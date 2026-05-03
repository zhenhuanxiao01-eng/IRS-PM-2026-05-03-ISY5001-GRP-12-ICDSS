"""
Phase 3 — Track 3: Cognitive Systems (Graph-Augmented RAG).

Workflow: symptom extraction -> graph retrieval -> cognitive synthesis.
Llama 3 generates diagnostic rationale grounded in graph evidence and rule alerts.
"""

import json
import os
import re

import requests
from neo4j import GraphDatabase

from ingest import normalize_entity
from trace_logger import log_event

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD", "icdss_secure_2026")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")


def _llm_call(prompt: str, temperature: float = 0.3) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        log_event("CognitiveLLMError", {"error": str(e)}, status="Error")
        return ""


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _fallback_extract_symptoms(text: str) -> list[str]:
    """Deterministic keyword fallback when LLM is unavailable or fails."""
    from ingest import NORMALIZATION_MAP
    text_lower = text.lower()
    found = set()
    sorted_keys = sorted(NORMALIZATION_MAP.keys(), key=len, reverse=True)
    for term in sorted_keys:
        if term in text_lower:
            normalized = normalize_entity(term)
            if normalized and normalized != "Normal":
                found.add(normalized)
    return list(found)


def _fallback_extract_params(text: str) -> dict:
    """Deterministic regex fallback for numeric parameters and lifestyle flags."""
    params = {}
    text_lower = text.lower()

    age_match = re.search(r"(\d{1,3})\s*[-–]?\s*year", text_lower)
    if age_match:
        params["age"] = int(age_match.group(1))

    bp_match = re.search(r"(?:bp|blood pressure)[:\s]*(\d{2,3})\s*/\s*(\d{2,3})", text_lower)
    if not bp_match:
        bp_match = re.search(r"(?:bp|blood pressure).{0,40}?(\d{2,3})\s*/\s*(\d{2,3})", text_lower)
    if not bp_match:
        bp_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s*hg", text_lower)
    if bp_match:
        params["resting_bp"] = int(bp_match.group(1))

    ldl_match = re.search(r"ldl(?:\s+cholesterol)?.{0,30}?(\d{2,4})\s*mg\s*/?\s*dl", text_lower)
    if ldl_match:
        params["ldl_cholesterol"] = int(ldl_match.group(1))

    chol_match = re.search(r"cholesterol[:\s]*(\d{2,4})", text_lower)
    if not chol_match:
        chol_match = re.search(r"(\d{2,4})\s*mg\s*/?\s*dl", text_lower)
    if chol_match:
        params["cholesterol"] = int(chol_match.group(1))

    hr_match = re.search(r"(?:heart rate|hr|pulse)[:\s]*(\d{2,3})", text_lower)
    if hr_match:
        params["max_heart_rate"] = int(hr_match.group(1))

    if re.search(r"\b(?:male|man|boy)\b", text_lower):
        params["sex"] = "Male"
    elif re.search(r"\b(?:female|woman|girl)\b", text_lower):
        params["sex"] = "Female"

    if re.search(r"\b(?:smok|tobacco|cigarette|nicotine)\w*\b", text_lower):
        params["smoking"] = "TRUE"

    if re.search(r"\bfamily history\b|first-degree relative|first degree relative", text_lower):
        params["family_history_cvd"] = "TRUE"

    bmi_match = re.search(r"\bbmi\b.{0,20}?(\d{2}(?:\.\d+)?)", text_lower)
    if bmi_match:
        params["bmi"] = float(bmi_match.group(1))

    ascvd_match = re.search(r"\bascvd\b.{0,40}?(\d{1,3})(?:\s*%)?", text_lower)
    if ascvd_match:
        params["ascvd_risk"] = int(ascvd_match.group(1))

    if re.search(r"\b(?:chest pain|chest discomfort|chest tight|angina)\b", text_lower):
        params["chest_pain_present"] = "TRUE"

    if re.search(r"\b(?:sedentary|unhealthy|lack of (?:exercise|sleep)|poor (?:diet|sleep|lifestyle)|inactive)\b", text_lower):
        params["sedentary_lifestyle"] = "TRUE"

    if re.search(r"\b(?:exercise.{0,5}angina|angina.{0,10}exercise)\b", text_lower):
        params["exercise_angina"] = "TRUE"

    if re.search(r"\b(?:fasting.{0,10}(?:sugar|glucose|blood sugar).{0,10}(?:high|elevated|above|>))\b", text_lower):
        params["fasting_blood_sugar"] = "TRUE"
    if re.search(r"fasting.{0,20}(?:plasma\s+)?glucose.{0,30}(?:100\s*[-–]\s*125|between\s+100)", text_lower):
        params["prediabetes"] = "TRUE"
        params["fasting_blood_sugar"] = "TRUE"

    if re.search(r"\b(?:faint|syncope|loss of consciousness|blackout|passed out)\w*\b", text_lower):
        params["syncope"] = "TRUE"
        params["fainting"] = "TRUE"

    if re.search(r"(?:shortness of breath.{0,15}(?:at rest|even at rest)|severe (?:shortness of breath|dyspnea)|breathless.{0,10}rest|dyspnea at rest)", text_lower):
        params["dyspnea_at_rest"] = "TRUE"

    st_match = re.search(r"\bst[\s-]*(?:segment|depression|elevation|changes)\b", text_lower)
    if st_match and not re.search(r"normal.{0,15}st|st.{0,15}normal", text_lower):
        params["st_changes"] = "TRUE"

    return params


def extract_symptoms_from_text(user_input: str) -> list[str]:
    """Extract symptoms via LLM with deterministic keyword fallback."""
    prompt = (
        "You are a clinical NLP system. Extract all medical symptoms, conditions, "
        "and clinical findings from the following patient description.\n\n"
        "Return ONLY a JSON array of strings, each being a symptom or condition.\n"
        "Be specific and use standard medical terminology.\n\n"
        f"Patient description: {user_input}\n\n"
        "JSON array:"
    )

    response = _llm_call(prompt, temperature=0.1)
    llm_results = []
    if response:
        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                raw = json.loads(match.group())
                llm_results = [n for s in raw if isinstance(s, str) for n in [normalize_entity(s)] if n]
        except (json.JSONDecodeError, ValueError):
            pass

    fallback_results = _fallback_extract_symptoms(user_input)

    merged = list(dict.fromkeys(llm_results + fallback_results))
    return merged


def extract_patient_params(user_input: str) -> dict:
    """Extract clinical parameters via LLM with deterministic regex fallback."""
    prompt = (
        "You are a clinical data extraction system. From the following patient description, "
        "extract any clinical parameters and lifestyle risk factors present.\n\n"
        "Return ONLY a JSON object with these possible keys (omit any not mentioned):\n"
        '  "age": integer,\n'
        '  "sex": "Male" or "Female",\n'
        '  "resting_bp": integer (systolic blood pressure in mmHg),\n'
        '  "cholesterol": integer (in mg/dL),\n'
        '  "ldl_cholesterol": integer (LDL cholesterol in mg/dL),\n'
        '  "max_heart_rate": integer (in bpm),\n'
        '  "bmi": float,\n'
        '  "ascvd_risk": integer (10-year ASCVD risk percentage),\n'
        '  "fasting_blood_sugar": "TRUE" or "FALSE",\n'
        '  "prediabetes": "TRUE" if fasting glucose is 100-125 mg/dL, otherwise "FALSE",\n'
        '  "exercise_angina": "TRUE" or "FALSE",\n'
        '  "st_depression": float,\n'
        '  "num_vessels": integer (0-3),\n'
        '  "smoking": "TRUE" if patient smokes or uses tobacco, otherwise "FALSE",\n'
        '  "family_history_cvd": "TRUE" if premature ASCVD/CVD family history is mentioned, otherwise "FALSE",\n'
        '  "chest_pain_present": "TRUE" if any chest pain/discomfort is mentioned, otherwise "FALSE",\n'
        '  "sedentary_lifestyle": "TRUE" if unhealthy lifestyle, lack of exercise, poor sleep, or poor diet is mentioned, otherwise "FALSE",\n'
        '  "syncope": "TRUE" if fainting, syncope, loss of consciousness, or blackout is mentioned, otherwise "FALSE",\n'
        '  "fainting": "TRUE" (same as syncope — set if any fainting event described),\n'
        '  "dyspnea_at_rest": "TRUE" if shortness of breath at rest, severe dyspnea, or breathlessness even at rest is mentioned, otherwise "FALSE"\n\n'
        f"Patient description: {user_input}\n\n"
        "JSON object:"
    )

    llm_params = {}
    response = _llm_call(prompt, temperature=0.1)
    if response:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                llm_params = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    fallback_params = _fallback_extract_params(user_input)

    merged = {**fallback_params, **llm_params}
    return merged


def retrieve_graph_context(driver, symptoms: list[str]) -> dict:
    """Step 2: Query Neo4j for connected sub-graph based on extracted symptoms."""
    context = {
        "guideline_links": [],
        "disease_symptoms": [],
        "disease_treatments": [],
    }

    with driver.session() as session:
        for symptom in symptoms:
            records = session.run(
                """
                MATCH (s:Symptom {name: $symptom})<-[:HAS_SYMPTOM]-(d:Disease)
                OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
                OPTIONAL MATCH (g:Guideline)-[:REFERENCES]->(d)
                RETURN d.name AS disease, COLLECT(DISTINCT t.name) AS treatments,
                       COLLECT(DISTINCT g.name) AS guidelines
                """,
                symptom=symptom,
            )
            for record in records:
                if record["disease"]:
                    context["disease_symptoms"].append({
                        "disease": record["disease"],
                        "symptom": symptom,
                    })
                    for t in record["treatments"]:
                        if t:
                            context["disease_treatments"].append({
                                "disease": record["disease"],
                                "treatment": t,
                            })
                    for g in record["guidelines"]:
                        if g:
                            context["guideline_links"].append({
                                "guideline": g,
                                "disease": record["disease"],
                            })

    log_event("GraphRetrieval", {
        "symptoms_queried": symptoms,
        "diseases_found": len(context["disease_symptoms"]),
        "treatments_found": len(context["disease_treatments"]),
        "guidelines_found": len(context["guideline_links"]),
    })

    return context


def synthesize_rationale(
    user_input: str,
    symptoms: list[str],
    graph_context: dict,
    rule_alerts: list[dict],
    similar_cases: list[dict],
) -> str:
    """Step 3: Llama 3 generates a diagnostic report grounded in graph evidence."""

    graph_evidence = []
    for ds in graph_context.get("disease_symptoms", []):
        graph_evidence.append(f"- Symptom '{ds['symptom']}' is associated with {ds['disease']}")
    for dt in graph_context.get("disease_treatments", []):
        graph_evidence.append(f"- {dt['disease']} is treated by {dt['treatment']}")
    for gl in graph_context.get("guideline_links", []):
        graph_evidence.append(f"- Guideline '{gl['guideline']}' references {gl['disease']}")

    alert_text = ""
    if rule_alerts:
        alert_lines = [f"- [{a['severity'].upper()}] {a['message']}" for a in rule_alerts]
        alert_text = "SAFETY ALERTS (from Rule Engine):\n" + "\n".join(alert_lines)

    similar_text = ""
    if similar_cases:
        case_lines = []
        for i, case in enumerate(similar_cases, 1):
            case_lines.append(
                f"- Case {i}: Age {case.get('age', 'N/A')}, {case.get('sex', 'N/A')}, "
                f"Similarity={case.get('similarity', 0):.2%}, "
                f"Diagnosis={'Heart Disease' if case.get('diagnosis_code', 0) > 0 else 'No Disease'}, "
                f"Matching symptoms: {', '.join(case.get('matching_symptoms', []))}"
            )
        similar_text = "SIMILAR HISTORICAL CASES:\n" + "\n".join(case_lines)

    prompt = (
        "You are a clinical decision support system. Based ONLY on the evidence provided below, "
        "generate a structured diagnostic assessment for the patient. "
        "Do NOT invent information not present in the evidence. If graph evidence is missing or weak, "
        "state that limitation clearly instead of filling the gap from general medical knowledge.\n\n"
        f"PATIENT PRESENTATION:\n{user_input}\n\n"
        f"EXTRACTED SYMPTOMS: {', '.join(symptoms)}\n\n"
        f"KNOWLEDGE GRAPH EVIDENCE:\n" + ("\n".join(graph_evidence) if graph_evidence else "No direct graph matches found.") + "\n\n"
        f"{alert_text}\n\n"
        f"{similar_text}\n\n"
        "Provide your assessment in the following format:\n"
        "1. CLINICAL SUMMARY: Brief overview of the patient's presentation\n"
        "2. RISK ASSESSMENT: Key risk factors identified from the evidence\n"
        "3. DIFFERENTIAL CONSIDERATIONS: Possible conditions based on graph evidence\n"
        "4. RECOMMENDED ACTIONS: Evidence-based next steps\n"
        "5. EVIDENCE SOURCES: List which evidence sources support your conclusions\n\n"
        "IMPORTANT: This is a decision SUPPORT tool. All outputs require clinician review.\n\n"
        "Assessment:"
    )

    log_event("CognitiveSynthesis", {
        "prompt_length": len(prompt),
        "symptoms_count": len(symptoms),
        "graph_evidence_count": len(graph_evidence),
        "alerts_count": len(rule_alerts),
        "similar_cases_count": len(similar_cases),
        "assembled_prompt": prompt[:500] + "...",
    })

    return _llm_call(prompt, temperature=0.3)


def run_cognitive_pipeline(
    user_input: str,
    rule_alerts: list[dict],
    similar_cases: list[dict],
    symptoms: list[str] | None = None,
    patient_params: dict | None = None,
) -> dict:
    """Execute the full cognitive pipeline: extract -> retrieve -> synthesize."""
    symptoms = symptoms if symptoms is not None else extract_symptoms_from_text(user_input)
    patient_params = patient_params if patient_params is not None else extract_patient_params(user_input)

    driver = _get_driver()
    try:
        graph_context = retrieve_graph_context(driver, symptoms)
    finally:
        driver.close()

    rationale = synthesize_rationale(user_input, symptoms, graph_context, rule_alerts, similar_cases)

    return {
        "symptoms": symptoms,
        "patient_params": patient_params,
        "graph_context": graph_context,
        "rationale": rationale,
        "llm_available": bool(rationale),
    }
