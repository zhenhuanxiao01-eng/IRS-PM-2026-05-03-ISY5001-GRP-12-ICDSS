"""
Phase 2 — Task 2.2: Hybrid Ingestion Logic.

Handles structured CSV data loading into Neo4j, NLP enrichment via Llama 3,
PDF knowledge extraction (pdfplumber + Llama 3 triple extraction), and
entity normalization.
"""

import json
import os
import re
from glob import glob

import pandas as pd
import pdfplumber
import requests
from neo4j import GraphDatabase

from trace_logger import log_event

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD", "icdss_secure_2026")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

NORMALIZATION_MAP = {
    # --- Blood pressure / Hypertension ---
    "high bp": "Hypertension",
    "high blood pressure": "Hypertension",
    "elevated blood pressure": "Hypertension",
    "hypertension": "Hypertension",
    "htn": "Hypertension",
    # --- Diabetes ---
    "diabetes": "Diabetes Mellitus",
    "diabetes mellitus": "Diabetes Mellitus",
    "dm": "Diabetes Mellitus",
    "type 2 diabetes": "Diabetes Mellitus",
    # --- MI ---
    "heart attack": "Myocardial Infarction",
    "mi": "Myocardial Infarction",
    "myocardial infarction": "Myocardial Infarction",
    # --- Chest pain / Angina (lay-language -> medical terms) ---
    "chest pain": "Atypical Angina",
    "occasional chest pain": "Atypical Angina",
    "intermittent chest pain": "Atypical Angina",
    "mild chest pain": "Atypical Angina",
    "chest discomfort": "Atypical Angina",
    "chest tightness": "Atypical Angina",
    "chest pressure": "Atypical Angina",
    "chest heaviness": "Atypical Angina",
    "typical chest pain": "Typical Angina",
    "severe chest pain": "Typical Angina",
    "exertional chest pain": "Typical Angina",
    "crushing chest pain": "Typical Angina",
    "crushing chest pressure": "Typical Angina",
    "crushing pain": "Typical Angina",
    "crushing pressure": "Typical Angina",
    "substernal pain": "Typical Angina",
    "substernal pressure": "Typical Angina",
    "radiating chest pain": "Typical Angina",
    "radiating pain": "Typical Angina",
    "pain radiating to arm": "Typical Angina",
    "pain radiating to neck": "Typical Angina",
    "pain radiating to jaw": "Typical Angina",
    "arm pain": "Typical Angina",
    "left arm pain": "Typical Angina",
    "jaw pain": "Typical Angina",
    "neck pain": "Typical Angina",
    "resting chest pain": "Typical Angina",
    "pain at rest": "Typical Angina",
    "chest pain at rest": "Typical Angina",
    "angina": "Atypical Angina",
    "angina pectoris": "Atypical Angina",
    "typical angina": "Typical Angina",
    "atypical angina": "Atypical Angina",
    "non-anginal": "Non-Anginal Pain",
    "non-anginal pain": "Non-Anginal Pain",
    "non-anginal chest pain": "Non-Anginal Pain",
    "no chest pain": "Asymptomatic",
    "asymptomatic": "Asymptomatic",
    # --- Cholesterol ---
    "high cholesterol": "Hypercholesterolemia",
    "hypercholesterolemia": "Hypercholesterolemia",
    "elevated cholesterol": "Hypercholesterolemia",
    "ldl cholesterol": "Hypercholesterolemia",
    "elevated ldl": "Hypercholesterolemia",
    "high ldl": "Hypercholesterolemia",
    "prediabetes": "Prediabetes",
    "impaired fasting glucose": "Prediabetes",
    "high ascvd risk": "High ASCVD Risk",
    "ascvd risk": "High ASCVD Risk",
    # --- Dyspnea ---
    "shortness of breath": "Dyspnea",
    "dyspnea": "Dyspnea",
    "sob": "Dyspnea",
    "breathlessness": "Dyspnea",
    # --- Cardiovascular diseases ---
    "coronary artery disease": "Coronary Artery Disease",
    "cad": "Coronary Artery Disease",
    "heart failure": "Heart Failure",
    "chf": "Heart Failure",
    "congestive heart failure": "Heart Failure",
    "heart disease": "Coronary Artery Disease",
    # --- Lifestyle risk factors ---
    "smoking": "Smoking",
    "smoker": "Smoking",
    "heavy smoker": "Smoking",
    "long-term smoker": "Smoking",
    "long term smoker": "Smoking",
    "chronic smoker": "Smoking",
    "chain smoker": "Smoking",
    "tobacco use": "Smoking",
    "tobacco": "Smoking",
    "cigarette": "Smoking",
    "cigarettes": "Smoking",
    "nicotine": "Smoking",
    "obesity": "Obesity",
    "overweight": "Obesity",
    "obese": "Obesity",
    "sedentary": "Sedentary Lifestyle",
    "sedentary lifestyle": "Sedentary Lifestyle",
    "physical inactivity": "Sedentary Lifestyle",
    "lack of exercise": "Sedentary Lifestyle",
    "inactive lifestyle": "Sedentary Lifestyle",
    "lack of sleep": "Poor Sleep",
    "poor sleep": "Poor Sleep",
    "insomnia": "Poor Sleep",
    "sleep deprivation": "Poor Sleep",
    "unhealthy lifestyle": "Unhealthy Lifestyle",
    "unhealthy diet": "Unhealthy Lifestyle",
    "poor diet": "Unhealthy Lifestyle",
    "alcohol": "Alcohol Use",
    "alcohol use": "Alcohol Use",
    "heavy drinking": "Alcohol Use",
    "stress": "Psychological Stress",
    "anxiety": "Psychological Stress",
    "family history": "Family History of CVD",
    "family history of heart disease": "Family History of CVD",
    "family history of cvd": "Family History of CVD",
    # --- Stroke ---
    "stroke": "Stroke",
    "cerebrovascular accident": "Stroke",
    "cva": "Stroke",
    # --- Kidney ---
    "kidney disease": "Chronic Kidney Disease",
    "chronic kidney disease": "Chronic Kidney Disease",
    "ckd": "Chronic Kidney Disease",
    # --- ECG / Cardiac findings ---
    "exercise-induced angina": "Exercise-Induced Angina",
    "exercise angina": "Exercise-Induced Angina",
    "unstable angina": "Unstable Angina",
    "acute coronary syndrome": "Unstable Angina",
    "acs": "Unstable Angina",
    "lv hypertrophy": "Left Ventricular Hypertrophy",
    "left ventricular hypertrophy": "Left Ventricular Hypertrophy",
    "stt abnormality": "ST-T Wave Abnormality",
    "st-t abnormality": "ST-T Wave Abnormality",
    "st-segment changes": "ST-T Wave Abnormality",
    "st segment changes": "ST-T Wave Abnormality",
    "st changes": "ST-T Wave Abnormality",
    "st elevation": "ST-T Wave Abnormality",
    "st depression": "ST-T Wave Abnormality",
    "st-segment depression": "ST-T Wave Abnormality",
    "st-segment elevation": "ST-T Wave Abnormality",
    "ecg abnormality": "ST-T Wave Abnormality",
    "abnormal ecg": "ST-T Wave Abnormality",
    "ekg abnormality": "ST-T Wave Abnormality",
    "normal": "Normal",
    "normal ecg": "Normal",
    "fixed defect": "Fixed Defect",
    "reversable defect": "Reversible Defect",
    "reversible defect": "Reversible Defect",
    # --- Syncope / Fainting ---
    "fainting": "Syncope",
    "fainting spells": "Syncope",
    "fainted": "Syncope",
    "faint": "Syncope",
    "syncope": "Syncope",
    "loss of consciousness": "Syncope",
    "blackout": "Syncope",
    "passed out": "Syncope",
    "dizzy": "Dizziness",
    "dizziness": "Dizziness",
    "lightheaded": "Dizziness",
    "light-headed": "Dizziness",
    "vertigo": "Dizziness",
    # --- Dyspnea at rest / Severity ---
    "shortness of breath at rest": "Dyspnea at Rest",
    "breathless at rest": "Dyspnea at Rest",
    "dyspnea at rest": "Dyspnea at Rest",
    "severe shortness of breath": "Dyspnea at Rest",
    "severe dyspnea": "Dyspnea at Rest",
    # --- Palpitations ---
    "palpitations": "Palpitations",
    "palpitation": "Palpitations",
    "irregular heartbeat": "Palpitations",
    "racing heart": "Palpitations",
    "heart racing": "Palpitations",
    "tachycardia": "Tachycardia",
    # --- Edema ---
    "swelling": "Peripheral Edema",
    "leg swelling": "Peripheral Edema",
    "ankle swelling": "Peripheral Edema",
    "peripheral edema": "Peripheral Edema",
    "edema": "Peripheral Edema",
    # --- Fatigue ---
    "fatigue": "Fatigue",
    "tiredness": "Fatigue",
    "exhaustion": "Fatigue",
    "weakness": "Fatigue",
    "lethargy": "Fatigue",
    # --- Nausea ---
    "nausea": "Nausea",
    "vomiting": "Nausea",
    "nauseous": "Nausea",
    # --- Cyanosis ---
    "cyanosis": "Cyanosis",
    "blue lips": "Cyanosis",
    "bluish skin": "Cyanosis",
    # --- Treatments ---
    "ace inhibitor": "ACE Inhibitor",
    "ace inhibitors": "ACE Inhibitor",
    "arb": "ARB",
    "beta blocker": "Beta Blocker",
    "beta-blocker": "Beta Blocker",
    "beta blockers": "Beta Blocker",
    "calcium channel blocker": "Calcium Channel Blocker",
    "ccb": "Calcium Channel Blocker",
    "diuretic": "Diuretic",
    "diuretics": "Diuretic",
    "statin": "Statin",
    "statins": "Statin",
    "aspirin": "Aspirin",
    "nitroglycerin": "Nitroglycerin",
    "anticoagulant": "Anticoagulant",
    "anticoagulants": "Anticoagulant",
    "lifestyle modification": "Lifestyle Modification",
    "diet and exercise": "Lifestyle Modification",
    "weight loss": "Lifestyle Modification",
}

RISK_FACTOR_TERMS = {
    "Smoking", "Obesity", "Sedentary Lifestyle", "Poor Sleep",
    "Unhealthy Lifestyle", "Alcohol Use", "Psychological Stress",
    "Family History of CVD", "High ASCVD Risk",
}


_NULL_STRINGS = {"nan", "none", "null", "na", "n/a", ""}


def _is_valid_entity(name) -> bool:
    """Check if a value is a usable entity name (not null, NaN, or empty)."""
    if name is None:
        return False
    if not isinstance(name, str):
        return False
    return name.strip().lower() not in _NULL_STRINGS


def normalize_entity(name) -> str | None:
    if not _is_valid_entity(name):
        return None
    key = name.strip().lower()
    return NORMALIZATION_MAP.get(key, name.strip().title())


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _llm_call(prompt: str, temperature: float = 0.1) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        log_event("LLMCallError", {"error": str(e)}, status="Error")
        return ""


def create_constraints(driver):
    """Create uniqueness constraints and indexes for core node types."""
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.hashed_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:AgeGroup) REQUIRE a.label IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Guideline) REQUIRE g.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.name IS UNIQUE",
    ]
    with driver.session() as session:
        for c in constraints:
            session.run(c)
    log_event("Neo4jConstraints", {"constraints_created": len(constraints)})


def ingest_structured_data(driver, df: pd.DataFrame):
    """Load patient data as graph nodes and relationships."""
    with driver.session() as session:
        for _, row in df.iterrows():
            hashed_id = row.get("hashed_patient_id", "")
            age_group = row.get("age_group", "Unknown")
            bp_cat = row.get("bp_category", "Unknown")
            chol_cat = row.get("chol_category", "Unknown")
            hr_cat = row.get("hr_category", "Unknown")
            diagnosis = int(row.get("diagnosis", 0)) if pd.notna(row.get("diagnosis")) else 0
            cp = normalize_entity(row.get("chest_pain_type"))
            ecg = normalize_entity(row.get("resting_ecg"))
            thal = normalize_entity(row.get("thal_defect"))
            dataset = str(row.get("dataset", "Unknown"))

            props = {
                "hashed_id": hashed_id,
                "age": int(row["age"]) if pd.notna(row.get("age")) else None,
                "sex": str(row.get("sex", "")),
                "resting_bp": float(row["resting_bp"]) if pd.notna(row.get("resting_bp")) else None,
                "cholesterol": float(row["cholesterol"]) if pd.notna(row.get("cholesterol")) else None,
                "max_heart_rate": float(row["max_heart_rate"]) if pd.notna(row.get("max_heart_rate")) else None,
                "fasting_blood_sugar": str(row.get("fasting_blood_sugar", "")),
                "exercise_angina": str(row.get("exercise_angina", "")),
                "st_depression": float(row["st_depression"]) if pd.notna(row.get("st_depression")) else None,
                "slope": str(row.get("slope", "")),
                "num_vessels": str(row.get("num_vessels", "")),
                "diagnosis": diagnosis,
                "center": dataset,
                "description": str(row.get("description", "")),
                "bp_category": bp_cat,
                "chol_category": chol_cat,
                "hr_category": hr_cat,
            }

            session.run(
                """
                MERGE (p:Patient {hashed_id: $hashed_id})
                SET p += $props
                MERGE (ag:AgeGroup {label: $age_group})
                MERGE (p)-[:IN_AGE_GROUP]->(ag)
                """,
                hashed_id=hashed_id, props=props, age_group=age_group,
            )

            if cp and cp != "Normal":
                session.run(
                    """
                    MATCH (p:Patient {hashed_id: $hashed_id})
                    MERGE (s:Symptom {name: $symptom})
                    MERGE (p)-[:HAS_SYMPTOM]->(s)
                    """,
                    hashed_id=hashed_id, symptom=cp,
                )

            if ecg and ecg != "Normal":
                session.run(
                    """
                    MATCH (p:Patient {hashed_id: $hashed_id})
                    MERGE (c:Condition {name: $condition})
                    MERGE (p)-[:HAS_CONDITION]->(c)
                    """,
                    hashed_id=hashed_id, condition=ecg,
                )

            if thal and thal != "Normal":
                session.run(
                    """
                    MATCH (p:Patient {hashed_id: $hashed_id})
                    MERGE (c:Condition {name: $condition})
                    MERGE (p)-[:HAS_CONDITION]->(c)
                    """,
                    hashed_id=hashed_id, condition=thal,
                )

            exang_str = str(row.get("exercise_angina", "")).strip().upper()
            if exang_str in ("TRUE", "1", "YES"):
                session.run(
                    """
                    MATCH (p:Patient {hashed_id: $hashed_id})
                    MERGE (s:Symptom {name: 'Exercise-Induced Angina'})
                    MERGE (p)-[:HAS_SYMPTOM]->(s)
                    """,
                    hashed_id=hashed_id,
                )

            fbs_str = str(row.get("fasting_blood_sugar", "")).strip().upper()
            if fbs_str in ("TRUE", "1", "YES"):
                session.run(
                    """
                    MATCH (p:Patient {hashed_id: $hashed_id})
                    MERGE (c:Condition {name: 'Elevated Fasting Blood Sugar'})
                    MERGE (p)-[:HAS_CONDITION]->(c)
                    """,
                    hashed_id=hashed_id,
                )

            if diagnosis > 0:
                severity = {1: "Mild", 2: "Moderate", 3: "Severe", 4: "Critical"}.get(diagnosis, "Unknown")
                session.run(
                    """
                    MATCH (p:Patient {hashed_id: $hashed_id})
                    MERGE (d:Disease {name: 'Heart Disease'})
                    MERGE (p)-[:DIAGNOSED_WITH {severity: $severity}]->(d)
                    """,
                    hashed_id=hashed_id, severity=severity,
                )

    log_event("StructuredDataIngestion", {"patients_loaded": len(df)})


def nlp_enrich_patients(driver, df: pd.DataFrame, batch_size: int = 20):
    """Use Llama 3 to extract additional symptoms from synthetic clinical notes."""
    prompt_template = (
        "You are a clinical NLP system. Extract medical symptoms and conditions "
        "from the following clinical note. Return ONLY a JSON array of strings, "
        "each being a symptom or condition name. Do not include explanations.\n\n"
        "Clinical note: {text}\n\n"
        "JSON array:"
    )

    enriched_count = 0
    with driver.session() as session:
        for _, row in df.iterrows():
            desc = row.get("description", "")
            hashed_id = row.get("hashed_patient_id", "")
            if not desc or not hashed_id:
                continue

            prompt = prompt_template.format(text=desc)
            response = _llm_call(prompt)
            if not response:
                continue

            try:
                match = re.search(r"\[.*?\]", response, re.DOTALL)
                if match:
                    symptoms = json.loads(match.group())
                else:
                    continue
            except (json.JSONDecodeError, ValueError):
                continue

            for symptom in symptoms:
                normalized = normalize_entity(symptom) if isinstance(symptom, str) else None
                if not normalized:
                    continue
                if normalized in RISK_FACTOR_TERMS:
                    session.run(
                        """
                        MATCH (p:Patient {hashed_id: $hashed_id})
                        MERGE (r:RiskFactor {name: $name})
                        MERGE (p)-[:HAS_RISK_FACTOR]->(r)
                        """,
                        hashed_id=hashed_id, name=normalized,
                    )
                else:
                    session.run(
                        """
                        MATCH (p:Patient {hashed_id: $hashed_id})
                        MERGE (s:Symptom {name: $symptom})
                        MERGE (p)-[:HAS_SYMPTOM]->(s)
                        """,
                        hashed_id=hashed_id, symptom=normalized,
                    )
            enriched_count += 1

    log_event("NLPEnrichment", {"patients_enriched": enriched_count})


def deterministic_enrich_patients(driver, df: pd.DataFrame):
    """Add graph terms already implied by structured fields before optional LLM enrichment."""
    enriched_count = 0
    with driver.session() as session:
        for _, row in df.iterrows():
            hashed_id = row.get("hashed_patient_id", "")
            if not hashed_id:
                continue

            terms: list[tuple[str, str]] = []
            if pd.notna(row.get("resting_bp")) and float(row.get("resting_bp")) >= 130:
                terms.append(("Condition", "Hypertension"))
            if pd.notna(row.get("cholesterol")) and float(row.get("cholesterol")) >= 200:
                terms.append(("Condition", "Hypercholesterolemia"))
            if str(row.get("exercise_angina", "")).strip().upper() in ("TRUE", "1", "YES"):
                terms.append(("Symptom", "Exercise-Induced Angina"))
            if str(row.get("fasting_blood_sugar", "")).strip().upper() in ("TRUE", "1", "YES"):
                terms.append(("Condition", "Elevated Fasting Blood Sugar"))

            for label, name in terms:
                if label == "Symptom":
                    session.run(
                        """
                        MATCH (p:Patient {hashed_id: $hashed_id})
                        MERGE (s:Symptom {name: $name})
                        MERGE (p)-[:HAS_SYMPTOM]->(s)
                        """,
                        hashed_id=hashed_id, name=name,
                    )
                else:
                    session.run(
                        """
                        MATCH (p:Patient {hashed_id: $hashed_id})
                        MERGE (c:Condition {name: $name})
                        MERGE (p)-[:HAS_CONDITION]->(c)
                        """,
                        hashed_id=hashed_id, name=name,
                    )
            if terms:
                enriched_count += 1

    log_event("DeterministicEnrichment", {"patients_enriched": enriched_count})


def extract_pdf_text(pdf_path: str) -> list[str]:
    """Extract text from PDF, one string per page."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text)
                else:
                    log_event("PDFPageSkipped", {"file": pdf_path, "page": i}, status="Warning")
    except Exception as e:
        log_event("PDFExtractionError", {"file": pdf_path, "error": str(e)}, status="Error")
    return pages


def chunk_text(pages: list[str], max_tokens: int = 800) -> list[str]:
    """Split page texts into chunks of approximately max_tokens words."""
    chunks = []
    for page_text in pages:
        words = page_text.split()
        for i in range(0, len(words), max_tokens):
            chunk = " ".join(words[i : i + max_tokens])
            if chunk.strip():
                chunks.append(chunk)
    return chunks


def extract_triples_from_chunk(chunk: str) -> list[dict]:
    """Send a text chunk to Llama 3 to extract disease-symptom-treatment triples."""
    prompt = (
        "You are a medical knowledge extraction system. From the text below, extract "
        "relationships in the form of triples. Each triple should be one of:\n"
        '  {"head": "<Disease>", "relation": "HAS_SYMPTOM", "tail": "<Symptom>"}\n'
        '  {"head": "<Disease>", "relation": "TREATED_BY", "tail": "<Treatment>"}\n\n'
        "Return ONLY a JSON array of these triple objects. If no triples can be extracted, "
        "return an empty array [].\n\n"
        f"Text:\n{chunk}\n\n"
        "JSON array:"
    )

    response = _llm_call(prompt)
    if not response:
        return []

    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            triples = json.loads(match.group())
            return [t for t in triples if isinstance(t, dict) and "head" in t and "relation" in t and "tail" in t]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def ingest_pdf_triples(driver, raw_data_dir: str = "/app/raw_data"):
    """Full PDF pipeline: extract text, chunk, extract triples, load to Neo4j."""
    pdf_files = glob(os.path.join(raw_data_dir, "*.pdf"))
    if not pdf_files:
        log_event("PDFIngestion", {"message": "No PDF files found"}, status="Warning")
        return

    total_triples = 0
    with driver.session() as session:
        for pdf_path in pdf_files:
            pdf_name = os.path.basename(pdf_path)
            log_event("PDFProcessingStart", {"file": pdf_name})

            pages = extract_pdf_text(pdf_path)
            chunks = chunk_text(pages)

            session.run(
                "MERGE (g:Guideline {name: $name}) SET g.source = $source",
                name=pdf_name, source=pdf_path,
            )

            for chunk in chunks:
                triples = extract_triples_from_chunk(chunk)
                for triple in triples:
                    head = normalize_entity(triple.get("head"))
                    relation = (triple.get("relation") or "").upper().replace(" ", "_")
                    tail = normalize_entity(triple.get("tail"))

                    if not head or not tail or not relation:
                        continue

                    if relation == "HAS_SYMPTOM":
                        session.run(
                            """
                            MERGE (d:Disease {name: $disease})
                            MERGE (s:Symptom {name: $symptom})
                            MERGE (d)-[:HAS_SYMPTOM]->(s)
                            MERGE (g:Guideline {name: $guideline})
                            MERGE (g)-[:REFERENCES]->(d)
                            """,
                            disease=head, symptom=tail, guideline=pdf_name,
                        )
                    elif relation == "TREATED_BY":
                        session.run(
                            """
                            MERGE (d:Disease {name: $disease})
                            MERGE (t:Treatment {name: $treatment})
                            MERGE (d)-[:TREATED_BY]->(t)
                            MERGE (g:Guideline {name: $guideline})
                            MERGE (g)-[:REFERENCES]->(d)
                            """,
                            disease=head, treatment=tail, guideline=pdf_name,
                        )
                    total_triples += 1

            log_event("PDFProcessingComplete", {"file": pdf_name, "chunks": len(chunks)})

    log_event("PDFIngestionComplete", {"pdf_count": len(pdf_files), "total_triples": total_triples})


def run_ingestion(
    df: pd.DataFrame,
    raw_data_dir: str = "/app/raw_data",
    include_nlp: bool = True,
    include_pdf: bool = True,
):
    """Execute the full ingestion pipeline."""
    if df is None or df.empty:
        raise ValueError("No processed patient data available. Run the data pipeline first.")

    driver = _get_driver()
    try:
        create_constraints(driver)
        ingest_structured_data(driver, df)
        deterministic_enrich_patients(driver, df)
        if include_nlp:
            nlp_enrich_patients(driver, df)
        else:
            log_event("NLPEnrichmentSkipped", {"reason": "disabled_by_user"}, status="Warning")
        if include_pdf:
            ingest_pdf_triples(driver, raw_data_dir)
        else:
            log_event("PDFIngestionSkipped", {"reason": "disabled_by_user"}, status="Warning")
        log_event("IngestionComplete", {"include_nlp": include_nlp, "include_pdf": include_pdf})
    finally:
        driver.close()
