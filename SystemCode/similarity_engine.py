"""
Phase 3 — Track 2: Knowledge Discovery (Similarity Engine).

Uses label-informed learned similarity to find the top-3 most similar
historical cases ("Digital Twins").
"""

import math
import os

from neo4j import GraphDatabase

from trace_logger import log_event

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("DB_PASSWORD", "icdss_secure_2026")

TERM_WEIGHT_PRIORS = {
    "Typical Angina": 3.0,
    "Exercise-Induced Angina": 2.5,
    "Hypertension": 2.0,
    "Hypercholesterolemia": 2.0,
    "Elevated Fasting Blood Sugar": 1.7,
    "Prediabetes": 1.5,
    "Smoking": 1.5,
    "Family History of CVD": 1.3,
    "Obesity": 1.2,
    "High ASCVD Risk": 1.2,
    "Atypical Angina": 1.2,
}

NUMERIC_FEATURE_RANGES = {
    "age": 40,
    "resting_bp": 80,
    "cholesterol": 200,
    "max_heart_rate": 120,
}


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def find_similar_cases(symptoms: list[str], top_k: int = 3, patient_params: dict | None = None) -> list[dict]:
    """
    Given normalized clinical terms and extracted numeric parameters, learn
    feature weights from historical diagnosis labels and return top-k matches.
    """
    patient_params = patient_params or {}
    if not symptoms and not patient_params:
        return []

    driver = _get_driver()
    try:
        return _hybrid_similarity_cypher(driver, symptoms, top_k, patient_params)
    finally:
        driver.close()


def _term_weight(term: str, learned_weights: dict[str, float]) -> float:
    return learned_weights.get(term, TERM_WEIGHT_PRIORS.get(term, 1.0))


def _numeric_similarity(patient: dict, patient_params: dict, numeric_weights: dict[str, float]) -> float:
    scores = []

    for key, spread in NUMERIC_FEATURE_RANGES.items():
        query_value = patient_params.get(key)
        patient_value = patient.get(key)
        if query_value is None or patient_value is None:
            continue
        try:
            diff = abs(float(query_value) - float(patient_value))
        except (TypeError, ValueError):
            continue
        feature_score = max(0.0, 1.0 - (diff / spread))
        scores.append((feature_score, numeric_weights.get(key, 1.0)))

    # LDL is not present in the UCI dataset, so use total cholesterol only as a weak proxy.
    if patient_params.get("ldl_cholesterol") is not None and patient.get("cholesterol") is not None:
        try:
            diff = abs(float(patient_params["ldl_cholesterol"]) - float(patient["cholesterol"]))
            feature_score = max(0.0, 1.0 - (diff / 220)) * 0.7
            scores.append((feature_score, numeric_weights.get("cholesterol", 1.0)))
        except (TypeError, ValueError):
            pass

    total_weight = sum(weight for _, weight in scores)
    if not total_weight:
        return 0.0
    return sum(score * weight for score, weight in scores) / total_weight


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_burden(symptoms: list[str], patient_params: dict) -> int:
    risk_terms = {
        "Hypertension",
        "Hypercholesterolemia",
        "Elevated Fasting Blood Sugar",
        "Prediabetes",
        "Smoking",
        "Family History of CVD",
        "Obesity",
        "High ASCVD Risk",
    }
    burden = len(set(symptoms) & risk_terms)
    if _to_float(patient_params.get("resting_bp")) >= 140:
        burden += 1
    if _to_float(patient_params.get("ldl_cholesterol")) >= 160:
        burden += 1
    if _to_float(patient_params.get("bmi")) >= 30:
        burden += 1
    if _to_float(patient_params.get("ascvd_risk")) >= 20:
        burden += 1
    return burden


def _learn_term_weights(candidates: list[dict]) -> dict[str, float]:
    """Learn term importance from diagnosis lift, with priors as a stabilizer."""
    positive_total = sum(1 for row in candidates if (row.get("diagnosis_code") or 0) > 0)
    negative_total = max(0, len(candidates) - positive_total)
    if not positive_total or not negative_total:
        return dict(TERM_WEIGHT_PRIORS)

    term_counts: dict[str, dict[str, int]] = {}
    for row in candidates:
        bucket = "positive" if (row.get("diagnosis_code") or 0) > 0 else "negative"
        for term in set(row.get("symptoms", [])):
            term_counts.setdefault(term, {"positive": 0, "negative": 0})
            term_counts[term][bucket] += 1

    learned = {}
    for term, counts in term_counts.items():
        positive_rate = (counts["positive"] + 1) / (positive_total + 2)
        negative_rate = (counts["negative"] + 1) / (negative_total + 2)
        lift = positive_rate / negative_rate
        data_weight = min(3.0, max(0.6, 1.0 + math.log(lift)))
        prior = TERM_WEIGHT_PRIORS.get(term, 1.0)
        learned[term] = round((0.65 * data_weight) + (0.35 * prior), 3)
    for term, prior in TERM_WEIGHT_PRIORS.items():
        learned.setdefault(term, prior)
    return learned


def _learn_numeric_weights(candidates: list[dict]) -> dict[str, float]:
    """Learn numeric feature importance from diseased vs non-diseased mean separation."""
    weights = {}
    for key, spread in NUMERIC_FEATURE_RANGES.items():
        positive_values = [
            _to_float(row.get(key), None)
            for row in candidates
            if (row.get("diagnosis_code") or 0) > 0 and row.get(key) is not None
        ]
        negative_values = [
            _to_float(row.get(key), None)
            for row in candidates
            if (row.get("diagnosis_code") or 0) == 0 and row.get(key) is not None
        ]
        positive_values = [value for value in positive_values if value is not None]
        negative_values = [value for value in negative_values if value is not None]
        if not positive_values or not negative_values:
            weights[key] = 1.0
            continue
        positive_mean = sum(positive_values) / len(positive_values)
        negative_mean = sum(negative_values) / len(negative_values)
        separation = abs(positive_mean - negative_mean) / spread
        weights[key] = round(min(2.0, max(0.5, 0.75 + (separation * 4))), 3)
    return weights


def _weighted_term_similarity(
    input_terms: list[str],
    patient_terms: list[str],
    corpus_terms: set[str],
    learned_weights: dict[str, float],
) -> tuple[float, list[str]]:
    supported_input = [term for term in input_terms if term in corpus_terms]
    input_set = set(supported_input)
    patient_set = set(patient_terms)
    intersection = sorted(input_set & patient_set)
    union = input_set | patient_set
    if not union:
        return 0.0, intersection

    intersection_weight = sum(_term_weight(term, learned_weights) for term in intersection)
    union_weight = sum(_term_weight(term, learned_weights) for term in union)
    return intersection_weight / union_weight if union_weight else 0.0, intersection


def _hybrid_similarity_cypher(driver, symptoms: list[str], top_k: int, patient_params: dict) -> list[dict]:
    """
    Rank similar cases with a hybrid score:
    supported weighted graph-term overlap + numeric clinical similarity.
    """
    query = """
    MATCH (p:Patient)
    OPTIONAL MATCH (p)-[:HAS_SYMPTOM]->(s:Symptom)
    OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition)
    OPTIONAL MATCH (p)-[:HAS_RISK_FACTOR]->(r:RiskFactor)
    WITH p, COLLECT(DISTINCT s.name) + COLLECT(DISTINCT c.name) + COLLECT(DISTINCT r.name) AS rawTerms
    WITH p, [x IN rawTerms WHERE x IS NOT NULL] AS patientTerms
    WHERE SIZE(patientTerms) > 0
    OPTIONAL MATCH (p)-[diag:DIAGNOSED_WITH]->(d:Disease)
    OPTIONAL MATCH (p)-[:HAS_CONDITION]->(cond:Condition)
    RETURN p.hashed_id AS patient_id,
           p.age AS age,
           p.sex AS sex,
           p.center AS center,
           p.diagnosis AS diagnosis_code,
           p.resting_bp AS resting_bp,
           p.cholesterol AS cholesterol,
           p.max_heart_rate AS max_heart_rate,
           p.description AS description,
           patientTerms AS symptoms,
           COLLECT(DISTINCT d.name) AS diseases,
           diag.severity AS severity,
           COLLECT(DISTINCT cond.name) AS conditions
    """

    candidates = []
    with driver.session() as session:
        records = session.run(query)
        for record in records:
            candidates.append({
                "patient_id": record["patient_id"][:12] + "..." if record["patient_id"] else "N/A",
                "age": record["age"],
                "sex": record["sex"],
                "center": record["center"],
                "diagnosis_code": record["diagnosis_code"],
                "resting_bp": record["resting_bp"],
                "cholesterol": record["cholesterol"],
                "max_heart_rate": record["max_heart_rate"],
                "symptoms": record["symptoms"],
                "diseases": record["diseases"],
                "severity": record["severity"],
                "conditions": record["conditions"],
                "description": record["description"],
            })

    corpus_terms = {term for candidate in candidates for term in candidate["symptoms"]}
    learned_term_weights = _learn_term_weights(candidates)
    learned_numeric_weights = _learn_numeric_weights(candidates)
    risk_burden = _risk_burden(symptoms, patient_params)
    results = []
    for candidate in candidates:
        term_score, matching_terms = _weighted_term_similarity(
            symptoms,
            candidate["symptoms"],
            corpus_terms,
            learned_term_weights,
        )
        numeric_score = _numeric_similarity(candidate, patient_params, learned_numeric_weights)
        diagnosis_boost = 0.08 if risk_burden >= 4 and (candidate.get("diagnosis_code") or 0) > 0 else 0.0
        hybrid_score = min(1.0, (term_score * 0.55) + (numeric_score * 0.35) + diagnosis_boost)
        if hybrid_score <= 0:
            continue
        raw_input = set(symptoms)
        patient_set = set(candidate["symptoms"])
        raw_union = raw_input | patient_set
        raw_intersection = raw_input & patient_set
        jaccard_score = len(raw_intersection) / len(raw_union) if raw_union else 0

        candidate.update({
            "similarity": round(hybrid_score, 4),
            "jaccard_similarity": round(jaccard_score, 4),
            "term_similarity": round(term_score, 4),
            "numeric_similarity": round(numeric_score, 4),
            "matching_symptoms": matching_terms,
            "matched_term_weights": {
                term: learned_term_weights.get(term, 1.0)
                for term in matching_terms
            },
            "numeric_weights": learned_numeric_weights,
            "scoring_method": "learned_similarity",
        })
        results.append(candidate)

    results.sort(
        key=lambda row: (
            row["similarity"],
            1 if (row.get("diagnosis_code") or 0) > 0 else 0,
            row.get("jaccard_similarity", 0),
        ),
        reverse=True,
    )
    results = results[:top_k]

    log_event("SimilaritySearch", {
        "input_symptoms": symptoms,
        "matches_found": len(results),
        "top_score": results[0]["similarity"] if results else 0,
        "scoring": "learned_similarity",
        "risk_burden": risk_burden,
        "learned_numeric_weights": learned_numeric_weights,
        "cypher_query": query.strip()[:200] + "...",
    })

    return results


def project_gds_graph(driver):
    """
    Project the Patient-Symptom bipartite graph into GDS for
    advanced similarity computations. Called once after ingestion.
    """
    with driver.session() as session:
        session.run("CALL gds.graph.drop('patient-symptom', false)")

        session.run("""
            CALL gds.graph.project(
                'patient-symptom',
                ['Patient', 'Symptom'],
                {
                    HAS_SYMPTOM: {
                        type: 'HAS_SYMPTOM',
                        orientation: 'UNDIRECTED'
                    }
                }
            )
        """)

    log_event("GDSProjection", {"graph_name": "patient-symptom"})
