"""
Phase 3 — Track 1: Decision Automation (Rule Engine).

Deterministic clinical safety rules derived from ACC/AHA guidelines.
Each rule fires alerts based on patient parameter thresholds.
"""

from trace_logger import log_event


RULES = [
    {
        "id": "HTN_ELDERLY",
        "name": "Hypertension Route (Elderly)",
        "condition": lambda p: p.get("age", 0) > 65 and p.get("resting_bp", 0) > 140,
        "severity": "critical",
        "message": "Pre-warning: Hypertension Route — Age > 65 with systolic BP > 140 mmHg. "
                   "ACC/AHA guideline recommends treatment target < 130/80 mmHg for high-risk patients.",
    },
    {
        "id": "HTN_STAGE1",
        "name": "Stage 1 Hypertension",
        "condition": lambda p: 130 <= p.get("resting_bp", 0) < 140,
        "severity": "warning",
        "message": "Stage 1 Hypertension detected — systolic BP 130-139 mmHg. "
                   "ACC/AHA recommends lifestyle modification and reassessment.",
    },
    {
        "id": "HTN_STAGE2",
        "name": "Stage 2 Hypertension",
        "condition": lambda p: p.get("resting_bp", 0) >= 140,
        "severity": "critical",
        "message": "Stage 2 Hypertension detected — systolic BP ≥ 140 mmHg. "
                   "ACC/AHA recommends combination antihypertensive therapy.",
    },
    {
        "id": "DM_SCREENING",
        "name": "Diabetes Screening",
        "condition": lambda p: (
            str(p.get("fasting_blood_sugar", "")).strip().upper() in ("TRUE", "1", "YES")
            and p.get("age", 0) > 50
        ),
        "severity": "warning",
        "message": "Diabetes screening recommended — fasting blood sugar > 120 mg/dL in patient aged > 50.",
    },
    {
        "id": "CARDIAC_STRESS",
        "name": "Cardiac Stress Risk",
        "condition": lambda p: (
            p.get("max_heart_rate", 999) < 100
            and str(p.get("exercise_angina", "")).strip().upper() in ("TRUE", "1", "YES")
        ),
        "severity": "critical",
        "message": "Cardiac stress risk — low maximum heart rate (< 100 bpm) with exercise-induced angina. "
                   "Consider stress testing and cardiology referral.",
    },
    {
        "id": "HIGH_CHOL",
        "name": "High Cholesterol",
        "condition": lambda p: p.get("cholesterol", 0) > 240,
        "severity": "warning",
        "message": "High serum cholesterol (> 240 mg/dL). Statin therapy and dietary counseling recommended.",
    },
    {
        "id": "ST_DEPRESSION",
        "name": "Significant ST Depression",
        "condition": lambda p: p.get("st_depression", 0) > 2.0,
        "severity": "warning",
        "message": "Significant ST depression (> 2.0 mm) during exercise — may indicate myocardial ischemia.",
    },
    {
        "id": "MULTI_VESSEL",
        "name": "Multi-Vessel Disease",
        "condition": lambda p: _safe_int(p.get("num_vessels", 0)) >= 2,
        "severity": "critical",
        "message": "Multiple vessels (≥ 2) show significant stenosis on fluoroscopy. "
                   "High risk for coronary artery disease — consider angiography.",
    },
    # --- Lifestyle / qualitative rules ---
    {
        "id": "SMOKING_CVD_RISK",
        "name": "Smoking — Cardiovascular Risk",
        "condition": lambda p: _is_true(p.get("smoking")),
        "severity": "warning",
        "message": "Smoking is a major modifiable cardiovascular risk factor (ACC/AHA). "
                   "Smoking cessation counseling and pharmacotherapy strongly recommended.",
    },
    {
        "id": "SMOKING_CHEST_PAIN",
        "name": "Smoker with Chest Pain",
        "condition": lambda p: (
            _is_true(p.get("smoking"))
            and _is_true(p.get("chest_pain_present"))
            and p.get("age", 0) >= 40
        ),
        "severity": "critical",
        "message": "Smoker aged ≥ 40 presenting with chest pain — strongly recommend cardiac workup. "
                   "Smoking combined with chest pain significantly elevates acute coronary syndrome risk.",
    },
    {
        "id": "CHEST_PAIN_EVAL",
        "name": "Chest Pain Reported",
        "condition": lambda p: _is_true(p.get("chest_pain_present")),
        "severity": "warning",
        "message": "Chest pain reported — clinical evaluation recommended. "
                   "Consider ECG, troponin, and cardiac risk stratification.",
    },
    {
        "id": "FEMALE_ATYPICAL",
        "name": "Female with Chest Pain",
        "condition": lambda p: (
            str(p.get("sex", "")).strip().lower() == "female"
            and _is_true(p.get("chest_pain_present"))
            and p.get("age", 0) >= 40
        ),
        "severity": "warning",
        "message": "Women ≥ 40 may present with atypical angina symptoms — do not dismiss chest pain. "
                   "ACC/AHA recommends same risk stratification pathway as male patients.",
    },
    {
        "id": "SEDENTARY_LIFESTYLE",
        "name": "Sedentary / Unhealthy Lifestyle",
        "condition": lambda p: _is_true(p.get("sedentary_lifestyle")),
        "severity": "info",
        "message": "Sedentary or unhealthy lifestyle identified — WHO PEN guideline recommends "
                   "at least 150 minutes/week moderate physical activity, balanced diet, and adequate sleep.",
    },
    # --- Combined / escalation rules ---
    {
        "id": "ELDERLY_HIGH_CHOL",
        "name": "Elderly with High Cholesterol",
        "condition": lambda p: p.get("age", 0) > 65 and p.get("cholesterol", 0) > 240,
        "severity": "critical",
        "message": "Patient aged > 65 with cholesterol > 240 mg/dL — high-risk for atherosclerotic "
                   "cardiovascular event. ACC/AHA recommends high-intensity statin therapy.",
    },
    {
        "id": "HTN_PLUS_HIGH_CHOL",
        "name": "Hypertension + Hypercholesterolemia",
        "condition": lambda p: p.get("resting_bp", 0) >= 140 and p.get("cholesterol", 0) > 200,
        "severity": "critical",
        "message": "Co-existing Stage 2 Hypertension and elevated cholesterol — combined cardiovascular risk "
                   "is multiplicative. Dual-target therapy (antihypertensive + statin) strongly recommended.",
    },
    {
        "id": "SMOKING_HTN_COMBO",
        "name": "Smoker with Hypertension",
        "condition": lambda p: _is_true(p.get("smoking")) and p.get("resting_bp", 0) >= 130,
        "severity": "critical",
        "message": "Smoking combined with hypertension greatly accelerates atherosclerosis. "
                   "Immediate smoking cessation and antihypertensive therapy essential.",
    },
    {
        "id": "ELDERLY_MULTI_RISK",
        "name": "Elderly Multi-Risk Profile",
        "condition": lambda p: (
            p.get("age", 0) > 65
            and p.get("resting_bp", 0) >= 130
            and _is_true(p.get("smoking"))
        ),
        "severity": "critical",
        "message": "Patient > 65 with hypertension and smoking history — very high 10-year ASCVD risk. "
                   "ACC/AHA recommends comprehensive risk factor management and cardiology referral.",
    },
    {
        "id": "SYNCOPE_CARDIAC",
        "name": "Syncope — Cardiac Evaluation",
        "condition": lambda p: (
            _is_true(p.get("syncope"))
            or _is_true(p.get("fainting"))
        ),
        "severity": "critical",
        "message": "Syncope / fainting reported — in the presence of cardiac risk factors, "
                   "cardiac syncope must be ruled out. ECG, echocardiogram, and Holter monitoring recommended.",
    },
    {
        "id": "DYSPNEA_AT_REST",
        "name": "Dyspnea at Rest",
        "condition": lambda p: _is_true(p.get("dyspnea_at_rest")),
        "severity": "critical",
        "message": "Shortness of breath at rest is an alarming sign — may indicate heart failure, "
                   "pulmonary embolism, or severe valvular disease. Urgent evaluation with BNP/NT-proBNP, "
                   "chest X-ray, and echocardiogram recommended.",
    },
]


def _safe_int(val) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _safe_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _is_true(val) -> bool:
    if val is None:
        return False
    return str(val).strip().upper() in ("TRUE", "1", "YES")


def normalize_patient_data(patient_data: dict) -> dict:
    """Coerce extracted clinical fields into stable types before rule evaluation."""
    normalized = dict(patient_data or {})
    for key in ("age", "resting_bp", "cholesterol", "max_heart_rate", "num_vessels"):
        if key in normalized:
            normalized[key] = _safe_int(normalized.get(key))
    if "st_depression" in normalized:
        normalized["st_depression"] = _safe_float(normalized.get("st_depression"))
    for key in (
        "fasting_blood_sugar",
        "exercise_angina",
        "smoking",
        "chest_pain_present",
        "sedentary_lifestyle",
        "syncope",
        "fainting",
        "dyspnea_at_rest",
    ):
        if key in normalized:
            normalized[key] = "TRUE" if _is_true(normalized.get(key)) else "FALSE"
    return normalized


def evaluate_rules(patient_data: dict) -> list[dict]:
    """
    Evaluate all rules against patient data.
    Returns a list of fired alerts sorted by severity (critical first).
    """
    normalized_data = normalize_patient_data(patient_data)
    alerts = []
    for rule in RULES:
        try:
            if rule["condition"](normalized_data):
                alerts.append({
                    "id": rule["id"],
                    "name": rule["name"],
                    "severity": rule["severity"],
                    "message": rule["message"],
                    "source": rule.get("source", "Clinical rule"),
                    "priority": rule.get("priority", 50),
                })
        except Exception:
            continue

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: (severity_order.get(a["severity"], 99), a.get("priority", 50)))

    log_event("RuleEvaluation", {
        "patient_data_keys": list(normalized_data.keys()),
        "alerts_fired": len(alerts),
        "alert_ids": [a["id"] for a in alerts],
    })

    return alerts
