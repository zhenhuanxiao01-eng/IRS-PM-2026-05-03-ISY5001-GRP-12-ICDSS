"""
ICDSS — Intelligent Clinical Decision Support System
Phase 4: Streamlit Dashboard with Triple-Track Reasoning.
"""

import os
import time
from html import escape

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from dotenv import load_dotenv

load_dotenv()

from data_pipeline import load_processed_data, run_pipeline
from ingest import run_ingestion, normalize_entity, _get_driver
from rule_engine import evaluate_rules
from similarity_engine import find_similar_cases
from cognitive_engine import run_cognitive_pipeline, extract_symptoms_from_text, extract_patient_params


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_true(value) -> bool:
    return str(value).strip().upper() in ("TRUE", "1", "YES")


def _status_from_score(score: int) -> str:
    if score >= 2:
        return "Good"
    if score == 1:
        return "Moderate"
    return "Needs Review"


def derive_symptoms_from_params(params: dict) -> list[str]:
    """Bridge numeric clinical parameters to semantic symptom/condition terms
    so Track 2 Jaccard similarity can match against enriched patient nodes."""
    derived = []
    bp = _to_float(params.get("resting_bp"))
    if bp >= 130:
        derived.append("Hypertension")

    chol = _to_float(params.get("cholesterol"))
    ldl = _to_float(params.get("ldl_cholesterol"))
    if chol >= 200 or ldl >= 160:
        derived.append("Hypercholesterolemia")

    if _is_true(params.get("smoking")):
        derived.append("Smoking")

    if _is_true(params.get("family_history_cvd")):
        derived.append("Family History of CVD")

    if _to_float(params.get("bmi")) >= 30:
        derived.append("Obesity")

    if _to_float(params.get("ascvd_risk")) >= 20:
        derived.append("High ASCVD Risk")

    if _is_true(params.get("fasting_blood_sugar")):
        derived.append("Elevated Fasting Blood Sugar")

    if _is_true(params.get("prediabetes")):
        derived.append("Prediabetes")

    if _is_true(params.get("exercise_angina")):
        derived.append("Exercise-Induced Angina")

    if _is_true(params.get("chest_pain_present")):
        derived.append("Atypical Angina")

    return derived


def build_track_metrics(results: dict) -> dict:
    """Compute MVP-visible metrics for the three reasoning tracks."""
    alerts = results.get("rule_alerts", [])
    cases = results.get("similar_cases", [])
    cognitive = results.get("cognitive", {})
    graph_context = cognitive.get("graph_context", {})

    severity_counts = {"critical": 0, "warning": 0, "info": 0}
    for alert in alerts:
        severity = alert.get("severity", "info")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    similarity_scores = [case.get("similarity", 0) for case in cases]
    top_score = max(similarity_scores) if similarity_scores else 0
    avg_top3 = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    if top_score >= 0.5:
        retrieval_confidence = "Strong"
    elif top_score >= 0.25:
        retrieval_confidence = "Moderate"
    elif top_score > 0:
        retrieval_confidence = "Weak"
    else:
        retrieval_confidence = "No Match"

    disease_count = len(graph_context.get("disease_symptoms", []))
    treatment_count = len(graph_context.get("disease_treatments", []))
    guideline_count = len(graph_context.get("guideline_links", []))
    graph_evidence_count = disease_count + treatment_count + guideline_count
    extracted_terms = len(results.get("canonical_symptoms", []))
    input_params = len(results.get("patient_params", {}))

    track1_score = 0
    track1_reasons = []
    if input_params >= 3:
        track1_score += 1
        track1_reasons.append("enough structured parameters extracted")
    else:
        track1_reasons.append("few structured parameters extracted")
    if alerts or input_params >= 5:
        track1_score += 1
        track1_reasons.append("rule engine produced an interpretable safety signal")
    else:
        track1_reasons.append("rule engine had limited input to evaluate")

    track2_score = 0
    track2_reasons = []
    if len(cases) >= 3:
        track2_score += 1
        track2_reasons.append("top-3 historical cases returned")
    else:
        track2_reasons.append("fewer than three historical cases returned")
    if top_score >= 0.35:
        track2_score += 1
        track2_reasons.append("top hybrid similarity is clinically usable")
    elif top_score >= 0.2:
        track2_reasons.append("top hybrid similarity is weak-to-moderate")
    else:
        track2_reasons.append("top hybrid similarity is weak")

    track3_score = 0
    track3_reasons = []
    if graph_evidence_count >= 3:
        track3_score += 1
        track3_reasons.append("graph evidence found")
    else:
        track3_reasons.append("limited graph evidence found")
    if cognitive.get("rationale"):
        track3_score += 1
        track3_reasons.append("LLM rationale generated")
    else:
        track3_reasons.append("LLM rationale missing")

    overall_score = track1_score + track2_score + track3_score
    if overall_score >= 5:
        overall_status = "Good"
    elif overall_score >= 3:
        overall_status = "Moderate"
    else:
        overall_status = "Needs Review"

    return {
        "track1": {
            "alerts_fired": len(alerts),
            "critical": severity_counts.get("critical", 0),
            "warning": severity_counts.get("warning", 0),
            "info": severity_counts.get("info", 0),
            "input_params": input_params,
            "status": _status_from_score(track1_score),
            "reason": "; ".join(track1_reasons),
        },
        "track2": {
            "matches_found": len(cases),
            "top_score": top_score,
            "avg_top3": avg_top3,
            "retrieval_confidence": retrieval_confidence,
            "zero_match": len(cases) == 0,
            "status": _status_from_score(track2_score),
            "reason": "; ".join(track2_reasons),
        },
        "track3": {
            "extracted_terms": extracted_terms,
            "graph_evidence": graph_evidence_count,
            "diseases": disease_count,
            "treatments": treatment_count,
            "guidelines": guideline_count,
            "rationale_generated": bool(cognitive.get("rationale")),
            "llm_available": cognitive.get("llm_available", False),
            "status": _status_from_score(track3_score),
            "reason": "; ".join(track3_reasons),
        },
        "overall": {
            "query_seconds": results.get("query_seconds", 0),
            "status": overall_status,
            "score": overall_score,
        },
    }


st.set_page_config(
    page_title="ICDSS — Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    [data-testid="stSidebar"] .stButton button {
        border-radius: 10px;
        font-weight: 700;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 52%, #0f766e 100%);
        color: #ffffff;
        padding: 30px 34px;
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
        margin-bottom: 24px;
    }
    .hero-eyebrow {
        color: #bae6fd;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        max-width: 900px;
        color: #e2e8f0;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    .hero-meta {
        margin-top: 18px;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .hero-pill {
        border: 1px solid rgba(255,255,255,0.22);
        background: rgba(255,255,255,0.12);
        color: #f8fafc;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .section-title {
        margin: 8px 0 14px;
    }
    .section-title h2 {
        font-size: 1.25rem;
        margin: 0;
        letter-spacing: -0.015em;
    }
    .section-title p {
        color: #64748b;
        margin: 4px 0 0;
        font-size: 0.92rem;
    }
    .metric-card,
    .content-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 18px;
        padding: 18px 18px 16px;
        min-height: 158px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
        margin-bottom: 14px;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 850;
        letter-spacing: -0.035em;
        margin-bottom: 6px;
    }
    .metric-subtitle {
        color: #475569;
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .status-badge {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }
    .status-good {
        color: #065f46;
        background: #d1fae5;
        border: 1px solid #a7f3d0;
    }
    .status-moderate {
        color: #92400e;
        background: #fef3c7;
        border: 1px solid #fde68a;
    }
    .status-review {
        color: #991b1b;
        background: #fee2e2;
        border: 1px solid #fecaca;
    }
    .critical-alert {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        color: #ffffff;
        padding: 14px 16px;
        border-radius: 14px;
        margin-bottom: 10px;
        font-weight: 650;
        box-shadow: 0 10px 20px rgba(153, 27, 27, 0.18);
    }
    .warning-alert {
        background: linear-gradient(135deg, #b45309 0%, #f59e0b 100%);
        color: #ffffff;
        padding: 14px 16px;
        border-radius: 14px;
        margin-bottom: 10px;
        font-weight: 650;
        box-shadow: 0 10px 20px rgba(180, 83, 9, 0.16);
    }
    .info-alert {
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%);
        color: #ffffff;
        padding: 14px 16px;
        border-radius: 14px;
        margin-bottom: 10px;
    }
    .similarity-card {
        background: #ffffff;
        color: #0f172a;
        padding: 16px;
        border-radius: 16px;
        margin-bottom: 12px;
        border: 1px solid rgba(59, 130, 246, 0.18);
        border-left: 5px solid #2563eb;
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
    }
    .rationale-box {
        background: #ffffff;
        color: #111827;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(34, 197, 94, 0.18);
        border-left: 5px solid #16a34a;
        line-height: 1.6;
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
    }
    .small-muted {
        color: #64748b;
        font-size: 0.84rem;
        line-height: 1.45;
    }
    .divider-soft {
        height: 1px;
        background: linear-gradient(90deg, rgba(148,163,184,0), rgba(148,163,184,0.45), rgba(148,163,184,0));
        margin: 24px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _status_class(status: str) -> str:
    status_key = status.lower().replace(" ", "-")
    if status_key == "needs-review":
        return "status-review"
    if status_key == "moderate":
        return "status-moderate"
    return "status-good"


def render_section(title: str, subtitle: str = ""):
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="section-title"><h2>{escape(title)}</h2>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, subtitle: str = "", status: str | None = None):
    badge = ""
    if status:
        badge = f'<span class="status-badge {_status_class(status)}">{escape(status)}</span>'
    st.markdown(
        f"""
        <div class="metric-card">
            {badge}
            <div class="metric-label">{escape(title)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-subtitle">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _wrap_label(label: str, max_len: int = 18) -> str:
    label = str(label or "")
    words = label.split()
    if not words:
        return label

    lines = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + len(word) + 1 <= max_len:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _check_neo4j_has_data() -> bool:
    """Check if Neo4j already contains patient data from a previous ingestion."""
    try:
        driver = _get_driver()
        with driver.session() as session:
            result = session.run("MATCH (p:Patient) RETURN count(p) AS cnt")
            count = result.single()["cnt"]
        driver.close()
        return count > 0
    except Exception:
        return False


def init_session_state():
    defaults = {
        "pipeline_run": False,
        "ingestion_run": False,
        "messages": [],
        "last_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state["ingestion_run"] and _check_neo4j_has_data():
        st.session_state["pipeline_run"] = True
        st.session_state["ingestion_run"] = True

    if "processed_df" not in st.session_state:
        cached_df = load_processed_data()
        if cached_df is not None:
            st.session_state["processed_df"] = cached_df
            st.session_state["pipeline_run"] = True


init_session_state()


# ─── Sidebar: Data Pipeline Controls ────────────────────────────────────────

with st.sidebar:
    st.markdown("## ICDSS Console")
    st.caption("Data operations, graph readiness, and model orchestration.")
    st.markdown("---")

    st.subheader("1. Data Pipeline")
    raw_dir = "/app/raw_data"
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")] if os.path.isdir(raw_dir) else []
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")] if os.path.isdir(raw_dir) else []

    st.caption(f"**CSVs found:** {len(csv_files)}")
    for f in csv_files:
        st.text(f"  • {f}")
    st.caption(f"**PDFs found:** {len(pdf_files)}")
    for f in pdf_files:
        st.text(f"  • {f}")

    if st.button("Run Data Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running aggregation, anonymization, and note generation..."):
            try:
                df = run_pipeline(raw_dir)
                st.session_state["processed_df"] = df
                st.session_state["pipeline_run"] = True
                st.success(f"Pipeline complete — {len(df)} records processed.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")

    st.markdown("---")

    st.subheader("2. Knowledge Graph Ingestion")
    ingestion_disabled = not st.session_state.get("pipeline_run", False)
    include_nlp = st.checkbox("Run LLM note enrichment", value=True)
    include_pdf = st.checkbox("Run PDF guideline extraction", value=True)
    if st.button(
        "Ingest into Neo4j",
        type="primary",
        use_container_width=True,
        disabled=ingestion_disabled,
    ):
        with st.spinner("Loading data into Neo4j. Optional LLM/PDF steps may take several minutes..."):
            try:
                df = st.session_state.get("processed_df")
                if df is None:
                    df = load_processed_data()
                run_ingestion(df, raw_dir, include_nlp=include_nlp, include_pdf=include_pdf)
                st.session_state["ingestion_run"] = True
                st.success("Ingestion complete — graph populated.")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    if ingestion_disabled:
        st.caption("Run the data pipeline first.")

    st.markdown("---")

    st.subheader("3. System Status")
    col_a, col_b = st.columns(2)
    col_a.metric("Pipeline", "Ready" if st.session_state["pipeline_run"] else "Pending")
    col_b.metric("Graph", "Ready" if st.session_state["ingestion_run"] else "Pending")

    st.markdown("---")
    st.caption(
        "**Disclaimer:** ICDSS is a decision *support* tool for educational purposes. "
        "All outputs require clinician review. Synthetic PII was generated via Faker for "
        "anonymization demonstration only."
    )


# ─── Main Area ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">Intelligent Clinical Decision Support System</div>
        <div class="hero-title">Triple-Track Cardiovascular Risk Review</div>
        <div class="hero-subtitle">
            A clinical decision support MVP combining deterministic safety rules,
            learned historical-case similarity, and graph-grounded cognitive synthesis.
            Designed for transparent review, not autonomous diagnosis.
        </div>
        <div class="hero-meta">
            <span class="hero-pill">Track 1: Rules</span>
            <span class="hero-pill">Track 2: Learned Similarity</span>
            <span class="hero-pill">Track 3: Cognitive RAG</span>
            <span class="hero-pill">Clinician-in-the-loop</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

user_input = st.chat_input("Describe a patient case (e.g., 'Patient aged 70 with typical angina, BP 150 mmHg, cholesterol 280 mg/dL')")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    if not st.session_state.get("ingestion_run", False):
        st.warning("Please run the data pipeline and ingestion first (sidebar controls).")
    else:
        with st.spinner("Analyzing patient case across all three tracks..."):
            query_start = time.perf_counter()

            # Track 3 (step 1): Extract symptoms and parameters
            symptoms = extract_symptoms_from_text(user_input)
            patient_params = extract_patient_params(user_input)

            # Track 1: Rule Engine
            rule_alerts = evaluate_rules(patient_params)

            # Track 2: Similarity Engine
            normalized_symptoms = [n for s in symptoms for n in [normalize_entity(s)] if n]
            derived = derive_symptoms_from_params(patient_params)
            normalized_symptoms = list(dict.fromkeys(normalized_symptoms + derived))
            similar_cases = find_similar_cases(normalized_symptoms, top_k=3, patient_params=patient_params)

            # Track 3 (steps 2-3): Graph retrieval + Cognitive synthesis
            cognitive_result = run_cognitive_pipeline(
                user_input,
                rule_alerts,
                similar_cases,
                symptoms=normalized_symptoms,
                patient_params=patient_params,
            )
            query_seconds = time.perf_counter() - query_start

            st.session_state["last_results"] = {
                "symptoms": symptoms,
                "canonical_symptoms": normalized_symptoms,
                "patient_params": patient_params,
                "rule_alerts": rule_alerts,
                "similar_cases": similar_cases,
                "cognitive": cognitive_result,
                "query_seconds": query_seconds,
            }

# ─── Display Results ─────────────────────────────────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

results = st.session_state.get("last_results")
if results:
    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)

    # Extracted info summary
    with st.expander("Extracted Patient Information", expanded=True):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.markdown("**Extracted Symptoms:**")
            if results["symptoms"]:
                for s in results["symptoms"]:
                    st.markdown(f"- {s}")
                canonical = results.get("canonical_symptoms", [])
                if canonical and canonical != results["symptoms"]:
                    st.markdown("**Canonical Track Terms:**")
                    for s in canonical:
                        st.markdown(f"- {s}")
            else:
                st.caption("No symptoms extracted.")
        with ecol2:
            st.markdown("**Clinical Parameters:**")
            if results["patient_params"]:
                for k, v in results["patient_params"].items():
                    st.markdown(f"- **{k}:** {v}")
            else:
                st.caption("No parameters extracted.")

    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)

    metrics = build_track_metrics(results)
    render_section(
        "MVP Three-Track Performance Metrics",
        "These metrics make each reasoning track explicit: Track 1 measures rule coverage, "
        "Track 2 measures retrieval strength, and Track 3 measures graph grounding and LLM availability.",
    )

    overall_col, t1_col, t2_col, t3_col = st.columns(4)
    with overall_col:
        render_metric_card(
            "Overall Query",
            f"{metrics['overall']['query_seconds']:.1f}s",
            f"Composite signal score: {metrics['overall']['score']}/6. MVP signal quality, not clinical accuracy.",
            metrics["overall"]["status"],
        )

    with t1_col:
        t1 = metrics["track1"]
        render_metric_card(
            "Track 1: Rules",
            f"{t1['alerts_fired']} alerts",
            f"Critical: {t1['critical']} | Warnings: {t1['warning']} | Params used: {t1['input_params']}. {t1['reason']}.",
            t1["status"],
        )

    with t2_col:
        t2 = metrics["track2"]
        render_metric_card(
            "Track 2: Similarity",
            f"{t2['top_score']:.1%}",
            f"{t2['matches_found']} matches | Avg top-3: {t2['avg_top3']:.1%} | Confidence: {t2['retrieval_confidence']}. {t2['reason']}.",
            t2["status"],
        )

    with t3_col:
        t3 = metrics["track3"]
        rationale_status = "Yes" if t3["rationale_generated"] else "No"
        render_metric_card(
            "Track 3: Cognitive RAG",
            f"{t3['graph_evidence']} evidence links",
            f"Guidelines: {t3['guidelines']} | Terms: {t3['extracted_terms']} | Rationale: {rationale_status}. {t3['reason']}.",
            t3["status"],
        )

    with st.expander("How To Interpret These MVP Performance Metrics"):
        st.markdown(
            """
            **Good** means the track has enough evidence to be useful for this query.
            **Moderate** means the track ran, but confidence or evidence is limited.
            **Needs Review** means the track output should not be trusted without checking data quality, graph ingestion, or extracted terms.

            - **Track 1 is good** when enough structured clinical parameters are extracted and the rule engine can produce an interpretable safety signal.
            - **Track 2 is good** when it returns top-3 historical cases and the top learned similarity is at least 35%.
            - **Track 3 is good** when graph evidence is found and the LLM generates a rationale grounded in that evidence.

            These are MVP signal-quality metrics. They show whether the system is working well enough for demonstration and review, but they are not a substitute for clinical validation with labeled test cases.
            """
        )

    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)

    # Three-pane layout
    render_section(
        "Reasoning Outputs",
        "Side-by-side outputs preserve interpretability across deterministic, statistical, and cognitive reasoning.",
    )
    col1, col2, col3 = st.columns([1, 1, 1.2])

    # ── Track 1: Rule Alerts ──
    with col1:
        st.markdown("#### Track 1: Safety Alerts")
        alerts = results.get("rule_alerts", [])
        if alerts:
            for alert in alerts:
                severity = alert["severity"]
                if severity == "critical":
                    css_class = "critical-alert"
                elif severity == "warning":
                    css_class = "warning-alert"
                else:
                    css_class = "info-alert"
                st.markdown(
                    f'<div class="{css_class}">[{severity.upper()}] {alert["name"]}<br>'
                    f'<small>{alert["message"]}</small></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No safety alerts triggered for this case.")

    # ── Track 2: Similar Cases ──
    with col2:
        st.markdown("#### Track 2: Similar Cases")
        cases = results.get("similar_cases", [])
        if cases:
            for i, case in enumerate(cases, 1):
                diag_text = "Heart Disease" if case.get("diagnosis_code", 0) > 0 else "No Disease"
                st.markdown(
                    f'<div class="similarity-card">'
                    f'<strong>Case {i}</strong> — Learned similarity: <strong>{case["similarity"]:.1%}</strong><br>'
                    f'Jaccard: {case.get("jaccard_similarity", 0):.1%} | Numeric: {case.get("numeric_similarity", 0):.1%}<br>'
                    f'Age: {case.get("age", "N/A")} | Sex: {case.get("sex", "N/A")} | '
                    f'Center: {case.get("center", "N/A")}<br>'
                    f'Diagnosis: {diag_text}'
                    f'{" (" + case.get("severity", "") + ")" if case.get("severity") else ""}<br>'
                    f'Matching symptoms: {", ".join(case.get("matching_symptoms", []))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                matched_weights = case.get("matched_term_weights", {})
                if matched_weights:
                    weight_text = ", ".join(
                        f"{term}: {weight:.2f}" for term, weight in matched_weights.items()
                    )
                    st.caption(f"Learned matched-term weights: {weight_text}")
        else:
            st.info("No similar cases found. Ensure data has been ingested.")

    # ── Track 3: AI Rationale ──
    with col3:
        st.markdown("#### Track 3: AI Rationale")
        rationale = results.get("cognitive", {}).get("rationale", "")
        if rationale:
            rationale_html = escape(rationale).replace("\n", "<br>")
            st.markdown(f'<div class="rationale-box">{rationale_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No AI rationale generated.")

    st.session_state["messages"].append({
        "role": "assistant",
        "content": "Analysis complete. See the three tracks above for detailed results.",
    })

    # ── Graph Visualization ──
    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)
    render_section(
        "Knowledge Graph Visualization",
        "A summarized view of the strongest patient, disease, guideline, treatment, and similar-case links.",
    )

    graph_ctx = results.get("cognitive", {}).get("graph_context", {})
    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    with gcol1:
        max_diseases = st.slider("Diseases", 3, 12, 6)
    with gcol2:
        max_treatments = st.slider("Treatments", 0, 12, 5)
    with gcol3:
        max_guidelines = st.slider("Guidelines", 0, 6, 2)
    with gcol4:
        show_edge_labels = st.toggle("Edge labels", value=False)

    st.caption(
        "The graph is intentionally summarized to avoid visual clutter. Increase the sliders only when you need deeper evidence exploration."
    )

    nodes = []
    edges = []
    node_ids = set()
    edge_set = set()

    def _add_node(nid, label, size, color, shape, title=None):
        if nid not in node_ids:
            nodes.append(
                Node(
                    id=nid,
                    label=_wrap_label(label),
                    title=title or label,
                    size=size,
                    color=color,
                    shape=shape,
                    font={"size": 10, "strokeWidth": 2, "strokeColor": "#0f172a"},
                )
            )
            node_ids.add(nid)

    def _add_edge(src, tgt, label, color, dashes=False):
        key = (src, tgt, label)
        if key not in edge_set:
            edge_label = label if show_edge_labels else ""
            edges.append(
                Edge(
                    source=src,
                    target=tgt,
                    label=edge_label,
                    color=color,
                    dashes=dashes,
                    font={"size": 9, "color": color},
                )
            )
            edge_set.add(key)

    # --- Layer 1: Patient symptoms (red, center) ---
    input_symptoms = results.get("canonical_symptoms", []) or results.get("symptoms", [])
    patient_id = "patient_query"
    _add_node(patient_id, "Patient Query", 34, "#dc2626", "dot")

    for s in input_symptoms[:10]:
        nid = f"input_{s}"
        _add_node(nid, s, 20, "#ef4444", "dot")
        _add_edge(patient_id, nid, "presents", "#ef4444")

    # --- Layer 2: Diseases linked to symptoms (green diamonds) ---
    seen_diseases = set()
    disease_links = graph_ctx.get("disease_symptoms", [])
    for ds in disease_links[:max_diseases]:
        d_id = f"disease_{ds['disease']}"
        _add_node(d_id, ds["disease"], 26, "#16a34a", "diamond")
        seen_diseases.add(ds["disease"])

        input_id = f"input_{ds['symptom']}"
        if input_id in node_ids:
            _add_edge(input_id, d_id, "linked to", "#f97316")
        else:
            s_id = f"symptom_{ds['symptom']}"
            _add_node(s_id, ds["symptom"], 16, "#22c55e", "dot")
            _add_edge(d_id, s_id, "HAS_SYMPTOM", "#86efac")

    # --- Layer 3: Treatments (limited globally to preserve readability) ---
    treatment_counts = {}
    treatments_added = 0
    for dt in graph_ctx.get("disease_treatments", []):
        if treatments_added >= max_treatments:
            break
        disease = dt["disease"]
        if disease not in seen_diseases:
            continue
        treatment_counts.setdefault(disease, 0)
        if treatment_counts[disease] >= 2:
            continue
        treatment_counts[disease] += 1
        treatments_added += 1

        d_id = f"disease_{disease}"
        t_id = f"treatment_{dt['treatment']}"
        if d_id in node_ids:
            _add_node(t_id, dt["treatment"], 15, "#15803d", "triangle")
            _add_edge(d_id, t_id, "TREATED_BY", "#86efac")

    # --- Layer 4: Guidelines (green squares, shortened labels) ---
    seen_guidelines = set()
    for gl in graph_ctx.get("guideline_links", []):
        if len(seen_guidelines) >= max_guidelines:
            break
        d_id = f"disease_{gl['disease']}"
        if d_id not in node_ids:
            continue
        g_name = gl["guideline"]
        if g_name in seen_guidelines:
            continue
        seen_guidelines.add(g_name)

        g_id = f"guideline_{g_name}"
        _add_node(g_id, g_name, 18, "#166534", "square")
        _add_edge(g_id, d_id, "REFERENCES", "#bbf7d0", dashes=True)

    # --- Layer 5: Similar cases (blue stars) ---
    cases_data = results.get("similar_cases", [])
    for i, case in enumerate(cases_data):
        c_id = f"case_{i}"
        _add_node(
            c_id,
            f"Case {i+1} ({case['similarity']:.0%})",
            22, "#2563eb", "star",
        )
        for ms in case.get("matching_symptoms", [])[:3]:
            ms_id = f"input_{ms}"
            if ms_id in node_ids:
                _add_edge(c_id, ms_id, "shares", "#60a5fa", dashes=True)

    if nodes:
        config = Config(
            width="100%",
            height=720,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#ffffff",
            collapsible=False,
            node={"labelProperty": "label"},
            link={"renderLabel": show_edge_labels},
            physics_solver="forceAtlas2Based",
            forceAtlas2Based={
                "gravitationalConstant": -85,
                "centralGravity": 0.008,
                "springLength": 175,
                "springConstant": 0.08,
                "avoidOverlap": 1.0,
            },
        )
        st.markdown(
            '<div style="width:100%;min-width:0;">',
            unsafe_allow_html=True,
        )
        agraph(nodes=nodes, edges=edges, config=config)
        st.markdown("</div>", unsafe_allow_html=True)

        lcol1, lcol2, lcol3 = st.columns(3)
        lcol1.markdown("**Red** — Patient's symptoms (input)")
        lcol2.markdown("**Blue** — Historical similar cases")
        lcol3.markdown("**Green** — Diseases, treatments & guidelines")
    else:
        st.caption("No graph data to visualize. Try a query with recognizable symptoms.")
