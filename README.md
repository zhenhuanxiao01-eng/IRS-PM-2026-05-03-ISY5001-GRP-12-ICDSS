# [ Practice Module ] Project Submission

---

## SECTION 1 : PROJECT TITLE
## Intelligent Clinical Decision Support System (ICDSS)

**Project Theme:** A Hybrid Reasoning Framework for Explainable Cardiovascular Decision Support


---

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT

Cardiovascular diseases remain one of the leading causes of mortality globally. In clinical environments, healthcare practitioners must interpret structured patient measurements, risk factors, guideline knowledge, and historical case evidence under time pressure. While artificial intelligence can support this process, many healthcare AI systems face trust and adoption barriers because their outputs are difficult to explain.

This project develops an **Intelligent Clinical Decision Support System (ICDSS)** that focuses on explainable cardiovascular risk review. Instead of relying on one black-box prediction model, the system integrates three intelligent reasoning tracks:

1. **Decision Automation** - deterministic clinical safety alerts using a Python rule engine.
2. **Knowledge Discovery** - learned similarity matching against historical UCI Heart Disease patient cases.
3. **Cognitive Systems** - Neo4j graph retrieval and Llama-based graph-grounded clinical rationale generation.

The system uses the UCI Heart Disease dataset as the structured clinical data source. The dataset is processed through a privacy-aware data pipeline that validates clinical fields, generates synthetic PII for demonstration, hashes synthetic patient identifiers, bucketizes key medical measurements, and creates synthetic clinical notes for enrichment. Patient records and clinical concepts are then ingested into Neo4j as a knowledge graph.

The application is delivered as a Dockerized Streamlit dashboard. A one-click Windows launcher, `start.bat`, starts Neo4j, Ollama, and the Streamlit application. The user enters a natural-language patient case, and the system displays extracted patient information, rule alerts, learned similar cases, AI rationale, three-track performance metrics, and an interactive knowledge graph visualization.

The key contribution of the project is the transparent reasoning chain. Each system output is connected to deterministic rules, similar historical cases, graph evidence, or constrained LLM synthesis. The system is intended as an educational clinical decision support MVP and not as a certified medical diagnostic tool.

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name | Student ID | Work Items (Who Did What) | Email (Optional) |
| :--- | :---: | :--- | :--- |
| Xiao Zhenhuan | A0340523R | Individual project owner. Designed the ICDSS concept, implemented the Dockerized architecture, built the Streamlit UI, developed the data pipeline, Neo4j ingestion, rule engine, learned similarity engine, cognitive Graph-RAG workflow, metrics dashboard, graph visualization, one-click launcher, README, and project report. | - |

---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

[![Watch the video](https://img.youtube.com/vi/SfBAu9MkzxA/maxresdefault.jpg)](https://www.youtube.com/watch?v=SfBAu9MkzxA)
[![Watch the video](https://img.youtube.com/vi/yxjuJdJjMc4/maxresdefault.jpg)](https://www.youtube.com/watch?v=yxjuJdJjMc4)
---

## SECTION 5 : USER GUIDE

### [ 1 ] Prerequisites

The system is designed to run locally through Docker.

Required software:

- Windows machine
- Docker Desktop installed and running
- Git, if cloning from a repository
- Recommended 16 GB RAM for the default `llama3:8b` model

If the machine has limited GPU memory, the launcher can automatically switch to a lighter model, `llama3.2:1b`, for more reliable startup.

### [ 2 ] One-Click Startup

From the project folder:

```text
RS_Practice_Module/
```

Double-click:

```text
start.bat
```

The launcher will automatically:

- check whether Docker is installed
- check whether Docker Desktop is running
- create required folders if missing
- create `.env` if missing
- detect NVIDIA GPU support
- configure CPU or GPU mode
- build and start Docker services
- wait for the Streamlit health check
- open the browser at `http://127.0.0.1:8501`

### [ 3 ] Manual Startup

If manual startup is preferred:

```powershell
cd C:\Users\zhenh\Documents\PROJECTS\RS_Practice_Module
docker compose up -d --build
```

Open the browser:

```text
http://127.0.0.1:8501
```

Neo4j browser is available at:

```text
http://127.0.0.1:7474
```

### [ 4 ] Running the ICDSS Workflow

1. Start the system using `start.bat`.
2. Confirm that the Streamlit dashboard opens.
3. In the sidebar, confirm that `heart_disease_uci.csv` is detected under `raw_data/`.
4. Click **Run Data Pipeline**.
5. Click **Ingest into Neo4j**.
6. Enter a patient case in the chat input box.
7. Review the outputs:
   - extracted clinical parameters
   - Track 1 safety alerts
   - Track 2 learned similar historical cases
   - Track 3 AI clinical rationale
   - MVP performance metrics
   - knowledge graph visualization

### [ 5 ] Main System Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard and triple-track orchestration |
| `data_pipeline.py` | CSV validation, anonymization, bucketization, synthetic note generation |
| `ingest.py` | Neo4j graph ingestion and optional PDF/LLM enrichment |
| `rule_engine.py` | Track 1 decision automation |
| `similarity_engine.py` | Track 2 learned similarity matching |
| `cognitive_engine.py` | Track 3 graph retrieval and LLM rationale |
| `trace_logger.py` | Audit trace logging |
| `docker-compose.yml` | Docker service orchestration |
| `start.bat` | One-click Windows launcher |

---

## SECTION 6 : PROJECT REPORT / PAPER

Refer to the project report in this repository:

```text
./projectreport/IRS-PM-2026-05-03-ISY5001-GRP-12-ICDSS-Group-Report.pdf
```

---

## SECTION 7 : MISCELLANEOUS

### Dataset

The project uses the UCI Heart Disease dataset:

```text
raw_data/heart_disease_uci.csv
```

The dataset is de-identified and does not contain real PII. Synthetic PII is generated only to demonstrate anonymization and is removed after hashing.

### Generated Runtime Folders

The following folders are generated or populated during runtime and should not be treated as source files:

```text
data/
processed_data/
```


### References

- Grand View Research. (2024). Clinical Decision Support Systems Market Size, Share & Trends Analysis Report.
- McKinsey Global Institute. (2011). Big Data: The Next Frontier for Innovation, Competition, and Productivity.
- National Institutes of Health. (2023). Explainable Artificial Intelligence for Healthcare.
- UCI Machine Learning Repository. (1988). Heart Disease Data Set.
- World Health Organization. (2021). Cardiovascular diseases fact sheet.

---

**Course Context:** Practice Module project for intelligent reasoning systems.
