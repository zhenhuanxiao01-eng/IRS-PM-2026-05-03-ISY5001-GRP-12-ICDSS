"""
Phase 2 — Task 2.1: Data Aggregation, Synthetic PII, Anonymization, Bucketization,
and Synthetic Clinical Note Generation.

TRANSPARENCY NOTE: This pipeline adds *synthetic* PII columns (Name, DOB, Patient ID)
via the Faker library purely to demonstrate a GDPR-style anonymization workflow.
The original UCI Heart Disease dataset contains NO real PII.
"""

import hashlib
import hmac
import os
from glob import glob

import pandas as pd
from faker import Faker

from trace_logger import log_event

fake = Faker()
Faker.seed(42)

SALT_KEY = os.getenv("SALT_KEY", "default-salt-key")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "/app/processed_data")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "patient_master.csv")

REQUIRED_COLUMNS = {
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "st_depression",
    "num_vessels",
    "thal_defect",
    "diagnosis",
}

COLUMN_MAP = {
    "id": "id",
    "age": "age",
    "sex": "sex",
    "dataset": "dataset",
    "cp": "chest_pain_type",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "thalch": "max_heart_rate",
    "exang": "exercise_angina",
    "oldpeak": "st_depression",
    "slope": "slope",
    "ca": "num_vessels",
    "thal": "thal_defect",
    "num": "diagnosis",
}

BP_CATEGORIES = [
    (120, "Normal"),
    (130, "Elevated"),
    (140, "Stage 1 Hypertension"),
    (180, "Stage 2 Hypertension"),
    (float("inf"), "Hypertensive Crisis"),
]

CHOL_CATEGORIES = [
    (200, "Desirable"),
    (240, "Borderline High"),
    (float("inf"), "High"),
]


def _categorize(value, categories):
    if pd.isna(value):
        return "Unknown"
    for threshold, label in categories:
        if value < threshold:
            return label
    return categories[-1][1]


def _age_bucket(age):
    if pd.isna(age):
        return "Unknown"
    decade = int(age) // 10 * 10
    return f"{decade}s"


def _heart_rate_category(hr):
    if pd.isna(hr):
        return "Unknown"
    if hr < 60:
        return "Bradycardia"
    if hr <= 100:
        return "Normal"
    return "Tachycardia"


def aggregate_csvs(raw_data_dir: str = "/app/raw_data") -> pd.DataFrame:
    csv_files = glob(os.path.join(raw_data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.rename(columns=COLUMN_MAP, inplace=True)
    log_event("DataAggregation", {"csv_count": len(csv_files), "total_rows": len(combined)})
    return combined


def validate_input_data(df: pd.DataFrame) -> dict:
    """Validate required clinical fields before downstream anonymization/ingestion."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    numeric_columns = [
        "age",
        "resting_bp",
        "cholesterol",
        "max_heart_rate",
        "st_depression",
        "num_vessels",
        "diagnosis",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    diagnosis_values = set(df["diagnosis"].dropna().astype(int).unique())
    invalid_diagnoses = sorted(diagnosis_values - {0, 1, 2, 3, 4})
    if invalid_diagnoses:
        raise ValueError(f"Invalid diagnosis values: {invalid_diagnoses}")

    missing_summary = {
        col: int(df[col].isna().sum())
        for col in REQUIRED_COLUMNS
        if int(df[col].isna().sum()) > 0
    }
    report = {
        "row_count": len(df),
        "missing_values": missing_summary,
        "diagnosis_distribution": df["diagnosis"].value_counts(dropna=False).to_dict(),
    }
    log_event("DataQualityCheck", report)
    return report


def augment_synthetic_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic Name, DOB, and Patient ID columns (for anonymization demo only)."""
    df["synthetic_name"] = [fake.name() for _ in range(len(df))]
    df["synthetic_dob"] = [fake.date_of_birth(minimum_age=20, maximum_age=90).isoformat() for _ in range(len(df))]
    df["synthetic_patient_id"] = [fake.uuid4() for _ in range(len(df))]
    log_event("SyntheticPIIAugmentation", {"rows_augmented": len(df)})
    return df


def anonymize(df: pd.DataFrame) -> pd.DataFrame:
    """Hash synthetic Patient IDs with HMAC-SHA256, then drop PII columns."""
    def _hash_id(pid):
        return hmac.new(
            SALT_KEY.encode(), str(pid).encode(), hashlib.sha256
        ).hexdigest()

    df["hashed_patient_id"] = df["synthetic_patient_id"].apply(_hash_id)

    sample_hash = df["hashed_patient_id"].iloc[0] if len(df) > 0 else "N/A"
    log_event("Anonymization", {
        "hashed_id": sample_hash[:12] + "...",
        "pii_removed": True,
    })

    df.drop(columns=["synthetic_name", "synthetic_dob", "synthetic_patient_id"], inplace=True)
    return df


def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert continuous metrics to categorical labels."""
    df["age_group"] = df["age"].apply(_age_bucket)
    df["bp_category"] = df["resting_bp"].apply(lambda v: _categorize(v, BP_CATEGORIES))
    df["chol_category"] = df["cholesterol"].apply(lambda v: _categorize(v, CHOL_CATEGORIES))
    df["hr_category"] = df["max_heart_rate"].apply(lambda v: _heart_rate_category(v))
    log_event("Bucketization", {"rows_processed": len(df)})
    return df


def _build_description(row) -> str:
    """Generate a synthetic clinical note from structured fields."""
    parts = []

    age = row.get("age")
    sex = row.get("sex")
    if pd.notna(age) and pd.notna(sex):
        parts.append(f"Patient is a {int(age)}-year-old {sex.lower()}")

    cp = row.get("chest_pain_type")
    if pd.notna(cp):
        parts.append(f"presenting with {cp}")

    bp = row.get("resting_bp")
    if pd.notna(bp):
        parts.append(f"resting blood pressure {int(bp)} mmHg")

    chol = row.get("cholesterol")
    if pd.notna(chol):
        parts.append(f"serum cholesterol {int(chol)} mg/dL")

    hr = row.get("max_heart_rate")
    if pd.notna(hr):
        parts.append(f"maximum heart rate achieved {int(hr)} bpm")

    fbs = row.get("fasting_blood_sugar")
    if pd.notna(fbs):
        fbs_str = str(fbs).strip().upper()
        if fbs_str in ("TRUE", "1", "YES"):
            parts.append("fasting blood sugar above 120 mg/dL")
        else:
            parts.append("fasting blood sugar within normal range")

    exang = row.get("exercise_angina")
    if pd.notna(exang):
        exang_str = str(exang).strip().upper()
        if exang_str in ("TRUE", "1", "YES"):
            parts.append("exercise-induced angina present")
        else:
            parts.append("no exercise-induced angina")

    ecg = row.get("resting_ecg")
    if pd.notna(ecg):
        parts.append(f"resting ECG shows {ecg}")

    if not parts:
        return "No clinical information available."

    return ". ".join(parts) + "."


def generate_synthetic_notes(df: pd.DataFrame) -> pd.DataFrame:
    df["description"] = df.apply(_build_description, axis=1)
    log_event("SyntheticNoteGeneration", {"notes_generated": len(df)})
    return df


def save_processed_data(df: pd.DataFrame, output_path: str = PROCESSED_DATA_FILE) -> str:
    """Persist processed data so ingestion can be rerun without repeating the pipeline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    log_event("ProcessedDataSaved", {"path": output_path, "rows": len(df)})
    return output_path


def load_processed_data(path: str = PROCESSED_DATA_FILE) -> pd.DataFrame | None:
    """Load the last processed dataset if it exists."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    log_event("ProcessedDataLoaded", {"path": path, "rows": len(df)})
    return df


def run_pipeline(raw_data_dir: str = "/app/raw_data") -> pd.DataFrame:
    """Execute the full data pipeline and return the processed DataFrame."""
    df = aggregate_csvs(raw_data_dir)
    validate_input_data(df)
    df = augment_synthetic_pii(df)
    df = anonymize(df)
    df = bucketize(df)
    df = generate_synthetic_notes(df)
    save_processed_data(df)
    log_event("PipelineComplete", {"final_row_count": len(df), "columns": list(df.columns)})
    return df
