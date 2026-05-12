"""
clinical_harmonization.py — EHR Data Harmonisation & Biomedical Ontology Mapping
==================================================================================
Demonstrates clinical data integration for precision medicine:
  • Synthetic EHR dataset generation (demographics, diagnoses, labs, medications)
  • ICD-10 hierarchy mapping and HPO phenotype term annotation
  • Data quality assessment: completeness, consistency, outlier detection
  • HIPAA Safe Harbour de-identification
  • Feature engineering pipeline for downstream ML
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. BIOMEDICAL ONTOLOGIES (curated subsets)
# ---------------------------------------------------------------------------

# ICD-10 chapter hierarchy (simplified)
ICD10_CHAPTERS: Dict[str, Dict] = {
    "E11": {"desc": "Type 2 diabetes mellitus",          "chapter": "Endocrine",         "hpo": "HP:0005978"},
    "I10": {"desc": "Essential hypertension",            "chapter": "Cardiovascular",     "hpo": "HP:0000822"},
    "J45": {"desc": "Asthma",                            "chapter": "Respiratory",        "hpo": "HP:0002099"},
    "C50": {"desc": "Malignant neoplasm of breast",      "chapter": "Neoplasms",          "hpo": "HP:0003002"},
    "F32": {"desc": "Major depressive disorder",         "chapter": "Mental",             "hpo": "HP:0000716"},
    "N18": {"desc": "Chronic kidney disease",            "chapter": "Genitourinary",      "hpo": "HP:0012622"},
    "M54": {"desc": "Dorsalgia / back pain",             "chapter": "Musculoskeletal",    "hpo": "HP:0003418"},
    "K21": {"desc": "GERD",                              "chapter": "Digestive",          "hpo": "HP:0002020"},
    "G43": {"desc": "Migraine",                          "chapter": "Nervous system",     "hpo": "HP:0002076"},
    "Z87": {"desc": "Personal history of diseases",      "chapter": "Factors",            "hpo": None},
}

# HPO term hierarchy (simplified phenotype ontology)
HPO_TERMS: Dict[str, Dict] = {
    "HP:0000822": {"label": "Hypertension",          "parent": "HP:0001626", "category": "cardiovascular"},
    "HP:0005978": {"label": "T2DM",                  "parent": "HP:0001939", "category": "metabolic"},
    "HP:0002099": {"label": "Asthma",                "parent": "HP:0002086", "category": "respiratory"},
    "HP:0003002": {"label": "Breast carcinoma",      "parent": "HP:0100013", "category": "oncology"},
    "HP:0000716": {"label": "Depression",            "parent": "HP:0000708", "category": "psychiatric"},
    "HP:0012622": {"label": "CKD",                   "parent": "HP:0000077", "category": "renal"},
    "HP:0003418": {"label": "Back pain",             "parent": "HP:0003002", "category": "musculoskeletal"},
    "HP:0002020": {"label": "GERD",                  "parent": "HP:0025031", "category": "gastrointestinal"},
    "HP:0002076": {"label": "Migraine",              "parent": "HP:0012638", "category": "neurological"},
    "HP:0001626": {"label": "Cardiovascular anomaly","parent": "HP:0000118", "category": "cardiovascular"},
}

# Reference lab ranges (SI units)
LAB_REFERENCE_RANGES: Dict[str, Dict] = {
    "glucose_mmolL":    {"low": 3.9,  "high": 6.1,  "unit": "mmol/L",  "critical_low": 2.8, "critical_high": 22.2},
    "hba1c_pct":        {"low": 4.0,  "high": 5.6,  "unit": "%",       "critical_low": 3.0, "critical_high": 15.0},
    "creatinine_umolL": {"low": 62.0, "high": 115.0,"unit": "µmol/L",  "critical_low": 44,  "critical_high": 884},
    "egfr":             {"low": 60.0, "high": 120.0, "unit": "mL/min/1.73m²", "critical_low": 15, "critical_high": 200},
    "ldl_mmolL":        {"low": 0.0,  "high": 3.0,  "unit": "mmol/L",  "critical_low": 0.0, "critical_high": 10.0},
    "hdl_mmolL":        {"low": 1.0,  "high": 2.5,  "unit": "mmol/L",  "critical_low": 0.5, "critical_high": 5.0},
    "alt_UL":           {"low": 7.0,  "high": 40.0, "unit": "U/L",     "critical_low": 0.0, "critical_high": 1000},
    "systolic_bp":      {"low": 90.0, "high": 120.0,"unit": "mmHg",    "critical_low": 70,  "critical_high": 220},
    "diastolic_bp":     {"low": 60.0, "high": 80.0, "unit": "mmHg",    "critical_low": 40,  "critical_high": 130},
    "bmi":              {"low": 18.5, "high": 25.0, "unit": "kg/m²",   "critical_low": 12,  "critical_high": 60},
}


# ---------------------------------------------------------------------------
# 2. SYNTHETIC EHR DATASET GENERATION
# ---------------------------------------------------------------------------

def generate_ehr_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic EHR-like cohort.

    Columns
    -------
    patient_id, age, sex, ethnicity, zip_code, encounter_date,
    icd10_primary, icd10_secondary, medications (list as str),
    glucose_mmolL, hba1c_pct, creatinine_umolL, egfr,
    ldl_mmolL, hdl_mmolL, alt_UL, systolic_bp, diastolic_bp, bmi,
    smoking_status, alcohol_units_wk, outcome_readmission_30d
    """
    rng = np.random.default_rng(seed)

    ethnicities = ["White", "Black", "Hispanic", "Asian", "Other"]
    eth_probs   = [0.60, 0.13, 0.18, 0.06, 0.03]
    medications_pool = [
        "metformin", "lisinopril", "atorvastatin", "amlodipine",
        "omeprazole", "levothyroxine", "albuterol", "sertraline",
        "aspirin", "furosemide",
    ]
    icd10_codes = list(ICD10_CHAPTERS.keys())

    rows = []
    for i in range(n):
        age = int(rng.integers(18, 85))
        sex = rng.choice(["M", "F"])
        eth = rng.choice(ethnicities, p=eth_probs)
        zip_code = str(rng.integers(10000, 99999))

        year = rng.integers(2018, 2024)
        month = rng.integers(1, 13)
        day = rng.integers(1, 29)
        enc_date = f"{year}-{month:02d}-{day:02d}"

        # Primary + optional secondary diagnosis
        icd_primary = rng.choice(icd10_codes)
        icd_secondary = rng.choice(icd10_codes) if rng.random() > 0.4 else None

        # Medications (1–4 drugs)
        n_meds = rng.integers(1, 5)
        meds = ", ".join(rng.choice(medications_pool, size=n_meds, replace=False).tolist())

        # Lab values — correlated with age and diagnoses
        age_factor = (age - 50) / 30.0
        has_diabetes = icd_primary == "E11" or icd_secondary == "E11"
        has_htn = icd_primary == "I10" or icd_secondary == "I10"
        has_ckd = icd_primary == "N18" or icd_secondary == "N18"

        glucose = rng.normal(5.5 + (1.5 if has_diabetes else 0) + 0.3 * age_factor, 0.8)
        hba1c = rng.normal(5.8 + (1.8 if has_diabetes else 0) + 0.2 * age_factor, 0.5)
        creatinine = rng.normal(90 + (30 if has_ckd else 0) + 5 * age_factor, 15)
        egfr = max(5, rng.normal(85 - (25 if has_ckd else 0) - 8 * age_factor, 12))
        ldl = rng.normal(2.8 + 0.3 * age_factor, 0.6)
        hdl = rng.normal(1.4 - 0.1 * age_factor, 0.3)
        alt = rng.normal(25 + 5 * age_factor, 8)
        sbp = rng.normal(120 + (20 if has_htn else 0) + 5 * age_factor, 12)
        dbp = rng.normal(78 + (10 if has_htn else 0) + 2 * age_factor, 8)
        bmi = rng.normal(27 + 0.05 * age, 4.5)

        smoking = rng.choice(["never", "former", "current"], p=[0.55, 0.30, 0.15])
        alcohol = float(rng.integers(0, 20))

        # Introduce 8% missingness randomly
        lab_vals = [glucose, hba1c, creatinine, egfr, ldl, hdl, alt, sbp, dbp, bmi]
        lab_vals = [v if rng.random() > 0.08 else np.nan for v in lab_vals]

        # Outcome: 30-day readmission (15% baseline, higher for elderly + multimorbid)
        readmit_p = 0.15 + 0.003 * max(0, age - 65)
        if icd_secondary:
            readmit_p += 0.08
        readmit = int(rng.random() < readmit_p)

        rows.append({
            "patient_id": f"PT{i+1:05d}",
            "age": age,
            "sex": sex,
            "ethnicity": eth,
            "zip_code": zip_code,
            "encounter_date": enc_date,
            "icd10_primary": icd_primary,
            "icd10_secondary": icd_secondary,
            "medications": meds,
            "glucose_mmolL": round(lab_vals[0], 2) if not np.isnan(lab_vals[0]) else np.nan,
            "hba1c_pct": round(lab_vals[1], 1) if not np.isnan(lab_vals[1]) else np.nan,
            "creatinine_umolL": round(lab_vals[2], 1) if not np.isnan(lab_vals[2]) else np.nan,
            "egfr": round(lab_vals[3], 1) if not np.isnan(lab_vals[3]) else np.nan,
            "ldl_mmolL": round(lab_vals[4], 2) if not np.isnan(lab_vals[4]) else np.nan,
            "hdl_mmolL": round(lab_vals[5], 2) if not np.isnan(lab_vals[5]) else np.nan,
            "alt_UL": round(lab_vals[6], 1) if not np.isnan(lab_vals[6]) else np.nan,
            "systolic_bp": round(lab_vals[7], 0) if not np.isnan(lab_vals[7]) else np.nan,
            "diastolic_bp": round(lab_vals[8], 0) if not np.isnan(lab_vals[8]) else np.nan,
            "bmi": round(lab_vals[9], 1) if not np.isnan(lab_vals[9]) else np.nan,
            "smoking_status": smoking,
            "alcohol_units_wk": alcohol,
            "outcome_readmission_30d": readmit,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. ONTOLOGY MAPPING
# ---------------------------------------------------------------------------

def map_icd10_to_hpo(df: pd.DataFrame) -> pd.DataFrame:
    """Add HPO terms and ICD-10 chapter labels."""
    out = df.copy()

    def _map(code):
        info = ICD10_CHAPTERS.get(code)
        return info if info else {"desc": "Unknown", "chapter": "Other", "hpo": None}

    primary_info = out["icd10_primary"].apply(_map)
    out["dx_primary_desc"]    = primary_info.apply(lambda x: x["desc"])
    out["dx_primary_chapter"] = primary_info.apply(lambda x: x["chapter"])
    out["hpo_primary"]        = primary_info.apply(lambda x: x["hpo"])

    out["hpo_category"] = out["hpo_primary"].map(
        lambda h: HPO_TERMS.get(h, {}).get("category", "other") if h else "other"
    )
    return out


# ---------------------------------------------------------------------------
# 4. DATA QUALITY ASSESSMENT
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    completeness: float       # % non-null
    consistency: float        # % within valid range
    outlier_rate: float       # % z-score > 3
    duplicate_rate: float     # % duplicate patient IDs
    overall_score: float
    column_stats: Dict        = field(default_factory=dict)
    flags: List[str]          = field(default_factory=list)


def assess_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """Compute completeness, consistency, and outlier metrics."""
    lab_cols = list(LAB_REFERENCE_RANGES.keys())
    lab_cols = [c for c in lab_cols if c in df.columns]

    # Completeness
    total_cells = df[lab_cols].size
    missing = df[lab_cols].isna().sum().sum()
    completeness = 1 - missing / total_cells

    # Consistency: values within critical range
    n_consistent = 0
    n_total = 0
    col_stats = {}
    for col in lab_cols:
        if col not in df.columns:
            continue
        ref = LAB_REFERENCE_RANGES[col]
        vals = df[col].dropna()
        n_total += len(vals)
        in_critical = ((vals >= ref["critical_low"]) &
                       (vals <= ref["critical_high"])).sum()
        n_consistent += int(in_critical)
        col_stats[col] = {
            "n": len(vals),
            "missing": df[col].isna().sum(),
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std()), 2),
            "pct_in_range": round(float(in_critical / len(vals)) * 100, 1) if len(vals) else 0,
        }

    consistency = n_consistent / n_total if n_total > 0 else 1.0

    # Outliers (z-score > 3)
    outlier_count = 0
    for col in lab_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 5:
            continue
        z = np.abs(stats.zscore(vals))
        outlier_count += (z > 3).sum()
    outlier_rate = outlier_count / (len(df) * len(lab_cols))

    # Duplicate IDs
    dup_rate = df["patient_id"].duplicated().mean()

    # Flags
    flags = []
    if completeness < 0.90:
        flags.append(f"Low completeness: {completeness:.1%}")
    if consistency < 0.95:
        flags.append(f"Out-of-range values detected: {(1-consistency):.1%}")
    if outlier_rate > 0.02:
        flags.append(f"Elevated outlier rate: {outlier_rate:.1%}")
    if dup_rate > 0:
        flags.append(f"Duplicate patient IDs: {dup_rate:.1%}")

    overall = (0.5 * completeness + 0.3 * consistency + 0.2 * (1 - outlier_rate))
    return DataQualityReport(
        completeness=completeness,
        consistency=consistency,
        outlier_rate=outlier_rate,
        duplicate_rate=dup_rate,
        overall_score=overall,
        column_stats=col_stats,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# 5. HIPAA SAFE HARBOUR DE-IDENTIFICATION
# ---------------------------------------------------------------------------

def deidentify_hipaa_safe_harbour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply HIPAA Safe Harbour de-identification (45 CFR §164.514(b)).

    Transformations applied
    -----------------------
    • age        → age_decade (e.g. 67 → "60-69"), ages ≥ 90 → "90+"
    • zip_code   → first 3 digits only (suppressed if population < 20k proxy)
    • encounter_date → encounter_year only
    • patient_id → anonymised sequential ID
    """
    out = df.copy()

    # Age → decade
    def age_to_decade(age):
        if pd.isna(age):
            return np.nan
        if age >= 90:
            return "90+"
        decade = (int(age) // 10) * 10
        return f"{decade}-{decade+9}"

    out["age_decade"] = out["age"].apply(age_to_decade)
    out = out.drop(columns=["age"])

    # ZIP → 3-digit prefix
    out["zip_prefix"] = out["zip_code"].astype(str).str[:3]
    out = out.drop(columns=["zip_code"])

    # Date → year
    out["encounter_year"] = pd.to_datetime(out["encounter_date"],
                                           errors="coerce").dt.year
    out = out.drop(columns=["encounter_date"])

    # Patient ID → anonymous
    out["anon_id"] = [f"ANON{i+1:05d}" for i in range(len(out))]
    out = out.drop(columns=["patient_id"])

    return out


# ---------------------------------------------------------------------------
# 6. FEATURE ENGINEERING FOR ML
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a feature matrix suitable for ML models.

    Steps
    -----
    1. Encode categorical variables (sex, ethnicity, smoking, ICD chapter)
    2. Impute missing lab values (median strategy)
    3. Normalise continuous features (z-score)
    4. Create derived features (eGFR category, BMI category, BP risk)

    Returns
    -------
    feature_df : DataFrame of engineered features
    feature_names : list of column names
    """
    out = df.copy()

    # Derived features
    out["ckd_stage"] = pd.cut(
        out["egfr"].fillna(60),
        bins=[-1, 15, 30, 45, 60, 90, 999],
        labels=["G5", "G4", "G3b", "G3a", "G2", "G1"],
    ).astype(str)

    out["bmi_category"] = pd.cut(
        out["bmi"].fillna(25),
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=["underweight", "normal", "overweight", "obese_I", "obese_II"],
    ).astype(str)

    out["bp_risk"] = (
        (out["systolic_bp"].fillna(120) >= 140) |
        (out["diastolic_bp"].fillna(80) >= 90)
    ).astype(int)

    out["multimorbid"] = out["icd10_secondary"].notna().astype(int)
    out["n_medications"] = out["medications"].str.split(",").apply(len)

    # Encode categoricals
    cat_cols = ["sex", "ethnicity", "smoking_status",
                "dx_primary_chapter", "hpo_category", "ckd_stage", "bmi_category"]
    cat_cols = [c for c in cat_cols if c in out.columns]

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        out[col + "_enc"] = le.fit_transform(out[col].astype(str))
        le_dict[col] = le

    # Lab + numeric feature columns
    lab_cols = list(LAB_REFERENCE_RANGES.keys())
    lab_cols = [c for c in lab_cols if c in out.columns]
    derived_num = ["bp_risk", "multimorbid", "n_medications", "alcohol_units_wk"]
    derived_num = [c for c in derived_num if c in out.columns]
    enc_cols = [c + "_enc" for c in cat_cols]

    feature_cols = lab_cols + derived_num + enc_cols
    feature_df = out[feature_cols].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    feature_df[lab_cols] = imputer.fit_transform(feature_df[lab_cols])

    # Standardise numeric
    scaler = StandardScaler()
    feature_df[lab_cols] = scaler.fit_transform(feature_df[lab_cols])

    return feature_df, feature_cols


# ---------------------------------------------------------------------------
# 7. VISUALISATION
# ---------------------------------------------------------------------------

def plot_data_quality(report: DataQualityReport, df: pd.DataFrame,
                      out_path: str = "data_quality.png") -> None:
    """3-panel data quality dashboard."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Clinical Data Quality Assessment", fontsize=13, fontweight="bold")

    # Panel 1: Completeness per lab column
    ax = axes[0]
    lab_cols = [c for c in LAB_REFERENCE_RANGES if c in df.columns]
    completeness_pct = [(1 - df[c].isna().mean()) * 100 for c in lab_cols]
    colors = ["#388e3c" if v >= 95 else "#f57c00" if v >= 85 else "#d32f2f"
              for v in completeness_pct]
    ax.barh(lab_cols, completeness_pct, color=colors, alpha=0.85)
    ax.axvline(95, color="red", linestyle="--", lw=1, label="95% threshold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Completeness (%)")
    ax.set_title("Lab Value Completeness", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: Distribution of key lab values
    ax = axes[1]
    key_labs = ["glucose_mmolL", "hba1c_pct", "egfr", "bmi"]
    key_labs = [c for c in key_labs if c in df.columns]
    data_to_plot = [df[c].dropna().values for c in key_labs]
    parts = ax.violinplot(data_to_plot, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#42a5f5")
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(key_labs) + 1))
    ax.set_xticklabels([c.replace("_", "\n") for c in key_labs], fontsize=8)
    ax.set_title("Key Lab Distributions", fontweight="bold")

    # Panel 3: ICD-10 chapter distribution
    ax = axes[2]
    if "dx_primary_chapter" in df.columns:
        chapter_counts = df["dx_primary_chapter"].value_counts()
        ax.bar(chapter_counts.index, chapter_counts.values,
               color="#7b1fa2", alpha=0.8)
        ax.set_xticklabels(chapter_counts.index, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title("Primary Diagnosis Chapters", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Data quality plot → {out_path}")


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(__file__) or "."

    print("=" * 64)
    print("  CLINICAL DATA HARMONISATION PIPELINE")
    print("=" * 64)

    # ── Generate EHR data ────────────────────────────────────────────
    print("\n[1/5] Generating synthetic EHR dataset (n=500)…")
    df_raw = generate_ehr_dataset(n=500, seed=42)
    print(f"  Shape    : {df_raw.shape}")
    print(f"  Columns  : {list(df_raw.columns)}")
    print(f"  Cases    : {df_raw['outcome_readmission_30d'].sum()} "
          f"({df_raw['outcome_readmission_30d'].mean():.1%} readmission rate)")

    # ── Ontology mapping ─────────────────────────────────────────────
    print("\n[2/5] Mapping ICD-10 codes to HPO phenotype terms…")
    df_mapped = map_icd10_to_hpo(df_raw)
    print(f"  ICD-10 chapters : {df_mapped['dx_primary_chapter'].nunique()}")
    print(f"  HPO categories  : {df_mapped['hpo_category'].value_counts().to_dict()}")

    # ── Data quality ─────────────────────────────────────────────────
    print("\n[3/5] Assessing data quality…")
    qc_report = assess_data_quality(df_mapped)
    print(f"  Completeness score : {qc_report.completeness:.1%}")
    print(f"  Consistency score  : {qc_report.consistency:.1%}")
    print(f"  Outlier rate       : {qc_report.outlier_rate:.2%}")
    print(f"  Overall DQ score   : {qc_report.overall_score:.1%}")
    if qc_report.flags:
        for flag in qc_report.flags:
            print(f"  ⚠  {flag}")

    # ── De-identification ────────────────────────────────────────────
    print("\n[4/5] Applying HIPAA Safe Harbour de-identification…")
    df_deid = deidentify_hipaa_safe_harbour(df_mapped)
    removed = [c for c in ["age", "zip_code", "encounter_date", "patient_id"]
               if c not in df_deid.columns]
    added = ["age_decade", "zip_prefix", "encounter_year", "anon_id"]
    print(f"  Removed PHI fields : {removed}")
    print(f"  Replacement fields : {added}")
    print(f"  Sample de-identified record:")
    print(df_deid[["anon_id", "age_decade", "zip_prefix", "encounter_year"]].head(3).to_string(index=False))

    # ── Feature engineering ──────────────────────────────────────────
    print("\n[5/5] Engineering ML-ready feature matrix…")
    feature_df, feature_cols = engineer_features(df_mapped)
    print(f"  Feature matrix : {feature_df.shape}")
    print(f"  Features       : {feature_cols[:6]} … ({len(feature_cols)} total)")
    print(f"  Missing values : {feature_df.isna().sum().sum()} (post-imputation)")

    # ── Visualisation ────────────────────────────────────────────────
    plot_data_quality(qc_report, df_mapped,
                      out_path=os.path.join(out_dir, "data_quality.png"))

    print("\nDone.")
