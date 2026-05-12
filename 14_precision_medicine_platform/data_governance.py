"""
data_governance.py — Data Governance, Compliance & Audit Trail
===============================================================
Regulatory-grade data governance framework for precision medicine pipelines:
  • Data quality scoring (completeness, uniqueness, validity, timeliness)
  • HIPAA Safe Harbour & GDPR pseudonymisation compliance checklist
  • Provenance tracking: per-transformation audit trail
  • Data lineage graph (NetworkX DAG)
  • Quality report generation
"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. DATA QUALITY DIMENSIONS
# ---------------------------------------------------------------------------

@dataclass
class QualityDimension:
    name: str
    score: float          # 0–1
    weight: float         # contribution to overall score
    details: Dict         = field(default_factory=dict)
    issues: List[str]     = field(default_factory=list)


@dataclass
class DataQualityAssessment:
    dataset_id: str
    assessed_at: str
    n_records: int
    n_columns: int
    dimensions: List[QualityDimension] = field(default_factory=list)
    overall_score: float = 0.0
    grade: str = "F"
    recommendations: List[str] = field(default_factory=list)

    def compute_overall(self) -> None:
        total_w = sum(d.weight for d in self.dimensions)
        self.overall_score = sum(d.score * d.weight for d in self.dimensions) / (total_w or 1)
        if self.overall_score >= 0.90:
            self.grade = "A"
        elif self.overall_score >= 0.80:
            self.grade = "B"
        elif self.overall_score >= 0.70:
            self.grade = "C"
        elif self.overall_score >= 0.60:
            self.grade = "D"
        else:
            self.grade = "F"


def assess_quality_dimensions(
    df: pd.DataFrame,
    dataset_id: str = "dataset_001",
    required_cols: Optional[List[str]] = None,
    valid_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> DataQualityAssessment:
    """
    Score a DataFrame across 4 quality dimensions.

    Dimensions
    ----------
    Completeness  — non-null rate across required columns
    Uniqueness    — absence of duplicate records
    Validity      — values within expected ranges / formats
    Timeliness    — presence and recency of date fields
    """
    assessment = DataQualityAssessment(
        dataset_id=dataset_id,
        assessed_at=datetime.utcnow().isoformat(),
        n_records=len(df),
        n_columns=len(df.columns),
    )

    # ── Completeness ──────────────────────────────────────────────────
    cols_to_check = required_cols or list(df.columns)
    cols_present = [c for c in cols_to_check if c in df.columns]
    if cols_present:
        non_null = df[cols_present].notna().mean().mean()
    else:
        non_null = 0.0
    missing_cols = [c for c in cols_to_check if c not in df.columns]
    issues_comp = []
    if missing_cols:
        issues_comp.append(f"Missing required columns: {missing_cols}")
    for col in cols_present:
        miss = df[col].isna().mean()
        if miss > 0.10:
            issues_comp.append(f"{col}: {miss:.1%} missing (>10% threshold)")
    assessment.dimensions.append(QualityDimension(
        name="Completeness", score=float(non_null), weight=0.30,
        details={"non_null_rate": round(float(non_null), 4),
                 "missing_cols": missing_cols},
        issues=issues_comp,
    ))

    # ── Uniqueness ────────────────────────────────────────────────────
    dup_rate = df.duplicated().mean()
    id_cols = [c for c in ["patient_id", "anon_id", "subject_id"] if c in df.columns]
    id_dup_rate = 0.0
    if id_cols:
        id_dup_rate = df[id_cols[0]].duplicated().mean()
    uniqueness_score = 1.0 - max(dup_rate, id_dup_rate)
    issues_uniq = []
    if dup_rate > 0:
        issues_uniq.append(f"{df.duplicated().sum()} duplicate rows detected")
    if id_dup_rate > 0:
        issues_uniq.append(f"{int(id_dup_rate * len(df))} duplicate patient IDs")
    assessment.dimensions.append(QualityDimension(
        name="Uniqueness", score=float(uniqueness_score), weight=0.25,
        details={"row_dup_rate": round(float(dup_rate), 4),
                 "id_dup_rate": round(float(id_dup_rate), 4)},
        issues=issues_uniq,
    ))

    # ── Validity ──────────────────────────────────────────────────────
    ranges = valid_ranges or {}
    in_range_rates = []
    issues_valid = []
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        in_range = ((vals >= lo) & (vals <= hi)).mean()
        in_range_rates.append(float(in_range))
        if in_range < 0.95:
            n_out = int((~((vals >= lo) & (vals <= hi))).sum())
            issues_valid.append(f"{col}: {n_out} values outside [{lo}, {hi}]")
    validity_score = float(np.mean(in_range_rates)) if in_range_rates else 1.0
    assessment.dimensions.append(QualityDimension(
        name="Validity", score=validity_score, weight=0.30,
        details={"n_range_checks": len(in_range_rates),
                 "mean_in_range": round(validity_score, 4)},
        issues=issues_valid,
    ))

    # ── Timeliness ────────────────────────────────────────────────────
    date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower()]
    timeliness_score = 1.0
    issues_time = []
    if not date_cols:
        timeliness_score = 0.7
        issues_time.append("No date/timestamp columns found — timeliness cannot be assessed")
    else:
        for col in date_cols:
            null_rate = df[col].isna().mean()
            if null_rate > 0.05:
                timeliness_score = min(timeliness_score, 1 - null_rate)
                issues_time.append(f"{col}: {null_rate:.1%} missing timestamps")
    assessment.dimensions.append(QualityDimension(
        name="Timeliness", score=timeliness_score, weight=0.15,
        details={"date_columns": date_cols},
        issues=issues_time,
    ))

    assessment.compute_overall()

    # Recommendations
    for dim in assessment.dimensions:
        if dim.score < 0.80:
            assessment.recommendations.append(
                f"Improve {dim.name} (current: {dim.score:.1%}): {dim.issues[0] if dim.issues else 'see details'}"
            )

    return assessment


# ---------------------------------------------------------------------------
# 2. COMPLIANCE CHECKLIST
# ---------------------------------------------------------------------------

@dataclass
class ComplianceItem:
    id: str
    requirement: str
    framework: str          # HIPAA | GDPR | FDA_21CFR11
    status: str             # PASS | FAIL | PARTIAL | N/A
    evidence: str = ""
    remediation: str = ""


def run_compliance_checklist(df: pd.DataFrame) -> List[ComplianceItem]:
    """
    Evaluate a DataFrame against HIPAA Safe Harbour and GDPR requirements.
    Returns list of ComplianceItem with PASS/FAIL/PARTIAL status.
    """
    items = []

    # ── HIPAA Safe Harbour (18 identifiers) ──────────────────────────
    phi_identifiers = {
        "name": ["name", "first_name", "last_name", "patient_name"],
        "geographic": ["address", "street", "city", "zip_code", "county", "state"],
        "dates": ["dob", "date_of_birth", "admission_date", "discharge_date",
                  "death_date", "encounter_date"],
        "phone": ["phone", "telephone", "fax"],
        "email": ["email", "email_address"],
        "ssn": ["ssn", "social_security"],
        "mrn": ["mrn", "medical_record_number"],
        "account": ["account_number", "account_no"],
        "certificate": ["certificate", "license_number"],
        "vehicle": ["vehicle_id", "license_plate"],
        "device": ["device_id", "serial_number"],
        "url": ["url", "website"],
        "ip": ["ip_address", "ip"],
        "biometric": ["fingerprint", "retinal", "voice_print"],
        "photo": ["photo", "image_file"],
        "other": ["patient_id"],  # direct identifiers
    }

    col_lower = {c.lower(): c for c in df.columns}
    phi_present = []
    for category, keywords in phi_identifiers.items():
        for kw in keywords:
            if kw in col_lower:
                phi_present.append((category, col_lower[kw]))

    # Age compliance: ages ≥ 90 must be grouped
    age_cols = [c for c in df.columns if c.lower() in ("age",)]
    age_compliant = True
    if age_cols:
        ages = df[age_cols[0]].dropna()
        if (ages >= 90).any():
            age_compliant = False

    items.append(ComplianceItem(
        id="HIPAA-01",
        requirement="Remove or transform all 18 PHI identifiers (Safe Harbour method)",
        framework="HIPAA",
        status="PASS" if not phi_present else "FAIL",
        evidence=f"PHI columns found: {[v for _, v in phi_present]}" if phi_present else "No direct PHI columns detected",
        remediation="Apply deidentify_hipaa_safe_harbour() to remove remaining PHI" if phi_present else "",
    ))

    items.append(ComplianceItem(
        id="HIPAA-02",
        requirement="Ages ≥ 90 must be aggregated into a single category",
        framework="HIPAA",
        status="PASS" if age_compliant else "FAIL",
        evidence="No raw age column present" if not age_cols else
                 ("Ages ≥ 90 not present" if age_compliant else "Ages ≥ 90 found in raw age column"),
        remediation="Replace exact ages ≥ 90 with '90+' group" if not age_compliant else "",
    ))

    zip_cols = [c for c in df.columns if "zip" in c.lower()]
    zip_ok = all(df[c].astype(str).str.len().max() <= 3 for c in zip_cols) if zip_cols else True
    items.append(ComplianceItem(
        id="HIPAA-03",
        requirement="ZIP codes: only first 3 digits may be retained",
        framework="HIPAA",
        status="PASS" if zip_ok else "FAIL" if zip_cols else "N/A",
        evidence=f"ZIP columns: {zip_cols}" if zip_cols else "No ZIP columns present",
        remediation="Truncate ZIP codes to 3-digit prefix" if not zip_ok else "",
    ))

    date_raw = [c for c in df.columns
                if any(d in c.lower() for d in ["encounter_date", "dob", "admission"])]
    items.append(ComplianceItem(
        id="HIPAA-04",
        requirement="Full dates must be generalised to year only",
        framework="HIPAA",
        status="FAIL" if date_raw else "PASS",
        evidence=f"Full date columns present: {date_raw}" if date_raw else "No full date columns detected",
        remediation="Replace full dates with encounter_year" if date_raw else "",
    ))

    # ── GDPR ─────────────────────────────────────────────────────────
    items.append(ComplianceItem(
        id="GDPR-01",
        requirement="Personal data must be pseudonymised (Art. 4(5))",
        framework="GDPR",
        status="PASS" if not phi_present else "PARTIAL",
        evidence="anon_id present — direct identifiers replaced" if "anon_id" in df.columns else "Pseudonymisation not confirmed",
        remediation="Ensure patient_id replaced with anonymous token",
    ))

    items.append(ComplianceItem(
        id="GDPR-02",
        requirement="Special category data (health data) requires explicit consent or Article 9 exemption",
        framework="GDPR",
        status="PARTIAL",
        evidence="Health data present — consent/legal basis must be documented externally",
        remediation="Document legal basis in data processing agreement (DPA)",
    ))

    items.append(ComplianceItem(
        id="GDPR-03",
        requirement="Data minimisation — only collect data necessary for stated purpose (Art. 5(1)(c))",
        framework="GDPR",
        status="PARTIAL",
        evidence=f"Dataset contains {len(df.columns)} columns — review for necessity",
        remediation="Remove columns not required for the defined analytical purpose",
    ))

    # ── FDA 21 CFR Part 11 (Electronic Records) ───────────────────────
    items.append(ComplianceItem(
        id="FDA-01",
        requirement="Audit trail: system-generated, tamper-evident record of all changes",
        framework="FDA_21CFR11",
        status="PARTIAL",
        evidence="ProvenanceTracker provides transformation audit trail",
        remediation="Ensure audit trail is stored separately from mutable data",
    ))

    return items


# ---------------------------------------------------------------------------
# 3. PROVENANCE TRACKER (AUDIT TRAIL)
# ---------------------------------------------------------------------------

@dataclass
class TransformationRecord:
    step_id: str
    name: str
    description: str
    input_hash: str
    output_hash: str
    n_rows_in: int
    n_rows_out: int
    n_cols_in: int
    n_cols_out: int
    timestamp: str
    parameters: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class ProvenanceTracker:
    """
    Tracks data transformations as an immutable audit trail.
    Each transformation is hashed for tamper detection.
    """

    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id
        self.records: List[TransformationRecord] = []
        self._step_counter = 0

    def _hash_df(self, df: pd.DataFrame) -> str:
        """Compute a stable hash of a DataFrame's content."""
        h = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()
        return h[:12]

    def record(
        self,
        name: str,
        description: str,
        df_in: pd.DataFrame,
        df_out: pd.DataFrame,
        parameters: Optional[Dict] = None,
        warnings: Optional[List[str]] = None,
    ) -> TransformationRecord:
        """Record a transformation step."""
        self._step_counter += 1
        rec = TransformationRecord(
            step_id=f"STEP_{self._step_counter:03d}",
            name=name,
            description=description,
            input_hash=self._hash_df(df_in),
            output_hash=self._hash_df(df_out),
            n_rows_in=len(df_in),
            n_rows_out=len(df_out),
            n_cols_in=len(df_in.columns),
            n_cols_out=len(df_out.columns),
            timestamp=datetime.utcnow().isoformat(),
            parameters=parameters or {},
            warnings=warnings or [],
        )
        self.records.append(rec)
        return rec

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    def verify_chain(self) -> bool:
        """
        Verify hash chain integrity — each step's input_hash must match
        the previous step's output_hash.
        """
        for i in range(1, len(self.records)):
            if self.records[i].input_hash != self.records[i - 1].output_hash:
                return False
        return True


# ---------------------------------------------------------------------------
# 4. DATA LINEAGE GRAPH
# ---------------------------------------------------------------------------

def build_lineage_graph(tracker: ProvenanceTracker) -> Optional[Any]:
    """Build a NetworkX DAG of the data transformation pipeline."""
    if not HAS_NX:
        return None

    G = nx.DiGraph()
    G.graph["pipeline_id"] = tracker.pipeline_id

    # Add source node
    G.add_node("SOURCE", type="source", label="Raw Data")

    prev_node = "SOURCE"
    for rec in tracker.records:
        node_id = rec.step_id
        G.add_node(node_id,
                   type="transformation",
                   label=rec.name,
                   description=rec.description,
                   rows_in=rec.n_rows_in,
                   rows_out=rec.n_rows_out,
                   timestamp=rec.timestamp)
        G.add_edge(prev_node, node_id,
                   input_hash=rec.input_hash,
                   output_hash=rec.output_hash)
        prev_node = node_id

    # Add sink node
    G.add_node("SINK", type="sink", label="ML-Ready Dataset")
    G.add_edge(prev_node, "SINK")

    return G


def plot_lineage_graph(
    G,
    out_path: str = "data_lineage.png",
) -> None:
    """Visualise the data lineage DAG."""
    if G is None:
        print("  NetworkX not available — skipping lineage graph.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_title("Data Lineage Graph", fontsize=12, fontweight="bold")

    # Manual layout: left to right
    nodes = list(nx.topological_sort(G))
    pos = {node: (i * 2.5, 0) for i, node in enumerate(nodes)}

    node_colors = []
    node_labels = {}
    for n in nodes:
        t = G.nodes[n].get("type", "transformation")
        node_colors.append(
            "#388e3c" if t == "source" else
            "#d32f2f" if t == "sink" else "#1976d2"
        )
        node_labels[n] = G.nodes[n].get("label", n)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=1200, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=7, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowsize=20, edge_color="#555555",
                           connectionstyle="arc3,rad=0.1")

    legend_items = [
        plt.Rectangle((0, 0), 1, 1, fc="#388e3c", label="Source"),
        plt.Rectangle((0, 0), 1, 1, fc="#1976d2", label="Transformation"),
        plt.Rectangle((0, 0), 1, 1, fc="#d32f2f", label="Sink"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Lineage graph → {out_path}")


# ---------------------------------------------------------------------------
# 5. QUALITY REPORT GENERATOR
# ---------------------------------------------------------------------------

def generate_quality_report(
    assessment: DataQualityAssessment,
    compliance: List[ComplianceItem],
    tracker: ProvenanceTracker,
    out_path: str = "quality_report.txt",
) -> str:
    """Generate a formatted text quality report."""
    lines = [
        "╔" + "═" * 70 + "╗",
        "║  DATA GOVERNANCE & QUALITY REPORT" + " " * 35 + "║",
        "╠" + "═" * 70 + "╣",
        f"║  Dataset ID   : {assessment.dataset_id:<53} ║",
        f"║  Assessed at  : {assessment.assessed_at[:19]:<53} ║",
        f"║  Records      : {assessment.n_records:<53} ║",
        f"║  Columns      : {assessment.n_columns:<53} ║",
        "╠" + "═" * 70 + "╣",
        "║  DATA QUALITY SCORES" + " " * 49 + "║",
        "╠" + "═" * 70 + "╣",
    ]

    for dim in assessment.dimensions:
        bar_len = int(dim.score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        status = "✓" if dim.score >= 0.80 else "⚠" if dim.score >= 0.60 else "✗"
        line = f"  {status} {dim.name:<14} [{bar}] {dim.score:.1%}"
        lines.append(f"║{line:<70} ║")

    lines += [
        "╠" + "═" * 70 + "╣",
        f"║  Overall Score : {assessment.overall_score:.1%}   Grade: {assessment.grade}" + " " * 42 + "║",
        "╠" + "═" * 70 + "╣",
        "║  COMPLIANCE STATUS" + " " * 51 + "║",
        "╠" + "═" * 70 + "╣",
    ]

    for item in compliance:
        icon = "✓" if item.status == "PASS" else "✗" if item.status == "FAIL" else "~"
        line = f"  {icon} [{item.framework}] {item.id}: {item.status}"
        lines.append(f"║{line:<70} ║")

    pass_count = sum(1 for i in compliance if i.status == "PASS")
    fail_count = sum(1 for i in compliance if i.status == "FAIL")

    lines += [
        "╠" + "═" * 70 + "╣",
        f"║  Compliance: {pass_count} PASS  {fail_count} FAIL  "
        f"{len(compliance)-pass_count-fail_count} PARTIAL/N/A" + " " * 28 + "║",
        "╠" + "═" * 70 + "╣",
        "║  AUDIT TRAIL SUMMARY" + " " * 49 + "║",
        "╠" + "═" * 70 + "╣",
    ]

    for rec in tracker.records:
        line = f"  {rec.step_id}  {rec.name:<25}  rows: {rec.n_rows_in}→{rec.n_rows_out}"
        lines.append(f"║{line:<70} ║")

    chain_ok = tracker.verify_chain()
    lines += [
        "╠" + "═" * 70 + "╣",
        f"║  Hash chain integrity: {'VERIFIED ✓' if chain_ok else 'BROKEN ✗'}" + " " * 43 + "║",
        "╚" + "═" * 70 + "╝",
    ]

    if assessment.recommendations:
        lines += ["", "RECOMMENDATIONS:", ""]
        for i, rec in enumerate(assessment.recommendations, 1):
            lines.append(f"  {i}. {rec}")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)

    return report


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from clinical_harmonization import (
        generate_ehr_dataset,
        map_icd10_to_hpo,
        deidentify_hipaa_safe_harbour,
        engineer_features,
        LAB_REFERENCE_RANGES,
    )

    out_dir = os.path.dirname(__file__) or "."

    print("=" * 64)
    print("  DATA GOVERNANCE & COMPLIANCE FRAMEWORK")
    print("=" * 64)

    # ── Build pipeline with provenance tracking ──────────────────────
    tracker = ProvenanceTracker(pipeline_id="PREC_MED_PIPELINE_001")

    print("\n[1/5] Generating raw EHR dataset…")
    df_raw = generate_ehr_dataset(n=500, seed=42)
    print(f"  Raw shape: {df_raw.shape}")

    print("\n[2/5] Running pipeline with provenance tracking…")

    # Step 1: Ontology mapping
    df_mapped = map_icd10_to_hpo(df_raw)
    tracker.record(
        name="OntologyMapping",
        description="Map ICD-10 codes to HPO phenotype terms and chapter labels",
        df_in=df_raw, df_out=df_mapped,
        parameters={"ontology": "ICD-10 + HPO", "version": "2024"},
    )

    # Step 2: HIPAA de-identification
    df_deid = deidentify_hipaa_safe_harbour(df_mapped)
    tracker.record(
        name="DeIdentification",
        description="HIPAA Safe Harbour: age→decade, zip→prefix, date→year, ID→anon",
        df_in=df_mapped, df_out=df_deid,
        parameters={"method": "HIPAA Safe Harbour", "standard": "45 CFR §164.514(b)"},
    )

    # Step 3: Feature engineering
    feat_df, feat_cols = engineer_features(df_deid)
    tracker.record(
        name="FeatureEngineering",
        description="Encode categoricals, impute missing values, standardise labs",
        df_in=df_deid, df_out=feat_df,
        parameters={"imputation": "median", "scaling": "z-score"},
    )

    print(f"  Recorded {len(tracker.records)} transformation steps")
    print(f"  Hash chain integrity: {'VERIFIED' if tracker.verify_chain() else 'BROKEN'}")

    # ── Quality assessment ───────────────────────────────────────────
    print("\n[3/5] Assessing data quality on de-identified dataset…")
    valid_ranges = {k: (v["critical_low"], v["critical_high"])
                   for k, v in LAB_REFERENCE_RANGES.items()}
    assessment = assess_quality_dimensions(
        df_deid,
        dataset_id="PREC_MED_DEID_V1",
        valid_ranges={k: v for k, v in valid_ranges.items() if k in df_deid.columns},
    )
    print(f"  Overall DQ score : {assessment.overall_score:.1%}  Grade: {assessment.grade}")
    for dim in assessment.dimensions:
        icon = "✓" if dim.score >= 0.80 else "⚠"
        print(f"  {icon} {dim.name:<15} {dim.score:.1%}")

    # ── Compliance checklist ─────────────────────────────────────────
    print("\n[4/5] Running compliance checklist (HIPAA + GDPR)…")
    compliance = run_compliance_checklist(df_deid)
    for item in compliance:
        icon = "✓" if item.status == "PASS" else "✗" if item.status == "FAIL" else "~"
        print(f"  {icon} [{item.framework}] {item.id}: {item.status}")
        if item.remediation:
            print(f"      → {item.remediation}")

    pass_n = sum(1 for i in compliance if i.status == "PASS")
    print(f"\n  {pass_n}/{len(compliance)} checks passed")

    # ── Reports & lineage graph ──────────────────────────────────────
    print("\n[5/5] Generating quality report and lineage graph…")
    report = generate_quality_report(
        assessment, compliance, tracker,
        out_path=os.path.join(out_dir, "quality_report.txt"),
    )
    print(report)

    if HAS_NX:
        G = build_lineage_graph(tracker)
        plot_lineage_graph(G, out_path=os.path.join(out_dir, "data_lineage.png"))
    else:
        print("  (networkx not installed — lineage graph skipped)")

    print("\nDone.")
