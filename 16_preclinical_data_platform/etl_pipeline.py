"""
etl_pipeline.py — 4-step ETL pipeline for preclinical assay data.

Mirrors Azure Data Factory / Airflow patterns: Extract → Validate → Transform → Load.
Includes FAIR metadata capture, unit normalisation, audit trail, and error log.
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed
except ImportError:
    def set_global_seed(s=42): np.random.seed(s)

from schema_design import create_database, populate_database, ASSAY_CATALOGUE

# ──────────────────────────────────────────────────────────────────────────────
# Data contracts
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RawRecord:
    compound_id:  str
    assay_id:     str
    raw_value:    float
    raw_unit:     str
    operator:     str
    run_date:     str
    instrument:   str
    batch_id:     str
    n_replicates: int = 1
    notes:        str = ""


@dataclass
class ValidationError:
    record_id:  str
    field:      str
    message:    str
    severity:   str   # "error" | "warning"


@dataclass
class TransformedRecord:
    compound_id:  str
    assay_id:     str
    value_si:     float          # canonical SI unit (nM for IC50)
    unit_si:      str
    ic50_nm:      Optional[float]
    qualifier:    str
    hill_slope:   float
    r_squared:    float
    n_replicates: int
    cv_pct:       float
    operator:     str
    run_date:     str
    batch_id:     str
    qc_flag:      str
    # FAIR metadata
    source_file:  str = ""
    source_hash:  str = ""
    etl_version:  str = "1.0.0"
    etl_timestamp: str = ""


@dataclass
class ETLReport:
    n_extracted:   int = 0
    n_validated:   int = 0
    n_rejected:    int = 0
    n_transformed: int = 0
    n_loaded:      int = 0
    errors:        List[ValidationError] = field(default_factory=list)
    warnings:      List[ValidationError] = field(default_factory=list)
    fair_score:    float = 0.0
    run_timestamp: str = ""
    source_hash:   str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Unit conversion table
# ──────────────────────────────────────────────────────────────────────────────

UNIT_CONVERSIONS: Dict[str, Tuple[float, str]] = {
    # (multiplier_to_SI, canonical_unit)
    "nM":       (1.0,     "nM"),
    "nm":       (1.0,     "nM"),
    "nanomolar":(1.0,     "nM"),
    "µM":       (1000.0,  "nM"),
    "uM":       (1000.0,  "nM"),
    "micromolar":(1000.0, "nM"),
    "mM":       (1e6,     "nM"),
    "millimolar":(1e6,    "nM"),
    "M":        (1e9,     "nM"),
    "pM":       (0.001,   "nM"),
    "picomolar":(0.001,   "nM"),
    # DMPK
    "µL/min/mg":(1.0,     "µL/min/mg"),
    "uL/min/mg":(1.0,     "µL/min/mg"),
    # hERG (stored as µM)
    "µM_herg":  (1.0,     "µM"),
}

ASSAY_EXPECTED_UNITS: Dict[str, List[str]] = {
    "ASY001": ["nM", "nm", "nanomolar", "µM", "uM", "micromolar", "pM"],
    "ASY002": ["nM", "nm", "nanomolar", "µM", "uM", "micromolar"],
    "ASY003": ["nM", "nm", "nanomolar", "µM", "uM", "micromolar"],
    "ASY004": ["µL/min/mg", "uL/min/mg"],
    "ASY005": ["µM", "uM", "micromolar", "µM_herg"],
}

ASSAY_VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    "ASY001": (0.01, 1e5),   # nM after conversion
    "ASY002": (0.01, 1e5),
    "ASY003": (0.1,  1e5),
    "ASY004": (0.0,  500.0), # Clint raw
    "ASY005": (0.01, 1e5),   # µM after conversion
}

REQUIRED_METADATA = ["operator", "run_date", "instrument", "batch_id"]


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Extract
# ──────────────────────────────────────────────────────────────────────────────

def extract(raw_data: List[dict], source_name: str = "synthetic") -> Tuple[List[RawRecord], str]:
    records = []
    for row in raw_data:
        try:
            rec = RawRecord(
                compound_id  = str(row["compound_id"]),
                assay_id     = str(row["assay_id"]),
                raw_value    = float(row["value"]),
                raw_unit     = str(row.get("unit", "")),
                operator     = str(row.get("operator", "unknown")),
                run_date     = str(row.get("run_date", "1900-01-01")),
                instrument   = str(row.get("instrument", "unknown")),
                batch_id     = str(row.get("batch_id", "BATCH000")),
                n_replicates = int(row.get("n_replicates", 1)),
                notes        = str(row.get("notes", "")),
            )
            records.append(rec)
        except (KeyError, ValueError, TypeError):
            pass   # malformed rows silently dropped at extract

    source_hash = hashlib.md5(
        json.dumps(raw_data, sort_keys=True, default=str).encode()
    ).hexdigest()
    return records, source_hash


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Validate
# ──────────────────────────────────────────────────────────────────────────────

_KNOWN_ASSAYS = {a[0] for a in ASSAY_CATALOGUE}


def validate(records: List[RawRecord]) -> Tuple[List[RawRecord], List[ValidationError]]:
    valid, errors = [], []

    for rec in records:
        rec_errors: List[ValidationError] = []
        rid = f"{rec.compound_id}/{rec.assay_id}"

        # Assay ID must exist
        if rec.assay_id not in _KNOWN_ASSAYS:
            rec_errors.append(ValidationError(rid, "assay_id",
                f"Unknown assay '{rec.assay_id}'", "error"))

        # Unit must be recognised
        expected = ASSAY_EXPECTED_UNITS.get(rec.assay_id, [])
        if rec.raw_unit not in UNIT_CONVERSIONS:
            rec_errors.append(ValidationError(rid, "raw_unit",
                f"Unrecognised unit '{rec.raw_unit}'", "error"))
        elif expected and rec.raw_unit not in expected:
            rec_errors.append(ValidationError(rid, "raw_unit",
                f"Unit '{rec.raw_unit}' unexpected for {rec.assay_id}", "warning"))

        # Value range (in raw units)
        if not (0 < rec.raw_value < 1e8):
            rec_errors.append(ValidationError(rid, "raw_value",
                f"Value {rec.raw_value} out of plausible range", "error"))

        # Required metadata
        for field_name in REQUIRED_METADATA:
            val = getattr(rec, field_name, "")
            if not val or val in ("unknown", ""):
                rec_errors.append(ValidationError(rid, field_name,
                    f"Missing required metadata '{field_name}'", "warning"))

        # Date format
        try:
            datetime.strptime(rec.run_date, "%Y-%m-%d")
        except ValueError:
            rec_errors.append(ValidationError(rid, "run_date",
                f"Invalid date '{rec.run_date}' (expected YYYY-MM-DD)", "error"))

        hard_errors = [e for e in rec_errors if e.severity == "error"]
        errors.extend(rec_errors)
        if not hard_errors:
            valid.append(rec)

    return valid, errors


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Transform
# ──────────────────────────────────────────────────────────────────────────────

def _convert_to_si(value: float, unit: str) -> Tuple[float, str]:
    mult, canonical = UNIT_CONVERSIONS.get(unit, (1.0, unit))
    return value * mult, canonical


def _simulate_curve_stats(rng: np.random.Generator) -> Tuple[float, float, float]:
    hill  = float(rng.uniform(0.7, 2.2))
    rsq   = float(rng.uniform(0.88, 0.999))
    cv    = float(rng.uniform(1, 22))
    return hill, rsq, cv


def transform(
    records: List[RawRecord],
    source_hash: str,
    source_file: str = "synthetic",
    seed: int = 42,
) -> List[TransformedRecord]:
    rng = np.random.default_rng(seed)
    ts  = datetime.utcnow().isoformat()
    transformed = []

    for rec in records:
        value_si, unit_si = _convert_to_si(rec.raw_value, rec.raw_unit)
        hill, rsq, cv = _simulate_curve_stats(rng)

        # IC50 in nM for applicable assays
        ic50_nm: Optional[float] = None
        if unit_si == "nM":
            ic50_nm = value_si
        elif unit_si == "µM" and rec.assay_id not in ("ASY004",):
            ic50_nm = value_si * 1000.0

        qc_flag = "pass" if cv < 20 and rsq > 0.90 else "review"

        transformed.append(TransformedRecord(
            compound_id  = rec.compound_id,
            assay_id     = rec.assay_id,
            value_si     = round(value_si, 4),
            unit_si      = unit_si,
            ic50_nm      = round(ic50_nm, 3) if ic50_nm else None,
            qualifier    = "=",
            hill_slope   = round(hill, 3),
            r_squared    = round(rsq, 4),
            n_replicates = rec.n_replicates,
            cv_pct       = round(cv, 1),
            operator     = rec.operator,
            run_date     = rec.run_date,
            batch_id     = rec.batch_id,
            qc_flag      = qc_flag,
            source_file  = source_file,
            source_hash  = source_hash,
            etl_version  = "1.0.0",
            etl_timestamp = ts,
        ))

    return transformed


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Load
# ──────────────────────────────────────────────────────────────────────────────

def load(conn: sqlite3.Connection, records: List[TransformedRecord]) -> int:
    n = 0
    for i, r in enumerate(records):
        result_id = f"ETL{i+1:06d}"
        cur = conn.execute(
            """INSERT OR IGNORE INTO results
               (result_id, compound_id, assay_id, value, unit, qualifier,
                ic50_nm, hill_slope, r_squared, n_replicates, cv_pct,
                operator, run_date, batch_id, qc_flag)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result_id, r.compound_id, r.assay_id, r.value_si, r.unit_si,
             r.qualifier, r.ic50_nm, r.hill_slope, r.r_squared,
             r.n_replicates, r.cv_pct, r.operator, r.run_date,
             r.batch_id, r.qc_flag),
        )
        n += cur.rowcount
    conn.commit()
    return n


# ──────────────────────────────────────────────────────────────────────────────
# FAIR compliance scoring
# ──────────────────────────────────────────────────────────────────────────────

def compute_fair_score(records: List[TransformedRecord]) -> float:
    if not records:
        return 0.0

    scores = []
    for r in records:
        f = 1.0 if r.compound_id and r.assay_id else 0.0        # Findable
        a = 1.0 if r.run_date != "1900-01-01" else 0.5          # Accessible
        i = 1.0 if r.unit_si in ("nM", "µM", "µL/min/mg") else 0.5  # Interoperable
        re = 1.0 if r.source_hash and r.operator != "unknown" else 0.5  # Reusable
        scores.append((f + a + i + re) / 4)

    return round(float(np.mean(scores)) * 100, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_etl(
    raw_data: List[dict],
    conn: sqlite3.Connection,
    source_name: str = "synthetic",
    seed: int = 42,
) -> ETLReport:
    report = ETLReport(run_timestamp=datetime.utcnow().isoformat())

    # Step 1
    extracted, src_hash = extract(raw_data, source_name)
    report.n_extracted = len(extracted)
    report.source_hash = src_hash

    # Step 2
    validated, all_errors = validate(extracted)
    report.n_validated = len(validated)
    report.n_rejected  = len(extracted) - len(validated)
    report.errors   = [e for e in all_errors if e.severity == "error"]
    report.warnings = [e for e in all_errors if e.severity == "warning"]

    # Step 3
    transformed = transform(validated, src_hash, source_name, seed=seed)
    report.n_transformed = len(transformed)

    # Step 4
    report.n_loaded = load(conn, transformed)

    report.fair_score = compute_fair_score(transformed)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic raw data generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_raw_data(n: int = 200, seed: int = 42) -> List[dict]:
    rng = np.random.default_rng(seed)
    assay_ids  = [a[0] for a in ASSAY_CATALOGUE]
    unit_map   = {a[0]: a[6] for a in ASSAY_CATALOGUE}
    instruments = ["Envision-01", "PHERAstar-02", "FLIPR-03", "PatchXpress-04"]
    operators   = ["analyst_01", "analyst_02", "analyst_03"]

    # Inject some bad records
    bad_unit_idx = set(rng.choice(n, size=int(n * 0.04), replace=False))
    bad_val_idx  = set(rng.choice(n, size=int(n * 0.02), replace=False))

    records = []
    for i in range(n):
        asy_id   = str(rng.choice(assay_ids))
        unit     = unit_map[asy_id]
        value    = float(10 ** rng.uniform(-1, 4)) if asy_id != "ASY004" \
                   else float(rng.uniform(0, 200))

        if i in bad_unit_idx:
            unit = "mg/mL"    # wrong unit
        if i in bad_val_idx:
            value = -999.0    # out-of-range

        records.append({
            "compound_id":  f"CPD{rng.integers(1, 51):04d}",
            "assay_id":     asy_id,
            "value":        round(value, 3),
            "unit":         unit,
            "operator":     str(rng.choice(operators)),
            "run_date":     f"2024-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}",
            "instrument":   str(rng.choice(instruments)),
            "batch_id":     f"BATCH{rng.integers(1,20):03d}",
            "n_replicates": int(rng.integers(2, 4)),
        })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────────────────

def print_etl_report(report: ETLReport) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║         ETL PIPELINE REPORT           ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Run timestamp : {report.run_timestamp}")
    print(f"  Source hash   : {report.source_hash[:12]}…")
    print(f"  Extracted     : {report.n_extracted:>6}")
    print(f"  Validated     : {report.n_validated:>6}")
    print(f"  Rejected      : {report.n_rejected:>6}  "
          f"({report.n_rejected/max(report.n_extracted,1):.1%})")
    print(f"  Transformed   : {report.n_transformed:>6}")
    print(f"  Loaded        : {report.n_loaded:>6}")
    print(f"  FAIR score    : {report.fair_score:>5.1f}%")
    if report.errors:
        print(f"\n  Errors ({len(report.errors)}):")
        for e in report.errors[:5]:
            print(f"    [{e.record_id}] {e.field}: {e.message}")
        if len(report.errors) > 5:
            print(f"    … and {len(report.errors)-5} more")
    if report.warnings:
        print(f"\n  Warnings ({len(report.warnings)}):")
        for w in report.warnings[:3]:
            print(f"    [{w.record_id}] {w.field}: {w.message}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    # Build DB with reference data (compounds, assays)
    conn  = create_database()
    stats = populate_database(conn, seed=42)
    print(f"Reference data loaded: {stats}")

    # Generate synthetic raw upload
    raw_data = generate_raw_data(n=200, seed=42)
    print(f"Raw records to process: {len(raw_data)}")

    # Run ETL
    report = run_etl(raw_data, conn, source_name="synthetic_upload_2024", seed=42)
    print_etl_report(report)

    # Verify loaded results
    n_results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    print(f"\nTotal results in DB after ETL: {n_results}")

    conn.close()
