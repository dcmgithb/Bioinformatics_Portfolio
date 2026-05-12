"""
workflow_dag.py — Prefect 2.x workflow DAG for the preclinical ML pipeline.

7 tasks: ingest_data → validate_schema → run_etl → train_models →
evaluate_models → register_best → generate_report.

Falls back to a plain Python task graph if Prefect is not installed.
"""

from __future__ import annotations

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed
except ImportError:
    def set_global_seed(s=42): np.random.seed(s)

try:
    from prefect import flow, task
    from prefect.logging import get_run_logger
    HAS_PREFECT = True
except ImportError:
    HAS_PREFECT = False
    # Minimal shims so the rest of the code is identical
    def task(fn=None, *, retries=0, retry_delay_seconds=5, name=None, **kw):
        def decorator(f):
            f._task_name    = name or f.__name__
            f._retries      = retries
            f._retry_delay  = retry_delay_seconds
            return f
        return decorator(fn) if fn else decorator

    def flow(fn=None, *, name=None, **kw):
        def decorator(f):
            f._flow_name = name or f.__name__
            return f
        return decorator(fn) if fn else decorator

    def get_run_logger():
        import logging
        return logging.getLogger("workflow_dag")

from experiment_tracker import (
    generate_bioactivity_dataset, MODEL_CONFIGS,
    train_and_track, promote_best_to_production,
    plot_run_comparison,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared context (artifact bag passed between tasks)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineContext:
    raw_records:    int         = 0
    validated_pct:  float       = 0.0
    n_features:     int         = 0
    n_train:        int         = 0
    n_val:          int         = 0
    model_results:  List[dict]  = field(default_factory=list)
    best_model:     str         = ""
    best_auc:       float       = 0.0
    report_path:    str         = ""
    run_id:         str         = ""
    errors:         List[str]   = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────────────────────────────────────

@task(retries=2, retry_delay_seconds=5, name="ingest_data")
def ingest_data(n_records: int = 500, seed: int = 42) -> dict:
    logger = get_run_logger()
    logger.info(f"Ingesting {n_records} synthetic bioactivity records …")

    X, y = generate_bioactivity_dataset(n=n_records, seed=seed)
    source_hash = hashlib.md5(str(seed).encode()).hexdigest()[:8]

    result = {
        "X": X.to_dict(orient="list"),
        "y": y.tolist(),
        "n_records": n_records,
        "source_hash": source_hash,
        "ingest_timestamp": datetime.utcnow().isoformat(),
    }
    logger.info(f"Ingested {n_records} records  hash={source_hash}")
    return result


@task(retries=1, retry_delay_seconds=2, name="validate_schema")
def validate_schema(ingested: dict) -> dict:
    logger = get_run_logger()
    X = pd.DataFrame(ingested["X"])
    y = np.array(ingested["y"])

    errors = []
    if X.isnull().any().any():
        n_null = int(X.isnull().sum().sum())
        errors.append(f"{n_null} null values detected")
    if not set(["mw", "logp", "tpsa"]).issubset(X.columns):
        errors.append("Missing required columns: mw, logp, tpsa")
    if len(X) != len(y):
        errors.append(f"Length mismatch: X={len(X)} y={len(y)}")
    if not np.isin(y, [0, 1]).all():
        errors.append("Labels contain non-binary values")

    pass_rate = 1.0 - len(errors) / max(len(X.columns), 1) / 10
    logger.info(f"Validation: {len(errors)} issues  pass_rate={pass_rate:.1%}")

    if errors:
        logger.warning(f"Validation warnings: {errors}")

    return {**ingested, "validation_errors": errors, "pass_rate": pass_rate}


@task(retries=1, retry_delay_seconds=5, name="run_etl")
def run_etl(validated: dict) -> dict:
    logger = get_run_logger()
    X = pd.DataFrame(validated["X"])
    y = np.array(validated["y"])

    # Normalise features (z-score)
    means = X.mean()
    stds  = X.std().replace(0, 1)
    X_norm = (X - means) / stds

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info(f"ETL complete: {len(X_train)} train, {len(X_val)} val  "
                f"active={y.mean():.1%}")
    return {
        "X_train": X_train.to_dict(orient="list"),
        "X_val":   X_val.to_dict(orient="list"),
        "y_train": y_train.tolist(),
        "y_val":   y_val.tolist(),
        "feature_names": list(X.columns),
        "n_train": len(X_train),
        "n_val":   len(X_val),
    }


@task(retries=3, retry_delay_seconds=10, name="train_models")
def train_models(etl_output: dict) -> dict:
    logger = get_run_logger()
    X_train = pd.DataFrame(etl_output["X_train"])
    X_val   = pd.DataFrame(etl_output["X_val"])
    y_train = np.array(etl_output["y_train"])
    y_val   = np.array(etl_output["y_val"])

    results = []
    for config in MODEL_CONFIGS:
        logger.info(f"Training {config['name']} …")
        res = train_and_track(
            X_train, y_train, X_val, y_val,
            config, run_name=f"dag_{config['name']}",
        )
        results.append({
            "model_name": res.model_name,
            "auc_roc":    round(res.auc_roc, 4),
            "auc_pr":     round(res.auc_pr, 4),
            "cv_auc":     round(res.cv_auc, 4),
            "run_id":     res.run_id,
        })
        logger.info(f"  {config['name']}: AUC={res.auc_roc:.4f}")

    return {"model_results": results, **{k: v for k, v in etl_output.items()
                                         if k in ("n_train", "n_val", "feature_names")}}


@task(retries=1, retry_delay_seconds=5, name="evaluate_models")
def evaluate_models(trained: dict) -> dict:
    logger = get_run_logger()
    results = trained["model_results"]

    # Rank by AUC-ROC
    ranked = sorted(results, key=lambda r: r["auc_roc"], reverse=True)
    best   = ranked[0]

    logger.info("Model evaluation ranking:")
    for i, r in enumerate(ranked):
        logger.info(f"  {i+1}. {r['model_name']}  AUC={r['auc_roc']:.4f}")

    return {**trained, "ranked_results": ranked,
            "best_model": best["model_name"], "best_auc": best["auc_roc"]}


@task(retries=2, retry_delay_seconds=5, name="register_best")
def register_best(evaluated: dict) -> dict:
    logger = get_run_logger()
    best_name = evaluated["best_model"]
    best_auc  = evaluated["best_auc"]

    # In real MLflow this would call MlflowClient().transition_model_version_stage(...)
    registry_entry = {
        "model_name":     best_name,
        "stage":          "Production",
        "auc_roc":        best_auc,
        "registered_at":  datetime.utcnow().isoformat(),
        "registered_by":  "workflow_dag",
    }
    logger.info(f"Registered {best_name} → Production  AUC={best_auc:.4f}")
    return {**evaluated, "registry_entry": registry_entry}


@task(retries=1, retry_delay_seconds=2, name="generate_report")
def generate_report(registered: dict, out_dir: str = ".") -> str:
    logger = get_run_logger()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pipeline_report.txt")

    lines = [
        "MLOPS PIPELINE REPORT",
        "=" * 50,
        f"Run timestamp  : {datetime.utcnow().isoformat()}",
        f"Train samples  : {registered.get('n_train', 'N/A')}",
        f"Val samples    : {registered.get('n_val', 'N/A')}",
        f"Features       : {len(registered.get('feature_names', []))}",
        "",
        "── Model Results ──",
    ]
    for r in registered.get("ranked_results", []):
        lines.append(
            f"  {r['model_name']:<20}  AUC-ROC={r['auc_roc']:.4f}"
            f"  AUC-PR={r['auc_pr']:.4f}  CV-AUC={r['cv_auc']:.4f}"
        )
    reg = registered.get("registry_entry", {})
    lines += [
        "",
        "── Registry ──",
        f"  Production model : {reg.get('model_name', 'N/A')}",
        f"  Registered at    : {reg.get('registered_at', 'N/A')}",
        f"  AUC-ROC          : {reg.get('auc_roc', 0):.4f}",
    ]

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info(f"Report written → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Flow
# ──────────────────────────────────────────────────────────────────────────────

@flow(name="preclinical_ml_pipeline")
def preclinical_ml_pipeline(
    n_records: int = 500,
    seed: int = 42,
    out_dir: str = ".",
) -> PipelineContext:
    ctx = PipelineContext()
    ctx.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    ingested  = ingest_data(n_records=n_records, seed=seed)
    validated = validate_schema(ingested)
    etl_out   = run_etl(validated)
    trained   = train_models(etl_out)
    evaluated = evaluate_models(trained)
    registered = register_best(evaluated)
    report_path = generate_report(registered, out_dir=out_dir)

    ctx.raw_records   = ingested["n_records"]
    ctx.validated_pct = validated["pass_rate"]
    ctx.n_train       = etl_out["n_train"]
    ctx.n_val         = etl_out["n_val"]
    ctx.n_features    = len(etl_out["feature_names"])
    ctx.model_results = evaluated["ranked_results"]
    ctx.best_model    = evaluated["best_model"]
    ctx.best_auc      = evaluated["best_auc"]
    ctx.report_path   = report_path
    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# Plain Python runner (no Prefect server needed)
# ──────────────────────────────────────────────────────────────────────────────

def run_without_prefect(n_records: int = 500, seed: int = 42, out_dir: str = ".") -> PipelineContext:
    """Execute the same task graph without Prefect orchestration."""
    print("Running pipeline in standalone mode (Prefect not available) …")
    ingested   = ingest_data(n_records=n_records, seed=seed)
    validated  = validate_schema(ingested)
    etl_out    = run_etl(validated)
    trained    = train_models(etl_out)
    evaluated  = evaluate_models(trained)
    registered = register_best(evaluated)
    report_path = generate_report(registered, out_dir=out_dir)

    ctx = PipelineContext(
        raw_records   = ingested["n_records"],
        validated_pct = validated["pass_rate"],
        n_train       = etl_out["n_train"],
        n_val         = etl_out["n_val"],
        n_features    = len(etl_out["feature_names"]),
        model_results = evaluated["ranked_results"],
        best_model    = evaluated["best_model"],
        best_auc      = evaluated["best_auc"],
        report_path   = report_path,
        run_id        = datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    )
    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)
    os.makedirs("figures", exist_ok=True)

    if HAS_PREFECT:
        print("Running with Prefect orchestration …")
        ctx = preclinical_ml_pipeline(n_records=500, seed=42, out_dir=".")
    else:
        ctx = run_without_prefect(n_records=500, seed=42, out_dir=".")

    print("\n── Pipeline Summary ──")
    print(f"  Records ingested  : {ctx.raw_records}")
    print(f"  Validation pass   : {ctx.validated_pct:.1%}")
    print(f"  Train / Val       : {ctx.n_train} / {ctx.n_val}")
    print(f"  Features          : {ctx.n_features}")
    print(f"  Best model        : {ctx.best_model}  AUC={ctx.best_auc:.4f}")
    print(f"  Report            : {ctx.report_path}")
    if ctx.model_results:
        print("\n  Model ranking:")
        for r in ctx.model_results:
            print(f"    {r['model_name']:<20}  AUC-ROC={r['auc_roc']:.4f}")
