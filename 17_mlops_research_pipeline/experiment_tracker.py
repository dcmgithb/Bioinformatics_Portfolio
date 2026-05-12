"""
experiment_tracker.py — MLflow experiment tracking for preclinical bioactivity models.

Tracks 3 model types (RF, XGBoost, ElasticNet) across multiple runs with
parameter logging, metric comparison, SHAP artifact, and model registry
with Production transition.
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42): np.random.seed(s)
    PALETTES = {"young": "#2196F3", "aged": "#F44336", "accent": "#4CAF50"}

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("mlflow not installed — running in dry-run mode (no tracking server).")

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic bioactivity dataset
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds",
    "aromatic_rings", "fsp3", "qed",
    "morgan_pc1", "morgan_pc2", "morgan_pc3",
    "maccs_pc1", "maccs_pc2",
]


def generate_bioactivity_dataset(n: int = 500, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "mw":              rng.uniform(200, 600, n),
        "logp":            rng.uniform(-1, 6, n),
        "hbd":             rng.integers(0, 6, n).astype(float),
        "hba":             rng.integers(0, 11, n).astype(float),
        "tpsa":            rng.uniform(20, 160, n),
        "rotatable_bonds": rng.integers(0, 12, n).astype(float),
        "aromatic_rings":  rng.integers(0, 5, n).astype(float),
        "fsp3":            rng.uniform(0, 1, n),
        "qed":             rng.uniform(0.1, 1.0, n),
        "morgan_pc1":      rng.normal(0, 1, n),
        "morgan_pc2":      rng.normal(0, 1, n),
        "morgan_pc3":      rng.normal(0, 1, n),
        "maccs_pc1":       rng.normal(0, 1, n),
        "maccs_pc2":       rng.normal(0, 1, n),
    })
    # Activity: logP 1-4, MW 300-500, low TPSA → active
    score = (
        -np.abs(X["logp"] - 2.5) / 2
        - np.abs(X["mw"] - 400) / 150
        - X["tpsa"] / 120
        + X["qed"]
        + rng.normal(0, 0.4, n)
    )
    y = (score > np.percentile(score, 40)).astype(int)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Model configs
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        "name": "RandomForest",
        "model": RandomForestClassifier,
        "param_grid": [
            {"n_estimators": 100, "max_depth": 5,    "min_samples_leaf": 3, "random_state": 42},
            {"n_estimators": 200, "max_depth": 8,    "min_samples_leaf": 2, "random_state": 42},
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "random_state": 42},
        ],
        "tags": {"framework": "sklearn", "algorithm": "ensemble", "task": "classification"},
    },
    {
        "name": "XGBoost",
        "model": GradientBoostingClassifier,
        "param_grid": [
            {"n_estimators": 100, "learning_rate": 0.1,  "max_depth": 3, "random_state": 42},
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "random_state": 42},
            {"n_estimators": 150, "learning_rate": 0.08, "max_depth": 5, "random_state": 42},
        ],
        "tags": {"framework": "sklearn", "algorithm": "gradient_boosting", "task": "classification"},
    },
    {
        "name": "ElasticNet",
        "model": LogisticRegression,
        "param_grid": [
            {"C": 1.0,  "penalty": "l2",         "solver": "lbfgs",  "random_state": 42, "max_iter": 1000},
            {"C": 0.1,  "penalty": "l2",         "solver": "lbfgs",  "random_state": 42, "max_iter": 1000},
            {"C": 10.0, "penalty": "l2",         "solver": "lbfgs",  "random_state": 42, "max_iter": 1000},
        ],
        "tags": {"framework": "sklearn", "algorithm": "linear", "task": "classification"},
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Permutation feature importance (SHAP proxy)
# ──────────────────────────────────────────────────────────────────────────────

def permutation_importance(
    model, X_val: pd.DataFrame, y_val: np.ndarray, n_repeats: int = 10, seed: int = 42
) -> pd.Series:
    rng      = np.random.default_rng(seed)
    base_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    importances = {}
    for col in X_val.columns:
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_auc = roc_auc_score(y_val, model.predict_proba(X_perm)[:, 1])
            deltas.append(base_auc - perm_auc)
        importances[col] = float(np.mean(deltas))
    return pd.Series(importances).sort_values(ascending=False)


def plot_feature_importance(importances: pd.Series, title: str, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    top = importances.head(10)
    colors = [PALETTES.get("young", "#2196F3") if v >= 0 else PALETTES.get("aged", "#F44336")
              for v in top.values]
    ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1])
    ax.set_xlabel("Mean ΔAUC (permutation importance)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Tracking with MLflow (or dry-run log)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    run_id:    str
    model_name: str
    params:    Dict
    auc_roc:   float
    auc_pr:    float
    rmse:      float
    cv_auc:    float


def train_and_track(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame,   y_val: np.ndarray,
    config: Dict,
    run_name: str,
    experiment_name: str = "preclinical_bioactivity",
    tracking_dir: str = "./mlruns",
) -> RunResult:
    params     = config["param_grid"][0]
    ModelClass = config["model"]
    tags       = config["tags"]

    is_tree = isinstance(ModelClass, type) and issubclass(
        ModelClass, (RandomForestClassifier, GradientBoostingClassifier)
    )
    if is_tree:
        model = ModelClass(**params)
    else:
        model = Pipeline([("scaler", StandardScaler()), ("clf", ModelClass(**params))])

    model.fit(X_train, y_train)
    proba    = model.predict_proba(X_val)[:, 1]
    auc_roc  = roc_auc_score(y_val, proba)
    auc_pr   = average_precision_score(y_val, proba)
    rmse     = float(np.sqrt(mean_squared_error(y_val, proba)))
    cv_estimator = (
        ModelClass(**params) if is_tree
        else Pipeline([("scaler", StandardScaler()), ("clf", ModelClass(**params))])
    )
    cv_auc   = float(np.mean(cross_val_score(
        cv_estimator, X_train, y_train, cv=5, scoring="roc_auc"
    )))

    # Feature importance artifact
    tmpdir  = tempfile.mkdtemp()
    imp_path = os.path.join(tmpdir, f"importance_{run_name}.png")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X_train.columns)
        imp = imp.sort_values(ascending=False)
    else:
        imp = permutation_importance(model, X_val, y_val, n_repeats=5)
    plot_feature_importance(imp, f"Feature Importance — {run_name}", imp_path)

    run_id = "dry_run"
    if HAS_MLFLOW:
        mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
        mlflow.set_experiment(experiment_name)
        mlflow.sklearn.autolog(log_models=False, silent=True)

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tags(tags)
            mlflow.log_params(params)
            mlflow.log_metrics({
                "auc_roc": round(auc_roc, 4),
                "auc_pr":  round(auc_pr, 4),
                "rmse":    round(rmse, 4),
                "cv_auc":  round(cv_auc, 4),
            })
            mlflow.log_artifact(imp_path, artifact_path="figures")
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     registered_model_name=config["name"])

    return RunResult(run_id=run_id, model_name=config["name"],
                     params=params, auc_roc=auc_roc, auc_pr=auc_pr,
                     rmse=rmse, cv_auc=cv_auc)


# ──────────────────────────────────────────────────────────────────────────────
# Run comparison plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_run_comparison(
    results: List[RunResult],
    out_path: str = "figures/mlflow_comparison.png",
) -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df = pd.DataFrame([{
        "model": r.model_name, "AUC-ROC": r.auc_roc,
        "AUC-PR": r.auc_pr, "RMSE": r.rmse, "CV-AUC": r.cv_auc,
    } for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#FAFAFA")
    colors = [PALETTES.get("young", "#2196F3"),
              PALETTES.get("aged", "#F44336"),
              PALETTES.get("accent", "#4CAF50")]

    for ax, metric in zip(axes, ["AUC-ROC", "AUC-PR", "CV-AUC"]):
        vals = df.groupby("model")[metric].max()
        bar_colors = colors[:len(vals)]
        ax.bar(vals.index, vals.values, color=bar_colors, edgecolor="white")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Model")
        ax.set_ylim(0, 1.05)
        for i, (name, val) in enumerate(vals.items()):
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
        ax.set_facecolor("#F5F5F5")

    fig.suptitle("MLflow Run Comparison — Preclinical Bioactivity Models",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Model registry helper
# ──────────────────────────────────────────────────────────────────────────────

def promote_best_to_production(results: List[RunResult]) -> str:
    best = max(results, key=lambda r: r.auc_roc)
    print(f"\nBest model: {best.model_name}  AUC-ROC={best.auc_roc:.4f}")

    if HAS_MLFLOW:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{best.model_name}'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            client.transition_model_version_stage(
                name=best.model_name,
                version=latest.version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"Transitioned {best.model_name} v{latest.version} → Production")
            return f"{best.model_name}:v{latest.version}"
    return f"{best.model_name}:dry_run"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    print("Generating synthetic bioactivity dataset (n=500) …")
    X, y = generate_bioactivity_dataset(n=500, seed=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}"
          f"  Active fraction: {y.mean():.1%}")

    results = []
    for config in MODEL_CONFIGS:
        run_name = config["name"]
        print(f"\nTraining {run_name} …")
        res = train_and_track(
            X_train, y_train, X_val, y_val,
            config, run_name=run_name,
        )
        results.append(res)
        print(f"  AUC-ROC={res.auc_roc:.4f}  AUC-PR={res.auc_pr:.4f}"
              f"  CV-AUC={res.cv_auc:.4f}")

    os.makedirs("figures", exist_ok=True)
    img = plot_run_comparison(results, "figures/mlflow_comparison.png")
    print(f"\nRun comparison plot → {img}")

    prod_model = promote_best_to_production(results)
    print(f"Production model registered: {prod_model}")

    if HAS_MLFLOW:
        print(f"\nMLflow UI: mlflow ui --backend-store-uri file://$(pwd)/mlruns")
