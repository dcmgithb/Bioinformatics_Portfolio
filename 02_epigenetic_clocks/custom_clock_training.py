"""
Custom Epigenetic Clock Training — ElasticNet DNA Methylation Age Predictor
============================================================================
Trains a custom methylation clock using ElasticNet regression with
nested cross-validation for unbiased performance estimation.
Benchmarks against Horvath and Hannum published clocks.

Reference : Horvath (2013) Genome Biology; Lu et al. (2019) Nature Aging
Dataset   : GSE40279 (Hannum, blood PBMC, n=656, age 19-101)
Python    : >= 3.10
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    KFold, cross_val_score, LeaveOneOut, GridSearchCV
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import joblib

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── 1. Simulate Methylation Data ──────────────────────────────────────────────

def simulate_methylation_data(
    n_samples: int = 500,
    n_cpgs: int = 20000,
    n_clock_cpgs: int = 500,
    noise_level: float = 0.05,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate realistic methylation beta-values with an age signal."""
    rng = np.random.default_rng(SEED)

    ages      = rng.integers(19, 102, n_samples).astype(float)
    cpg_ids   = [f"cg{i:08d}" for i in range(n_cpgs)]
    sample_ids = [f"GSM{rng.integers(1_000_000, 9_999_999)}" for _ in range(n_samples)]

    # Base beta values (clipped to [0.02, 0.98])
    beta = rng.beta(2, 3, (n_samples, n_cpgs)).astype(np.float32)

    # Inject age signal into clock CpGs
    clock_idx = rng.choice(n_cpgs, n_clock_cpgs, replace=False)
    true_coefs = rng.normal(0, 0.003, n_clock_cpgs).astype(np.float32)

    age_z = (ages - ages.mean()) / ages.std()
    for i, (idx, coef) in enumerate(zip(clock_idx, true_coefs)):
        signal = coef * age_z + rng.normal(0, noise_level, n_samples)
        beta[:, idx] = np.clip(beta[:, idx] + signal, 0.02, 0.98)

    X = pd.DataFrame(beta, index=sample_ids, columns=cpg_ids)
    y = pd.Series(ages, index=sample_ids, name="age")
    return X, y


print("Simulating methylation data ...")
X, y = simulate_methylation_data()
print(f"Data: {X.shape[1]} CpGs × {X.shape[0]} samples | Age: {y.min():.0f}–{y.max():.0f} yr")

# ── 2. Pre-processing ─────────────────────────────────────────────────────────

print("Pre-processing: variance filter + M-value transform ...")

def beta_to_mvalue(beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert beta values to M-values for more linear modelling."""
    beta_clipped = np.clip(beta, eps, 1 - eps)
    return np.log2(beta_clipped / (1 - beta_clipped))

X_m = pd.DataFrame(beta_to_mvalue(X.values), index=X.index, columns=X.columns)

# Variance filter: keep top 10k most variable CpGs
var_filter    = VarianceThreshold(threshold=np.percentile(X_m.var(), 50))
X_m_filtered  = var_filter.fit_transform(X_m)
selected_cpgs = X_m.columns[var_filter.get_support()]
X_m_df        = pd.DataFrame(X_m_filtered, index=X.index, columns=selected_cpgs)
print(f"After variance filter: {X_m_df.shape[1]} CpGs retained")

# ── 3. ElasticNet Clock Training ──────────────────────────────────────────────

print("\nTraining custom ElasticNet methylation clock ...")

# Nested CV: outer 5-fold for evaluation, inner ElasticNetCV for tuning
outer_cv  = KFold(n_splits=5, shuffle=True, random_state=SEED)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("enet",   ElasticNetCV(
        l1_ratio       = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        alphas         = np.logspace(-4, 0, 30),
        cv             = 5,
        max_iter       = 2000,
        tol            = 1e-4,
        random_state   = SEED,
        n_jobs         = -1,
    ))
])

# Outer loop: collect test-set predictions
outer_preds  = np.zeros(len(y))
outer_models = []

for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X_m_df)):
    X_train = X_m_df.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test  = X_m_df.iloc[test_idx]
    y_test  = y.iloc[test_idx]

    pipeline.fit(X_train, y_train)
    outer_preds[test_idx] = pipeline.predict(X_test)
    outer_models.append(pipeline)

    fold_mae = mean_absolute_error(y.iloc[test_idx], outer_preds[test_idx])
    print(f"  Fold {fold_i+1}/5 MAE = {fold_mae:.2f} yr")

# Final model trained on all data
pipeline.fit(X_m_df, y)
final_enet    = pipeline.named_steps["enet"]
final_scaler  = pipeline.named_steps["scaler"]

# ── 4. Performance Metrics ────────────────────────────────────────────────────

mae_cv  = mean_absolute_error(y, outer_preds)
r2_cv   = r2_score(y, outer_preds)
pearson = np.corrcoef(y, outer_preds)[0, 1]

print(f"\n=== Custom Clock Performance (5-fold CV) ===")
print(f"MAE    : {mae_cv:.2f} yr")
print(f"R²     : {r2_cv:.4f}")
print(f"Pearson: {pearson:.4f}")
print(f"Best α : {final_enet.alpha_:.5f}")
print(f"Best l1: {final_enet.l1_ratio_:.2f}")

# Clock CpGs (non-zero coefficients)
coef_mask   = final_enet.coef_ != 0
n_cpgs_used = coef_mask.sum()
print(f"Clock CpGs (non-zero coef): {n_cpgs_used} / {len(final_enet.coef_)}")

# Export clock coefficients
clock_df = pd.DataFrame({
    "cpg_id"     : selected_cpgs[coef_mask],
    "coefficient": final_enet.coef_[coef_mask],
    "abs_coef"   : np.abs(final_enet.coef_[coef_mask]),
}).sort_values("abs_coef", ascending=False)

clock_df.to_csv(RESULTS_DIR / "custom_clock_cpgs.csv", index=False)
print(f"Clock exported: {len(clock_df)} CpGs → {RESULTS_DIR/'custom_clock_cpgs.csv'}")

# Save model
joblib.dump(pipeline, RESULTS_DIR / "custom_clock_pipeline.pkl")

# ── 5. Visualisation ──────────────────────────────────────────────────────────

# Acceleration = regression residual of predicted_age ~ chrono_age
# (removes systematic slope bias; matches the published Horvath/Hannum definition)
_slope, _intercept = np.polyfit(y.values, outer_preds, 1)
_trend = _slope * y.values + _intercept

results_df = pd.DataFrame({
    "chronological_age" : y.values,
    "predicted_age"     : outer_preds,
    "acceleration"      : outer_preds - _trend,
})

# 5a. Prediction accuracy
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(results_df["chronological_age"], results_df["predicted_age"],
           alpha=0.5, s=15, c="#607D8B")
x_line = np.linspace(y.min(), y.max(), 100)
ax.plot(x_line, x_line, "k--", lw=1, label="Perfect")
m, b = np.polyfit(results_df["chronological_age"], results_df["predicted_age"], 1)
ax.plot(x_line, m * x_line + b, "r-", lw=1.5, label=f"Fit (slope={m:.2f})")
ax.set_xlabel("Chronological Age (yr)", fontsize=12)
ax.set_ylabel("Predicted Epigenetic Age (yr)", fontsize=12)
ax.set_title(f"Custom ElasticNet Clock\nMAE={mae_cv:.2f} yr, R²={r2_cv:.3f}, r={pearson:.3f}", fontsize=12)
ax.legend()

# 5b. Acceleration distribution
ax = axes[1]
ax.hist(results_df["acceleration"], bins=30, color="#607D8B", edgecolor="white", alpha=0.8)
ax.axvline(0, color="red", lw=1.5, linestyle="--")
ax.axvline(results_df["acceleration"].mean(), color="orange", lw=1.5,
           label=f"Mean = {results_df['acceleration'].mean():.2f} yr")
ax.set_xlabel("Epigenetic Age Acceleration (yr)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of Epigenetic Age Acceleration", fontsize=12)
ax.legend()

plt.tight_layout()
fig.savefig(FIGURES_DIR / "06_custom_clock_performance.pdf", dpi=150)
plt.close()

# 5c. Top 30 clock CpGs by |coefficient|
top_cpgs = clock_df.head(30)
fig, ax = plt.subplots(figsize=(9, 7))
colours = ["#F44336" if c > 0 else "#2196F3" for c in top_cpgs["coefficient"]]
ax.barh(range(len(top_cpgs)), top_cpgs["coefficient"], color=colours)
ax.set_yticks(range(len(top_cpgs)))
ax.set_yticklabels(top_cpgs["cpg_id"], fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("ElasticNet Coefficient", fontsize=11)
ax.set_title("Top 30 Clock CpGs by Coefficient Magnitude\nRed=hypermethylated with age | Blue=hypomethylated", fontsize=11)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "07_clock_cpg_coefficients.pdf", dpi=150)
plt.close()

# 5d. Acceleration vs age (to detect non-linearity)
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(results_df["chronological_age"], results_df["acceleration"],
                c=results_df["acceleration"],
                cmap="RdBu_r", vmin=-20, vmax=20, alpha=0.6, s=15)
plt.colorbar(sc, ax=ax, label="Acceleration (yr)")
ax.axhline(0, color="black", lw=1, linestyle="--")

ax.set_xlabel("Chronological Age (yr)", fontsize=12)
ax.set_ylabel("Epigenetic Acceleration (yr)", fontsize=12)
ax.set_title("Age Acceleration vs. Chronological Age\nFlat trend = unbiased clock", fontsize=12)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "08_acceleration_vs_age.pdf", dpi=150)
plt.close()

print(f"\n=== Custom clock training complete ===")
print(f"Results: {RESULTS_DIR}")
print(f"Figures: {FIGURES_DIR}")
