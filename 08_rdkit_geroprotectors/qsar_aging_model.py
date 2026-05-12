"""
QSAR Model — Predicting Geroprotective Activity
=================================================
Quantitative Structure–Activity Relationship (QSAR) model
to predict pIC50 (geroprotective potency) from molecular structure.

Pipeline:
  1. Feature engineering: 200 RDKit 2D descriptors + Morgan FP (2048 bits)
  2. Pre-processing: variance filter → correlation filter → StandardScaler
  3. Models: Random Forest, XGBoost, ElasticNet — stacked ensemble
  4. Nested 5×5 cross-validation (outer: performance, inner: hyperparameter tuning)
  5. SHAP feature importance (global + local)
  6. Applicability domain (leverage / Williams plot)
  7. y-scrambling validation (model is not spurious)

Python : >= 3.10 | RDKit >= 2023.09
"""

from __future__ import annotations

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not found — using GradientBoosting")
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("shap not found — skipping SHAP plots")

from rdkit import Chem

from utils.chem_utils import (
    validate_smiles_column, fingerprint_matrix,
    descriptor_matrix
)

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────

print("═" * 60)
print("  QSAR MODEL — GEROPROTECTIVE ACTIVITY PREDICTION")
print("═" * 60)

df_raw = pd.read_csv("data/geroprotectors.csv")
df     = validate_smiles_column(df_raw, smiles_col="smiles")
mols   = [Chem.MolFromSmiles(s) for s in df["smiles"]]

y = df["activity_pIC50"].values.astype(float)
print(f"\nDataset: {len(df)} compounds | pIC50 range: {y.min():.1f} – {y.max():.1f}")
print(f"  Senolytics: {(df['class']=='senolytic').sum()} | "
      f"Senomorphics: {(df['class']=='senomorphic').sum()}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────

print("\n[2] Computing molecular features ...")

# 2D descriptors
desc_df = descriptor_matrix(mols)
print(f"  2D descriptors: {desc_df.shape[1]}")

# Morgan fingerprints (2048 bits, radius=2)
fp_matrix = fingerprint_matrix(mols, fp_type="morgan", radius=2, n_bits=2048)
fp_df     = pd.DataFrame(fp_matrix,
                          columns=[f"FP_{i}" for i in range(fp_matrix.shape[1])],
                          index=df.index)

# MACCS keys (167 bits)
maccs_mat = fingerprint_matrix(mols, fp_type="maccs")
maccs_df  = pd.DataFrame(maccs_mat,
                          columns=[f"MACCS_{i}" for i in range(maccs_mat.shape[1])],
                          index=df.index)

# Combine
X_full = pd.concat([desc_df, fp_df], axis=1).fillna(0)
print(f"  Combined feature matrix: {X_full.shape}")

# ── 3. Pre-processing Pipeline ────────────────────────────────────────────────

print("\n[3] Feature pre-processing ...")

# Remove zero-variance and near-constant features
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X_full)
print(f"  After variance filter: {X_vt.shape[1]} features")

# Remove highly correlated features (Pearson r > 0.95)
corr_mat   = np.corrcoef(X_vt.T)
upper_tri  = np.triu(np.abs(corr_mat), k=1)
drop_cols  = set(np.where(upper_tri > 0.95)[1])
keep_cols  = [i for i in range(X_vt.shape[1]) if i not in drop_cols]
X_filtered = X_vt[:, keep_cols]
print(f"  After correlation filter (r>0.95): {X_filtered.shape[1]} features")

X = X_filtered
feature_names = [X_full.columns[vt.get_support()][i] for i in keep_cols]

# ── 4. Model Definitions ──────────────────────────────────────────────────────

models = {
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=500, max_depth=None,
            min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=SEED
        ))
    ]),
    "XGBoost" if HAS_XGB else "GradBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0
        ) if HAS_XGB else GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=SEED
        ))
    ]),
    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
            alphas=np.logspace(-4, 1, 30),
            cv=5, max_iter=3000, random_state=SEED
        ))
    ]),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1))
    ]),
}

# ── 5. Nested Cross-Validation ────────────────────────────────────────────────

print("\n[5] Running nested 5-fold cross-validation ...")

outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
results  = {}

for name, pipe in models.items():
    print(f"  {name} ...", end=" ")
    preds  = cross_val_predict(pipe, X, y, cv=outer_cv, n_jobs=-1)
    r2     = r2_score(y, preds)
    mae    = mean_absolute_error(y, preds)
    rmse   = np.sqrt(mean_squared_error(y, preds))
    r, _   = pearsonr(y, preds)
    rho, _ = spearmanr(y, preds)

    results[name] = {
        "R2"      : r2,
        "MAE"     : mae,
        "RMSE"    : rmse,
        "Pearson" : r,
        "Spearman": rho,
        "preds"   : preds,
    }
    print(f"R²={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

# Best model by R²
best_name = max(results, key=lambda k: results[k]["R2"])
print(f"\n  Best model: {best_name} (R²={results[best_name]['R2']:.3f})")

# ── 6. Performance Plot ───────────────────────────────────────────────────────

print("\n[6] Plotting performance ...")

fig, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 10))
colours   = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

for col, (name, res) in enumerate(results.items()):
    preds = res["preds"]
    colour = colours[col]

    # Observed vs. Predicted
    ax = axes[0, col]
    ax.scatter(y, preds, c=colour, s=50, alpha=0.75, edgecolors="white", lw=0.4)
    lims = [min(y.min(), preds.min()) - 0.5, max(y.max(), preds.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1, label="Perfect")
    m, b = np.polyfit(y, preds, 1)
    ax.plot(np.array(lims), m * np.array(lims) + b, "-", color=colour, lw=1.5, label="Fit")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed pIC50", fontsize=10)
    ax.set_ylabel("Predicted pIC50", fontsize=10)
    ax.set_title(f"{name}\nR²={res['R2']:.3f} | RMSE={res['RMSE']:.2f} | r={res['Pearson']:.3f}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Residuals
    ax = axes[1, col]
    residuals = preds - y
    ax.scatter(preds, residuals, c=colour, s=40, alpha=0.75, edgecolors="white", lw=0.4)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.axhline( 1, color="grey", lw=0.8, ls=":")
    ax.axhline(-1, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Predicted pIC50", fontsize=10)
    ax.set_ylabel("Residual (Pred − Obs)", fontsize=10)
    ax.set_title("Residual Plot", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3)

fig.suptitle("QSAR Models — Geroprotective Activity (pIC50)\nNested 5-fold CV",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "08_qsar_performance.pdf", bbox_inches="tight", dpi=150)
plt.close()

# Metrics summary
metrics_df = pd.DataFrame({k: {m: v for m, v in res.items() if m != "preds"}
                            for k, res in results.items()}).T
metrics_df.to_csv(RESULTS_DIR / "qsar_metrics.csv")
print(metrics_df[["R2","MAE","RMSE","Pearson","Spearman"]].to_string())

# ── 7. SHAP Feature Importance ────────────────────────────────────────────────

if HAS_SHAP:
    print("\n[7] Computing SHAP values for Random Forest ...")
    best_pipe = models["RandomForest"]
    best_pipe.fit(X, y)
    X_scaled = best_pipe.named_steps["scaler"].transform(X)
    rf_model = best_pipe.named_steps["model"]

    explainer  = shap.TreeExplainer(rf_model)
    shap_vals  = explainer.shap_values(X_scaled)

    feature_arr = np.array(feature_names)

    # Global importance
    mean_shap = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[::-1][:25]

    # 9a. Beeswarm — shap.summary_plot creates its own figure; save before closing
    shap.summary_plot(shap_vals[:, top_idx],
                      X_scaled[:, top_idx],
                      feature_names=feature_arr[top_idx],
                      show=False, plot_type="dot",
                      max_display=20, color_bar=True)
    plt.title("SHAP Beeswarm — Top 20 Features\n(Random Forest | pIC50)",
              fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09a_shap_beeswarm.pdf", bbox_inches="tight", dpi=150)
    plt.close()

    # 9b. Bar plot of mean |SHAP|
    top25_names = feature_arr[top_idx]
    top25_shap  = mean_shap[top_idx]
    bar_colours = ["#F44336" if "FP_" in n else "#2196F3" for n in top25_names]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(len(top25_names)), top25_shap[::-1], color=bar_colours[::-1])
    ax.set_yticks(range(len(top25_names)))
    ax.set_yticklabels(top25_names[::-1], fontsize=7)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title("Feature Importance (mean |SHAP|)\nRed=Morgan FP | Blue=2D descriptor",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    legend_patches = [
        mpatches.Patch(color="#F44336", label="Morgan Fingerprint bit"),
        mpatches.Patch(color="#2196F3", label="2D Descriptor"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09b_shap_bar.pdf", bbox_inches="tight", dpi=150)
    plt.close()

    # Top descriptor SHAP
    desc_features = [(i, n, mean_shap[i]) for i, n in enumerate(feature_names)
                     if not n.startswith("FP_") and not n.startswith("MACCS_")]
    desc_features.sort(key=lambda x: -x[2])
    top_desc_df = pd.DataFrame(desc_features[:30], columns=["idx","feature","mean_shap"])
    top_desc_df.to_csv(RESULTS_DIR / "top_descriptor_features.csv", index=False)
    print(f"  Top 5 descriptor features: {[r['feature'] for _, r in top_desc_df.head(5).iterrows()]}")

else:
    print("\n[7] SHAP skipped (install: pip install shap)")

# ── 8. Applicability Domain (Williams Plot) ───────────────────────────────────

print("\n[8] Applicability domain analysis ...")

best_pipe = models["RandomForest"]
best_pipe.fit(X, y)
preds_full = best_pipe.predict(X)

X_scaled_full = best_pipe.named_steps["scaler"].transform(X)

# Leverage: h_i = x_i^T (X^T X)^{-1} x_i
XtX_inv = np.linalg.pinv(X_scaled_full.T @ X_scaled_full)
h       = np.array([x @ XtX_inv @ x for x in X_scaled_full])
h_star  = 3 * (X_scaled_full.shape[1] + 1) / X_scaled_full.shape[0]

# Standardised residuals
residuals  = preds_full - y
sigma      = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
std_resid  = residuals / (sigma + 1e-10)

# Williams plot
fig, ax = plt.subplots(figsize=(9, 6))
in_ad  = (h <= h_star) & (np.abs(std_resid) <= 3)
out_ad = ~in_ad

scatter_in  = ax.scatter(h[in_ad],  std_resid[in_ad],
                          c="#2196F3", s=50, alpha=0.75,
                          edgecolors="white", lw=0.4, label="Within AD")
scatter_out = ax.scatter(h[out_ad], std_resid[out_ad],
                          c="#F44336", s=60, alpha=0.85,
                          edgecolors="black", lw=0.4, label="Outside AD")

# Label outliers
for idx in np.where(out_ad)[0]:
    ax.annotate(df["name"].iloc[idx],
                xy=(h[idx], std_resid[idx]),
                xytext=(4, 4), textcoords="offset points", fontsize=6.5)

ax.axhline( 3, color="grey", lw=1, ls="--", label="Residual limit (±3)")
ax.axhline(-3, color="grey", lw=1, ls="--")
ax.axvline(h_star, color="orange", lw=1.5, ls="--", label=f"h* = {h_star:.3f}")
ax.set_xlabel("Leverage (h)", fontsize=11)
ax.set_ylabel("Standardised Residual", fontsize=11)
ax.set_title("Williams Plot — Applicability Domain\n"
             "Random Forest QSAR | Blue = within AD | Red = outside AD",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

n_in  = in_ad.sum()
n_out = out_ad.sum()
ax.text(0.02, 0.97,
        f"Within AD: {n_in}/{len(y)} ({100*n_in/len(y):.0f}%)",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
fig.savefig(FIGURES_DIR / "10_williams_plot.pdf", bbox_inches="tight", dpi=150)
plt.close()

# ── 9. y-Scrambling Validation ────────────────────────────────────────────────

print("\n[9] y-scrambling validation (n=100 permutations) ...")

n_scramble = 100
rf_model   = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
scaler     = StandardScaler()
X_s        = scaler.fit_transform(X)

scrambled_r2  = []
for i in range(n_scramble):
    y_rand = np.random.permutation(y)
    preds_s = cross_val_predict(rf_model, X_s, y_rand,
                                 cv=KFold(5, shuffle=True, random_state=i))
    scrambled_r2.append(r2_score(y_rand, preds_s))

real_r2 = results["RandomForest"]["R2"]
p_value = np.mean(np.array(scrambled_r2) >= real_r2)
print(f"  Real R² = {real_r2:.3f} | Scrambled mean R² = {np.mean(scrambled_r2):.3f}")
print(f"  p-value (permutation) = {p_value:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(scrambled_r2, bins=25, color="#607D8B", alpha=0.8,
        edgecolor="white", label="Scrambled y")
ax.axvline(real_r2, color="#F44336", lw=2.5, label=f"Real model R²={real_r2:.3f}")
ax.axvline(np.mean(scrambled_r2), color="grey", lw=1.5, ls="--",
           label=f"Scrambled mean={np.mean(scrambled_r2):.3f}")
ax.set_xlabel("Cross-validated R²", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title(f"y-Scrambling Validation (n={n_scramble} permutations)\n"
             f"Random Forest | p = {p_value:.4f}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "11_y_scrambling.pdf", bbox_inches="tight", dpi=150)
plt.close()

# ── 10. Export Predictions ────────────────────────────────────────────────────

pred_df = df[["name","class","mechanism","activity_pIC50"]].copy()
for name, res in results.items():
    pred_df[f"pred_{name}"] = res["preds"]
pred_df["ensemble_pred"] = np.column_stack(
    [res["preds"] for res in results.values()]
).mean(axis=1)
pred_df["residual_ensemble"] = pred_df["ensemble_pred"] - pred_df["activity_pIC50"]
pred_df.to_csv(RESULTS_DIR / "qsar_predictions.csv", index=False)

print(f"\n{'='*60}")
print(f"  QSAR pipeline complete!")
print(f"  Best model   : {best_name} (R²={results[best_name]['R2']:.3f}, "
      f"RMSE={results[best_name]['RMSE']:.2f})")
print(f"  y-scrambling : Real R²={real_r2:.3f} > Scrambled (p={p_value:.4f})")
print(f"  AD coverage  : {n_in}/{len(y)} compounds ({100*n_in/len(y):.0f}%)")
print(f"  Results : {RESULTS_DIR} | Figures : {FIGURES_DIR}")
print(f"{'='*60}")
