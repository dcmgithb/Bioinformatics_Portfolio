"""
Geroprotector Chemical Space Analysis
======================================
Comprehensive cheminformatics characterisation of 75 curated
geroprotective compounds using RDKit.

Analyses:
  1. Property profiling (MW, LogP, HBD, HBA, TPSA, QED, SA score)
  2. Drug-likeness filtering (Lipinski Ro5, Veber, PAINS, MPO)
  3. Fingerprint-based chemical space (PCA + UMAP)
  4. Scaffold diversity (Murcko scaffolds, Bemis-Murcko tree)
  5. Tanimoto similarity clustering (heatmap + dendrogram)
  6. Mechanism-class chemical profile comparisons
  7. Structural alert analysis

Dataset : data/geroprotectors.csv (75 compounds, annotated)
Python  : >= 3.10 | RDKit >= 2023.09
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# RDKit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not found — skipping UMAP plot")

from utils.chem_utils import (
    validate_smiles_column, enrich_dataframe, fingerprint_matrix,
    tanimoto_matrix, get_scaffold, morgan_fp, calc_all_descriptors,
    descriptor_matrix, sa_score, mpo_score, is_pains
)

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── Palettes ──────────────────────────────────────────────────────────────────

CLASS_PAL   = {"senolytic": "#F44336", "senomorphic": "#2196F3"}
MECHANISM_PAL = {
    "BCL-2 inhibition"     : "#C62828",
    "mTORC1 inhibition"    : "#1565C0",
    "CDK4/6 inhibition"    : "#E65100",
    "AMPK activation"      : "#2E7D32",
    "SIRT1 activation"     : "#6A1B9A",
    "NAD+ precursor"       : "#00695C",
    "JAK1/2 inhibition"    : "#4527A0",
    "PARP inhibition"      : "#AD1457",
    "PI3K inhibition"      : "#F57F17",
    "BTK inhibition"       : "#37474F",
    "Autophagy induction"  : "#558B2F",
    "Antioxidant"          : "#EF6C00",
    "COX inhibition"       : "#795548",
    "NF-kB inhibition"     : "#0277BD",
    "other"                : "#9E9E9E",
}

# ── 1. Load & Enrich Compound Database ───────────────────────────────────────

print("═" * 60)
print("  GEROPROTECTOR CHEMICAL SPACE ANALYSIS")
print("═" * 60)

df_raw = pd.read_csv("data/geroprotectors.csv")
print(f"\nLoaded {len(df_raw)} compounds from database")

df = enrich_dataframe(df_raw, smiles_col="smiles")
mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]

print(f"After SMILES validation: {len(df)} compounds")
print(f"  Senolytics:    {(df['class']=='senolytic').sum()}")
print(f"  Senomorphics:  {(df['class']=='senomorphic').sum()}")
print(f"  Lipinski pass: {df['lipinski_ok'].sum()}")
print(f"  PAINS alerts:  {df['pains'].sum()}")
print(f"  Drug-like:     {df['drug_like'].sum()}")

df.to_csv(RESULTS_DIR / "geroprotectors_enriched.csv", index=False)

# ── 2. Property Distribution Comparison ───────────────────────────────────────

print("\n[2] Computing property distributions ...")

props = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotatableBonds",
         "QED", "mpo_score", "sa_score", "Fsp3", "AromaticRings"]

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.flatten()

for ax, prop in zip(axes, props):
    for cls, colour in CLASS_PAL.items():
        sub = df[df["class"] == cls][prop].dropna()
        ax.hist(sub, bins=18, alpha=0.65, color=colour, label=cls,
                density=True, edgecolor="white", linewidth=0.4)
        ax.axvline(sub.median(), color=colour, lw=1.5, ls="--")
    ax.set_title(prop, fontsize=10, fontweight="bold")
    ax.set_xlabel(prop, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

# Lipinski rules guide lines
GUIDES = {"MW": 500, "LogP": 5, "HBD": 5, "HBA": 10, "TPSA": 140}
for ax, prop in zip(axes, props):
    if prop in GUIDES:
        ax.axvline(GUIDES[prop], color="black", lw=1, ls=":", alpha=0.6,
                   label=f"Ro5 limit ({GUIDES[prop]})")

axes[-1].remove()   # remove last empty subplot
fig.suptitle("Physicochemical Properties: Senolytics vs. Senomorphics\n"
             "Dashed = median | Dotted = Lipinski limit",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "01_property_distributions.pdf",
            bbox_inches="tight", dpi=150)
plt.close()

# Statistical comparison
from scipy.stats import mannwhitneyu
stat_rows = []
for prop in props:
    a = df[df["class"]=="senolytic"][prop].dropna()
    b = df[df["class"]=="senomorphic"][prop].dropna()
    if len(a) > 2 and len(b) > 2:
        stat, pval = mannwhitneyu(a, b, alternative="two-sided")
        stat_rows.append({"property": prop,
                          "median_senolytic"  : round(a.median(), 3),
                          "median_senomorphic": round(b.median(), 3),
                          "MW_statistic"      : stat,
                          "p_value"           : pval,
                          "significant"       : pval < 0.05})
stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(RESULTS_DIR / "property_stats.csv", index=False)

print("  Significant property differences (p<0.05):")
print(stats_df[stats_df["significant"]][["property","median_senolytic","median_senomorphic","p_value"]].to_string())

# ── 3. Drug-likeness Radar Chart ─────────────────────────────────────────────

print("\n[3] Drug-likeness radar charts ...")

# Normalise each property to [0,1] for radar
radar_props = ["QED", "mpo_score", "Fsp3", "AromaticRings"]
radar_labels = ["QED", "MPO Score", "Fsp3", "Aromatic Rings (norm)"]

# Also add normalised versions of Lipinski properties
df["MW_norm"]   = 1 - (df["MW"]   / 800).clip(0, 1)
df["LogP_norm"] = 1 - (df["LogP"].abs() / 5).clip(0, 1)
df["TPSA_norm"] = 1 - (df["TPSA"] / 150).clip(0, 1)

radar_cols   = ["QED", "mpo_score", "MW_norm", "LogP_norm", "TPSA_norm", "Fsp3"]
radar_labels = ["QED", "MPO", "MW\n(inv)", "logP\n(inv)", "TPSA\n(inv)", "Fsp3"]
n_sides      = len(radar_cols)
angles       = np.linspace(0, 2 * np.pi, n_sides, endpoint=False).tolist()
angles      += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25","0.50","0.75","1.0"], fontsize=7)

for cls, colour in CLASS_PAL.items():
    means = df[df["class"] == cls][radar_cols].mean().tolist()
    means += means[:1]
    ax.plot(angles, means, "o-", lw=2, color=colour, label=cls)
    ax.fill(angles, means, alpha=0.15, color=colour)

ax.set_title("Drug-likeness Radar\nSenolytics vs. Senomorphics",
             fontsize=12, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
fig.savefig(FIGURES_DIR / "02_radar_chart.pdf", bbox_inches="tight", dpi=150)
plt.close()

# ── 4. Chemical Space — PCA on Morgan Fingerprints ───────────────────────────

print("\n[4] Chemical space mapping (Morgan FP → PCA + UMAP) ...")

fp_matrix = fingerprint_matrix(mols, fp_type="morgan", radius=2, n_bits=2048)
print(f"  Fingerprint matrix: {fp_matrix.shape}")

# PCA
pca  = PCA(n_components=10, random_state=SEED)
pca_coords = pca.fit_transform(fp_matrix)
df["PC1"], df["PC2"] = pca_coords[:, 0], pca_coords[:, 1]

explained = pca.explained_variance_ratio_
print(f"  PC1: {explained[0]*100:.1f}%  PC2: {explained[1]*100:.1f}%")

# PCA scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, colour_by, title, palette in [
    (axes[0], "class",     "Coloured by Class",     CLASS_PAL),
    (axes[1], "mechanism", "Coloured by Mechanism", MECHANISM_PAL),
]:
    groups = df[colour_by].apply(
        lambda x: x if x in palette else "other"
    )
    for grp in groups.unique():
        mask   = groups == grp
        colour = palette.get(grp, "#9E9E9E")
        ax.scatter(df.loc[mask, "PC1"], df.loc[mask, "PC2"],
                   c=colour, s=60, alpha=0.85, label=grp, edgecolors="white", lw=0.4)

    # Annotate a few key compounds
    for _, row in df.nlargest(5, "activity_pIC50").iterrows():
        ax.annotate(row["name"][:12],
                    xy=(row["PC1"], row["PC2"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="grey"))

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2 if colour_by=="mechanism" else 1,
              loc="lower right", markerscale=1.5)
    ax.grid(alpha=0.3)

fig.suptitle("Chemical Space of Geroprotectors — Morgan Fingerprint PCA",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "03_chemical_space_pca.pdf",
            bbox_inches="tight", dpi=150)
plt.close()

# UMAP
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.3,
                        random_state=SEED, metric="jaccard")
    umap_coords = reducer.fit_transform(fp_matrix)
    df["UMAP1"], df["UMAP2"] = umap_coords[:, 0], umap_coords[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, colour_by, title, palette in [
        (axes[0], "class",     "UMAP — Class",     CLASS_PAL),
        (axes[1], "mechanism", "UMAP — Mechanism", MECHANISM_PAL),
    ]:
        groups = df[colour_by].apply(lambda x: x if x in palette else "other")
        for grp in groups.unique():
            mask   = groups == grp
            colour = palette.get(grp, "#9E9E9E")
            ax.scatter(df.loc[mask, "UMAP1"], df.loc[mask, "UMAP2"],
                       c=colour, s=65, alpha=0.85, label=grp,
                       edgecolors="white", lw=0.4)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP1", fontsize=10); ax.set_ylabel("UMAP2", fontsize=10)
        ax.legend(fontsize=7, ncol=1, loc="best", markerscale=1.5)
        ax.grid(alpha=0.3)

    fig.suptitle("Chemical Space — UMAP (Jaccard metric on Morgan FP)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_chemical_space_umap.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("  UMAP saved.")
else:
    print("  UMAP skipped (install umap-learn)")

# ── 5. Tanimoto Similarity Heatmap ────────────────────────────────────────────

print("\n[5] Tanimoto similarity heatmap ...")

sim_matrix = tanimoto_matrix(fp_matrix)
dist_matrix = 1 - sim_matrix
np.fill_diagonal(dist_matrix, 0)

# Hierarchical clustering
linkage = hierarchy.linkage(squareform(dist_matrix), method="ward")
order   = hierarchy.leaves_list(linkage)

sim_sorted = sim_matrix[np.ix_(order, order)]
names_sorted = df["name"].iloc[order].tolist()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.tril(np.ones_like(sim_sorted, dtype=bool), k=-1)
sns.heatmap(
    sim_sorted, ax=ax,
    cmap=sns.color_palette("YlOrRd", as_cmap=True),
    vmin=0, vmax=1,
    xticklabels=names_sorted, yticklabels=names_sorted,
    linewidths=0, mask=mask,
    cbar_kws={"label": "Tanimoto Similarity", "shrink": 0.7},
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=5.5)
ax.set_title("Tanimoto Similarity Heatmap — Geroprotectors\n"
             "Hierarchical clustering (Ward linkage, Morgan FP)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "05_tanimoto_heatmap.pdf",
            bbox_inches="tight", dpi=150)
plt.close()

# Most similar pairs
pairs = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        pairs.append({"compound_A": df["name"].iloc[i],
                      "compound_B": df["name"].iloc[j],
                      "tanimoto"  : sim_matrix[i, j],
                      "same_class": df["class"].iloc[i] == df["class"].iloc[j]})
pairs_df = pd.DataFrame(pairs).sort_values("tanimoto", ascending=False)
pairs_df.to_csv(RESULTS_DIR / "tanimoto_pairs.csv", index=False)

print(f"  Most similar pair: {pairs_df.iloc[0]['compound_A']} — "
      f"{pairs_df.iloc[0]['compound_B']} "
      f"(Tanimoto = {pairs_df.iloc[0]['tanimoto']:.3f})")

# ── 6. Scaffold Analysis ─────────────────────────────────────────────────────

print("\n[6] Scaffold diversity analysis ...")

scaffold_counts = Counter(df["scaffold"])
n_unique  = len(scaffold_counts)
n_singletons = sum(1 for v in scaffold_counts.values() if v == 1)
diversity_ratio = n_unique / len(df)

print(f"  Total compounds       : {len(df)}")
print(f"  Unique scaffolds      : {n_unique}")
print(f"  Singletons            : {n_singletons}")
print(f"  Scaffold diversity    : {diversity_ratio:.2f}")

top_scaffolds = pd.DataFrame(
    scaffold_counts.most_common(15),
    columns=["scaffold_smiles", "count"]
).assign(fraction=lambda d: d["count"] / len(df))
top_scaffolds.to_csv(RESULTS_DIR / "scaffold_diversity.csv", index=False)

# Bar chart of top scaffolds
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(top_scaffolds)), top_scaffolds["count"],
               color="#607D8B", alpha=0.85)
ax.set_yticks(range(len(top_scaffolds)))
ax.set_yticklabels(
    [f"Scaffold {i+1}\n({row.scaffold_smiles[:30]}...)"
     for i, (_, row) in enumerate(top_scaffolds.iterrows())],
    fontsize=7
)
ax.set_xlabel("Number of Compounds", fontsize=11)
ax.set_title(f"Top 15 Murcko Scaffolds in Geroprotector Set\n"
             f"Scaffold diversity ratio = {diversity_ratio:.2f} | "
             f"{n_unique} unique scaffolds from {len(df)} compounds",
             fontsize=11, fontweight="bold")
for bar, count in zip(bars, top_scaffolds["count"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            str(count), va="center", fontsize=8)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "06_scaffold_diversity.pdf",
            bbox_inches="tight", dpi=150)
plt.close()

# ── 7. PAINS & Drug-likeness Summary Dashboard ───────────────────────────────

print("\n[7] Drug-likeness / PAINS summary ...")

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 7a. Lipinski pie per class
for i, cls in enumerate(["senolytic", "senomorphic"]):
    ax = fig.add_subplot(gs[0, i])
    sub    = df[df["class"] == cls]
    counts = [sub["drug_like"].sum(), (~sub["drug_like"]).sum()]
    colours = ["#4CAF50", "#F44336"]
    ax.pie(counts, labels=["Drug-like", "Fails Ro5+Veber"],
           colors=colours, autopct="%1.0f%%",
           startangle=140, textprops={"fontsize": 9})
    ax.set_title(f"{cls.title()}\n(n={len(sub)})", fontsize=10, fontweight="bold")

# 7b. PAINS
ax = fig.add_subplot(gs[0, 2])
pains_counts = df.groupby(["class","pains"]).size().unstack(fill_value=0)
pains_counts.plot(kind="bar", ax=ax, color=["#4CAF50","#F44336"],
                  width=0.6, edgecolor="white")
ax.set_title("PAINS Alerts by Class", fontsize=10, fontweight="bold")
ax.set_xlabel(None); ax.set_ylabel("Count", fontsize=9)
ax.set_xticklabels(["Senolytic","Senomorphic"], rotation=0)
ax.legend(["No alert","PAINS alert"], fontsize=8)

# 7c. QED violin
ax = fig.add_subplot(gs[1, 0])
for x, (cls, colour) in enumerate(CLASS_PAL.items()):
    data = df[df["class"] == cls]["QED"]
    vp = ax.violinplot(data, positions=[x], widths=0.6)
    for pc in vp["bodies"]:
        pc.set_facecolor(colour); pc.set_alpha(0.8)
    ax.boxplot(data, positions=[x], widths=0.2,
               patch_artist=True, boxprops=dict(facecolor="white"),
               medianprops=dict(color="black", lw=1.5), flierprops=dict(markersize=3))
ax.set_xticks([0, 1]); ax.set_xticklabels(["Senolytic","Senomorphic"])
ax.set_ylabel("QED Drug-likeness Score", fontsize=9)
ax.set_title("QED Distribution", fontsize=10, fontweight="bold")

# 7d. MPO score scatter vs. pIC50
ax = fig.add_subplot(gs[1, 1])
for cls, colour in CLASS_PAL.items():
    sub = df[df["class"] == cls]
    ax.scatter(sub["mpo_score"], sub["activity_pIC50"],
               c=colour, s=40, alpha=0.75, label=cls, edgecolors="white", lw=0.4)
ax.set_xlabel("MPO Score (lead-likeness)", fontsize=9)
ax.set_ylabel("Activity (pIC50)", fontsize=9)
ax.set_title("MPO Score vs. Activity", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)

# 7e. SA score
ax = fig.add_subplot(gs[1, 2])
for cls, colour in CLASS_PAL.items():
    sub = df[df["class"] == cls]["sa_score"]
    ax.hist(sub, bins=12, alpha=0.7, color=colour, label=cls,
            density=True, edgecolor="white")
ax.set_xlabel("Synthetic Accessibility Score\n(1=easy, 10=hard)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Synthetic Accessibility", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.axvline(5, color="grey", ls="--", lw=1, label="SA=5 threshold")

fig.suptitle("Geroprotector Drug-likeness Dashboard",
             fontsize=14, fontweight="bold", y=1.01)
fig.savefig(FIGURES_DIR / "07_druglikeness_dashboard.pdf",
            bbox_inches="tight", dpi=150)
plt.close()

# ── 8. Top Candidates Summary ────────────────────────────────────────────────

print("\n[8] Top geroprotector candidates ...")

top_candidates = (
    df.sort_values(["drug_like","mpo_score","activity_pIC50"],
                   ascending=[False, False, False])
    .head(20)
    [["name","class","mechanism","activity_pIC50",
      "MW","LogP","QED","mpo_score","pains","drug_like"]]
)
top_candidates.to_csv(RESULTS_DIR / "top_candidates.csv", index=False)
print(top_candidates[["name","class","activity_pIC50","QED","mpo_score"]].to_string())

print(f"\n{'='*60}")
print(f"  Analysis complete!")
print(f"  Results : {RESULTS_DIR}")
print(f"  Figures : {FIGURES_DIR}")
print(f"  Key facts:")
print(f"    • {len(df)} geroprotectors analysed")
print(f"    • {n_unique} unique Murcko scaffolds (diversity = {diversity_ratio:.2f})")
print(f"    • {df['drug_like'].sum()} drug-like compounds ({df['drug_like'].mean()*100:.0f}%)")
print(f"    • {df['pains'].sum()} PAINS alerts flagged")
print(f"    • Best QED: {df.loc[df['QED'].idxmax(),'name']} ({df['QED'].max():.3f})")
print(f"{'='*60}")
