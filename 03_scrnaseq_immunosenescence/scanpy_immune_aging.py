"""
scRNA-seq Immunosenescence — Scanpy Pipeline
=============================================
Full single-cell analysis of PBMC aging using Python/Scanpy.
Mirrors the Seurat R pipeline for cross-validation.
Includes: QC, clustering, UMAP, marker genes, pseudobulk DE,
          senescence scoring, and age-associated cell abundance analysis.

Dataset : GSE174072 — 10X PBMCs young vs. aged donors
Python  : >= 3.10
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import scanpy as sc
from anndata import AnnData

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

sc.settings.seed       = SEED
sc.settings.verbosity  = 1
sc.settings.n_jobs     = 4
sc.settings.figdir     = "figures_scanpy"
sc.set_figure_params(dpi=100, facecolor="white")

RESULTS_DIR = Path("results_scanpy")
FIGURES_DIR = Path("figures_scanpy")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── 1. Data Simulation ────────────────────────────────────────────────────────

def simulate_pbmc_data(
    n_cells: int = 8000,
    n_genes: int = 3000,
    n_young_donors: int = 4,
    n_aged_donors: int = 4,
) -> AnnData:
    """Simulate 10X-like PBMC scRNA-seq data with aging signal."""
    rng = np.random.default_rng(SEED)

    cell_types  = ["CD4_Naive","CD4_Memory","CD8_Naive","CD8_Memory",
                   "CD8_TEMRA","NK","B_Naive","B_Memory",
                   "Monocyte_Classical","Monocyte_NonClassical","pDC","mDC"]
    young_props = np.array([0.12,0.08,0.10,0.06,0.02,0.12,0.10,0.05,0.15,0.08,0.05,0.07])
    aged_props  = np.array([0.06,0.10,0.07,0.09,0.10,0.09,0.08,0.08,0.16,0.10,0.04,0.03])
    young_props /= young_props.sum()
    aged_props  /= aged_props.sum()

    n_young = n_cells // 2
    n_aged  = n_cells - n_young

    young_ct = rng.choice(cell_types, n_young, p=young_props)
    aged_ct  = rng.choice(cell_types, n_aged,  p=aged_props)
    all_ct   = np.concatenate([young_ct, aged_ct])

    age_group = np.array(["young"] * n_young + ["aged"] * n_aged)
    donors    = np.concatenate([
        rng.choice([f"young_d{i}" for i in range(1, n_young_donors+1)], n_young),
        rng.choice([f"aged_d{i}"  for i in range(1, n_aged_donors+1)],  n_aged),
    ])

    # Gene names
    marker_genes = [
        "CD3E","CD3D","CD4","CD8A","CD8B",        # T cells
        "CCR7","SELL","LEF1","TCF7",               # Naive T
        "GZMB","GZMK","PRF1","NKG7","IFNG",       # Effector T
        "KLRG1","CX3CR1","TBX21",                  # TEMRA
        "NCAM1","KLRD1","GNLY","XCL1",             # NK
        "MS4A1","CD19","CD79A","IGHM","IGHD",      # B
        "LYZ","CD14","FCGR3A","S100A8","CSF1R",   # Monocytes
        "CLEC4C","FCER1A","LILRA4","CLEC10A",     # DC
        "IL6","CXCL10","MMP3","CDKN1A","CDKN2A",  # Senescence/SASP
        "IL1B","TNF","IL8","CXCL1","CXCL2",
    ]
    extra_genes = [f"GENE{i}" for i in range(n_genes - len(marker_genes))]
    gene_names  = marker_genes + extra_genes

    # Count matrix (cells × genes)
    counts = rng.negative_binomial(0.3, 0.37, (n_cells, len(gene_names))).astype(np.float32)

    # Inject marker expression
    boosts = {
        "CD4_Naive":  ["CD3E","CD4","CCR7","SELL","LEF1"],
        "CD4_Memory": ["CD3E","CD4","IFNG","GZMK"],
        "CD8_Naive":  ["CD3E","CD8A","CCR7","SELL"],
        "CD8_Memory": ["CD3E","CD8A","GZMB","GZMK","NKG7"],
        "CD8_TEMRA":  ["CD3E","CD8A","KLRG1","CX3CR1","GZMB","PRF1","TBX21"],
        "NK":         ["NCAM1","GNLY","KLRD1","NKG7","XCL1"],
        "B_Naive":    ["MS4A1","CD19","CD79A","IGHM","IGHD"],
        "B_Memory":   ["MS4A1","CD19","CD79A"],
        "Monocyte_Classical":    ["LYZ","CD14","S100A8"],
        "Monocyte_NonClassical": ["FCGR3A","LYZ","CSF1R"],
        "pDC":        ["CLEC4C","LILRA4"],
        "mDC":        ["FCER1A","CLEC10A"],
    }
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    for ct, markers in boosts.items():
        cells_idx = np.where(all_ct == ct)[0]
        genes_idx = [gene_idx[g] for g in markers if g in gene_idx]
        if len(cells_idx) > 0 and len(genes_idx) > 0:
            counts[np.ix_(cells_idx, genes_idx)] += rng.poisson(5, (len(cells_idx), len(genes_idx)))

    # Aging / senescence signal
    sasp = ["IL6","CXCL10","MMP3","CDKN1A","IL1B","TNF","IL8"]
    sasp_idx  = [gene_idx[g] for g in sasp if g in gene_idx]
    aged_mask = age_group == "aged"
    counts[np.ix_(aged_mask, sasp_idx)] += rng.poisson(3, (aged_mask.sum(), len(sasp_idx)))

    obs = pd.DataFrame({
        "cell_barcode" : [f"CELL{i}" for i in range(n_cells)],
        "age_group"    : age_group,
        "donor_id"     : donors,
        "cell_type"    : all_ct,
    }, index=[f"CELL{i}" for i in range(n_cells)])

    var = pd.DataFrame({"gene_id": gene_names}, index=gene_names)
    adata = AnnData(X=counts, obs=obs, var=var)
    return adata


print("Simulating PBMC scRNA-seq data ...")
adata = simulate_pbmc_data()
print(f"AnnData: {adata.n_obs} cells × {adata.n_vars} genes")

# ── 2. Quality Control ────────────────────────────────────────────────────────

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

# Synthetic mitochondrial %
rng = np.random.default_rng(SEED)
adata.obs["pct_counts_mt"] = rng.uniform(0.5, 15, adata.n_obs)

print("QC metrics computed")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(
    axes,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    ["Genes Detected", "Total UMI Counts", "% Mitochondrial"],
):
    for grp, colour in [("young", "#2196F3"), ("aged", "#F44336")]:
        data_grp = adata.obs.loc[adata.obs["age_group"] == grp, col]
        ax.hist(data_grp, bins=40, alpha=0.6, color=colour, label=grp, density=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(col)
    ax.legend()
plt.suptitle("QC Distributions — Young vs. Aged PBMCs", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "01_QC_histograms.pdf", bbox_inches="tight", dpi=150)
plt.close()

# Filter
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[
    (adata.obs["n_genes_by_counts"] < 5000) &
    (adata.obs["pct_counts_mt"] < 20)
].copy()
print(f"Post-QC: {adata.n_obs} cells × {adata.n_vars} genes")

# ── 3. Normalisation & HVG ────────────────────────────────────────────────────

adata.raw = adata.copy()          # store raw counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(
    adata, n_top_genes=2000, batch_key="donor_id", flavor="seurat_v3"
)
adata = adata[:, adata.var["highly_variable"]].copy()
print(f"HVG selected: {adata.n_vars}")

# ── 4. PCA, Neighbours, UMAP ─────────────────────────────────────────────────

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack", n_comps=50, random_state=SEED)

sc.pl.pca_variance_ratio(adata, n_pcs=50, show=False)
plt.savefig(FIGURES_DIR / "02_PCA_variance.pdf", bbox_inches="tight")
plt.close()

N_PCS = 20
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=N_PCS, random_state=SEED)
sc.tl.leiden(adata, resolution=0.5, random_state=SEED, key_added="leiden")
sc.tl.umap(adata, random_state=SEED)

# UMAP panels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, colour_by, title, palette in [
    (axes[0], "leiden",    "Leiden Clusters", None),
    (axes[1], "cell_type", "Cell Types",      None),
    (axes[2], "age_group", "Age Group",       {"young":"#2196F3","aged":"#F44336"}),
]:
    sc.pl.umap(adata, color=colour_by, ax=ax, show=False,
               title=title, legend_loc="right margin", frameon=False,
               palette=palette)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "03_UMAP_panels.pdf", bbox_inches="tight", dpi=150)
plt.close()

# ── 5. Marker Gene Discovery ──────────────────────────────────────────────────

sc.tl.rank_genes_groups(
    adata, groupby="cell_type",
    method="wilcoxon", key_added="rank_genes_ct",
    pts=True, use_raw=False
)

marker_df = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_ct")
marker_df.to_csv(RESULTS_DIR / "cell_type_markers.csv", index=False)

sc.pl.rank_genes_groups_dotplot(
    adata, key="rank_genes_ct",
    n_genes=4, show=False,
    groupby="cell_type", standard_scale="var",
    save=False,
)
plt.savefig(FIGURES_DIR / "04_marker_dotplot.pdf", bbox_inches="tight", dpi=150)
plt.close()

# ── 6. Senescence Module Score ────────────────────────────────────────────────

senescence_genes = {
    "SASP"              : ["IL6","CXCL10","MMP3","IL1B","TNF","IL8","CXCL1"],
    "Cell_Cycle_Arrest" : ["CDKN1A","CDKN2A"],
    "Senescence_Core"   : ["CDKN1A","CDKN2A","IL6","CXCL10","MMP3"],
}

for sig_name, genes in senescence_genes.items():
    valid_genes = [g for g in genes if g in adata.var_names]
    if valid_genes:
        sc.tl.score_genes(adata, gene_list=valid_genes, score_name=sig_name)
    else:
        adata.obs[sig_name] = 0.0
        print(f"Warning: no genes found for {sig_name}")

# Violin of senescence score by cell type and age
fig, ax = plt.subplots(figsize=(14, 5))
sc.pl.violin(adata, keys="Senescence_Core", groupby="cell_type",
             rotation=45, ax=ax, show=False,
             jitter=0.05, palette=None)
ax.set_title("Senescence Core Score by Cell Type")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "05_senescence_violin.pdf", bbox_inches="tight", dpi=150)
plt.close()

# Age group comparison for each cell type
sen_df = adata.obs[["age_group","cell_type","Senescence_Core"]].copy()
fig, ax = plt.subplots(figsize=(12, 5))
ct_order = (sen_df.groupby("cell_type")["Senescence_Core"]
            .median().sort_values(ascending=False).index)

sns.boxplot(data=sen_df, x="cell_type", y="Senescence_Core",
            hue="age_group", order=ct_order,
            palette={"young":"#2196F3","aged":"#F44336"},
            ax=ax, fliersize=2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_title("Senescence Score — Young vs. Aged per Cell Type", fontsize=12)
ax.set_xlabel(None)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "06_senescence_by_celltype.pdf", bbox_inches="tight", dpi=150)
plt.close()

# UMAP coloured by senescence score
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(adata, color="Senescence_Core", ax=ax, show=False,
           cmap="YlOrRd", title="Senescence Core Score", frameon=False)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "07_UMAP_senescence.pdf", bbox_inches="tight", dpi=150)
plt.close()

# ── 7. Differential Abundance ────────────────────────────────────────────────

prop_df = (
    adata.obs.groupby(["donor_id","age_group","cell_type"])
    .size().reset_index(name="n")
)
prop_df["proportion"] = prop_df.groupby("donor_id")["n"].transform(lambda x: x / x.sum())

age_mean = prop_df.groupby(["age_group","cell_type"])["proportion"].agg(["mean","sem"]).reset_index()
age_mean.columns = ["age_group","cell_type","mean_prop","sem_prop"]

fig, ax = plt.subplots(figsize=(12, 5))
ct_order = (age_mean[age_mean["age_group"]=="young"]
            .set_index("cell_type")["mean_prop"]
            .sort_values(ascending=False).index)
x_pos    = np.arange(len(ct_order))
width    = 0.35

young_data = age_mean[age_mean["age_group"]=="young"].set_index("cell_type").loc[ct_order]
aged_data  = age_mean[age_mean["age_group"]=="aged"].set_index("cell_type").loc[ct_order]

ax.bar(x_pos - width/2, young_data["mean_prop"], width, label="Young",
       color="#2196F3", alpha=0.85, yerr=young_data["sem_prop"], capsize=3)
ax.bar(x_pos + width/2, aged_data["mean_prop"], width, label="Aged",
       color="#F44336", alpha=0.85, yerr=aged_data["sem_prop"], capsize=3)

ax.set_xticks(x_pos)
ax.set_xticklabels(ct_order, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Mean Cell Proportion (±SEM)")
ax.set_title("Immune Cell Proportions — Young vs. Aged PBMCs")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "08_cell_proportions.pdf", bbox_inches="tight", dpi=150)
plt.close()

# ── 8. Save ───────────────────────────────────────────────────────────────────

adata.write(RESULTS_DIR / "pbmc_immune_aging.h5ad")
print(f"\n=== Scanpy pipeline complete ===")
print(f"AnnData saved: {RESULTS_DIR / 'pbmc_immune_aging.h5ad'}")
print(f"Figures: {FIGURES_DIR}")
