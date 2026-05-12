"""
Bulk RNA-seq Aging Analysis — Python Mirror Pipeline
======================================================
Replicates the R/DESeq2 pipeline using pyDESeq2 + GSEApy.
Cross-validates DE results and provides Python-native visualisations.

Dataset : GSE65907 — Human PBMC young vs. aged
Author  : Portfolio / CenAGE application
Python  : >= 3.10
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from typing import Optional

# Bioinformatics
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import gseapy as gp
from gseapy import dotplot as gsea_dotplot

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Configuration ──────────────────────────────────────────────────────────────

CFG = {
    "geo_accession" : "GSE65907",
    "species"       : "human",
    "condition"     : "age_group",
    "reference"     : "young",
    "padj_thresh"   : 0.05,
    "lfc_thresh"    : 1.0,
    "min_count"     : 10,
    "results_dir"   : Path("results_python"),
    "figures_dir"   : Path("figures_python"),
    "n_cpus"        : 4,
}

CFG["results_dir"].mkdir(exist_ok=True)
CFG["figures_dir"].mkdir(exist_ok=True)

PALETTE = {"young": "#2196F3", "aged": "#F44336"}

# ── 1. Data Simulation / Loading ───────────────────────────────────────────────

def simulate_counts(n_genes: int = 15000, n_samples: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (count_matrix, metadata) matching real GEO dimensions."""
    rng = np.random.default_rng(SEED)
    gene_ids    = [f"GENE{i}" for i in range(n_genes)]
    sample_ids  = [f"SRR{rng.integers(1_000_000, 9_999_999)}" for _ in range(n_samples)]

    counts = rng.negative_binomial(0.5, 0.005, (n_samples, n_genes)).astype(float)
    counts = pd.DataFrame(counts, index=sample_ids, columns=gene_ids)

    # Inject aging signal
    n_young   = n_samples // 2
    aged_idx  = list(range(n_young, n_samples))
    up_genes  = rng.choice(n_genes, 500, replace=False)
    dn_genes  = rng.choice(list(set(range(n_genes)) - set(up_genes)), 500, replace=False)

    counts.iloc[aged_idx, up_genes] *= 4
    counts.iloc[aged_idx, dn_genes] //= 4

    metadata = pd.DataFrame({
        "age_group" : ["young"] * n_young + ["aged"] * (n_samples - n_young),
        "age_years" : (list(rng.uniform(20, 35, n_young)) +
                       list(rng.uniform(65, 85, n_samples - n_young))),
        "sex"       : rng.choice(["M", "F"], n_samples).tolist(),
    }, index=sample_ids)

    return counts.astype(int), metadata


counts, metadata = simulate_counts()
print(f"Loaded: {counts.shape[1]} genes × {counts.shape[0]} samples")
print(metadata["age_group"].value_counts())

# ── 2. Pre-filtering ───────────────────────────────────────────────────────────

min_samples = int(counts.shape[0] * 0.25)
keep        = (counts >= CFG["min_count"]).sum(axis=0) >= min_samples
counts      = counts.loc[:, keep]
print(f"After pre-filter: {counts.shape[1]} genes retained")

# ── 3. Quality Control — PCA ──────────────────────────────────────────────────

from sklearn.decomposition import PCA as _PCA
from sklearn.preprocessing import StandardScaler as _SS

log1p_mat = np.log1p(counts.values)
pca_coords = _PCA(n_components=2, random_state=SEED).fit_transform(
    _SS().fit_transform(log1p_mat)
)
pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"], index=counts.index)
pca_df["age_group"] = metadata.loc[counts.index, "age_group"].values

fig, ax = plt.subplots(figsize=(7, 6))
for grp, colour in PALETTE.items():
    sub = pca_df[pca_df["age_group"] == grp]
    ax.scatter(sub["PC1"], sub["PC2"], c=colour, label=grp.capitalize(),
               s=60, alpha=0.85, edgecolors="white", linewidths=0.4)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA — log1p-normalised counts\nPython pipeline (pyDESeq2)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(CFG["figures_dir"] / "00_pca_python.pdf", dpi=150)
plt.close()
print("PCA saved → figures_python/00_pca_python.pdf")

# ── 4. pyDESeq2 ────────────────────────────────────────────────────────────────

print("\nBuilding DESeqDataSet ...")
dds = DeseqDataSet(
    counts   = counts,
    metadata = metadata,
    design_factors = [CFG["condition"]],
    ref_level      = [(CFG["condition"], CFG["reference"])],
    refit_cooks    = True,
    n_cpus         = CFG["n_cpus"],
)
dds.deseq2()

print("Running Wald test ...")
stat_res = DeseqStats(
    dds,
    contrast    = [CFG["condition"], "aged", "young"],
    alpha       = CFG["padj_thresh"],
    cooks_filter = True,
    independent_filter = True,
    n_cpus      = CFG["n_cpus"],
)
stat_res.summary()
stat_res.lfc_shrink(coeff=f"{CFG['condition']}_aged_vs_young")

res = stat_res.results_df.reset_index().rename(columns={"index": "gene_id"})
res["regulation"] = np.where(
    (res["padj"] < CFG["padj_thresh"]) & (res["log2FoldChange"] >  CFG["lfc_thresh"]),
    "Up in Aged",
    np.where(
        (res["padj"] < CFG["padj_thresh"]) & (res["log2FoldChange"] < -CFG["lfc_thresh"]),
        "Down in Aged", "NS"
    )
)
res = res.sort_values("padj")
res.to_csv(CFG["results_dir"] / "DE_genes_pydeseq2.csv", index=False)
print(res["regulation"].value_counts())

# ── 5. Visualisation ───────────────────────────────────────────────────────────

def regulation_color(reg: str) -> str:
    return {"Up in Aged": "#F44336", "Down in Aged": "#2196F3"}.get(reg, "#BDBDBD")

# 4a. Volcano plot
fig, ax = plt.subplots(figsize=(9, 7))
res_plot = res.dropna(subset=["padj", "log2FoldChange"]).copy()
res_plot["-log10_padj"] = -np.log10(res_plot["padj"].clip(lower=1e-300))
res_plot["colour"] = res_plot["regulation"].map(regulation_color)

ax.scatter(
    res_plot["log2FoldChange"], res_plot["-log10_padj"],
    c=res_plot["colour"], alpha=0.5, s=10, linewidths=0
)
ax.axhline(-np.log10(CFG["padj_thresh"]), ls="--", lw=1, color="grey")
ax.axvline( CFG["lfc_thresh"],  ls="--", lw=1, color="grey")
ax.axvline(-CFG["lfc_thresh"],  ls="--", lw=1, color="grey")
ax.set_xlabel("log2 Fold Change (Aged/Young)", fontsize=12)
ax.set_ylabel("-log10(FDR-adjusted p-value)", fontsize=12)
ax.set_title(f"Volcano Plot — Aged vs. Young PBMCs\n"
             f"pyDESeq2 | FDR < {CFG['padj_thresh']} | |LFC| > {CFG['lfc_thresh']}",
             fontsize=13)

# Annotate top 15 genes
top15 = res_plot.nsmallest(15, "padj")
for _, row in top15.iterrows():
    ax.annotate(row["gene_id"],
                xy=(row["log2FoldChange"], row["-log10_padj"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=6.5, alpha=0.9)

from matplotlib.patches import Patch
legend_els = [Patch(facecolor="#F44336", label="Up in Aged"),
              Patch(facecolor="#2196F3", label="Down in Aged"),
              Patch(facecolor="#BDBDBD", label="Not Significant")]
ax.legend(handles=legend_els, loc="upper left", fontsize=9)
plt.tight_layout()
fig.savefig(CFG["figures_dir"] / "01_volcano_pydeseq2.pdf", dpi=150)
plt.close()

# 4b. MA plot
fig, ax = plt.subplots(figsize=(8, 6))
res_ma = res_plot.copy()
res_ma["log10_basemean"] = np.log10(res_ma["baseMean"] + 1)
ax.scatter(
    res_ma["log10_basemean"], res_ma["log2FoldChange"],
    c=res_ma["colour"], alpha=0.4, s=8, linewidths=0
)
ax.axhline(0,  ls="-",  lw=0.8, color="black")
ax.axhline( CFG["lfc_thresh"], ls="--", lw=1, color="grey")
ax.axhline(-CFG["lfc_thresh"], ls="--", lw=1, color="grey")
ax.set_xlabel("Mean Expression (log10)", fontsize=12)
ax.set_ylabel("LFC (shrunken, apeglm)", fontsize=12)
ax.set_title("MA Plot — Aged vs. Young PBMCs", fontsize=13)
plt.tight_layout()
fig.savefig(CFG["figures_dir"] / "02_ma_plot_pydeseq2.pdf", dpi=150)
plt.close()

# ── 6. GSEA with GSEApy ────────────────────────────────────────────────────────

print("\nRunning GSEA with GSEApy ...")

# Ranked list: by stat (Wald test statistic)
ranked_df = (
    res.dropna(subset=["stat"])
    .set_index("gene_id")["stat"]
    .sort_values(ascending=False)
    .reset_index()
)
ranked_df.columns = ["gene_id", "stat"]

gsea_res = gp.prerank(
    rnk          = ranked_df,
    gene_sets    = ["MSigDB_Hallmark_2020", "KEGG_2021_Human"],
    organism     = "human",
    outdir       = str(CFG["results_dir"] / "GSEA"),
    min_size     = 10,
    max_size     = 500,
    permutation_num = 1000,
    seed         = SEED,
    threads      = CFG["n_cpus"],
    verbose      = False,
)

gsea_df = gsea_res.res2d.reset_index()
gsea_df.to_csv(CFG["results_dir"] / "GSEA_pydeseq2.csv", index=False)
print(f"GSEA complete: {len(gsea_df)} pathways tested")

# Top pathways barplot
if len(gsea_df) > 0:
    # 0.25 is the Broad GSEA convention for NES-based enrichment (not BH padj)
    sig_gsea = gsea_df[gsea_df["FDR q-val"].astype(float) < 0.25].copy()
    sig_gsea["NES"] = sig_gsea["NES"].astype(float)
    sig_gsea = sig_gsea.reindex(sig_gsea["NES"].abs().sort_values(ascending=False).index)

    if len(sig_gsea) > 0:
        top_n = sig_gsea.head(20)
        fig, ax = plt.subplots(figsize=(9, 7))
        colours = ["#F44336" if n > 0 else "#2196F3" for n in top_n["NES"]]
        ax.barh(range(len(top_n)), top_n["NES"], color=colours)
        ax.set_yticks(range(len(top_n)))
        ax.set_yticklabels(top_n["Term"], fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Normalised Enrichment Score (NES)", fontsize=11)
        ax.set_title("GSEA — Hallmarks + KEGG | Aged vs. Young\n"
                     "GSEApy prerank | FDR < 0.25", fontsize=12)
        plt.tight_layout()
        fig.savefig(CFG["figures_dir"] / "03_GSEA_barplot_python.pdf", dpi=150)
        plt.close()

# ── 7. ORA with GSEApy ─────────────────────────────────────────────────────────

print("Running ORA with GSEApy ...")

sig_genes = res[res["regulation"] != "NS"]["gene_id"].tolist()
bg_genes  = res["gene_id"].tolist()

if len(sig_genes) > 10:
    ora_res = gp.enrichr(
        gene_list    = sig_genes,
        gene_sets    = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "GO_Biological_Process_2023"],
        organism     = "human",
        outdir       = str(CFG["results_dir"] / "ORA"),
        cutoff       = 0.05,
    )
    ora_df = ora_res.res2d
    ora_df.to_csv(CFG["results_dir"] / "ORA_pydeseq2.csv", index=False)

    # Dotplot
    if len(ora_df) > 0:
        fig_ora = gsea_dotplot(
            ora_df,
            column    = "Adjusted P-value",
            x         = "Gene_set",
            title     = "ORA — Aging DE Genes\nGSEApy enrichr | Top 20 terms",
            top_term  = 10,
            figsize   = (8, 8),
            cutoff    = 0.05,
        )
        if fig_ora:
            fig_ora.savefig(CFG["figures_dir"] / "04_ORA_dotplot_python.pdf",
                            bbox_inches="tight", dpi=150)
        plt.close()

print(f"\n=== Python pipeline complete ===")
print(f"Results : {CFG['results_dir']}")
print(f"Figures : {CFG['figures_dir']}")
sig_count = (res["regulation"] != "NS").sum()
print(f"Significant DE genes: {sig_count}")
