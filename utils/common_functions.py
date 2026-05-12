"""
Common Bioinformatics Utility Functions (Python)
=================================================
Shared across all portfolio projects.
Mirrors the R utilities in common_functions.R.

Python : >= 3.10
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import random
from pathlib import Path
from typing import Any, Optional, Union
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Colour Palettes ───────────────────────────────────────────────────────────

PALETTES: dict[str, dict | list] = {
    "age_group"   : {"young": "#2196F3", "aged": "#F44336"},
    "age_3groups" : {"young": "#4CAF50", "middle": "#FF9800", "aged": "#F44336"},
    "regulation"  : {"Up in Aged": "#F44336", "Down in Aged": "#2196F3", "NS": "#BDBDBD"},
    "sex"         : {"M": "#9C27B0", "F": "#FF9800"},
    "species"     : {"human": "#1565C0", "mouse": "#2E7D32"},
    "diverging"   : ["#2196F3", "white", "#F44336"],
}

def make_colormap(colours: list[str], name: str = "custom") -> matplotlib.colors.LinearSegmentedColormap:
    """Create a matplotlib colormap from a list of hex colours."""
    return matplotlib.colors.LinearSegmentedColormap.from_list(name, colours)

CMAP_AGING = make_colormap(PALETTES["diverging"])

# ── Statistical Utilities ─────────────────────────────────────────────────────

def add_multiple_testing(
    df: pd.DataFrame,
    p_col: str = "pvalue",
    methods: list[str] = ("BH", "bonferroni"),
) -> pd.DataFrame:
    """Add FDR and Bonferroni corrected p-values as new columns."""
    from statsmodels.stats.multitest import multipletests

    df = df.copy()
    for method in methods:
        reject, corrected, _, _ = multipletests(
            df[p_col].fillna(1.0), method=method
        )
        col = f"p_{method}"
        df[col] = corrected
    return df


def classify_de_genes(
    df: pd.DataFrame,
    lfc_col: str = "log2FoldChange",
    padj_col: str = "padj",
    lfc_thresh: float = 1.0,
    padj_thresh: float = 0.05,
) -> pd.DataFrame:
    """Classify genes into Up/Down/NS based on LFC and adjusted p-value."""
    df = df.copy()
    df["regulation"] = "NS"
    df.loc[
        (df[padj_col] < padj_thresh) & (df[lfc_col] >  lfc_thresh), "regulation"
    ] = "Up in Aged"
    df.loc[
        (df[padj_col] < padj_thresh) & (df[lfc_col] < -lfc_thresh), "regulation"
    ] = "Down in Aged"
    return df


def quick_stats(x: np.ndarray | pd.Series, label: str = "x") -> pd.DataFrame:
    """Compute descriptive statistics for a numeric array."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return pd.DataFrame([{
        "variable": label,
        "n"       : len(x),
        "mean"    : np.mean(x),
        "sd"      : np.std(x, ddof=1),
        "median"  : np.median(x),
        "iqr"     : stats.iqr(x),
        "min"     : np.min(x),
        "max"     : np.max(x),
    }])


def significance_stars(pval: float) -> str:
    """Convert p-value to significance stars."""
    if pval < 0.001: return "***"
    if pval < 0.01:  return "**"
    if pval < 0.05:  return "*"
    return "ns"

# ── Count Data Utilities ──────────────────────────────────────────────────────

def filter_low_counts(
    counts: pd.DataFrame,
    min_count: int = 10,
    min_samples_frac: float = 0.25,
) -> pd.DataFrame:
    """Remove genes with insufficient counts across samples."""
    min_samples = int(np.ceil(counts.shape[0] * min_samples_frac))
    keep        = (counts >= min_count).sum(axis=0) >= min_samples
    n_before    = counts.shape[1]
    result      = counts.loc[:, keep]
    log.info(f"Filtering: {n_before} → {result.shape[1]} genes "
             f"({100*result.shape[1]/n_before:.1f}% retained)")
    return result


def calc_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """Calculate counts per million (CPM)."""
    lib_sizes = counts.sum(axis=1)
    return counts.div(lib_sizes, axis=0) * 1e6


def calc_tpm(counts: pd.DataFrame, gene_lengths: pd.Series) -> pd.DataFrame:
    """Calculate transcripts per million (TPM) given gene lengths in bp."""
    if len(gene_lengths) != counts.shape[1]:
        raise ValueError("gene_lengths must match number of columns in counts")
    rpk = counts.div(gene_lengths / 1000, axis=1)
    return rpk.div(rpk.sum(axis=1), axis=0) * 1e6


def beta_to_mvalue(beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert DNA methylation beta values to M-values (logit scale)."""
    beta_clipped = np.clip(beta, eps, 1 - eps)
    return np.log2(beta_clipped / (1 - beta_clipped))


def mvalue_to_beta(mvalue: np.ndarray) -> np.ndarray:
    """Convert M-values back to beta values."""
    return 2**mvalue / (2**mvalue + 1)

# ── Visualisation Helpers ─────────────────────────────────────────────────────

def volcano_plot(
    df: pd.DataFrame,
    lfc_col: str = "log2FoldChange",
    padj_col: str = "padj",
    label_col: Optional[str] = "gene_id",
    lfc_thresh: float = 1.0,
    padj_thresh: float = 0.05,
    title: str = "Volcano Plot",
    n_label: int = 15,
    figsize: tuple = (9, 7),
) -> plt.Figure:
    """Publication-ready volcano plot."""
    df = classify_de_genes(df, lfc_col, padj_col, lfc_thresh, padj_thresh)
    df["-log10_padj"] = -np.log10(df[padj_col].clip(lower=1e-300))

    colour_map = {
        "Up in Aged"   : PALETTES["regulation"]["Up in Aged"],
        "Down in Aged" : PALETTES["regulation"]["Down in Aged"],
        "NS"           : PALETTES["regulation"]["NS"],
    }

    fig, ax = plt.subplots(figsize=figsize)

    for reg, colour in colour_map.items():
        sub = df[df["regulation"] == reg]
        ax.scatter(sub[lfc_col], sub["-log10_padj"],
                   c=colour, alpha=0.5, s=12, label=reg, linewidths=0)

    ax.axhline(-np.log10(padj_thresh), ls="--", lw=1, color="grey60")
    ax.axvline( lfc_thresh,            ls="--", lw=1, color="grey60")
    ax.axvline(-lfc_thresh,            ls="--", lw=1, color="grey60")

    # Label top genes
    if label_col and label_col in df.columns:
        top_genes = df.nsmallest(n_label, padj_col)
        for _, row in top_genes.iterrows():
            ax.annotate(
                row[label_col],
                xy=(row[lfc_col], row["-log10_padj"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, alpha=0.9,
            )

    ax.set_xlabel(f"log₂ Fold Change", fontsize=12)
    ax.set_ylabel("-log₁₀(FDR-adjusted p)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(markerscale=2, fontsize=9, loc="upper left")
    plt.tight_layout()
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    title: str = "Correlation Heatmap",
    figsize: tuple = (8, 7),
    annot: bool = True,
) -> plt.Figure:
    """Styled correlation heatmap."""
    corr  = df.corr(method=method)
    mask  = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle

    fig, ax = plt.subplots(figsize=figsize)
    cmap = make_colormap(["#2196F3", "white", "#F44336"])
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap=cmap, center=0, vmin=-1, vmax=1,
        annot=annot, fmt=".2f", annot_kws={"size": 8},
        square=True, linewidths=0.5,
    )
    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    return fig

# ── Reproducibility Utilities ─────────────────────────────────────────────────

def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    log.info(f"Global seed set: {seed}")


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute MD5 hash of a DataFrame for reproducibility checks."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def save_analysis_metadata(
    params: dict[str, Any],
    outpath: Path | str,
    prefix: str = "run",
) -> None:
    """Save analysis parameters and environment info as JSON."""
    import sys
    import platform

    metadata = {
        "params"    : params,
        "python"    : sys.version,
        "platform"  : platform.platform(),
        "packages"  : {
            pkg: __import__(pkg).__version__
            for pkg in ["numpy","pandas","scipy","sklearn","scanpy","anndata"]
            if pkg in sys.modules or _can_import(pkg)
        },
    }
    outpath = Path(outpath) / f"{prefix}_metadata.json"
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f"Analysis metadata saved: {outpath}")


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# ── SRA Utilities ────────────────────────────────────────────────────────────

def build_sra_download_script(
    srr_list: list[str],
    outdir: str = "data/raw_reads",
    n_parallel: int = 4,
    prefetch_max_gb: int = 50,
) -> str:
    """Generate a shell script for parallel SRA download and FASTQ conversion."""
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated SRA download script",
        f"set -euo pipefail",
        f'OUTDIR="{outdir}"',
        f"mkdir -p \"$OUTDIR\" logs",
        "",
        "# Parallel download function",
        "download_srr() {",
        "    local SRR=$1",
        "    echo \"Downloading $SRR...\"",
        f"    prefetch --max-size {prefetch_max_gb}G --output-directory data/sra \"$SRR\"",
        "    parallel-fastq-dump \\",
        "        --sra-id \"$SRR\" \\",
        "        --threads 4 \\",
        "        --split-files \\",
        "        --gzip \\",
        f"        --outdir \"$OUTDIR\" \\",
        "        2>> \"logs/${SRR}.log\"",
        "    echo \"$SRR done\"",
        "}",
        "export -f download_srr",
        "",
        "# Run downloads in parallel",
        f"printf '%s\\n' \\ ",
    ] + [f"    {srr} \\" for srr in srr_list] + [
        f"    | parallel -j {n_parallel} download_srr {{}}",
        "",
        "echo \"All downloads complete: $OUTDIR\"",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    set_global_seed(42)
    print("Common functions available:")
    print("  Statistics  : classify_de_genes, add_multiple_testing, quick_stats")
    print("  Count data  : filter_low_counts, calc_cpm, calc_tpm")
    print("  Methylation : beta_to_mvalue, mvalue_to_beta")
    print("  Plots       : volcano_plot, correlation_heatmap")
    print("  Reproducib. : set_global_seed, hash_dataframe, save_analysis_metadata")
    print("  SRA         : build_sra_download_script")
