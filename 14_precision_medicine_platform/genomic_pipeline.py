"""
genomic_pipeline.py — Genomic Variant Processing & GWAS Pipeline
=================================================================
Demonstrates end-to-end variant data handling:
  • Synthetic VCF-like genotype matrix generation
  • Variant QC: call rate, MAF, Hardy-Weinberg equilibrium
  • Population stratification via PCA
  • GWAS: logistic regression association testing per variant
  • Manhattan plot + QQ plot visualisation

All data is synthetic with realistic allele frequency distributions.
No real patient data is used.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. SYNTHETIC GENOTYPE DATA GENERATION
# ---------------------------------------------------------------------------

# Chromosome sizes (number of SNPs per chromosome, simplified)
CHROM_SNP_COUNTS = {
    "1": 120, "2": 110, "3": 95, "4": 85, "5": 80,
    "6": 75,  "7": 70,  "8": 65, "9": 60, "10": 55,
    "11": 50, "12": 50, "13": 40, "14": 35, "15": 30,
    "16": 30, "17": 25, "18": 20, "19": 15, "20": 15,
    "21": 10, "22": 10,
}
TOTAL_SNPS = sum(CHROM_SNP_COUNTS.values())  # ~1000


def generate_variant_manifest(seed: int = 42) -> pd.DataFrame:
    """
    Generate a SNP manifest with realistic allele frequencies.

    Returns DataFrame with columns:
      variant_id, chrom, pos, ref, alt, maf_population, functional_class
    """
    rng = np.random.default_rng(seed)
    rows = []
    bases = ["A", "C", "G", "T"]

    for chrom, n_snps in CHROM_SNP_COUNTS.items():
        # Positions spread across a 100 Mb chromosome
        positions = sorted(rng.integers(1, 100_000_000, size=n_snps))
        for i, pos in enumerate(positions):
            ref = rng.choice(bases)
            alt = rng.choice([b for b in bases if b != ref])
            # MAF from beta distribution — most SNPs are rare
            maf = float(np.clip(rng.beta(0.5, 5), 0.01, 0.49))
            func_class = rng.choice(
                ["intergenic", "intronic", "synonymous", "missense", "UTR3", "UTR5"],
                p=[0.45, 0.30, 0.10, 0.08, 0.04, 0.03],
            )
            rows.append({
                "variant_id": f"chr{chrom}:{pos}:{ref}:{alt}",
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "maf_population": round(maf, 4),
                "functional_class": func_class,
            })

    return pd.DataFrame(rows)


def generate_genotype_matrix(
    manifest: pd.DataFrame,
    n_samples: int = 500,
    n_causal: int = 10,
    heritability: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.Series, List[str]]:
    """
    Generate a synthetic genotype matrix under an additive liability model.

    Parameters
    ----------
    manifest    : variant manifest from generate_variant_manifest()
    n_samples   : number of individuals
    n_causal    : number of causal SNPs for the binary phenotype
    heritability: proportion of variance explained by genetics

    Returns
    -------
    G        : (n_samples, n_snps) int8 array  0/1/2 dosage
    phenotype: binary Series (0=control, 1=case)
    causal_ids: list of causal variant_ids
    """
    rng = np.random.default_rng(seed)
    n_snps = len(manifest)

    # Sample genotypes under HWE
    mafs = manifest["maf_population"].values
    # P(AA) = (1-maf)^2, P(Aa) = 2*maf*(1-maf), P(aa) = maf^2
    p_hom_ref = (1 - mafs) ** 2
    p_het = 2 * mafs * (1 - mafs)
    # p_hom_alt = mafs ** 2

    G = np.zeros((n_samples, n_snps), dtype=np.int8)
    for j in range(n_snps):
        probs = [p_hom_ref[j], p_het[j], mafs[j] ** 2]
        probs = np.array(probs)
        probs /= probs.sum()
        G[:, j] = rng.choice([0, 1, 2], size=n_samples, p=probs)

    # Select causal SNPs (prefer common, missense/synonymous)
    eligible = manifest[manifest["maf_population"] > 0.05].index.tolist()
    causal_idx = rng.choice(eligible, size=min(n_causal, len(eligible)), replace=False)
    causal_ids = manifest.loc[causal_idx, "variant_id"].tolist()

    # Build polygenic score
    effect_sizes = rng.normal(0, 1, size=len(causal_idx))
    G_causal = G[:, causal_idx].astype(float)
    # Standardise each causal SNP
    G_std = (G_causal - G_causal.mean(0)) / (G_causal.std(0) + 1e-8)
    pgs = G_std @ effect_sizes

    # Add environmental noise
    sigma_e = np.sqrt((1 - heritability) / heritability) * pgs.std()
    liability = pgs + rng.normal(0, sigma_e, n_samples)

    # Binary phenotype: top 30% are cases
    threshold = np.percentile(liability, 70)
    phenotype = pd.Series((liability >= threshold).astype(int),
                           name="phenotype")

    return G, phenotype, causal_ids


# ---------------------------------------------------------------------------
# 2. VARIANT QUALITY CONTROL
# ---------------------------------------------------------------------------

def compute_variant_qc(
    G: np.ndarray,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-variant QC metrics.

    Returns manifest augmented with:
      call_rate, obs_maf, hwe_p, missing_rate, qc_pass
    """
    n_samples = G.shape[0]

    call_rate = (G >= 0).mean(axis=0)
    obs_maf = G.sum(axis=0) / (2 * n_samples)
    obs_maf = np.minimum(obs_maf, 1 - obs_maf)

    # Hardy-Weinberg exact test (chi-squared approximation)
    hwe_p = np.ones(G.shape[1])
    for j in range(G.shape[1]):
        n_aa = (G[:, j] == 0).sum()
        n_Aa = (G[:, j] == 1).sum()
        n_AA = (G[:, j] == 2).sum()
        n = n_aa + n_Aa + n_AA
        if n == 0:
            continue
        p = (2 * n_AA + n_Aa) / (2 * n)
        q = 1 - p
        exp_aa = n * q ** 2
        exp_Aa = n * 2 * p * q
        exp_AA = n * p ** 2
        obs = np.array([n_aa, n_Aa, n_AA], dtype=float)
        exp = np.array([exp_aa, exp_Aa, exp_AA], dtype=float)
        # Chi-squared with 1 df (HWE)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum((obs - exp) ** 2 / np.where(exp > 0, exp, np.inf))
        hwe_p[j] = float(stats.chi2.sf(chi2, df=1))

    qc_pass = (
        (call_rate >= 0.95) &
        (obs_maf >= 0.01) &
        (hwe_p >= 1e-6)
    )

    qc_df = manifest.copy()
    qc_df["call_rate"] = call_rate
    qc_df["obs_maf"] = obs_maf.round(4)
    qc_df["hwe_p"] = hwe_p
    qc_df["qc_pass"] = qc_pass

    return qc_df


# ---------------------------------------------------------------------------
# 3. POPULATION STRATIFICATION (PCA)
# ---------------------------------------------------------------------------

def compute_population_pcs(
    G: np.ndarray,
    qc_df: pd.DataFrame,
    n_components: int = 10,
) -> Tuple[np.ndarray, PCA]:
    """
    Compute principal components from QC-passed variants.

    Returns
    -------
    pcs  : (n_samples, n_components) array
    pca  : fitted PCA object
    """
    pass_idx = qc_df["qc_pass"].values
    G_pass = G[:, pass_idx].astype(float)

    # Mean-centre by 2*maf (standard genomic PCA)
    mafs = qc_df.loc[pass_idx, "obs_maf"].values
    G_centred = G_pass - 2 * mafs[np.newaxis, :]

    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(G_centred)

    return pcs, pca


# ---------------------------------------------------------------------------
# 4. GWAS — LOGISTIC REGRESSION
# ---------------------------------------------------------------------------

def run_gwas(
    G: np.ndarray,
    phenotype: pd.Series,
    qc_df: pd.DataFrame,
    pcs: np.ndarray,
    n_pcs_covariate: int = 3,
) -> pd.DataFrame:
    """
    Run per-variant logistic regression GWAS with PC covariates.

    Returns qc_df augmented with: beta, se, p_value, neg_log10_p
    """
    y = phenotype.values
    covariates = pcs[:, :n_pcs_covariate]   # first 3 PCs as covariates
    scaler = StandardScaler()
    cov_scaled = scaler.fit_transform(covariates)

    pass_mask = qc_df["qc_pass"].values
    n_pass = pass_mask.sum()
    betas = np.zeros(len(qc_df))
    ses = np.ones(len(qc_df))
    pvals = np.ones(len(qc_df))

    pass_indices = np.where(pass_mask)[0]

    for i, j in enumerate(pass_indices):
        x = G[:, j].astype(float).reshape(-1, 1)
        X = np.hstack([x, cov_scaled])
        try:
            model = LogisticRegression(max_iter=200, solver="lbfgs",
                                       random_state=42, C=1e6)
            model.fit(X, y)
            beta = model.coef_[0, 0]

            # Wald test: SE from Fisher information (approximate)
            p_hat = model.predict_proba(X)[:, 1]
            W = np.diag(p_hat * (1 - p_hat))
            XtWX = X.T @ W @ X
            try:
                cov_mat = np.linalg.inv(XtWX)
                se = np.sqrt(abs(cov_mat[0, 0]))
            except np.linalg.LinAlgError:
                se = 1.0
            z = beta / (se + 1e-10)
            pval = float(2 * stats.norm.sf(abs(z)))

            betas[j] = beta
            ses[j] = se
            pvals[j] = max(pval, 1e-300)
        except Exception:
            pass

    gwas_df = qc_df.copy()
    gwas_df["beta"] = betas
    gwas_df["se"] = ses
    gwas_df["p_value"] = pvals
    gwas_df["neg_log10_p"] = -np.log10(np.clip(pvals, 1e-300, 1))

    return gwas_df


# ---------------------------------------------------------------------------
# 5. VARIANT ANNOTATION
# ---------------------------------------------------------------------------

# Functional impact scores (CADD-like proxy)
FUNCTIONAL_IMPACT = {
    "missense":    0.85,
    "synonymous":  0.15,
    "UTR3":        0.25,
    "UTR5":        0.30,
    "intronic":    0.05,
    "intergenic":  0.02,
}


def annotate_variants(gwas_df: pd.DataFrame) -> pd.DataFrame:
    """Add functional impact score and combined prioritisation score."""
    df = gwas_df.copy()
    df["functional_impact"] = df["functional_class"].map(FUNCTIONAL_IMPACT).fillna(0.05)

    # Combined score: -log10(p) × functional_impact × MAF_weight
    maf_weight = 1 - df["obs_maf"]   # rarer = higher weight
    df["priority_score"] = (
        df["neg_log10_p"] * df["functional_impact"] * maf_weight
    )
    df["priority_score"] = df["priority_score"].round(4)

    return df


# ---------------------------------------------------------------------------
# 6. VISUALISATION
# ---------------------------------------------------------------------------

def plot_manhattan(
    gwas_df: pd.DataFrame,
    out_path: str = "manhattan_plot.png",
    significance_threshold: float = 5e-8,
) -> None:
    """Manhattan plot with genome-wide significance line."""
    df = gwas_df[gwas_df["qc_pass"]].copy()

    chrom_order = [str(i) for i in range(1, 23)]
    chrom_colors = ["#1976d2", "#e53935"]

    fig, ax = plt.subplots(figsize=(18, 6))
    x_offset = 0
    x_ticks, x_labels = [], []
    chrom_offsets = {}

    for i, chrom in enumerate(chrom_order):
        sub = df[df["chrom"] == chrom].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("pos")
        xs = sub["pos"].values + x_offset
        ys = sub["neg_log10_p"].values
        color = chrom_colors[i % 2]
        ax.scatter(xs, ys, c=color, s=6, alpha=0.7, linewidths=0)

        x_ticks.append(x_offset + sub["pos"].median())
        x_labels.append(chrom)
        chrom_offsets[chrom] = x_offset
        x_offset += sub["pos"].max() + 5_000_000

    # Significance lines
    sig_y = -np.log10(significance_threshold)
    suggestive_y = -np.log10(1e-5)
    ax.axhline(sig_y, color="red", lw=1.2, linestyle="--",
               label=f"Genome-wide sig (p={significance_threshold:.0e})")
    ax.axhline(suggestive_y, color="orange", lw=0.8, linestyle=":",
               label="Suggestive (p=1e-5)")

    # Label top hits
    top = df.nlargest(5, "neg_log10_p")
    for _, row in top.iterrows():
        x = row["pos"] + chrom_offsets.get(row["chrom"], 0)
        ax.annotate(
            row["variant_id"].split(":")[0] + ":" + str(row["pos"]),
            xy=(x, row["neg_log10_p"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=6, color="black",
        )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log₁₀(p)")
    ax.set_title("GWAS Manhattan Plot", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(df["neg_log10_p"].max() * 1.1, sig_y * 1.2))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Manhattan plot → {out_path}")


def plot_qq(
    gwas_df: pd.DataFrame,
    out_path: str = "qq_plot.png",
) -> None:
    """QQ plot of observed vs expected -log10(p) with genomic inflation λ."""
    df = gwas_df[gwas_df["qc_pass"] & (gwas_df["p_value"] < 1)].copy()
    obs = np.sort(df["neg_log10_p"].values)[::-1]
    n = len(obs)
    expected = -np.log10(np.arange(1, n + 1) / n)

    # Genomic inflation factor
    chi2_obs = stats.chi2.isf(df["p_value"].values, df=1)
    lambda_gc = np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected, obs, s=6, alpha=0.6, color="#1976d2")
    max_val = max(obs.max(), expected.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", lw=1.2, label="Expected")
    ax.set_xlabel("Expected -log₁₀(p)")
    ax.set_ylabel("Observed -log₁₀(p)")
    ax.set_title(f"QQ Plot  (λ_GC = {lambda_gc:.3f})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  QQ plot → {out_path}")


def plot_pca_stratification(
    pcs: np.ndarray,
    phenotype: pd.Series,
    out_path: str = "pca_stratification.png",
) -> None:
    """Scatter plot of PC1 vs PC2 coloured by phenotype."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {0: "#1976d2", 1: "#e53935"}
    labels = {0: "Control", 1: "Case"}

    for val in [0, 1]:
        mask = phenotype.values == val
        axes[0].scatter(pcs[mask, 0], pcs[mask, 1],
                        c=colors[val], s=15, alpha=0.6, label=labels[val])
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Population Stratification (PC1 vs PC2)", fontweight="bold")
    axes[0].legend(fontsize=9)

    axes[1].scatter(pcs[:, 2], pcs[:, 3],
                    c=[colors[v] for v in phenotype.values],
                    s=15, alpha=0.6)
    axes[1].set_xlabel("PC3")
    axes[1].set_ylabel("PC4")
    axes[1].set_title("PC3 vs PC4", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA stratification plot → {out_path}")


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(__file__) or "."

    print("=" * 64)
    print("  GENOMIC VARIANT PROCESSING & GWAS PIPELINE")
    print("=" * 64)

    # ── Generate data ────────────────────────────────────────────────
    print("\n[1/5] Generating variant manifest and synthetic cohort…")
    manifest = generate_variant_manifest(seed=42)
    print(f"  Variants  : {len(manifest):,}")
    print(f"  Chromosomes: {manifest['chrom'].nunique()}")

    G, phenotype, causal_ids = generate_genotype_matrix(
        manifest, n_samples=500, n_causal=10, seed=42
    )
    print(f"  Samples   : {G.shape[0]}")
    print(f"  Cases     : {phenotype.sum()}  Controls: {(phenotype==0).sum()}")
    print(f"  Causal SNPs (ground truth): {len(causal_ids)}")

    # ── Variant QC ───────────────────────────────────────────────────
    print("\n[2/5] Variant QC…")
    qc_df = compute_variant_qc(G, manifest)
    n_pass = qc_df["qc_pass"].sum()
    print(f"  Pass QC   : {n_pass:,} / {len(qc_df):,} variants")
    print(f"  Fail MAF  : {(qc_df['obs_maf'] < 0.01).sum()}")
    print(f"  Fail HWE  : {(qc_df['hwe_p'] < 1e-6).sum()}")
    print(f"  Fail call : {(qc_df['call_rate'] < 0.95).sum()}")

    # ── Population stratification ────────────────────────────────────
    print("\n[3/5] Population stratification (PCA)…")
    pcs, pca = compute_population_pcs(G, qc_df, n_components=10)
    evr = pca.explained_variance_ratio_
    print(f"  PC1: {evr[0]:.2%}  PC2: {evr[1]:.2%}  PC3: {evr[2]:.2%}")

    # ── GWAS ─────────────────────────────────────────────────────────
    print("\n[4/5] Running GWAS (logistic regression + PC covariates)…")
    gwas_df = run_gwas(G, phenotype, qc_df, pcs, n_pcs_covariate=3)
    gwas_df = annotate_variants(gwas_df)

    sig_hits = gwas_df[gwas_df["p_value"] < 5e-8]
    suggestive = gwas_df[(gwas_df["p_value"] < 1e-5) & (gwas_df["p_value"] >= 5e-8)]
    print(f"  Genome-wide significant hits : {len(sig_hits)}")
    print(f"  Suggestive hits (p<1e-5)     : {len(suggestive)}")

    if not sig_hits.empty:
        top = sig_hits.nsmallest(5, "p_value")[
            ["variant_id", "chrom", "pos", "p_value", "beta",
             "obs_maf", "functional_class", "priority_score"]
        ]
        print("\n  Top significant variants:")
        print(top.to_string(index=False))

    # Check overlap with causal SNPs
    hit_ids = set(sig_hits["variant_id"].tolist() +
                  suggestive["variant_id"].tolist())
    recovered = len(set(causal_ids) & hit_ids)
    print(f"\n  Causal SNPs recovered: {recovered}/{len(causal_ids)}")

    # ── Visualisations ───────────────────────────────────────────────
    print("\n[5/5] Generating plots…")
    plot_manhattan(gwas_df, out_path=os.path.join(out_dir, "manhattan_plot.png"))
    plot_qq(gwas_df, out_path=os.path.join(out_dir, "qq_plot.png"))
    plot_pca_stratification(pcs, phenotype,
                            out_path=os.path.join(out_dir, "pca_stratification.png"))

    print("\nDone.")
