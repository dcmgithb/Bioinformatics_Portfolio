"""
sequence_analysis.py — RNA secondary structure, motif enrichment, and
target site accessibility analysis for oligonucleotide design (Project 18).
"""

from __future__ import annotations

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42): np.random.seed(s)
    PALETTES = {"young": "#2196F3", "aged": "#F44336", "accent": "#4CAF50"}

from oligo_designer import (
    generate_kras_mrna, generate_transcriptome, gc_content,
    nearest_neighbour_tm, reverse_complement,
    tile_and_score, OligoType, candidates_to_df,
)
from activity_predictor import (
    generate_training_data, FEATURE_COLS,
)

# ──────────────────────────────────────────────────────────────────────────────
# Nussinov RNA secondary structure (dot-bracket notation)
# ──────────────────────────────────────────────────────────────────────────────

RNA_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def nussinov_fold(seq: str, min_loop: int = 3) -> str:
    """
    Nussinov dynamic programming algorithm.
    Returns dot-bracket notation: '.' = unpaired, '(' / ')' = paired.
    """
    n   = len(seq)
    dp  = np.zeros((n, n), dtype=int)

    # Fill DP table (upper triangle)
    for span in range(min_loop + 1, n):
        for i in range(n - span):
            j = i + span
            # Unpaired options
            best = max(dp[i+1, j], dp[i, j-1])
            # Paired
            if (seq[i], seq[j]) in RNA_PAIRS:
                best = max(best, dp[i+1, j-1] + 1)
            # Bifurcation
            for k in range(i + 1, j):
                best = max(best, dp[i, k] + dp[k+1, j])
            dp[i, j] = best

    # Traceback
    structure = ["."] * n

    def traceback(i: int, j: int) -> None:
        if i >= j:
            return
        if dp[i, j] == dp[i+1, j]:
            traceback(i+1, j)
        elif dp[i, j] == dp[i, j-1]:
            traceback(i, j-1)
        elif (seq[i], seq[j]) in RNA_PAIRS and dp[i, j] == dp[i+1, j-1] + 1:
            structure[i] = "("
            structure[j] = ")"
            traceback(i+1, j-1)
        else:
            for k in range(i+1, j):
                if dp[i, j] == dp[i, k] + dp[k+1, j]:
                    traceback(i, k)
                    traceback(k+1, j)
                    return

    traceback(0, n-1)
    return "".join(structure)


def count_base_pairs(structure: str) -> int:
    return structure.count("(")


def mfe_estimate(structure: str, seq: str, gc_kcal: float = -3.0,
                 au_kcal: float = -2.0, gu_kcal: float = -1.0) -> float:
    """Approximate MFE from base pair counts (coarse estimate)."""
    pairs = []
    stack = []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            pairs.append((j, i))

    mfe = 0.0
    for i, j in pairs:
        pair = (seq[i].replace("T", "U"), seq[j].replace("T", "U"))
        if pair in {("G", "C"), ("C", "G")}:
            mfe += gc_kcal
        elif pair in {("A", "U"), ("U", "A")}:
            mfe += au_kcal
        else:
            mfe += gu_kcal
    return mfe


# ──────────────────────────────────────────────────────────────────────────────
# Target site accessibility window
# ──────────────────────────────────────────────────────────────────────────────

def window_accessibility(mrna: str, window: int = 20, step: int = 1) -> np.ndarray:
    """Slide window along mRNA; accessibility = 1 - GC (proxy for open structure)."""
    scores = []
    for i in range(0, len(mrna) - window, step):
        region = mrna[i:i+window]
        scores.append(1.0 - gc_content(region))
    return np.array(scores)


def secondary_structure_accessibility(
    mrna: str, window: int = 50, step: int = 10
) -> np.ndarray:
    """
    Sliding-window Nussinov fold → fraction unpaired per window.
    Computationally limited to short windows for speed.
    """
    fractions = []
    for i in range(0, len(mrna) - window, step):
        region = mrna[i:i+window].replace("T", "U")
        struct  = nussinov_fold(region, min_loop=3)
        frac_unpaired = struct.count(".") / len(struct)
        fractions.append(frac_unpaired)
    return np.array(fractions)


# ──────────────────────────────────────────────────────────────────────────────
# Motif enrichment
# ──────────────────────────────────────────────────────────────────────────────

def kmer_frequencies(sequences: List[str], k: int = 3) -> pd.Series:
    """Compute k-mer frequencies for a list of sequences."""
    counts: Dict[str, List[int]] = {}
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            counts.setdefault(kmer, []).append(1)
    totals = {kmer: sum(v) for kmer, v in counts.items()}
    return pd.Series(totals).sort_values(ascending=False)


def motif_enrichment(
    active_seqs: List[str], inactive_seqs: List[str], k: int = 3
) -> pd.DataFrame:
    """Log-odds enrichment of k-mers in active vs inactive oligos."""
    act_counts = kmer_frequencies(active_seqs, k)
    ina_counts = kmer_frequencies(inactive_seqs, k)

    all_kmers = set(act_counts.index) | set(ina_counts.index)
    act_total = sum(act_counts)
    ina_total = sum(ina_counts)

    rows = []
    for kmer in all_kmers:
        act_f = (act_counts.get(kmer, 0) + 0.5) / (act_total + 0.5)
        ina_f = (ina_counts.get(kmer, 0) + 0.5) / (ina_total + 0.5)
        log_odds = float(np.log2(act_f / ina_f))
        rows.append({"kmer": kmer, "act_freq": act_f, "ina_freq": ina_f,
                     "log_odds": log_odds})

    df = pd.DataFrame(rows).sort_values("log_odds", ascending=False)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Sequence logo (positional base frequency)
# ──────────────────────────────────────────────────────────────────────────────

def compute_pwm(sequences: List[str]) -> pd.DataFrame:
    """Position weight matrix from aligned sequences (all same length)."""
    n    = len(sequences[0])
    pwm  = pd.DataFrame(0.0, index=list("ACGT"), columns=range(n))
    for seq in sequences:
        for i, b in enumerate(seq):
            if b in pwm.index:
                pwm.loc[b, i] += 1
    return pwm / len(sequences)


def plot_sequence_logo(pwm: pd.DataFrame, title: str, ax: plt.Axes) -> None:
    """Simplified sequence logo: bar stacked by base proportion per position."""
    colors_map = {"A": "#4CAF50", "C": "#2196F3", "G": "#FF9800", "T": "#F44336"}
    bottom = np.zeros(pwm.shape[1])
    for base in ["A", "C", "G", "T"]:
        heights = pwm.loc[base].values
        ax.bar(range(pwm.shape[1]), heights, bottom=bottom,
               color=colors_map[base], label=base, edgecolor="none", width=0.95)
        bottom += heights
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.set_ylim(0, 1.05)


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_sequence_analysis(out_dir: str = "figures") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Example secondary structure
    print("── Nussinov secondary structure (example 40-nt) ──")
    example = "GGCAUUGCAGUAGCGCAUAGCUAGCUGCAUAGCUGCAUGCUA"
    struct   = nussinov_fold(example)
    mfe      = mfe_estimate(struct, example)
    pairs    = count_base_pairs(struct)
    print(f"  Seq : {example}")
    print(f"  Fold: {struct}")
    print(f"  Base pairs: {pairs}  MFE estimate: {mfe:.1f} kcal/mol")

    # 2. Target site accessibility along KRAS mRNA
    mrna = generate_kras_mrna(length=2000, seed=42)
    gc_acc  = window_accessibility(mrna, window=20, step=1)
    ss_acc  = secondary_structure_accessibility(mrna, window=50, step=10)

    # 3. Get training data for motif analysis
    df_train = generate_training_data(n=400, seed=42)
    active_seqs   = df_train[df_train["active"] == 1]["sequence"].tolist()
    inactive_seqs = df_train[df_train["active"] == 0]["sequence"].tolist()
    enrichment    = motif_enrichment(active_seqs, inactive_seqs, k=3)

    print(f"\nTop enriched 3-mers in active oligos:")
    print(enrichment.head(5)[["kmer", "log_odds"]].to_string(index=False))
    print(f"\nTop depleted 3-mers in active oligos:")
    print(enrichment.tail(5)[["kmer", "log_odds"]].to_string(index=False))

    # 4. PWM for active vs inactive
    min_len = min(min(len(s) for s in active_seqs), min(len(s) for s in inactive_seqs))
    active_trimmed   = [s[:min_len] for s in active_seqs[:100]]
    inactive_trimmed = [s[:min_len] for s in inactive_seqs[:100]]
    pwm_active   = compute_pwm(active_trimmed)
    pwm_inactive = compute_pwm(inactive_trimmed)

    # ── Plots ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#FAFAFA")

    # Panel 1: GC-based accessibility along mRNA
    ax1 = fig.add_subplot(3, 2, 1)
    positions = np.arange(len(gc_acc))
    ax1.plot(positions, gc_acc, color=PALETTES.get("young", "#2196F3"),
             linewidth=0.6, alpha=0.8)
    ax1.fill_between(positions, gc_acc, alpha=0.15,
                     color=PALETTES.get("young", "#2196F3"))
    ax1.axhline(0.4, color="#333", linestyle="--", linewidth=0.8, label="Threshold 0.40")
    ax1.set_xlabel("Target Position (nt)")
    ax1.set_ylabel("Accessibility Score")
    ax1.set_title("Target Site Accessibility (GC-based)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_facecolor("#F5F5F5")

    # Panel 2: Secondary structure accessibility (sliding Nussinov)
    ax2 = fig.add_subplot(3, 2, 2)
    ss_positions = np.arange(len(ss_acc)) * 10
    ax2.plot(ss_positions, ss_acc, color=PALETTES.get("accent", "#4CAF50"),
             linewidth=1.5, marker="o", markersize=3)
    ax2.axhline(0.60, color="#333", linestyle="--", linewidth=0.8,
                label="≥60% unpaired (accessible)")
    ax2.set_xlabel("Window Start (nt)")
    ax2.set_ylabel("Fraction Unpaired")
    ax2.set_title("Nussinov Structure Accessibility", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.set_facecolor("#F5F5F5")

    # Panel 3: Tm distribution (active vs inactive)
    trs = generate_transcriptome(n_transcripts=300, length=400, seed=42)
    candidates = tile_and_score(mrna, trs, step=2, seed=42)
    cand_df    = candidates_to_df(candidates)
    merged     = cand_df.merge(
        df_train[["sequence", "active", "knockdown_pct"]], on="sequence", how="inner"
    )
    ax3 = fig.add_subplot(3, 2, 3)
    for act_val, label, color in [(1, "Active (≥70% KD)", PALETTES.get("young", "#2196F3")),
                                   (0, "Inactive",         PALETTES.get("aged", "#F44336"))]:
        sub = merged[merged["active"] == act_val]
        if not sub.empty:
            ax3.hist(sub["tm_celsius"], bins=20, alpha=0.6,
                     color=color, label=f"{label} (n={len(sub)})", edgecolor="white")
    ax3.set_xlabel("Melting Temperature (°C)")
    ax3.set_ylabel("Count")
    ax3.set_title("Tm Distribution: Active vs Inactive", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.set_facecolor("#F5F5F5")

    # Panel 4: Motif enrichment (log-odds)
    ax4 = fig.add_subplot(3, 2, 4)
    top_enrich = pd.concat([enrichment.head(8), enrichment.tail(8)])
    colors_enrich = [PALETTES.get("young", "#2196F3") if v >= 0
                     else PALETTES.get("aged", "#F44336")
                     for v in top_enrich["log_odds"]]
    ax4.barh(top_enrich["kmer"], top_enrich["log_odds"],
             color=colors_enrich, edgecolor="white")
    ax4.axvline(0, color="#333", linewidth=0.8)
    ax4.set_xlabel("Log₂ Odds (active / inactive)")
    ax4.set_title("3-mer Motif Enrichment in Active Oligos", fontweight="bold")
    ax4.set_facecolor("#F5F5F5")

    # Panel 5: Sequence logo — active oligos
    ax5 = fig.add_subplot(3, 2, 5)
    plot_sequence_logo(pwm_active.iloc[:, :20], "Sequence Logo — Active Oligos (pos 1-20)", ax5)
    ax5.set_facecolor("#F5F5F5")

    # Panel 6: Sequence logo — inactive oligos
    ax6 = fig.add_subplot(3, 2, 6)
    plot_sequence_logo(pwm_inactive.iloc[:, :20], "Sequence Logo — Inactive Oligos (pos 1-20)", ax6)
    ax6.set_facecolor("#F5F5F5")

    plt.suptitle("RNA Oligonucleotide Sequence Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "sequence_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nPlot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)
    os.makedirs("figures", exist_ok=True)
    run_sequence_analysis(out_dir="figures")
