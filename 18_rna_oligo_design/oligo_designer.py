"""
oligo_designer.py — Oligonucleotide design and thermodynamic scoring.

Tiles ASO / siRNA / gapmer candidates over a target mRNA, scores each
for GC content, nearest-neighbour Tm, self-complementarity, seed off-target
burden, and 5' accessibility, then ranks and reports top candidates.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42): np.random.seed(s)
    PALETTES = {"young": "#2196F3", "aged": "#F44336", "accent": "#4CAF50"}

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

class OligoType(Enum):
    ASO    = "antisense_oligonucleotide"
    SIRNA  = "siRNA"
    GAPMER = "gapmer"


# Nearest-neighbour thermodynamic parameters (SantaLucia 1998, 1M NaCl)
# DNA/DNA ΔH (kcal/mol) and ΔS (cal/mol/K) for each dinucleotide
NN_PARAMS: Dict[str, Tuple[float, float]] = {
    "AA": (-7.9, -22.2), "AT": (-7.2, -20.4), "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7), "GT": (-8.4, -22.4), "CT": (-7.8, -21.0),
    "GA": (-8.2, -22.2), "CG": (-10.6,-27.2), "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9),
    # complements (for antisense Tm calculation)
    "TT": (-7.9, -22.2), "AC": (-8.5, -22.7), "TG": (-8.4, -22.4),
    "AG": (-7.8, -21.0), "TC": (-8.2, -22.2), "CC": (-8.0, -19.9),
}
INIT_DH = 0.2    # initiation ΔH correction kcal/mol
INIT_DS = -5.7   # initiation ΔS correction cal/mol/K
R       = 1.987  # cal/mol/K


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic mRNA target
# ──────────────────────────────────────────────────────────────────────────────

def generate_kras_mrna(length: int = 2000, seed: int = 42) -> str:
    """Synthetic KRAS-like mRNA sequence (realistic GC content ~50%)."""
    rng = np.random.default_rng(seed)
    # GC-biased in coding region (pos 200-1800), AT-rich in UTRs
    bases = []
    for i in range(length):
        if 200 <= i < 1800:
            b = rng.choice(["A", "T", "G", "C"], p=[0.22, 0.22, 0.28, 0.28])
        else:
            b = rng.choice(["A", "T", "G", "C"], p=[0.30, 0.30, 0.20, 0.20])
        bases.append(b)
    return "".join(bases)


def generate_transcriptome(n_transcripts: int = 500, length: int = 500, seed: int = 42) -> List[str]:
    """Synthetic off-target transcriptome for seed-match scoring."""
    rng = np.random.default_rng(seed)
    transcripts = []
    for _ in range(n_transcripts):
        seq = "".join(rng.choice(["A", "T", "G", "C"],
                                  p=[0.25, 0.25, 0.25, 0.25], size=length))
        transcripts.append(seq)
    return transcripts


# ──────────────────────────────────────────────────────────────────────────────
# Thermodynamic scoring
# ──────────────────────────────────────────────────────────────────────────────

COMPLEMENT = str.maketrans("ATGC", "TACG")


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def nearest_neighbour_tm(
    seq: str,
    oligo_conc_M: float = 250e-9,
    salt_conc_M:  float = 0.05,
) -> float:
    """
    Nearest-neighbour Tm (°C) for DNA oligo / RNA target duplex.
    Uses SantaLucia 1998 DNA/DNA parameters as approximation.
    """
    dh = INIT_DH
    ds = INIT_DS
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        if dinuc in NN_PARAMS:
            ddh, dds = NN_PARAMS[dinuc]
            dh += ddh
            ds += dds

    # Salt correction (owczarzy 2004 approximation)
    ln_salt = np.log(salt_conc_M)
    ds_corrected = ds + 0.368 * (len(seq) - 1) * ln_salt

    tm_K = (dh * 1000) / (ds_corrected + R * np.log(oligo_conc_M / 4)) - 273.15
    return float(tm_K)


def self_complementarity_score(seq: str, min_stem: int = 4) -> int:
    """Count maximum stem length in self-complementary folding."""
    rc = reverse_complement(seq)
    max_stem = 0
    for i in range(len(seq) - min_stem):
        for length in range(min_stem, len(seq) // 2 + 1):
            if seq[i:i+length] == rc[i:i+length]:
                max_stem = max(max_stem, length)
    return max_stem


def seed_off_target_score(
    antisense_seq: str,
    transcriptome: List[str],
    seed_start: int = 1,
    seed_end:   int = 7,
) -> int:
    """Count transcriptome sequences with a 6-mer seed match."""
    seed = antisense_seq[seed_start:seed_end + 1]   # positions 2-8 (0-indexed 1-7)
    seed_rc = reverse_complement(seed)
    count = sum(1 for t in transcriptome if seed_rc in t)
    return count


def accessibility_score(mrna: str, pos: int, window: int = 30) -> float:
    """
    Simplified 5' accessibility: low GC in a window around the target site
    correlates with less secondary structure and better accessibility.
    """
    start = max(0, pos - window // 2)
    end   = min(len(mrna), pos + window // 2)
    region = mrna[start:end]
    return 1.0 - gc_content(region)   # higher = more accessible


# ──────────────────────────────────────────────────────────────────────────────
# Oligo dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OligoCandidate:
    oligo_id:        str
    oligo_type:      OligoType
    sequence:        str          # antisense DNA sequence
    target_start:    int
    target_end:      int
    gc_pct:          float
    tm_celsius:      float
    self_comp_score: int
    seed_off_targets: int
    accessibility:   float
    composite_score: float = 0.0
    rank:            int   = 0
    pass_filter:     bool  = True
    fail_reasons:    List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Design filters
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DesignFilters:
    gc_min:         float = 0.35
    gc_max:         float = 0.65
    tm_min:         float = 50.0
    tm_max:         float = 80.0
    max_self_comp:  int   = 4
    max_seed_hits:  int   = 10
    min_accessibility: float = 0.30


# ──────────────────────────────────────────────────────────────────────────────
# Tiling and scoring
# ──────────────────────────────────────────────────────────────────────────────

def tile_and_score(
    mrna:          str,
    transcriptome: List[str],
    oligo_length:  int = 20,
    oligo_type:    OligoType = OligoType.ASO,
    filters:       Optional[DesignFilters] = None,
    step:          int = 1,
    seed:          int = 42,
) -> List[OligoCandidate]:
    if filters is None:
        filters = DesignFilters()

    candidates = []
    for i in range(0, len(mrna) - oligo_length, step):
        target_region = mrna[i:i + oligo_length]
        antisense_seq = reverse_complement(target_region)

        gc    = gc_content(antisense_seq)
        tm    = nearest_neighbour_tm(antisense_seq)
        sc    = self_complementarity_score(antisense_seq)
        seed_hits = seed_off_target_score(antisense_seq, transcriptome)
        acc   = accessibility_score(mrna, i + oligo_length // 2)

        # Composite score (higher = better)
        tm_score    = max(0, 1 - abs(tm - 62) / 15)
        gc_score    = max(0, 1 - abs(gc - 0.50) / 0.20)
        acc_score   = acc
        seed_score  = max(0, 1 - seed_hits / 20)
        sc_score    = max(0, 1 - sc / 6)
        composite   = (0.30 * tm_score + 0.25 * gc_score + 0.20 * acc_score +
                       0.15 * seed_score + 0.10 * sc_score)

        # Filter
        fail_reasons = []
        if not (filters.gc_min <= gc <= filters.gc_max):
            fail_reasons.append(f"GC={gc:.0%} outside [{filters.gc_min:.0%},{filters.gc_max:.0%}]")
        if not (filters.tm_min <= tm <= filters.tm_max):
            fail_reasons.append(f"Tm={tm:.1f}°C outside [{filters.tm_min},{filters.tm_max}]")
        if sc > filters.max_self_comp:
            fail_reasons.append(f"Self-comp={sc} > {filters.max_self_comp}")
        if seed_hits > filters.max_seed_hits:
            fail_reasons.append(f"Seed hits={seed_hits} > {filters.max_seed_hits}")
        if acc < filters.min_accessibility:
            fail_reasons.append(f"Accessibility={acc:.2f} < {filters.min_accessibility}")

        candidates.append(OligoCandidate(
            oligo_id        = f"{oligo_type.value[:3].upper()}{i:05d}",
            oligo_type      = oligo_type,
            sequence        = antisense_seq,
            target_start    = i,
            target_end      = i + oligo_length,
            gc_pct          = round(gc * 100, 1),
            tm_celsius      = round(tm, 1),
            self_comp_score = sc,
            seed_off_targets = seed_hits,
            accessibility   = round(acc, 3),
            composite_score = round(composite, 4),
            pass_filter     = len(fail_reasons) == 0,
            fail_reasons    = fail_reasons,
        ))

    # Rank passing candidates
    passing = [c for c in candidates if c.pass_filter]
    passing.sort(key=lambda c: c.composite_score, reverse=True)
    for rank, c in enumerate(passing, 1):
        c.rank = rank

    return candidates


def candidates_to_df(candidates: List[OligoCandidate]) -> pd.DataFrame:
    return pd.DataFrame([{
        "oligo_id":        c.oligo_id,
        "oligo_type":      c.oligo_type.value,
        "sequence":        c.sequence,
        "target_start":    c.target_start,
        "target_end":      c.target_end,
        "gc_pct":          c.gc_pct,
        "tm_celsius":      c.tm_celsius,
        "self_comp_score": c.self_comp_score,
        "seed_off_targets":c.seed_off_targets,
        "accessibility":   c.accessibility,
        "composite_score": c.composite_score,
        "pass_filter":     c.pass_filter,
        "rank":            c.rank,
    } for c in candidates])


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_candidate_landscape(
    df: pd.DataFrame, out_path: str = "figures/oligo_landscape.png"
) -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#FAFAFA")

    pass_df = df[df["pass_filter"]]
    fail_df = df[~df["pass_filter"]]

    # Panel 1: Tm distribution
    ax = axes[0, 0]
    ax.hist(fail_df["tm_celsius"], bins=30, alpha=0.5, color=PALETTES.get("aged", "#F44336"),
            label="Filtered", edgecolor="white")
    ax.hist(pass_df["tm_celsius"], bins=30, alpha=0.7, color=PALETTES.get("young", "#2196F3"),
            label="Passing", edgecolor="white")
    ax.axvline(55, color="#333", linestyle="--", linewidth=1)
    ax.axvline(75, color="#333", linestyle="--", linewidth=1)
    ax.set_xlabel("Tm (°C)")
    ax.set_ylabel("Count")
    ax.set_title("Melting Temperature Distribution")
    ax.legend()
    ax.set_facecolor("#F5F5F5")

    # Panel 2: GC% vs Tm scatter
    ax = axes[0, 1]
    ax.scatter(fail_df["gc_pct"], fail_df["tm_celsius"],
               alpha=0.2, s=4, color=PALETTES.get("aged", "#F44336"), label="Filtered")
    sc = ax.scatter(pass_df["gc_pct"], pass_df["tm_celsius"],
                    c=pass_df["composite_score"], cmap="viridis",
                    alpha=0.7, s=8, label="Passing")
    plt.colorbar(sc, ax=ax, label="Composite score")
    ax.axvline(35, color="#999", linestyle=":", linewidth=0.8)
    ax.axvline(65, color="#999", linestyle=":", linewidth=0.8)
    ax.set_xlabel("GC (%)")
    ax.set_ylabel("Tm (°C)")
    ax.set_title("GC% vs Melting Temperature")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F5F5F5")

    # Panel 3: Composite score along mRNA
    ax = axes[1, 0]
    ax.scatter(df["target_start"], df["composite_score"],
               c=[PALETTES.get("young", "#2196F3") if p else PALETTES.get("aged", "#F44336")
                  for p in df["pass_filter"]],
               s=3, alpha=0.5)
    top5 = pass_df.nlargest(5, "composite_score")
    for _, row in top5.iterrows():
        ax.annotate(row["oligo_id"], (row["target_start"], row["composite_score"]),
                    fontsize=6, ha="center", va="bottom")
    ax.set_xlabel("Target Position (nt)")
    ax.set_ylabel("Composite Score")
    ax.set_title("Composite Score Along mRNA")
    ax.set_facecolor("#F5F5F5")

    # Panel 4: Failure reasons
    ax = axes[1, 1]
    reason_counts = {
        "GC out of range": int((df["gc_pct"] < 35).sum() + (df["gc_pct"] > 65).sum()),
        "Tm out of range": int((df["tm_celsius"] < 50).sum() + (df["tm_celsius"] > 80).sum()),
        "High self-comp":  int((df["self_comp_score"] > 4).sum()),
        "High seed hits":  int((df["seed_off_targets"] > 10).sum()),
        "Low accessibility": int((df["accessibility"] < 0.30).sum()),
    }
    rc_df = pd.Series(reason_counts).sort_values(ascending=True)
    ax.barh(rc_df.index, rc_df.values, color=PALETTES.get("aged", "#F44336"),
            edgecolor="white")
    ax.set_xlabel("Candidates Affected")
    ax.set_title("Filter Failure Reasons")
    ax.set_facecolor("#F5F5F5")

    plt.suptitle("Oligonucleotide Candidate Landscape", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    print("Generating synthetic KRAS mRNA (2000 nt) …")
    mrna = generate_kras_mrna(length=2000, seed=42)
    print(f"  GC content: {gc_content(mrna):.1%}")

    print("Generating off-target transcriptome (500 × 500 nt) …")
    transcriptome = generate_transcriptome(n_transcripts=500, length=500, seed=42)

    print("Tiling and scoring ASO candidates (20-nt, step=1) …")
    candidates = tile_and_score(
        mrna, transcriptome,
        oligo_length=20, oligo_type=OligoType.ASO,
        step=1, seed=42,
    )

    df = candidates_to_df(candidates)
    n_pass = df["pass_filter"].sum()
    print(f"  Total candidates: {len(df)}")
    print(f"  Passing filters : {n_pass}  ({n_pass/len(df):.1%})")
    print(f"\nTop 10 candidates:")
    top = df[df["pass_filter"]].nsmallest(10, "rank")
    print(top[["oligo_id", "target_start", "gc_pct", "tm_celsius",
               "seed_off_targets", "accessibility", "composite_score"]].to_string(index=False))

    os.makedirs("figures", exist_ok=True)
    img = plot_candidate_landscape(df, "figures/oligo_landscape.png")
    print(f"\nLandscape plot → {img}")
