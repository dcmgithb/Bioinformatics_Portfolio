"""
Antibody Fitness Landscape Scanning & Directed Evolution
==========================================================
Maps the mutational fitness landscape of antibody CDR loops and
simulates directed evolution campaigns to identify improved variants.

Methods
-------
- Single-point mutation scanning: exhaustive testing of all 19 AA
  substitutions at each position
- Greedy combinatorial search: forward selection of additive mutations
- Genetic algorithm: population-based directed evolution simulation
- Fitness landscape visualisation: heatmap + trajectory plots

Python  : >= 3.10
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ── Single-point mutation scanning ───────────────────────────────────────────

def scan_single_point_mutations(
    wt_sequence: str,
    scoring_fn: Callable[[str], float],
    positions: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Exhaustively test all 19 AA substitutions at each position.

    Parameters
    ----------
    wt_sequence : wild-type amino acid sequence
    scoring_fn  : function str → float (higher = better fitness)
    positions   : which positions to scan (default: all)

    Returns
    -------
    pd.DataFrame with columns:
        position, wt_aa, mut_aa, fitness, delta_fitness
    sorted by delta_fitness descending.
    """
    if positions is None:
        positions = list(range(len(wt_sequence)))

    wt_fitness = scoring_fn(wt_sequence)
    rows = []

    for pos in positions:
        wt_aa = wt_sequence[pos]
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue
            mutant = wt_sequence[:pos] + mut_aa + wt_sequence[pos + 1:]
            fitness = scoring_fn(mutant)
            rows.append({
                "position":      pos,
                "wt_aa":         wt_aa,
                "mut_aa":        mut_aa,
                "fitness":       fitness,
                "delta_fitness": fitness - wt_fitness,
                "mutant_seq":    mutant,
            })

    df = pd.DataFrame(rows).sort_values("delta_fitness", ascending=False)
    return df.reset_index(drop=True)


# ── Fitness landscape visualisation ──────────────────────────────────────────

def visualize_fitness_landscape(
    scan_df: pd.DataFrame,
    wt_sequence: str,
    title: str = "Fitness Landscape",
    positions: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Heatmap of delta_fitness: x = position, y = mutant AA.
    Wild-type residues are marked with '×'.
    Top-5 mutations are annotated in white bold.

    Returns
    -------
    matplotlib Figure
    """
    if positions is None:
        positions = sorted(scan_df["position"].unique())

    # Build matrix [20 AAs × n_positions]
    matrix = np.zeros((20, len(positions)))
    wt_map = {pos: wt_sequence[pos] for pos in positions}

    for _, row in scan_df.iterrows():
        if row["position"] not in positions:
            continue
        pos_idx = positions.index(row["position"])
        aa_idx  = AMINO_ACIDS.index(row["mut_aa"])
        matrix[aa_idx, pos_idx] = row["delta_fitness"]

    # WT row: delta = 0 by definition
    for pi, pos in enumerate(positions):
        wt_aa = wt_sequence[pos]
        if wt_aa in AMINO_ACIDS:
            matrix[AMINO_ACIDS.index(wt_aa), pi] = 0.0

    vmax  = max(abs(matrix).max(), 0.01)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    plt.colorbar(im, ax=ax, label="ΔFitness vs WT", shrink=0.8)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(
        [f"{wt_sequence[p]}{p}" for p in positions], fontsize=7, rotation=45, ha="right"
    )
    ax.set_yticks(range(20))
    ax.set_yticklabels(AMINO_ACIDS, fontsize=8)
    ax.set_xlabel("Position (WT residue + index)", fontsize=11)
    ax.set_ylabel("Mutant Amino Acid", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Mark WT residues
    for pi, pos in enumerate(positions):
        wt_aa = wt_sequence[pos]
        if wt_aa in AMINO_ACIDS:
            ai = AMINO_ACIDS.index(wt_aa)
            ax.text(pi, ai, "×", ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")

    # Annotate top-5 beneficial mutations
    top5 = scan_df[scan_df["delta_fitness"] > 0].head(5)
    for _, row in top5.iterrows():
        if row["position"] not in positions:
            continue
        pi = positions.index(row["position"])
        ai = AMINO_ACIDS.index(row["mut_aa"])
        ax.text(pi, ai, f"{row['delta_fitness']:.2f}",
                ha="center", va="center", color="white",
                fontsize=6, fontweight="bold")

    plt.tight_layout()
    return fig


# ── Top variant selection ─────────────────────────────────────────────────────

def select_top_variants(
    scan_df: pd.DataFrame,
    top_k: int   = 10,
    min_delta: float = 0.02,
    prefer_nonredundant: bool = True,
) -> List[Dict]:
    """
    Pick top beneficial mutations, optionally from distinct positions.

    Returns
    -------
    List of dicts: position, wt_aa, mut_aa, delta_fitness, mutant_seq
    """
    candidates = scan_df[scan_df["delta_fitness"] >= min_delta].copy()
    if candidates.empty:
        return []

    if prefer_nonredundant:
        # One mutation per position (the best one)
        candidates = (candidates
                      .sort_values("delta_fitness", ascending=False)
                      .drop_duplicates("position"))

    return candidates.head(top_k).to_dict("records")


# ── Greedy combinatorial search ───────────────────────────────────────────────

def greedy_combinatorial_search(
    wt_sequence: str,
    scoring_fn: Callable[[str], float],
    top_single_mutations: List[Dict],
    max_mutations: int = 3,
    verbose: bool = True,
) -> List[Dict]:
    """
    Greedy forward selection: at each round, apply the single mutation
    that most improves the current best sequence.

    Parameters
    ----------
    wt_sequence          : starting sequence
    scoring_fn           : fitness function
    top_single_mutations : list of candidate mutations from scan_single_point_mutations
    max_mutations        : max number of mutations to add

    Returns
    -------
    List of dicts per round: sequence, fitness, mutations_applied, delta_from_wt
    """
    wt_fitness    = scoring_fn(wt_sequence)
    current_seq   = wt_sequence
    current_fit   = wt_fitness
    applied       = []
    history       = [{"round": 0, "sequence": wt_sequence,
                       "fitness": wt_fitness, "mutations": [], "delta_from_wt": 0.0}]

    remaining = list(top_single_mutations)

    for rnd in range(1, max_mutations + 1):
        best_gain = -np.inf
        best_mut  = None
        best_seq  = None

        for mut in remaining:
            pos = mut["position"]
            if pos >= len(current_seq):
                continue
            # Apply mutation to current sequence
            new_seq = current_seq[:pos] + mut["mut_aa"] + current_seq[pos + 1:]
            new_fit = scoring_fn(new_seq)
            gain    = new_fit - current_fit
            if gain > best_gain:
                best_gain = gain
                best_mut  = mut
                best_seq  = new_seq

        if best_mut is None or best_gain <= 0:
            if verbose:
                print(f"  Round {rnd}: no improving mutation found. Stopping.")
            break

        current_seq = best_seq
        current_fit = current_fit + best_gain   # already computed in inner loop
        applied.append(best_mut)
        remaining   = [m for m in remaining if m["position"] != best_mut["position"]]

        history.append({
            "round":        rnd,
            "sequence":     current_seq,
            "fitness":      current_fit,
            "mutations":    [f"{m['wt_aa']}{m['position']}{m['mut_aa']}" for m in applied],
            "delta_from_wt": current_fit - wt_fitness,
        })

        if verbose:
            mut_label = f"{best_mut['wt_aa']}{best_mut['position']}{best_mut['mut_aa']}"
            print(f"  Round {rnd}: +{mut_label} → fitness={current_fit:.4f} "
                  f"(Δwt={current_fit - wt_fitness:+.4f})")

    return history


# ── Directed evolution by genetic algorithm ───────────────────────────────────

def simulate_directed_evolution(
    wt_sequence: str,
    scoring_fn: Callable[[str], float],
    n_rounds: int          = 10,
    population_size: int   = 50,
    selection_pressure: float = 0.3,   # fraction of population kept per round
    mutation_rate: float   = 0.05,     # per-position mutation probability
    seed: int              = 42,
) -> pd.DataFrame:
    """
    Simple genetic algorithm with tournament selection.

    Each round:
    1. Evaluate fitness for all sequences
    2. Select top (selection_pressure × population) survivors
    3. Reproduce with point mutations (rate = mutation_rate)
    4. Add WT as elitist carry-over

    Returns
    -------
    pd.DataFrame with per-round stats: round, max_fitness, mean_fitness, std_fitness
    """
    rng = np.random.default_rng(seed)
    L   = len(wt_sequence)

    # Initialise population: WT + random single-mutants
    population = [wt_sequence]
    while len(population) < population_size:
        pos    = rng.integers(L)
        mut_aa = rng.choice([aa for aa in AMINO_ACIDS if aa != wt_sequence[pos]])
        mutant = wt_sequence[:pos] + mut_aa + wt_sequence[pos + 1:]
        population.append(mutant)

    history = []

    for rnd in range(n_rounds):
        # Evaluate fitness
        fitness_vals = np.array([scoring_fn(seq) for seq in population])

        # Record stats
        history.append({
            "round":        rnd,
            "max_fitness":  float(fitness_vals.max()),
            "mean_fitness": float(fitness_vals.mean()),
            "std_fitness":  float(fitness_vals.std()),
            "best_seq":     population[int(np.argmax(fitness_vals))],
        })

        # Selection: top fraction survive
        n_survivors = max(2, int(population_size * selection_pressure))
        top_idx     = np.argsort(fitness_vals)[::-1][:n_survivors]
        survivors   = [population[i] for i in top_idx]

        # Reproduction with mutation
        new_population = [wt_sequence]  # elitist: always keep WT
        new_population.append(survivors[0])  # keep best

        while len(new_population) < population_size:
            parent = survivors[rng.integers(len(survivors))]
            child  = list(parent)
            for pos in range(L):
                if rng.random() < mutation_rate:
                    child[pos] = rng.choice(
                        [aa for aa in AMINO_ACIDS if aa != parent[pos]]
                    )
            new_population.append("".join(child))

        population = new_population

    df = pd.DataFrame(history)
    return df


# ── Plot evolution trajectory ─────────────────────────────────────────────────

def plot_evolution_trajectory(
    history_df: pd.DataFrame,
    title: str = "Directed Evolution Trajectory",
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Line plot of max and mean fitness per round with shaded ±1σ band.
    """
    fig, ax = plt.subplots(figsize=figsize)
    rounds = history_df["round"].values
    maxf   = history_df["max_fitness"].values
    meanf  = history_df["mean_fitness"].values
    stdf   = history_df["std_fitness"].values

    ax.plot(rounds, maxf,  color="#E74C3C", lw=2, label="Best individual")
    ax.plot(rounds, meanf, color="#3498DB", lw=2, label="Population mean")
    ax.fill_between(rounds, meanf - stdf, meanf + stdf,
                    alpha=0.2, color="#3498DB", label="±1 σ")

    ax.set_xlabel("Evolution Round", fontsize=12)
    ax.set_ylabel("Fitness Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate improvement
    delta = maxf[-1] - maxf[0]
    ax.annotate(
        f"Δ = {delta:+.3f}",
        xy=(rounds[-1], maxf[-1]),
        xytext=(rounds[-1] * 0.6, maxf[-1] * 0.97),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, fontweight="bold",
    )

    plt.tight_layout()
    return fig
