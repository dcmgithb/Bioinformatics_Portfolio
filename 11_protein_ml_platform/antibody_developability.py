"""
Antibody Developability Prediction & Optimisation Pipeline
============================================================
Full end-to-end pipeline for therapeutic antibody candidate profiling:

  1. Generate 100 synthetic VH sequences with realistic CDR3 diversity
  2. Compute physicochemical + ESM-mock features
  3. Train multi-task model: binding affinity / aggregation / Tm
  4. Fitness landscape scan across CDR-H3
  5. Directed evolution simulation
  6. 6-panel developability dashboard
  7. Rank candidates and export CSV

Usage
-----
    python antibody_developability.py

Python  : >= 3.10
PyTorch : >= 2.0
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from protein_features import (
    compute_sequence_features, compute_antibody_features,
    extract_cdr_regions, encode_sequence, AMINO_ACIDS, AA_PROPERTIES
)
from esm_antibody_model import ESMEmbedder, ESMAntibodyModel
from fitness_landscape import (
    scan_single_point_mutations, visualize_fitness_landscape,
    select_top_variants, greedy_combinatorial_search,
    simulate_directed_evolution, plot_evolution_trajectory,
)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ── VH framework regions (human IGHV1-2*02-based) ─────────────────────────────

FR1 = "QVQLVQSGAEVKKPGASVKVSCKASGYTFT"
FR2 = "WVRQAPGQGLEWMG"
FR3 = "RVTMTRDTSTSTVYMELSSLRSEDTAVYYCAR"
FR4 = "WGQGTLVTVSS"

CDR1_TEMPLATES = [
    "SYAMS", "GYTFT", "SYWMH", "DYNMD", "SYGMH",
    "NYGMN", "SYGIS", "DYYWS", "SYWIT", "GYSFN",
]
CDR2_TEMPLATES = [
    "WINPNSGGTNYAQKFQG", "SISSGGSSTYYADSVKG", "EINPSNGRTNYNEKFKG",
    "SISSSGGSTYYADSVKG", "WINTYTGEPTYAQKFQG", "EISPSGGSTYYADSVKG",
]

CDR3_RESIDUES = list("ACDEFGHIKLMNQRSTVWY")  # no P (rare in CDR3)
CDR3_LENGTH_DIST = {8: 0.10, 9: 0.15, 10: 0.20, 11: 0.20,
                    12: 0.15, 13: 0.10, 14: 0.07, 15: 0.03}


def generate_vh_sequence(rng: np.random.Generator, cdr3_bias: Optional[str] = None) -> str:
    """Generate a realistic human VH sequence with random CDR loops."""
    cdr1 = rng.choice(CDR1_TEMPLATES)
    cdr2 = rng.choice(CDR2_TEMPLATES)

    # CDR3 length drawn from realistic distribution
    lengths = list(CDR3_LENGTH_DIST.keys())
    probs   = list(CDR3_LENGTH_DIST.values())
    l3      = rng.choice(lengths, p=probs)

    if cdr3_bias:
        # Bias CDR3 toward a given sequence with mutations
        base = list(cdr3_bias[:l3].ljust(l3, "G"))
        for i in range(len(base)):
            if rng.random() < 0.3:
                base[i] = rng.choice(CDR3_RESIDUES)
        cdr3 = "".join(base)
    else:
        # Preference for aromatic/charged residues in CDR3 (biologically motivated)
        weights = np.array([
            1 if aa in "YWFHRD" else 0.5
            for aa in CDR3_RESIDUES
        ])
        weights /= weights.sum()
        cdr3 = "".join(rng.choice(CDR3_RESIDUES, size=l3, p=weights))

    return FR1 + cdr1 + FR2 + cdr2 + FR3 + cdr3 + FR4



def generate_synthetic_antibody_dataset(
    n: int = 100, seed: int = SEED
) -> pd.DataFrame:
    """
    Generate a synthetic antibody dataset with realistic property distributions.

    Labels are computed from physicochemical rules + noise:
    - Binding affinity (log pKd): correlated with CDR3 aromaticity & length
    - Aggregation probability: correlated with hydrophobic patch score
    - Thermal stability (Tm °C): correlated with GRAVY and net charge
    """
    rng = np.random.default_rng(seed)

    # Generate a lead CDR3 for a biased subset
    lead_cdr3 = "ARDYYGSSGWYFDV"

    rows = []
    for i in range(n):
        # 40% of sequences biased toward the lead CDR3 (simulates a lead series)
        bias = lead_cdr3 if i < n * 0.4 else None
        vh   = generate_vh_sequence(rng, cdr3_bias=bias)

        # Compute features
        seq_feats  = compute_sequence_features(vh)
        ab_feats   = compute_antibody_features(vh)
        cdrs       = extract_cdr_regions(vh)
        cdr3       = cdrs.get("CDR-H3", "")

        # Synthetic labels based on physicochemical rules
        aromatic_cdr3 = sum(1 for aa in cdr3 if aa in "YWF") / max(len(cdr3), 1)
        hydro_cdr3    = ab_feats.get("CDR_mean_hydrophobicity", 0.0)
        net_charge    = seq_feats.get("net_charge_pH7", 0.0)
        gravy         = seq_feats.get("gravy", 0.0)
        instab        = seq_feats.get("instability_index", 40.0)

        # Binding affinity: aromatics in CDR3 → tighter binding
        bind_base = 8.5 + 2.0 * aromatic_cdr3 - 0.3 * abs(net_charge) + rng.normal(0, 0.4)
        bind_affinity = float(np.clip(bind_base, 5.0, 12.0))

        # Aggregation: hydrophobic CDR → higher aggregation risk
        aggr_logit = 2.0 * hydro_cdr3 - 1.5 + rng.normal(0, 0.5)
        aggr_prob  = float(1 / (1 + np.exp(-aggr_logit)))

        # Thermal stability: lower GRAVY and more charged → more stable
        tm_base = 72.0 - 5.0 * gravy + 1.5 * abs(net_charge) - 0.1 * (instab - 40)
        tm       = float(np.clip(tm_base + rng.normal(0, 3.0), 50, 90))

        rows.append({
            "sequence":     vh,
            "CDR3":         cdr3,
            "CDR3_length":  len(cdr3),
            "binding":      bind_affinity,
            "aggregation":  float(aggr_prob > 0.5),   # binary label
            "aggr_prob_gt": aggr_prob,                 # continuous (for analysis)
            "stability":    tm,
            "GRAVY":        gravy,
            "net_charge":   net_charge,
            "CDR3_aromatic":aromatic_cdr3,
            "CDR3_hydro":   hydro_cdr3,
            "instability":  instab,
            "is_lead_series": int(i < n * 0.4),
        })

    return pd.DataFrame(rows)


# ── Feature matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, embedder: ESMEmbedder) -> np.ndarray:
    """Combine ESM embeddings + hand-crafted features."""
    sequences = df["sequence"].tolist()

    # ESM embeddings
    emb = embedder.encode(sequences)

    # Hand-crafted features
    hand_feats = []
    for seq in sequences:
        sf = compute_sequence_features(seq)
        af = compute_antibody_features(seq)
        row = [
            sf.get("gravy", 0), sf.get("net_charge_pH7", 0),
            sf.get("instability_index", 40), sf.get("aromatic_fraction", 0),
            sf.get("hydrophobic_fraction", 0), sf.get("mean_flexibility", 0.45),
            af.get("CDR-H3_length", 10), af.get("CDR-H3_hydrophobicity", 0),
            af.get("CDR-H3_charge", 0), af.get("CDR-H3_aromatic_frac", 0),
            af.get("total_CDR_length", 30), af.get("CDR_mean_hydrophobicity", 0),
        ]
        hand_feats.append(row)

    return np.hstack([emb, np.array(hand_feats, dtype=np.float32)])


# ── Dashboard ─────────────────────────────────────────────────────────────────

def plot_developability_dashboard(
    df: pd.DataFrame,
    scan_df: pd.DataFrame,
    evo_df: pd.DataFrame,
    greedy_hist: list,
    wt_seq: str,
    results: dict,
    out_path: Path,
):
    """6-panel developability dashboard."""
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ── Panel 1: Developability radar ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0], projection="polar")
    categories  = ["Binding\n(norm)", "Stability\n(norm)",
                   "Aggregation\n(low=good)", "CDR3\naromatics", "Net charge\n(abs inv)"]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]

    # Top 5 vs bottom 5 average
    df_sorted = df.sort_values("binding", ascending=False)
    top5   = df_sorted.head(5)
    bot5   = df_sorted.tail(5)

    def radar_vals(sub):
        return [
            sub["binding"].mean() / 12,
            sub["stability"].mean() / 90,
            1 - sub["aggr_prob_gt"].mean(),
            sub["CDR3_aromatic"].mean(),
            1 - min(abs(sub["net_charge"].mean()) / 5, 1),
        ]

    for vals, label, col in [(radar_vals(top5), "Top-5", "#2ECC71"),
                              (radar_vals(bot5),  "Bot-5", "#E74C3C")]:
        vals_plot = vals + [vals[0]]
        ax1.plot(angles, vals_plot, "o-", lw=2, color=col, label=label)
        ax1.fill(angles, vals_plot, alpha=0.15, color=col)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=7.5)
    ax1.set_ylim(0, 1)
    ax1.set_title("Developability Radar\n(Top-5 vs Bot-5)", fontsize=10, pad=15)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    # ── Panel 2: Property scatter matrix ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(df["binding"], df["stability"],
                     c=df["aggr_prob_gt"], cmap="RdYlGn_r",
                     s=50, alpha=0.75, edgecolors="none")
    plt.colorbar(sc, ax=ax2, label="Aggregation prob.")
    ax2.set_xlabel("Binding affinity (log pKd)", fontsize=10)
    ax2.set_ylabel("Thermal stability (Tm °C)", fontsize=10)
    ax2.set_title("Property Landscape\n(colour = aggregation risk)", fontsize=10)
    # Quadrant labels
    bm, sm = df["binding"].median(), df["stability"].median()
    ax2.axvline(bm, ls="--", color="grey", alpha=0.5, lw=1)
    ax2.axhline(sm, ls="--", color="grey", alpha=0.5, lw=1)
    ax2.text(bm + 0.05, sm + 0.5, "Ideal", fontsize=8, color="green", style="italic")
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: CDR-H3 fitness landscape heatmap (condensed) ──────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if scan_df is not None and not scan_df.empty:
        cdr3_seq  = extract_cdr_regions(wt_seq).get("CDR-H3", "")
        positions = sorted(scan_df["position"].unique())[:10]  # first 10 pos
        aa_order  = list("ACDEFGHIKLMNPQRSTVWY")
        mat = np.zeros((20, len(positions)))
        for _, row in scan_df[scan_df["position"].isin(positions)].iterrows():
            pi = positions.index(row["position"])
            ai = aa_order.index(row["mut_aa"]) if row["mut_aa"] in aa_order else -1
            if ai >= 0:
                mat[ai, pi] = row["delta_fitness"]
        vmax = max(abs(mat).max(), 0.01)
        im3 = ax3.imshow(mat, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax)
        plt.colorbar(im3, ax=ax3, label="ΔFitness", shrink=0.8)
        ax3.set_xticks(range(len(positions)))
        ax3.set_xticklabels([f"P{p}" for p in positions], fontsize=7, rotation=45)
        ax3.set_yticks(range(20))
        ax3.set_yticklabels(aa_order, fontsize=6)
        ax3.set_title("CDR-H3 Fitness Landscape\n(single-point mutations)", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "No scan data", ha="center", va="center")
        ax3.set_title("CDR-H3 Fitness Landscape", fontsize=10)

    # ── Panel 4: Directed evolution trajectory ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if evo_df is not None:
        rounds = evo_df["round"].values
        ax4.plot(rounds, evo_df["max_fitness"],  "r-o", lw=2, ms=4, label="Best")
        ax4.plot(rounds, evo_df["mean_fitness"], "b-o", lw=2, ms=4, label="Mean")
        ax4.fill_between(
            rounds,
            evo_df["mean_fitness"] - evo_df["std_fitness"],
            evo_df["mean_fitness"] + evo_df["std_fitness"],
            alpha=0.2, color="blue"
        )
        ax4.set_xlabel("Evolution Round", fontsize=10)
        ax4.set_ylabel("Fitness", fontsize=10)
        ax4.set_title("Directed Evolution Trajectory", fontsize=10)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    # ── Panel 5: Greedy optimisation path ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if greedy_hist:
        g_rounds  = [h["round"]   for h in greedy_hist]
        g_fitness = [h["fitness"] for h in greedy_hist]
        ax5.plot(g_rounds, g_fitness, "gs-", lw=2, ms=8)
        for h in greedy_hist[1:]:
            muts = h.get("mutations", [])
            label = " + ".join(muts) if muts else ""
            ax5.annotate(
                label, xy=(h["round"], h["fitness"]),
                xytext=(h["round"] - 0.1, h["fitness"] + 0.01),
                fontsize=7, color="#2C3E50",
            )
        ax5.set_xlabel("Greedy Round", fontsize=10)
        ax5.set_ylabel("Fitness", fontsize=10)
        ax5.set_title("Greedy Combinatorial Optimisation", fontsize=10)
        ax5.grid(True, alpha=0.3)

    # ── Panel 6: Model performance + top candidates ─────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    top_cands = df.sort_values("binding", ascending=False).head(8)
    col_labels = ["CDR3", "Bind", "Stab(Tm)", "Aggr"]
    table_data = [
        [row["CDR3"][:12] + "…" if len(row["CDR3"]) > 12 else row["CDR3"],
         f"{row['binding']:.2f}",
         f"{row['stability']:.1f}°C",
         "✓" if row["aggregation"] == 0 else "✗"]
        for _, row in top_cands.iterrows()
    ]
    tbl = ax6.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.5)
    # Colour header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax6.set_title("Top-8 Candidates by Binding Affinity", fontsize=10, pad=12)

    fig.suptitle(
        "Antibody Developability Platform — VH Engineering Dashboard",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


# ── Main pipeline ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Antibody Developability Prediction & Optimisation Pipeline")
    print("=" * 65)

    # 1. Generate synthetic dataset
    print("\n[1/6] Generating 100 synthetic VH sequences...")
    df = generate_synthetic_antibody_dataset(n=100, seed=SEED)
    print(f"  Binding:    {df['binding'].mean():.2f} ± {df['binding'].std():.2f} log pKd")
    print(f"  Stability:  {df['stability'].mean():.1f} ± {df['stability'].std():.1f} °C")
    print(f"  Aggregators: {df['aggregation'].sum():.0f}/{len(df)}")

    # 2. Build features + train model
    print("\n[2/6] Building ESM-mock embeddings + training multi-task model...")
    model = ESMAntibodyModel(mock=True, device="cpu", epochs=30, batch_size=16)
    losses = model.fit(
        sequences   = df["sequence"].tolist(),
        binding     = df["binding"].values,
        aggregation = df["aggregation"].values,
        stability   = df["stability"].values,
    )
    print(f"  Final training loss: {losses[-1]:.4f}")

    # 3. Cross-validated predictions using sklearn GBM (interpretable baseline)
    print("\n[3/6] Cross-validated predictions (GBM on physicochemical features)...")
    embedder = ESMEmbedder(mock=True)
    X = build_feature_matrix(df, embedder)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Binding affinity (regression)
    gbm_bind = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=SEED)
    bind_cv  = cross_val_predict(gbm_bind, X_sc, df["binding"].values,
                                  cv=KFold(5, shuffle=True, random_state=SEED))
    r2_bind   = r2_score(df["binding"].values, bind_cv)
    mae_bind  = mean_absolute_error(df["binding"].values, bind_cv)

    # Aggregation (classification)
    gbm_aggr = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=SEED)
    aggr_cv  = cross_val_predict(gbm_aggr, X_sc, df["aggregation"].values.astype(int),
                                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  method="predict_proba")[:, 1]
    auc_aggr = roc_auc_score(df["aggregation"].values, aggr_cv)

    # Thermal stability (regression)
    gbm_stab = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=SEED)
    stab_cv  = cross_val_predict(gbm_stab, X_sc, df["stability"].values,
                                  cv=KFold(5, shuffle=True, random_state=SEED))
    r2_stab  = r2_score(df["stability"].values, stab_cv)

    print(f"  Binding    → R²={r2_bind:.3f}, MAE={mae_bind:.3f}")
    print(f"  Aggregation→ AUC={auc_aggr:.3f}")
    print(f"  Stability  → R²={r2_stab:.3f}")

    # 4. Fitness landscape scan on best candidate's CDR-H3
    print("\n[4/6] Fitness landscape scan on lead CDR-H3...")
    best_idx   = int(df["binding"].idxmax())
    lead_seq   = df.loc[best_idx, "sequence"]
    lead_cdrs  = extract_cdr_regions(lead_seq)
    cdr3_seq   = lead_cdrs.get("CDR-H3", "")

    # Locate CDR3 in full VH sequence
    cdr3_start = lead_seq.find(cdr3_seq) if cdr3_seq else 0
    cdr3_positions = list(range(cdr3_start, cdr3_start + len(cdr3_seq)))

    # Simple fitness function: predicted binding from GBM
    gbm_bind.fit(X_sc, df["binding"].values)

    def fitness_fn(seq: str) -> float:
        feats = build_feature_matrix(
            pd.DataFrame({"sequence": [seq]}), embedder
        )
        return float(gbm_bind.predict(scaler.transform(feats))[0])

    scan_df = scan_single_point_mutations(
        lead_seq, fitness_fn, positions=cdr3_positions[:8]
    )
    print(f"  Scanned {len(cdr3_positions[:8])} CDR-H3 positions × 19 mutations "
          f"= {len(scan_df)} variants")
    top_muts = select_top_variants(scan_df, top_k=5, min_delta=0.01)
    print(f"  Top mutation: {top_muts[0]['wt_aa']}{top_muts[0]['position']}"
          f"{top_muts[0]['mut_aa']} (Δ={top_muts[0]['delta_fitness']:+.3f})"
          if top_muts else "  No beneficial mutations found")

    # 5. Greedy combinatorial search
    print("\n[5/6] Greedy combinatorial optimisation (max 3 mutations)...")
    greedy_hist = greedy_combinatorial_search(
        lead_seq, fitness_fn, top_muts, max_mutations=3, verbose=True
    )
    best_variant = greedy_hist[-1]
    print(f"  Best variant: {best_variant['mutations']} "
          f"→ fitness={best_variant['fitness']:.4f} "
          f"(Δwt={best_variant['delta_from_wt']:+.4f})")

    # 6. Directed evolution simulation
    print("\n[6/6] Directed evolution simulation (10 rounds, pop=30)...")
    evo_df = simulate_directed_evolution(
        lead_seq, fitness_fn,
        n_rounds=10, population_size=30,
        selection_pressure=0.3, mutation_rate=0.04, seed=SEED
    )
    print(f"  Start fitness: {evo_df['max_fitness'].iloc[0]:.4f} "
          f"→ End: {evo_df['max_fitness'].iloc[-1]:.4f} "
          f"(+{evo_df['max_fitness'].iloc[-1] - evo_df['max_fitness'].iloc[0]:.4f})")

    # 7. Dashboard
    print("\nGenerating 6-panel developability dashboard...")
    plot_developability_dashboard(
        df, scan_df, evo_df, greedy_hist, lead_seq,
        results={"r2_bind": r2_bind, "auc_aggr": auc_aggr, "r2_stab": r2_stab},
        out_path=OUT_DIR / "antibody_developability_dashboard.png",
    )

    # Export candidate ranking
    df["bind_cv_pred"]  = bind_cv
    df["aggr_cv_pred"]  = aggr_cv
    df["stab_cv_pred"]  = stab_cv
    df["dev_score"]     = (
        0.5 * (df["binding"] / df["binding"].max()) +
        0.3 * (df["stability"] / df["stability"].max()) +
        0.2 * (1 - df["aggr_prob_gt"])
    )
    df.sort_values("dev_score", ascending=False).to_csv(
        OUT_DIR / "antibody_candidates_ranked.csv", index=False
    )
    print("Saved results/antibody_candidates_ranked.csv")
    print("\n✓ Pipeline complete.")
