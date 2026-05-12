"""
run_discovery.py — Autonomous Drug Discovery Campaign Runner
=============================================================
Entry point for the agentic drug discovery workflow.

Usage
-----
    python run_discovery.py
    python run_discovery.py --target CDK4/6 --max-steps 20
    python run_discovery.py --seed-smiles "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)CCN1"

Outputs
-------
    /tmp/memory.json            — full campaign memory (molecules + trace)
    /tmp/discovery_campaign.png — 4-panel results figure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Lazy imports — tools and agent may not exist yet
# ---------------------------------------------------------------------------
try:
    from agent.discovery_agent import DrugDiscoveryAgent, AgentConfig
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False

try:
    from agent.memory import AgentMemory, MoleculeEntry
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False

try:
    from agent.tools import TOOLS, execute_tool
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False


# ---------------------------------------------------------------------------
# 1. CAMPAIGN RUNNER
# ---------------------------------------------------------------------------

def run_discovery_campaign(
    target: str = "CDK4/6",
    max_steps: int = 20,
    seed_smiles: Optional[str] = None,
    verbose: bool = True,
) -> "AgentMemory":
    """
    Run an autonomous drug discovery campaign and return populated AgentMemory.

    Parameters
    ----------
    target      : biological target name (used for literature search + scoring)
    max_steps   : maximum ReAct iterations
    seed_smiles : starting molecule; defaults to a CDK4/6-like scaffold
    verbose     : print each step to stdout

    Returns
    -------
    AgentMemory populated with all discovered molecules and reasoning traces.
    """
    if not HAS_AGENT:
        raise ImportError(
            "agent/discovery_agent.py not found. "
            "Please write that module before running the campaign."
        )

    if seed_smiles is None:
        seed_smiles = "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)CCN1"

    config = AgentConfig(
        target=target,
        max_steps=max_steps,
        seed_smiles=seed_smiles,
        mock_llm=True,
        verbose=verbose,
    )

    if verbose:
        print("=" * 64)
        print(f"  AUTONOMOUS DRUG DISCOVERY CAMPAIGN")
        print(f"  Target  : {target}")
        print(f"  Steps   : {max_steps}")
        print(f"  Seed    : {seed_smiles[:50]}{'…' if len(seed_smiles) > 50 else ''}")
        print("=" * 64)

    agent = DrugDiscoveryAgent(config=config)
    memory = agent.run()

    if verbose:
        print("\n" + memory.summary())
        print("\n[Report]")
        print(agent.generate_report())

    return memory


# ---------------------------------------------------------------------------
# 2. RESULTS VISUALISATION
# ---------------------------------------------------------------------------

def visualize_campaign_results(
    memory: "AgentMemory",
    out_path: str = "/tmp/discovery_campaign.png",
) -> None:
    """
    4-panel campaign results figure:
      1. pIC50 trajectory vs step (coloured by source)
      2. Property distributions (MW, LogP, QED violin plots)
      3. Chemical space PCA (Morgan FP, coloured by pIC50)
      4. Top-10 candidates table
    """
    df = memory.molecules.to_dataframe()
    if df.empty:
        print("No molecules to visualise.")
        return

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Autonomous Drug Discovery — Campaign Results",
                 fontsize=15, fontweight="bold", y=0.98)

    # ── Panel 1: pIC50 discovery trajectory ────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    source_colors = {
        "similarity_search": "#1976d2",
        "scaffold_hop":      "#f57c00",
        "optimization":      "#388e3c",
        "unknown":           "#9e9e9e",
    }
    scored = df[df["binding_score"].notna()].copy()
    if not scored.empty:
        for src, grp in scored.groupby("source"):
            col = source_colors.get(src, "#9e9e9e")
            ax1.scatter(grp["step_discovered"], grp["binding_score"],
                        c=col, s=60, alpha=0.8, label=src, zorder=3)

        # Running best line
        steps_sorted = scored.sort_values("step_discovered")
        best_so_far = steps_sorted["binding_score"].cummax()
        ax1.plot(steps_sorted["step_discovered"], best_so_far,
                 color="black", lw=1.5, linestyle="--", alpha=0.5, label="best so far")

        ax1.axhline(8.0, color="#d32f2f", lw=1, linestyle=":", alpha=0.7, label="target pIC50=8.0")
        ax1.set_xlabel("Discovery Step")
        ax1.set_ylabel("Predicted pIC50")
        ax1.set_title("Discovery Trajectory", fontweight="bold")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

    # ── Panel 2: Property distributions ────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    props = ["MW", "LogP", "QED"]
    available = [p for p in props if p in df.columns and df[p].notna().sum() > 3]

    if available:
        plot_data = [df[p].dropna().values for p in available]
        parts = ax2.violinplot(plot_data, positions=range(len(available)),
                               showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#42a5f5")
            pc.set_alpha(0.7)
        ax2.set_xticks(range(len(available)))
        ax2.set_xticklabels(available)
        ax2.set_ylabel("Value")
        ax2.set_title("Property Distributions", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "Insufficient property data",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Property Distributions", fontweight="bold")

    # ── Panel 3: Chemical space PCA ─────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from sklearn.decomposition import PCA

        fps = []
        valid_idx = []
        for i, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
                fps.append(np.array(fp))
                valid_idx.append(i)

        if len(fps) >= 4:
            X = np.vstack(fps)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X)
            sub_df = df.iloc[valid_idx].reset_index(drop=True)

            scores = sub_df["binding_score"].values
            has_score = ~pd.isna(scores)

            sc = ax3.scatter(
                coords[has_score, 0], coords[has_score, 1],
                c=scores[has_score], cmap="RdYlGn",
                vmin=5.0, vmax=10.0, s=40, alpha=0.8, zorder=3
            )
            ax3.scatter(
                coords[~has_score, 0], coords[~has_score, 1],
                c="#bdbdbd", s=20, alpha=0.5, zorder=2
            )
            plt.colorbar(sc, ax=ax3, label="pIC50")

            # Star = best compound
            if memory.current_best is not None:
                best_smi = memory.current_best.canonical_smiles
                for j, row in sub_df.iterrows():
                    if row["smiles"] == best_smi:
                        ax3.scatter(coords[j, 0], coords[j, 1],
                                    marker="*", s=300, c="gold",
                                    edgecolors="black", zorder=5, label="best")
                        break

            ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax3.legend(fontsize=8)
        else:
            ax3.text(0.5, 0.5, "Too few molecules for PCA",
                     ha="center", va="center", transform=ax3.transAxes)
    except Exception as e:
        ax3.text(0.5, 0.5, f"PCA unavailable\n{e}",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=8)

    ax3.set_title("Chemical Space (PCA)", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Top-10 candidates table ───────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    ax4.set_title("Top-10 Candidates", fontweight="bold")

    top10 = memory.molecules.get_top_k(k=10, by="binding_score")
    if top10:
        col_labels = ["Rank", "Name", "pIC50", "Source", "Step"]
        rows_tbl = []
        for i, e in enumerate(top10, 1):
            rows_tbl.append([
                str(i),
                e.name[:16],
                f"{e.binding_score:.2f}" if e.binding_score else "—",
                e.source[:14],
                str(e.step_discovered),
            ])

        tbl = ax4.table(
            cellText=rows_tbl,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 1.5)

        # Colour header
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#1565c0")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Colour top-3 rows
        rank_colors = ["#fff9c4", "#f3e5f5", "#e8f5e9"]
        for i in range(min(3, len(rows_tbl))):
            for j in range(len(col_labels)):
                tbl[i + 1, j].set_facecolor(rank_colors[i])
    else:
        ax4.text(0.5, 0.5, "No scored molecules yet",
                 ha="center", va="center", transform=ax4.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nResults figure saved → {out_path}")


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous drug discovery campaign against a molecular target."
    )
    parser.add_argument("--target", default="CDK4/6",
                        help="Biological target (default: CDK4/6)")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Maximum ReAct steps (default: 20)")
    parser.add_argument("--seed-smiles", default=None,
                        help="Starting molecule SMILES (optional)")
    parser.add_argument("--out-dir", default="/tmp",
                        help="Directory for output files (default: /tmp)")
    args = parser.parse_args()

    memory_path = os.path.join(args.out_dir, "memory.json")
    figure_path = os.path.join(args.out_dir, "discovery_campaign.png")

    t0 = time.time()
    memory = run_discovery_campaign(
        target=args.target,
        max_steps=args.max_steps,
        seed_smiles=args.seed_smiles,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nCampaign completed in {elapsed:.1f}s")

    # Save memory
    memory.save(memory_path)
    print(f"Memory saved  → {memory_path}")

    # Save figure
    visualize_campaign_results(memory, out_path=figure_path)

    # Print top-5 summary
    top5 = memory.molecules.get_top_k(k=5)
    if top5:
        print("\nTop-5 compounds:")
        for i, e in enumerate(top5, 1):
            score = f"{e.binding_score:.2f}" if e.binding_score else "n/a"
            print(f"  {i}. {e.name:<20} pIC50={score}  source={e.source}")


if __name__ == "__main__":
    main()
