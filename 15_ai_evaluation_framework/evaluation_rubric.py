"""
evaluation_rubric.py — Structured Evaluation Rubric for AI-Generated
Precision Medicine Content
=====================================================================
Provides a 6-dimension scoring framework for assessing AI outputs in
genomics and clinical contexts:
  • Scientific Accuracy
  • Clinical Relevance
  • Data Integrity
  • Completeness
  • Reasoning Quality
  • Regulatory Compliance

Each dimension scored 1–5 with weighted composite score and calibration
analysis against expert ground truth.
"""

from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. RUBRIC DEFINITION
# ---------------------------------------------------------------------------

@dataclass
class EvaluationDimension:
    name: str
    weight: float               # contribution to composite (weights sum to 1)
    description: str
    scoring_guide: Dict[int, str]   # 1-5 → description
    examples: Dict[int, str] = field(default_factory=dict)  # 1-5 → example


RUBRIC_DIMENSIONS: List[EvaluationDimension] = [
    EvaluationDimension(
        name="Scientific Accuracy",
        weight=0.30,
        description="Factual correctness of genomic/molecular claims against established literature.",
        scoring_guide={
            1: "Multiple critical factual errors (e.g., wrong gene function, incorrect variant impact)",
            2: "One critical or several minor factual errors affecting core claims",
            3: "Mostly accurate with 1-2 minor inaccuracies that do not change conclusions",
            4: "Accurate with only trivial or style-level imprecisions",
            5: "Fully accurate, well-supported by evidence, no errors detected",
        },
        examples={
            1: "States BRCA1 is a proto-oncogene (incorrect — it is a tumour suppressor)",
            3: "Correctly describes TP53 as a tumour suppressor but overstates penetrance",
            5: "Accurately describes KRAS G12D as a gain-of-function oncogenic mutation with correct prevalence",
        },
    ),
    EvaluationDimension(
        name="Clinical Relevance",
        weight=0.25,
        description="Applicability and utility of the output for clinical decision-making.",
        scoring_guide={
            1: "Output is irrelevant or potentially harmful in a clinical context",
            2: "Limited clinical utility; missing key actionable information",
            3: "Moderately useful; addresses the clinical question with some gaps",
            4: "Clinically useful with minor omissions; appropriate for professional use",
            5: "Highly actionable, directly addresses the clinical question with guideline-aligned recommendations",
        },
    ),
    EvaluationDimension(
        name="Data Integrity",
        weight=0.20,
        description="Correctness of data citations, statistics, and numerical claims.",
        scoring_guide={
            1: "Fabricated statistics or data; hallucinated citations",
            2: "Significant numerical errors or unverifiable statistics",
            3: "Statistics plausible but insufficiently sourced or rounded imprecisely",
            4: "Data claims largely verifiable with minor rounding/sourcing issues",
            5: "All statistics accurate, properly sourced, and reproducible",
        },
    ),
    EvaluationDimension(
        name="Completeness",
        weight=0.10,
        description="Coverage of the topic relative to the question asked.",
        scoring_guide={
            1: "Major components of the answer are missing",
            2: "Several important aspects omitted",
            3: "Core question addressed but supplementary depth missing",
            4: "Comprehensive with minor omissions",
            5: "Thorough coverage of all relevant aspects",
        },
    ),
    EvaluationDimension(
        name="Reasoning Quality",
        weight=0.10,
        description="Logical coherence, appropriate uncertainty, and absence of non-sequiturs.",
        scoring_guide={
            1: "Contradictory or logically flawed reasoning",
            2: "Reasoning gaps that undermine the conclusion",
            3: "Sound reasoning with minor logical leaps",
            4: "Well-reasoned with appropriate hedging",
            5: "Exemplary reasoning; uncertainty quantified; alternative hypotheses considered",
        },
    ),
    EvaluationDimension(
        name="Regulatory Compliance",
        weight=0.05,
        description="Adherence to reporting standards (HIPAA, GDPR, clinical guidelines).",
        scoring_guide={
            1: "Contains PHI or violates regulatory requirements",
            2: "Regulatory concerns not addressed despite being relevant",
            3: "Regulatory context acknowledged but incompletely addressed",
            4: "Compliant with minor guidance gaps",
            5: "Fully compliant; explicitly addresses relevant regulatory framework",
        },
    ),
]

DIMENSION_NAMES = [d.name for d in RUBRIC_DIMENSIONS]
DIMENSION_WEIGHTS = {d.name: d.weight for d in RUBRIC_DIMENSIONS}


# ---------------------------------------------------------------------------
# 2. RUBRIC SCORER
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    item_id: str
    scores: Dict[str, int]           # dimension → 1-5
    composite_score: float           # weighted average
    notes: Dict[str, str] = field(default_factory=dict)
    evaluator_id: str = "auto"
    timestamp: str = ""

    @property
    def grade(self) -> str:
        s = self.composite_score
        return "A" if s >= 4.5 else "B" if s >= 3.5 else "C" if s >= 2.5 else "D" if s >= 1.5 else "F"


class RubricScorer:
    """Scores AI outputs on the 6-dimension precision medicine rubric."""

    def __init__(self, rubric: List[EvaluationDimension] = None) -> None:
        self.rubric = rubric or RUBRIC_DIMENSIONS

    def score(
        self,
        item_id: str,
        scores: Dict[str, int],
        notes: Optional[Dict[str, str]] = None,
        evaluator_id: str = "auto",
    ) -> EvaluationResult:
        """
        Compute composite score from per-dimension scores.

        Parameters
        ----------
        scores : dict mapping dimension name → integer score 1-5
        """
        # Validate scores
        for dim_name, s in scores.items():
            if not (1 <= s <= 5):
                raise ValueError(f"Score for '{dim_name}' must be 1-5, got {s}")

        # Weighted composite
        composite = sum(
            scores.get(d.name, 3) * d.weight
            for d in self.rubric
        )

        return EvaluationResult(
            item_id=item_id,
            scores=scores,
            composite_score=round(composite, 3),
            notes=notes or {},
            evaluator_id=evaluator_id,
            timestamp=datetime.datetime.utcnow().isoformat(),
        )

    def score_batch(self, items: List[Dict]) -> pd.DataFrame:
        """
        Score a batch of items.

        items : list of dicts with keys: item_id, scores, notes (optional)
        Returns DataFrame with scores + composite.
        """
        results = []
        for item in items:
            r = self.score(
                item_id=item["item_id"],
                scores=item["scores"],
                notes=item.get("notes", {}),
                evaluator_id=item.get("evaluator_id", "auto"),
            )
            row = {"item_id": r.item_id, "composite": r.composite_score,
                   "grade": r.grade}
            row.update({f"score_{k.replace(' ', '_')}": v
                        for k, v in r.scores.items()})
            results.append(row)
        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 3. SYNTHETIC AI EVALUATION DATASET
# ---------------------------------------------------------------------------

# Topics for synthetic Q&A pairs
TOPICS = [
    "BRCA1/2 pathogenic variant interpretation",
    "KRAS oncogenic mutation mechanisms",
    "CYP2D6 pharmacogenomics",
    "HbA1c reference range interpretation",
    "EGFR targeted therapy resistance",
    "Lynch syndrome mismatch repair genes",
    "APOE4 Alzheimer's disease risk",
    "CKD staging by eGFR",
    "PD-L1 immunotherapy biomarker",
    "Warfarin pharmacogenomics (CYP2C9/VKORC1)",
]


def generate_evaluation_dataset(
    n: int = 50,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic AI outputs with expert ground-truth scores.

    Returns
    -------
    items_df      : DataFrame with item_id, topic, ai_output_quality
    ground_truth  : DataFrame with item_id + per-dimension expert scores
    """
    rng = np.random.default_rng(seed)

    # Quality tiers: poor / moderate / good / excellent
    tier_probs = [0.15, 0.25, 0.40, 0.20]
    tier_score_ranges = {
        "poor":      {d: (1, 2) for d in DIMENSION_NAMES},
        "moderate":  {d: (2, 3) for d in DIMENSION_NAMES},
        "good":      {d: (3, 4) for d in DIMENSION_NAMES},
        "excellent": {d: (4, 5) for d in DIMENSION_NAMES},
    }
    # Scientific Accuracy is harder to get right
    for tier in ["moderate", "good"]:
        tier_score_ranges[tier]["Scientific Accuracy"] = (
            tier_score_ranges[tier]["Scientific Accuracy"][0],
            tier_score_ranges[tier]["Scientific Accuracy"][1] - 1,
        )

    items, gt_rows = [], []
    tiers = ["poor", "moderate", "good", "excellent"]
    for i in range(n):
        tier = rng.choice(tiers, p=tier_probs)
        topic = rng.choice(TOPICS)
        item_id = f"ITEM_{i+1:03d}"

        scores = {}
        for dim in DIMENSION_NAMES:
            lo, hi = tier_score_ranges[tier][dim]
            score = int(rng.integers(lo, hi + 1))
            scores[dim] = score

        # Composite ground truth
        scorer = RubricScorer()
        result = scorer.score(item_id, scores, evaluator_id="expert_gt")
        composite = result.composite_score

        items.append({
            "item_id": item_id,
            "topic": topic,
            "quality_tier": tier,
            "gt_composite": composite,
        })
        gt_row = {"item_id": item_id, "composite": composite}
        gt_row.update({f"gt_{d.replace(' ', '_')}": scores[d]
                       for d in DIMENSION_NAMES})
        gt_rows.append(gt_row)

    return pd.DataFrame(items), pd.DataFrame(gt_rows)


# ---------------------------------------------------------------------------
# 4. CALIBRATION ANALYSIS
# ---------------------------------------------------------------------------

def calibration_analysis(
    ground_truth: pd.DataFrame,
    predicted_scores: pd.DataFrame,
    dimension: str = "composite",
) -> Dict:
    """
    Compare predicted scores against ground truth.

    Returns Pearson r, MAE, bias, and kappa (rounded to integer bins).
    """
    gt_col = "composite" if dimension == "composite" else f"gt_{dimension.replace(' ', '_')}"
    pred_col = "composite" if dimension == "composite" else f"pred_{dimension.replace(' ', '_')}"

    merged = ground_truth[["item_id", gt_col]].merge(
        predicted_scores[["item_id", pred_col]], on="item_id"
    )
    if merged.empty:
        return {}

    gt = merged[gt_col].values
    pred = merged[pred_col].values

    r, p = stats.pearsonr(gt, pred)
    mae = mean_absolute_error(gt, pred)
    bias = float(np.mean(pred - gt))

    # Kappa on integer-rounded scores (1-5)
    gt_int = np.clip(np.round(gt).astype(int), 1, 5)
    pred_int = np.clip(np.round(pred).astype(int), 1, 5)
    kappa = cohen_kappa_score(gt_int, pred_int,
                               labels=[1, 2, 3, 4, 5],
                               weights="quadratic")

    return {
        "dimension":  dimension,
        "n":          len(merged),
        "pearson_r":  round(float(r), 4),
        "p_value":    round(float(p), 4),
        "mae":        round(float(mae), 4),
        "bias":       round(float(bias), 4),
        "kappa_qwt":  round(float(kappa), 4),
    }


# ---------------------------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------------------------

def plot_rubric_results(
    scores_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    out_path: str = "evaluation_rubric.png",
) -> None:
    """3-panel rubric results figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("AI Output Evaluation — Rubric Scoring Results",
                 fontsize=13, fontweight="bold")

    # Panel 1: Score distribution per dimension (violin)
    ax = axes[0]
    dim_score_cols = [c for c in scores_df.columns if c.startswith("score_")]
    dim_labels = [c.replace("score_", "").replace("_", " ") for c in dim_score_cols]
    data = [scores_df[c].dropna().values for c in dim_score_cols]
    if data:
        parts = ax.violinplot(data, positions=range(len(dim_labels)),
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#42a5f5")
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(dim_labels)))
        ax.set_xticklabels(dim_labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Score (1-5)")
        ax.set_ylim(0.5, 5.5)
        ax.set_title("Score Distributions by Dimension", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(3.0, color="red", lw=1, linestyle="--", alpha=0.5, label="Threshold")
        ax.legend(fontsize=8)

    # Panel 2: Composite score histogram by quality tier
    ax = axes[1]
    if "quality_tier" in ground_truth.columns and "gt_composite" in ground_truth.columns:
        tiers = ["poor", "moderate", "good", "excellent"]
        tier_colors = ["#d32f2f", "#f57c00", "#1976d2", "#388e3c"]
        for tier, col in zip(tiers, tier_colors):
            sub = ground_truth[ground_truth["quality_tier"] == tier]["gt_composite"]
            if not sub.empty:
                ax.hist(sub, bins=10, alpha=0.6, color=col, label=tier, density=True)
        ax.set_xlabel("Composite Score")
        ax.set_ylabel("Density")
        ax.set_title("Score Distribution by Quality Tier", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 3: Dimension weight radar
    ax = axes[2]
    ax.set_aspect("equal")
    dim_names = [d.name for d in RUBRIC_DIMENSIONS]
    weights = [d.weight for d in RUBRIC_DIMENSIONS]
    N = len(dim_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    weights_plot = weights + weights[:1]

    ax_polar = fig.add_axes([0.68, 0.1, 0.28, 0.8], polar=True)
    ax.remove()
    ax_polar.plot(angles, weights_plot, "o-", color="#1976d2", lw=2)
    ax_polar.fill(angles, weights_plot, alpha=0.25, color="#1976d2")
    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels([d.replace(" ", "\n") for d in dim_names], fontsize=7)
    ax_polar.set_ylim(0, 0.35)
    ax_polar.set_title("Dimension Weights", fontsize=10, fontweight="bold", pad=20)

    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Rubric results plot → {out_path}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(__file__) or "."

    print("=" * 64)
    print("  AI EVALUATION RUBRIC — PRECISION MEDICINE")
    print("=" * 64)

    # ── Generate dataset ─────────────────────────────────────────────
    print("\n[1/3] Generating synthetic evaluation dataset (n=50)…")
    items_df, gt_df = generate_evaluation_dataset(n=50, seed=42)
    print(f"  Items        : {len(items_df)}")
    print(f"  Tier counts  : {items_df['quality_tier'].value_counts().to_dict()}")
    print(f"  Mean GT score: {gt_df['composite'].mean():.2f} ± {gt_df['composite'].std():.2f}")

    # ── Score with simulated automated scorer ────────────────────────
    print("\n[2/3] Scoring with automated rubric (simulated evaluator)…")
    scorer = RubricScorer()
    rng = np.random.default_rng(99)

    # Simulate automated scorer with noise (bias + random error)
    auto_items = []
    for _, row in gt_df.iterrows():
        auto_scores = {}
        for d in RUBRIC_DIMENSIONS:
            col = f"gt_{d.name.replace(' ', '_')}"
            gt_score = row[col]
            # Add noise: ±1 with 30% probability
            noise = rng.choice([-1, 0, 0, 0, 1])
            auto_scores[d.name] = int(np.clip(gt_score + noise, 1, 5))
        auto_items.append({"item_id": row["item_id"], "scores": auto_scores})

    scores_df = scorer.score_batch(auto_items)

    # Rename for calibration
    pred_df = scores_df.rename(columns={"composite": "composite"})
    pred_df_cal = pred_df[["item_id", "composite"]].copy()

    print(f"  Mean auto score: {scores_df['composite'].mean():.2f}")
    print(f"\n  Grade distribution:")
    print(scores_df["grade"].value_counts().sort_index().to_string())

    # ── Calibration ──────────────────────────────────────────────────
    print("\n[3/3] Calibration analysis (auto scorer vs expert GT)…")
    cal = calibration_analysis(gt_df, pred_df_cal, dimension="composite")
    print(f"  Pearson r    : {cal['pearson_r']:.4f}  (p={cal['p_value']:.4f})")
    print(f"  MAE          : {cal['mae']:.4f}")
    print(f"  Bias         : {cal['bias']:+.4f}")
    print(f"  QW Kappa     : {cal['kappa_qwt']:.4f}")

    # Print rubric summary
    print("\n  Rubric Dimensions:")
    print(f"  {'Dimension':<25} {'Weight':>7}  Scoring guide (score 5)")
    print("  " + "-" * 70)
    for d in RUBRIC_DIMENSIONS:
        print(f"  {d.name:<25} {d.weight:>6.0%}  {d.scoring_guide[5][:45]}…")

    # ── Visualisation ────────────────────────────────────────────────
    merged_for_plot = scores_df.merge(items_df[["item_id", "quality_tier"]], on="item_id")
    merged_for_plot = merged_for_plot.merge(
        gt_df[["item_id", "gt_composite"]], on="item_id"
    )
    plot_rubric_results(
        scores_df,
        merged_for_plot,
        out_path=os.path.join(out_dir, "evaluation_rubric.png"),
    )

    print("\nDone.")
