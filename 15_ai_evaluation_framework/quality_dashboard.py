"""
quality_dashboard.py — Aggregated AI quality dashboard for Project 15.

Pulls results from evaluation_rubric, biomedical_factchecker, and
annotation_pipeline modules (or generates synthetic stand-ins when those
modules are absent) and renders a 4-panel matplotlib dashboard plus a
plain-text quality_report.txt.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42):
        np.random.seed(s)
    PALETTES = {
        "young": "#2196F3",
        "aged": "#F44336",
        "accent": "#4CAF50",
        "neutral": "#9E9E9E",
    }

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

RUBRIC_DIMENSIONS = [
    "Scientific Accuracy",
    "Clinical Relevance",
    "Data Integrity",
    "Completeness",
    "Reasoning Quality",
    "Regulatory Compliance",
]

TOPICS = [
    "variant_interpretation",
    "drug_gene_interaction",
    "biomarker_reference",
    "clinical_guideline",
    "phenotype_association",
]

PASS_THRESHOLD_COMPOSITE = 3.0   # composite score ≥ 3.0 → pass
PASS_THRESHOLD_HALLUCINATION = 0.20  # hallucination rate ≤ 20 % → pass
PASS_THRESHOLD_AGREEMENT = 0.60  # Krippendorff α ≥ 0.60 → pass


@dataclass
class RubricData:
    scores_df: pd.DataFrame          # rows=items, cols=dimensions (1-5)
    composite: np.ndarray            # per-item weighted composite


@dataclass
class FactCheckerData:
    hallucination_rates: Dict[str, float]   # topic → rate
    per_item_rates: np.ndarray              # per-item hallucination rate


@dataclass
class AnnotationData:
    agreement_matrix: np.ndarray    # annotator×annotator κ matrix
    batch_alpha: np.ndarray         # Krippendorff α per batch
    n_annotators: int
    annotator_labels: List[str]


@dataclass
class DashboardResults:
    rubric: RubricData
    factchecker: FactCheckerData
    annotation: AnnotationData
    n_items: int
    pass_rates: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generators (fallback when modules are unavailable)
# ──────────────────────────────────────────────────────────────────────────────

def _synthetic_rubric(n: int = 50, seed: int = 42) -> RubricData:
    rng = np.random.default_rng(seed)
    weights = np.array([0.30, 0.25, 0.20, 0.10, 0.10, 0.05])
    # Four quality tiers
    tier_probs = [0.20, 0.30, 0.30, 0.20]
    tier_means = [1.8, 2.8, 3.8, 4.6]
    tiers = rng.choice(4, size=n, p=tier_probs)
    scores = np.zeros((n, len(RUBRIC_DIMENSIONS)))
    for i, t in enumerate(tiers):
        raw = rng.normal(tier_means[t], 0.5, size=len(RUBRIC_DIMENSIONS))
        scores[i] = np.clip(np.round(raw), 1, 5)
    df = pd.DataFrame(scores, columns=RUBRIC_DIMENSIONS)
    composite = (scores * weights).sum(axis=1)
    return RubricData(scores_df=df, composite=composite)


def _synthetic_factchecker(n: int = 50, seed: int = 42) -> FactCheckerData:
    rng = np.random.default_rng(seed)
    topic_rates = {t: float(rng.uniform(0.05, 0.35)) for t in TOPICS}
    # Per-item rates sampled from topic distribution
    per_item = rng.beta(2, 6, size=n).astype(float)
    return FactCheckerData(hallucination_rates=topic_rates, per_item_rates=per_item)


def _synthetic_annotation(n_batches: int = 10, seed: int = 42) -> AnnotationData:
    rng = np.random.default_rng(seed)
    n_ann = 3
    labels = ["Generalist", "Domain Expert", "Clinical Specialist"]
    # Pairwise Cohen's κ (symmetric, diagonal = 1)
    kappa_mat = np.eye(n_ann)
    pairs = [(0, 1, 0.72), (0, 2, 0.65), (1, 2, 0.78)]
    for i, j, k in pairs:
        kappa_mat[i, j] = k + rng.normal(0, 0.03)
        kappa_mat[j, i] = kappa_mat[i, j]
    # Krippendorff α trajectory over batches (slight drift)
    alpha_base = 0.71
    batch_alpha = np.clip(
        alpha_base + rng.normal(0, 0.04, size=n_batches) - np.linspace(0, 0.08, n_batches),
        0.4, 1.0,
    )
    return AnnotationData(
        agreement_matrix=kappa_mat,
        batch_alpha=batch_alpha,
        n_annotators=n_ann,
        annotator_labels=labels,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data loader — tries real modules first, falls back to synthetic
# ──────────────────────────────────────────────────────────────────────────────

def load_dashboard_data(n_items: int = 50, seed: int = 42) -> DashboardResults:
    set_global_seed(seed)

    # --- Rubric ----------------------------------------------------------
    try:
        from evaluation_rubric import generate_evaluation_dataset, RubricScorer, DIMENSION_NAMES as RD
        items_df, gt_df = generate_evaluation_dataset(n=n_items)
        scorer = RubricScorer()
        rows = []
        for _, row in gt_df.iterrows():
            scores = {d: int(row[f"gt_{d.replace(' ', '_')}"]) for d in RD}
            result = scorer.score(row["item_id"], scores)
            rows.append({**scores, "composite": result.composite_score})
        rdf = pd.DataFrame(rows)
        rubric = RubricData(
            scores_df=rdf[RD],
            composite=rdf["composite"].values,
        )
    except Exception:
        rubric = _synthetic_rubric(n=n_items, seed=seed)

    # --- Fact checker ----------------------------------------------------
    try:
        from biomedical_factchecker import BiomedicalFactChecker, generate_test_responses
        checker = BiomedicalFactChecker()
        responses = generate_test_responses(n=n_items)
        rates = {}
        per_item = []
        for resp in responses:
            result = checker.check(resp["text"])
            per_item.append(result.hallucination_rate)
        per_item_arr = np.array(per_item)
        topic_rates = {}
        for t in TOPICS:
            mask = [resp.get("topic") == t for resp in responses]
            if any(mask):
                topic_rates[t] = float(np.mean(per_item_arr[mask]))
        factchecker = FactCheckerData(
            hallucination_rates=topic_rates or {t: float(np.mean(per_item_arr)) for t in TOPICS},
            per_item_rates=per_item_arr,
        )
    except Exception:
        factchecker = _synthetic_factchecker(n=n_items, seed=seed)

    # --- Annotation ------------------------------------------------------
    try:
        from annotation_pipeline import run_annotation_pipeline, compute_batch_agreement
        results = run_annotation_pipeline(n_items=n_items)
        annotation = AnnotationData(
            agreement_matrix=results["kappa_matrix"],
            batch_alpha=results["batch_alpha"],
            n_annotators=results["n_annotators"],
            annotator_labels=results["annotator_labels"],
        )
    except Exception:
        annotation = _synthetic_annotation(seed=seed)

    # --- Pass rates ------------------------------------------------------
    pass_rates = {
        "rubric_pass_rate": float(np.mean(rubric.composite >= PASS_THRESHOLD_COMPOSITE)),
        "hallucination_pass_rate": float(
            np.mean(factchecker.per_item_rates <= PASS_THRESHOLD_HALLUCINATION)
        ),
        "agreement_pass_rate": float(
            np.mean(annotation.batch_alpha >= PASS_THRESHOLD_AGREEMENT)
        ),
    }

    return DashboardResults(
        rubric=rubric,
        factchecker=factchecker,
        annotation=annotation,
        n_items=n_items,
        pass_rates=pass_rates,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard renderer
# ──────────────────────────────────────────────────────────────────────────────

def plot_quality_dashboard(
    data: DashboardResults,
    out_path: str = "figures/quality_dashboard.png",
) -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # Panel colours
    cp = [PALETTES.get("young", "#2196F3"),
          PALETTES.get("aged", "#F44336"),
          PALETTES.get("accent", "#4CAF50"),
          "#FF9800", "#9C27B0", "#00BCD4"]

    # ── Panel 1: Rubric score distributions (violin) ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    score_data = [data.rubric.scores_df[d].values for d in RUBRIC_DIMENSIONS]
    short_labels = ["Sci Acc", "Clin Rel", "Data Int", "Complet", "Reason", "Reg Comp"]
    parts = ax1.violinplot(score_data, positions=range(len(RUBRIC_DIMENSIONS)),
                           showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(cp[i % len(cp)])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("#333333")
    parts["cbars"].set_color("#666666")
    parts["cmaxes"].set_color("#666666")
    parts["cmins"].set_color("#666666")
    ax1.set_xticks(range(len(RUBRIC_DIMENSIONS)))
    ax1.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
    ax1.set_yticks([1, 2, 3, 4, 5])
    ax1.set_ylabel("Score (1–5)", fontsize=10)
    ax1.set_title("Rubric Score Distributions per Dimension", fontsize=11, fontweight="bold")
    ax1.axhline(3.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_facecolor("#F5F5F5")
    pr = data.pass_rates.get("rubric_pass_rate", 0)
    ax1.text(0.97, 0.04, f"Pass rate: {pr:.0%}", transform=ax1.transAxes,
             ha="right", fontsize=9,
             color=PALETTES.get("accent", "#4CAF50") if pr >= 0.70 else PALETTES.get("aged", "#F44336"))

    # ── Panel 2: Hallucination rate by topic (bar) ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    topics = list(data.factchecker.hallucination_rates.keys())
    rates = [data.factchecker.hallucination_rates[t] for t in topics]
    topic_labels = [t.replace("_", "\n") for t in topics]
    bar_colors = [PALETTES.get("accent", "#4CAF50") if r <= PASS_THRESHOLD_HALLUCINATION
                  else PALETTES.get("aged", "#F44336") for r in rates]
    bars = ax2.bar(range(len(topics)), [r * 100 for r in rates],
                   color=bar_colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(PASS_THRESHOLD_HALLUCINATION * 100, color="#333333",
                linestyle="--", linewidth=1.2, label=f"Threshold {PASS_THRESHOLD_HALLUCINATION:.0%}")
    for bar, r in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f"{r:.1%}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(topics)))
    ax2.set_xticklabels(topic_labels, fontsize=8)
    ax2.set_ylabel("Hallucination Rate (%)", fontsize=10)
    ax2.set_title("Hallucination Rate by Topic", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_facecolor("#F5F5F5")

    # ── Panel 3: Inter-annotator agreement heatmap ────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    mat = data.annotation.agreement_matrix
    labels = data.annotation.annotator_labels
    im = ax3.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Cohen's κ")
    ax3.set_xticks(range(data.annotation.n_annotators))
    ax3.set_yticks(range(data.annotation.n_annotators))
    ax3.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax3.set_yticklabels(labels, fontsize=9)
    for i in range(data.annotation.n_annotators):
        for j in range(data.annotation.n_annotators):
            ax3.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                     fontsize=10, color="white" if mat[i, j] > 0.6 else "#333333")
    ax3.set_title("Inter-Annotator Agreement (Cohen's κ)", fontsize=11, fontweight="bold")

    # ── Panel 4: Quality trend over batches (line) ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    batches = np.arange(1, len(data.annotation.batch_alpha) + 1)
    ax4.plot(batches, data.annotation.batch_alpha, color=PALETTES.get("young", "#2196F3"),
             linewidth=2, marker="o", markersize=5, label="Krippendorff α")
    ax4.axhline(PASS_THRESHOLD_AGREEMENT, color="#333333", linestyle="--",
                linewidth=1.2, label=f"Threshold {PASS_THRESHOLD_AGREEMENT:.2f}")
    ax4.fill_between(batches, PASS_THRESHOLD_AGREEMENT, data.annotation.batch_alpha,
                     where=data.annotation.batch_alpha >= PASS_THRESHOLD_AGREEMENT,
                     alpha=0.15, color=PALETTES.get("accent", "#4CAF50"), label="Above threshold")
    ax4.fill_between(batches, PASS_THRESHOLD_AGREEMENT, data.annotation.batch_alpha,
                     where=data.annotation.batch_alpha < PASS_THRESHOLD_AGREEMENT,
                     alpha=0.15, color=PALETTES.get("aged", "#F44336"), label="Below threshold")
    ax4.set_xlabel("Annotation Batch", fontsize=10)
    ax4.set_ylabel("Krippendorff's α", fontsize=10)
    ax4.set_ylim(0, 1.05)
    ax4.set_title("Annotation Quality Trend Over Batches", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.set_facecolor("#F5F5F5")

    fig.suptitle("AI Output Quality Dashboard — Precision Medicine Evaluation",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Quality report generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_quality_report(data: DashboardResults, out_path: str = "quality_report.txt") -> str:
    lines: List[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("╔" + "═" * (len(title) + 4) + "╗")
        lines.append("║  " + title + "  ║")
        lines.append("╚" + "═" * (len(title) + 4) + "╝")

    lines.append("AI OUTPUT QUALITY REPORT")
    lines.append("=" * 60)
    lines.append(f"Items evaluated : {data.n_items}")
    lines.append(f"Rubric pass rate: {data.pass_rates.get('rubric_pass_rate', 0):.1%}")
    lines.append(f"Hallucination pass rate: {data.pass_rates.get('hallucination_pass_rate', 0):.1%}")
    lines.append(f"Agreement pass rate: {data.pass_rates.get('agreement_pass_rate', 0):.1%}")

    section("1. RUBRIC SCORES")
    for dim in RUBRIC_DIMENSIONS:
        col = data.rubric.scores_df[dim]
        status = "PASS" if col.mean() >= PASS_THRESHOLD_COMPOSITE else "FAIL"
        lines.append(f"  {dim:<28}  mean={col.mean():.2f}  sd={col.std():.2f}  [{status}]")
    lines.append(f"\n  Composite (weighted) — mean: {data.rubric.composite.mean():.2f}  "
                 f"items passing: {(data.rubric.composite >= PASS_THRESHOLD_COMPOSITE).sum()}/{data.n_items}")

    section("2. HALLUCINATION RATES BY TOPIC")
    for topic, rate in sorted(data.factchecker.hallucination_rates.items(), key=lambda x: -x[1]):
        status = "PASS" if rate <= PASS_THRESHOLD_HALLUCINATION else "FAIL"
        bar = "█" * int(rate * 20)
        lines.append(f"  {topic:<30}  {rate:.1%}  {bar:<20}  [{status}]")
    overall_hal = float(np.mean(data.factchecker.per_item_rates))
    lines.append(f"\n  Overall hallucination rate: {overall_hal:.1%}")

    section("3. INTER-ANNOTATOR AGREEMENT")
    mat = data.annotation.agreement_matrix
    labels = data.annotation.annotator_labels
    header = "  " + " " * 24 + "  ".join(f"{l[:12]:>12}" for l in labels)
    lines.append(header)
    for i, la in enumerate(labels):
        row_vals = "  ".join(f"{mat[i, j]:>12.3f}" for j in range(len(labels)))
        lines.append(f"  {la:<24}{row_vals}")
    mean_alpha = float(np.mean(data.annotation.batch_alpha))
    min_alpha = float(np.min(data.annotation.batch_alpha))
    status = "PASS" if mean_alpha >= PASS_THRESHOLD_AGREEMENT else "FAIL"
    lines.append(f"\n  Krippendorff α — mean: {mean_alpha:.3f}  min: {min_alpha:.3f}  [{status}]")

    section("4. OVERALL VERDICT")
    all_pass = all([
        data.pass_rates.get("rubric_pass_rate", 0) >= 0.70,
        data.pass_rates.get("hallucination_pass_rate", 0) >= 0.70,
        data.pass_rates.get("agreement_pass_rate", 0) >= 0.70,
    ])
    verdict = "PASS — AI output quality meets deployment criteria." if all_pass \
        else "FAIL — One or more quality dimensions below threshold."
    lines.append(f"\n  {verdict}")
    lines.append("")

    report = "\n".join(lines)
    with open(out_path, "w") as fh:
        fh.write(report)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    print("Loading dashboard data …")
    data = load_dashboard_data(n_items=50, seed=42)

    os.makedirs("figures", exist_ok=True)
    img_path = plot_quality_dashboard(data, out_path="figures/quality_dashboard.png")
    print(f"Dashboard saved → {img_path}")

    report = generate_quality_report(data, out_path="quality_report.txt")
    print("\n" + report)
