"""
annotation_pipeline.py — Multi-annotator simulation, inter-annotator agreement,
adjudication workflow, and quality drift detection for Project 15.
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
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42):
        np.random.seed(s)
    PALETTES = {"young": "#2196F3", "aged": "#F44336", "accent": "#4CAF50"}

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

ANNOTATION_SCHEMA = {
    "factual_accuracy":  {"type": "ordinal",  "scale": [1, 2, 3, 4, 5]},
    "clinical_safety":   {"type": "ordinal",  "scale": [1, 2, 3, 4, 5]},
    "completeness":      {"type": "ordinal",  "scale": [1, 2, 3, 4, 5]},
    "hallucination_flag":{"type": "binary",   "scale": [0, 1]},
    "requires_review":   {"type": "binary",   "scale": [0, 1]},
}

ANNOTATOR_PROFILES = {
    "Generalist": {
        "bias": 0.1,       # tendency to score higher
        "noise": 0.6,      # ordinal noise std
        "flag_sens": 0.70, # sensitivity for binary flags
        "flag_spec": 0.80,
    },
    "Domain Expert": {
        "bias": -0.1,
        "noise": 0.35,
        "flag_sens": 0.85,
        "flag_spec": 0.90,
    },
    "Clinical Specialist": {
        "bias": 0.0,
        "noise": 0.45,
        "flag_sens": 0.80,
        "flag_spec": 0.88,
    },
}


@dataclass
class AnnotationTask:
    item_id: str
    content: str
    true_scores: Dict[str, float]   # ground-truth ordinal + binary labels
    topic: str
    batch_id: int


@dataclass
class AnnotatorResponse:
    item_id: str
    annotator: str
    scores: Dict[str, float]
    confidence: float   # self-reported 0-1


@dataclass
class AgreementResult:
    dimension: str
    cohen_kappa: Optional[float]         # binary/ordinal pairwise
    krippendorff_alpha: float
    pairwise_matrix: np.ndarray          # annotator × annotator
    annotator_labels: List[str]
    flagged_items: List[str]             # low-agreement items


@dataclass
class AdjudicationRecord:
    item_id: str
    dimension: str
    annotator_scores: Dict[str, float]
    adjudicated_score: float
    reason: str
    requires_expert: bool


# ──────────────────────────────────────────────────────────────────────────────
# Dataset generator
# ──────────────────────────────────────────────────────────────────────────────

TOPICS = [
    "variant_interpretation",
    "drug_gene_interaction",
    "biomarker_reference",
    "clinical_guideline",
    "phenotype_association",
]


def generate_annotation_tasks(
    n: int = 100, n_batches: int = 10, seed: int = 42
) -> List[AnnotationTask]:
    rng = np.random.default_rng(seed)
    tasks = []
    items_per_batch = n // n_batches

    for batch_id in range(n_batches):
        for j in range(items_per_batch):
            item_id = f"item_{batch_id:02d}_{j:03d}"
            topic = rng.choice(TOPICS)
            # Ground-truth scores with batch-level quality trend (slight decline)
            quality = max(1.5, 4.0 - batch_id * 0.05 + rng.normal(0, 0.3))
            true_scores = {
                "factual_accuracy":  float(np.clip(round(quality + rng.normal(0, 0.4)), 1, 5)),
                "clinical_safety":   float(np.clip(round(quality + rng.normal(0, 0.5)), 1, 5)),
                "completeness":      float(np.clip(round(quality + rng.normal(0, 0.6)), 1, 5)),
                "hallucination_flag": float(1 if quality < 3.0 else 0),
                "requires_review":   float(1 if quality < 2.5 else 0),
            }
            tasks.append(AnnotationTask(
                item_id=item_id,
                content=f"AI response about {topic.replace('_', ' ')} (batch {batch_id})",
                true_scores=true_scores,
                topic=topic,
                batch_id=batch_id,
            ))
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# Annotator simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_annotator(
    tasks: List[AnnotationTask],
    annotator_name: str,
    seed: int = 0,
) -> List[AnnotatorResponse]:
    rng = np.random.default_rng(seed)
    profile = ANNOTATOR_PROFILES[annotator_name]
    responses = []

    for task in tasks:
        scores: Dict[str, float] = {}
        for dim, meta in ANNOTATION_SCHEMA.items():
            true_val = task.true_scores[dim]
            if meta["type"] == "ordinal":
                noisy = true_val + profile["bias"] + rng.normal(0, profile["noise"])
                scores[dim] = float(np.clip(round(noisy), 1, 5))
            else:
                # Binary: simulate sensitivity/specificity
                if true_val == 1:
                    scores[dim] = float(1 if rng.random() < profile["flag_sens"] else 0)
                else:
                    scores[dim] = float(1 if rng.random() > profile["flag_spec"] else 0)

        confidence = float(np.clip(rng.normal(0.75, 0.12), 0.3, 1.0))
        responses.append(AnnotatorResponse(
            item_id=task.item_id,
            annotator=annotator_name,
            scores=scores,
            confidence=confidence,
        ))

    return responses


def collect_all_annotations(
    tasks: List[AnnotationTask], seed: int = 42
) -> pd.DataFrame:
    all_rows = []
    for i, name in enumerate(ANNOTATOR_PROFILES):
        responses = simulate_annotator(tasks, name, seed=seed + i * 17)
        for r in responses:
            row = {"item_id": r.item_id, "annotator": r.annotator,
                   "confidence": r.confidence, **r.scores}
            all_rows.append(row)
    return pd.DataFrame(all_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Agreement metrics
# ──────────────────────────────────────────────────────────────────────────────

def cohen_kappa(a: np.ndarray, b: np.ndarray, ordinal: bool = False) -> float:
    """Unweighted (binary) or linear-weighted (ordinal) Cohen's κ."""
    unique = sorted(set(a) | set(b))
    k = len(unique)
    if k < 2:
        return 1.0

    idx = {v: i for i, v in enumerate(unique)}
    conf = np.zeros((k, k))
    for ai, bi in zip(a, b):
        conf[idx[ai], idx[bi]] += 1

    n = conf.sum()
    if n == 0:
        return 0.0

    po = np.trace(conf) / n

    if ordinal and k > 2:
        # Linear weights
        weights = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                weights[i, j] = 1 - abs(unique[i] - unique[j]) / (max(unique) - min(unique) + 1e-9)
        po = (weights * conf / n).sum()
        row_marg = conf.sum(axis=1) / n
        col_marg = conf.sum(axis=0) / n
        pe = (weights * np.outer(row_marg, col_marg)).sum()
    else:
        row_marg = conf.sum(axis=1) / n
        col_marg = conf.sum(axis=0) / n
        pe = (row_marg * col_marg).sum()

    if abs(1 - pe) < 1e-9:
        return 1.0
    return float((po - pe) / (1 - pe))


def krippendorff_alpha(data_matrix: np.ndarray, level: str = "ordinal") -> float:
    """
    Krippendorff's alpha for reliability.

    data_matrix: shape (n_annotators, n_items) — NaN = missing
    level: 'nominal', 'ordinal', 'interval'
    """
    m, n = data_matrix.shape
    # Observed disagreement
    do = 0.0
    n_pairs = 0
    for j in range(n):
        col = data_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        for a in range(len(valid)):
            for b in range(a + 1, len(valid)):
                diff = valid[a] - valid[b]
                if level == "ordinal":
                    do += diff ** 2
                elif level == "interval":
                    do += diff ** 2
                else:
                    do += float(valid[a] != valid[b])
                n_pairs += 1

    if n_pairs == 0:
        return 1.0

    # Expected disagreement
    all_vals = data_matrix[~np.isnan(data_matrix)]
    de = 0.0
    total = len(all_vals)
    for a in range(total):
        for b in range(a + 1, total):
            diff = all_vals[a] - all_vals[b]
            if level == "ordinal":
                de += diff ** 2
            elif level == "interval":
                de += diff ** 2
            else:
                de += float(all_vals[a] != all_vals[b])

    n_de_pairs = total * (total - 1) / 2
    if n_de_pairs == 0 or de == 0:
        return 1.0

    # Scale observed to match expected denominator
    do_scaled = do / n_pairs * n_de_pairs
    return float(1 - do_scaled / de)


def compute_agreement(
    ann_df: pd.DataFrame, dimension: str, low_agreement_threshold: float = 0.60
) -> AgreementResult:
    annotators = list(ANNOTATOR_PROFILES.keys())
    items = ann_df["item_id"].unique()
    n_ann = len(annotators)

    # Build annotator × item matrix
    pivot = ann_df.pivot_table(index="annotator", columns="item_id",
                                values=dimension, aggfunc="first")
    mat = pivot.reindex(index=annotators, columns=items).values.astype(float)

    schema_type = ANNOTATION_SCHEMA[dimension]["type"]

    # Pairwise Cohen's κ
    pairwise = np.eye(n_ann)
    for i in range(n_ann):
        for j in range(i + 1, n_ann):
            a_vals = mat[i]
            b_vals = mat[j]
            mask = ~(np.isnan(a_vals) | np.isnan(b_vals))
            if mask.sum() < 5:
                k = np.nan
            else:
                k = cohen_kappa(a_vals[mask], b_vals[mask],
                                ordinal=(schema_type == "ordinal"))
            pairwise[i, j] = k
            pairwise[j, i] = k

    level = "ordinal" if schema_type == "ordinal" else "nominal"
    alpha = krippendorff_alpha(mat, level=level)

    # Flag low-agreement items (std across annotators > 1.5 for ordinal)
    flagged = []
    for item_id in items:
        col = pivot[item_id].values.astype(float)
        col = col[~np.isnan(col)]
        if len(col) >= 2:
            if schema_type == "ordinal" and np.std(col) > 1.5:
                flagged.append(item_id)
            elif schema_type == "binary" and len(set(col)) > 1:
                flagged.append(item_id)

    mean_kappa = float(np.nanmean(pairwise[np.triu_indices(n_ann, k=1)]))

    return AgreementResult(
        dimension=dimension,
        cohen_kappa=mean_kappa,
        krippendorff_alpha=alpha,
        pairwise_matrix=pairwise,
        annotator_labels=annotators,
        flagged_items=flagged,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Adjudication workflow
# ──────────────────────────────────────────────────────────────────────────────

def adjudicate_low_agreement(
    ann_df: pd.DataFrame,
    agreement: AgreementResult,
    dimension: str,
) -> List[AdjudicationRecord]:
    records = []
    schema_type = ANNOTATION_SCHEMA[dimension]["type"]

    for item_id in agreement.flagged_items:
        item_ann = ann_df[ann_df["item_id"] == item_id]
        ann_scores: Dict[str, float] = {}
        for _, row in item_ann.iterrows():
            ann_scores[row["annotator"]] = float(row[dimension])

        vals = list(ann_scores.values())
        if schema_type == "ordinal":
            # Weighted average by annotator expertise (Domain Expert > others)
            weights = {"Generalist": 1.0, "Domain Expert": 2.0, "Clinical Specialist": 1.5}
            w_sum = sum(weights[a] for a in ann_scores)
            adj_score = sum(ann_scores[a] * weights[a] for a in ann_scores) / w_sum
            reason = f"Weighted expert adjudication (range {min(vals):.0f}–{max(vals):.0f})"
        else:
            # Majority vote; tie → conservative (flag = 1)
            pos = sum(1 for v in vals if v == 1)
            adj_score = float(1 if pos >= len(vals) / 2 else 0)
            reason = f"Majority vote ({pos}/{len(vals)} flagged)"

        requires_expert = (schema_type == "ordinal" and max(vals) - min(vals) > 2) or \
                          (schema_type == "binary" and len(set(vals)) > 1)

        records.append(AdjudicationRecord(
            item_id=item_id,
            dimension=dimension,
            annotator_scores=ann_scores,
            adjudicated_score=round(adj_score, 2),
            reason=reason,
            requires_expert=requires_expert,
        ))

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Batch drift detection
# ──────────────────────────────────────────────────────────────────────────────

def compute_batch_agreement(
    ann_df: pd.DataFrame,
    tasks: List[AnnotationTask],
    dimension: str = "factual_accuracy",
) -> np.ndarray:
    task_lookup = {t.item_id: t.batch_id for t in tasks}
    ann_df = ann_df.copy()
    ann_df["batch_id"] = ann_df["item_id"].map(task_lookup)

    batch_ids = sorted(ann_df["batch_id"].dropna().unique())
    alphas = []
    for bid in batch_ids:
        batch_df = ann_df[ann_df["batch_id"] == bid]
        pivot = batch_df.pivot_table(index="annotator", columns="item_id",
                                      values=dimension, aggfunc="first")
        mat = pivot.values.astype(float)
        if mat.shape[1] < 3:
            alphas.append(np.nan)
            continue
        level = "ordinal" if ANNOTATION_SCHEMA[dimension]["type"] == "ordinal" else "nominal"
        alphas.append(krippendorff_alpha(mat, level=level))

    return np.array(alphas)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper (called by quality_dashboard)
# ──────────────────────────────────────────────────────────────────────────────

def run_annotation_pipeline(n_items: int = 100, seed: int = 42) -> dict:
    tasks = generate_annotation_tasks(n=n_items, seed=seed)
    ann_df = collect_all_annotations(tasks, seed=seed)

    # Agreement on primary ordinal dimension
    agr = compute_agreement(ann_df, "factual_accuracy")
    batch_alpha = compute_batch_agreement(ann_df, tasks, dimension="factual_accuracy")

    return {
        "kappa_matrix": agr.pairwise_matrix,
        "batch_alpha": batch_alpha,
        "n_annotators": len(ANNOTATOR_PROFILES),
        "annotator_labels": list(ANNOTATOR_PROFILES.keys()),
        "flagged_items": agr.flagged_items,
        "krippendorff_alpha": agr.krippendorff_alpha,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_annotation_results(
    ann_df: pd.DataFrame,
    tasks: List[AnnotationTask],
    out_path: str = "figures/annotation_results.png",
) -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#FAFAFA")

    annotators = list(ANNOTATOR_PROFILES.keys())
    colors = [PALETTES.get("young", "#2196F3"),
              PALETTES.get("aged", "#F44336"),
              PALETTES.get("accent", "#4CAF50")]

    # Panel 1: Score distributions per annotator
    ax = axes[0]
    for i, ann in enumerate(annotators):
        vals = ann_df[ann_df["annotator"] == ann]["factual_accuracy"].values
        ax.hist(vals, bins=np.arange(0.5, 6, 1), alpha=0.5, color=colors[i],
                label=ann, edgecolor="white")
    ax.set_xlabel("Factual Accuracy Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distributions by Annotator")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F5F5F5")

    # Panel 2: Pairwise Cohen's κ heatmap
    ax = axes[1]
    agr = compute_agreement(ann_df, "factual_accuracy")
    mat = agr.pairwise_matrix
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(annotators)))
    ax.set_yticks(range(len(annotators)))
    ax.set_xticklabels([a.split()[0] for a in annotators], rotation=20, ha="right")
    ax.set_yticklabels([a.split()[0] for a in annotators])
    for i in range(len(annotators)):
        for j in range(len(annotators)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if mat[i, j] > 0.6 else "#333333")
    ax.set_title("Pairwise Cohen's κ")

    # Panel 3: Krippendorff α drift
    ax = axes[2]
    batch_alpha = compute_batch_agreement(ann_df, tasks, "factual_accuracy")
    batches = np.arange(1, len(batch_alpha) + 1)
    ax.plot(batches, batch_alpha, color=PALETTES.get("young", "#2196F3"),
            marker="o", linewidth=2)
    ax.axhline(0.60, linestyle="--", color="#333333", linewidth=1)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Krippendorff's α")
    ax.set_title("Agreement Drift Over Batches")
    ax.set_ylim(0, 1.05)
    ax.set_facecolor("#F5F5F5")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    print("Generating annotation tasks …")
    tasks = generate_annotation_tasks(n=100, n_batches=10, seed=42)
    print(f"  {len(tasks)} tasks across {len(set(t.batch_id for t in tasks))} batches")

    print("Simulating annotators …")
    ann_df = collect_all_annotations(tasks, seed=42)
    print(f"  {len(ann_df)} annotation responses")

    print("\nAgreement results:")
    for dim in ANNOTATION_SCHEMA:
        agr = compute_agreement(ann_df, dim)
        print(f"  {dim:<22}  κ={agr.cohen_kappa:.3f}  α={agr.krippendorff_alpha:.3f}"
              f"  flagged={len(agr.flagged_items)}")

    print("\nAdjudication (factual_accuracy) …")
    agr_main = compute_agreement(ann_df, "factual_accuracy")
    adj_records = adjudicate_low_agreement(ann_df, agr_main, "factual_accuracy")
    print(f"  {len(adj_records)} items adjudicated, "
          f"{sum(r.requires_expert for r in adj_records)} require expert review")

    os.makedirs("figures", exist_ok=True)
    img_path = plot_annotation_results(ann_df, tasks, out_path="figures/annotation_results.png")
    print(f"\nPlot saved → {img_path}")
