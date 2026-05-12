"""
patient_stratification.py — Unsupervised Patient Subgroup Discovery
=====================================================================
Identifies clinically meaningful patient subgroups using:
  • Consensus clustering (K-Means ensemble for stability)
  • Optimal k selection: silhouette, gap statistic, elbow
  • Subgroup characterisation: differential feature profiles
  • Kaplan-Meier survival curves per subgroup (synthetic time-to-event)
  • UMAP-like dimensionality reduction (PCA-based proxy without umap-learn)
  • Heatmap of cluster feature profiles
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. SYNTHETIC PATIENT FEATURE MATRIX
# ---------------------------------------------------------------------------

def generate_patient_features(
    n: int = 500,
    n_clusters_true: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate a synthetic patient feature matrix with known cluster structure.

    Returns
    -------
    df         : DataFrame with clinical + lab features
    true_labels: ground-truth cluster assignments
    """
    rng = np.random.default_rng(seed)

    # Cluster centres in feature space
    cluster_profiles = {
        0: {  # Young, low-risk metabolic
            "age": (42, 8), "bmi": (24, 3), "glucose_mmolL": (5.2, 0.5),
            "hba1c_pct": (5.4, 0.3), "egfr": (92, 10), "ldl_mmolL": (2.6, 0.5),
            "systolic_bp": (118, 10), "crp_mgL": (1.5, 1.0), "n_meds": (1, 0.8),
        },
        1: {  # Middle-aged cardiometabolic
            "age": (58, 7), "bmi": (30, 4), "glucose_mmolL": (6.8, 1.0),
            "hba1c_pct": (7.1, 0.8), "egfr": (75, 12), "ldl_mmolL": (3.4, 0.7),
            "systolic_bp": (142, 14), "crp_mgL": (4.2, 2.5), "n_meds": (3, 1.2),
        },
        2: {  # Elderly, CKD-dominant
            "age": (72, 6), "bmi": (26, 3.5), "glucose_mmolL": (6.2, 0.9),
            "hba1c_pct": (6.5, 0.6), "egfr": (38, 10), "ldl_mmolL": (2.2, 0.6),
            "systolic_bp": (148, 16), "crp_mgL": (5.8, 3.0), "n_meds": (5, 1.5),
        },
        3: {  # High-complexity multimorbid
            "age": (65, 8), "bmi": (34, 5), "glucose_mmolL": (8.2, 1.5),
            "hba1c_pct": (8.8, 1.0), "egfr": (52, 15), "ldl_mmolL": (3.8, 0.8),
            "systolic_bp": (158, 18), "crp_mgL": (9.0, 4.0), "n_meds": (7, 2.0),
        },
    }

    k = n_clusters_true
    sizes = rng.multinomial(n, [1 / k] * k)
    rows = []
    labels = []

    for cluster_id, size in enumerate(sizes):
        profile = cluster_profiles[cluster_id]
        for _ in range(size):
            row = {}
            for feat, (mu, sigma) in profile.items():
                row[feat] = float(rng.normal(mu, sigma))
            row["cluster_true"] = cluster_id
            rows.append(row)
            labels.append(cluster_id)

    df = pd.DataFrame(rows)

    # Add time-to-event (survival): higher-risk clusters have shorter time
    hazard_by_cluster = {0: 0.02, 1: 0.06, 2: 0.12, 3: 0.20}
    times, events = [], []
    for _, row in df.iterrows():
        h = hazard_by_cluster[int(row["cluster_true"])]
        t = rng.exponential(1 / h)
        t = min(t, 60.0)  # max follow-up 60 months
        event = int(t < 60.0)
        times.append(round(t, 1))
        events.append(event)

    df["survival_time_months"] = times
    df["event_observed"] = events
    df = df.drop(columns=["cluster_true"])

    return df, np.array(labels)


# ---------------------------------------------------------------------------
# 2. OPTIMAL K SELECTION
# ---------------------------------------------------------------------------

def select_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
) -> Dict[str, object]:
    """
    Compute silhouette scores and inertia (elbow) for k selection.

    Returns dict with scores and recommended k.
    """
    silhouette_scores = []
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil)
        inertias.append(km.inertia_)

    best_k = list(k_range)[int(np.argmax(silhouette_scores))]

    return {
        "k_range":            list(k_range),
        "silhouette_scores":  silhouette_scores,
        "inertias":           inertias,
        "recommended_k":      best_k,
        "best_silhouette":    max(silhouette_scores),
    }


# ---------------------------------------------------------------------------
# 3. CONSENSUS CLUSTERING
# ---------------------------------------------------------------------------

def consensus_clustering(
    X_scaled: np.ndarray,
    k: int,
    n_iter: int = 50,
    subsample_rate: float = 0.8,
    seed: int = 42,
) -> np.ndarray:
    """
    Consensus clustering: ensemble of K-Means on subsampled data.
    Returns final cluster labels from hierarchical clustering of
    the consensus (co-occurrence) matrix.
    """
    rng = np.random.default_rng(seed)
    n = X_scaled.shape[0]
    co_occur = np.zeros((n, n), dtype=np.float32)
    counts   = np.zeros((n, n), dtype=np.float32)

    for _ in range(n_iter):
        idx = rng.choice(n, size=int(n * subsample_rate), replace=False)
        km = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(1e6)))
        sub_labels = km.fit_predict(X_scaled[idx])

        for i_pos, i in enumerate(idx):
            for j_pos, j in enumerate(idx):
                counts[i, j] += 1
                if sub_labels[i_pos] == sub_labels[j_pos]:
                    co_occur[i, j] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus_matrix = np.where(counts > 0, co_occur / counts, 0)

    # Final clustering on consensus matrix
    distance = 1 - consensus_matrix
    agg = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                  linkage="average")
    labels = agg.fit_predict(distance)

    return labels


# ---------------------------------------------------------------------------
# 4. KAPLAN-MEIER SURVIVAL ANALYSIS
# ---------------------------------------------------------------------------

def kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-parametric Kaplan-Meier estimator.

    Returns
    -------
    times    : unique event times
    survival : S(t) at each time
    """
    order = np.argsort(time)
    t = time[order]
    e = event[order]

    unique_times = np.unique(t[e == 1])
    survival = np.ones(len(unique_times) + 1)
    times_out = np.concatenate([[0], unique_times])
    n_at_risk = len(t)

    for i, ti in enumerate(unique_times):
        d_i = ((t == ti) & (e == 1)).sum()
        survival[i + 1] = survival[i] * (1 - d_i / max(n_at_risk, 1))
        n_at_risk -= (t == ti).sum()

    return times_out, survival


def logrank_test(
    t1: np.ndarray, e1: np.ndarray,
    t2: np.ndarray, e2: np.ndarray,
) -> float:
    """Log-rank test p-value between two survival curves."""
    all_times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    O1 = E1 = O2 = E2 = 0.0

    for t in all_times:
        n1 = (t1 >= t).sum()
        n2 = (t2 >= t).sum()
        d1 = ((t1 == t) & (e1 == 1)).sum()
        d2 = ((t2 == t) & (e2 == 1)).sum()
        n = n1 + n2
        d = d1 + d2
        if n < 2:
            continue
        E1 += d * n1 / n
        E2 += d * n2 / n
        O1 += d1
        O2 += d2

    if E1 == 0 or E2 == 0:
        return 1.0
    chi2 = (O1 - E1) ** 2 / E1 + (O2 - E2) ** 2 / E2
    return float(stats.chi2.sf(chi2, df=1))


# ---------------------------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------------------------

CLUSTER_COLORS = ["#1976d2", "#e53935", "#388e3c", "#f57c00", "#7b1fa2", "#00796b"]


def plot_stratification_dashboard(
    df: pd.DataFrame,
    labels: np.ndarray,
    true_labels: np.ndarray,
    k_results: Dict,
    out_path: str = "patient_stratification.png",
) -> None:
    """5-panel patient stratification figure."""
    feat_cols = [c for c in df.columns
                 if c not in ("survival_time_months", "event_observed")]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feat_cols])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle("Patient Stratification — Unsupervised Subgroup Discovery",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: K selection ──────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ks = k_results["k_range"]
    sil = k_results["silhouette_scores"]
    ax1.plot(ks, sil, "o-", color="#1976d2", lw=2)
    ax1.axvline(k_results["recommended_k"], color="red", lw=1.5,
                linestyle="--", label=f"k={k_results['recommended_k']}")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Optimal k Selection", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: PCA scatter coloured by consensus cluster ────────────
    ax2 = fig.add_subplot(2, 3, 2)
    k = len(np.unique(labels))
    for c in range(k):
        mask = labels == c
        ax2.scatter(coords[mask, 0], coords[mask, 1],
                    c=CLUSTER_COLORS[c], s=20, alpha=0.7, label=f"Cluster {c+1}")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title("Patient Subgroups (PCA)", fontweight="bold")
    ax2.legend(fontsize=8, markerscale=1.5)

    # ── Panel 3: Kaplan-Meier curves ──────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    t_all = df["survival_time_months"].values
    e_all = df["event_observed"].values
    km_curves = {}
    for c in range(k):
        mask = labels == c
        t_c, s_c = kaplan_meier(t_all[mask], e_all[mask])
        km_curves[c] = (t_c, s_c)
        ax3.step(t_c, s_c, where="post", lw=2, color=CLUSTER_COLORS[c],
                 label=f"Cluster {c+1} (n={mask.sum()})")

    # Log-rank p-value (cluster 0 vs worst cluster)
    worst = max(range(k), key=lambda c: e_all[labels == c].mean())
    if worst != 0:
        p = logrank_test(t_all[labels == 0], e_all[labels == 0],
                         t_all[labels == worst], e_all[labels == worst])
        ax3.text(0.6, 0.85, f"log-rank p={p:.3e}", transform=ax3.transAxes, fontsize=8)

    ax3.set_xlabel("Time (months)")
    ax3.set_ylabel("Survival Probability")
    ax3.set_title("Kaplan-Meier Survival by Subgroup", fontweight="bold")
    ax3.legend(fontsize=7)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Feature profile heatmap ─────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    profile_rows = []
    for c in range(k):
        mask = labels == c
        profile_rows.append(X_scaled[mask].mean(axis=0))
    profile_mat = np.vstack(profile_rows)

    im = ax4.imshow(profile_mat, aspect="auto", cmap="RdBu_r",
                    vmin=-2, vmax=2)
    ax4.set_xticks(range(len(feat_cols)))
    ax4.set_xticklabels(feat_cols, rotation=45, ha="right", fontsize=7)
    ax4.set_yticks(range(k))
    ax4.set_yticklabels([f"Cluster {c+1}" for c in range(k)], fontsize=9)
    ax4.set_title("Cluster Feature Profiles (z-score)", fontweight="bold")
    plt.colorbar(im, ax=ax4, fraction=0.04)

    # ── Panel 5: Cluster size + event rate ───────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    sizes = [int((labels == c).sum()) for c in range(k)]
    event_rates = [float(e_all[labels == c].mean()) for c in range(k)]
    x = np.arange(k)
    width = 0.4
    bars = ax5.bar(x - width / 2, sizes, width, label="Cluster size",
                   color=[CLUSTER_COLORS[c] for c in range(k)], alpha=0.85)
    ax5r = ax5.twinx()
    ax5r.bar(x + width / 2, [r * 100 for r in event_rates], width,
             label="Event rate (%)", color="grey", alpha=0.5)
    ax5r.set_ylabel("Event Rate (%)")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"Cluster {c+1}" for c in range(k)])
    ax5.set_ylabel("Cluster Size (n)")
    ax5.set_title("Cluster Sizes & Event Rates", fontweight="bold")
    ax5.legend(loc="upper left", fontsize=8)
    ax5r.legend(loc="upper right", fontsize=8)

    # ── Panel 6: Silhouette width plot ───────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    from sklearn.metrics import silhouette_samples
    sil_samples = silhouette_samples(X_scaled, labels)
    y_lower = 10
    for c in range(k):
        c_sil = np.sort(sil_samples[labels == c])
        size_c = c_sil.shape[0]
        y_upper = y_lower + size_c
        ax6.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                          alpha=0.75, color=CLUSTER_COLORS[c])
        ax6.text(-0.05, y_lower + size_c / 2, f"C{c+1}", fontsize=8)
        y_lower = y_upper + 10
    ax6.axvline(sil_samples.mean(), color="red", linestyle="--", lw=1.5,
                label=f"Mean = {sil_samples.mean():.3f}")
    ax6.set_xlabel("Silhouette Coefficient")
    ax6.set_ylabel("Cluster")
    ax6.set_title("Silhouette Width Plot", fontweight="bold")
    ax6.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Stratification dashboard → {out_path}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(__file__) or "."

    print("=" * 64)
    print("  PATIENT STRATIFICATION — SUBGROUP DISCOVERY")
    print("=" * 64)

    # ── Generate data ────────────────────────────────────────────────
    print("\n[1/4] Generating patient feature matrix (n=500, k_true=4)…")
    df, true_labels = generate_patient_features(n=500, n_clusters_true=4, seed=42)
    feat_cols = [c for c in df.columns
                 if c not in ("survival_time_months", "event_observed")]
    print(f"  Features    : {len(feat_cols)}")
    print(f"  Event rate  : {df['event_observed'].mean():.1%}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feat_cols])

    # ── Optimal k ───────────────────────────────────────────────────
    print("\n[2/4] Selecting optimal number of clusters…")
    k_results = select_optimal_k(X_scaled, k_range=range(2, 7))
    print(f"  Silhouette scores : {dict(zip(k_results['k_range'], [f'{s:.3f}' for s in k_results['silhouette_scores']]))}")
    print(f"  Recommended k     : {k_results['recommended_k']}  "
          f"(silhouette={k_results['best_silhouette']:.3f})")

    # ── Consensus clustering ─────────────────────────────────────────
    k = k_results["recommended_k"]
    print(f"\n[3/4] Consensus clustering (k={k}, 50 iterations)…")
    labels = consensus_clustering(X_scaled, k=k, n_iter=50, seed=42)
    sil_final = silhouette_score(X_scaled, labels)
    print(f"  Final silhouette : {sil_final:.4f}")

    # Subgroup characterisation
    print("\n  Subgroup profiles:")
    df_labeled = df[feat_cols].copy()
    df_labeled["cluster"] = labels
    df_labeled["survival_time"] = df["survival_time_months"].values
    df_labeled["event"] = df["event_observed"].values
    profile = df_labeled.groupby("cluster")[feat_cols].mean().round(2)
    print(profile.T.to_string())

    # Survival summary
    print("\n  Survival by cluster:")
    t_all = df["survival_time_months"].values
    e_all = df["event_observed"].values
    for c in sorted(np.unique(labels)):
        mask = labels == c
        er = e_all[mask].mean()
        med_t = np.median(t_all[mask])
        print(f"    Cluster {c+1}: n={mask.sum()}, event_rate={er:.1%}, median_surv={med_t:.1f}mo")

    # ── Visualisation ────────────────────────────────────────────────
    print("\n[4/4] Generating stratification dashboard…")
    plot_stratification_dashboard(
        df, labels, true_labels, k_results,
        out_path=os.path.join(out_dir, "patient_stratification.png"),
    )

    print("\nDone.")
