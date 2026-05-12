"""
analytics_dashboard.py — Interactive Plotly dashboard for preclinical assay data.

4-panel HTML dashboard: dose-response curves, activity heatmap,
selectivity scatter, FAIR compliance score. Falls back to matplotlib
static PNG if Plotly is unavailable.
"""

from __future__ import annotations

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42): np.random.seed(s)
    PALETTES = {"young": "#2196F3", "aged": "#F44336", "accent": "#4CAF50"}

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from schema_design import (
    create_database, populate_database,
    run_query, ASSAY_CATALOGUE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Hill equation / dose-response
# ──────────────────────────────────────────────────────────────────────────────

def hill_equation(conc: np.ndarray, ic50: float, hill: float,
                  top: float = 100.0, bottom: float = 0.0) -> np.ndarray:
    return bottom + (top - bottom) / (1 + (ic50 / conc) ** hill)


def fit_dose_response(
    concentrations: np.ndarray, responses: np.ndarray
) -> Tuple[float, float, float, float]:
    """Returns (ic50, hill_slope, top, bottom)."""
    try:
        p0 = [np.median(concentrations), 1.0, 100.0, 0.0]
        bounds = ([1e-3, 0.3, 80, -10], [1e6, 5.0, 120, 20])
        popt, _ = curve_fit(hill_equation, concentrations, responses,
                            p0=p0, bounds=bounds, maxfev=5000)
        return tuple(popt)
    except Exception:
        return (np.median(concentrations), 1.0, 100.0, 0.0)


def generate_dose_response_data(
    compounds: List[str], ic50_values: List[float], seed: int = 42
) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    concs = np.array([0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000])
    rows  = []
    for cpd, ic50 in zip(compounds, ic50_values):
        hill = rng.uniform(0.8, 1.8)
        for c in concs:
            resp = hill_equation(c, ic50, hill) + rng.normal(0, 3)
            resp = np.clip(resp, 0, 110)
            rows.append({"compound_id": cpd, "concentration_nM": c,
                         "response_pct": round(resp, 1), "ic50_nm": ic50,
                         "hill": hill})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def load_dashboard_data(seed: int = 42) -> Dict:
    conn  = create_database()
    stats = populate_database(conn, seed=seed)

    top10 = run_query(conn, "top10_cdk4_ic50")
    sel   = run_query(conn, "selectivity_ratio")
    cov   = run_query(conn, "assay_coverage")

    # Heatmap: compound × assay Z-score
    heatmap_q = """
        SELECT c.compound_id, r.assay_id,
               COALESCE(r.ic50_nm, r.value) AS value
        FROM   results r
        JOIN   compounds c ON r.compound_id = c.compound_id
        WHERE  r.qc_flag = 'pass'
          AND  r.assay_id IN ('ASY001','ASY002','ASY003')
    """
    hm_df = pd.read_sql_query(heatmap_q, conn)
    hm_pivot = hm_df.pivot_table(index="compound_id", columns="assay_id",
                                  values="value", aggfunc="min")
    # Log-transform + z-score
    hm_log = np.log10(hm_pivot.fillna(hm_pivot.max()) + 1)
    hm_z   = (hm_log - hm_log.mean()) / (hm_log.std() + 1e-9)

    # FAIR score by assay
    fair_q = """
        SELECT r.assay_id,
               ROUND(100.0 * SUM(CASE WHEN r.operator != 'unknown'
                                       AND r.run_date != '1900-01-01' THEN 1 ELSE 0 END)
                    / COUNT(*), 1) AS fair_score,
               COUNT(*) AS n_records
        FROM   results r
        GROUP  BY r.assay_id
    """
    fair_df = pd.read_sql_query(fair_q, conn)
    conn.close()

    # Top 5 compounds for dose-response
    top5 = top10.head(5)
    dr_df = generate_dose_response_data(
        top5["compound_id"].tolist(),
        top5["ic50_nm"].tolist(),
        seed=seed,
    )

    return {
        "top10": top10, "selectivity": sel, "coverage": cov,
        "heatmap_z": hm_z, "fair_df": fair_df, "dr_df": dr_df,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotly dashboard
# ──────────────────────────────────────────────────────────────────────────────

def build_plotly_dashboard(data: Dict, out_path: str = "figures/dashboard.html") -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Dose-Response Curves (CDK4 Biochemical)",
            "Compound Activity Heatmap (Z-score)",
            "Selectivity: CDK4 vs CDK2",
            "FAIR Compliance Score by Assay",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
    )

    # ── Panel 1: Dose-response curves ──────────────────────────────────────
    dr_df = data["dr_df"]
    concs_fine = np.logspace(-1, 4, 200)
    for i, cpd in enumerate(dr_df["compound_id"].unique()):
        sub   = dr_df[dr_df["compound_id"] == cpd]
        ic50  = sub["ic50_nm"].iloc[0]
        hill  = sub["hill"].iloc[0]
        curve = hill_equation(concs_fine, ic50, hill)

        fig.add_trace(go.Scatter(
            x=sub["concentration_nM"], y=sub["response_pct"],
            mode="markers", marker=dict(color=colors[i], size=5),
            name=cpd, legendgroup=cpd, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=concs_fine, y=curve,
            mode="lines", line=dict(color=colors[i], width=2),
            name=f"{cpd} fit", legendgroup=cpd, showlegend=False,
            hovertemplate=f"IC50={ic50:.1f} nM<extra></extra>",
        ), row=1, col=1)
        fig.add_vline(x=ic50, line_dash="dot",
                      line_color=colors[i], opacity=0.5, row=1, col=1)

    fig.update_xaxes(type="log", title_text="Concentration (nM)",
                     row=1, col=1)
    fig.update_yaxes(title_text="Response (%)", range=[-5, 110], row=1, col=1)

    # ── Panel 2: Heatmap ───────────────────────────────────────────────────
    hm = data["heatmap_z"]
    if not hm.empty:
        cpds    = hm.index.tolist()[:30]   # top 30 for readability
        assays  = hm.columns.tolist()
        z_vals  = hm.loc[cpds, assays].values

        asy_labels = {
            "ASY001": "CDK4 IC50",
            "ASY002": "CDK2 IC50",
            "ASY003": "MCF7 GI50",
        }

        fig.add_trace(go.Heatmap(
            z=z_vals,
            x=[asy_labels.get(a, a) for a in assays],
            y=cpds,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Z-score", x=1.0, len=0.45, y=0.78),
            showscale=True,
        ), row=1, col=2)

    # ── Panel 3: Selectivity scatter ───────────────────────────────────────
    sel_df = data["selectivity"]
    if not sel_df.empty:
        sel_df = sel_df.dropna(subset=["cdk4_ic50", "cdk2_ic50"])
        fig.add_trace(go.Scatter(
            x=sel_df["cdk4_ic50"],
            y=sel_df["cdk2_ic50"],
            mode="markers+text",
            text=sel_df["compound_id"],
            textposition="top center",
            textfont=dict(size=7),
            marker=dict(
                size=10,
                color=np.log10(sel_df["selectivity_ratio"] + 0.1),
                colorscale="Viridis",
                colorbar=dict(title="log₁₀(sel.)", x=0.48, len=0.45, y=0.24),
                showscale=True,
            ),
            hovertemplate="%{text}<br>CDK4=%{x:.1f} nM<br>CDK2=%{y:.1f} nM"
                          "<br>Sel=%{customdata:.1f}×<extra></extra>",
            customdata=sel_df["selectivity_ratio"],
            showlegend=False,
        ), row=2, col=1)

        # Diagonal (iso-selectivity line)
        lim = max(sel_df["cdk4_ic50"].max(), sel_df["cdk2_ic50"].max()) * 1.2
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                      line=dict(dash="dash", color="gray"), row=2, col=1)

        fig.update_xaxes(type="log", title_text="CDK4 IC50 (nM)", row=2, col=1)
        fig.update_yaxes(type="log", title_text="CDK2 IC50 (nM)", row=2, col=1)

    # ── Panel 4: FAIR score bar ─────────────────────────────────────────────
    fair_df = data["fair_df"]
    asy_name_map = {a[0]: a[1][:25] for a in ASSAY_CATALOGUE}
    fair_df["assay_name"] = fair_df["assay_id"].map(asy_name_map)

    bar_colors = ["#4CAF50" if s >= 85 else "#FF9800" if s >= 70 else "#F44336"
                  for s in fair_df["fair_score"]]

    fig.add_trace(go.Bar(
        x=fair_df["assay_name"],
        y=fair_df["fair_score"],
        marker_color=bar_colors,
        text=[f"{s:.0f}%" for s in fair_df["fair_score"]],
        textposition="outside",
        showlegend=False,
    ), row=2, col=2)
    fig.add_hline(y=85, line_dash="dash", line_color="green",
                  annotation_text="Target 85%", row=2, col=2)
    fig.update_yaxes(title_text="FAIR Score (%)", range=[0, 110], row=2, col=2)

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="Preclinical Research Data Platform — Analytics Dashboard",
            font=dict(size=16),
        ),
        height=850,
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#F5F5F5",
        legend=dict(x=1.02, y=0.99),
        font=dict(family="Arial", size=11),
    )

    fig.write_html(out_path)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib fallback
# ──────────────────────────────────────────────────────────────────────────────

def build_matplotlib_dashboard(
    data: Dict, out_path: str = "figures/dashboard_static.png"
) -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#FAFAFA")
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    # Panel 1: Dose-response
    ax = axes[0, 0]
    dr_df    = data["dr_df"]
    concs_fine = np.logspace(-1, 4, 200)
    for i, cpd in enumerate(dr_df["compound_id"].unique()):
        sub  = dr_df[dr_df["compound_id"] == cpd]
        ic50 = sub["ic50_nm"].iloc[0]
        hill = sub["hill"].iloc[0]
        curve = hill_equation(concs_fine, ic50, hill)
        ax.scatter(sub["concentration_nM"], sub["response_pct"],
                   color=colors[i], s=20, alpha=0.7)
        ax.plot(concs_fine, curve, color=colors[i], linewidth=1.5,
                label=f"{cpd} IC50={ic50:.0f} nM")
        ax.axvline(ic50, color=colors[i], linestyle=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3000)
    ax.set_ylim(-5, 115)
    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("Response (%)")
    ax.set_title("Dose-Response Curves (CDK4)", fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_facecolor("#F5F5F5")

    # Panel 2: Heatmap
    ax = axes[0, 1]
    hm = data["heatmap_z"]
    if not hm.empty:
        im = ax.imshow(hm.values[:25], cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(hm.columns)))
        ax.set_xticklabels(hm.columns, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(min(25, len(hm.index))))
        ax.set_yticklabels(hm.index[:25], fontsize=6)
        plt.colorbar(im, ax=ax, label="Z-score", fraction=0.046)
    ax.set_title("Activity Heatmap (Z-score)", fontweight="bold")

    # Panel 3: Selectivity scatter
    ax = axes[1, 0]
    sel_df = data["selectivity"].dropna(subset=["cdk4_ic50", "cdk2_ic50"])
    if not sel_df.empty:
        sc = ax.scatter(sel_df["cdk4_ic50"], sel_df["cdk2_ic50"],
                        c=np.log10(sel_df["selectivity_ratio"] + 0.1),
                        cmap="viridis", s=50, alpha=0.8)
        plt.colorbar(sc, ax=ax, label="log₁₀(selectivity)")
        lim = max(sel_df[["cdk4_ic50", "cdk2_ic50"]].max()) * 1.5
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("CDK4 IC50 (nM)")
        ax.set_ylabel("CDK2 IC50 (nM)")
    ax.set_title("CDK4 vs CDK2 Selectivity", fontweight="bold")
    ax.set_facecolor("#F5F5F5")

    # Panel 4: FAIR score
    ax = axes[1, 1]
    fair_df = data["fair_df"]
    bar_colors = ["#4CAF50" if s >= 85 else "#FF9800" if s >= 70 else "#F44336"
                  for s in fair_df["fair_score"]]
    ax.bar(range(len(fair_df)), fair_df["fair_score"], color=bar_colors, edgecolor="white")
    ax.axhline(85, color="green", linestyle="--", linewidth=1.2, label="Target 85%")
    ax.set_xticks(range(len(fair_df)))
    ax.set_xticklabels(fair_df["assay_id"], rotation=20)
    ax.set_ylabel("FAIR Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("FAIR Compliance by Assay", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F5F5F5")

    plt.suptitle("Preclinical Data Platform — Analytics Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    print("Loading dashboard data …")
    data = load_dashboard_data(seed=42)

    os.makedirs("figures", exist_ok=True)

    if HAS_PLOTLY:
        out = build_plotly_dashboard(data, "figures/dashboard.html")
        print(f"Interactive Plotly dashboard → {out}")
    else:
        print("Plotly not installed — building matplotlib fallback.")

    out_static = build_matplotlib_dashboard(data, "figures/dashboard_static.png")
    print(f"Static dashboard → {out_static}")

    top10 = data["top10"]
    print(f"\nTop 3 CDK4 hits:")
    print(top10[["compound_id", "name", "ic50_nm", "mw", "logp"]].head(3).to_string(index=False))
