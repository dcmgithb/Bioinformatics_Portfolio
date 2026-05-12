"""
Network Pharmacology of Geroprotectors
========================================
Constructs a drug–target–pathway network to reveal the polypharmacology
landscape of geroprotective compounds and identify key aging pathway hubs.

Approach:
  1. Build drug–target bipartite network from curated geroprotector targets
  2. Overlay pathway annotations (mTOR, Senescence, Inflammation, DNA Repair)
  3. Compute network metrics (degree, betweenness, eigenvector centrality)
  4. Identify polypharmacology: drugs hitting ≥3 aging-relevant targets
  5. Target vulnerability: targets hit by ≥3 known geroprotectors
  6. Chemical similarity-weighted network (structure-activity landscape)
  7. Export Cytoscape-ready GraphML + interactive plotly network

Python : >= 3.10 | RDKit >= 2023.09
"""

from __future__ import annotations

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter
from rdkit import Chem

from utils.chem_utils import (
    validate_smiles_column, fingerprint_matrix, tanimoto_matrix, mpo_score
)

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── 1. Drug–Target Interaction Database ───────────────────────────────────────

print("═" * 60)
print("  NETWORK PHARMACOLOGY — AGING DRUG-TARGET LANDSCAPE")
print("═" * 60)

df_drugs = pd.read_csv("data/geroprotectors.csv")
df_drugs = validate_smiles_column(df_drugs, smiles_col="smiles")

# Parse multi-target annotations
def parse_targets(target_str: str) -> list[str]:
    if pd.isna(target_str): return []
    return [t.strip() for t in str(target_str).split(";") if t.strip()]

df_drugs["target_list"] = df_drugs["target"].apply(parse_targets)

# Build drug-target pairs
dti_rows = []
for _, row in df_drugs.iterrows():
    for tgt in row["target_list"]:
        dti_rows.append({
            "drug"          : row["name"],
            "target"        : tgt,
            "class"         : row["class"],
            "mechanism"     : row["mechanism"],
            "activity_pIC50": row["activity_pIC50"],
        })
dti_df = pd.DataFrame(dti_rows)
print(f"\nDrug-target interactions: {len(dti_df)}")
print(f"  Unique drugs  : {dti_df['drug'].nunique()}")
print(f"  Unique targets: {dti_df['target'].nunique()}")

# ── 2. Aging Pathway Annotations ──────────────────────────────────────────────

TARGET_PATHWAYS = {
    # mTOR / longevity signalling
    "mTORC1"  : "mTOR_Longevity",  "mTORC2" : "mTOR_Longevity",
    "FKBP12"  : "mTOR_Longevity",  "AMPK"   : "mTOR_Longevity",
    "FOXO3"   : "mTOR_Longevity",  "SIRT1"  : "Sirtuin_NAD",
    "SIRT3"   : "Sirtuin_NAD",     "NAMPT"  : "Sirtuin_NAD",
    # Senescence
    "CDKN1A"  : "Senescence",      "CDKN2A" : "Senescence",
    "TP53"    : "Senescence",      "RB1"    : "Senescence",
    "MDM2"    : "Senescence",      "CDK2"   : "Senescence",
    "CDK4"    : "Senescence",      "CDK6"   : "Senescence",
    "CDK1"    : "Senescence",      "CDK9"   : "Senescence",
    "CDK5"    : "Senescence",      "WEE1"   : "Senescence",
    "CHEK1"   : "DNA_Repair",      "CHEK2"  : "DNA_Repair",
    "PARP1"   : "DNA_Repair",      "PARP2"  : "DNA_Repair",
    "ATM"     : "DNA_Repair",
    # Apoptosis / anti-senescent
    "BCL-2"   : "Apoptosis",       "BCL-XL" : "Apoptosis",
    # Inflammation / SASP
    "NF-kB"   : "Inflammation",    "IL6"    : "Inflammation",
    "IL1B"    : "Inflammation",    "TNF"    : "Inflammation",
    "PTGS1"   : "Inflammation",    "PTGS2"  : "Inflammation",
    "IL6R"    : "Inflammation",    "IL1R1"  : "Inflammation",
    "JAK1"    : "Inflammation",    "JAK2"   : "Inflammation",
    "JAK3"    : "Inflammation",    "STAT3"  : "Inflammation",
    "TGFBR1"  : "Inflammation",    "TGFB1"  : "Inflammation",
    # Kinase signalling
    "BCR-ABL" : "Kinase",          "SRC"    : "Kinase",
    "PDGFR"   : "Kinase",          "PIK3CA" : "Kinase",
    "PIK3CD"  : "Kinase",          "PI3K"   : "Kinase",
    "MEK1"    : "Kinase",          "MEK2"   : "Kinase",
    "BTK"     : "Kinase",          "ITK"    : "Kinase",
    "SYK"     : "Kinase",
    # Autophagy / proteostasis
    "cGAS"    : "Autophagy",       "STING"  : "Autophagy",
    "SMPD3"   : "Autophagy",       "TLR9"   : "Autophagy",
    "HDAC"    : "Autophagy",       "HDAC1"  : "Autophagy",
    "NRF2"    : "Autophagy",       "KEAP1"  : "Autophagy",
    # Centromere / genomic stability (CenAGE specific!)
    "FGFR1"   : "Genomic_Stability", "VEGFR2": "Genomic_Stability",
    "PDGFRA"  : "Genomic_Stability",
}

PATHWAY_COLOURS = {
    "mTOR_Longevity"    : "#1565C0",
    "Sirtuin_NAD"       : "#6A1B9A",
    "Senescence"        : "#F57F17",
    "DNA_Repair"        : "#AD1457",
    "Apoptosis"         : "#C62828",
    "Inflammation"      : "#E65100",
    "Kinase"            : "#2E7D32",
    "Autophagy"         : "#00695C",
    "Genomic_Stability" : "#37474F",
    "Other"             : "#9E9E9E",
}

dti_df["pathway"] = dti_df["target"].map(TARGET_PATHWAYS).fillna("Other")

# ── 3. Build Bipartite Drug–Target Network ────────────────────────────────────

print("\n[3] Building drug–target bipartite network ...")

G = nx.Graph()

# Add drug nodes
for _, row in df_drugs.iterrows():
    G.add_node(row["name"], node_type="drug",
               drug_class=row["class"],
               mechanism=row["mechanism"],
               pIC50=row["activity_pIC50"],
               mpo=mpo_score(Chem.MolFromSmiles(row["smiles"])) if row["smiles"] else 0)

# Add target nodes
for tgt in dti_df["target"].unique():
    pathway = TARGET_PATHWAYS.get(tgt, "Other")
    G.add_node(tgt, node_type="target",
               pathway=pathway,
               n_drugs=int((dti_df["target"] == tgt).sum()))

# Add edges
for _, row in dti_df.iterrows():
    if G.has_node(row["drug"]) and G.has_node(row["target"]):
        G.add_edge(row["drug"], row["target"],
                   weight=row["activity_pIC50"],
                   drug_class=row["class"])

print(f"  Nodes: {G.number_of_nodes()} ({len(df_drugs)} drugs + "
      f"{dti_df['target'].nunique()} targets)")
print(f"  Edges: {G.number_of_edges()} drug-target interactions")

# ── 4. Network Metrics ────────────────────────────────────────────────────────

print("\n[4] Computing network centrality metrics ...")

degree_cent    = nx.degree_centrality(G)
betweenness    = nx.betweenness_centrality(G, weight="weight")
eigenvector    = nx.eigenvector_centrality(G, max_iter=1000)

for node in G.nodes():
    G.nodes[node]["degree_centrality"] = degree_cent[node]
    G.nodes[node]["betweenness"]       = betweenness[node]
    G.nodes[node]["eigenvector"]       = eigenvector[node]
    G.nodes[node]["degree"]            = G.degree(node)

# Hub targets: most connected to drugs
target_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "target"]
drug_nodes   = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]

hub_targets  = sorted(target_nodes, key=lambda n: G.degree(n), reverse=True)
hub_drugs    = sorted(drug_nodes,   key=lambda n: G.degree(n), reverse=True)

print(f"\n  Top 10 target hubs (most drugs):")
for t in hub_targets[:10]:
    pathway = G.nodes[t].get("pathway","?")
    print(f"    {t:15s} degree={G.degree(t):3d}  pathway={pathway}")

print(f"\n  Top 10 polypharmacology drugs (most targets):")
for d in hub_drugs[:10]:
    cls = G.nodes[d].get("drug_class","?")
    print(f"    {d:20s} targets={G.degree(d):3d}  class={cls}")

# Export centrality
centrality_df = pd.DataFrame({
    "node"             : list(G.nodes()),
    "node_type"        : [G.nodes[n].get("node_type","") for n in G.nodes()],
    "degree"           : [G.degree(n) for n in G.nodes()],
    "degree_centrality": [degree_cent[n] for n in G.nodes()],
    "betweenness"      : [betweenness[n] for n in G.nodes()],
    "eigenvector"      : [eigenvector[n] for n in G.nodes()],
    "pathway"          : [G.nodes[n].get("pathway", G.nodes[n].get("drug_class","")) for n in G.nodes()],
}).sort_values("degree", ascending=False)
centrality_df.to_csv(RESULTS_DIR / "network_centrality_pharma.csv", index=False)

# ── 5. Drug–Drug Similarity Network (SAR Landscape) ─────────────────────────

print("\n[5] Building structure-activity landscape (SAL) network ...")

mols    = [Chem.MolFromSmiles(s) for s in df_drugs["smiles"]]
fp_mat  = fingerprint_matrix(mols, fp_type="morgan", radius=2, n_bits=2048)
sim_mat = tanimoto_matrix(fp_mat)

# Drug-drug network with Tanimoto ≥ 0.35 edges
G_sal = nx.Graph()
for i, row in df_drugs.iterrows():
    G_sal.add_node(row["name"],
                   drug_class=row["class"],
                   mechanism=row["mechanism"],
                   pIC50=row["activity_pIC50"],
                   n_targets=len(row["target_list"]))

for i in range(len(df_drugs)):
    for j in range(i + 1, len(df_drugs)):
        sim = sim_mat[i, j]
        if sim >= 0.30:
            G_sal.add_edge(df_drugs["name"].iloc[i],
                           df_drugs["name"].iloc[j],
                           tanimoto=float(sim))

n_edges_sal = G_sal.number_of_edges()
print(f"  SAL network: {G_sal.number_of_nodes()} drugs, {n_edges_sal} edges (Tanimoto≥0.30)")

# ── 6. Visualisation ─────────────────────────────────────────────────────────

print("\n[6] Visualising networks ...")

# 6a. Drug-target bipartite network
CLASS_COL  = {"senolytic": "#F44336", "senomorphic": "#2196F3"}
PATHWAY_COL = PATHWAY_COLOURS

fig, ax = plt.subplots(figsize=(18, 14))

# Use spring layout for aesthetic placement
pos = nx.spring_layout(G, k=2.5, iterations=60, seed=SEED, weight="weight")

# Separate node types for drawing
drug_n  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
tgt_n   = [n for n, d in G.nodes(data=True) if d.get("node_type") == "target"]

# Draw edges
edge_alphas = [min(0.8, G[u][v].get("weight", 5) / 12) for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.8, edge_color="#BDBDBD")

# Draw drug nodes (squares via scatter)
for cls, colour in CLASS_COL.items():
    nodes_cls = [n for n in drug_n if G.nodes[n].get("drug_class") == cls]
    sizes     = [300 + 200 * G.degree(n) for n in nodes_cls]
    pos_cls   = np.array([pos[n] for n in nodes_cls])
    if len(pos_cls):
        ax.scatter(pos_cls[:, 0], pos_cls[:, 1],
                   s=sizes, c=colour, alpha=0.85, marker="o",
                   edgecolors="white", linewidth=0.8, zorder=3, label=f"{cls} drug")

# Draw target nodes (triangles)
for pwy, colour in PATHWAY_COL.items():
    nodes_pwy = [n for n in tgt_n if G.nodes[n].get("pathway") == pwy]
    sizes     = [400 + 300 * G.degree(n) for n in nodes_pwy]
    pos_pwy   = np.array([pos[n] for n in nodes_pwy])
    if len(pos_pwy):
        ax.scatter(pos_pwy[:, 0], pos_pwy[:, 1],
                   s=sizes, c=colour, alpha=0.80, marker="^",
                   edgecolors="white", linewidth=0.8, zorder=3, label=pwy)

# Labels for high-degree nodes only
label_nodes = {n: n for n in G.nodes() if G.degree(n) >= 3}
nx.draw_networkx_labels(G, pos, labels=label_nodes, ax=ax,
                         font_size=6.5, font_weight="bold")

ax.set_title("Geroprotector Drug–Target Interaction Network\n"
             "Circles = drugs (red=senolytic, blue=senomorphic) | "
             "Triangles = targets (coloured by pathway) | Size ∝ degree",
             fontsize=12, fontweight="bold")
ax.axis("off")
ax.legend(fontsize=7, ncol=3, loc="lower left",
          markerscale=1.0,
          title="Node type / Pathway", title_fontsize=8)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "15_drug_target_network.pdf",
            bbox_inches="tight", dpi=150)
plt.close()

# 6b. Pathway enrichment bar chart
pathway_counts = dti_df.groupby(["pathway","class"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
pathway_counts.plot(kind="bar", ax=ax,
                    color=["#F44336","#2196F3"],
                    width=0.7, edgecolor="white")
ax.set_title("Drug–Target Interactions by Aging Pathway & Class",
             fontsize=12, fontweight="bold")
ax.set_xlabel(None)
ax.set_ylabel("Number of Interactions", fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
ax.legend(["Senolytic","Senomorphic"], fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "16_pathway_interaction_counts.pdf",
            bbox_inches="tight", dpi=150)
plt.close()

# 6c. Polypharmacology bubble chart
poly_df = (dti_df.groupby("drug")
           .agg(n_targets=("target","nunique"),
                n_pathways=("pathway","nunique"),
                mean_pIC50=("activity_pIC50","mean"),
                drug_class=("class","first"))
           .reset_index()
           .sort_values("n_targets", ascending=False))

fig, ax = plt.subplots(figsize=(10, 7))
for cls, colour in CLASS_COL.items():
    sub = poly_df[poly_df["drug_class"] == cls]
    ax.scatter(sub["n_pathways"], sub["n_targets"],
               s=sub["mean_pIC50"] * 30,
               c=colour, alpha=0.75, label=cls,
               edgecolors="white", lw=0.5)
for _, row in poly_df[poly_df["n_targets"] >= 3].iterrows():
    ax.annotate(row["drug"][:14],
                xy=(row["n_pathways"], row["n_targets"]),
                xytext=(4, 3), textcoords="offset points", fontsize=6.5)

ax.set_xlabel("Number of Distinct Pathways Targeted", fontsize=11)
ax.set_ylabel("Number of Distinct Targets", fontsize=11)
ax.set_title("Polypharmacology Landscape of Geroprotectors\n"
             "Bubble size ∝ mean pIC50 | Red=senolytic | Blue=senomorphic",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10, markerscale=2)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "17_polypharmacology.pdf", bbox_inches="tight", dpi=150)
plt.close()

# 6d. SAL network (drug-drug similarity)
fig, ax = plt.subplots(figsize=(14, 12))
pos_sal = nx.spring_layout(G_sal, k=2, seed=SEED, weight="tanimoto")

edge_weights = [G_sal[u][v]["tanimoto"] for u, v in G_sal.edges()]
nx.draw_networkx_edges(G_sal, pos_sal, ax=ax,
                        width=[w * 3 for w in edge_weights],
                        alpha=0.35, edge_color="#607D8B")
for cls, colour in CLASS_COL.items():
    nodes_c = [n for n in G_sal.nodes()
               if G_sal.nodes[n].get("drug_class") == cls]
    sizes   = [200 + 150 * G_sal.degree(n) for n in nodes_c]
    pos_c   = np.array([pos_sal[n] for n in nodes_c])
    if len(pos_c):
        ax.scatter(pos_c[:, 0], pos_c[:, 1],
                   s=sizes, c=colour, alpha=0.85,
                   edgecolors="white", lw=0.6, label=cls, zorder=3)

nx.draw_networkx_labels(G_sal, pos_sal, ax=ax, font_size=6, font_weight="bold")
ax.set_title("Structure-Activity Landscape (SAL) — Geroprotector Chemical Space\n"
             "Edges: Tanimoto ≥ 0.30 | Node size ∝ number of targets",
             fontsize=12, fontweight="bold")
ax.axis("off")
ax.legend(fontsize=9, loc="lower left")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "18_sal_network.pdf", bbox_inches="tight", dpi=150)
plt.close()

# ── 7. Export for Cytoscape ───────────────────────────────────────────────────

nx.write_graphml(G,     str(RESULTS_DIR / "drug_target_network.graphml"))
nx.write_graphml(G_sal, str(RESULTS_DIR / "sal_network.graphml"))

# Node & edge tables
pd.DataFrame([{
    "id"       : n,
    "node_type": G.nodes[n].get("node_type"),
    "pathway"  : G.nodes[n].get("pathway", G.nodes[n].get("drug_class","")),
    "degree"   : G.degree(n),
    "betweenness": betweenness[n],
} for n in G.nodes()]).to_csv(RESULTS_DIR / "dtn_nodes.csv", index=False)

pd.DataFrame([{
    "source"    : u,
    "target"    : v,
    "pIC50"     : G[u][v].get("weight", 0),
    "drug_class": G[u][v].get("drug_class",""),
} for u, v in G.edges()]).to_csv(RESULTS_DIR / "dtn_edges.csv", index=False)

print(f"\n{'='*60}")
print(f"  Network pharmacology complete!")
print(f"  Drug-target network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  SAL network        : {G_sal.number_of_nodes()} nodes, {n_edges_sal} edges")
print(f"  Top hub target     : {hub_targets[0]} (degree={G.degree(hub_targets[0])})")
print(f"  Top polypharmacol. : {hub_drugs[0]} (targets={G.degree(hub_drugs[0])})")
print(f"  Cytoscape export   : drug_target_network.graphml")
print(f"  Results : {RESULTS_DIR} | Figures : {FIGURES_DIR}")
print(f"{'='*60}")
