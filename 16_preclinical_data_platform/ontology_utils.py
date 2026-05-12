"""
ontology_utils.py — ChEBI + Gene Ontology annotation utilities.

Curated controlled vocabularies for compound classification (ChEBI) and
gene function annotation (GO slim). Provides semantic similarity scoring
using information-content-based Resnik approach.
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42): np.random.seed(s)
    PALETTES = {"young": "#2196F3", "aged": "#F44336", "accent": "#4CAF50"}

# ──────────────────────────────────────────────────────────────────────────────
# ChEBI hierarchy (curated subset relevant to drug discovery)
# ──────────────────────────────────────────────────────────────────────────────

CHEBI_HIERARCHY: Dict[str, Dict] = {
    "CHEBI:24431": {"name": "chemical entity",     "parents": [],
                    "depth": 0, "frequency": 1.0},
    "CHEBI:23367": {"name": "molecular entity",    "parents": ["CHEBI:24431"],
                    "depth": 1, "frequency": 0.95},
    "CHEBI:33285": {"name": "compound",            "parents": ["CHEBI:23367"],
                    "depth": 2, "frequency": 0.90},
    "CHEBI:23888": {"name": "drug",                "parents": ["CHEBI:33285"],
                    "depth": 3, "frequency": 0.60},
    "CHEBI:35222": {"name": "inhibitor",           "parents": ["CHEBI:23888"],
                    "depth": 4, "frequency": 0.30},
    "CHEBI:38637": {"name": "kinase inhibitor",    "parents": ["CHEBI:35222"],
                    "depth": 5, "frequency": 0.12},
    "CHEBI:33281": {"name": "antimicrobial agent", "parents": ["CHEBI:23888"],
                    "depth": 4, "frequency": 0.15},
    "CHEBI:35176": {"name": "antineoplastic agent","parents": ["CHEBI:23888"],
                    "depth": 4, "frequency": 0.20},
    "CHEBI:50914": {"name": "PARP inhibitor",      "parents": ["CHEBI:35222"],
                    "depth": 5, "frequency": 0.04},
    "CHEBI:75771": {"name": "CDK inhibitor",       "parents": ["CHEBI:38637"],
                    "depth": 6, "frequency": 0.05},
    "CHEBI:79998": {"name": "CDK4/6 inhibitor",    "parents": ["CHEBI:75771"],
                    "depth": 7, "frequency": 0.02},
    "CHEBI:47779": {"name": "organochlorine compound", "parents": ["CHEBI:33285"],
                    "depth": 3, "frequency": 0.10},
    "CHEBI:26347": {"name": "organic heteromonocyclic compound",
                    "parents": ["CHEBI:33285"], "depth": 3, "frequency": 0.25},
    "CHEBI:25806": {"name": "sulfonamide",         "parents": ["CHEBI:26347"],
                    "depth": 4, "frequency": 0.08},
}

# MeSH pharmacological actions mapped to ChEBI IDs
MESH_PHARM_ACTIONS: Dict[str, List[str]] = {
    "Antineoplastic Agents":  ["CHEBI:35176", "CHEBI:79998", "CHEBI:50914"],
    "Enzyme Inhibitors":      ["CHEBI:35222", "CHEBI:38637"],
    "Kinase Inhibitors":      ["CHEBI:38637", "CHEBI:75771", "CHEBI:79998"],
    "Antimicrobial Agents":   ["CHEBI:33281"],
    "Anti-Bacterial Agents":  ["CHEBI:33281", "CHEBI:25806"],
}

# Reverse: chebi → MeSH terms
_CHEBI_TO_MESH: Dict[str, List[str]] = {}
for mesh_term, chebi_ids in MESH_PHARM_ACTIONS.items():
    for cid in chebi_ids:
        _CHEBI_TO_MESH.setdefault(cid, []).append(mesh_term)

# ──────────────────────────────────────────────────────────────────────────────
# Gene Ontology slim (curated subset)
# ──────────────────────────────────────────────────────────────────────────────

GO_TERMS: Dict[str, Dict] = {
    # Molecular Function
    "GO:0003674": {"name": "molecular_function",         "aspect": "MF",
                   "parents": [], "depth": 0, "frequency": 1.0},
    "GO:0004672": {"name": "protein kinase activity",    "aspect": "MF",
                   "parents": ["GO:0003674"], "depth": 2, "frequency": 0.12},
    "GO:0016301": {"name": "kinase activity",            "aspect": "MF",
                   "parents": ["GO:0003674"], "depth": 1, "frequency": 0.18},
    "GO:0004693": {"name": "cyclin-dependent protein serine/threonine kinase activity",
                   "aspect": "MF",
                   "parents": ["GO:0004672"], "depth": 3, "frequency": 0.04},
    "GO:0005245": {"name": "voltage-gated calcium channel activity", "aspect": "MF",
                   "parents": ["GO:0003674"], "depth": 2, "frequency": 0.03},
    "GO:0003714": {"name": "transcription corepressor activity", "aspect": "MF",
                   "parents": ["GO:0003674"], "depth": 2, "frequency": 0.05},
    "GO:0016538": {"name": "cyclin-dependent protein kinase regulator activity",
                   "aspect": "MF",
                   "parents": ["GO:0003674"], "depth": 2, "frequency": 0.03},
    # Biological Process
    "GO:0008150": {"name": "biological_process",         "aspect": "BP",
                   "parents": [], "depth": 0, "frequency": 1.0},
    "GO:0007049": {"name": "cell cycle",                 "aspect": "BP",
                   "parents": ["GO:0008150"], "depth": 1, "frequency": 0.15},
    "GO:0045736": {"name": "negative regulation of cyclin-dependent kinase activity",
                   "aspect": "BP",
                   "parents": ["GO:0007049"], "depth": 3, "frequency": 0.02},
    "GO:0006977": {"name": "DNA damage response, signal transduction by p53",
                   "aspect": "BP",
                   "parents": ["GO:0008150"], "depth": 2, "frequency": 0.04},
    "GO:0006813": {"name": "potassium ion transport",    "aspect": "BP",
                   "parents": ["GO:0008150"], "depth": 2, "frequency": 0.05},
    "GO:0045786": {"name": "negative regulation of cell cycle", "aspect": "BP",
                   "parents": ["GO:0007049"], "depth": 2, "frequency": 0.06},
    "GO:0060307": {"name": "regulation of ventricular cardiac muscle cell action potential",
                   "aspect": "BP",
                   "parents": ["GO:0008150"], "depth": 3, "frequency": 0.01},
    "GO:0007050": {"name": "cell cycle arrest",          "aspect": "BP",
                   "parents": ["GO:0007049"], "depth": 2, "frequency": 0.04},
    # Cellular Component
    "GO:0005575": {"name": "cellular_component",         "aspect": "CC",
                   "parents": [], "depth": 0, "frequency": 1.0},
    "GO:0005737": {"name": "cytoplasm",                  "aspect": "CC",
                   "parents": ["GO:0005575"], "depth": 1, "frequency": 0.40},
    "GO:0005634": {"name": "nucleus",                    "aspect": "CC",
                   "parents": ["GO:0005575"], "depth": 1, "frequency": 0.45},
    "GO:0005886": {"name": "plasma membrane",            "aspect": "CC",
                   "parents": ["GO:0005575"], "depth": 1, "frequency": 0.30},
}

# Gene → GO term sets (curated)
GENE_GO_ANNOTATIONS: Dict[str, Dict[str, List[str]]] = {
    "CDK4":  {
        "MF": ["GO:0004693", "GO:0004672"],
        "BP": ["GO:0007049", "GO:0006977", "GO:0045786"],
        "CC": ["GO:0005737"],
    },
    "CDK2":  {
        "MF": ["GO:0004693", "GO:0004672"],
        "BP": ["GO:0007049", "GO:0045786", "GO:0007050"],
        "CC": ["GO:0005737"],
    },
    "KCNH2": {
        "MF": ["GO:0005245"],
        "BP": ["GO:0006813", "GO:0060307"],
        "CC": ["GO:0005886"],
    },
    "RB1":   {
        "MF": ["GO:0003714"],
        "BP": ["GO:0045736", "GO:0007050"],
        "CC": ["GO:0005634"],
    },
    "CCND1": {
        "MF": ["GO:0016538"],
        "BP": ["GO:0007049", "GO:0045786"],
        "CC": ["GO:0005737"],
    },
    "TP53":  {
        "MF": ["GO:0003714"],
        "BP": ["GO:0006977", "GO:0007050", "GO:0045786"],
        "CC": ["GO:0005634", "GO:0005737"],
    },
    "EGFR":  {
        "MF": ["GO:0004672"],
        "BP": ["GO:0007049", "GO:0006977"],
        "CC": ["GO:0005886"],
    },
    "KRAS":  {
        "MF": ["GO:0004672"],
        "BP": ["GO:0007049"],
        "CC": ["GO:0005737", "GO:0005886"],
    },
}

# Pathway DB (KEGG-style)
PATHWAY_MEMBERSHIP: Dict[str, List[str]] = {
    "CDK4":  ["hsa04110 Cell cycle", "R-HSA-69278 Cell Cycle G1/S"],
    "CDK2":  ["hsa04110 Cell cycle", "R-HSA-69278 Cell Cycle G1/S"],
    "CCND1": ["hsa04110 Cell cycle", "R-HSA-69278 Cell Cycle G1/S"],
    "RB1":   ["hsa04110 Cell cycle", "R-HSA-69278 Cell Cycle G1/S"],
    "TP53":  ["hsa04115 p53 signalling", "R-HSA-5633007 Regulation by TP53"],
    "EGFR":  ["hsa04012 ErbB signalling", "R-HSA-177929 EGFR Downregulation"],
    "KRAS":  ["hsa04010 MAPK signalling", "R-HSA-5672965 RAS processing"],
    "KCNH2": ["hsa04022 cGMP-PKG", "R-HSA-5576891 Cardiac conduction"],
}


# ──────────────────────────────────────────────────────────────────────────────
# ChEBI annotation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CompoundAnnotation:
    compound_id:   str
    chebi_id:      str
    chebi_name:    str
    chebi_depth:   int
    chebi_parents: List[str]
    mesh_actions:  List[str]


def annotate_compound(compound_id: str, chebi_id: str) -> CompoundAnnotation:
    entry = CHEBI_HIERARCHY.get(chebi_id, {})
    return CompoundAnnotation(
        compound_id   = compound_id,
        chebi_id      = chebi_id,
        chebi_name    = entry.get("name", "unknown"),
        chebi_depth   = entry.get("depth", -1),
        chebi_parents = entry.get("parents", []),
        mesh_actions  = _CHEBI_TO_MESH.get(chebi_id, []),
    )


def get_chebi_ancestors(chebi_id: str) -> Set[str]:
    ancestors: Set[str] = set()
    queue = [chebi_id]
    while queue:
        cid = queue.pop()
        if cid in ancestors:
            continue
        ancestors.add(cid)
        queue.extend(CHEBI_HIERARCHY.get(cid, {}).get("parents", []))
    return ancestors


# ──────────────────────────────────────────────────────────────────────────────
# GO annotation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneAnnotation:
    gene_symbol:  str
    go_mf:        List[str]
    go_bp:        List[str]
    go_cc:        List[str]
    pathways:     List[str]
    go_names:     Dict[str, str]  # term_id → name


def annotate_gene(gene_symbol: str) -> Optional[GeneAnnotation]:
    anns = GENE_GO_ANNOTATIONS.get(gene_symbol.upper())
    if not anns:
        return None
    all_terms = anns.get("MF", []) + anns.get("BP", []) + anns.get("CC", [])
    go_names  = {t: GO_TERMS[t]["name"] for t in all_terms if t in GO_TERMS}
    return GeneAnnotation(
        gene_symbol = gene_symbol,
        go_mf       = anns.get("MF", []),
        go_bp       = anns.get("BP", []),
        go_cc       = anns.get("CC", []),
        pathways    = PATHWAY_MEMBERSHIP.get(gene_symbol.upper(), []),
        go_names    = go_names,
    )


def get_go_ancestors(go_id: str) -> Set[str]:
    ancestors: Set[str] = set()
    queue = [go_id]
    while queue:
        tid = queue.pop()
        if tid in ancestors:
            continue
        ancestors.add(tid)
        queue.extend(GO_TERMS.get(tid, {}).get("parents", []))
    return ancestors


# ──────────────────────────────────────────────────────────────────────────────
# Semantic similarity (Resnik)
# ──────────────────────────────────────────────────────────────────────────────

def _ic(term_id: str, ontology: Dict[str, Dict]) -> float:
    freq = ontology.get(term_id, {}).get("frequency", 1.0)
    return -math.log(max(freq, 1e-9))


def resnik_similarity(term_a: str, term_b: str) -> float:
    """Information-content of the most informative common ancestor."""
    anc_a = get_go_ancestors(term_a)
    anc_b = get_go_ancestors(term_b)
    common = anc_a & anc_b
    if not common:
        return 0.0
    return max(_ic(t, GO_TERMS) for t in common)


def gene_semantic_similarity(
    gene_a: str, gene_b: str, aspect: str = "BP"
) -> float:
    """BMA (best-match average) semantic similarity between two gene's GO sets."""
    ann_a = GENE_GO_ANNOTATIONS.get(gene_a.upper(), {}).get(aspect, [])
    ann_b = GENE_GO_ANNOTATIONS.get(gene_b.upper(), {}).get(aspect, [])
    if not ann_a or not ann_b:
        return 0.0

    # Best-match average
    scores_ab = [max(resnik_similarity(a, b) for b in ann_b) for a in ann_a]
    scores_ba = [max(resnik_similarity(b, a) for a in ann_a) for b in ann_b]
    return float(np.mean(scores_ab + scores_ba))


def gene_similarity_matrix(genes: List[str], aspect: str = "BP") -> pd.DataFrame:
    mat = np.zeros((len(genes), len(genes)))
    for i, ga in enumerate(genes):
        for j, gb in enumerate(genes):
            mat[i, j] = gene_semantic_similarity(ga, gb, aspect)
    return pd.DataFrame(mat, index=genes, columns=genes)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_ontology_overview(
    out_path: str = "figures/ontology_overview.png",
) -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    genes = list(GENE_GO_ANNOTATIONS.keys())
    sim_mat = gene_similarity_matrix(genes, aspect="BP")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#FAFAFA")

    # Panel 1: GO annotation heatmap
    ax = axes[0]
    all_go_bp = sorted({t for g in genes for t in GENE_GO_ANNOTATIONS[g].get("BP", [])})
    ann_matrix = np.zeros((len(genes), len(all_go_bp)))
    for i, g in enumerate(genes):
        for j, t in enumerate(all_go_bp):
            ann_matrix[i, j] = 1 if t in GENE_GO_ANNOTATIONS[g].get("BP", []) else 0
    ax.imshow(ann_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_go_bp)))
    ax.set_xticklabels([GO_TERMS[t]["name"][:20] for t in all_go_bp],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9)
    ax.set_title("Gene → GO Biological Process Annotations", fontsize=11, fontweight="bold")

    # Panel 2: Semantic similarity matrix
    ax = axes[1]
    im = ax.imshow(sim_mat.values, cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Resnik IC")
    ax.set_xticks(range(len(genes)))
    ax.set_yticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(genes, fontsize=9)
    for i in range(len(genes)):
        for j in range(len(genes)):
            ax.text(j, i, f"{sim_mat.values[i,j]:.1f}", ha="center", va="center",
                    fontsize=7, color="white" if sim_mat.values[i,j] > 2 else "#333333")
    ax.set_title("Gene Semantic Similarity (BP, Resnik)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    print("── ChEBI compound annotation ──")
    for chebi_id in ["CHEBI:79998", "CHEBI:38637", "CHEBI:35176"]:
        ann = annotate_compound("CPD0001", chebi_id)
        ancestors = get_chebi_ancestors(chebi_id)
        mesh = ann.mesh_actions
        print(f"  {ann.chebi_name:<30}  depth={ann.chebi_depth}"
              f"  ancestors={len(ancestors)}  MeSH={mesh}")

    print("\n── GO gene annotation ──")
    for gene in ["CDK4", "CDK2", "KCNH2", "RB1"]:
        ann = annotate_gene(gene)
        if ann:
            print(f"  {gene:<8}  MF={len(ann.go_mf)}  BP={len(ann.go_bp)}"
                  f"  CC={len(ann.go_cc)}  pathways={len(ann.pathways)}")
            for t, name in list(ann.go_names.items())[:2]:
                print(f"           {t} — {name}")

    print("\n── Gene semantic similarity (BP) ──")
    genes = ["CDK4", "CDK2", "RB1", "KCNH2"]
    mat   = gene_similarity_matrix(genes, aspect="BP")
    print(mat.round(2).to_string())

    os.makedirs("figures", exist_ok=True)
    img = plot_ontology_overview("figures/ontology_overview.png")
    print(f"\nPlot saved → {img}")
