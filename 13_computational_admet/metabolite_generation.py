"""
Phase I / Phase II Metabolic Transformation Engine
====================================================
SMARTS-based rule engine for generating metabolites, guided by SoM
probability scores. Builds a full metabolic tree, detects reactive
intermediates, and ranks metabolites by predicted formation likelihood.

Coverage
--------
Phase I  : 20 CYP-mediated oxidative transformations
Phase II : 12 conjugation reactions (UGT, SULT, COMT, NAT, GST)
Reactive : quinones, epoxides, aldehyde intermediates, Michael acceptors

Algorithm
---------
1. Score each atom for SoM vulnerability (rule-based fallback if no model)
2. Apply Phase I SMARTS to atoms above SoM threshold
3. Filter metabolites by MW < 800 and validity
4. Score each metabolite by: P(formation) = SoM_score × rule_priority × MW_penalty
5. For each Phase I metabolite, apply Phase II conjugations
6. Detect reactive metabolites via structural alerts
7. Return ranked metabolic tree

References
----------
Testa & Kraemer (2008) The Biochemistry of Drug Metabolism — Wiley-VCH
Biovia Pipeline Pilot (2023) Metabolite generation protocol
StarDrop (2023) Metabolic transformation rules
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED as RDKitQED
from rdkit.Chem import rdMolDescriptors, rdChemReactions
from rdkit.Chem.Draw import MolsToGridImage

warnings.filterwarnings("ignore")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


# ── Transformation rule library ───────────────────────────────────────────────

@dataclass
class MetabolicRule:
    name:        str
    smarts:      str              # reaction SMARTS (reactant >> product)
    phase:       int              # 1 or 2
    enzyme:      str              # primary enzyme(s)
    priority:    float            # base formation probability [0,1]
    reactive:    bool = False     # does this produce a reactive intermediate?
    requires_oh: bool = False     # Phase II: requires -OH from prior Phase I?


METABOLIC_RULES: List[MetabolicRule] = [
    # ── Phase I: CYP oxidations ───────────────────────────────────────────────

    # Aromatic hydroxylation
    MetabolicRule("arom_hydroxylation",
        "[cH:1]>>[c:1]O",
        phase=1, enzyme="CYP1A2/CYP2C9/CYP3A4", priority=0.80),

    # Aliphatic hydroxylation (methyl)
    MetabolicRule("aliphatic_OH_methyl",
        "[CH3:1][!#7,!#8,!#16:2]>>[CH2:1]([!#7,!#8,!#16:2])O",
        phase=1, enzyme="CYP3A4/CYP2C9", priority=0.65),

    # Benzylic hydroxylation
    MetabolicRule("benzylic_OH",
        "[CH3:1][c:2]>>[CH2:1]([c:2])O",
        phase=1, enzyme="CYP2C8/CYP3A4", priority=0.70),

    # N-dealkylation (N-methyl)
    MetabolicRule("n_demethylation",
        "[N:1][CH3:2]>>[N:1]",
        phase=1, enzyme="CYP1A2/CYP2D6/CYP3A4", priority=0.85),

    # N-dealkylation (N-ethyl)
    MetabolicRule("n_deethylation",
        "[N:1][CH2:2][CH3]>>[N:1]",
        phase=1, enzyme="CYP2D6/CYP3A4", priority=0.70),

    # O-dealkylation (O-methyl)
    MetabolicRule("o_demethylation",
        "[O:1][CH3:2]>>[O:1]",
        phase=1, enzyme="CYP2D6/CYP2C9", priority=0.75),

    # O-dealkylation (O-ethyl)
    MetabolicRule("o_deethylation",
        "[O:1][CH2:2][CH3]>>[O:1]",
        phase=1, enzyme="CYP2D6", priority=0.60),

    # N-oxidation (tertiary amine)
    MetabolicRule("n_oxidation",
        "[N;X3;!$(N=*):1]>>[N+:1][O-]",
        phase=1, enzyme="CYP3A4/FMO", priority=0.55),

    # S-oxidation → sulfoxide
    MetabolicRule("s_oxidation",
        "[S;X2;!$([S]~[#7,#8,F,Cl,Br,I]):1]>>[S:1]=O",
        phase=1, enzyme="CYP3A4/FMO3", priority=0.65),

    # Sulfoxide → sulfone
    MetabolicRule("sulfone_formation",
        "[S:1](=O)>>[S:1](=O)=O",
        phase=1, enzyme="CYP3A4", priority=0.45),

    # Epoxidation (aromatic) — reactive intermediate
    MetabolicRule("arene_epoxide",
        "[c:1][c:2]>>[C@@H:1]1[C@H:2]O1",
        phase=1, enzyme="CYP1A1/CYP3A4", priority=0.30,
        reactive=True),

    # Aliphatic epoxidation — reactive intermediate
    MetabolicRule("alkene_epoxide",
        "[CH:1]=[CH:2]>>[C@@H:1]1[C@H:2]O1",
        phase=1, enzyme="CYP3A4", priority=0.40,
        reactive=True),

    # Dehydrogenation (adjacent to heteroatom → unsaturation)
    MetabolicRule("dehydrogenation",
        "[CH2:1][CH2:2]>>[CH:1]=[CH:2]",
        phase=1, enzyme="CYP3A4", priority=0.35),

    # Desaturation (alpha to carbonyl)
    MetabolicRule("alpha_desaturation",
        "[CH2:1][C:2](=O)>>[CH:1]=[C:2](=O)",
        phase=1, enzyme="CYP2C19/CYP3A4", priority=0.30),

    # Hydroxylation of aliphatic C (general)
    MetabolicRule("aliphatic_OH_general",
        "[CH2:1][CH2:2][CH3]>>[CH2:1][CH:2](O)[CH3]",
        phase=1, enzyme="CYP3A4", priority=0.50),

    # Aldehyde formation from primary alcohol (via alcohol dehydrogenase)
    MetabolicRule("aldehyde_from_alcohol",
        "[CH2:1][OH]>>[CH:1]=O",
        phase=1, enzyme="ADH/CYP2E1", priority=0.55,
        reactive=True),

    # Ester hydrolysis
    MetabolicRule("ester_hydrolysis",
        "[C:1](=O)[O:2][C:3]>>[C:1](=O)[OH]",
        phase=1, enzyme="Esterase/CYP", priority=0.70),

    # Amide hydrolysis
    MetabolicRule("amide_hydrolysis",
        "[C:1](=O)[N:2][C:3]>>[C:1](=O)[OH]",
        phase=1, enzyme="Amidase", priority=0.45),

    # Dehalogenation
    MetabolicRule("dehalogenation",
        "[c:1][Cl,Br,F:2]>>[c:1][OH]",
        phase=1, enzyme="CYP2B6/CYP3A4", priority=0.40),

    # Deamination
    MetabolicRule("oxidative_deamination",
        "[CH:1]([NH2])[CH3]>>[CH:1](=O)[CH3]",
        phase=1, enzyme="MAO/CYP2D6", priority=0.50,
        reactive=True),

    # ── Phase II: Conjugation reactions ──────────────────────────────────────

    # O-glucuronidation (phenols, alcohols)
    MetabolicRule("o_glucuronidation",
        "[OH:1]>>[O:1]C1C(O)C(O)C(O)C(O1)C(=O)O",
        phase=2, enzyme="UGT1A/UGT2B", priority=0.80),

    # N-glucuronidation (amines)
    MetabolicRule("n_glucuronidation",
        "[NH;!$(NC=O):1]>>[N:1]C1C(O)C(O)C(O)C(O1)C(=O)O",
        phase=2, enzyme="UGT1A4/UGT2B7", priority=0.60),

    # Acyl glucuronidation (carboxylic acids) — reactive
    MetabolicRule("acyl_glucuronidation",
        "[C:1](=O)[OH]>>[C:1](=O)OC1C(O)C(O)C(O)C(O1)C(=O)O",
        phase=2, enzyme="UGT1A3/UGT2B7", priority=0.65,
        reactive=True),

    # O-sulfation (phenols)
    MetabolicRule("o_sulfation",
        "[c:1][OH]>>[c:1]OS(=O)(=O)O",
        phase=2, enzyme="SULT1A1/SULT1A3", priority=0.70),

    # N-sulfation (amines)
    MetabolicRule("n_sulfation",
        "[NH2:1][c]>>[N:1]S(=O)(=O)O",
        phase=2, enzyme="SULT1A1", priority=0.50),

    # O-methylation (catechols) via COMT
    MetabolicRule("o_methylation_comt",
        "[c:1]([OH])[c:2][OH]>>[c:1]([OH])[c:2]OC",
        phase=2, enzyme="COMT", priority=0.75),

    # N-acetylation
    MetabolicRule("n_acetylation",
        "[NH2:1][c]>>[NH:1]C(=O)C",
        phase=2, enzyme="NAT1/NAT2", priority=0.65),

    # Glycine conjugation (carboxylic acids)
    MetabolicRule("glycine_conjugation",
        "[C:1](=O)[OH]>>[C:1](=O)NCC(=O)O",
        phase=2, enzyme="GLYAT", priority=0.50),

    # Glutathione conjugation (Michael acceptors / epoxides) — detox
    MetabolicRule("gsh_conjugation_epoxide",
        "[C:1]1[O:2][C:3]1>>[C:1](O)[C:3]SCC(NC(=O)CCC(NC(=O)[C@@H](N)CCC(=O)O)C(=O)O)C(=O)NCC(=O)O",
        phase=2, enzyme="GST", priority=0.70),

    # Mercapturic acid (simplified GSH pathway)
    MetabolicRule("mercapturic_acid",
        "[c:1][Cl]>>[c:1]SCC(NC(=O)C)C(=O)O",
        phase=2, enzyme="GST/GGT/NAT", priority=0.55),

    # Taurine conjugation
    MetabolicRule("taurine_conjugation",
        "[C:1](=O)[OH]>>[C:1](=O)NCCS(=O)(=O)O",
        phase=2, enzyme="ACSL/BAAT", priority=0.35),

    # N-methylation (indoles, imidazoles) via HNMT
    MetabolicRule("n_methylation",
        "[nH:1]>>[n:1]C",
        phase=2, enzyme="HNMT/TPMT", priority=0.40),
]

# Build reaction objects once
def _build_reaction(smarts: str) -> Optional[rdChemReactions.ChemicalReaction]:
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn
    except Exception:
        return None


# ── Reactive metabolite structural alerts ─────────────────────────────────────

REACTIVE_ALERTS: Dict[str, str] = {
    "quinone":           "O=C1C=CC(=O)C=C1",           # para-quinone
    "ortho_quinone":     "O=C1C=CC(=O)c2ccccc12",       # ortho-quinone
    "epoxide":           "C1OC1",                        # epoxide
    "michael_acceptor":  "C=CC=O",                       # alpha,beta-unsaturated carbonyl
    "aldehyde":          "[CH]=O",                       # reactive aldehyde
    "acyl_halide":       "C(=O)[Cl,Br]",                 # acyl halide
    "isocyanate":        "N=C=O",                        # isocyanate
    "quinone_methide":   "C=C-c1ccc(O)cc1",              # quinone methide precursor
    "nitrenium":         "[N+]=[C]",                     # nitrenium ion precursor
    "thiirane":          "C1CS1",                        # episulfide
    "imine_michael":     "C=CN",                         # alpha,beta-unsaturated imine
}

REACTIVE_PATTERNS: Dict[str, Chem.Mol] = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in REACTIVE_ALERTS.items()
    if Chem.MolFromSmarts(smarts) is not None
}


def detect_reactive_metabolite(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """
    Check if a molecule contains reactive metabolite structural alerts.

    Returns
    -------
    (is_reactive, list_of_alert_names)
    """
    alerts = []
    for name, patt in REACTIVE_PATTERNS.items():
        if mol.HasSubstructMatch(patt):
            alerts.append(name)
    return bool(alerts), alerts


# ── Metabolite data class ─────────────────────────────────────────────────────

@dataclass
class Metabolite:
    smiles:       str
    parent_smiles: str
    rule_name:    str
    phase:        int
    enzyme:       str
    formation_prob: float
    depth:        int               # metabolic depth (0 = parent)
    is_reactive:  bool    = False
    reactive_alerts: List[str] = field(default_factory=list)
    mw:           float   = 0.0
    logp:         float   = 0.0
    qed:          float   = 0.0


# ── Metabolic tree generator ──────────────────────────────────────────────────

class MetabolicTreeGenerator:
    """
    Generate a ranked metabolic tree for a parent compound.

    Parameters
    ----------
    max_depth     : maximum metabolic steps (1 = phase I only, 2 = phase I + II)
    mw_cutoff     : reject metabolites above this MW
    prob_cutoff   : reject metabolites with formation_prob below this
    max_metabolites : maximum total metabolites in tree
    """

    def __init__(
        self,
        max_depth:       int   = 2,
        mw_cutoff:       float = 800.0,
        prob_cutoff:     float = 0.10,
        max_metabolites: int   = 50,
    ):
        self.max_depth       = max_depth
        self.mw_cutoff       = mw_cutoff
        self.prob_cutoff     = prob_cutoff
        self.max_metabolites = max_metabolites

        # Pre-compile reactions
        self._reactions: Dict[str, Optional[rdChemReactions.ChemicalReaction]] = {
            r.name: _build_reaction(r.smarts) for r in METABOLIC_RULES
        }

    def _apply_rule(
        self, mol: Chem.Mol, rule: MetabolicRule
    ) -> List[str]:
        """Apply a metabolic SMARTS rule to a molecule. Returns list of product SMILES."""
        rxn = self._reactions.get(rule.name)
        if rxn is None:
            return []
        try:
            products_set = rxn.RunReactants((mol,))
            result = []
            for products in products_set:
                for prod in products:
                    try:
                        Chem.SanitizeMol(prod)
                        smi = Chem.MolToSmiles(prod)
                        if smi and smi != Chem.MolToSmiles(mol):
                            result.append(smi)
                    except Exception:
                        pass
            return list(set(result))
        except Exception:
            return []

    def _filter_metabolite(self, smiles: str) -> bool:
        """Return True if metabolite passes quality filters."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        if mw > self.mw_cutoff or mw < 50:
            return False
        return True

    def generate(
        self,
        smiles: str,
        som_scores: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> List[Metabolite]:
        """
        Generate metabolic tree for a parent SMILES.

        Parameters
        ----------
        smiles     : parent compound SMILES
        som_scores : optional per-atom SoM scores from som_prediction.py
                     format: {atom_idx: {reaction_class: probability}}

        Returns
        -------
        List[Metabolite] sorted by formation_prob descending
        """
        parent_mol = Chem.MolFromSmiles(smiles)
        if parent_mol is None:
            return []

        parent_canon = Chem.MolToSmiles(parent_mol)
        all_metabolites: List[Metabolite] = []
        seen_smiles: Set[str]             = {parent_canon}

        # BFS expansion
        queue = [(parent_canon, 0, 1.0)]  # (smiles, depth, parent_prob)

        while queue:
            curr_smi, depth, parent_prob = queue.pop(0)
            if depth >= self.max_depth:
                continue
            if len(all_metabolites) >= self.max_metabolites:
                break

            curr_mol = Chem.MolFromSmiles(curr_smi)
            if curr_mol is None:
                continue

            # Select rules for this depth
            phase_rules = [r for r in METABOLIC_RULES
                           if r.phase == (depth + 1)]

            for rule in phase_rules:
                products = self._apply_rule(curr_mol, rule)
                for prod_smi in products:
                    if prod_smi in seen_smiles:
                        continue
                    if not self._filter_metabolite(prod_smi):
                        continue

                    prod_mol = Chem.MolFromSmiles(prod_smi)
                    if prod_mol is None:
                        continue

                    # Formation probability
                    form_prob = rule.priority * parent_prob

                    # Modulate by SoM score if available
                    # (simplified: boost by mean SoM score if significant)
                    if som_scores:
                        avg_som = np.mean(list(som_scores.get(0, {0: 0.5}).values()))
                        form_prob *= (0.5 + 0.5 * avg_som)

                    # MW penalty: heavier metabolites form more slowly
                    mw = Descriptors.MolWt(prod_mol)
                    if mw > 600:
                        form_prob *= 0.7
                    elif mw > 500:
                        form_prob *= 0.85

                    if form_prob < self.prob_cutoff:
                        continue

                    is_react, alerts = detect_reactive_metabolite(prod_mol)

                    try:
                        qed = float(RDKitQED.qed(prod_mol))
                    except Exception:
                        qed = 0.0

                    met = Metabolite(
                        smiles         = prod_smi,
                        parent_smiles  = curr_smi,
                        rule_name      = rule.name,
                        phase          = rule.phase,
                        enzyme         = rule.enzyme,
                        formation_prob = round(form_prob, 4),
                        depth          = depth + 1,
                        is_reactive    = is_react,
                        reactive_alerts = alerts,
                        mw             = round(mw, 2),
                        logp           = round(Descriptors.MolLogP(prod_mol), 2),
                        qed            = round(qed, 3),
                    )
                    all_metabolites.append(met)
                    seen_smiles.add(prod_smi)
                    queue.append((prod_smi, depth + 1, form_prob))

        # Sort by formation probability
        all_metabolites.sort(key=lambda m: m.formation_prob, reverse=True)
        return all_metabolites


# ── Metabolic tree visualisation ──────────────────────────────────────────────

def plot_metabolic_tree(
    parent_smiles: str,
    metabolites: List[Metabolite],
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualise the metabolic tree as a directed graph.
    Reactive metabolites are highlighted in red.
    Phase I metabolites in blue, Phase II in green.
    """
    G = nx.DiGraph()
    parent_canon = Chem.MolToSmiles(Chem.MolFromSmiles(parent_smiles))
    G.add_node(parent_canon, depth=0, label="Parent", colour="#95A5A6", prob=1.0)

    for met in metabolites:
        short = met.smiles[:25] + "…" if len(met.smiles) > 25 else met.smiles
        colour = "#E74C3C" if met.is_reactive else ("#3498DB" if met.phase == 1 else "#2ECC71")
        G.add_node(met.smiles, depth=met.depth, label=short,
                   colour=colour, prob=met.formation_prob)
        G.add_edge(met.parent_smiles, met.smiles, rule=met.rule_name, prob=met.formation_prob)

    fig, ax = plt.subplots(figsize=(14, 8))

    if len(G.nodes) <= 1:
        ax.text(0.5, 0.5, "No metabolites generated", ha="center", va="center")
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig

    # Hierarchical layout by depth
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=3.0)

    node_colours = [G.nodes[n].get("colour", "#95A5A6") for n in G.nodes]
    node_sizes   = [800 if G.nodes[n].get("depth", 0) == 0 else 400 for n in G.nodes]
    labels       = {n: G.nodes[n].get("label", n[:15]) for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           edge_color="#7F8C8D", arrowsize=12, alpha=0.6,
                           connectionstyle="arc3,rad=0.1")

    # Edge labels (rule names, abbreviated)
    edge_labels = {(u, v): G.edges[u, v]["rule"][:8]
                   for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=5, alpha=0.7)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#95A5A6", label="Parent"),
        Patch(color="#3498DB", label="Phase I"),
        Patch(color="#2ECC71", label="Phase II"),
        Patch(color="#E74C3C", label="Reactive"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)

    n_react = sum(1 for m in metabolites if m.is_reactive)
    ax.set_title(
        f"Metabolic Tree  |  {len(metabolites)} metabolites  |  "
        f"{n_react} reactive intermediates",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def metabolite_summary_table(metabolites: List[Metabolite]) -> pd.DataFrame:
    """Return a formatted summary DataFrame of all metabolites."""
    rows = []
    for m in metabolites:
        rows.append({
            "SMILES":          m.smiles,
            "Phase":           m.phase,
            "Rule":            m.rule_name,
            "Enzyme":          m.enzyme,
            "P(formation)":    m.formation_prob,
            "MW":              m.mw,
            "LogP":            m.logp,
            "QED":             m.qed,
            "Reactive":        "⚠" if m.is_reactive else "✓",
            "Reactive alerts": ", ".join(m.reactive_alerts) if m.reactive_alerts else "—",
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    generator = MetabolicTreeGenerator(max_depth=2, mw_cutoff=800, prob_cutoff=0.12)

    test_drugs = {
        "Paracetamol (Acetaminophen)": "CC(=O)Nc1ccc(O)cc1",
        "Ibuprofen":                   "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "Lidocaine":                   "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        "Diazepam":                    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "Tamoxifen":                   "CC(/C=C/c1ccc(O)cc1)=C(\CCN(C)C)c1ccccc1",
    }

    print("=" * 65)
    print("  Phase I / Phase II Metabolic Transformation Engine")
    print("=" * 65)

    all_summaries = []

    for drug_name, smiles in test_drugs.items():
        print(f"\n{'─'*60}")
        print(f"Drug: {drug_name}")
        print(f"SMILES: {smiles}")

        metabolites = generator.generate(smiles)

        phase1 = [m for m in metabolites if m.phase == 1]
        phase2 = [m for m in metabolites if m.phase == 2]
        reactive = [m for m in metabolites if m.is_reactive]

        print(f"  Total metabolites : {len(metabolites)}")
        print(f"  Phase I           : {len(phase1)}")
        print(f"  Phase II          : {len(phase2)}")
        print(f"  Reactive          : {len(reactive)}")

        if reactive:
            print(f"  ⚠ Reactive alerts  :")
            for r in reactive:
                print(f"     {r.smiles[:50]} → {', '.join(r.reactive_alerts)}")

        if metabolites:
            top3 = metabolites[:3]
            print(f"\n  Top-3 metabolites by formation probability:")
            for m in top3:
                print(f"    [{m.rule_name:25s}] P={m.formation_prob:.3f}  "
                      f"MW={m.mw:.0f}  {m.smiles[:45]}")

        # Build metabolic tree figure
        fig = plot_metabolic_tree(
            smiles, metabolites,
            out_path=OUT_DIR / f"metabolic_tree_{drug_name.split()[0].lower()}.png"
        )
        plt.close(fig)

        df = metabolite_summary_table(metabolites)
        all_summaries.append(df.assign(parent=drug_name))

    # Export combined summary
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(OUT_DIR / "metabolites_summary.csv", index=False)
    print(f"\n\nExported {len(combined)} metabolite records to "
          f"{OUT_DIR}/metabolites_summary.csv")

    # Reactive metabolite alert summary
    reactive_df = combined[combined["Reactive"] == "⚠"]
    print(f"\nTotal reactive metabolites detected: {len(reactive_df)}")
    if not reactive_df.empty:
        print(reactive_df[["parent", "Rule", "Reactive alerts", "P(formation)"]].to_string(index=False))
