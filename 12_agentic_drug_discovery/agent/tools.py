"""
tools.py — Cheminformatics Tool Registry for Drug Discovery Agent
=================================================================
8 tools callable by the ReAct agent via execute_tool().

Tools
-----
  SimilaritySearchTool      — Tanimoto search against CDK4/6 reference library
  PropertyCalculatorTool    — MW, LogP, QED, SA, TPSA, PAINS, MPO
  ADMETPredictorTool        — rule-based BBB, hERG, CYP3A4, solubility, F%
  BindingAffinityPredictorTool — RF model → predicted pIC50
  ScaffoldHopperTool        — bioisostere SMARTS transformations
  LeadOptimizerTool         — MedChem modification suggestions
  LiteratureSearchTool      — curated knowledge base keyword search
  MoleculeVisualizerTool    — saves property summary figure
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. REFERENCE LIBRARY — 50 CDK4/6-like compounds with known pIC50
# ---------------------------------------------------------------------------

REFERENCE_LIBRARY: List[Dict] = [
    # Palbociclib-like: pyrido[2,3-d]pyrimidine + piperazine core
    {"smiles": "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)CCN1",          "name": "CDK46_001", "pIC50": 8.2},
    {"smiles": "CC1=C(C(=O)Nc2ncnc3ccccc23)CCN1",             "name": "CDK46_002", "pIC50": 7.8},
    {"smiles": "CN1CCN(c2ccc(NC(=O)c3cccnc3)cc2)CC1",         "name": "CDK46_003", "pIC50": 9.1},
    {"smiles": "CC(=O)Nc1ccc(-c2nc3ccccc3[nH]2)cc1",          "name": "CDK46_004", "pIC50": 7.3},
    {"smiles": "C1CN(c2ccc(Nc3ncnc4[nH]ccc34)cc2)CCN1",       "name": "CDK46_005", "pIC50": 8.5},
    # Ribociclib-like: amino-pyrrolo-pyrimidine
    {"smiles": "Nc1ncnc2[nH]ccc12",                            "name": "CDK46_006", "pIC50": 6.5},
    {"smiles": "CN1CCC(Nc2ncnc3[nH]ccc23)CC1",                "name": "CDK46_007", "pIC50": 8.8},
    {"smiles": "CC1CCN(c2ccc(Nc3ncnc4ccc[nH]34)cc2)CC1",      "name": "CDK46_008", "pIC50": 8.6},
    {"smiles": "CN1CCN(C(=O)c2ccc(Nc3ncnc4[nH]ccc34)cc2)CC1","name": "CDK46_009", "pIC50": 7.9},
    {"smiles": "Cc1ccc(NC(=O)c2cc(-c3ccncc3)ccn2)cc1",        "name": "CDK46_010", "pIC50": 7.4},
    # Abemaciclib-like: dimethyl-aminopyrimidine
    {"smiles": "CN(C)c1ncnc2[nH]ccc12",                       "name": "CDK46_011", "pIC50": 7.0},
    {"smiles": "CN(C)c1ccc(NC(=O)c2ccncc2)cc1",               "name": "CDK46_012", "pIC50": 7.6},
    {"smiles": "c1ccc(-c2ccncc2)nc1",                          "name": "CDK46_013", "pIC50": 6.8},
    {"smiles": "CN1CCN(c2ccnc(Nc3ccc(F)cc3)n2)CC1",           "name": "CDK46_014", "pIC50": 8.3},
    {"smiles": "Fc1ccc(Nc2ncnc3[nH]ccc23)cc1",                "name": "CDK46_015", "pIC50": 8.0},
    # Indazole / benzimidazole scaffolds
    {"smiles": "c1ccc2[nH]ncc2c1",                             "name": "CDK46_016", "pIC50": 6.6},
    {"smiles": "CN1C=NC2=CC=CC=C21",                           "name": "CDK46_017", "pIC50": 6.9},
    {"smiles": "CC1=NC2=CC=CC=C2N1",                           "name": "CDK46_018", "pIC50": 7.1},
    {"smiles": "c1ccc2nc[nH]c2c1",                             "name": "CDK46_019", "pIC50": 6.7},
    {"smiles": "CN1C(=O)c2ccccc2N=C1N",                       "name": "CDK46_020", "pIC50": 7.2},
    # Pyrimidine + morpholine variants
    {"smiles": "C1COCCN1",                                     "name": "CDK46_021", "pIC50": 6.5},
    {"smiles": "CN1CCN(c2ncnc3ccc[nH]23)CC1",                 "name": "CDK46_022", "pIC50": 8.1},
    {"smiles": "C1CN(c2ncccn2)CCO1",                           "name": "CDK46_023", "pIC50": 7.5},
    {"smiles": "CC1=CC(=NC(=N1)N)N2CCOCC2",                   "name": "CDK46_024", "pIC50": 8.4},
    {"smiles": "CN1CCN(c2nccc(N)n2)CC1",                       "name": "CDK46_025", "pIC50": 7.7},
    # Sulfonamide / amide linker variants
    {"smiles": "NS(=O)(=O)c1ccc(Nc2ncnc3[nH]ccc23)cc1",      "name": "CDK46_026", "pIC50": 8.7},
    {"smiles": "CC(=O)Nc1ccc(Nc2ncnc3[nH]ccc23)cc1",          "name": "CDK46_027", "pIC50": 8.9},
    {"smiles": "O=C(Nc1ccccc1)c1ccncc1",                       "name": "CDK46_028", "pIC50": 7.0},
    {"smiles": "CC(=O)Nc1ccc(-c2ccncc2)cc1",                   "name": "CDK46_029", "pIC50": 7.3},
    {"smiles": "O=C(Nc1ccc(F)cc1)c1ccncc1",                    "name": "CDK46_030", "pIC50": 7.6},
    # Fluorinated analogues
    {"smiles": "Fc1cccc(Nc2ncnc3[nH]ccc23)c1",                "name": "CDK46_031", "pIC50": 8.1},
    {"smiles": "Fc1ccc(CN2CCN(c3ncnc4[nH]ccc34)CC2)cc1",      "name": "CDK46_032", "pIC50": 8.6},
    {"smiles": "CC1CN(c2ccc(NC(=O)c3cccnc3F)cc2)CCN1",        "name": "CDK46_033", "pIC50": 9.0},
    {"smiles": "Fc1cncc(NC2CCNCC2)c1",                         "name": "CDK46_034", "pIC50": 7.8},
    {"smiles": "CC(F)(F)c1ccc(Nc2ncnc3[nH]ccc23)cc1",         "name": "CDK46_035", "pIC50": 8.3},
    # Methyl / methoxy substituents
    {"smiles": "COc1ccc(Nc2ncnc3[nH]ccc23)cc1",               "name": "CDK46_036", "pIC50": 7.9},
    {"smiles": "Cc1cncc(NC2CCN(C)CC2)c1",                      "name": "CDK46_037", "pIC50": 8.2},
    {"smiles": "COc1cccc(CN2CCN(c3ncnc4ccc[nH]34)CC2)c1",     "name": "CDK46_038", "pIC50": 8.5},
    {"smiles": "Cc1cc(NC(=O)c2ccncc2)ccc1N1CCNCC1",           "name": "CDK46_039", "pIC50": 8.8},
    {"smiles": "COc1ccc(C(=O)Nc2ccc(N3CCNCC3)cc2)cc1",        "name": "CDK46_040", "pIC50": 7.4},
    # Chloro analogues
    {"smiles": "Clc1ccc(Nc2ncnc3[nH]ccc23)cc1",               "name": "CDK46_041", "pIC50": 7.7},
    {"smiles": "Clc1cncc(NC2CCNCC2)c1",                        "name": "CDK46_042", "pIC50": 8.0},
    {"smiles": "CC1=C(Cl)N=C(N)N=C1N1CCNCC1",                 "name": "CDK46_043", "pIC50": 8.4},
    {"smiles": "Clc1ccc(CN2CCN(c3ncnc4[nH]ccc34)CC2)cc1",     "name": "CDK46_044", "pIC50": 8.7},
    {"smiles": "Clc1cccc(NC(=O)c2ccncc2)c1",                   "name": "CDK46_045", "pIC50": 7.2},
    # Cyclopentyl / cyclohexyl variants
    {"smiles": "N#Cc1ccc(Nc2ncnc3[nH]ccc23)cc1",              "name": "CDK46_046", "pIC50": 8.0},
    {"smiles": "O=C1CCCCN1c1ccc(Nc2ncnc3[nH]ccc23)cc1",       "name": "CDK46_047", "pIC50": 8.3},
    {"smiles": "CC1(C)CCN(c2ccc(Nc3ncnc4[nH]ccc34)cc2)CC1",   "name": "CDK46_048", "pIC50": 8.6},
    {"smiles": "C1CCC(Nc2ccc(N3CCNCC3)nc2)CC1",               "name": "CDK46_049", "pIC50": 7.5},
    {"smiles": "CC1=NC(=O)c2cc(Nc3ncnc4[nH]ccc34)ccc21",      "name": "CDK46_050", "pIC50": 9.2},
]

# Validate and canonicalise at import time; drop any invalid entries
_VALIDATED_LIBRARY: List[Dict] = []
for _entry in REFERENCE_LIBRARY:
    _mol = Chem.MolFromSmiles(_entry["smiles"])
    if _mol is not None:
        _entry = dict(_entry)
        _entry["smiles"] = Chem.MolToSmiles(_mol)
        _entry.setdefault("target", "CDK4/6")
        _VALIDATED_LIBRARY.append(_entry)

REFERENCE_LIBRARY = _VALIDATED_LIBRARY


# ---------------------------------------------------------------------------
# 2. SIMILARITY SEARCH TOOL
# ---------------------------------------------------------------------------

@dataclass
class SimilaritySearchTool:
    """Tanimoto similarity search against the CDK4/6 reference library."""

    name: str = field(default="similarity_search", init=False)
    description: str = field(
        default=(
            "Search CDK4/6 reference library by Tanimoto similarity (Morgan FP). "
            "Args: smiles (str), top_k (int=5). "
            "Returns hits sorted by similarity descending."
        ),
        init=False,
    )

    def run(self, smiles: str, top_k: int = 5) -> Dict:
        """
        Parameters
        ----------
        smiles : query molecule SMILES
        top_k  : number of hits to return

        Returns
        -------
        {
            "hits": [{"smiles", "name", "pIC50", "target", "similarity"}, ...],
            "query_valid": bool,
            "n_searched": int,
        }
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"hits": [], "query_valid": False, "n_searched": 0,
                        "error": f"Invalid SMILES: {smiles}"}

            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

            scored: List[Tuple[float, Dict]] = []
            for entry in REFERENCE_LIBRARY:
                ref_mol = Chem.MolFromSmiles(entry["smiles"])
                if ref_mol is None:
                    continue
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=1024)
                sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
                scored.append((sim, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            hits = [
                {
                    "smiles":     entry["smiles"],
                    "name":       entry["name"],
                    "pIC50":      entry["pIC50"],
                    "target":     entry.get("target", "CDK4/6"),
                    "similarity": round(sim, 4),
                }
                for sim, entry in scored[:top_k]
            ]

            return {
                "hits":       hits,
                "query_valid": True,
                "n_searched": len(REFERENCE_LIBRARY),
            }

        except Exception as exc:
            return {"hits": [], "query_valid": False, "n_searched": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# 3. BINDING AFFINITY PREDICTOR
# ---------------------------------------------------------------------------

@dataclass
class BindingAffinityPredictorTool:
    """
    Random Forest model trained on REFERENCE_LIBRARY + synthetic decoys.
    Predicts pIC50 against CDK4/6 from Morgan fingerprints.
    Model is fitted once at instantiation.
    """

    name: str = field(default="binding_affinity", init=False)
    description: str = field(
        default=(
            "Predict pIC50 against CDK4/6 using an RF model trained on the "
            "reference library. Args: smiles (str). "
            "Returns predicted_pIC50, confidence, activity_class."
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        self._model = None
        self._train()

    # ------------------------------------------------------------------
    def _mol_to_fp(self, smiles: str) -> Optional[np.ndarray]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return np.array(fp, dtype=np.float32)

    # ------------------------------------------------------------------
    def _train(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        X, y = [], []

        # Positive set: reference library
        for entry in REFERENCE_LIBRARY:
            fp = self._mol_to_fp(entry["smiles"])
            if fp is not None:
                X.append(fp)
                y.append(entry["pIC50"])

        # Synthetic decoys: random drug-like SMILES with pIC50 4.0–6.0
        rng = np.random.default_rng(42)
        decoy_smiles = [
            "c1ccccc1", "CC(C)O", "CCO", "CCCC", "c1ccncc1",
            "CC(=O)O", "CCC(=O)N", "c1ccc(O)cc1", "CC(N)=O", "CCOC(=O)C",
            "c1ccoc1",  "c1ccsc1",  "C1CCNCC1",   "C1CCOCC1", "CC1=CC=CC=C1",
            "c1ccc(Cl)cc1", "c1ccc(F)cc1", "c1ccc(Br)cc1", "CCN(CC)CC", "CCCCO",
        ]
        for smi in decoy_smiles * 5:   # 100 decoys
            fp = self._mol_to_fp(smi)
            if fp is not None:
                X.append(fp)
                y.append(float(rng.uniform(4.0, 6.0)))

        X = np.vstack(X)
        y = np.array(y, dtype=np.float32)

        self._model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        self._model.fit(X, y)

    # ------------------------------------------------------------------
    def run(self, smiles: str) -> Dict:
        """
        Parameters
        ----------
        smiles : molecule SMILES

        Returns
        -------
        {
            "predicted_pIC50": float,
            "confidence": "high" | "medium" | "low",
            "activity_class": "inactive" | "moderate" | "active" | "highly_active",
            "smiles_input": str,
        }
        """
        try:
            fp = self._mol_to_fp(smiles)
            if fp is None:
                return {"error": f"Invalid SMILES: {smiles}",
                        "predicted_pIC50": None, "confidence": "low",
                        "activity_class": "unknown"}

            fp_2d = fp.reshape(1, -1)
            pred = float(self._model.predict(fp_2d)[0])

            # Confidence from std across trees
            tree_preds = np.array([t.predict(fp_2d)[0] for t in self._model.estimators_])
            std = float(tree_preds.std())
            if std < 0.3:
                confidence = "high"
            elif std < 0.6:
                confidence = "medium"
            else:
                confidence = "low"

            # Activity class
            if pred < 6.0:
                activity_class = "inactive"
            elif pred < 7.0:
                activity_class = "moderate"
            elif pred < 8.0:
                activity_class = "active"
            else:
                activity_class = "highly_active"

            return {
                "predicted_pIC50": round(pred, 3),
                "confidence":      confidence,
                "activity_class":  activity_class,
                "smiles_input":    smiles,
            }

        except Exception as exc:
            return {"error": str(exc), "predicted_pIC50": None,
                    "confidence": "low", "activity_class": "unknown"}


# ---------------------------------------------------------------------------
# 4. PROPERTY CALCULATOR TOOL
# ---------------------------------------------------------------------------

# 5 common PAINS SMARTS patterns
_PAINS_SMARTS = [
    Chem.MolFromSmarts("[#6]1~[#6]~[#6](~[#7])~[#6]~[#6]~[#6]1~[#8]"),  # aniline-quinone
    Chem.MolFromSmarts("[#16]~[#6]~[#6](=O)~[#7]"),                       # rhodanine-like
    Chem.MolFromSmarts("[OH]c1ccccc1[OH]"),                                # catechol
    Chem.MolFromSmarts("[#6]1(=O)~[#6]~[#6]~[#6](=O)~[#6]~[#6]1"),       # quinone
    Chem.MolFromSmarts("[CH]=[CH]-[C](=O)"),                               # michael acceptor
]
_PAINS_SMARTS = [p for p in _PAINS_SMARTS if p is not None]


def _sa_score(mol: Chem.Mol) -> float:
    """
    Approximate synthetic accessibility score (1=easy, 10=hard).
    Based on ring complexity and stereocentre count.
    """
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    mw = Descriptors.ExactMolWt(mol)

    complexity = (
        0.5 * n_rings
        + 1.0 * n_bridgehead
        + 0.8 * n_spiro
        + 0.3 * n_stereo
        + 0.002 * max(0, mw - 300)
    )
    score = 1.0 + min(9.0, complexity)
    return round(score, 2)


def _mpo_score(mol: Chem.Mol) -> float:
    """
    CNS MPO-style composite score (0–6).
    Each of 6 properties contributes 0 or 1 based on desirable ranges.
    """
    mw   = Descriptors.ExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    # pKa proxy: count basic nitrogens
    basic_n = sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() == 7 and a.GetTotalNumHs() > 0
    )

    score = (
        (1 if mw   <= 360  else 0) +
        (1 if logp <= 3.0  else 0) +
        (1 if tpsa >= 40 and tpsa <= 90 else 0) +
        (1 if hbd  <= 3    else 0) +
        (1 if hba  <= 7    else 0) +
        (1 if basic_n <= 2 else 0)
    )
    return float(score)


@dataclass
class PropertyCalculatorTool:
    """Calculate key physicochemical and drug-likeness properties."""

    name: str = field(default="property_calculator", init=False)
    description: str = field(
        default=(
            "Calculate MW, LogP, QED, TPSA, HBD, HBA, nRotB, SA_score, "
            "MPO_score, PAINS_alerts, Lipinski_pass. Args: smiles (str)."
        ),
        init=False,
    )

    def run(self, smiles: str) -> Dict:
        """
        Returns
        -------
        {
            "MW", "LogP", "QED", "TPSA", "HBD", "HBA", "nRotB",
            "nAr", "SA_score", "MPO_score", "PAINS_alerts",
            "Lipinski_pass", "smiles_input"
        }
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": f"Invalid SMILES: {smiles}"}

            mw   = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            hbd  = rdMolDescriptors.CalcNumHBD(mol)
            hba  = rdMolDescriptors.CalcNumHBA(mol)
            nrotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
            n_ar  = rdMolDescriptors.CalcNumAromaticRings(mol)
            qed   = QED.qed(mol)
            sa    = _sa_score(mol)
            mpo   = _mpo_score(mol)

            pains = sum(
                1 for pat in _PAINS_SMARTS if mol.HasSubstructMatch(pat)
            )

            lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

            return {
                "MW":           round(mw, 2),
                "LogP":         round(logp, 2),
                "QED":          round(qed, 3),
                "TPSA":         round(tpsa, 1),
                "HBD":          hbd,
                "HBA":          hba,
                "nRotB":        nrotb,
                "nAr":          n_ar,
                "SA_score":     sa,
                "MPO_score":    mpo,
                "PAINS_alerts": pains,
                "Lipinski_pass": lipinski,
                "smiles_input": smiles,
            }

        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# 5. ADMET PREDICTOR TOOL
# ---------------------------------------------------------------------------

@dataclass
class ADMETPredictorTool:
    """Rule-based ADMET predictor using RDKit physicochemical properties."""

    name: str = field(default="admet_predictor", init=False)
    description: str = field(
        default=(
            "Predict BBB penetration, hERG risk, CYP3A4 substrate likelihood, "
            "aqueous solubility (ESOL), and oral bioavailability. "
            "Args: smiles (str)."
        ),
        init=False,
    )

    def run(self, smiles: str) -> Dict:
        """
        Returns
        -------
        {
            "BBB_penetrant": bool,
            "hERG_risk": "low" | "medium" | "high",
            "CYP3A4_substrate": bool,
            "solubility_mgL": float,        # ESOL estimate
            "solubility_class": str,        # "insoluble" … "highly soluble"
            "bioavailability_pct": float,   # 0–100
            "overall_admet": "good" | "moderate" | "poor",
            "flags": [str],                 # human-readable liabilities
        }
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": f"Invalid SMILES: {smiles}"}

            mw   = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            hbd  = rdMolDescriptors.CalcNumHBD(mol)
            hba  = rdMolDescriptors.CalcNumHBA(mol)
            nrotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
            n_ar  = rdMolDescriptors.CalcNumAromaticRings(mol)

            # Basic nitrogen count (pKa proxy for hERG)
            basic_n = sum(
                1 for a in mol.GetAtoms()
                if a.GetAtomicNum() == 7 and a.GetTotalNumHs() > 0
            )

            flags = []

            # ── BBB penetration ────────────────────────────────────────
            # Lobes-Abraham-Martin criteria (CNS drug-likeness)
            bbb = (
                mw   <= 400 and
                1.0  <= logp <= 3.5 and
                tpsa <= 90  and
                hbd  <= 3   and
                hba  <= 7
            )
            if not bbb:
                flags.append("Low BBB penetration predicted")

            # ── hERG risk ──────────────────────────────────────────────
            # Key structural features: basic amine + aromatic bulk + MW
            herg_score = 0
            if basic_n >= 1:   herg_score += 1
            if n_ar    >= 3:   herg_score += 1
            if mw      > 350:  herg_score += 1
            if logp    > 3.0:  herg_score += 1

            if herg_score >= 3:
                herg_risk = "high"
                flags.append("hERG liability (basic amine + aromatic bulk)")
            elif herg_score == 2:
                herg_risk = "medium"
            else:
                herg_risk = "low"

            # ── CYP3A4 substrate ───────────────────────────────────────
            # Large, lipophilic, aromatic compounds tend to be 3A4 substrates
            cyp3a4 = (mw > 400 or n_ar >= 3 or logp > 3.0)
            if cyp3a4:
                flags.append("Likely CYP3A4 substrate — DDI risk")

            # ── Aqueous solubility (Delaney ESOL) ─────────────────────
            # log S = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*AP
            # where AP = fraction aromatic atoms
            n_atoms = mol.GetNumHeavyAtoms()
            n_ar_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
            ap = n_ar_atoms / n_atoms if n_atoms > 0 else 0.0
            log_s = (0.16
                     - 0.63 * logp
                     - 0.0062 * mw
                     + 0.066 * nrotb
                     - 0.74 * ap)
            sol_mol_L = 10 ** log_s                  # mol/L
            sol_mg_L  = sol_mol_L * mw * 1000        # mg/L

            if log_s < -5:
                sol_class = "insoluble"
                flags.append("Poor aqueous solubility")
            elif log_s < -4:
                sol_class = "poorly soluble"
            elif log_s < -2:
                sol_class = "moderately soluble"
            else:
                sol_class = "highly soluble"

            # ── Oral bioavailability (Fa × Fg estimate) ───────────────
            # Fa from TPSA (Egan model proxy)
            fa = max(0.0, min(1.0, 1.0 - (tpsa / 150.0)))
            # Fg: first-pass gut wall (CYP3A4 in enterocytes)
            fg = 0.6 if cyp3a4 else 0.9
            # Fh: hepatic first-pass (simple estimate from LogP)
            fh = max(0.3, 1.0 - 0.08 * max(0, logp - 2))
            f_oral = fa * fg * fh * 100.0
            if f_oral < 20:
                flags.append(f"Low predicted bioavailability ({f_oral:.0f}%)")

            # ── Overall ADMET rating ───────────────────────────────────
            n_flags = len(flags)
            if n_flags == 0:
                overall = "good"
            elif n_flags <= 2:
                overall = "moderate"
            else:
                overall = "poor"

            return {
                "BBB_penetrant":     bbb,
                "hERG_risk":         herg_risk,
                "CYP3A4_substrate":  cyp3a4,
                "solubility_mgL":    round(sol_mg_L, 2),
                "solubility_class":  sol_class,
                "bioavailability_pct": round(f_oral, 1),
                "overall_admet":     overall,
                "flags":             flags,
                "smiles_input":      smiles,
            }

        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# 6. SCAFFOLD HOPPER TOOL
# ---------------------------------------------------------------------------

# Each transformation: (name, from_smarts, to_smarts)
_BIOISOSTERE_TRANSFORMS = [
    ("pyridine→pyrimidine",  "c1ccncc1",          "c1ccncn1"),
    ("benzene→thiophene",    "c1ccccc1",           "c1ccsc1"),
    ("piperazine→morpholine","C1CCNCCN1",          "C1COCCN1"),
    ("NH→O (amide→ester)",   "[NH1][C:1](=O)",     "[O][C:1](=O)"),
    ("C=O→SO2",              "[C:1](=O)[N:2]",     "[S:1](=O)(=O)[N:2]"),
    ("Cl→F",                 "[Cl:1]",             "[F:1]"),
    ("phenyl→cyclohexyl",    "c1ccccc1",           "C1CCCCC1"),
]


def _apply_transform(smiles: str, from_smarts: str, to_smarts: str) -> Optional[str]:
    """Apply a single SMARTS replacement; return canonical SMILES or None."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        rxn = AllChem.ReactionFromSmarts(f"{from_smarts}>>{to_smarts}")
        if rxn is None:
            return None
        products = rxn.RunReactants((mol,))
        if not products:
            return None
        for prod_tuple in products:
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                    smi = Chem.MolToSmiles(prod)
                    if smi and Chem.MolFromSmiles(smi) is not None:
                        return smi
                except Exception:
                    continue
        return None
    except Exception:
        return None


@dataclass
class ScaffoldHopperTool:
    """Generate scaffold hops via bioisostere replacements."""

    name: str = field(default="scaffold_hopper", init=False)
    description: str = field(
        default=(
            "Apply bioisostere SMARTS transformations to generate scaffold hops. "
            "Args: smiles (str), n_hops (int=5). "
            "Returns valid transformed SMILES with transformation labels."
        ),
        init=False,
    )

    def run(self, smiles: str, n_hops: int = 5) -> Dict:
        """
        Returns
        -------
        {
            "hops": [{"smiles", "transformation", "valid"}, ...],
            "n_valid": int,
            "query_valid": bool,
        }
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"hops": [], "n_valid": 0, "query_valid": False,
                        "error": f"Invalid SMILES: {smiles}"}

            hops = []
            seen = {Chem.MolToSmiles(mol)}

            for name, from_smarts, to_smarts in _BIOISOSTERE_TRANSFORMS:
                if len(hops) >= n_hops:
                    break
                new_smi = _apply_transform(smiles, from_smarts, to_smarts)
                valid = new_smi is not None and new_smi not in seen
                if valid:
                    seen.add(new_smi)
                hops.append({
                    "smiles":         new_smi if valid else smiles,
                    "transformation": name,
                    "valid":          valid,
                })

            n_valid = sum(1 for h in hops if h["valid"])
            return {"hops": hops, "n_valid": n_valid, "query_valid": True}

        except Exception as exc:
            return {"hops": [], "n_valid": 0, "query_valid": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 7. LEAD OPTIMIZER TOOL
# ---------------------------------------------------------------------------

# Modification recipes: (goal, smarts_from, smarts_to, rationale)
_OPT_RECIPES = {
    "potency": [
        (
            "[c:1][H]",
            "[c:1]F",
            "Add fluorine to aromatic ring — blocks CYP oxidation and can improve potency via C–F···π or C–H···F interactions",
        ),
        (
            "[c:1][H]",
            "[c:1]Cl",
            "Add chlorine — increases lipophilicity and enhances hydrophobic binding pocket contacts",
        ),
        (
            "[NH1:1][C:2](=O)",
            "[NH1:1][C:2](=O)[C@@H](N)C",
            "Add alpha-methyl group to amide — conformational rigidification can improve binding selectivity",
        ),
    ],
    "solubility": [
        (
            "[c:1][H]",
            "[c:1]OC",
            "Add methoxy group — improves aqueous solubility by increasing polarity",
        ),
        (
            "[C:1](=O)[N:2]",
            "[C:1](=O)[N:2]CC[OH]",
            "Append hydroxyl-ethyl to nitrogen — HBD addition improves solubility and reduces logP",
        ),
        (
            "[c:1][H]",
            "[c:1]CN1CCNCC1",
            "Introduce piperazinylmethyl — classic solubility handle; basic amine raises aqueous solubility",
        ),
    ],
    "selectivity": [
        (
            "[c:1][H]",
            "[c:1]C(C)(C)C",
            "Add tert-butyl — steric bulk exploits CDK4 vs CDK6 pocket volume differences",
        ),
        (
            "[CH2:1][N:2]",
            "[C@@H:1](C)[N:2]",
            "Alpha-methyl substitution — rigidifies conformation, may improve isoform selectivity",
        ),
        (
            "[c:1][H]",
            "[c:1]C#N",
            "Add nitrile — hydrogen-bond acceptor that can discriminate between binding sites",
        ),
    ],
}


@dataclass
class LeadOptimizerTool:
    """Suggest medicinal chemistry modifications to improve a lead compound."""

    name: str = field(default="lead_optimizer", init=False)
    description: str = field(
        default=(
            "Suggest SMARTS-based modifications for potency, solubility, or selectivity. "
            "Args: smiles (str), optimize_for (str='potency'). "
            "Returns up to 3 suggestions with rationale and new SMILES."
        ),
        init=False,
    )

    def run(self, smiles: str, optimize_for: str = "potency") -> Dict:
        """
        Parameters
        ----------
        smiles       : lead molecule SMILES
        optimize_for : "potency" | "solubility" | "selectivity"

        Returns
        -------
        {
            "suggestions": [
                {"modification": str, "rationale": str, "new_smiles": str, "valid": bool},
                ...
            ],
            "optimize_for": str,
            "n_valid": int,
        }
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"suggestions": [], "optimize_for": optimize_for, "n_valid": 0,
                        "error": f"Invalid SMILES: {smiles}"}

            goal = optimize_for.lower()
            if goal not in _OPT_RECIPES:
                goal = "potency"

            recipes = _OPT_RECIPES[goal]
            suggestions = []
            seen = {Chem.MolToSmiles(mol)}

            for from_sma, to_sma, rationale in recipes:
                new_smi = _apply_transform(smiles, from_sma, to_sma)
                valid = new_smi is not None and new_smi not in seen
                if valid:
                    seen.add(new_smi)
                suggestions.append({
                    "modification": f"{from_sma} → {to_sma}",
                    "rationale":    rationale,
                    "new_smiles":   new_smi if valid else smiles,
                    "valid":        valid,
                })

            n_valid = sum(1 for s in suggestions if s["valid"])
            return {
                "suggestions":  suggestions,
                "optimize_for": optimize_for,
                "n_valid":      n_valid,
            }

        except Exception as exc:
            return {"suggestions": [], "optimize_for": optimize_for,
                    "n_valid": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# 8. LITERATURE SEARCH TOOL
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE = {
    "CDK4/6": {
        "keywords": ["cdk4", "cdk6", "cdk4/6", "cyclin", "palbociclib", "ribociclib",
                     "abemaciclib", "kinase", "cell cycle", "rb", "retinoblastoma"],
        "summary": (
            "CDK4/6 inhibitors block cyclin-dependent kinases 4 and 6, preventing "
            "Rb phosphorylation and arresting the cell cycle at G1/S. Key pharmacophore: "
            "hinge-binding NH (critical HBD), hydrophobic back-pocket (aromatic ring), "
            "solvent-exposed basic amine (improves solubility and selectivity). "
            "Therapeutic pIC50 range: >8.0. Primary metabolising enzyme: CYP3A4. "
            "Approved inhibitors share a pyrimidine or pyrido[2,3-d]pyrimidine core."
        ),
    },
    "hERG": {
        "keywords": ["herg", "qtc", "qt prolongation", "cardiac", "ion channel",
                     "potassium", "cardiotoxicity", "torsades"],
        "summary": (
            "hERG (Kv11.1) channel blockade causes QT prolongation and potentially "
            "fatal torsades de pointes. Key risk factors: basic amine (pKa > 7.0), "
            "MW > 450, multiple aromatic rings, lipophilic cationic amphiphiles. "
            "Mitigation strategies: reduce basicity (pKa < 6.5), introduce polar groups "
            "to reduce logP, add bulky ortho-substituents to aromatic rings. "
            "In vitro hERG IC50 > 30× therapeutic free Cmax is generally acceptable."
        ),
    },
    "DILI": {
        "keywords": ["dili", "liver", "hepatotoxicity", "hepatic", "jaundice",
                     "alt", "ast", "reactive metabolite", "idiosyncratic"],
        "summary": (
            "Drug-induced liver injury (DILI) is a leading cause of post-market "
            "withdrawal. Key risk factors: reactive metabolite formation (quinones, "
            "epoxides, Michael acceptors), mitochondrial liability (high logP, "
            "uncoupling), elevated daily dose (>100 mg/day), biliary excretion. "
            "Structural alerts: anilines, nitroaromatics, hydrazines, thiophenes. "
            "Minimise covalent binding potential and reactive intermediate formation."
        ),
    },
    "CYP3A4": {
        "keywords": ["cyp3a4", "cyp", "cytochrome", "p450", "metabolism",
                     "clearance", "ddi", "drug interaction", "inhibition"],
        "summary": (
            "CYP3A4 is responsible for ~50% of oxidative drug metabolism. "
            "Substrates tend to be large (MW > 400), lipophilic (logP > 2), and "
            "aromatic. Strong inhibitors include ketoconazole, ritonavir, and "
            "clarithromycin — co-administration increases substrate exposure significantly. "
            "Inducers (rifampicin, carbamazepine) accelerate clearance. "
            "For CDK4/6 inhibitors, CYP3A4 DDI should be assessed early; "
            "palbociclib and ribociclib are sensitive CYP3A4 substrates."
        ),
    },
    "solubility": {
        "keywords": ["solubility", "dissolution", "aqueous", "logd", "logp",
                     "bcs", "absorption", "bioavailability", "formulation"],
        "summary": (
            "Aqueous solubility is a major determinant of oral absorption. "
            "BCS Class II/IV compounds (low solubility) require formulation strategies. "
            "Solubility positively correlates with polar surface area and HBD/HBA count; "
            "negatively with logP and crystal packing energy. "
            "Design strategies: introduce ionisable groups (piperazine, morpholine), "
            "reduce logP below 3, add hydroxyl or amide groups. "
            "Target: thermodynamic solubility > 50 µg/mL for oral drugs."
        ),
    },
    "selectivity": {
        "keywords": ["selectivity", "cdk2", "off-target", "kinome", "specificity",
                     "isoform", "profiling", "panel"],
        "summary": (
            "CDK4/6 selectivity over CDK2 is important to avoid anti-proliferative "
            "toxicity in normal tissues. CDK4/6 have a bulkier gatekeeper residue "
            "(Phe/Thr vs Phe in CDK2) and a larger hydrophobic back-pocket. "
            "Strategies to improve selectivity: exploit size differences in the "
            "back-pocket with bulky substituents, target CDK4/6-specific allosteric "
            "sites, use conformational restriction to favour the CDK4/6 DFG-out state. "
            "Broader kinase panel profiling at 1 µM is standard practice."
        ),
    },
}


@dataclass
class LiteratureSearchTool:
    """Keyword search against a curated drug discovery knowledge base."""

    name: str = field(default="literature_search", init=False)
    description: str = field(
        default=(
            "Search curated knowledge base covering CDK4/6, hERG, DILI, CYP3A4, "
            "solubility, and selectivity. Args: query (str). "
            "Returns top-2 most relevant entries."
        ),
        init=False,
    )

    def run(self, query: str, top_k: int = 2) -> Dict:
        """
        Parameters
        ----------
        query : free-text search string
        top_k : number of results to return

        Returns
        -------
        {"results": [{"topic", "summary", "relevance"}, ...]}
        """
        try:
            q = query.lower()
            scored = []
            for topic, data in _KNOWLEDGE_BASE.items():
                hits = sum(1 for kw in data["keywords"] if kw in q)
                # Partial credit: topic name itself in query
                if topic.lower() in q:
                    hits += 3
                if hits > 0:
                    scored.append((hits, topic, data["summary"]))

            # Fallback: return top-2 by keyword density if nothing matched
            if not scored:
                scored = [(0, t, d["summary"]) for t, d in _KNOWLEDGE_BASE.items()]

            scored.sort(key=lambda x: x[0], reverse=True)
            results = [
                {"topic": topic, "summary": summary, "relevance": hits}
                for hits, topic, summary in scored[:top_k]
            ]
            return {"results": results}

        except Exception as exc:
            return {"results": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# 9. MOLECULE VISUALIZER TOOL
# ---------------------------------------------------------------------------

@dataclass
class MoleculeVisualizerTool:
    """Save a matplotlib property-summary figure for a list of molecules."""

    name: str = field(default="visualize", init=False)
    description: str = field(
        default=(
            "Save a property summary figure for up to 8 molecules. "
            "Args: smiles_list (list[str]), title (str='Molecules'). "
            "Returns saved_to path and n_molecules."
        ),
        init=False,
    )

    def run(self, smiles_list: List[str], title: str = "Molecules") -> Dict:
        """
        Returns
        -------
        {"saved_to": str, "n_molecules": int}
        """
        try:
            valid = [(s, Chem.MolFromSmiles(s)) for s in smiles_list]
            valid = [(s, m) for s, m in valid if m is not None][:8]

            if not valid:
                return {"saved_to": None, "n_molecules": 0,
                        "error": "No valid SMILES provided"}

            n = len(valid)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(ncols * 4, nrows * 3.5))
            axes = np.array(axes).flatten() if n > 1 else [axes]

            for ax, (smi, mol) in zip(axes, valid):
                mw   = Descriptors.ExactMolWt(mol)
                logp = Crippen.MolLogP(mol)
                qed  = QED.qed(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                hbd  = rdMolDescriptors.CalcNumHBD(mol)
                hba  = rdMolDescriptors.CalcNumHBA(mol)

                props = (
                    f"MW={mw:.0f}  LogP={logp:.1f}\n"
                    f"QED={qed:.2f}  TPSA={tpsa:.0f}\n"
                    f"HBD={hbd}  HBA={hba}"
                )
                ax.text(0.5, 0.55, smi[:40] + ("…" if len(smi) > 40 else ""),
                        ha="center", va="center", fontsize=7,
                        wrap=True, transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", ec="#90caf9"))
                ax.text(0.5, 0.18, props,
                        ha="center", va="center", fontsize=8,
                        transform=ax.transAxes, family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#f1f8e9", ec="#a5d6a7"))
                ax.axis("off")

            # Hide unused axes
            for ax in axes[n:]:
                ax.axis("off")

            fig.suptitle(title, fontsize=13, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
            out_path = f"/tmp/molecules_{safe_title}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            return {"saved_to": out_path, "n_molecules": n}

        except Exception as exc:
            return {"saved_to": None, "n_molecules": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# 10. TOOL REGISTRY
# ---------------------------------------------------------------------------

# Instantiate all tools (BindingAffinityPredictorTool trains its RF here)
_TOOL_INSTANCES = [
    SimilaritySearchTool(),
    PropertyCalculatorTool(),
    ADMETPredictorTool(),
    BindingAffinityPredictorTool(),
    ScaffoldHopperTool(),
    LeadOptimizerTool(),
    LiteratureSearchTool(),
    MoleculeVisualizerTool(),
]

TOOLS: Dict[str, Any] = {t.name: t for t in _TOOL_INSTANCES}


def execute_tool(tool_name: str, **kwargs) -> Dict:
    """
    Dispatch a tool call by name.

    Parameters
    ----------
    tool_name : one of the keys in TOOLS
    **kwargs  : forwarded to tool.run()

    Returns
    -------
    dict — tool result (always a dict, never raises)
    """
    if tool_name not in TOOLS:
        return {"error": f"Unknown tool '{tool_name}'. Available: {list(TOOLS)}"}
    try:
        return TOOLS[tool_name].run(**kwargs)
    except Exception as exc:
        return {"error": f"Tool '{tool_name}' raised: {exc}"}


def get_tool_descriptions() -> str:
    """Return a formatted string of all tool names and descriptions."""
    lines = ["Available tools:", ""]
    for name, tool in TOOLS.items():
        lines.append(f"  {name}")
        lines.append(f"    {tool.description}")
        lines.append("")
    return "\n".join(lines)
