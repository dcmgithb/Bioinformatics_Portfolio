"""
metabolite_toxicity.py — Metabolite Toxicity Profiling Module
=============================================================
Structural alert detection, ML-based endpoint prediction, and composite
safety scoring for drug metabolites.

Endpoints modelled
------------------
  DILI              — Drug-Induced Liver Injury
  Mutagenicity      — Ames test proxy (genotoxic alerts)
  hERG_inhibition   — Cardiac ion-channel blockade (QT risk)
  Hepatotoxicity    — Broad hepatocellular damage
  GSH_trapping      — Glutathione-reactive metabolites (reactive metabolite flag)
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    QED,
    rdMolDescriptors,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. STRUCTURAL ALERT LIBRARY
# ---------------------------------------------------------------------------

STRUCTURAL_ALERTS: List[Dict] = [
    {
        "id": "quinone",
        "smarts": "[#6]1(=O)[#6]=[#6][#6](=O)[#6]=[#6]1",
        "name": "Para-quinone",
        "category": "reactive",
        "severity": 3,
        "mechanism": "Arylation of protein nucleophiles; GSH conjugation; oxidative stress",
    },
    {
        "id": "ortho_quinone",
        "smarts": "[#6]1(=O)[#6](=O)[#6]=[#6][#6]=[#6]1",
        "name": "Ortho-quinone",
        "category": "reactive",
        "severity": 3,
        "mechanism": "Redox cycling; catechol oxidation product; DILI risk",
    },
    {
        "id": "epoxide",
        "smarts": "[C;X3]1[O]1",
        "name": "Epoxide",
        "category": "reactive",
        "severity": 3,
        "mechanism": "Alkylation of DNA and proteins; mutagenic potential",
    },
    {
        "id": "michael_acceptor",
        "smarts": "[C]=[C]-[C]=O",
        "name": "Michael acceptor (α,β-unsaturated carbonyl)",
        "category": "reactive",
        "severity": 2,
        "mechanism": "1,4-addition by GSH thiol; covalent modification",
    },
    {
        "id": "aldehyde",
        "smarts": "[CH]=O",
        "name": "Aldehyde",
        "category": "reactive",
        "severity": 2,
        "mechanism": "Schiff base formation with Lys/protein; mutagenicity",
    },
    {
        "id": "acyl_halide",
        "smarts": "[C](=O)[F,Cl,Br,I]",
        "name": "Acyl halide",
        "category": "reactive",
        "severity": 3,
        "mechanism": "Highly reactive acylating agent; protein binding",
    },
    {
        "id": "isocyanate",
        "smarts": "[N]=[C]=[O]",
        "name": "Isocyanate",
        "category": "reactive",
        "severity": 3,
        "mechanism": "Carbamoylation of amines; immunogenic hapten formation",
    },
    {
        "id": "nitroso",
        "smarts": "[N]=[O;!$([N](=O)=O)]",
        "name": "Nitroso",
        "category": "genotoxic",
        "severity": 3,
        "mechanism": "DNA alkylation; N-nitroso carcinogen formation",
    },
    {
        "id": "aromatic_amine",
        "smarts": "[NH2,NH1;$([N]c)]",
        "name": "Aromatic amine",
        "category": "genotoxic",
        "severity": 2,
        "mechanism": "N-hydroxylation → nitrenium ion → DNA adducts",
    },
    {
        "id": "nitro",
        "smarts": "[$([N+](=O)[O-]),$([N](=O)=O)]",
        "name": "Nitro group",
        "category": "genotoxic",
        "severity": 2,
        "mechanism": "Nitroreduction to hydroxylamine/nitroso; mutagenic",
    },
    {
        "id": "hydrazine",
        "smarts": "[NH][NH]",
        "name": "Hydrazine",
        "category": "genotoxic",
        "severity": 3,
        "mechanism": "Free radical generation; DILI; monoamine oxidase inhibition",
    },
    {
        "id": "halo_alkene",
        "smarts": "[Cl,Br,I][C]=[C]",
        "name": "Vinyl halide",
        "category": "genotoxic",
        "severity": 2,
        "mechanism": "Bioactivation to haloacetaldehyde; DNA adducts",
    },
    {
        "id": "peroxide",
        "smarts": "[O]-[O]",
        "name": "Peroxide / hydroperoxide",
        "category": "reactive",
        "severity": 2,
        "mechanism": "Oxidative stress; Fenton reaction; lipid peroxidation",
    },
    {
        "id": "anhydride",
        "smarts": "[C](=O)[O][C](=O)",
        "name": "Acid anhydride",
        "category": "reactive",
        "severity": 2,
        "mechanism": "Acylation of nucleophilic residues; protein modification",
    },
    {
        "id": "aziridine",
        "smarts": "[N]1[C][C]1",
        "name": "Aziridine",
        "category": "genotoxic",
        "severity": 3,
        "mechanism": "Ring-opening alkylation of DNA; antineoplastic but mutagenic",
    },
    {
        "id": "thiirane",
        "smarts": "[S]1[C][C]1",
        "name": "Thiirane (episulfide)",
        "category": "reactive",
        "severity": 2,
        "mechanism": "Alkylation similar to epoxide; CYP-mediated bioactivation",
    },
    {
        "id": "diazo",
        "smarts": "[N]=[N+]=[#6]",
        "name": "Diazo compound",
        "category": "genotoxic",
        "severity": 3,
        "mechanism": "Carbocation formation; DNA alkylation",
    },
    {
        "id": "beta_lactam",
        "smarts": "[N]1[C](=O)[C][C]1",
        "name": "β-Lactam",
        "category": "reactive",
        "severity": 1,
        "mechanism": "Acylation of penicillin-binding proteins and serine residues",
    },
    {
        "id": "quinone_methide",
        "smarts": "[CH2]=[c](-[OH])",
        "name": "Quinone methide precursor",
        "category": "reactive",
        "severity": 2,
        "mechanism": "Tautomerises to electrophilic quinone methide; protein alkylation",
    },
    {
        "id": "thiol_reactive",
        "smarts": "[S;H1,H0;!$([S](=O)(=O))]",
        "name": "Thiol / sulfide",
        "category": "reactive",
        "severity": 1,
        "mechanism": "Disulfide exchange; metal chelation; DILI in high dose",
    },
]

# Compile SMARTS patterns once
_ALERT_MOLS: List[Chem.Mol | None] = []
for _a in STRUCTURAL_ALERTS:
    try:
        _ALERT_MOLS.append(Chem.MolFromSmarts(_a["smarts"]))
    except Exception:
        _ALERT_MOLS.append(None)


# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def _count_basic_nitrogens(mol: Chem.Mol) -> int:
    """Estimate number of basic nitrogen atoms (pKa proxy)."""
    basic_smarts = Chem.MolFromSmarts("[N;!$(N-C=O);!$(N~[#6]=[#7,#8,#16]);H1,H2,H3]")
    if basic_smarts is None:
        return 0
    matches = mol.GetSubstructMatches(basic_smarts)
    return len(matches)


def mol_to_tox_features(smiles: str) -> np.ndarray | None:
    """
    Convert SMILES to toxicity feature vector.

    Layout (1221 dims total):
      [0:1024]  Morgan FP radius=2 (1024 bits)
      [1024:1191] MACCS keys (167 bits)
      [1191:1211] Structural alert counts (20 alerts)
      [1211:1221] Physicochemical descriptors (10)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan FP
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    morgan = np.array(fp, dtype=np.float32)

    # MACCS keys
    maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    maccs = np.array(maccs_fp, dtype=np.float32)

    # Structural alert counts
    alert_counts = np.zeros(len(STRUCTURAL_ALERTS), dtype=np.float32)
    for i, pat in enumerate(_ALERT_MOLS):
        if pat is not None:
            alert_counts[i] = float(len(mol.GetSubstructMatches(pat)))

    # Physicochemical descriptors
    mw = Descriptors.ExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    nrotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_het = rdMolDescriptors.CalcNumHeteroatoms(mol)
    n_basic = float(_count_basic_nitrogens(mol))

    physchem = np.array(
        [mw, logp, tpsa, hbd, hba, nrotb, n_ar, n_rings, n_het, n_basic],
        dtype=np.float32,
    )

    return np.concatenate([morgan, maccs, alert_counts, physchem])


# ---------------------------------------------------------------------------
# 3. SYNTHETIC DATASET
# ---------------------------------------------------------------------------

# Scaffold library: 30 drug-like fragments
_SCAFFOLDS = [
    "c1ccc(N)cc1",                  # aniline
    "c1ccc(O)cc1",                  # phenol
    "c1ccc(-c2ccccn2)cc1",          # 2-phenylpyridine
    "c1cnc2ccccc2c1",               # isoquinoline
    "c1ccc2[nH]ccc2c1",             # indole
    "c1ccc2c(c1)CCNC2",             # tetrahydroisoquinoline
    "C1CCNCC1",                     # piperidine
    "C1COCCN1",                     # morpholine
    "c1ccc(C(=O)O)cc1",             # benzoic acid
    "c1cc([NH2])ccc1O",             # aminophenol
    "c1ccc(Cl)cc1",                 # chlorobenzene
    "c1ccc(Br)cc1",                 # bromobenzene
    "c1ccc(F)cc1",                  # fluorobenzene
    "c1ccc([N+](=O)[O-])cc1",       # nitrobenzene
    "CC(=O)Nc1ccccc1",              # acetanilide
    "Cc1ccccc1NC(=O)C",             # N-methyl acetanilide
    "c1ccc(CC(=O)O)cc1",            # phenylacetic acid
    "c1cc2ccccc2cc1",               # naphthalene
    "c1ccc2ccccc2c1",               # naphthalene isomer
    "c1ccc(-c2ccccc2)cc1",          # biphenyl
    "Cc1nc2ccccc2o1",               # benzoxazole
    "c1ccc2sc(N)nc2c1",             # benzothiazolamine
    "c1cnc2ccccn12",                # imidazo[1,2-a]pyridine
    "C(F)(F)F",                     # CF3
    "CC(C)CC(=O)O",                 # isobutyric acid
    "CC(=O)c1ccccc1",               # acetophenone
    "CCCC(=O)O",                    # butanoic acid
    "c1ccc(SC)cc1",                 # thioanisole
    "c1ccc(OC)cc1",                 # anisole
    "c1ccc(CC#N)cc1",               # phenylacetonitrile
]


def _generate_smiles_from_scaffold(scaffold: str, rng: np.random.Generator) -> str:
    """Attach a random substituent to a scaffold."""
    substituents = [
        "", "C", "CC", "CCC", "OC", "OCC", "F", "Cl", "N", "NC",
        "C(=O)O", "C(=O)N", "S(=O)(=O)N", "CF", "CCF",
    ]
    # Try to add substituent; if invalid, return scaffold
    mol = Chem.MolFromSmiles(scaffold)
    if mol is None:
        return scaffold
    sub = rng.choice(substituents)
    if sub == "":
        return scaffold
    new_smi = scaffold.rstrip() + sub
    test = Chem.MolFromSmiles(new_smi)
    if test is not None:
        return Chem.MolToSmiles(test)
    return scaffold


def generate_tox_dataset(
    n: int = 500, seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic toxicity dataset.

    Returns
    -------
    df : DataFrame with columns [smiles, DILI, mutagenicity, hERG_inhibition,
                                  hepatotoxicity, GSH_trapping, MW, LogP, …]
    X  : Feature matrix (n × 1221)
    """
    rng = np.random.default_rng(seed)
    rows = []
    features = []

    scaffold_cycle = _SCAFFOLDS * (n // len(_SCAFFOLDS) + 2)
    rng.shuffle(scaffold_cycle)

    idx = 0
    for scaffold in scaffold_cycle:
        if len(rows) >= n:
            break
        smi = _generate_smiles_from_scaffold(scaffold, rng)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        feat = mol_to_tox_features(smi)
        if feat is None:
            continue

        # Alert flags
        alert_flags: Dict[str, bool] = {}
        for i, alert in enumerate(STRUCTURAL_ALERTS):
            alert_flags[alert["id"]] = feat[1191 + i] > 0

        # Physicochemical
        mw, logp, tpsa, hbd, hba, nrotb, n_ar, n_rings, n_het, n_basic = feat[1211:]

        # SAR-based labelling
        dili = int(
            (alert_flags.get("quinone", False) or
             alert_flags.get("michael_acceptor", False) or
             alert_flags.get("aromatic_amine", False))
            and mw < 600
        )
        mutagenicity = int(
            alert_flags.get("nitro", False) or
            alert_flags.get("aromatic_amine", False) or
            alert_flags.get("diazo", False) or
            alert_flags.get("halo_alkene", False) or
            alert_flags.get("hydrazine", False)
        )
        herg = int(n_ar >= 3 and n_basic >= 1 and mw > 300)
        n_alerts = sum(1 for v in alert_flags.values() if v)
        hepatotox = int(dili or (n_alerts >= 2 and logp > 2))
        gsh = int(
            alert_flags.get("michael_acceptor", False) or
            alert_flags.get("epoxide", False) or
            alert_flags.get("quinone", False) or
            alert_flags.get("ortho_quinone", False)
        )

        # 10% label noise
        noise = rng.random(5) < 0.10
        dili = int(dili) ^ int(noise[0])
        mutagenicity = int(mutagenicity) ^ int(noise[1])
        herg = int(herg) ^ int(noise[2])
        hepatotox = int(hepatotox) ^ int(noise[3])
        gsh = int(gsh) ^ int(noise[4])

        rows.append({
            "smiles": smi,
            "DILI": dili,
            "mutagenicity": mutagenicity,
            "hERG_inhibition": herg,
            "hepatotoxicity": hepatotox,
            "GSH_trapping": gsh,
            "MW": float(mw),
            "LogP": float(logp),
            "TPSA": float(tpsa),
            "HBD": int(hbd),
            "HBA": int(hba),
            "nAr": int(n_ar),
            "n_basic": int(n_basic),
        })
        features.append(feat)
        idx += 1

    df = pd.DataFrame(rows[:n])
    X = np.vstack(features[:n])
    return df, X


# ---------------------------------------------------------------------------
# 4. TOXICITY PROFILER CLASS
# ---------------------------------------------------------------------------

class ToxicityProfiler:
    """
    Multi-endpoint toxicity predictor combining structural alerts with
    calibrated GBM classifiers.
    """

    ENDPOINTS = ["DILI", "mutagenicity", "hERG_inhibition", "hepatotoxicity", "GSH_trapping"]

    # Endpoint-specific weights for safety score calculation (higher = more critical)
    _WEIGHTS = {
        "DILI": 0.30,
        "mutagenicity": 0.30,
        "hERG_inhibition": 0.20,
        "hepatotoxicity": 0.15,
        "GSH_trapping": 0.05,
    }

    def __init__(self) -> None:
        self._models: Dict[str, CalibratedClassifierCV] = {}
        self._scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, smiles_list: List[str], labels_df: pd.DataFrame) -> None:
        """Train one GBM per endpoint with Platt calibration."""
        X_list = []
        valid_idx = []
        for i, smi in enumerate(smiles_list):
            feat = mol_to_tox_features(smi)
            if feat is not None:
                X_list.append(feat)
                valid_idx.append(i)

        X = np.vstack(X_list)
        X_scaled = self._scaler.fit_transform(X)
        labels_df = labels_df.iloc[valid_idx].reset_index(drop=True)

        for ep in self.ENDPOINTS:
            y = labels_df[ep].values
            base = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
            cal = CalibratedClassifierCV(base, cv=3, method="sigmoid")
            cal.fit(X_scaled, y)
            self._models[ep] = cal

        self._fitted = True

    # ------------------------------------------------------------------
    def predict_proba(self, smiles: str) -> Dict[str, float]:
        """Return dict of endpoint → probability (0–1)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        feat = mol_to_tox_features(smiles)
        if feat is None:
            return {ep: 0.0 for ep in self.ENDPOINTS}
        X = self._scaler.transform(feat.reshape(1, -1))
        return {ep: float(self._models[ep].predict_proba(X)[0, 1]) for ep in self.ENDPOINTS}

    # ------------------------------------------------------------------
    def check_structural_alerts(self, smiles: str) -> List[Dict]:
        """Return list of triggered structural alerts."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        triggered = []
        for alert, pat in zip(STRUCTURAL_ALERTS, _ALERT_MOLS):
            if pat is not None and mol.HasSubstructMatch(pat):
                triggered.append(dict(alert))
        return triggered

    # ------------------------------------------------------------------
    def full_toxicity_profile(self, smiles: str) -> Dict:
        """Combine structural alerts and ML endpoint probabilities."""
        alerts = self.check_structural_alerts(smiles)
        probs = self.predict_proba(smiles) if self._fitted else {}
        score = self.safety_score(smiles)
        return {
            "smiles": smiles,
            "structural_alerts": alerts,
            "n_alerts": len(alerts),
            "max_alert_severity": max((a["severity"] for a in alerts), default=0),
            "alert_categories": list({a["category"] for a in alerts}),
            "tox_probs": probs,
            "safety_score": score,
        }

    # ------------------------------------------------------------------
    def safety_score(self, smiles: str) -> float:
        """
        Composite safety score 0–100 (higher = safer).

        Deductions:
          • ML endpoint risk (weighted)
          • Structural alert severity penalty
        """
        probs = self.predict_proba(smiles) if self._fitted else {ep: 0.0 for ep in self.ENDPOINTS}
        alerts = self.check_structural_alerts(smiles)

        # ML-based deduction (0–70 points)
        ml_risk = sum(probs.get(ep, 0.0) * self._WEIGHTS[ep] for ep in self.ENDPOINTS)
        ml_deduction = ml_risk * 70.0

        # Alert-based deduction (0–30 points)
        severity_sum = sum(a["severity"] for a in alerts)
        alert_deduction = min(30.0, severity_sum * 5.0)

        return max(0.0, 100.0 - ml_deduction - alert_deduction)


# ---------------------------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------------------------

def _safety_gauge(ax: plt.Axes, score: float) -> None:
    """Half-circle gauge showing safety score."""
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.2, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Safety Score", fontsize=11, fontweight="bold")

    # Background arc (grey)
    theta = np.linspace(np.pi, 0, 200)
    ax.fill_between(np.cos(theta), np.sin(theta),
                    0.75 * np.sin(theta), color="#e0e0e0", zorder=1)

    # Colour zones
    zones = [(0, 40, "#d32f2f"), (40, 70, "#f57c00"), (70, 100, "#388e3c")]
    for lo, hi, col in zones:
        t0 = np.pi * (1 - lo / 100)
        t1 = np.pi * (1 - hi / 100)
        th = np.linspace(t0, t1, 50)
        ax.fill_between(np.cos(th), np.sin(th), 0.75 * np.sin(th), color=col, alpha=0.6, zorder=2)

    # Needle
    needle_angle = np.pi * (1 - score / 100)
    ax.annotate("", xy=(0.9 * np.cos(needle_angle), 0.9 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    ax.text(0, -0.15, f"{score:.0f}/100", ha="center", va="center",
            fontsize=16, fontweight="bold",
            color="#d32f2f" if score < 40 else "#f57c00" if score < 70 else "#388e3c")


def plot_toxicity_dashboard(
    profiler: ToxicityProfiler,
    query_smiles: str,
    query_name: str,
    train_X: np.ndarray,
    train_labels: pd.DataFrame,
    out_path: str = "toxicity_dashboard.png",
) -> None:
    """5-panel toxicity dashboard for a query molecule."""

    profile = profiler.full_toxicity_profile(query_smiles)
    probs = profile["tox_probs"]
    alerts = profile["structural_alerts"]
    score = profile["safety_score"]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Toxicity Dashboard — {query_name}", fontsize=15, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ── Panel 1: Radar chart ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    eps = ToxicityProfiler.ENDPOINTS
    values = [probs.get(ep, 0.0) for ep in eps]
    N = len(eps)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    values_plot = values + values[:1]
    ax1.plot(angles, values_plot, "o-", color="#d32f2f", lw=2)
    ax1.fill(angles, values_plot, alpha=0.25, color="#d32f2f")
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([e.replace("_", "\n") for e in eps], fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.25, 0.5, 0.75])
    ax1.set_yticklabels(["0.25", "0.5", "0.75"], fontsize=7)
    ax1.set_title("Endpoint Risk Radar", fontsize=11, fontweight="bold", pad=15)

    # ── Panel 2: Structural alerts ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    all_names = [a["name"][:28] for a in STRUCTURAL_ALERTS]
    triggered_ids = {a["id"] for a in alerts}
    colors_bar = ["#d32f2f" if a["id"] in triggered_ids else "#81c784"
                  for a in STRUCTURAL_ALERTS]
    y_pos = range(len(STRUCTURAL_ALERTS))
    ax2.barh(list(y_pos), [1] * len(STRUCTURAL_ALERTS), color=colors_bar, alpha=0.85)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(all_names, fontsize=7)
    ax2.set_xticks([])
    ax2.set_title(f"Structural Alerts  ({len(alerts)} triggered)", fontsize=11, fontweight="bold")
    red_p = mpatches.Patch(color="#d32f2f", label="Triggered")
    grn_p = mpatches.Patch(color="#81c784", label="Clear")
    ax2.legend(handles=[red_p, grn_p], fontsize=8, loc="lower right")

    # ── Panel 3: Endpoint probabilities ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    prob_vals = [probs.get(ep, 0.0) for ep in eps]
    bar_colors = ["#d32f2f" if v > 0.5 else "#f57c00" if v > 0.3 else "#81c784"
                  for v in prob_vals]
    bars = ax3.barh(eps, prob_vals, color=bar_colors, alpha=0.85)
    ax3.set_xlim(0, 1)
    ax3.axvline(0.5, color="grey", linestyle="--", lw=1, alpha=0.6)
    ax3.set_xlabel("Predicted Probability")
    ax3.set_title("Endpoint Probabilities (ML)", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, prob_vals):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)

    # ── Panel 4: Safety gauge ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _safety_gauge(ax4, score)

    # ── Panel 5: Chemical space PCA ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(train_X[:, :1024])  # Use Morgan FP only

    dili_labels = train_labels["DILI"].values
    sc = ax5.scatter(X_pca[dili_labels == 0, 0], X_pca[dili_labels == 0, 1],
                     c="#81c784", alpha=0.4, s=15, label="DILI-neg")
    sc2 = ax5.scatter(X_pca[dili_labels == 1, 0], X_pca[dili_labels == 1, 1],
                      c="#d32f2f", alpha=0.4, s=15, label="DILI-pos")

    # Project query
    q_feat = mol_to_tox_features(query_smiles)
    if q_feat is not None:
        q_pca = pca.transform(q_feat[:1024].reshape(1, -1))
        ax5.scatter(q_pca[0, 0], q_pca[0, 1], marker="*", s=300,
                    c="gold", edgecolors="black", zorder=5, label=query_name)

    ax5.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax5.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax5.set_title("Chemical Space (PCA, coloured by DILI)", fontsize=11, fontweight="bold")
    ax5.legend(fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard saved → {out_path}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("  METABOLITE TOXICITY PROFILING PLATFORM")
    print("=" * 64)

    # ── Generate synthetic training data ────────────────────────────────
    print("\n[1/3] Generating synthetic training dataset (n=500)…")
    df_train, X_train = generate_tox_dataset(n=500, seed=42)
    print(f"  Dataset shape : {X_train.shape}")
    for ep in ToxicityProfiler.ENDPOINTS:
        pos = df_train[ep].sum()
        print(f"  {ep:<20} positives: {pos}/{len(df_train)}  ({pos/len(df_train):.0%})")

    # ── Train profiler ───────────────────────────────────────────────────
    print("\n[2/3] Training ToxicityProfiler (5 endpoints, calibrated GBM)…")
    profiler = ToxicityProfiler()
    profiler.fit(df_train["smiles"].tolist(), df_train[ToxicityProfiler.ENDPOINTS])
    print("  Training complete.")

    # ── Test compounds ───────────────────────────────────────────────────
    print("\n[3/3] Predicting toxicity for benchmark compounds…\n")

    test_drugs = [
        ("CC(=O)Nc1ccc(O)cc1",
         "Acetaminophen",
         "CYP2E1 → NAPQI quinone; major DILI risk in overdose"),
        ("O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
         "Diclofenac",
         "Acyl glucuronide + CYP3A4 reactive metabolites; idiosyncratic DILI"),
        ("[H][C@@]12CC[C@H](C(C)=O)[C@@]1(C)CC[C@@]1(C)[C@@H]2CC[C@@H]1O",
         "Cortisol (steroid reference)",
         "Phase II conjugation; low structural alert load"),
        ("c1ccc2c(c1)-c1ccccc1N2",
         "Acridine",
         "Polycyclic amine; intercalation + mutagenicity concern"),
        ("CC(N)Cc1ccc(O)cc1",
         "Tyramine",
         "MAO substrate; phenylethylamine class; aromatic amine"),
    ]

    print(f"{'Compound':<25} {'Score':>6} {'DILI':>6} {'Mutag':>6} {'hERG':>6} "
          f"{'Hepato':>7} {'GSH':>6}  Alerts")
    print("-" * 85)

    for smi, name, note in test_drugs:
        profile = profiler.full_toxicity_profile(smi)
        probs = profile["tox_probs"]
        alerts = profile["structural_alerts"]
        score = profile["safety_score"]
        alert_names = ", ".join(a["name"] for a in alerts) if alerts else "none"
        print(
            f"{name:<25} {score:>5.0f}  "
            f"{probs['DILI']:>5.2f}  "
            f"{probs['mutagenicity']:>5.2f}  "
            f"{probs['hERG_inhibition']:>5.2f}  "
            f"{probs['hepatotoxicity']:>6.2f}  "
            f"{probs['GSH_trapping']:>5.2f}  "
            f"{alert_names[:35]}"
        )
        print(f"  ({note})\n")

    # ── Save dashboard for acetaminophen ────────────────────────────────
    print("\nGenerating toxicity dashboard for Acetaminophen…")
    out = os.path.join(os.path.dirname(__file__) or ".", "toxicity_dashboard.png")
    plot_toxicity_dashboard(
        profiler,
        query_smiles="CC(=O)Nc1ccc(O)cc1",
        query_name="Acetaminophen",
        train_X=X_train,
        train_labels=df_train[ToxicityProfiler.ENDPOINTS],
        out_path=out,
    )

    print("\nDone.")
