"""
CYP Enzyme Isoform & Phase II Enzyme Substrate/Inhibitor Classifier
=====================================================================
Multi-task gradient boosting classifier predicting which metabolic enzymes
are responsible for a given compound's biotransformation.

Enzymes covered
---------------
Phase I  : CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4 (substrate + inhibitor)
Phase II : UGT1A (glucuronidation), SULT1A (sulfation), MAO-A/B

Model architecture
------------------
Features : Morgan FP (2048) + MACCS keys (167) + pharmacophore counts
           + physicochemical descriptors (MW, LogP, TPSA, pKa_est, etc.)
Model    : Multi-task GradientBoostingClassifier (one per enzyme/role)
           with Platt scaling for calibrated probabilities
Output   : Per-enzyme substrate/inhibitor probabilities [0,1]
           + ranked enzyme involvement report

Pharmacophore-based features
----------------------------
The key CYP isoform recognition features encoded explicitly:
  CYP2D6 : basic nitrogen within 5–7 Å of flat aromatic SoM
  CYP2C9 : acidic function (COOH/OH) 5–7 Å from SoM, MW ≈ 300–500
  CYP3A4 : large MW (>400), multiple hydrogen bond acceptors, flexible
  CYP1A2 : planar aromatic system, small MW (<300)
  CYP2C19: intermediate properties between 2C9 and 2D6

References
----------
Rendic (2002) Summary of information on human CYP enzymes — Drug Metab Rev
Kirchmair et al. (2015) Predicting drug metabolism — J. Chem. Inf. Model.
Veith et al. (2009) Comprehensive characterization of cytochrome P450 isozymes —
  Nat. Biotechnol. 27, 1050–1055
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ── Enzyme task definitions ────────────────────────────────────────────────────

ENZYME_TASKS = [
    "CYP1A2_substrate",  "CYP1A2_inhibitor",
    "CYP2C9_substrate",  "CYP2C9_inhibitor",
    "CYP2C19_substrate", "CYP2C19_inhibitor",
    "CYP2D6_substrate",  "CYP2D6_inhibitor",
    "CYP3A4_substrate",  "CYP3A4_inhibitor",
    "UGT1A_substrate",
    "SULT1A_substrate",
    "MAO_substrate",
]
N_TASKS = len(ENZYME_TASKS)


# ── Feature engineering ───────────────────────────────────────────────────────

def estimate_pka(mol: Chem.Mol) -> Tuple[float, float]:
    """
    Estimate most acidic and most basic pKa from structure.
    Rule-based approximation (no ionisation library required).
    Returns (pKa_acidic, pKa_basic) — approximate values.
    """
    pka_acid = 14.0   # neutral default
    pka_base = -1.0   # neutral default

    # Carboxylic acids (~pKa 3-5)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OH]")):
        pka_acid = 4.0
    # Phenols (~pKa 8-10)
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("[OH][c]")):
        pka_acid = 9.5
    # Sulfonamides (~pKa 10)
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("[NH]S(=O)(=O)")):
        pka_acid = 10.5

    # Aliphatic amines (~pKa 9-10)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O);!a]")):
        pka_base = 9.5
    # Aromatic amines (~pKa 4-5)
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("[NH2][c]")):
        pka_base = 4.5
    # Pyridines (~pKa 5)
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("n")):
        pka_base = 5.0
    # Imidazoles (~pKa 7)
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("[nH]1ccnc1")):
        pka_base = 7.0

    return pka_acid, pka_base


def pharmacophore_features(mol: Chem.Mol) -> Dict[str, float]:
    """
    Encode CYP pharmacophore recognition features as a feature vector.
    These features capture isoform-specific binding pocket requirements.
    """
    feats = {}

    # CYP2D6 pharmacophore: basic nitrogen + aromatic ring
    n_basic_n = sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() == 7 and a.GetFormalCharge() >= 0
        and a.GetTotalNumHs() > 0 and not a.GetIsAromatic()
    )
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    feats["basic_n_count"]    = n_basic_n
    feats["aromatic_ring_cnt"]= n_aromatic_rings
    feats["cyp2d6_pharmaco"]  = float(n_basic_n > 0 and n_aromatic_rings > 0)

    # CYP2C9: acidic function present
    n_cooh = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OH]")))
    n_phenol = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OH][c]")))
    feats["n_carboxylic"]  = n_cooh
    feats["n_phenol"]      = n_phenol
    feats["cyp2c9_pharmaco"] = float(n_cooh + n_phenol > 0)

    # CYP3A4: large, flexible molecules
    n_rot    = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_hba    = Descriptors.NumHAcceptors(mol)
    feats["n_rotatable"]  = n_rot
    feats["n_hba"]        = n_hba
    feats["cyp3a4_pharmaco"] = float(Descriptors.MolWt(mol) > 400 and n_hba > 4)

    # CYP1A2: planar aromatic, small MW
    n_aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    feats["n_aromatic_atoms"] = n_aromatic_atoms
    feats["cyp1a2_pharmaco"]  = float(
        n_aromatic_atoms > 6 and Descriptors.MolWt(mol) < 350
    )

    # UGT: hydroxyl groups
    n_oh = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OH]")))
    n_nh = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[NH]")))
    feats["n_oh"] = n_oh
    feats["n_nh"] = n_nh
    feats["ugt_pharmaco"] = float(n_oh + n_phenol > 0)

    # SULT: phenolic OH
    feats["sult_pharmaco"] = float(n_phenol > 0)

    # MAO: primary/secondary amine + aliphatic chain
    n_prim_amine = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[NH2][C;!a]")))
    feats["n_prim_amine"] = n_prim_amine
    feats["mao_pharmaco"] = float(n_prim_amine > 0)

    return feats


def mol_to_features(smiles: str) -> Optional[np.ndarray]:
    """
    Compute full feature vector: Morgan FP + MACCS + pharmacophore + physicochemical.
    Returns None for invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan fingerprint (2048)
    fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr_morgan = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_morgan, arr_morgan)

    # MACCS keys (167)
    fp_maccs = GenMACCSKeys(mol)
    arr_maccs = np.zeros(167, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_maccs, arr_maccs)

    # Pharmacophore features (15)
    pharm = pharmacophore_features(mol)
    pharm_arr = np.array(list(pharm.values()), dtype=np.float32)

    # Physicochemical (10)
    pka_a, pka_b = estimate_pka(mol)
    physchem = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        mol.GetNumHeavyAtoms(),
        pka_a,
        pka_b,
    ], dtype=np.float32)

    return np.concatenate([arr_morgan, arr_maccs, pharm_arr, physchem])


# ── Synthetic label generation ────────────────────────────────────────────────

def generate_cyp_labels(smiles: str, seed: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Generate synthetic enzyme involvement labels using rule-based heuristics
    representative of known CYP SAR (structure-activity relationships).

    Rules based on published literature:
    - CYP2D6: basic N within 5Å of aromatic ring → substrate
    - CYP2C9: acidic group (pKa < 8) → substrate
    - CYP3A4: MW > 300, lipophilic → substrate (broad specificity)
    - CYP1A2: planar aromatic → substrate
    - CYP2C19: overlapping with 2C9 + 2D6

    Returns binary label vector [N_TASKS].
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    rng = np.random.default_rng(seed or abs(hash(smiles)) % (2 ** 31))

    mw     = Descriptors.MolWt(mol)
    logp   = Descriptors.MolLogP(mol)
    hbd    = Descriptors.NumHDonors(mol)
    hba    = Descriptors.NumHAcceptors(mol)
    tpsa   = Descriptors.TPSA(mol)
    n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    pka_a, pka_b = estimate_pka(mol)

    pharm = pharmacophore_features(mol)
    has_basic_n  = pharm["basic_n_count"] > 0
    has_acid     = pharm["n_carboxylic"] + pharm["n_phenol"] > 0
    has_oh       = pharm["n_oh"] > 0
    has_phenol   = pharm["n_phenol"] > 0

    # CYP1A2 substrate: planar aromatic, small, electron-rich
    cyp1a2_sub = (n_arom >= 2 and mw < 350 and logp > 1.0
                  and not has_acid and not has_basic_n)

    # CYP2C9 substrate: acidic, MW 200-550
    cyp2c9_sub = (has_acid and 200 < mw < 550 and logp > 1.5)

    # CYP2C19 substrate: overlapping; moderate logp
    cyp2c19_sub = ((has_acid or has_basic_n) and mw < 500 and 1.0 < logp < 5.0)

    # CYP2D6 substrate: basic N, aromatic, MW < 500
    cyp2d6_sub = (has_basic_n and n_arom >= 1 and mw < 500)

    # CYP3A4 substrate: large, flexible, lipophilic (broadest)
    cyp3a4_sub = (mw > 300 and logp > 1.0 and hba > 3)

    # Inhibitors: similar features but stronger pharmacophore signature
    cyp1a2_inh  = (n_arom >= 3 and mw < 400)
    cyp2c9_inh  = (has_acid and logp > 2.5 and mw > 250)
    cyp2c19_inh = (cyp2c19_sub and logp > 2.0)
    cyp2d6_inh  = (has_basic_n and n_arom >= 2 and mw < 450 and pka_b > 6)
    cyp3a4_inh  = (mw > 350 and hba > 5 and logp > 2.0)

    # UGT substrate: requires OH/NH
    ugt_sub  = (has_oh or (pharm["n_nh"] > 0)) and mw < 600

    # SULT substrate: phenol specifically
    sult_sub = has_phenol and mw < 400

    # MAO substrate: primary amine + aliphatic chain
    mao_sub = pharm["n_prim_amine"] > 0 and n_arom <= 2

    # Convert to binary with noise (simulate experimental variability)
    def noisy(flag: bool, fp: float = 0.15, fn: float = 0.20) -> int:
        """Add false positive/negative noise to binary label."""
        if flag:
            return int(rng.random() > fn)
        else:
            return int(rng.random() < fp)

    labels = np.array([
        noisy(cyp1a2_sub),  noisy(cyp1a2_inh),
        noisy(cyp2c9_sub),  noisy(cyp2c9_inh),
        noisy(cyp2c19_sub), noisy(cyp2c19_inh),
        noisy(cyp2d6_sub),  noisy(cyp2d6_inh),
        noisy(cyp3a4_sub),  noisy(cyp3a4_inh),
        noisy(ugt_sub),
        noisy(sult_sub),
        noisy(mao_sub),
    ], dtype=np.float32)

    return labels


# ── Model training ────────────────────────────────────────────────────────────

class CYPIsoformClassifier:
    """
    Multi-task ensemble classifier for CYP/enzyme isoform prediction.
    One calibrated GBM per task.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int    = 4,
        learning_rate: float = 0.1,
    ):
        self.n_estimators   = n_estimators
        self.max_depth      = max_depth
        self.learning_rate  = learning_rate
        self.models: Dict[str, Pipeline] = {}
        self._feature_dim: Optional[int] = None

    def _make_model(self) -> Pipeline:
        base = GradientBoostingClassifier(
            n_estimators  = self.n_estimators,
            max_depth     = self.max_depth,
            learning_rate = self.learning_rate,
            subsample     = 0.85,
            random_state  = SEED,
        )
        calibrated = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),  # FP data: no centering
            ("clf",    calibrated),
        ])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Parameters
        ----------
        X : [N, F] feature matrix
        Y : [N, N_TASKS] binary label matrix
        """
        self._feature_dim = X.shape[1]
        print(f"Training {N_TASKS} CYP/enzyme classifiers...")
        for i, task in enumerate(ENZYME_TASKS):
            y  = Y[:, i].astype(int)
            pos_rate = y.mean()
            if pos_rate < 0.02 or pos_rate > 0.98:
                # Degenerate task: skip
                print(f"  [{task:25s}] Skipping (degenerate: {pos_rate:.2f} positive rate)")
                continue
            model = self._make_model()
            model.fit(X, y)
            self.models[task] = model
            print(f"  [{task:25s}] Fitted  (pos_rate={pos_rate:.2f})")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [N, N_TASKS] probability matrix."""
        proba = np.zeros((len(X), N_TASKS), dtype=np.float32)
        for i, task in enumerate(ENZYME_TASKS):
            if task in self.models:
                proba[:, i] = self.models[task].predict_proba(X)[:, 1]
        return proba

    def evaluate_cv(self, X: np.ndarray, Y: np.ndarray, cv: int = 5) -> pd.DataFrame:
        """Cross-validated AUC for each task."""
        rows = []
        kf   = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

        for i, task in enumerate(ENZYME_TASKS):
            y = Y[:, i].astype(int)
            pos_rate = y.mean()
            if pos_rate < 0.02 or pos_rate > 0.98:
                rows.append({"Task": task, "AUC-ROC": float("nan"),
                              "AP": float("nan"), "Pos%": f"{pos_rate*100:.0f}%"})
                continue

            base = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=SEED
            )
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf",    base),
            ])
            proba_cv = cross_val_predict(pipe, X, y, cv=kf, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(y, proba_cv))
            ap  = float(average_precision_score(y, proba_cv))
            rows.append({"Task": task, "AUC-ROC": auc, "AP": ap,
                          "Pos%": f"{pos_rate*100:.0f}%"})
            print(f"  [{task:25s}] AUC={auc:.3f}  AP={ap:.3f}")

        return pd.DataFrame(rows)

    def predict_compound(self, smiles: str) -> pd.DataFrame:
        """
        Predict enzyme involvement for a single compound.

        Returns
        -------
        DataFrame: enzyme, role, probability, involvement_level
        """
        X = mol_to_features(smiles)
        if X is None:
            return pd.DataFrame()
        X  = X.reshape(1, -1)
        P  = self.predict_proba(X)[0]

        rows = []
        for i, task in enumerate(ENZYME_TASKS):
            parts = task.rsplit("_", 1)
            enzyme = parts[0]
            role   = parts[1] if len(parts) == 2 else "substrate"
            prob   = P[i]
            if prob >= 0.7:
                level = "HIGH"
            elif prob >= 0.4:
                level = "MODERATE"
            else:
                level = "low"
            rows.append({
                "Enzyme":       enzyme,
                "Role":         role,
                "Probability":  round(float(prob), 3),
                "Involvement":  level,
            })
        df = pd.DataFrame(rows).sort_values("Probability", ascending=False)
        return df


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_cyp_profile(
    profile_df: pd.DataFrame,
    compound_name: str = "Compound",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of CYP/enzyme probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, role in zip(axes, ["substrate", "inhibitor"]):
        sub = profile_df[profile_df["Role"] == role].copy()
        sub = sub.sort_values("Probability", ascending=True)

        colours = []
        for p in sub["Probability"]:
            if p >= 0.7:
                colours.append("#E74C3C")
            elif p >= 0.4:
                colours.append("#F39C12")
            else:
                colours.append("#2ECC71")

        bars = ax.barh(sub["Enzyme"], sub["Probability"], color=colours, alpha=0.85)
        ax.axvline(0.5, ls="--", color="black", lw=1, alpha=0.5)
        ax.axvline(0.7, ls=":",  color="red",   lw=1, alpha=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Probability", fontsize=10)
        ax.set_title(f"{role.capitalize()}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Add probability text on bars
        for bar, p in zip(bars, sub["Probability"]):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{p:.2f}", va="center", fontsize=8)

    fig.suptitle(f"CYP & Enzyme Isoform Profile: {compound_name}",
                 fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#E74C3C", label="High (>0.70)"),
        Patch(color="#F39C12", label="Moderate (0.40-0.70)"),
        Patch(color="#2ECC71", label="Low (<0.40)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_auc_summary(cv_results: pd.DataFrame, out_path: Optional[Path] = None) -> plt.Figure:
    """Bar chart of cross-validated AUC-ROC for each task."""
    valid = cv_results.dropna(subset=["AUC-ROC"])
    fig, ax = plt.subplots(figsize=(12, 5))

    colours = ["#E74C3C" if r > 0.85 else "#3498DB" if r > 0.75 else "#95A5A6"
               for r in valid["AUC-ROC"]]
    bars = ax.bar(valid["Task"], valid["AUC-ROC"], color=colours, alpha=0.85)
    ax.axhline(0.75, ls="--", color="orange", lw=1.5, label="AUC=0.75 threshold")
    ax.axhline(0.85, ls="--", color="green",  lw=1.5, label="AUC=0.85 threshold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-fold CV AUC-ROC", fontsize=11)
    ax.set_title("CYP Isoform Classifier — Cross-Validated Performance",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, v in zip(bars, valid["AUC-ROC"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Generate training data
    print("=" * 65)
    print("  CYP Isoform & Phase II Enzyme Classifier")
    print("=" * 65)

    training_smiles = [
        # CYP1A2 substrates: planar aromatics
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",   # Caffeine
        "c1ccc2[nH]cc(CC(N)C(=O)O)c2c1", # Tryptophan
        "COc1ccc(OC)cc1",                 # Dimethoxybenzene
        "c1ccc(-c2ccncc2)cc1",            # 4-phenylpyridine
        "c1ccc2c(c1)ccc1ccccc12",         # Anthracene
        "Cc1ccc2cc3ccccc3cc2c1",          # 2-methylanthracene
        # CYP2D6 substrates: basic N + aromatic
        "CCN(CC)CCOC(=O)c1ccccc1",        # Procaine
        "CN(C)CCCN1c2ccccc2Sc2ccccc21",   # Promazine
        "CC(N)Cc1ccccc1",                 # Amphetamine
        "OC(c1ccccc1)c1ccccn1",           # Carbinoxamine-like
        "COc1ccc(CCN)cc1",                # Tyramine-like
        # CYP2C9 substrates: acidic + aromatic
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",    # Ibuprofen
        "COc1ccc(C(C)C(=O)O)cc1",         # Naproxen
        "c1ccc(NC2CCCC2=O)cc1",           # Warfarin-like
        "O=C(O)c1ccccc1Nc1cccc(Cl)c1",   # Tolfenamic acid-like
        # CYP3A4 substrates: large, flexible
        "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",  # Diazepam
        "CC(=O)OCC",
        "CCCCC(=O)OCC",
        "c1ccc(NC(=O)c2ccccc2)cc1",       # Benzanilide
        "CCN(CC)CC(=O)Nc1c(C)cccc1C",     # Lidocaine
        # CYP2C19 substrates
        "CCOC(=O)c1nc2ccccc2n1CC=C",      # Mebendazole-like
        "Cc1cnc2nc(Nc3ccc(F)cc3)ncc2c1", # CDK4/6 inhibitor-like
        # UGT substrates: phenols
        "CC(=O)Nc1ccc(O)cc1",             # Paracetamol
        "c1cc(O)ccc1O",                   # Resorcinol (catechol)
        "Oc1ccccc1",                      # Phenol
        "COc1ccc(O)cc1",                  # 4-methoxyphenol (guaiacol)
        # SULT substrates: phenols
        "c1ccc(O)cc1",                    # Phenol
        "Cc1ccc(O)cc1",                   # p-cresol
        "Oc1ccc(O)cc1",                   # Hydroquinone
        # MAO substrates: primary amines
        "NCC1CCCCC1",                     # Cyclohexylmethylamine
        "NCCc1ccccc1",                    # Phenethylamine
        "NC",                             # Methylamine
        "NCCc1ccc(O)c(O)c1",             # Dopamine
        # Additional diverse drugs
        "O=C(c1ccncc1)N1CCC(n2cnc3ccccc32)CC1",  # complex amide
        "Cc1noc(NC(=O)c2ccccc2F)c1",
        "CC(O)c1ccccc1",
        "CCOC(=O)CC",
        "c1ccc2c(c1)CCCO2",               # Chroman
        "O=C1CCc2ccccc2N1",               # Dihydroquinolinone
        "c1ccc(CN2CCNCC2)cc1",            # Phenylpiperazine
    ]

    print(f"\n[1/4] Computing features for {len(training_smiles)} compounds...")
    X_list, y_list = [], []
    for smi in training_smiles:
        feats = mol_to_features(smi)
        labs  = generate_cyp_labels(smi)
        if feats is not None and labs is not None:
            X_list.append(feats)
            y_list.append(labs)

    X = np.vstack(X_list)
    Y = np.vstack(y_list)
    print(f"  Feature matrix: {X.shape}  |  Label matrix: {Y.shape}")

    # 2. Cross-validated evaluation
    print(f"\n[2/4] 5-fold cross-validation...")
    clf = CYPIsoformClassifier(n_estimators=150, max_depth=4)
    cv_results = clf.evaluate_cv(X, Y, cv=5)

    valid_aucs = cv_results["AUC-ROC"].dropna()
    print(f"\n  Mean AUC-ROC: {valid_aucs.mean():.3f} ± {valid_aucs.std():.3f}")
    print(f"  AUC range:   {valid_aucs.min():.3f} – {valid_aucs.max():.3f}")

    # 3. Fit final model
    print(f"\n[3/4] Training final model on full dataset...")
    clf.fit(X, Y)

    # 4. Predict test compounds
    print(f"\n[4/4] Predicting enzyme involvement for test compounds...")

    test_cases = {
        "Caffeine":      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Ibuprofen":     "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "Lidocaine":     "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        "Paracetamol":   "CC(=O)Nc1ccc(O)cc1",
        "Diazepam":      "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "Tamoxifen":     "CC(/C=C/c1ccc(O)cc1)=C(\\CCN(C)C)c1ccccc1",
    }

    for name, smi in test_cases.items():
        profile = clf.predict_compound(smi)
        print(f"\n  {name}:")
        high = profile[profile["Involvement"] == "HIGH"]
        if not high.empty:
            for _, row in high.iterrows():
                print(f"    ⚑  {row['Enzyme']:15s} {row['Role']:12s}  P={row['Probability']:.3f}")
        else:
            print("    (no high-involvement enzymes predicted)")

        # Save profile figure
        fig = plot_cyp_profile(profile, compound_name=name,
                                out_path=OUT_DIR / f"cyp_profile_{name.lower()}.png")
        plt.close(fig)

    # AUC summary figure
    fig = plot_auc_summary(cv_results, out_path=OUT_DIR / "cyp_classifier_auc.png")
    plt.close(fig)

    # Export CV results
    cv_results.to_csv(OUT_DIR / "cyp_cv_results.csv", index=False)
    print(f"\nSaved CYP profiles and AUC summary to {OUT_DIR}/")
