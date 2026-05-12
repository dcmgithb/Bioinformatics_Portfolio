"""
Shared RDKit Cheminformatics Utilities
=======================================
Common functions for the geroprotector drug discovery pipeline.

Python : >= 3.10
RDKit  : >= 2023.09
"""

from __future__ import annotations

import warnings
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Lipinski, QED, AllChem, Draw,
    rdMolDescriptors, rdFMCS, FilterCatalog
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── SMILES Validation ─────────────────────────────────────────────────────────

def smiles_to_mol(smiles: str, sanitize: bool = True) -> Optional[Chem.Mol]:
    """Parse SMILES to RDKit Mol. Returns None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None:
            log.warning(f"Invalid SMILES: {smiles[:60]}")
        return mol
    except Exception as e:
        log.warning(f"SMILES parse error: {e}")
        return None


def validate_smiles_column(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Parse and validate a SMILES column, dropping invalid entries."""
    df = df.copy()
    df["mol"] = df[smiles_col].apply(smiles_to_mol)
    n_invalid = df["mol"].isna().sum()
    if n_invalid > 0:
        log.warning(f"Dropped {n_invalid} invalid SMILES")
    df = df[df["mol"].notna()].copy()
    # Canonicalise SMILES
    df[smiles_col] = df["mol"].apply(Chem.MolToSmiles)
    return df

# ── Lipinski / Drug-likeness Filters ─────────────────────────────────────────

LIPINSKI_RULES = {
    "MW"   : (0,   500),
    "LogP" : (-5,  5  ),
    "HBD"  : (0,   5  ),
    "HBA"  : (0,   10 ),
}

VEBER_RULES = {
    "TPSA"        : (0, 140),
    "RotatableBonds": (0, 10),
}

def calc_lipinski(mol: Chem.Mol) -> dict:
    """Compute Lipinski Ro5 properties."""
    return {
        "MW"   : Descriptors.MolWt(mol),
        "ExactMW": Descriptors.ExactMolWt(mol),
        "LogP" : Descriptors.MolLogP(mol),
        "HBD"  : Lipinski.NumHDonors(mol),
        "HBA"  : Lipinski.NumHAcceptors(mol),
        "TPSA" : Descriptors.TPSA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount"     : rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings" : rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtoms"    : mol.GetNumHeavyAtoms(),
        "QED"  : QED.qed(mol),
        "Fsp3" : rdMolDescriptors.CalcFractionCSP3(mol),
        "StereocentersCount": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    }


def lipinski_pass(mol: Chem.Mol, max_violations: int = 1) -> bool:
    """Return True if molecule passes Lipinski Ro5 (allows max_violations)."""
    props = calc_lipinski(mol)
    violations = sum([
        props["MW"]   > LIPINSKI_RULES["MW"][1],
        props["LogP"] > LIPINSKI_RULES["LogP"][1],
        props["HBD"]  > LIPINSKI_RULES["HBD"][1],
        props["HBA"]  > LIPINSKI_RULES["HBA"][1],
    ])
    return violations <= max_violations


def veber_pass(mol: Chem.Mol) -> bool:
    """Return True if molecule passes Veber oral bioavailability rules."""
    props = calc_lipinski(mol)
    return (props["TPSA"]           <= VEBER_RULES["TPSA"][1] and
            props["RotatableBonds"] <= VEBER_RULES["RotatableBonds"][1])

# ── PAINS / Structural Alerts ─────────────────────────────────────────────────

def _get_pains_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


_PAINS_CATALOG = _get_pains_catalog()


def is_pains(mol: Chem.Mol) -> tuple[bool, list[str]]:
    """
    Check if a molecule contains PAINS substructures.
    Returns (is_pains: bool, alert_names: list[str])
    """
    entry = _PAINS_CATALOG.GetFirstMatch(mol)
    if entry:
        matches = _PAINS_CATALOG.GetMatches(mol)
        names   = [m.GetDescription() for m in matches]
        return True, names
    return False, []

# ── Fingerprints ──────────────────────────────────────────────────────────────

def morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan (ECFP-like) fingerprint as numpy array."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_fp(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    """RDKit topological fingerprint as numpy array."""
    from rdkit.Chem import RDKFingerprint
    fp  = RDKFingerprint(mol, fpSize=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maccs_fp(mol: Chem.Mol) -> np.ndarray:
    """MACCS keys fingerprint (167 bits) as numpy array."""
    from rdkit.Chem.MACCSkeys import GenMACCSKeys
    fp  = GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def fingerprint_matrix(
    mols: list[Chem.Mol],
    fp_type: str = "morgan",
    **kwargs,
) -> np.ndarray:
    """
    Compute a fingerprint matrix (n_compounds × n_bits).
    fp_type: "morgan" | "rdkit" | "maccs"
    """
    fn_map = {"morgan": morgan_fp, "rdkit": rdkit_fp, "maccs": maccs_fp}
    fn = fn_map.get(fp_type)
    if fn is None:
        raise ValueError(f"Unknown fp_type: {fp_type}. Choose from {list(fn_map)}")
    return np.vstack([fn(m, **kwargs) for m in mols])

# ── Tanimoto Similarity ───────────────────────────────────────────────────────

def tanimoto_matrix(fp_matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Tanimoto similarity matrix from a binary fingerprint matrix.
    Efficient vectorised implementation.
    """
    # Tanimoto: |A ∩ B| / |A ∪ B|  =  dot(A,B) / (|A| + |B| - dot(A,B))
    X    = fp_matrix.astype(np.float32)
    dot  = X @ X.T
    norms = np.array(fp_matrix.sum(axis=1), dtype=np.float32)
    denom = norms[:, None] + norms[None, :] - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom > 0, dot / denom, 0.0)
    return sim


def top_k_similar(
    query_fp:   np.ndarray,
    library_fps: np.ndarray,
    names:      list[str],
    k:          int = 10,
) -> pd.DataFrame:
    """Return the top-k most similar compounds to a query fingerprint."""
    q = query_fp.astype(np.float32)
    L = library_fps.astype(np.float32)
    dot   = L @ q
    norms = L.sum(axis=1) + q.sum() - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(norms > 0, dot / norms, 0.0)
    top_idx = np.argsort(sim)[::-1][:k]
    return pd.DataFrame({"name": [names[i] for i in top_idx],
                          "tanimoto": sim[top_idx]})

# ── Murcko Scaffold ───────────────────────────────────────────────────────────

def get_scaffold(mol: Chem.Mol, generic: bool = False) -> Optional[str]:
    """
    Extract Bemis–Murcko scaffold SMILES.
    generic=True returns a framework with all atoms as C and all bonds as single.
    """
    try:
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol))
        else:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None

# ── 2D Descriptors ───────────────────────────────────────────────────────────

# Full set of 2D RDKit descriptors (200+)
_DESCRIPTOR_NAMES = [
    name for name, fn in Descriptors.descList
    if not name.startswith("fr_")     # skip very noisy fragment descriptors
]
_DESCRIPTOR_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(
    _DESCRIPTOR_NAMES
)


def calc_all_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Compute all 2D RDKit descriptors for a molecule."""
    vals = _DESCRIPTOR_CALC.CalcDescriptors(mol)
    return dict(zip(_DESCRIPTOR_NAMES, vals))


def descriptor_matrix(mols: list[Chem.Mol]) -> pd.DataFrame:
    """
    Compute a descriptor matrix (n_compounds × n_descriptors).
    Returns a DataFrame. NaN values indicate calculation failure.
    """
    rows = [calc_all_descriptors(m) for m in mols]
    df   = pd.DataFrame(rows, columns=_DESCRIPTOR_NAMES)
    # Remove constant and near-zero-variance columns
    df   = df.loc[:, df.std(axis=0) > 1e-6]
    return df

# ── Retrosynthetic Accessibility ─────────────────────────────────────────────

def sa_score(mol: Chem.Mol) -> float:
    """
    Synthetic Accessibility (SA) score [1=easy, 10=hard].
    Uses the Ertl & Schuffenhauer (2009) algorithm.
    Requires the sascorer module from RDKit contrib.
    """
    try:
        from rdkit.Chem import RDConfig
        import sys, os
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        return sascorer.calculateScore(mol)
    except ImportError:
        # Fallback: approximate SA via ring complexity
        n_rings   = rdMolDescriptors.CalcNumRings(mol)
        n_stereo  = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_heavy   = mol.GetNumHeavyAtoms()
        return min(10, 1 + 0.2 * n_rings + 0.3 * n_stereo + 0.02 * n_heavy)

# ── MPO (Multi-Parameter Optimisation) Score ─────────────────────────────────

def mpo_score(mol: Chem.Mol) -> float:
    """
    CNS-inspired MPO score adapted for geroprotector lead-likeness.
    Scores 7 desirable properties on [0,1], returns their mean.
    High MPO (>0.6) = drug-like geroprotector candidate.
    """
    props = calc_lipinski(mol)

    def score_range(val, lo, mid_lo, mid_hi, hi):
        """Trapezoidal scoring function."""
        if val <= lo or val >= hi:
            return 0.0
        if mid_lo <= val <= mid_hi:
            return 1.0
        if val < mid_lo:
            return (val - lo) / (mid_lo - lo)
        return (hi - val) / (hi - mid_hi)

    scores = [
        score_range(props["LogP"],         -2,  1,   4,   5),   # moderate lipophilicity
        score_range(props["MW"],            0, 200, 450, 600),   # lead-like MW
        score_range(props["TPSA"],          0,  20, 100, 150),   # permeability
        score_range(props["HBD"],          -1,   0,   3,   6),   # HBD
        score_range(props["HBA"],          -1,   0,   7,  12),   # HBA
        score_range(props["RotatableBonds"],-1,   0,   7,  12),  # flexibility
        props["QED"],                                             # drug-likeness
    ]
    return float(np.mean(scores))

# ── Utility: DataFrame enrichment ────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """
    Add mol, Lipinski properties, QED, scaffold, MPO, PAINS flag to a DataFrame.
    """
    df = validate_smiles_column(df, smiles_col)
    mols = df["mol"].tolist()

    lipinski_rows = [calc_lipinski(m) for m in mols]
    lipinski_df   = pd.DataFrame(lipinski_rows, index=df.index)

    scaffolds     = [get_scaffold(m)         for m in mols]
    pains_flags   = [is_pains(m)[0]          for m in mols]
    mpo_scores    = [mpo_score(m)            for m in mols]
    sa_scores_v   = [sa_score(m)             for m in mols]
    lips_pass     = [lipinski_pass(m)        for m in mols]
    veber_pass_v  = [veber_pass(m)           for m in mols]

    result = df.drop(columns=["mol"]).assign(
        **lipinski_df,
        scaffold     = scaffolds,
        pains        = pains_flags,
        mpo_score    = mpo_scores,
        sa_score     = sa_scores_v,
        lipinski_ok  = lips_pass,
        veber_ok     = veber_pass_v,
        drug_like    = [l and v for l, v in zip(lips_pass, veber_pass_v)],
    )
    return result


log.info("chem_utils loaded — RDKit utilities for geroprotector discovery")
