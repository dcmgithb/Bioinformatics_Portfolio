"""
Protein & Antibody Feature Engineering
========================================
Physicochemical amino acid descriptors, sequence-level features,
and antibody-specific CDR loop extraction for ML pipelines.

References
----------
- Kyte & Doolittle (1982) hydrophobicity scale
- Guruprasad instability index (1990)
- Chothia CDR numbering (approximated from conserved framework residues)
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Amino acid physicochemical properties ─────────────────────────────────────

AA_PROPERTIES: Dict[str, Dict] = {
    # AA : {hydrophobicity, charge, mw, polarity, aromaticity, flexibility}
    "A": {"hydrophobicity":  1.8, "charge":  0, "mw":  89.1, "polarity": False, "aromatic": False, "flexibility": 0.36},
    "R": {"hydrophobicity": -4.5, "charge": +1, "mw": 174.2, "polarity": True,  "aromatic": False, "flexibility": 0.53},
    "N": {"hydrophobicity": -3.5, "charge":  0, "mw": 132.1, "polarity": True,  "aromatic": False, "flexibility": 0.46},
    "D": {"hydrophobicity": -3.5, "charge": -1, "mw": 133.1, "polarity": True,  "aromatic": False, "flexibility": 0.51},
    "C": {"hydrophobicity":  2.5, "charge":  0, "mw": 121.2, "polarity": False, "aromatic": False, "flexibility": 0.35},
    "Q": {"hydrophobicity": -3.5, "charge":  0, "mw": 146.2, "polarity": True,  "aromatic": False, "flexibility": 0.49},
    "E": {"hydrophobicity": -3.5, "charge": -1, "mw": 147.1, "polarity": True,  "aromatic": False, "flexibility": 0.50},
    "G": {"hydrophobicity": -0.4, "charge":  0, "mw":  75.0, "polarity": False, "aromatic": False, "flexibility": 0.54},
    "H": {"hydrophobicity": -3.2, "charge":  0, "mw": 155.2, "polarity": True,  "aromatic": True,  "flexibility": 0.32},
    "I": {"hydrophobicity":  4.5, "charge":  0, "mw": 131.2, "polarity": False, "aromatic": False, "flexibility": 0.46},
    "L": {"hydrophobicity":  3.8, "charge":  0, "mw": 131.2, "polarity": False, "aromatic": False, "flexibility": 0.59},
    "K": {"hydrophobicity": -3.9, "charge": +1, "mw": 146.2, "polarity": True,  "aromatic": False, "flexibility": 0.47},
    "M": {"hydrophobicity":  1.9, "charge":  0, "mw": 149.2, "polarity": False, "aromatic": False, "flexibility": 0.41},
    "F": {"hydrophobicity":  2.8, "charge":  0, "mw": 165.2, "polarity": False, "aromatic": True,  "flexibility": 0.31},
    "P": {"hydrophobicity": -1.6, "charge":  0, "mw": 115.1, "polarity": False, "aromatic": False, "flexibility": 0.51},
    "S": {"hydrophobicity": -0.8, "charge":  0, "mw": 105.1, "polarity": True,  "aromatic": False, "flexibility": 0.51},
    "T": {"hydrophobicity": -0.7, "charge":  0, "mw": 119.1, "polarity": True,  "aromatic": False, "flexibility": 0.44},
    "W": {"hydrophobicity": -0.9, "charge":  0, "mw": 204.2, "polarity": False, "aromatic": True,  "flexibility": 0.31},
    "Y": {"hydrophobicity": -1.3, "charge":  0, "mw": 181.2, "polarity": True,  "aromatic": True,  "flexibility": 0.42},
    "V": {"hydrophobicity":  4.2, "charge":  0, "mw": 117.1, "polarity": False, "aromatic": False, "flexibility": 0.39},
}

AMINO_ACIDS = sorted(AA_PROPERTIES.keys())


# ── Per-residue descriptor vector (10-dim) ────────────────────────────────────

def aa_descriptor_vector(aa: str) -> np.ndarray:
    """
    10-dimensional physicochemical descriptor for a single amino acid.

    Dimensions
    ----------
    0  hydrophobicity   (Kyte-Doolittle, scaled [-1,1])
    1  charge           (-1, 0, +1)
    2  molecular_weight (scaled by /200)
    3  polarity         (0/1)
    4  aromaticity      (0/1)
    5  flexibility      (raw value ~0.31-0.59)
    6  is_proline       (0/1)  — breaks alpha helices
    7  is_glycine       (0/1)  — max flexibility
    8  is_cysteine      (0/1)  — disulfide bridges
    9  size_class       (tiny=0, small=0.25, med=0.5, large=0.75, xlarge=1)
    """
    props = AA_PROPERTIES.get(aa.upper(), {
        "hydrophobicity": 0.0, "charge": 0, "mw": 130.0,
        "polarity": False, "aromatic": False, "flexibility": 0.45
    })

    # Normalise hydrophobicity to [-1, 1] (range: -4.5 to 4.5)
    hydro_norm = float(props["hydrophobicity"]) / 4.5

    # Size class from MW
    mw = props["mw"]
    if mw < 90:      size_class = 0.0
    elif mw < 120:   size_class = 0.25
    elif mw < 150:   size_class = 0.5
    elif mw < 175:   size_class = 0.75
    else:            size_class = 1.0

    return np.array([
        hydro_norm,
        float(props["charge"]),
        float(props["mw"]) / 200.0,
        float(props["polarity"]),
        float(props["aromatic"]),
        float(props["flexibility"]),
        float(aa.upper() == "P"),
        float(aa.upper() == "G"),
        float(aa.upper() == "C"),
        size_class,
    ], dtype=np.float32)


# ── Sequence encoding ─────────────────────────────────────────────────────────

def encode_sequence(
    seq: str,
    method: str = "physicochemical",
    max_len: Optional[int] = None,
) -> np.ndarray:
    """
    Encode a protein sequence as a 2D feature matrix.

    Parameters
    ----------
    seq     : amino acid sequence (single-letter codes)
    method  : "one_hot" → [L,20], "physicochemical" → [L,10], "combined" → [L,30]
    max_len : if given, zero-pad or truncate to this length

    Returns
    -------
    np.ndarray [L, dim]
    """
    seq = seq.upper()
    if max_len:
        seq = seq[:max_len]

    if method == "one_hot":
        mat = np.zeros((len(seq), 20), dtype=np.float32)
        for i, aa in enumerate(seq):
            if aa in AMINO_ACIDS:
                mat[i, AMINO_ACIDS.index(aa)] = 1.0

    elif method == "physicochemical":
        mat = np.stack([aa_descriptor_vector(aa) for aa in seq], axis=0)

    elif method == "combined":
        oh  = encode_sequence(seq, "one_hot")
        phys = encode_sequence(seq, "physicochemical")
        mat  = np.concatenate([oh, phys], axis=1)

    else:
        raise ValueError(f"Unknown method: {method}. Choose one_hot/physicochemical/combined")

    if max_len and len(seq) < max_len:
        pad = max_len - len(seq)
        mat = np.pad(mat, ((0, pad), (0, 0)), mode="constant")

    return mat


def sequence_to_flat_features(seq: str, method: str = "physicochemical") -> np.ndarray:
    """Flatten encode_sequence output → 1D feature vector (for sklearn)."""
    return encode_sequence(seq, method).flatten()


# ── Sequence-level statistics ─────────────────────────────────────────────────

def compute_sequence_features(seq: str) -> Dict[str, float]:
    """
    Compute 15 sequence-level biophysical features.

    Returns
    -------
    dict with: length, MW, net_charge_pH7, pI_estimate,
               hydrophobic_fraction, aromatic_fraction, proline_fraction,
               gravy, aliphatic_index, instability_index,
               charge_asymmetry, mean_flexibility, cys_count, avg_mw_per_residue
    """
    seq = seq.upper()
    n   = len(seq)
    if n == 0:
        return {}

    counts = {aa: seq.count(aa) for aa in AMINO_ACIDS}

    # Total MW (approximate, ignoring water loss per peptide bond correctly)
    mw = sum(counts[aa] * AA_PROPERTIES[aa]["mw"] for aa in AMINO_ACIDS) - 18.02 * (n - 1)

    # Net charge at pH 7.4
    pos_aa = {"R": 1, "K": 1, "H": 0.1}   # H is ~10% protonated at pH 7.4
    neg_aa = {"D": -1, "E": -1}
    net_charge = (sum(pos_aa.get(aa, 0) * counts[aa] for aa in pos_aa) +
                  sum(neg_aa[aa] * counts[aa] for aa in neg_aa))

    # pI estimate (simple bisection not full Henderson-Hasselbalch)
    pI_estimate = 7.0 + net_charge * 0.5  # crude linear approx

    # GRAVY (grand average of hydropathicity)
    gravy = sum(AA_PROPERTIES[aa]["hydrophobicity"] * counts[aa]
                for aa in AMINO_ACIDS) / n

    # Aliphatic index: relative volume occupied by aliphatic AAs (A/V/I/L)
    ali = (counts["A"] + 2.9 * counts["V"] + 3.9 * (counts["I"] + counts["L"])) / n * 100

    # Instability index (Guruprasad 1990) — simplified DIWV table
    # Using a simplified version based on dipeptide instability weights
    instability_weights = {
        ("W", "W"): 1.0, ("C", "K"): 1.0, ("R", "R"): 0.83, ("A", "D"): -7.49,
        ("K", "K"): 1.0, ("R", "H"): 0.5,  ("S", "R"):  0.5,  ("G", "G"): -0.2,
    }
    instab = 0.0
    for i in range(len(seq) - 1):
        pair = (seq[i], seq[i + 1])
        instab += instability_weights.get(pair, 0.0)
    instability_index = (10.0 / n) * instab + 40.0  # baseline ~40 for stable

    # Charge asymmetry: difference between N-half and C-half charge
    half = n // 2
    charge_n = sum(AA_PROPERTIES[aa]["charge"] * seq[:half].count(aa)
                   for aa in AA_PROPERTIES)
    charge_c = sum(AA_PROPERTIES[aa]["charge"] * seq[half:].count(aa)
                   for aa in AA_PROPERTIES)
    charge_asymmetry = float(abs(charge_n - charge_c))

    # Mean flexibility
    mean_flex = np.mean([AA_PROPERTIES.get(aa, {"flexibility": 0.45})["flexibility"]
                         for aa in seq])

    return {
        "length":              n,
        "MW":                  mw,
        "net_charge_pH7":      net_charge,
        "pI_estimate":         pI_estimate,
        "hydrophobic_fraction": (counts.get("I", 0) + counts.get("L", 0) +
                                  counts.get("V", 0) + counts.get("F", 0) +
                                  counts.get("M", 0)) / n,
        "aromatic_fraction":   (counts.get("F", 0) + counts.get("W", 0) +
                                 counts.get("Y", 0) + counts.get("H", 0)) / n,
        "proline_fraction":    counts.get("P", 0) / n,
        "gravy":               gravy,
        "aliphatic_index":     ali,
        "instability_index":   instability_index,
        "charge_asymmetry":    charge_asymmetry,
        "mean_flexibility":    float(mean_flex),
        "cys_count":           counts.get("C", 0),
        "avg_mw_per_residue":  mw / n,
        "pos_charged_fraction":(counts.get("R", 0) + counts.get("K", 0)) / n,
    }


# ── Antibody CDR extraction (Chothia-based heuristic) ────────────────────────

# Conserved framework residues used to locate CDR boundaries
# Simple positional heuristic — production code should use ANARCI/IgBLAST
CDR_WINDOWS = {
    # (CDR_name, start_offset_from_conserved_cys, length_range)
    "CDR-H1": (6,  [5, 12]),
    "CDR-H2": (17, [3, 10]),
    "CDR-H3": (33, [3, 25]),
    "CDR-L1": (6,  [5, 17]),
    "CDR-L2": (16, [3, 7]),
    "CDR-L3": (33, [5, 11]),
}


def extract_cdr_regions(
    vh_sequence: str,
    vl_sequence: Optional[str] = None,
) -> Dict[str, str]:
    """
    Extract CDR loop sequences from VH (and optionally VL) using a
    simple pattern-based approach anchored on conserved cysteines.

    In production, replace with ANARCI or IgBLAST-based numbering.

    Returns
    -------
    dict: {CDR-H1: str, CDR-H2: str, CDR-H3: str, [CDR-L1..L3 if vl given]}
    """
    results: Dict[str, str] = {}
    vh = vh_sequence.upper()

    # Locate conserved Cys positions in VH (approximately pos 22 & 92 in Chothia)
    cys_positions = [i for i, aa in enumerate(vh) if aa == "C"]

    if len(cys_positions) >= 2:
        c1, c2 = cys_positions[0], cys_positions[1]
        span = c2 - c1
        # CDR-H1: ~6 residues after first Cys
        h1_start = max(0, c1 + 6)
        results["CDR-H1"] = vh[h1_start:h1_start + 8]
        # CDR-H2: ~midway between Cys1 and Cys2
        h2_start = c1 + span // 2 - 2
        results["CDR-H2"] = vh[h2_start:h2_start + 7]
        # CDR-H3: just before the second Cys
        h3_end = c2
        results["CDR-H3"] = vh[max(0, h3_end - 12):h3_end]
    else:
        # Fallback: return fixed windows
        l = len(vh)
        results["CDR-H1"] = vh[max(0, l // 6):max(0, l // 6) + 8]
        results["CDR-H2"] = vh[max(0, l // 3):max(0, l // 3) + 7]
        results["CDR-H3"] = vh[max(0, l * 2 // 3):max(0, l * 2 // 3) + 10]

    if vl_sequence:
        vl = vl_sequence.upper()
        cys_vl = [i for i, aa in enumerate(vl) if aa == "C"]
        if len(cys_vl) >= 2:
            c1, c2 = cys_vl[0], cys_vl[1]
            span = c2 - c1
            results["CDR-L1"] = vl[c1 + 6:c1 + 6 + 11]
            results["CDR-L2"] = vl[c1 + span // 2 - 1:c1 + span // 2 + 6]
            results["CDR-L3"] = vl[max(0, c2 - 9):c2]
        else:
            l = len(vl)
            results["CDR-L1"] = vl[max(0, l // 6):max(0, l // 6) + 11]
            results["CDR-L2"] = vl[max(0, l // 3):max(0, l // 3) + 6]
            results["CDR-L3"] = vl[max(0, l * 2 // 3):max(0, l * 2 // 3) + 9]

    return results


def compute_antibody_features(
    vh: str,
    vl: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute 20 antibody-specific features combining CDR and framework info.
    """
    cdrs = extract_cdr_regions(vh, vl)
    vh_feats = compute_sequence_features(vh)
    feats: Dict[str, float] = {}

    # CDR lengths
    for cdr_name, cdr_seq in cdrs.items():
        feats[f"{cdr_name}_length"] = len(cdr_seq)

    # CDR-H3 properties (most influential CDR)
    h3 = cdrs.get("CDR-H3", "")
    if h3:
        h3_feats = compute_sequence_features(h3)
        feats["CDR-H3_hydrophobicity"] = h3_feats.get("gravy", 0.0)
        feats["CDR-H3_charge"]         = h3_feats.get("net_charge_pH7", 0.0)
        feats["CDR-H3_aromatic_frac"]  = h3_feats.get("aromatic_fraction", 0.0)

    # Overall VH features
    feats["VH_GRAVY"]              = vh_feats.get("gravy", 0.0)
    feats["VH_instability"]        = vh_feats.get("instability_index", 40.0)
    feats["VH_charge"]             = vh_feats.get("net_charge_pH7", 0.0)
    feats["VH_hydrophobic_frac"]   = vh_feats.get("hydrophobic_fraction", 0.0)
    feats["total_CDR_length"]      = sum(feats.get(f"{k}_length", 0)
                                          for k in ["CDR-H1", "CDR-H2", "CDR-H3"])

    # Predicted aggregation hotspots: hydrophobic patches in CDRs
    hydro_vals = []
    for cdr_name, cdr_seq in cdrs.items():
        for aa in cdr_seq:
            hydro_vals.append(AA_PROPERTIES.get(aa, {"hydrophobicity": 0})["hydrophobicity"])
    feats["CDR_mean_hydrophobicity"] = float(np.mean(hydro_vals)) if hydro_vals else 0.0
    feats["CDR_max_hydrophobicity"]  = float(np.max(hydro_vals))  if hydro_vals else 0.0

    return feats


# ── Dataset class ─────────────────────────────────────────────────────────────

class SequenceDataset:
    """
    Batched feature extraction from protein sequences.
    Produces a feature matrix ready for sklearn or torch models.
    """

    def __init__(
        self,
        sequences: List[str],
        labels: Optional[List[float]] = None,
        antibody_mode: bool = False,
    ):
        self.sequences  = sequences
        self.labels     = labels
        self._antibody  = antibody_mode

    def compute_features(self) -> pd.DataFrame:
        """Return a DataFrame of computed features for all sequences."""
        rows = []
        for seq in self.sequences:
            f = compute_sequence_features(seq)
            if self._antibody:
                f.update(compute_antibody_features(seq))
            rows.append(f)
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, Optional[float]]:
        label = self.labels[idx] if self.labels else None
        return self.sequences[idx], label
