"""
Molecule → Graph Featurization for GNNs
=========================================
Converts RDKit molecules into graph-structured data with rich
atom and bond feature vectors for message-passing neural networks.

Atom features  : 50-dimensional (atomic type, charge, hybridisation, etc.)
Bond features  : 11-dimensional (type, conjugation, ring, stereo)

Python  : >= 3.10
RDKit   : >= 2023.09
PyTorch : >= 2.0
"""

from __future__ import annotations

import warnings
from typing import Optional, List, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

warnings.filterwarnings("ignore")

# ── Atom vocabulary ──────────────────────────────────────────────────────────

# Common elements in drug-like molecules (+ UNK bucket)
COMMON_ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H B C N O F Si P S Cl Br I

HYBRIDISATION_VOCAB = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

CHIRAL_VOCAB = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

BOND_TYPE_VOCAB = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

STEREO_VOCAB = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


# ── Encoding helpers ──────────────────────────────────────────────────────────

def one_hot(value, vocab: list, allow_unknown: bool = True) -> List[int]:
    """One-hot encode *value* against *vocab*; append UNK bucket when unknown."""
    enc = [int(value == v) for v in vocab]
    if allow_unknown:
        enc.append(int(value not in vocab))
    return enc


# ── Atom featurization (50-dim) ───────────────────────────────────────────────

def atom_features(atom: Chem.Atom) -> List[float]:
    """
    50-dimensional atom feature vector:
        atomic_num  : 12 + 1 UNK  = 13
        degree      : 0-10 + UNK  = 12
        formal_chg  : -2..+2 +UNK =  6
        total_hs    : 0-4 + UNK   =  6
        hybridistn  : 5 + UNK     =  6
        is_aromatic :              =  1
        is_in_ring  :              =  1
        chirality   : 3 + UNK     =  4
        gasteiger_q :              =  1
        ----------                   --
        Total                        50
    """
    feats: List[float] = []

    feats += one_hot(atom.GetAtomicNum(), COMMON_ATOMS)                # 13
    feats += one_hot(atom.GetDegree(), list(range(11)))                 # 12
    feats += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])        #  6
    feats += one_hot(atom.GetTotalNumHs(), list(range(5)))              #  6
    feats += one_hot(atom.GetHybridization(), HYBRIDISATION_VOCAB)      #  6
    feats += [int(atom.GetIsAromatic()), int(atom.IsInRing())]          #  2
    feats += one_hot(atom.GetChiralTag(), CHIRAL_VOCAB)                 #  4

    # Gasteiger partial charge (continuous)
    try:
        gc = float(atom.GetPropsAsDict().get("_GasteigerCharge", 0.0))
        if not np.isfinite(gc):
            gc = 0.0
    except Exception:
        gc = 0.0
    feats.append(gc)                                                    #  1

    return feats  # len == 50 (13+12+6+6+6+2+4+1)


ATOM_FEAT_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


# ── Bond featurization (11-dim) ───────────────────────────────────────────────

def bond_features(bond: Chem.Bond) -> List[float]:
    """
    11-dimensional bond feature vector:
        bond_type   : 4 + UNK = 5
        is_conjug   :         = 1
        is_in_ring  :         = 1
        stereo      : 3 + UNK = 4
        Total                  11
    """
    feats: List[float] = []
    feats += one_hot(bond.GetBondType(), BOND_TYPE_VOCAB)               #  5
    feats += [int(bond.GetIsConjugated()), int(bond.IsInRing())]        #  2
    feats += one_hot(bond.GetStereo(), STEREO_VOCAB)                    #  4
    return feats


BOND_FEAT_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


# ── Molecule → graph dict ─────────────────────────────────────────────────────

def mol_to_graph(
    mol: Chem.Mol,
    y: Optional[Union[float, List[float]]] = None,
    smiles: Optional[str] = None,
) -> Optional[dict]:
    """
    Convert an RDKit Mol to a graph dictionary compatible with
    PyTorch Geometric ``Data`` or any manual GNN implementation.

    Returns
    -------
    dict with keys:
        x           : np.ndarray [N_atoms, ATOM_FEAT_DIM]
        edge_index  : np.ndarray [2, 2*N_bonds]   (undirected → both directions)
        edge_attr   : np.ndarray [2*N_bonds, BOND_FEAT_DIM]
        y           : np.ndarray [1] or [T]        (target labels)
        smiles      : str
    """
    if mol is None:
        return None

    # Gasteiger charges needed before feature extraction
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    # ── Node features ─────────────────────────────────────────────────────────
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = np.array(node_feats, dtype=np.float32)  # [N, F_a]

    # ── Edge features ─────────────────────────────────────────────────────────
    rows, cols, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        rows += [i, j]
        cols += [j, i]
        edge_feats += [bf, bf]

    if rows:
        edge_index = np.array([rows, cols], dtype=np.int64)           # [2, 2E]
        edge_attr  = np.array(edge_feats, dtype=np.float32)           # [2E, F_b]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr  = np.zeros((0, BOND_FEAT_DIM), dtype=np.float32)

    graph = {
        "x":          x,
        "edge_index": edge_index,
        "edge_attr":  edge_attr,
        "smiles":     smiles or Chem.MolToSmiles(mol),
    }

    if y is not None:
        graph["y"] = np.array([y] if not isinstance(y, (list, tuple)) else y,
                               dtype=np.float32)

    return graph


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class MoleculeGraphDataset:
    """
    In-memory dataset of molecular graphs with optional activity labels.
    Compatible with any loop-based GNN training pipeline.
    """

    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[List[Union[float, List[float]]]] = None,
    ):
        self.graphs: List[dict] = []
        n_failed = 0

        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_failed += 1
                continue
            y = labels[i] if labels is not None else None
            g = mol_to_graph(mol, y=y, smiles=smi)
            if g is not None:
                self.graphs.append(g)

        if n_failed:
            print(f"[MoleculeGraphDataset] Dropped {n_failed} invalid SMILES "
                  f"({n_failed / len(smiles_list) * 100:.1f}%)")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        return self.graphs[idx]

    @property
    def num_node_features(self) -> int:
        return self.graphs[0]["x"].shape[1] if self.graphs else ATOM_FEAT_DIM

    @property
    def num_edge_features(self) -> int:
        return self.graphs[0]["edge_attr"].shape[1] if self.graphs else BOND_FEAT_DIM

    @property
    def targets(self) -> np.ndarray:
        return np.array([g["y"] for g in self.graphs if "y" in g])

    def stats(self) -> dict:
        n_atoms = [g["x"].shape[0] for g in self.graphs]
        n_bonds = [g["edge_index"].shape[1] // 2 for g in self.graphs]
        return {
            "n_molecules":     len(self),
            "mean_atoms":      float(np.mean(n_atoms)),
            "max_atoms":       int(np.max(n_atoms)),
            "mean_bonds":      float(np.mean(n_bonds)),
            "node_feat_dim":   self.num_node_features,
            "edge_feat_dim":   self.num_edge_features,
        }


# ── Mini-batch collation ──────────────────────────────────────────────────────

def collate_graphs(batch: List[dict]) -> dict:
    """
    Collate a list of graph dicts into a single batched graph dict.
    Node/edge offsets are computed for graph-level scatter operations.

    Returns
    -------
    dict:
        x           : [sum_N, F_a]
        edge_index  : [2, sum_2E]    (node indices offset per graph)
        edge_attr   : [sum_2E, F_b]
        y           : [B, T]
        batch       : [sum_N]        (graph index per node)
        ptr         : [B+1]          (cumulative node counts)
    """
    all_x, all_ei, all_ea, all_y, all_batch = [], [], [], [], []
    offset = 0

    for g_idx, g in enumerate(batch):
        n = g["x"].shape[0]
        all_x.append(g["x"])
        all_ei.append(g["edge_index"] + offset)
        all_ea.append(g["edge_attr"])
        if "y" in g:
            all_y.append(g["y"])
        all_batch.extend([g_idx] * n)
        offset += n

    collated = {
        "x":          np.concatenate(all_x, axis=0),
        "edge_index": np.concatenate(all_ei, axis=1) if all_ei else np.zeros((2, 0), dtype=np.int64),
        "edge_attr":  np.concatenate(all_ea, axis=0) if all_ea else np.zeros((0, BOND_FEAT_DIM), dtype=np.float32),
        "batch":      np.array(all_batch, dtype=np.int64),
    }
    if all_y:
        collated["y"] = np.stack(all_y, axis=0)

    return collated
