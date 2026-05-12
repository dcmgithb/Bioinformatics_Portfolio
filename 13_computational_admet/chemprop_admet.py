"""
chemprop_admet.py — Directed Message Passing Neural Network (D-MPNN) for ADMET.

Implements the ChemProp architecture (Yang et al. 2019, J. Chem. Inf. Model.)
from scratch in PyTorch — no torch_geometric required. Directed bond-level
message passing, atom readout, multi-task FFN head predicting CYP3A4
inhibition, hERG risk, and DILI from SMILES.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42):
        np.random.seed(s)
        torch.manual_seed(s)
    PALETTES = {"young": "#2196F3", "aged": "#F44336", "accent": "#4CAF50"}

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ──────────────────────────────────────────────────────────────────────────────
# Feature dimensions
# ──────────────────────────────────────────────────────────────────────────────

ATOM_FDIM = 73   # see atom_features()
BOND_FDIM = 13   # see bond_features()

ATOMIC_NUMS = list(range(1, 51))   # H–Sn (top-50 elements in drug molecules)
HYBRIDISATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
] if HAS_RDKIT else []

STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
] if HAS_RDKIT else []


def one_hot(value, choices: list) -> List[int]:
    enc = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    enc[idx] = 1
    return enc


# ──────────────────────────────────────────────────────────────────────────────
# Featurisation (RDKit path)
# ──────────────────────────────────────────────────────────────────────────────

def atom_features(atom) -> List[float]:
    """
    73-dimensional atom feature vector (matches ChemProp defaults):
      50+1 atomic num  |  6+1 degree  |  1 formal charge  |  5+1 Hs
      |  5+1 hybridisation  |  1 aromaticity  |  1 mass/100
    Each one_hot() adds 1 UNK bit → vocab of N → N+1 output bits.
    """
    return (
        one_hot(atom.GetAtomicNum(), ATOMIC_NUMS)          # 51  (50 + UNK)
        + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])   # 7   (6  + UNK)
        + [float(atom.GetFormalCharge())]                  # 1
        + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 6   (5  + UNK)
        + one_hot(atom.GetHybridization(), HYBRIDISATIONS) # 6   (5  + UNK)
        + [float(atom.GetIsAromatic())]                    # 1
        + [atom.GetMass() / 100.0]                         # 1
    )  # total = 51+7+1+6+6+1+1 = 73


def bond_features(bond) -> List[float]:
    """
    13-dimensional bond feature vector:
      4 bond type  |  1 conjugated  |  1 ring  |  6+1 stereo
    one_hot(stereo, 6 choices) → 7 bits (6 + UNK); total = 4+1+1+7 = 13.
    """
    bt = bond.GetBondTypeAsDouble()
    return (
        [float(bt == 1.0), float(bt == 1.5),
         float(bt == 2.0), float(bt == 3.0)]              # 4  single/aromatic/double/triple
        + [float(bond.GetIsConjugated())]                  # 1
        + [float(bond.IsInRing())]                         # 1
        + one_hot(bond.GetStereo(), STEREO_TYPES)          # 7  (6 + UNK)
    )  # total = 13


# ──────────────────────────────────────────────────────────────────────────────
# Molecule → directed graph tensors
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MolGraph:
    """Directed graph for one molecule."""
    n_atoms:      int
    n_bonds:      int                   # directed edges (= 2 × #bonds)
    atom_feats:   torch.Tensor          # [n_atoms, ATOM_FDIM]
    bond_feats:   torch.Tensor          # [n_bonds, BOND_FDIM]
    src:          torch.Tensor          # [n_bonds] — source atom of each directed edge
    dst:          torch.Tensor          # [n_bonds] — dest atom
    rev_idx:      torch.Tensor          # [n_bonds] — index of reverse directed edge


def mol_to_graph(smiles: str) -> Optional[MolGraph]:
    """Convert SMILES to directed MolGraph. Returns None on parse failure."""
    if not HAS_RDKIT:
        return _mock_mol_graph()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    a_feats = [atom_features(a) for a in mol.GetAtoms()]
    atom_feats = torch.tensor(a_feats, dtype=torch.float32)

    # Build directed edges: each bond → two directed edges
    src_list, dst_list, b_feats, rev_map = [], [], [], {}
    edge_idx = 0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf    = bond_features(bond)

        # i→j
        src_list.append(i); dst_list.append(j)
        b_feats.append(bf)
        fwd = edge_idx; edge_idx += 1

        # j→i
        src_list.append(j); dst_list.append(i)
        b_feats.append(bf)
        rev = edge_idx; edge_idx += 1

        rev_map[fwd] = rev
        rev_map[rev] = fwd

    if not src_list:
        # Single atom, no bonds
        src_list  = [0]; dst_list  = [0]
        b_feats   = [[0.0] * BOND_FDIM]
        rev_map   = {0: 0}

    n_bonds  = len(src_list)
    rev_idx  = torch.tensor([rev_map[i] for i in range(n_bonds)], dtype=torch.long)

    return MolGraph(
        n_atoms    = n_atoms,
        n_bonds    = n_bonds,
        atom_feats = atom_feats,
        bond_feats = torch.tensor(b_feats, dtype=torch.float32),
        src        = torch.tensor(src_list, dtype=torch.long),
        dst        = torch.tensor(dst_list, dtype=torch.long),
        rev_idx    = rev_idx,
    )


def _mock_mol_graph(n_atoms: int = 10) -> MolGraph:
    """Random graph used when RDKit is unavailable (mock/test mode)."""
    n_bonds = n_atoms * 2
    return MolGraph(
        n_atoms    = n_atoms,
        n_bonds    = n_bonds,
        atom_feats = torch.randn(n_atoms, ATOM_FDIM),
        bond_feats = torch.randn(n_bonds, BOND_FDIM),
        src        = torch.randint(0, n_atoms, (n_bonds,)),
        dst        = torch.randint(0, n_atoms, (n_bonds,)),
        rev_idx    = torch.arange(n_bonds),
    )


# ──────────────────────────────────────────────────────────────────────────────
# D-MPNN model
# ──────────────────────────────────────────────────────────────────────────────

class DMPNNLayer(nn.Module):
    """
    One directed message-passing step.

    For each directed edge (v→w):
        message(v→w) = Σ_{u∈N(v) \ w} h(u→v)   [excludes reverse edge]
        h_new(v→w)   = ReLU(h_init(v→w) + W_m · message(v→w))
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        h:       torch.Tensor,   # [n_bonds, hidden_dim] current bond hidden states
        h_init:  torch.Tensor,   # [n_bonds, hidden_dim] initial bond embeddings (residual)
        src:     torch.Tensor,   # [n_bonds] source atom index per directed edge
        n_atoms: int,
        rev_idx: torch.Tensor,   # [n_bonds] reverse edge index
    ) -> torch.Tensor:
        # Sum all incoming hidden states per atom: agg[atom] = Σ h(u→atom)
        agg = torch.zeros(n_atoms, h.size(1), device=h.device)
        agg.index_add_(0, src, h)                        # scatter-add into atoms

        # For each directed edge (v→w): message = agg[v] - h(w→v)
        # i.e. sum of all edges entering v, minus the reverse edge
        msg = agg[src] - h[rev_idx]                      # [n_bonds, hidden_dim]

        return F.relu(h_init + self.W_m(msg))


class DMPNN(nn.Module):
    """
    Full ChemProp D-MPNN.

    Architecture:
        bond init → depth × DMPNNLayer → atom readout → mean pool → FFN → n_tasks outputs
    """

    def __init__(
        self,
        n_tasks:    int   = 3,
        hidden_dim: int   = 300,
        depth:      int   = 3,
        dropout:    float = 0.0,
        ffn_hidden: int   = 300,
    ) -> None:
        super().__init__()
        self.depth  = depth
        self.hidden_dim = hidden_dim

        # Bond initialisation: [x_v ‖ e_vw] → hidden_dim
        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_dim, bias=False)

        # Message passing layers
        self.mp_layers = nn.ModuleList([DMPNNLayer(hidden_dim) for _ in range(depth)])

        # Atom readout: [x_v ‖ m_v] → hidden_dim
        self.W_a = nn.Linear(ATOM_FDIM + hidden_dim, hidden_dim)

        # Feed-forward head
        self.ffn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, n_tasks),
        )

    def forward(self, graphs: List[MolGraph]) -> torch.Tensor:
        """
        Process a batch of MolGraphs, return [batch, n_tasks] logits.
        """
        mol_vecs = [self._encode_mol(g) for g in graphs]
        batch    = torch.stack(mol_vecs, dim=0)          # [B, hidden_dim]
        return self.ffn(batch)                            # [B, n_tasks]

    def _encode_mol(self, g: MolGraph) -> torch.Tensor:
        dev = next(self.parameters()).device

        af  = g.atom_feats.to(dev)     # [n_atoms, ATOM_FDIM]
        bf  = g.bond_feats.to(dev)     # [n_bonds, BOND_FDIM]
        src = g.src.to(dev)
        dst = g.dst.to(dev)
        rev = g.rev_idx.to(dev)

        # Initial bond embedding: [x_src ‖ e_vw]
        h_init = F.relu(self.W_i(torch.cat([af[src], bf], dim=1)))  # [n_bonds, H]
        h      = h_init.clone()

        # Directed message passing
        for layer in self.mp_layers:
            h = layer(h, h_init, src, g.n_atoms, rev)

        # Atom-level readout: sum incoming bond hiddens per atom
        m_atom = torch.zeros(g.n_atoms, self.hidden_dim, device=dev)
        m_atom.index_add_(0, dst, h)                     # [n_atoms, H]

        h_atom = F.relu(self.W_a(torch.cat([af, m_atom], dim=1)))  # [n_atoms, H]

        # Mean pool over atoms → molecule vector
        return h_atom.mean(dim=0)                         # [H]


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic ADMET dataset
# ──────────────────────────────────────────────────────────────────────────────

SMILES_POOL = [
    "CC1=NC(=CC(=O)N1)C2=CC=C(C=C2)NC(=O)C3=CC=NC=C3",
    "CC(C)CC1=CC(=CC(=C1)C(C)C)C(C)C",
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "C1=CC(=CC=C1N)S(=O)(=O)N",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "CC(=O)Nc1ccc(cc1)O",
    "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)CCN1",
    "O=C(O)[C@@H](N)Cc1ccc(O)cc1",
    "CC(=O)c1ccc(cc1)N",
    "NC(=O)c1cccnc1",
    "O=C(O)c1cccc(c1)O",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "CC(C)(C)c1ccc(cc1)O",
    "O=C(O)CCc1ccccc1",
    "CC1=CC(=O)c2ccccc2C1=O",
    "c1ccc(cc1)CN",
    "O=C(O)[C@@H](N)CC(=O)O",
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "CCOC(=O)c1ccc(cc1)N",
    "CC1=CC=C(C=C1)C(=O)O",
    "NC(=O)c1cccc(c1)O",
    "Clc1ccc(cc1)C(=O)O",
    "CC(=O)Nc1cccc(c1)O",
    "O=C(O)c1ccc(O)c(O)c1",
    "CC(C)c1ccc(cc1)C(C)C",
    "c1cc(ccc1N)N",
    "O=S(=O)(O)c1ccccc1",
    "CC(C)(C)OC(=O)NC(Cc1ccccc1)C(=O)O",
    "O=C(O)c1ccc(cc1)Cl",
    "c1ccc(cc1)O",
    "O=C(O)C(O)(CC(=O)O)CC(=O)O",
    "OCC1OC(O)C(O)C(O)C1O",
    "O=C1CCCCC1",
    "c1ccc(cc1)CC(=O)O",
    "O=C(O)c1cccc(N)c1",
    "NC1=NC(=O)C2=C(N1)N=CN2",
    "O=C(O)[C@H](O)[C@@H](O)C(=O)O",
    "CCCCCCCCCCCCCCCC(=O)O",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "c1cnc(nc1)N",
    "O=C(O)CC(=O)O",
    "OC1=CC=CC=C1C(=O)O",
    "CN(C)c1ccc(cc1)C=O",
    "O=Cc1ccc(O)cc1",
    "CC(=O)NCCO",
    "NCCCO",
    "OCC(O)CO",
    "CC(O)CO",
    "OCCO",
    "CCO",
]

TASKS = ["CYP3A4_inhibition", "hERG_risk", "DILI"]


def _rdkit_descriptors(smiles: str, rng) -> np.ndarray:
    """Fallback descriptor vector when graph encoding is unavailable."""
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array([
                Descriptors.MolWt(mol) / 600,
                Descriptors.MolLogP(mol) / 6,
                rdMolDescriptors.CalcNumHBD(mol) / 5,
                rdMolDescriptors.CalcNumHBA(mol) / 10,
                rdMolDescriptors.CalcTPSA(mol) / 150,
                rdMolDescriptors.CalcNumRotatableBonds(mol) / 10,
                rdMolDescriptors.CalcNumAromaticRings(mol) / 5,
                float(rdMolDescriptors.CalcNumRings(mol)) / 5,
            ], dtype=np.float32)
    return rng.random(8).astype(np.float32)


def generate_admet_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic multi-task ADMET dataset.
    Labels follow realistic SAR rules:
      CYP3A4 inhibition — lipophilic, aromatic, basic nitrogen
      hERG risk         — high LogP, aromatic, cationic
      DILI              — reactive groups, high MW
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        smiles = SMILES_POOL[i % len(SMILES_POOL)]
        desc   = _rdkit_descriptors(smiles, rng)
        mw_n, logp_n, hbd_n, hba_n, tpsa_n, rot_n, arom_n, rings_n = desc

        noise = rng.normal(0, 0.1, 3)
        cyp_score  = 0.4 * logp_n + 0.3 * arom_n + 0.2 * hba_n + noise[0]
        herg_score = 0.5 * logp_n + 0.3 * arom_n + 0.2 * mw_n  + noise[1]
        dili_score = 0.3 * mw_n   + 0.3 * logp_n + 0.2 * arom_n + noise[2]

        rows.append({
            "smiles":            smiles,
            "CYP3A4_inhibition": int(cyp_score  > 0.5),
            "hERG_risk":         int(herg_score > 0.5),
            "DILI":              int(dili_score  > 0.45),
            "mw_norm":    float(mw_n),
            "logp_norm":  float(logp_n),
            "tpsa_norm":  float(tpsa_n),
            "arom_rings": float(arom_n),
        })

    df = pd.DataFrame(rows)
    # Re-threshold to ~40% positive rate per task
    for task in TASKS:
        threshold = df[task].quantile(0.60)
        df[task]  = (df[task] >= threshold).astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def build_graphs(smiles_list: List[str]) -> List[MolGraph]:
    graphs = []
    for smi in smiles_list:
        g = mol_to_graph(smi)
        if g is None:
            g = _mock_mol_graph()
        graphs.append(g)
    return graphs


def train_dmpnn(
    train_graphs: List[MolGraph],
    train_labels: np.ndarray,
    val_graphs:   List[MolGraph],
    val_labels:   np.ndarray,
    n_tasks:      int   = 3,
    hidden_dim:   int   = 128,
    depth:        int   = 3,
    epochs:       int   = 30,
    lr:           float = 1e-3,
    dropout:      float = 0.1,
    seed:         int   = 42,
) -> Tuple[DMPNN, List[float], List[float]]:
    torch.manual_seed(seed)
    device = torch.device("cpu")

    model = DMPNN(n_tasks=n_tasks, hidden_dim=hidden_dim, depth=depth,
                  dropout=dropout).to(device)
    opt   = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit  = nn.BCEWithLogitsLoss()

    y_train = torch.tensor(train_labels, dtype=torch.float32)
    y_val   = torch.tensor(val_labels,   dtype=torch.float32)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(train_graphs)              # [N_train, n_tasks]
        loss   = crit(logits, y_train)
        loss.backward()
        opt.step()
        train_losses.append(float(loss))

        model.eval()
        with torch.no_grad():
            val_logits = model(val_graphs)
            val_loss   = crit(val_logits, y_val)
        val_losses.append(float(val_loss))

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  train_loss={loss:.4f}  val_loss={val_loss:.4f}")

    return model, train_losses, val_losses


def evaluate(
    model: DMPNN, graphs: List[MolGraph], labels: np.ndarray
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(graphs)
        probs  = torch.sigmoid(logits).numpy()

    metrics = {}
    for i, task in enumerate(TASKS):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
            apr = average_precision_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = apr = float("nan")
        metrics[task] = {"auc_roc": auc, "auc_pr": apr}
    return metrics


def gbm_baseline(
    train_feats: np.ndarray, train_labels: np.ndarray,
    val_feats:   np.ndarray, val_labels:   np.ndarray,
) -> Dict[str, float]:
    """GBM baseline using RDKit descriptors for comparison."""
    metrics = {}
    for i, task in enumerate(TASKS):
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(train_feats, train_labels[:, i])
        probs = clf.predict_proba(val_feats)[:, 1]
        try:
            auc = roc_auc_score(val_labels[:, i], probs)
        except ValueError:
            auc = float("nan")
        metrics[task] = {"auc_roc": auc}
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_chemprop_results(
    model:        DMPNN,
    val_graphs:   List[MolGraph],
    val_labels:   np.ndarray,
    train_losses: List[float],
    val_losses:   List[float],
    gbm_metrics:  Dict,
    out_path:     str = "figures/chemprop_results.png",
) -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    from sklearn.metrics import roc_curve

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(val_graphs)).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#FAFAFA")
    colors = [PALETTES.get("young", "#2196F3"),
              PALETTES.get("aged", "#F44336"),
              PALETTES.get("accent", "#4CAF50")]

    # Panel 1: ROC curves per task
    ax = axes[0]
    for i, (task, color) in enumerate(zip(TASKS, colors)):
        try:
            fpr, tpr, _ = roc_curve(val_labels[:, i], probs[:, i])
            auc = roc_auc_score(val_labels[:, i], probs[:, i])
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{task.replace('_', ' ')} AUC={auc:.3f}")
        except ValueError:
            pass
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("D-MPNN ROC Curves (per task)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F5F5F5")

    # Panel 2: Training / validation loss curves
    ax = axes[1]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=PALETTES.get("young", "#2196F3"),
            linewidth=2, label="Train loss")
    ax.plot(epochs, val_losses,   color=PALETTES.get("aged", "#F44336"),
            linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("D-MPNN Training Curve", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor("#F5F5F5")

    # Panel 3: D-MPNN vs GBM AUC comparison
    ax = axes[2]
    dmpnn_aucs = []
    gbm_aucs   = []
    for i, task in enumerate(TASKS):
        try:
            dmpnn_aucs.append(roc_auc_score(val_labels[:, i], probs[:, i]))
        except ValueError:
            dmpnn_aucs.append(0.5)
        gbm_aucs.append(gbm_metrics.get(task, {}).get("auc_roc", 0.5))

    x = np.arange(len(TASKS))
    width = 0.35
    ax.bar(x - width/2, dmpnn_aucs, width, label="D-MPNN (ChemProp)",
           color=PALETTES.get("young", "#2196F3"), edgecolor="white")
    ax.bar(x + width/2, gbm_aucs, width, label="GBM baseline",
           color=PALETTES.get("neutral", "#9E9E9E"), edgecolor="white")
    for xi, (d, g) in enumerate(zip(dmpnn_aucs, gbm_aucs)):
        ax.text(xi - width/2, d + 0.01, f"{d:.2f}", ha="center", fontsize=8)
        ax.text(xi + width/2, g + 0.01, f"{g:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in TASKS], fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 1.12)
    ax.set_title("D-MPNN vs GBM Baseline", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F5F5F5")

    plt.suptitle("ChemProp D-MPNN — Multi-task ADMET Prediction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    print(f"RDKit available : {HAS_RDKIT}")
    print(f"PyTorch         : {torch.__version__}")

    print("\nGenerating synthetic ADMET dataset (n=500) …")
    df = generate_admet_dataset(n=500, seed=42)
    for task in TASKS:
        print(f"  {task:<25}  positive rate: {df[task].mean():.1%}")

    # Split
    idx = np.arange(len(df))
    tr_idx, va_idx = train_test_split(idx, test_size=0.20, random_state=42)
    train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]

    y_train = train_df[TASKS].values.astype(np.float32)
    y_val   = val_df[TASKS].values.astype(np.float32)

    # Build molecule graphs
    print("\nBuilding directed graphs …")
    train_graphs = build_graphs(train_df["smiles"].tolist())
    val_graphs   = build_graphs(val_df["smiles"].tolist())
    print(f"  Train: {len(train_graphs)} graphs  Val: {len(val_graphs)} graphs")
    g0 = train_graphs[0]
    print(f"  Example molecule: {g0.n_atoms} atoms, {g0.n_bonds} directed edges")
    print(f"  Atom feat dim: {g0.atom_feats.shape[1]}  Bond feat dim: {g0.bond_feats.shape[1]}")

    # Train D-MPNN
    print(f"\nTraining D-MPNN (hidden={128}, depth=3, epochs=30) …")
    model, tr_losses, va_losses = train_dmpnn(
        train_graphs, y_train, val_graphs, y_val,
        n_tasks=len(TASKS), hidden_dim=128, depth=3, epochs=30, seed=42,
    )

    # Evaluate
    dmpnn_metrics = evaluate(model, val_graphs, y_val)
    print("\n── D-MPNN results ──")
    for task, m in dmpnn_metrics.items():
        print(f"  {task:<25}  AUC-ROC={m['auc_roc']:.4f}  AUC-PR={m['auc_pr']:.4f}")

    # GBM baseline on RDKit descriptors
    desc_cols = ["mw_norm", "logp_norm", "tpsa_norm", "arom_rings"]
    X_tr = train_df[desc_cols].values
    X_va = val_df[desc_cols].values
    gbm_metrics = gbm_baseline(X_tr, y_train, X_va, y_val)
    print("\n── GBM baseline (RDKit descriptors) ──")
    for task, m in gbm_metrics.items():
        print(f"  {task:<25}  AUC-ROC={m['auc_roc']:.4f}")

    os.makedirs("figures", exist_ok=True)
    img = plot_chemprop_results(
        model, val_graphs, y_val, tr_losses, va_losses,
        gbm_metrics, out_path="figures/chemprop_results.png",
    )
    print(f"\nPlot saved → {img}")
