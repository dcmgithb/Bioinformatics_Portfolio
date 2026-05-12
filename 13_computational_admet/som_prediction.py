"""
Atom-Level Site of Metabolism (SoM) Prediction
================================================
Graph Neural Network that assigns a metabolic vulnerability score to every
heavy atom in a molecule, indicating the probability that the atom is a
Site of Metabolism (SoM) for Phase I oxidation or Phase II conjugation.

Architecture
------------
Input  : Molecule as a fully-connected atom graph with rich atomic features
         (Gasteiger charge, hybridisation, topological environment, CYP
         pharmacophore proximity features)
Model  : 3-layer MPNN → per-atom MLP → sigmoid probability
Output : Per-atom SoM scores [0,1] for six reaction classes:
         aromatic_hydroxylation, aliphatic_oxidation, n_dealkylation,
         o_dealkylation, glucuronidation, sulfation

Atom features (58-dim)
-----------------------
- One-hot: atomic number (common set + UNK)         13
- One-hot: degree 0-10 (+UNK)                       12
- One-hot: formal charge [-2,-1,0,1,2]               6
- One-hot: num total Hs [0-4]                        6
- One-hot: hybridisation [SP, SP2, SP3, SP3D, SP3D2] 6
- Binary:  is_aromatic, is_in_ring                   2
- One-hot: ring size [0,3,4,5,6,7,8+]               8
- Float:   Gasteiger charge (normalised)             1
- Float:   Crippen LogP contribution                 1
- Float:   TPSA contribution                         1
- One-hot: CYP pharmacophore class [none, basic_N,
           acid_O, aromatic_pi, aliphatic_C, heteroatom] (+UNK)  7
- Float:   distance to nearest N/O/S (topological)  1
Total:  64 features

References
----------
Zaretzki et al. (2012) RS-WebPredictor — Bioinformatics 28(4):497-505
Kirchmair et al. (2015) Metabolism prediction — J. Chem. Inf. Model. 55
Šícho et al. (2017) FAME 2 — J. Chem. Inf. Model. 57
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, rdMolDescriptors,
    Draw, rdDepictor, Crippen
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import DataStructs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ── Reaction class labels ─────────────────────────────────────────────────────

REACTION_CLASSES = [
    "aromatic_hydroxylation",
    "aliphatic_oxidation",
    "n_dealkylation",
    "o_dealkylation",
    "glucuronidation",
    "sulfation",
]
N_TASKS = len(REACTION_CLASSES)

COMMON_ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
RING_SIZES   = [0, 3, 4, 5, 6, 7, 8]


def one_hot(val, vocab, allow_unk=True):
    enc = [int(val == v) for v in vocab]
    if allow_unk:
        enc.append(int(val not in vocab))
    return enc


# ── CYP pharmacophore classification ─────────────────────────────────────────

def cyp_pharmacophore_class(atom: Chem.Atom) -> int:
    """
    Classify atom into CYP binding pocket pharmacophore class.
    0=none, 1=basic_N, 2=acid_O, 3=aromatic_pi, 4=aliphatic_C, 5=heteroatom
    """
    num = atom.GetAtomicNum()
    if num == 7 and atom.GetFormalCharge() >= 0 and atom.GetTotalNumHs() > 0:
        return 1  # basic nitrogen (common CYP2D6 feature)
    if num == 8 and atom.GetFormalCharge() < 0:
        return 2  # acidic oxygen (CYP2C9 feature)
    if num == 6 and atom.GetIsAromatic():
        return 3  # aromatic pi system (CYP1A2 feature)
    if num == 6 and not atom.GetIsAromatic():
        return 4  # aliphatic carbon
    if num in (7, 8, 16):
        return 5  # heteroatom
    return 0


def dist_to_heteroatom(mol: Chem.Mol, atom_idx: int) -> float:
    """Topological distance (bonds) to nearest N/O/S, normalised by molecule size."""
    dm = Chem.GetDistanceMatrix(mol)
    min_d = 9999
    for j, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() in (7, 8, 16) and j != atom_idx:
            min_d = min(min_d, dm[atom_idx, j])
    n = mol.GetNumHeavyAtoms()
    return float(min_d) / max(n, 1) if min_d < 9999 else 1.0


# ── Atom feature extraction (62-dim) ─────────────────────────────────────────

def atom_features_som(atom: Chem.Atom, mol: Chem.Mol) -> List[float]:
    """62-dimensional atom feature vector for SoM prediction."""
    feats: List[float] = []

    # Atomic number (13)
    feats += one_hot(atom.GetAtomicNum(), COMMON_ATOMS)
    # Degree (11)
    feats += one_hot(atom.GetDegree(), list(range(11)))
    # Formal charge (6)
    feats += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    # Num Hs (6)
    feats += one_hot(atom.GetTotalNumHs(), list(range(5)))
    # Hybridisation (6)
    feats += one_hot(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ])
    # Binary flags (2)
    feats += [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    # Ring size (8)
    ring_sz = 0
    if atom.IsInRing():
        for size in RING_SIZES[1:]:
            if atom.IsInRingSize(size):
                ring_sz = size
                break
        if ring_sz == 0:
            ring_sz = 8
    feats += one_hot(ring_sz, RING_SIZES)

    # Gasteiger charge (1)
    try:
        gc = float(atom.GetPropsAsDict().get("_GasteigerCharge", 0.0))
        if not np.isfinite(gc):
            gc = 0.0
    except Exception:
        gc = 0.0
    feats.append(gc)

    # Crippen LogP contribution (1)
    try:
        contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
        lp_contrib = float(contribs[atom.GetIdx()][0])
        if not np.isfinite(lp_contrib):
            lp_contrib = 0.0
    except Exception:
        lp_contrib = 0.0
    feats.append(lp_contrib)

    # TPSA contribution (1)
    try:
        tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        tpsa_c = float(tpsa_contribs[atom.GetIdx()])
        if not np.isfinite(tpsa_c):
            tpsa_c = 0.0
    except Exception:
        tpsa_c = 0.0
    feats.append(tpsa_c)

    # CYP pharmacophore class (6 one-hot)
    feats += one_hot(cyp_pharmacophore_class(atom), list(range(6)))

    # Distance to nearest heteroatom (1)
    feats.append(dist_to_heteroatom(mol, atom.GetIdx()))

    return feats


ATOM_FEAT_DIM = 64  # precomputed


def mol_to_atom_graph(
    mol: Chem.Mol,
    som_labels: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Convert molecule to atom graph dict.

    Parameters
    ----------
    mol        : RDKit molecule
    som_labels : [N_atoms, N_tasks] binary SoM labels (0/1 per atom per reaction class)

    Returns
    -------
    dict: x [N, 62], edge_index [2, 2E], y [N, N_tasks] or None
    """
    if mol is None:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    N = mol.GetNumHeavyAtoms()
    if N == 0:
        return None

    x = np.array([atom_features_som(a, mol) for a in mol.GetAtoms()], dtype=np.float32)

    rows, cols = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [i, j]
        cols += [j, i]

    edge_index = np.array([rows, cols], dtype=np.int64) if rows else np.zeros((2, 0), dtype=np.int64)

    graph = {"x": x, "edge_index": edge_index, "n_atoms": N,
             "smiles": Chem.MolToSmiles(mol)}
    if som_labels is not None:
        graph["y"] = som_labels.astype(np.float32)
    return graph


# ── GNN Model ─────────────────────────────────────────────────────────────────

class SoMLayer(nn.Module):
    """Single message-passing layer with residual connection."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.msg_net = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.update = nn.GRUCell(out_dim, in_dim)
        self.norm   = nn.LayerNorm(in_dim)
        self.proj   = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        if edge_index.shape[1] == 0:
            return x
        row, col = edge_index[0], edge_index[1]
        # Messages: concat source + target features
        msgs = self.msg_net(torch.cat([x[row], x[col]], dim=-1))
        # Aggregate (mean)
        agg = torch.zeros(x.size(0), msgs.size(-1), device=x.device)
        count = torch.zeros(x.size(0), 1, device=x.device)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msgs), msgs)
        count.scatter_add_(0, row.unsqueeze(-1),
                           torch.ones(row.size(0), 1, device=x.device))
        agg = agg / (count + 1e-8)
        x_new = self.update(agg, self.proj(x))
        return self.norm(x_new)


class SoMPredictor(nn.Module):
    """
    Atom-level SoM prediction network.

    Input  : per-atom features [N, 62]
    Output : per-atom SoM probabilities [N, N_tasks]
    """

    def __init__(
        self,
        in_dim: int    = ATOM_FEAT_DIM,
        hidden: int    = 128,
        n_layers: int  = 3,
        n_tasks: int   = N_TASKS,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            SoMLayer(hidden, hidden) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_tasks),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns [N, n_tasks] unnormalised logits."""
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.head(h)

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns [N, n_tasks] probabilities in [0,1]."""
        return torch.sigmoid(self.forward(x, edge_index))


# ── Synthetic training data ───────────────────────────────────────────────────

# Metabolic rules encoded as SMARTS for ground-truth label generation
SOM_RULES: Dict[str, Tuple[str, int]] = {
    # (SMARTS, reaction_class_index)
    "aromatic_C-H":        ("[cH]",         0),  # aromatic hydroxylation
    "aliphatic_CH3":       ("[CH3]",        1),  # aliphatic oxidation
    "aliphatic_CH2":       ("[CH2;!a]",     1),  # aliphatic oxidation
    "benzylic_CH":         ("[CH;$(CC~[cR])]", 1),
    "N_CH3":               ("[N;X3][CH3]",  2),  # N-dealkylation → label on N
    "O_CH3":               ("[O][CH3]",     3),  # O-dealkylation → label on O
    "OH_glucuronidation":  ("[OH;!$([OH]N)]", 4),  # glucuronidation
    "NH_glucuronidation":  ("[NH;!$([NH]C=O)]", 4),
    "COOH_glucuronidation":("[C](=O)[OH]",  4),
    "phenol_OH":           ("[OH;$(Oc1ccccc1)]", 4),  # phenol glucuronidation
    "sulfation_OH":        ("[OH;$(Oc1ccccc1)]", 5),  # phenol sulfation
    "sulfation_NH2":       ("[NH2;$(Nc1ccccc1)]", 5),
}


def generate_som_labels(mol: Chem.Mol) -> np.ndarray:
    """
    Generate binary SoM labels [N_atoms, N_tasks] using SMARTS rules.
    Adds 10% false positives and 15% false negatives to simulate
    experimental variability.
    """
    N      = mol.GetNumHeavyAtoms()
    labels = np.zeros((N, N_TASKS), dtype=np.float32)
    rng    = np.random.default_rng(abs(hash(Chem.MolToSmiles(mol))) % (2 ** 31))

    for rule_name, (smarts, cls_idx) in SOM_RULES.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                atom_idx = match[0]
                if atom_idx < N:
                    labels[atom_idx, cls_idx] = 1.0
        except Exception:
            pass

    # Add noise: 15% false negative (flip 1→0), 10% false positive (flip 0→1)
    for i in range(N):
        for j in range(N_TASKS):
            if labels[i, j] == 1.0 and rng.random() < 0.15:
                labels[i, j] = 0.0
            elif labels[i, j] == 0.0 and rng.random() < 0.05:
                labels[i, j] = 1.0

    return labels


# Drug-like SMILES for training/evaluation
TRAINING_SMILES = [
    "CC(=O)Nc1ccc(O)cc1",                         # Paracetamol
    "CC1=CC(=O)c2ccccc2C1=O",                      # Menadione
    "COc1ccc(CCN)cc1OC",                           # Dopamine analogue
    "Cn1ccnc1",                                    # Methylimidazole
    "c1ccc2[nH]ccc2c1",                           # Indole
    "CC(=O)Oc1ccccc1C(=O)O",                       # Aspirin
    "c1ccc(-c2ccncc2)cc1",                         # 4-phenylpyridine
    "Cc1ccc(S(N)(=O)=O)cc1",                       # Toluenesulfonamide
    "CC1CCCCC1N",                                  # Cyclohexylmethylamine
    "CCOCCO",                                      # Diethylene glycol
    "c1ccc(N)cc1",                                 # Aniline
    "CC(N)Cc1ccccc1",                              # Amphetamine
    "COc1cccc(OC)c1",                              # Dimethoxybenzene
    "CC(=O)c1ccccc1",                              # Acetophenone
    "O=C(O)c1ccccc1",                              # Benzoic acid
    "c1ccc(NC(=O)c2ccccc2)cc1",                    # Benzanilide
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                 # Ibuprofen
    "COc1ccc(C(C)C(=O)O)cc1",                      # Naproxen-like
    "Cc1cnc(NC(=O)c2ccccc2Cl)s1",                  # Chlormezanone-like
    "O=C(NCCN)c1ccccc1",                           # Benzamide amine
    "c1ccc2c(c1)CCCO2",                            # Chroman
    "CC(O)c1ccccc1",                               # 1-phenylethan-1-ol
    "c1ccnc(N)c1",                                 # 2-aminopyridine
    "O=C1CCCC(=O)N1",                              # Delta-valerolactam
    "CC1=CC(=O)NC(=S)N1",                          # Thiacetazone-like
    "c1ccc(-n2cccc2)cc1",                          # N-phenylpyrrole
    "CCC(CC)O",                                    # 3-pentanol
    "CC(N)C(=O)O",                                 # Alanine
    "c1cc(O)ccc1O",                                # Resorcinol
    "CCOC(=O)c1ccccc1",                            # Ethyl benzoate
    "CC1=CC(C)(C)CC(=O)O1",
    "Cc1ncc(CO)c(CN)c1O",                          # Pyridoxamine-like
    "c1cncc(N)c1",
    "CC(=O)OCC",
    "c1ccc(OC)cc1O",                               # Guaiacol (4-methoxyphenol)
    "CCN(CC)CCOC(=O)c1ccccc1",
    "c1cc2ccccc2[nH]1",                            # Indole
    "CNC(=O)c1ccccc1",
    "CCOC(=O)CC",
    "c1ccc2oc(=O)ccc2c1",                          # Coumarin
]


def create_training_dataset(smiles_list: List[str]) -> List[Dict]:
    """Convert SMILES list to labelled atom graph dataset."""
    dataset = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        labels = generate_som_labels(mol)
        graph  = mol_to_atom_graph(mol, som_labels=labels)
        if graph is not None:
            dataset.append(graph)
    return dataset


# ── Training ──────────────────────────────────────────────────────────────────

def graph_to_tensors(graph: Dict, device: str = "cpu"):
    x  = torch.tensor(graph["x"],          dtype=torch.float,  device=device)
    ei = torch.tensor(graph["edge_index"],  dtype=torch.long,   device=device)
    y  = torch.tensor(graph["y"],           dtype=torch.float,  device=device)
    return x, ei, y


def train_som_model(
    dataset: List[Dict],
    hidden: int    = 128,
    n_layers: int  = 3,
    epochs: int    = 60,
    lr: float      = 3e-4,
    device: str    = "cpu",
) -> Tuple[SoMPredictor, List[float]]:
    """Train SoM predictor on the atom graph dataset."""

    model = SoMPredictor(in_dim=ATOM_FEAT_DIM, hidden=hidden,
                          n_layers=n_layers, n_tasks=N_TASKS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Class imbalance: SoM atoms are minority → use weighted BCE
    # Positive weight ≈ N_neg / N_pos (roughly 10x)
    pos_weight = torch.tensor([10.0] * N_TASKS, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    losses = []
    print(f"Training SoM model on {len(dataset)} molecules × {N_TASKS} reaction classes...")

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(dataset))
        ep_loss = 0.0

        for idx in perm:
            g = dataset[idx]
            if "y" not in g:
                continue
            x, ei, y = graph_to_tensors(g, device)
            logits = model(x, ei)             # [N_atoms, N_tasks]
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()
        avg = ep_loss / max(len(dataset), 1)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.4f}")

    return model, losses


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_som(model: SoMPredictor, dataset: List[Dict], device: str = "cpu") -> Dict:
    """Compute per-class AUC and overall atom-level accuracy."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    model.eval()

    all_probs, all_true = [], []
    for g in dataset:
        if "y" not in g:
            continue
        x, ei, y = graph_to_tensors(g, device)
        probs = model.predict_proba(x, ei).cpu().numpy()
        all_probs.append(probs)
        all_true.append(g["y"])

    probs_arr = np.vstack(all_probs)   # [total_atoms, N_tasks]
    true_arr  = np.vstack(all_true)

    metrics = {}
    for i, cls in enumerate(REACTION_CLASSES):
        y_t = true_arr[:, i]
        y_p = probs_arr[:, i]
        if y_t.sum() > 0:
            metrics[f"auc_{cls}"]  = float(roc_auc_score(y_t, y_p))
            metrics[f"ap_{cls}"]   = float(average_precision_score(y_t, y_p))
        else:
            metrics[f"auc_{cls}"]  = float("nan")

    valid = [v for v in [metrics.get(f"auc_{c}") for c in REACTION_CLASSES]
             if np.isfinite(v)]
    metrics["mean_auc"] = float(np.mean(valid)) if valid else 0.0
    return metrics


# ── Inference + visualisation ─────────────────────────────────────────────────

@torch.no_grad()
def predict_som(
    model: SoMPredictor,
    smiles: str,
    device: str = "cpu",
) -> Optional[Dict]:
    """
    Predict SoM probabilities for a single molecule.

    Returns
    -------
    dict:
        smiles      : canonical SMILES
        atom_probs  : np.ndarray [N_atoms, N_tasks]
        top_sites   : list of (atom_idx, reaction_class, probability)
        dominant_class : most predicted reaction class
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    graph = mol_to_atom_graph(mol)
    x  = torch.tensor(graph["x"],         dtype=torch.float, device=device)
    ei = torch.tensor(graph["edge_index"], dtype=torch.long,  device=device)

    model.eval()
    probs = model.predict_proba(x, ei).cpu().numpy()  # [N, N_tasks]

    # Top sites: atoms with any reaction class probability > 0.4
    top_sites = []
    for i in range(probs.shape[0]):
        for j, cls in enumerate(REACTION_CLASSES):
            if probs[i, j] > 0.35:
                top_sites.append((i, cls, float(probs[i, j])))
    top_sites.sort(key=lambda x: x[2], reverse=True)

    # Dominant class overall
    class_sums = probs.sum(axis=0)
    dominant   = REACTION_CLASSES[int(class_sums.argmax())]

    return {
        "smiles":       Chem.MolToSmiles(mol),
        "atom_probs":   probs,
        "top_sites":    top_sites,
        "dominant_class": dominant,
        "n_atoms":      mol.GetNumHeavyAtoms(),
    }


def visualize_som(
    result: Dict,
    reaction_class: str = "aromatic_hydroxylation",
    out_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 4),
) -> plt.Figure:
    """
    Visualise SoM probabilities as a colour-coded atom heatmap.
    Atoms are coloured from white (low) to red (high) by SoM probability.
    """
    mol = Chem.MolFromSmiles(result["smiles"])
    if mol is None:
        raise ValueError("Invalid SMILES")

    rdDepictor.Compute2DCoords(mol)

    cls_idx = REACTION_CLASSES.index(reaction_class) if reaction_class in REACTION_CLASSES else 0
    probs   = result["atom_probs"][:, cls_idx]

    # Colour atoms by probability using a white→red gradient
    atom_colours = {}
    highlight_atoms = []
    for i, p in enumerate(probs):
        r, g, b = 1.0, max(0, 1.0 - p * 1.8), max(0, 1.0 - p * 1.8)
        atom_colours[i] = (r, g, b)
        if p > 0.25:
            highlight_atoms.append(i)

    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colours,
        highlightBonds=[],
        highlightBondColors={},
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={"width_ratios": [2, 1]})

    # Left: structure placeholder (use imshow for SVG substitute)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].axis("off")
    axes[0].set_title(f"SoM: {reaction_class}\n(red = high vulnerability)", fontsize=9)
    axes[0].text(0.5, 0.5, f"Structure: {result['smiles'][:40]}…",
                 ha="center", va="center", fontsize=7,
                 style="italic", color="grey",
                 transform=axes[0].transAxes)

    # Right: bar chart of top sites
    if result["top_sites"]:
        sites = result["top_sites"][:8]
        labels = [f"Atom {s[0]}\n({s[1][:6]})" for s in sites]
        values = [s[2] for s in sites]
        colours_bar = [cm.Reds(v) for v in values]
        axes[1].barh(range(len(sites)), values, color=colours_bar)
        axes[1].set_yticks(range(len(sites)))
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("SoM Probability", fontsize=9)
        axes[1].set_title("Top SoM Sites", fontsize=9)
        axes[1].axvline(0.5, ls="--", color="red", alpha=0.5, lw=1)
        axes[1].grid(True, alpha=0.3, axis="x")

    plt.suptitle(f"Site of Metabolism Prediction", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    print(f"\n[1/4] Building atom graph dataset ({len(TRAINING_SMILES)} molecules)...")
    dataset = create_training_dataset(TRAINING_SMILES)
    print(f"  {len(dataset)} molecules | "
          f"total atoms: {sum(g['n_atoms'] for g in dataset)}")

    # Count SoM atoms per class
    for j, cls in enumerate(REACTION_CLASSES):
        total_pos = sum(g["y"][:, j].sum() for g in dataset if "y" in g)
        total_neg = sum((1 - g["y"][:, j]).sum() for g in dataset if "y" in g)
        print(f"  {cls:30s}: {int(total_pos):4d} positive / {int(total_neg):4d} negative")

    train_d, val_d = train_test_split(dataset, test_size=0.2, random_state=SEED)

    print(f"\n[2/4] Training SoM predictor ({len(train_d)} train / {len(val_d)} val)...")
    model, losses = train_som_model(train_d, hidden=128, n_layers=3, epochs=50, device=DEVICE)

    print("\n[3/4] Evaluating SoM predictor...")
    metrics = evaluate_som(model, val_d, device=DEVICE)
    print(f"\n  {'Reaction Class':<32} {'AUC-ROC':>8} {'AP':>8}")
    print("  " + "-" * 50)
    for cls in REACTION_CLASSES:
        auc = metrics.get(f"auc_{cls}", float("nan"))
        ap  = metrics.get(f"ap_{cls}",  float("nan"))
        print(f"  {cls:<32} {auc:>8.3f} {ap:>8.3f}")
    print(f"\n  Mean AUC-ROC: {metrics['mean_auc']:.3f}")

    print("\n[4/4] Predicting SoM for example molecules...")
    test_compounds = {
        "Paracetamol":  "CC(=O)Nc1ccc(O)cc1",
        "Ibuprofen":    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "Caffeine":     "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Lidocaine":    "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        "Diazepam":     "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    }

    results_table = []
    for name, smi in test_compounds.items():
        result = predict_som(model, smi, device=DEVICE)
        if result:
            top = result["top_sites"][0] if result["top_sites"] else (None, "none", 0)
            results_table.append({
                "Compound":       name,
                "Top SoM atom":   top[0],
                "Reaction class": top[1],
                "Probability":    f"{top[2]:.3f}",
                "Dominant cls":   result["dominant_class"],
            })
            # Save visualisation
            fig = visualize_som(result, reaction_class=result["dominant_class"],
                                out_path=OUT_DIR / f"som_{name.lower()}.png")
            plt.close(fig)

    print("\n" + pd.DataFrame(results_table).to_string(index=False))
    print(f"\nSaved SoM heatmaps to {OUT_DIR}/")
