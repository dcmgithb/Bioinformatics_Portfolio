"""
De Novo CDK4/6 Inhibitor Generation Pipeline
=============================================
Full workflow: corpus prep → pre-training → fine-tuning → RL optimisation → analysis

Sections
--------
1. Simulate training corpus (3000 drug-like SMILES)
2. Build SMILES tokenizer vocabulary
3. Pre-train SmilesTransformer (language modelling)
4. Fine-tune on 50 CDK4/6 reference actives
5. RL optimisation (REINFORCE, 200 steps)
6. Analysis:
   a. Validity / uniqueness / novelty / diversity
   b. Property distributions vs training set
   c. t-SNE chemical space
   d. Scaffold novelty
   e. Top-10 by composite reward
7. Learning curves

Usage
-----
    python generation_pipeline.py

Python  : >= 3.10
PyTorch : >= 2.0
RDKit   : >= 2023.09
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED as RDKitQED
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from smiles_tokenizer import SmilesTokenizer
from smiles_transformer import SmilesTransformer
from rl_optimizer import RLOptimizer, MolecularReward, pretrain_on_smiles

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


# ── CDK4/6 reference actives (50 real/scaffold SMILES) ───────────────────────

CDK46_ACTIVES = [
    # Palbociclib & analogues
    "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)CCN1",
    "CC1=C(C(=O)Nc2ncnc3ccc(F)cc23)CCN1",
    "CC1=C(C(=O)Nc2nc(-c3ccccc3)nc3[nH]ccc23)CCN1",
    "C1CN(CCN1)c1ccc2nc(Nc3ccc(F)cc3)ncc2n1",
    "C1CN(CCN1)c1ccc2nc(Nc3cccc(F)c3)ncc2n1",
    "C1CN(CCN1)c1ccc2nc(Nc3ccc(Cl)cc3)ncc2n1",
    "C1CN(CCN1)c1ccc2nc(Nc3ccc(OC)cc3)ncc2n1",
    "CC1CCN(c2ccc3nc(Nc4ccc(F)cc4)ncc3n2)CC1",
    # Ribociclib & analogues
    "Cc1cnc2nc(Nc3ccccc3)ncc2c1N1CCCC1",
    "Cc1cnc2nc(Nc3cccc(F)c3)ncc2c1N1CCCC1",
    "Cc1cnc2nc(Nc3ccc(Cl)cc3)ncc2c1N1CCCC1",
    "Cc1cnc2nc(Nc3ccc(OC)cc3)ncc2c1N1CCOCC1",
    "Cc1cnc2nc(Nc3ccc(F)cc3F)ncc2c1N1CCNCC1",
    # Abemaciclib & analogues
    "Cc1ncc2nc(Nc3ccc(N4CCOCC4)cc3)ncc2c1",
    "Cc1ncc2nc(Nc3ccc(N4CCCC4)cc3)ncc2c1",
    "Cc1ncc2nc(Nc3ccc(N4CCNCC4)cc3F)ncc2c1",
    "Fc1ccc(Nc2ncnc3cc(N4CCOCC4)cnc23)cc1",
    "Fc1ccc(Nc2ncnc3cc(N4CCCC4)cnc23)cc1",
    # Trilaciclib & analogues
    "CC1(CCC(=O)Nc2ccc3c(ncnc3N3CCNCC3)c2)CC1",
    "CC1(CCC(=O)Nc2ccc3c(ncnc3N3CCOCC3)c2)CC1",
    # Novel scaffolds (kinase-like)
    "c1ccc2nc(N3CCNCC3)c(Nc3cccc(F)c3)nc2c1",
    "c1ccc2nc(N3CCNCC3)c(Nc3ccc(F)cc3)nc2c1",
    "Cc1cc(=O)[nH]c(=O)n1Cc1ccc(Nc2ncnc3ccccc23)cc1",
    "FC(F)(F)c1ccc(Nc2ncnc3cc(N4CCNCC4)cnc23)cc1",
    "CC(C)c1nc(Nc2ccc(F)cc2)c2cccnc2n1",
    # Pyrrolo[2,3-d]pyrimidines
    "c1cnc2[nH]ccc2c1Nc1ccc(N2CCNCC2)cc1",
    "c1cnc2[nH]ccc2c1Nc1cccc(F)c1",
    "Cc1ccc(Nc2ncnc3[nH]ccc23)cc1",
    "c1ccc(Nc2ncnc3[nH]ccc23)cc1",
    # Pyrido[2,3-d]pyrimidines
    "Nc1nc2ccccc2c(=O)[nH]1",
    "Cc1ccc2nc(Nc3ccc(F)cc3)ncc2c1",
    "c1ccc2nc(Nc3ccc(N4CCNCC4)cc3)ncc2c1",
    # Additional variants
    "O=C(Nc1ccc(Nc2ncnc3cccnc23)cc1)c1ccccc1",
    "CC1CCN(c2ncc3cnc(Nc4ccc(F)cc4)nc3c2)CC1",
    "FC(F)(F)c1ccc(Nc2ncc3cnc(N4CCNCC4)nc3c2)cc1",
    "Cc1cc2nc(Nc3cccc(Cl)c3)ncc2s1",
    "c1ccc(Nc2ncnc3cc(N4CCOCC4)cnc23)nc1",
    "Cc1ccc(NC(=O)c2cc(-c3cnc4ncccc4n3)ccc2F)cc1",
    "O=C(c1cncc(Cl)c1)Nc1ccc2nc(N3CCNCC3)cnc2c1",
    "CC(=O)Nc1ccc(Nc2ncnc3cc(N4CCCC4)cnc23)cc1",
    "CS(=O)(=O)c1ccc(Nc2ncnc3cc(N4CCCC4)cnc23)cc1",
    "Cc1nc2c(N3CCNCC3)ncnc2s1",
    "Cc1ccc2c(Nc3ccc(N4CCNCC4)cc3)ncnc2c1",
    "c1ccc2c(Nc3ccc(N4CCOCC4)cc3F)ncnc2c1",
    "C1CN(c2nc3c(cc2F)cccc3)CCN1",
    "Fc1ccc(Nc2ncnc3[nH]cc(-c4cccnc4)c23)cc1",
    "CC1CN(c2ncc3cnc(Nc4ccc(F)cc4F)nc3c2)CCN1",
    "c1nc2ccc(N3CCNCC3)nc2nc1Nc1cccc(F)c1",
    "Cc1ccc(N)c(Nc2ncnc3ccc(N4CCNCC4)nc23)c1",
]

# CDK2 anti-targets (enforce selectivity during RL)
CDK2_ANTITARGETS = [
    "c1ccc2nc(Nc3ccccc3)c(O)nc2c1",
    "CC1=CC(=O)c2ccccc2C1=O",
    "c1ccc(-c2cnc3ncccc3n2)cc1",
    "C1CC(=O)Nc2ccccc21",
    "O=C1c2ccccc2NC1=O",
]


# ── 1. Training corpus generation ─────────────────────────────────────────────

SCAFFOLD_SMARTS = [
    # Pyrimidine-based
    "c1cnc(N)nc1", "c1ncnc2[nH]ccc12", "c1cnccn1",
    # Benzimidazole
    "c1ccc2[nH]cnc2c1", "c1ccc2ncnc(N)c2c1",
    # Piperazine/piperidine decorations
    "C1CNCCN1", "C1CCNCC1", "C1COCCN1",
    # Fluorophenyl
    "Fc1ccccc1", "Fc1cccc(F)c1", "Fc1ccc(F)cc1",
    # Benzyl/phenyl
    "c1ccccc1", "c1ccncc1", "c1cncnc1",
]

def generate_corpus(n: int = 3000, seed: int = SEED) -> list[str]:
    """
    Generate a diverse training corpus of drug-like SMILES by combining
    common medicinal chemistry building blocks.
    Returns a list of valid canonical SMILES.
    """
    rng = np.random.default_rng(seed)
    corpus = list(CDK46_ACTIVES)  # seed corpus with known actives

    templates = [
        # Heteroaromatic cores with amine/amide connections
        "c1nc({A})nc({B})c1",
        "c1ccc(N{A})cc1",
        "c1cnc({A})nc1{B}",
        "C1CCN(CC1){A}",
        "C1CCNCC1{A}",
        "C1CNCCN1{A}",
        "{A}Nc1ccccc1{B}",
        "{A}Nc1ccc({B})cc1",
    ]
    fragment_a = [
        "c1ccc(F)cc1", "c1cccc(F)c1", "c1ccc(Cl)cc1",
        "c1ccc(OC)cc1", "c1ccncc1", "C(=O)C", "c1ccc2[nH]cnc2c1",
        "c1ncnc2[nH]ccc12",
    ]
    fragment_b = [
        "N1CCNCC1", "N1CCCCC1", "N1CCOCC1", "F", "Cl",
        "OC", "NC(C)=O", "C(F)(F)F",
    ]

    attempts = 0
    while len(corpus) < n and attempts < n * 10:
        attempts += 1
        # Randomly combine fragments
        a = rng.choice(fragment_a)
        b = rng.choice(fragment_b)
        t = rng.choice(templates)
        smi = t.replace("{A}", a).replace("{B}", b)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        # Lipinski filter
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        if mw < 150 or mw > 600 or logp > 6:
            continue
        corpus.append(canon)

    # Deduplicate
    corpus = list(dict.fromkeys(corpus))
    return corpus[:n]


# ── 2. Validity / diversity metrics ──────────────────────────────────────────

def compute_generation_metrics(
    generated: list[str],
    training_smiles: list[str],
) -> dict:
    """Validity, uniqueness, novelty, diversity (internal Tanimoto)."""
    valid_mols = []
    valid_smi  = []
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_mols.append(mol)
            valid_smi.append(Chem.MolToSmiles(mol))

    validity  = len(valid_smi) / max(len(generated), 1)
    unique    = set(valid_smi)
    uniqueness = len(unique) / max(len(valid_smi), 1)

    train_set = set(Chem.MolToSmiles(Chem.MolFromSmiles(s))
                    for s in training_smiles
                    if Chem.MolFromSmiles(s))
    novel     = unique - train_set
    novelty   = len(novel) / max(len(unique), 1)

    # Internal diversity: mean pairwise Tanimoto distance
    if len(valid_mols) > 1:
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in valid_mols[:100]]
        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        diversity = 1.0 - float(np.mean(sims)) if sims else 0.0
    else:
        diversity = 0.0

    return {
        "validity":   validity,
        "uniqueness": uniqueness,
        "novelty":    novelty,
        "diversity":  diversity,
        "n_valid":    len(valid_smi),
        "n_unique":   len(unique),
        "n_novel":    len(novel),
    }


def scaffold_novelty(generated_smiles: list[str], reference_smiles: list[str]) -> dict:
    """Count new Murcko scaffolds in generated set vs reference."""
    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        except Exception:
            return None

    ref_scaffolds = set(filter(None, [get_scaffold(s) for s in reference_smiles]))
    gen_scaffolds = set(filter(None, [get_scaffold(s) for s in generated_smiles]))
    new_scaffolds = gen_scaffolds - ref_scaffolds
    return {
        "ref_scaffolds":  len(ref_scaffolds),
        "gen_scaffolds":  len(gen_scaffolds),
        "new_scaffolds":  len(new_scaffolds),
        "scaffold_novelty": len(new_scaffolds) / max(len(gen_scaffolds), 1),
    }


# ── 3. Visualisations ─────────────────────────────────────────────────────────

def get_mol_props(smiles_list: list[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            rows.append({
                "smiles": smi,
                "MW":     Descriptors.MolWt(mol),
                "LogP":   Descriptors.MolLogP(mol),
                "QED":    float(RDKitQED.qed(mol)),
                "HBD":    Descriptors.NumHDonors(mol),
                "TPSA":   Descriptors.TPSA(mol),
            })
    return pd.DataFrame(rows)


def morgan_matrix(smiles_list: list[str], n_bits: int = 1024) -> np.ndarray:
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
        else:
            arr = np.zeros(n_bits, dtype=np.float32)
        fps.append(arr)
    return np.vstack(fps)


def plot_results(
    train_smiles: list[str],
    generated_smiles: list[str],
    rl_history: list[dict],
    reward_fn: MolecularReward,
    out_dir: Path,
):
    """4-panel results figure."""
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    valid_gen = [s for s in generated_smiles if Chem.MolFromSmiles(s)]

    # ── Panel 1: RL learning curve ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    steps       = [h.get("step", i) for i, h in enumerate(rl_history)]
    mean_reward = [h["mean_reward"] for h in rl_history]
    max_reward  = [h["max_reward"]  for h in rl_history]
    valid_frac  = [h["valid_frac"]  for h in rl_history]

    ax1.plot(steps, mean_reward, "b-", lw=2, label="Mean reward")
    ax1.plot(steps, max_reward,  "r-", lw=2, label="Max reward")
    ax1_r = ax1.twinx()
    ax1_r.plot(steps, [v * 100 for v in valid_frac], "g--", lw=1.5, alpha=0.7,
               label="Valid %")
    ax1_r.set_ylabel("Validity (%)", color="green", fontsize=9)
    ax1.set_xlabel("RL Step", fontsize=10)
    ax1.set_ylabel("Reward", fontsize=10)
    ax1.set_title("REINFORCE Learning Curve", fontsize=11, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: QED distribution comparison ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    train_props = get_mol_props(train_smiles[:200])
    gen_props   = get_mol_props(valid_gen[:200])

    for data, label, color in [(train_props["QED"], "Training", "#3498DB"),
                                (gen_props["QED"],   "Generated (RL)", "#E74C3C")]:
        if len(data) > 3:
            kde = gaussian_kde(data)
            x   = np.linspace(0, 1, 200)
            ax2.plot(x, kde(x), lw=2.5, color=color, label=label)
            ax2.fill_between(x, kde(x), alpha=0.15, color=color)
    ax2.set_xlabel("QED (Drug-likeness)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("QED Distribution: Training vs Generated", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Chemical space (PCA of Morgan FP) ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    n_train = min(150, len(train_smiles))
    n_gen   = min(150, len(valid_gen))
    all_smi = train_smiles[:n_train] + valid_gen[:n_gen]
    labels  = ["Training"] * n_train + ["Generated (RL)"] * n_gen
    X_fp    = morgan_matrix(all_smi)
    pca     = PCA(n_components=2, random_state=SEED)
    X_pca   = pca.fit_transform(X_fp)

    for group, col, mk in [("Training", "#3498DB", "o"), ("Generated (RL)", "#E74C3C", "^")]:
        idx = [i for i, l in enumerate(labels) if l == group]
        ax3.scatter(X_pca[idx, 0], X_pca[idx, 1], c=col, marker=mk,
                    s=20, alpha=0.6, label=group, edgecolors="none")
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
    ax3.set_title("Chemical Space (PCA of Morgan FP)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9, markerscale=2)
    ax3.grid(True, alpha=0.2)

    # ── Panel 4: Property violin comparison ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    prop = "MW"
    data_to_plot = [
        train_props[prop].dropna().values,
        gen_props[prop].dropna().values,
    ]
    vp = ax4.violinplot(data_to_plot, positions=[1, 2], showmedians=True, showextrema=True)
    for pc, col in zip(vp["bodies"], ["#3498DB", "#E74C3C"]):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels(["Training", "Generated"], fontsize=10)
    ax4.set_ylabel("Molecular Weight (Da)", fontsize=10)
    ax4.set_title("MW Distribution Comparison", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    # ── Panel 5: Reward component breakdown (top generated) ───────────────
    ax5 = fig.add_subplot(gs[1, 1])
    top_gen = sorted(
        [(s, reward_fn(s)) for s in valid_gen[:100] if Chem.MolFromSmiles(s)],
        key=lambda x: x[1]["total"], reverse=True
    )[:10]
    if top_gen:
        labels5   = [f"Gen-{i+1}" for i in range(len(top_gen))]
        qed_vals  = [r["qed"]   for _, r in top_gen]
        sa_vals   = [r["sa"]    for _, r in top_gen]
        sim_vals  = [r["sim"]   for _, r in top_gen]
        x5 = np.arange(len(labels5))
        w  = 0.25
        ax5.bar(x5 - w, qed_vals,  w, label="QED",        color="#3498DB")
        ax5.bar(x5,     sa_vals,   w, label="SA (scaled)", color="#2ECC71")
        ax5.bar(x5 + w, sim_vals,  w, label="Target sim.", color="#E74C3C")
        ax5.set_xticks(x5)
        ax5.set_xticklabels(labels5, rotation=45, ha="right", fontsize=8)
        ax5.set_ylabel("Score [0,1]", fontsize=10)
        ax5.set_title("Reward Components — Top-10 Generated", fontsize=11, fontweight="bold")
        ax5.legend(fontsize=8)
        ax5.set_ylim(0, 1.1)
        ax5.grid(True, alpha=0.3, axis="y")

    # ── Panel 6: Generation metrics summary ───────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    metrics = compute_generation_metrics(generated_smiles, train_smiles)
    scaf    = scaffold_novelty(valid_gen, CDK46_ACTIVES)
    lines = [
        ("Validity",          f"{metrics['validity']*100:.1f}%"),
        ("Uniqueness",        f"{metrics['uniqueness']*100:.1f}%"),
        ("Novelty",           f"{metrics['novelty']*100:.1f}%"),
        ("Diversity (1−Tan)", f"{metrics['diversity']:.3f}"),
        ("New scaffolds",     f"{scaf['new_scaffolds']}"),
        ("Scaffold novelty",  f"{scaf['scaffold_novelty']*100:.1f}%"),
        ("Mean QED (gen)",    f"{gen_props['QED'].mean():.3f}"),
        ("Mean QED (train)",  f"{train_props['QED'].mean():.3f}"),
    ]
    y_pos = 0.92
    ax6.text(0.5, 1.0, "Generation Metrics", ha="center", va="top",
             fontsize=12, fontweight="bold", transform=ax6.transAxes)
    for label, val in lines:
        ax6.text(0.1, y_pos, label + ":", fontsize=10, transform=ax6.transAxes)
        ax6.text(0.75, y_pos, val, fontsize=10, fontweight="bold",
                 color="#2C3E50", transform=ax6.transAxes, ha="right")
        y_pos -= 0.10

    fig.suptitle(
        "De Novo CDK4/6 Inhibitor Generation — SMILES Transformer + REINFORCE",
        fontsize=14, fontweight="bold"
    )
    plt.savefig(out_dir / "generation_results.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir}/generation_results.png")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # 1. Build corpus
    print("\n[1/7] Building training corpus...")
    corpus = generate_corpus(n=3000)
    print(f"  Corpus size: {len(corpus)} unique SMILES")

    # 2. Tokenizer
    print("[2/7] Building SMILES tokenizer...")
    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(corpus + CDK46_ACTIVES)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))

    # 3. Pre-train Transformer
    print("[3/7] Pre-training SMILES Transformer...")
    model = SmilesTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=128,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    pretrain_losses = pretrain_on_smiles(
        model, tokenizer, corpus,
        epochs=8, batch_size=64, lr=3e-4, device=DEVICE
    )

    # Quick validity check after pre-training
    pre_samples = model.generate(
        bos_token=tokenizer.bos_id,
        max_len=80, temperature=1.0, n_samples=100,
        tokenizer=tokenizer
    )
    pre_metrics = compute_generation_metrics(pre_samples, corpus)
    print(f"  Pre-train validity: {pre_metrics['validity']*100:.1f}%")

    # 4. Fine-tune on CDK4/6 actives
    print("\n[4/7] Fine-tuning on CDK4/6 reference actives...")
    ft_losses = pretrain_on_smiles(
        model, tokenizer, CDK46_ACTIVES,
        epochs=10, batch_size=16, lr=5e-5, device=DEVICE
    )

    # 5. RL optimisation
    print("\n[5/7] REINFORCE optimisation (200 steps)...")
    reward_fn = MolecularReward(
        reference_actives=CDK46_ACTIVES,
        anti_targets=CDK2_ANTITARGETS,
        w_qed=0.35, w_sa=0.15, w_sim=0.25, w_sel=0.15, w_novel=0.10,
    )
    rl_opt = RLOptimizer(
        model=model, tokenizer=tokenizer, reward_fn=reward_fn,
        lr=5e-5, baseline=0.40
    )
    rl_history = []
    for step in range(1, 201):
        metrics = rl_opt.step(n_samples=32, temperature=1.0)
        metrics["step"] = step
        rl_history.append(metrics)
        if step % 50 == 0:
            print(f"  Step {step:3d} | reward={metrics['mean_reward']:.3f} "
                  f"valid={metrics['valid_frac']*100:.0f}% "
                  f"novel={metrics['novel_frac']*100:.0f}% "
                  f"QED={metrics['mean_qed']:.3f}")

    # 6. Generate final molecules
    print("\n[6/7] Generating 500 molecules from optimised policy...")
    generated = model.generate(
        bos_token=tokenizer.bos_id,
        max_len=100, temperature=0.9, top_p=0.95,
        n_samples=500, tokenizer=tokenizer
    )
    final_metrics = compute_generation_metrics(generated, corpus)
    scaf_info     = scaffold_novelty(
        [s for s in generated if Chem.MolFromSmiles(s)], CDK46_ACTIVES
    )
    print(f"  Validity:   {final_metrics['validity']*100:.1f}%")
    print(f"  Uniqueness: {final_metrics['uniqueness']*100:.1f}%")
    print(f"  Novelty:    {final_metrics['novelty']*100:.1f}%")
    print(f"  New scaffolds: {scaf_info['new_scaffolds']}")

    # 7. Visualise
    print("\n[7/7] Generating result figures...")
    plot_results(corpus[:200], generated, rl_history, reward_fn, OUT_DIR)

    # Save top-10 candidates
    valid_gen = [s for s in generated if Chem.MolFromSmiles(s)]
    scored    = sorted(
        [(s, reward_fn(s)["total"]) for s in valid_gen],
        key=lambda x: x[1], reverse=True
    )[:20]
    top_df = pd.DataFrame(
        [{"smiles": s, "reward": r} for s, r in scored]
    )
    top_df.to_csv(OUT_DIR / "top_generated_molecules.csv", index=False)
    print("Saved results/top_generated_molecules.csv")
    print("\nDone.")
