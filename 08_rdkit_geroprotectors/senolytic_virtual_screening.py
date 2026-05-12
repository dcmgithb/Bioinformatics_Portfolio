"""
Virtual Screening for Novel Senolytic Candidates
=================================================
Identifies novel putative senolytic compounds by:
  1. Deriving a senolytic pharmacophore query from known senolytics
  2. Similarity-based screening (Tanimoto ≥ 0.4) against a mock library
  3. Multi-Parameter Optimisation (MPO) filtering
  4. Predicted ADMET profiling (rule-based + ML proxy)
  5. Scaffold hop analysis: finding novel scaffolds with senolytic features
  6. Ranking candidates by composite score

In a real pipeline: replace mock_library with ChEMBL / ZINC / Enamine
and add AutoDock/Glide docking for pose-based scoring.

Python : >= 3.10 | RDKit >= 2023.09
"""

from __future__ import annotations

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import DataStructs

from utils.chem_utils import (
    validate_smiles_column, enrich_dataframe, fingerprint_matrix,
    tanimoto_matrix, top_k_similar, morgan_fp, maccs_fp,
    mpo_score, sa_score, lipinski_pass, veber_pass, is_pains,
    calc_lipinski, get_scaffold
)

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── 1. Load Known Senolytics (Query Set) ──────────────────────────────────────

print("═" * 60)
print("  VIRTUAL SCREENING — NOVEL SENOLYTIC DISCOVERY")
print("═" * 60)

df_known = pd.read_csv("data/geroprotectors.csv")
df_known = validate_smiles_column(df_known, smiles_col="smiles")
senolytics = df_known[df_known["class"] == "senolytic"].copy()
senomorphics = df_known[df_known["class"] == "senomorphic"].copy()

print(f"\nQuery senolytics: {len(senolytics)}")
print(f"Reference senomorphics: {len(senomorphics)}")

# ── 2. Build Screening Library (Mock ChEMBL subset) ───────────────────────────

print("\n[2] Generating virtual screening library ...")

# Representative diverse compounds mimicking a ChEMBL drug-like subset
# In production: download from ChEMBL, ZINC, or Enamine REAL
LIBRARY_SMILES = [
    # Flavonoid-like scaffolds
    ("VL001", "O=c1cc(-c2ccc(F)c(F)c2)oc2cc(O)cc(O)c12"),
    ("VL002", "O=c1cc(-c2ccc(Cl)cc2)oc2cc(O)c(OCC)c(O)c12"),
    ("VL003", "O=c1c(O)c(-c2ccc(Br)cc2)oc2cc(O)cc(OC)c12"),
    ("VL004", "O=c1cc(-c2ccc(NC(=O)C)cc2)oc2cc(O)cc(O)c12"),
    ("VL005", "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2O)ccc1OCC(F)(F)F"),
    ("VL006", "O=c1c(O)c(-c2ccc(O)c(F)c2)oc2cc(O)cc(O)c12"),
    ("VL007", "O=c1cc(-c2ccc(O)c(O)c2F)oc2cc(O)cc(O)c12"),
    # Tyrosine kinase inhibitor scaffolds
    ("VL008", "Cc1cc(Nc2ncc(F)c(NC(=O)c3cc(Cl)ccc3)n2)ccc1"),
    ("VL009", "Cc1nc(Nc2ncc(s2)C(=O)Nc2c(Cl)cccc2C)cc(n1)N1CCN(CC)CC1"),
    ("VL010", "Cc1nc(Nc2ncc(s2)C(=O)Nc2ccccc2F)cc(n1)N1CCNCC1"),
    ("VL011", "Cc1nc(-c2cnn(C)c2)c2cc(NC(=O)c3ccc(F)cc3)ccc2n1"),
    ("VL012", "O=C(Nc1cc2c(s1)CCCC2)c1cnc2ccccc2c1"),
    # BCL-2 inhibitor scaffolds
    ("VL013", "O=C(Nc1ccc(F)cc1)c1cnc(-c2ccc(Cl)cc2)s1"),
    ("VL014", "O=C(Nc1ccc(Cl)cc1F)c1cn(-c2ccccc2)nc1-c1ccc(F)cc1"),
    ("VL015", "O=C(Nc1ccc(-c2ccccc2)cc1)c1cnc(-c2ccncc2)o1"),
    ("VL016", "CN(C(=O)c1ccc(-c2ccc(Cl)cc2)cc1)c1ccc(F)cc1"),
    # CDK inhibitor scaffolds
    ("VL017", "Cc1cn2c(n1)N=C(NC1CC1)C=C2"),
    ("VL018", "c1cc2c(nc1)c(-c1ccncc1)cn2"),
    ("VL019", "Cc1cc2cc(NC(=O)Cc3cc4ccccc4o3)ccc2c(=O)n1"),
    ("VL020", "Cc1ccc(-n2cc(-c3ccc(Cl)cc3)c(C(=O)O)c2)cc1"),
    # PI3K/mTOR dual inhibitors
    ("VL021", "Cc1cc2c(cc1N(CC)CC)OC(=O)C=C2"),
    ("VL022", "COc1cc2ncnc(N3CCOCC3)c2cc1OC"),
    ("VL023", "Cc1ccc(-c2nc3ccc(N4CCNCC4)cc3[nH]2)cc1"),
    ("VL024", "O=C(c1cccc(F)c1)Nc1ccc(cn1)-c1cnc2ccccc2n1"),
    # PARP inhibitor scaffolds
    ("VL025", "O=C1CCN(C(=O)c2ccc3cnccc3c2)C1"),
    ("VL026", "O=C1CCN(C(=O)c2cc3cc(F)ccc3[nH]2)C1"),
    ("VL027", "O=c1[nH]c(=O)c2ccc(F)cc2n1CC1CCN(CC1)c1ccc(F)cc1"),
    ("VL028", "Nc1ccc2c(c1)C(=O)N(CC1CCOCC1)CC2=O"),
    # Senomorphic-like (mTOR vicinity)
    ("VL029", "CCOc1cc2c(nc(NC3CC3)n2)cc1OC"),
    ("VL030", "O=C(Nc1ncc(F)c(Nc2ccc(OCC(F)(F)F)cc2)n1)c1ccc(Cl)cc1"),
    ("VL031", "Cc1ccc(Nc2nc3ccc(F)cn3c(=O)n2)cc1"),
    ("VL032", "CN(Cc1ccccc1)c1nc2ccc(F)cc2c(=O)n1CC(F)(F)F"),
    # Novel natural-product-like
    ("VL033", "OC1CC(c2ccc(O)c(OC)c2)Oc2cc(O)ccc21"),
    ("VL034", "COc1ccc(C2CC(=O)c3c(O)cc(O)cc3O2)cc1O"),
    ("VL035", "Oc1ccc(/C=C/c2cc(O)ccc2O)cc1"),
    ("VL036", "OC[C@H]1O[C@@H](Oc2cc(O)cc3oc(-c4ccc(O)c(O)c4)cc(O)c23)[C@H](O)[C@@H](O)[C@@H]1O"),
    # Diverse drug fragments
    ("VL037", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)Nc1cccc(n1)-c1ccc(F)cc1"),
    ("VL038", "O=C(Nc1ccc2[nH]cc(-c3ccc(Cl)cc3)c2c1)c1ccc(F)cc1"),
    ("VL039", "CC(=O)Nc1ccc(-c2nc3ccccc3c(=O)n2C)cc1"),
    ("VL040", "Cc1ccc(NC(=O)c2ccc(-c3ccc(Cl)cc3)cc2)cc1F"),
    ("VL041", "CC(=O)Nc1ccc(Nc2ncc(C(F)(F)F)cc2=O)cc1"),
    ("VL042", "Cc1ccc(C(=O)Nc2cccc(n2)-c2ccccc2F)cc1"),
    ("VL043", "O=C(c1ccccc1F)Nc1ccc(-c2ccc3[nH]cnc3c2)cc1"),
    ("VL044", "CC(NC(=O)c1cccc(c1)-c1ccc(F)cc1)c1cccc(F)c1"),
    # Polyphenol derivatives
    ("VL045", "COc1cc(OC)c(CC=C)cc1O"),
    ("VL046", "COc1cc(/C=C/C(=O)O)ccc1O"),
    ("VL047", "COc1ccc(-c2oc3cc(OC)cc(OC)c3c(=O)c2OC)cc1OC"),
    ("VL048", "OC(Cc1ccc(O)c(O)c1)C(=O)O"),
    # Senolytic-adjacent miscellaneous
    ("VL049", "O=C(c1ccc(Cl)cc1)N1CCN(c2ccc(F)cc2)CC1"),
    ("VL050", "Cc1ccc(cc1)-c1nc(-c2ccc(F)cc2)c[nH]1"),
]

lib_df = pd.DataFrame(LIBRARY_SMILES, columns=["id", "smiles"])
lib_df  = validate_smiles_column(lib_df, smiles_col="smiles")
lib_mols = [Chem.MolFromSmiles(s) for s in lib_df["smiles"]]
print(f"  Screening library: {len(lib_df)} compounds")

# ── 3. Fingerprint-Based Similarity Screening ─────────────────────────────────

print("\n[3] Fingerprint similarity screening ...")

# Compute Morgan FP for all query senolytics and library
query_fps = fingerprint_matrix(
    [Chem.MolFromSmiles(s) for s in senolytics["smiles"]],
    fp_type="morgan", radius=2, n_bits=2048
)
lib_fps = fingerprint_matrix(lib_mols, fp_type="morgan", radius=2, n_bits=2048)

# Max Tanimoto to any known senolytic
def max_tanimoto_to_class(lib_fp: np.ndarray, query_fps: np.ndarray) -> float:
    """Maximum Tanimoto similarity from one compound to a set of queries."""
    q = query_fps.astype(np.float32)
    l = lib_fp.astype(np.float32)
    dot   = q @ l
    norms = q.sum(axis=1) + l.sum() - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(norms > 0, dot / norms, 0.0)
    return float(np.max(sims))

senomorphic_fps = fingerprint_matrix(
    [Chem.MolFromSmiles(s) for s in senomorphics["smiles"]],
    fp_type="morgan", radius=2, n_bits=2048
)
lib_df["tanimoto_senolytic"]   = [max_tanimoto_to_class(fp, query_fps)      for fp in lib_fps]
lib_df["tanimoto_senomorphic"] = [max_tanimoto_to_class(fp, senomorphic_fps) for fp in lib_fps]
lib_df["selectivity_delta"] = lib_df["tanimoto_senolytic"] - lib_df["tanimoto_senomorphic"]

# ── 4. Physicochemical Properties & Filters ───────────────────────────────────

print("\n[4] Computing physicochemical properties & filters ...")

lib_df = enrich_dataframe(lib_df, smiles_col="smiles")
lib_mols_valid = [Chem.MolFromSmiles(s) for s in lib_df["smiles"]]

# ── 5. ADMET Prediction (Rule-Based + ML Proxy) ───────────────────────────────

print("\n[5] ADMET profiling ...")

def predict_admet(mol: Chem.Mol) -> dict:
    """
    Rule-based ADMET prediction using physicochemical thresholds.
    Loosely based on Lipinski, Veber, and published ADMET models.
    """
    props = calc_lipinski(mol)
    mw    = props["MW"]
    logp  = props["LogP"]
    tpsa  = props["TPSA"]
    hbd   = props["HBD"]
    hba   = props["HBA"]
    rb    = props["RotatableBonds"]

    # BBB permeability (CNS MPO-inspired)
    bbb_ok = (logp > 0) and (logp < 5) and (tpsa < 90) and (hbd <= 3) and (mw < 450)

    # GI absorption (Egan rule)
    gi_ok = (tpsa <= 131.6) and (logp <= 5.88)

    # Aqueous solubility (LogS estimate, Delaney-like)
    log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * hba - 0.74 * int(bbb_ok)
    sol_class = "Poor" if log_s < -4 else ("Moderate" if log_s < -2 else "Good")

    # CYP3A4 inhibition (logP + TPSA proxy)
    cyp3a4_inh = (logp > 3.5) and (tpsa < 100) and (mw > 300)

    # hERG liability (cationic + high logP)
    n_atoms = mol.GetNumHeavyAtoms()
    herg_flag = (logp > 3.7) and (n_atoms > 20) and any(
        a.GetFormalCharge() > 0 for a in mol.GetAtoms()
    )

    # Microsomal stability proxy (high Fsp3 = more stable)
    fsp3   = rdMolDescriptors.CalcFractionCSP3(mol)
    stable = fsp3 > 0.25

    # Overall drug score
    drug_score = sum([
        1.0 if gi_ok else 0.0,
        0.5 if bbb_ok else 0.0,
        1.0 if sol_class in ("Moderate","Good") else 0.0,
        -1.0 if cyp3a4_inh else 0.0,
        -0.5 if herg_flag else 0.0,
        0.5 if stable else 0.0,
    ])

    return {
        "bbb_permeable"    : bbb_ok,
        "gi_absorbed"      : gi_ok,
        "solubility"       : sol_class,
        "log_s_est"        : round(log_s, 2),
        "cyp3a4_inhibitor" : cyp3a4_inh,
        "herg_risk"        : herg_flag,
        "microsomal_stable": stable,
        "admet_score"      : round(drug_score, 2),
    }

admet_rows = [predict_admet(m) for m in lib_mols_valid]
admet_df   = pd.DataFrame(admet_rows, index=lib_df.index)
lib_df     = pd.concat([lib_df, admet_df], axis=1)

# ── 6. Composite Screening Score & Ranking ────────────────────────────────────

print("\n[6] Computing composite score and ranking ...")

# Composite score: weights each desirable property
lib_df["composite_score"] = (
    0.35 * lib_df["tanimoto_senolytic"]                 # similarity to senolytics
  + 0.15 * lib_df["selectivity_delta"].clip(-1, 1)      # senolytic vs. senomorphic selectivity
  + 0.20 * lib_df["mpo_score"]                          # lead-likeness
  + 0.10 * lib_df["QED"]                                # drug-likeness
  + 0.10 * lib_df["admet_score"].clip(0, 2) / 2         # ADMET
  - 0.10 * lib_df["pains"].astype(float)                # penalise PAINS
  + 0.10 * (1 - lib_df["sa_score"] / 10)                # synthetic accessibility
)

lib_df_sorted = lib_df.sort_values("composite_score", ascending=False)
lib_df_sorted.to_csv(RESULTS_DIR / "virtual_screening_hits.csv", index=False)

top20 = lib_df_sorted.head(20)
print(f"\n  Top 20 candidates written → virtual_screening_hits.csv")
print(f"  Best candidate: {top20.iloc[0]['id']} "
      f"(composite={top20.iloc[0]['composite_score']:.3f}, "
      f"Tanimoto={top20.iloc[0]['tanimoto_senolytic']:.3f})")

# ── 7. Scaffold Hop Analysis ──────────────────────────────────────────────────

print("\n[7] Scaffold hop analysis ...")

# Known senolytic scaffolds
known_scaffolds = set(senolytics["smiles"].apply(
    lambda s: get_scaffold(Chem.MolFromSmiles(s)) or ""
))

lib_df["is_novel_scaffold"] = lib_df["scaffold"].apply(
    lambda s: s not in known_scaffolds if pd.notna(s) else False
)

# Top novel-scaffold hits
novel_hits = lib_df_sorted[
    lib_df_sorted["is_novel_scaffold"] &
    (lib_df_sorted["tanimoto_senolytic"] >= 0.35) &
    lib_df_sorted["drug_like"]
]
novel_hits.to_csv(RESULTS_DIR / "scaffold_hop_candidates.csv", index=False)
print(f"  Novel scaffold candidates (Tanimoto≥0.35, drug-like): {len(novel_hits)}")

# ── 8. Visualisation ─────────────────────────────────────────────────────────

print("\n[8] Generating plots ...")

# 8a. Tanimoto distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(lib_df["tanimoto_senolytic"], bins=20, color="#F44336", alpha=0.7,
        label="vs. Senolytics", edgecolor="white")
ax.hist(lib_df["tanimoto_senomorphic"], bins=20, color="#2196F3", alpha=0.7,
        label="vs. Senomorphics", edgecolor="white")
ax.axvline(0.4, color="black", lw=1.5, ls="--", label="Hit threshold (0.4)")
ax.set_xlabel("Max Tanimoto Similarity", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Library Similarity to Known Geroprotectors\n(Morgan FP, radius=2)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 8b. Composite score distribution, coloured by scaffold novelty
ax = axes[1]
novel_mask = lib_df_sorted["is_novel_scaffold"]
ax.scatter(lib_df_sorted.loc[~novel_mask, "tanimoto_senolytic"],
           lib_df_sorted.loc[~novel_mask, "composite_score"],
           c="#607D8B", s=50, alpha=0.6, label="Known scaffold", edgecolors="white", lw=0.3)
ax.scatter(lib_df_sorted.loc[novel_mask, "tanimoto_senolytic"],
           lib_df_sorted.loc[novel_mask, "composite_score"],
           c="#FF9800", s=70, alpha=0.85, label="Novel scaffold", edgecolors="black", lw=0.5)
# Annotate top 5 novel
for _, row in novel_hits.head(5).iterrows():
    ax.annotate(row["id"],
                xy=(row["tanimoto_senolytic"], row["composite_score"]),
                xytext=(5, 3), textcoords="offset points", fontsize=7, fontweight="bold")
ax.axvline(0.4, color="black", lw=1, ls="--", alpha=0.5)
ax.set_xlabel("Tanimoto to Known Senolytics", fontsize=11)
ax.set_ylabel("Composite Score", fontsize=11)
ax.set_title("Virtual Screening Landscape\nOrange = novel scaffold",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "12_virtual_screening.pdf", bbox_inches="tight", dpi=150)
plt.close()

# 8c. ADMET spider chart for top 10 hits
top10 = lib_df_sorted.head(10)
admet_props = ["gi_absorbed","bbb_permeable","microsomal_stable"]
admet_scores_num = top10[admet_props].astype(int).values
admet_neg = ["cyp3a4_inhibitor","herg_risk","pains"]
admet_neg_num = top10[admet_neg].astype(int).values

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(top10))
w = 0.2
ax.bar(x - w, admet_scores_num[:, 0], w, label="GI absorbed",        color="#4CAF50", alpha=0.8)
ax.bar(x,     admet_scores_num[:, 1], w, label="BBB permeable",      color="#2196F3", alpha=0.8)
ax.bar(x + w, admet_scores_num[:, 2], w, label="Microsomal stable",  color="#9C27B0", alpha=0.8)
ax.bar(x + 2*w, admet_neg_num[:, 0], w, label="CYP3A4 inhibitor",   color="#FF9800", alpha=0.8)
ax.bar(x + 3*w, admet_neg_num[:, 1], w, label="hERG risk",           color="#F44336", alpha=0.8)
ax.set_xticks(x + w)
ax.set_xticklabels(top10["id"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Property (0=No, 1=Yes)", fontsize=10)
ax.set_title("ADMET Profile of Top 10 Virtual Screening Hits\n"
             "Green/Blue/Purple = desirable | Orange/Red = flag",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=3)
ax.set_ylim(-0.1, 1.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "13_admet_top10.pdf", bbox_inches="tight", dpi=150)
plt.close()

# 8d. Candidate summary table plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")
table_data = top10[["id","tanimoto_senolytic","composite_score",
                      "MW","LogP","QED","mpo_score",
                      "gi_absorbed","herg_risk","pains","is_novel_scaffold"]].copy()
table_data = table_data.round(3)
col_labels = ["ID","Tanimoto","Composite","MW","LogP","QED","MPO",
               "GI abs.","hERG","PAINS","Novel scaffold"]
table = ax.table(
    cellText   = table_data.values,
    colLabels  = col_labels,
    loc        = "center",
    cellLoc    = "center",
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.8)
ax.set_title("Top 10 Virtual Screening Hits — Summary Table",
             fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "14_screening_table.pdf", bbox_inches="tight", dpi=150)
plt.close()

print(f"\n{'='*60}")
print(f"  Virtual screening complete!")
print(f"  Library screened   : {len(lib_df)} compounds")
print(f"  Hits (Tan≥0.35)    : {(lib_df['tanimoto_senolytic']>=0.35).sum()}")
print(f"  Drug-like hits     : {((lib_df['tanimoto_senolytic']>=0.35)&lib_df['drug_like']).sum()}")
print(f"  Novel scaffolds    : {len(novel_hits)}")
print(f"  Results : {RESULTS_DIR} | Figures : {FIGURES_DIR}")
print(f"{'='*60}")
