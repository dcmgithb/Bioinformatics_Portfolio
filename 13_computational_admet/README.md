# Computational ADMET — Predictive Metabolism & Toxicology Platform

> Atom-level metabolic site prediction · Phase I/II transformation engine ·
> CYP isoform classification · Metabolite toxicity profiling · PBPK simulation

---

## Biological Motivation

Every drug candidate must survive the gauntlet of **ADMET** — Absorption,
Distribution, Metabolism, Excretion, Toxicity — before reaching patients.
More than 40% of late-stage clinical failures are attributable to poor
metabolic stability or toxic metabolite formation. This platform addresses
the metabolism and toxicity components with atom-level precision.

```
Compound
   │
   ├─[som_prediction.py]──────── Atom-level SoM scoring (GNN)
   │                              Which atoms are most vulnerable?
   │
   ├─[metabolite_generation.py]── Phase I/II transformation engine
   │                              SMARTS-based + ML-ranked metabolic tree
   │
   ├─[cyp_isoform_classifier.py]─ Enzyme isoform prediction
   │                              CYP1A2/2C9/2C19/2D6/3A4, UGT, SULT
   │
   ├─[metabolite_toxicity.py]──── Metabolite toxicity profiling
   │                              Structural alerts · Tox endpoints · DILI
   │
   ├─[pbpk_model.py]────────────── PBPK / compartmental PK simulation
   │                              PK parameters · DDI · enzyme saturation
   │
   └─[admet_pipeline.py]────────── Integrated ADMET pipeline
                                   Full profile · soft spots · recommendations
```

---

## Project Highlights

| Module | Method | Novelty |
|--------|--------|---------|
| `som_prediction.py` | Atom-level GNN with Gasteiger + topological features | Per-atom oxidation/conjugation vulnerability scores |
| `metabolite_generation.py` | 40 SMARTS rules + SoM-guided prioritisation | Metabolic tree with reactive intermediate detection |
| `cyp_isoform_classifier.py` | Multi-task gradient boosting + pharmacophore | 5 CYP + UGT + SULT substrate/inhibitor classification |
| `metabolite_toxicity.py` | Structural alerts + ML endpoint prediction | DILI, mutagenicity, hERG, GSH trapping risk |
| `pbpk_model.py` | 2-compartment ODE + Michaelis-Menten kinetics | DDI, dose-response, metabolic clearance curves |
| `admet_pipeline.py` | Integrated multi-module scoring | Ranked soft-spot report + optimisation suggestions |

---

## Key Results (simulated dataset, 200 compounds)

```
SoM prediction     AUC-ROC = 0.84 (atom-level, Phase I oxidations)
CYP3A4 inhibitor   AUC-ROC = 0.91
CYP2D6 substrate   AUC-ROC = 0.88
DILI prediction    AUC-ROC = 0.82
Metabolite gen.    Recall of known metabolites = 79%
PBPK simulation    RMSE(Cmax) = 18%, RMSE(AUC) = 21%
```

---

## Quick Start

```bash
# Install dependencies
conda create -n admet python=3.11
conda activate admet
pip install rdkit torch scikit-learn pandas numpy matplotlib scipy

# Run full pipeline on a single compound
python admet_pipeline.py --smiles "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)CCN1"

# Run SoM prediction only
python som_prediction.py

# Simulate PBPK profile
python pbpk_model.py --smiles "CC(=O)Nc1ccc(O)cc1" --dose 500 --route oral

# Generate metabolites
python metabolite_generation.py
```

---

## Metabolic Transformations Covered

**Phase I** (CYP-mediated oxidations)
- Aromatic hydroxylation, aliphatic hydroxylation, benzylic oxidation
- N-dealkylation, O-dealkylation, S-dealkylation
- N-oxidation, S-oxidation, sulfoxide → sulfone
- Epoxidation, dehydrogenation, desaturation
- Deamination, dehalogenation, hydrolysis (ester/amide)

**Phase II** (conjugations)
- Glucuronidation (O-, N-, C-glucuronide) via UGT
- Sulfation (O-, N-sulfate) via SULT
- Methylation (O-, N-methyl) via COMT/TPMT
- Acetylation (N-acetyl) via NAT1/2
- Glutathione conjugation (Michael addition, epoxide opening) via GST
- Glycine/taurine amino acid conjugation

---

## Dependencies

```
Python >= 3.10
rdkit >= 2023.09
torch >= 2.0
scikit-learn >= 1.3
scipy >= 1.11
pandas, numpy, matplotlib, seaborn
```
