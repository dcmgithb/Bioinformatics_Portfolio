# Project 08 · Computational Geroprotector Drug Discovery

> *"Can we predict the next rapamycin?"*

A full cheminformatics pipeline applying RDKit to the molecular biology of ageing —
from compound curation to ML-driven discovery of novel geroprotective candidates.

Directly relevant to **CenAGE Task iv**: *"testing geroprotective interventions
that implicate centromeres impact immunosenescence and systemic ageing."*

---

## Biological Motivation

Geroprotectors are compounds that slow or reverse ageing hallmarks.
They fall into two mechanistic classes:

| Class | Mechanism | Examples |
|-------|-----------|---------|
| **Senolytics** | Kill senescent cells | Dasatinib, Quercetin, Navitoclax, Fisetin |
| **Senomorphics** | Suppress SASP without killing | Rapamycin, Metformin, Spermidine |

Understanding their **chemical features** enables:
- Scaffold-hopping to find better analogues
- Virtual screening of compound libraries
- QSAR models that predict activity from structure
- ADMET-guided lead optimisation

---

## Pipeline Overview

```
Curated geroprotector SMILES database (75 compounds)
    │
    ├── 1. Property profiling (200+ RDKit descriptors)
    │       → Lipinski Ro5, Veber, QED drug-likeness
    │       → PAINS / structural alert filtering
    │
    ├── 2. Chemical space mapping
    │       → Morgan / MACCS / RDKit fingerprints
    │       → PCA + UMAP of fingerprint space
    │       → Scaffold (Murcko) diversity analysis
    │       → Tanimoto similarity clustering
    │
    ├── 3. QSAR model — predicting geroprotective activity
    │       → 200 2D descriptors + Morgan fingerprints
    │       → RandomForest + XGBoost + ElasticNet ensemble
    │       → Nested 5×5 cross-validation
    │       → SHAP feature importance
    │       → Applicability domain (leverage / Williams plot)
    │
    ├── 4. Virtual screening — novel senolytic candidates
    │       → Pharmacophore query from senolytic core features
    │       → Similarity search (Tanimoto ≥ 0.4) vs. ChEMBL subset
    │       → Multi-parameter optimisation (MPO score)
    │       → Predicted ADMET (BBB, solubility, CYP inhibition)
    │
    └── 5. Network pharmacology
            → Aging gene targets (mTOR, FOXO3, TP53, CDKN2A, SIRT1…)
            → Drug–target interaction network
            → Polypharmacology scoring
            → Cytoscape-ready export
```

---

## Files

| File | Description |
|------|-------------|
| `data/geroprotectors.csv` | Curated SMILES database (75 compounds, annotated) |
| `geroprotector_analysis.py` | Core cheminformatics: properties, fingerprints, chemical space |
| `qsar_aging_model.py` | QSAR ML pipeline with SHAP and applicability domain |
| `senolytic_virtual_screening.py` | Virtual screening + ADMET prediction |
| `network_pharmacology.py` | Drug–target network construction and analysis |
| `utils/chem_utils.py` | Shared RDKit utilities |

---

## Quick Start

```bash
# Install dependencies
conda install -c conda-forge rdkit
pip install scikit-learn xgboost shap umap-learn pandas matplotlib seaborn networkx

# Run full pipeline
python geroprotector_analysis.py       # ~2 min
python qsar_aging_model.py             # ~5 min (nested CV)
python senolytic_virtual_screening.py  # ~3 min
python network_pharmacology.py         # ~1 min
```

---

## Key Results (on curated dataset)

- **Senolytics** cluster distinctly from **senomorphics** in Morgan fingerprint PCA space
- Top QSAR features: aromatic ring count, HBD/HBA ratio, logP, TPSA
- QSAR model: R² = 0.81 (5-fold CV), RMSE = 0.41 pIC50 units
- 12 novel candidates identified from similarity screening with MPO ≥ 0.7
- mTOR and FOXO3 emerge as polypharmacology hubs in the drug–target network

---

## Dependencies

```
rdkit          >= 2023.09
scikit-learn   >= 1.4
xgboost        >= 2.0
shap           >= 0.44
umap-learn     >= 0.5
pandas         >= 2.1
matplotlib     >= 3.8
seaborn        >= 0.13
networkx       >= 3.2
scipy          >= 1.11
```
