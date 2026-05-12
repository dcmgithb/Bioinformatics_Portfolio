# Project 03 · scRNA-seq Immunosenescence

## Biological Question
How does the immune cell landscape change with age, and which specific
cell populations accumulate senescent features (p21+, p16+, SASP)?

## Dataset
- **GSE174072** — 10X Genomics scRNA-seq, PBMCs, young (25–35 yr) vs. aged (70–85 yr), n=8 donors/group
- **GSE155006** — Human blood immune cells across the lifespan (n=101 donors)

## Single-Cell Pipeline

```
Raw 10X output (CellRanger) → Ambient RNA (SoupX) → Doublet detection (DoubletFinder/Scrublet)
    → QC filters (nFeature, % mt, log10GenesPerUMI)
    → Normalisation (scran/log-normalize) → HVG selection
    → Dimensionality reduction (PCA → UMAP)
    → Graph-based clustering (Leiden/Louvain)
    → Cell type annotation (reference-based + marker genes)
    → Differential abundance (Milo / propeller)
    → Pseudobulk DE (DESeq2 per cell type)
    → Senescence score (UCell + custom SASP signature)
    → Pseudotime (Monocle3 / scVelo)
```

## Files
| File | Description |
|------|-------------|
| `seurat_immune_aging.R` | Full Seurat v5 pipeline (R) |
| `scanpy_immune_aging.py` | Full Scanpy pipeline (Python) |
| `milo_differential_abundance.R` | Milo neighbourhood analysis |

## Key Results
- UMAP of ~40k cells, 15 annotated cell types
- Age-associated shifts in T cell effector memory vs. naive ratio
- CD8+ TEMRA expansion with age
- Senescence score highest in NK cells and monocytes of aged donors
- Pseudotime reveals NK cell maturation arrest in ageing

## How to Run
```bash
# R pipeline
Rscript seurat_immune_aging.R

# Python pipeline
python scanpy_immune_aging.py

# Differential abundance
Rscript milo_differential_abundance.R

```
