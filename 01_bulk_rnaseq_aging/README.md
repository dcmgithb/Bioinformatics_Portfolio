# Project 01 · Bulk RNA-seq Aging Analysis

## Biological Question
How does physiological ageing reshape the transcriptome of immune tissues,
and which pathways show conserved dysregulation across human PBMCs and mouse spleen?

## Dataset
- **Human**: GSE65907 — PBMC RNA-seq, young (20–35 yr) vs. aged (65–85 yr) donors (n=20/group)
- **Mouse**: GSE132901 — Splenic CD8+ T cells, young (3 mo) vs. old (22 mo) C57BL/6J

## Pipeline Overview

```
Raw counts (GEO) → Normalisation (DESeq2 VST) → QC (PCA, heatmap)
    → Differential Expression (apeglm shrinkage)
    → ORA (clusterProfiler, MSigDB Hallmarks + KEGG)
    → GSEA (fgsea, ranked by stat)
    → Cross-species comparison (human/mouse ortholog mapping)
    → Visualisation (volcano, dotplot, ridgeplot, GSEA enrichment plots)
```

## Files
| File | Description |
|------|-------------|
| `deseq2_aging_pipeline.R` | Full DESeq2 analysis: QC → DE → export |
| `gsea_ora_pathway_analysis.R` | ORA + GSEA with MSigDB, KEGG, GO |
| `pydeseq2_pipeline.py` | Python mirror with pyDESeq2 + GSEApy |
| `cross_species_comparison.R` | Ortholog mapping & conserved aging signature |

## Key Outputs
- `results/DE_genes_aged_vs_young.csv` — full DE table with shrunk LFC
- `results/ORA_Hallmarks.csv` — MSigDB Hallmark enrichment
- `results/GSEA_ranked.csv` — GSEA results ranked by NES
- `figures/` — publication-ready ggplot2 figures

## How to Run

```bash
# R pipeline
Rscript deseq2_aging_pipeline.R

# Python pipeline
python pydeseq2_pipeline.py

# Full pathway analysis
Rscript gsea_ora_pathway_analysis.R
```

## Dependencies
```r
# R
BiocManager::install(c("DESeq2","apeglm","clusterProfiler",
                       "org.Hs.eg.db","org.Mm.eg.db","fgsea",
                       "msigdbr","ggplot2","pheatmap","EnhancedVolcano"))
```
```bash
# Python
pip install pydeseq2 gseapy pandas matplotlib seaborn
```
