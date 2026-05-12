# Project 05 · Longevity GWAS & Polygenic Scores

## Biological Question
Which genetic variants associate with exceptional longevity,
what are their molecular mechanisms, and can we build a polygenic
score predictive of lifespan?

## Datasets
- **CHARGE Longevity GWAS** — 389,166 participants, lifespan GWAS (Timmers et al. 2019 eLife)
- **Centenarian GWAS** — 3,836 centenarians vs. 31,444 controls (Sebastiani et al. 2012)
- **UK Biobank** — Parental age proxy for lifespan GWAS (Pilling et al. 2017)
- **LDHub** — Cross-trait genetic correlations

## Pipeline
```
Summary stats (GWAS catalog) → Harmonisation → LD-score regression (h² estimate)
    → MAGMA gene-level analysis → Gene-set enrichment
    → Polygenic Risk Score (PRSice-2)
    → Cross-trait genetic correlation (LDHub)
    → Mendelian Randomisation (TwoSampleMR)
    → Visualisation: Manhattan, QQ, PRS distribution
```

## Files
| File | Description |
|------|-------------|
| `longevity_gwas_analysis.R` | Full GWAS pipeline: Manhattan, QQ, MAGMA, PRS |
| `mendelian_randomisation.R` | MR analysis: risk factors → longevity |

## Key Findings
- Genome-wide significant loci: APOE, FOXO3, CHRNA3/5, LPA
- Gene-set enrichment: lipid metabolism, immune function, DNA repair
- PRS explains ~3% of longevity variance in held-out test set
- MR evidence: LDL-C and smoking causally reduce longevity
