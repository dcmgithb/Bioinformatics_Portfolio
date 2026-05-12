# Project 02 · Epigenetic Biological Age Clocks

## Biological Question
Can DNA methylation patterns accurately predict biological age in immune cells,
and do long-lived individuals show reduced epigenetic ageing (negative clock acceleration)?

## Dataset
- **GSE40279** — Hannum 2013 blood methylation array (n=656, ages 19–101)
- **GSE55763** — Large-scale PBMC methylation (n=2711)
- **GSE87571** — Whole blood methylation, age 14–94 (n=729)

## Clocks Implemented
| Clock | Year | CpG Sites | Tissue | Predicts |
|-------|------|-----------|--------|---------|
| Horvath | 2013 | 353 | Pan-tissue | Chronological age |
| Hannum | 2013 | 71 | Blood | Chronological age |
| PhenoAge | 2018 | 513 | Blood | Phenotypic age |
| GrimAge | 2019 | 1030 | Blood | Time-to-death |

## Pipeline Overview
```
450K/EPIC Array → Minfi normalisation → CpG QC
    → Clock CpG intersection
    → Age prediction (4 clocks)
    → Clock agreement & Bland-Altman
    → Epigenetic age acceleration (residuals)
    → Association with phenotype
    → Custom ElasticNet clock training (Python)
```

## Files
| File | Description |
|------|-------------|
| `biological_age_prediction.R` | Clock predictions, acceleration, inter-clock agreement, longevity analysis |
| `custom_clock_training.py` | Train custom ElasticNet methylation clock with nested CV |

## How to Run
```bash
Rscript biological_age_prediction.R
python custom_clock_training.py
```
