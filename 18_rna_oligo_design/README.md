# RNA Oligonucleotide Design Platform

> ASO / siRNA / gapmer design · Thermodynamic scoring · Off-target seed analysis ·
> ML activity prediction · RNA secondary structure

---

## Motivation

Oligonucleotide therapeutics (ASOs, siRNAs, gapmers) are a rapidly expanding
drug modality in genetic medicine. Optimal design requires balancing on-target
knockdown efficacy against thermodynamic stability, immunogenicity, and
off-target transcript binding. This platform implements the key computational
steps: tiling candidate sequences over a target mRNA, scoring thermodynamic
and structural properties, predicting off-target seed-match burden, and training
an ML model to predict percent knockdown from sequence features.

```
Target mRNA Sequence
        │
        ├─[oligo_designer.py]──── Sequence tiling + thermodynamic scoring
        │                          GC%, Tm (nearest-neighbour), self-complementarity
        │                          Seed off-target scoring · Accessibility window
        │
        ├─[activity_predictor.py] ML knockdown prediction
        │                          RF + XGBoost · SHAP waterfall
        │                          500 synthetic oligo–activity training pairs
        │
        └─[sequence_analysis.py]── RNA structure + motif analysis
                                    Nussinov dot-bracket · Target site map
                                    Sequence logo · Tm distribution
```

---

## Key Results (synthetic KRAS mRNA target, 1000 candidate oligos)

```
Candidates designed          1000  (20 nt ASOs, tiled every 1 nt)
Optimal GC% range            40–60%  (filter)
Tm range (37°C buffer)       55–75 °C
Off-target seed burden        mean 3.2 seed matches per oligo
Top predicted knockdown       94%  (XGBoost, 20-nt ASO position 847)
Model AUC-ROC                0.89  (≥70% knockdown active class)
Self-complementarity filter  removed 18%  of candidates
```

---

## Modules

| Module | Method | Highlights |
|--------|--------|------------|
| `oligo_designer.py` | Nearest-neighbour Tm, seed matching | ASO/siRNA/gapmer tiling, accessibility score, candidate ranking |
| `activity_predictor.py` | RF + XGBoost + SHAP | 500 synthetic training pairs, SHAP waterfall, AUC comparison |
| `sequence_analysis.py` | Nussinov algorithm | Dot-bracket structure, motif enrichment, sequence logo |

---

## Dependencies

```
Python >= 3.10
numpy, pandas, scipy, scikit-learn, matplotlib
```

## Quick Start

```bash
python oligo_designer.py        # tile + score candidates, rank top 20
python activity_predictor.py    # train ML model, SHAP waterfall
python sequence_analysis.py     # secondary structure + motif analysis
```
