# Preclinical Research Data Platform

> SQL schema design · ETL pipeline · FAIR metadata · ChEBI/GO ontologies ·
> ELN/LIMS simulation · Plotly interactive dashboards

---

## Motivation

Preclinical drug discovery generates heterogeneous experimental data — compound
structures, biochemical assay results, cellular readouts, DMPK profiles — across
dozens of projects and CROs. This platform demonstrates the full data engineering
lifecycle: from raw assay uploads through a validated ETL pipeline into a
relational database annotated with biomedical ontologies, surfaced via an
interactive Plotly dashboard. Built to mirror the data architecture of modern
pharma ELN/LIMS environments (Benchling, Revvity Signals, Azure Data Factory).

```
Raw Experimental Data
        │
        ├─[etl_pipeline.py]─────── Extract → Validate → Transform → Load
        │                           FAIR metadata · unit normalisation · audit log
        │
        ├─[schema_design.py]──────  SQLite relational DB
        │                           compounds · assays · biological_entities · results
        │
        ├─[ontology_utils.py]──────  ChEBI + Gene Ontology annotation
        │                            Semantic similarity · pathway membership
        │
        └─[analytics_dashboard.py]─ Interactive Plotly dashboard (HTML)
                                     Dose-response · heatmap · selectivity · FAIR score
```

---

## Key Results (50 compounds × 5 assays, synthetic dataset)

```
Compounds registered        50  (with SMILES, MW, LogP, ChEBI annotation)
Assay types                  5  (biochemical IC50, cell viability, selectivity,
                                  DMPK microsomal stability, hERG patch clamp)
ETL pass rate               94%  (6% rejected: out-of-range values, missing units)
FAIR compliance score       87%  (findable, accessible, interoperable, reusable)
Top hit IC50                 3.2 nM  (CDK4 biochemical assay)
Selectivity ratio           48×  (CDK4 vs. CDK2 counter-screen)
```

---

## Modules

| Module | Method | Highlights |
|--------|--------|------------|
| `schema_design.py` | SQLite star schema | Compound + assay + entity + results tables, advanced SQL queries |
| `etl_pipeline.py` | 4-step DAG pipeline | Validation, unit conversion, FAIR metadata, audit trail |
| `ontology_utils.py` | ChEBI + GO annotation | Semantic similarity, pathway enrichment, controlled vocabularies |
| `analytics_dashboard.py` | Plotly interactive HTML | Dose-response Hill fit, activity heatmap, selectivity scatter |

---

## Dependencies

```
Python >= 3.10
numpy, pandas, scipy, scikit-learn
matplotlib
plotly >= 5.0
sqlite3  (stdlib)
```

## Quick Start

```bash
python schema_design.py        # create DB, populate, run example queries
python etl_pipeline.py         # run ETL on synthetic raw data
python ontology_utils.py       # annotate compounds and genes with ontology terms
python analytics_dashboard.py  # generate interactive HTML dashboard
```
