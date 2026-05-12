# MLOps Research Pipeline

> MLflow experiment tracking · Prefect workflow DAG · PySpark batch processing ·
> Reproducible model registry · CI/CD-ready pipeline

---

## Motivation

Modern pharma data science requires not just good models but reproducible,
auditable, and production-ready ML pipelines. This project demonstrates the
full MLOps lifecycle for a preclinical bioactivity prediction task: structured
experiment tracking, automated workflow orchestration, and scalable batch data
processing — the three pillars expected in a cloud-native (Azure/Databricks)
research data environment.

```
Raw Assay Data
      │
      ├─[data_processor.py]──── PySpark batch pipeline
      │                          Filter · Aggregate · Join · Parquet output
      │
      ├─[workflow_dag.py]──────  Prefect flow (7 tasks, retries, state logging)
      │                          Ingest → Validate → ETL → Train → Evaluate
      │                          → Register → Report
      │
      └─[experiment_tracker.py] MLflow experiment tracking
                                 3 models · run comparison · model registry
                                 → Production transition
```

---

## Key Results (synthetic CDK4 bioactivity dataset, n=500)

```
Best model              XGBoost  (AUC-ROC 0.92, RMSE 0.41 log-units)
MLflow runs logged       9  (3 models × 3 hyperparameter configs)
Models in registry       3  (RF, XGBoost, ElasticNet)
Production model         XGBoost v2 (best val AUC)
Prefect flow tasks       7  (all with retry logic)
PySpark partitions       5  (by assay_type, Parquet output)
```

---

## Modules

| Module | Method | Highlights |
|--------|--------|------------|
| `experiment_tracker.py` | MLflow autolog + manual logging | Run comparison, model registry, production transition |
| `workflow_dag.py` | Prefect 2.x flow + tasks | Retry logic, state callbacks, artifact passing |
| `data_processor.py` | PySpark local mode | DataFrame ops, window functions, partitioned Parquet |

---

## Dependencies

```
Python >= 3.10
numpy, pandas, scipy, scikit-learn, matplotlib
mlflow >= 2.0        (pip install mlflow)
prefect >= 2.0       (pip install prefect)   — graceful fallback if absent
pyspark >= 3.0       (pip install pyspark)   — graceful fallback if absent
```

## Quick Start

```bash
python data_processor.py        # PySpark batch processing → Parquet
python experiment_tracker.py    # MLflow experiment tracking + model registry
python workflow_dag.py          # Prefect workflow DAG end-to-end
```
