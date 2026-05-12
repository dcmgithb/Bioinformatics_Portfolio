# Bioinformatics Portfolio — Aging, Genomics, Cheminformatics & Precision Medicine

> A curated collection of end-to-end bioinformatics projects spanning the molecular biology of ageing,
> centromere instability, immunosenescence, multi-omics systems biology, computational drug discovery,
> precision medicine, preclinical data science, and antibody/B cell data engineering.
> Built to demonstrate production-grade proficiency across the full computational biology stack.

---

## Who This Is For

This portfolio targets positions in **computational biology, cheminformatics, precision medicine, research data science, and data engineering** that demand:
- Deep wet-lab intuition translated into rigorous computational pipelines
- Fluency across R *and* Python ecosystems simultaneously
- Ability to go from raw sequencing reads or SMILES strings → publication-ready biology or deployable ML models
- Experience with AI output evaluation, data governance, and biomedical ontologies
- Familiarity with MLOps tooling, relational data modelling, and workflow orchestration
- Production data engineering skills: ETL pipelines, REST APIs, schema design, and automated data quality monitoring

---

## Portfolio Map

| # | Project | Core Skills | Key Tools |
|---|---------|-------------|-----------|
| 01 | [Bulk RNA-seq Aging Analysis](#01) | Differential expression, pathway analysis | DESeq2, clusterProfiler, GSEApy, MSigDB |
| 02 | [Epigenetic Age Clocks](#02) | DNA methylation, biological age prediction | Horvath/PhenoAge/GrimAge clocks, glmnet |
| 03 | [scRNA-seq Immunosenescence](#03) | Single-cell, immune cell heterogeneity | Seurat, Scanpy, trajectory inference |
| 04 | [Centromere Genomic Instability](#04) | Repeat genomics, satellite DNA, CNV | RepeatMasker, bedtools, GATK |
| 05 | [Longevity GWAS & Polygenic Scores](#05) | Population genetics, long-lived individuals | PLINK2, PRSice, LDscore |
| 06 | [Automated Sequencing Workflows](#06) | Reproducible pipelines, HPC | Snakemake, Nextflow, Docker |
| 07 | [Multi-omics Aging Network](#07) | Systems biology, network inference | WGCNA, MOFA2, networkx, igraph |
| 08 | [Geroprotector Cheminformatics](#08) | QSAR, molecular fingerprints, virtual screening | RDKit, XGBoost, SHAP, UMAP |
| 09 | [GNN Molecular Binding](#09) | Graph neural networks, binding affinity | PyTorch, MPNN, AttentiveFP |
| 10 | [Generative Molecule Design](#10) | Transformer SMILES generation, RL fine-tuning | PyTorch, REINFORCE, QED/SA scoring |
| 11 | [Protein ML & Fitness Landscape](#11) | ESM-2 embeddings, multi-task learning | transformers, fair-esm, directed evolution |
| 12 | [Agentic Drug Discovery](#12) | ReAct agent loop, cheminformatics tools | Claude API, RDKit, molecular memory |
| 13 | [Computational ADMET](#13) | ADMET prediction, metabolite generation, PBPK | RDKit, GBM, SMARTS, ODE modelling |
| 14 | [Precision Medicine Platform](#14) | GWAS, multi-modal ML, patient stratification | scikit-learn, SHAP, KM survival, HIPAA |
| 15 | [AI Evaluation Framework](#15) | Rubric scoring, fact-checking, IAA metrics | Cohen's κ, Krippendorff's α, KB verification |
| 16 | [Preclinical Data Platform](#16) | SQL schema, ETL, FAIR metadata, ontology | SQLite, Plotly, ChEBI, Gene Ontology |
| 17 | [MLOps Research Pipeline](#17) | Experiment tracking, workflow DAG, batch processing | MLflow, Prefect, PySpark |
| 18 | [RNA Oligo Design](#18) | ASO/siRNA design, thermodynamics, ML activity | Nearest-neighbour Tm, Nussinov, SHAP |
| utils | [Shared Utilities](#utils) | Data retrieval, GEO/SRA automation | SRA Toolkit, GEOquery, rentrez |
| 19 | [B Cell ETL Pipeline](#19) | API extraction, ETL, deduplication, concurrent I/O | requests, pandas, SQLAlchemy, structlog, Click |
| 20 | [Antibody Lab LIMS Schema](#20) | Relational schema design, migrations, analytical SQL | PostgreSQL, SQLAlchemy 2.0, Alembic |
| 21 | [Biodata REST API](#21) | REST API design, async DB access, auth, pagination | FastAPI, asyncpg, Pydantic v2, Docker |
| 22 | [Data Quality Monitor](#22) | Automated DQ checks, outlier detection, integrity audits | psycopg2, dataclasses, cron scheduling |

---

## Project Highlights

### 01 · Bulk RNA-seq Aging Analysis {#01}
**`01_bulk_rnaseq_aging/`**

Full DESeq2 pipeline from count matrix to publication-ready figures. Integrates:
- Variance-stabilising transformation & PCA quality control
- Shrinkage-corrected differential expression (apeglm)
- Over-Representation Analysis (ORA) via clusterProfiler + MSigDB Hallmarks
- Gene Set Enrichment Analysis (GSEA) with ranked gene lists
- Python mirror using pyDESeq2 + GSEApy for cross-validation

**Biological question**: How does ageing reshape the transcriptome of mouse immune tissues?

---

### 02 · Epigenetic Biological Age Clocks {#02}
**`02_epigenetic_clocks/`**

Implements four published methylation clocks and benchmarks them on public PBMC data:
- **Horvath (2013)** — pan-tissue clock
- **Hannum (2013)** — blood-specific clock
- **PhenoAge (2018)** — phenotypic age
- **GrimAge (2019)** — mortality predictor

Includes: clock residual analysis ("epigenetic acceleration"), visualisation of clock agreement, and a Python ML pipeline that trains a custom ElasticNet clock on held-out data.

---

### 03 · scRNA-seq Immunosenescence {#03}
**`03_scrnaseq_immunosenescence/`**

Dual-language (R/Seurat + Python/Scanpy) single-cell pipeline on PBMC data from young vs. aged donors:
- Ambient RNA removal, doublet detection, QC filtering
- Graph-based clustering & UMAP embedding
- Differential abundance testing across age groups (Milo)
- Cell-type-specific pseudobulk differential expression
- Pseudotime trajectory (Monocle3 / scVelo)
- Senescence score projection (custom gene signature + UCell)

---

### 04 · Centromere Genomic Instability {#04}
**`04_centromere_instability/`**

Genomic analysis pipeline targeting centromeric instability in ageing:
- Satellite repeat quantification from short-read WGS
- α-satellite copy number variation across age groups
- Centromere chromatin state inference from ATAC-seq signal
- Micronuclei burden as proxy for CIN — integration with expression data
- CENP-A/B occupancy analysis from ChIP-seq data

---

### 05 · Longevity GWAS & Polygenic Scores {#05}
**`05_longevity_gwas/`**

Population genetics pipeline for long-lived individuals (centenarians vs. controls):
- GWAS QC, imputation-ready file preparation with PLINK2
- LD-score regression for SNP heritability estimation
- Polygenic Risk Score (PRS) construction for longevity traits
- Gene-set enrichment on GWAS hits (MAGMA)
- Cross-trait genetic correlation with disease GWAS

---

### 06 · Automated Sequencing Workflows {#06}
**`06_automated_workflows/`**

Production-grade, containerised pipelines for RNA-seq data processing:
- **Snakemake** workflow: FastQC → Trim Galore → STAR → featureCounts → MultiQC
- **Nextflow** (DSL2) equivalent for HPC/cloud
- Docker + Conda environment definitions for full reproducibility
- Automatic GEO/SRA download and metadata parsing

---

### 07 · Multi-omics Aging Network {#07}
**`07_multiomics_aging_network/`**

Systems-biology integration of transcriptomics + methylomics + proteomics into an aging network:
- WGCNA co-expression network construction & module-trait correlation
- MOFA2 multi-omics factor analysis (latent factor decomposition)
- Network construction with igraph/networkx
- Hub gene identification & network resilience analysis
- Visualisation: Cytoscape-ready output + Python interactive plots

---

### 08 · Geroprotector Cheminformatics {#08}
**`08_geroprotector_cheminformatics/`**

End-to-end cheminformatics pipeline on a curated library of 75 geroprotector compounds:
- RDKit property profiling (Lipinski, QED, SA score, TPSA, PAINS)
- Molecular fingerprint PCA/UMAP for chemical space mapping
- QSAR ensemble: Random Forest + XGBoost + ElasticNet with SHAP explainability
- Virtual screening against longevity-associated targets
- Network pharmacology: target–compound interaction graph

---

### 09 · GNN Molecular Binding {#09}
**`09_gnn_molecular_binding/`**

Graph neural network models for binding affinity prediction on a CDK4/6 dataset:
- RDKit → PyTorch graph featurisation (49-dim node, 11-dim edge features)
- Two GNN architectures: MPNN and AttentiveFP (per-atom attention weights)
- Random Forest baseline with Morgan fingerprints
- Train/val/test split with AUC-ROC and MAE evaluation

---

### 10 · Generative Molecule Design {#10}
**`10_generative_molecule_design/`**

Three-stage de novo molecule generation pipeline targeting CDK4/6 inhibitors:
- GPT-style Transformer pre-trained on SMILES character sequences
- Fine-tuning on CDK4/6-active compounds
- REINFORCE RL with multi-objective reward: QED · SA score · Tanimoto · selectivity
- Validity, uniqueness, and novelty metrics for generated libraries

---

### 11 · Protein ML & Fitness Landscape {#11}
**`11_protein_ml_fitness/`**

ESM-2 (650M parameter) protein language model embeddings for multi-task property prediction:
- Binding affinity · aggregation risk · thermal stability (uncertainty-weighted loss)
- Physicochemical descriptor fusion
- Fitness landscape scanning and directed evolution simulation
- Full `mock=True` mode for CPU-only runs without ESM weights

---

### 12 · Agentic Drug Discovery {#12}
**`12_agentic_drug_discovery/`**

ReAct-style autonomous agent for multi-step drug discovery campaigns:
- 8 cheminformatics tools: similarity search, binding affinity predictor, ADMET, scaffold hopper, lead optimiser, literature search, property calculator, visualiser
- `MolecularMemory` with canonical SMILES deduplication and Bemis-Murcko scaffold clustering
- MockLLM deterministic 20-step decision schedule (no API key required)
- Session-persistent JSON memory with audit trail

---

### 13 · Computational ADMET {#13}
**`13_computational_admet/`**

Modular ADMET prediction pipeline for drug candidate profiling:
- SoM prediction via atom-level GNN scoring
- 40 SMARTS-based Phase I/II metabolite generation rules
- CYP isoform multi-task classifier (CYP3A4/2D6/2C9/1A2/2C19)
- D-MPNN (directed message passing neural network, ChemProp architecture) for multi-task ADMET
- Structural alerts (20 SMARTS patterns) + DILI/hERG/mutagenicity classifiers
- 2-compartment PBPK ODE model with Michaelis-Menten elimination

---

### 14 · Precision Medicine Platform {#14}
**`14_precision_medicine_platform/`**

Full lifecycle precision medicine platform on a synthetic 500-patient cohort:
- Genomic pipeline: GWAS with PCA covariates, Manhattan + QQ plots, variant annotation
- Clinical harmonisation: ICD-10 → HPO ontology mapping, HIPAA Safe Harbour de-identification
- Multi-modal biomarker discovery: RF + XGBoost + SHAP, RFECV panel selection, NRI/IDI
- Patient stratification: consensus clustering, Kaplan-Meier survival, UMAP
- Data governance: provenance DAG, audit trail, HIPAA/GDPR/FDA compliance checklist

**Key results**: Biomarker panel AUC 0.87 · 4 patient subgroups (silhouette 0.61) · 100% de-identification

---

### 15 · AI Evaluation Framework {#15}
**`15_ai_evaluation_framework/`**

Structured quality pipeline for evaluating AI-generated precision medicine content:
- 6-dimension weighted rubric (Scientific Accuracy 0.30 · Clinical Relevance 0.25 · Data Integrity 0.20 · Completeness 0.10 · Reasoning Quality 0.10 · Regulatory Compliance 0.05)
- Biomedical fact-checker: entity extraction (genes, variants, drugs, biomarkers), KB verification, hallucination rate metrics
- Multi-annotator pipeline: Cohen's κ (weighted ordinal), Krippendorff's α, adjudication workflow, batch drift detection
- Aggregated 4-panel quality dashboard + pass/fail quality report

**Key results**: Hallucination rate 18% · Inter-annotator κ 0.72 · Krippendorff's α 0.69

---

### 16 · Preclinical Data Platform {#16}
**`16_preclinical_data_platform/`**

End-to-end data engineering platform for preclinical drug discovery data:
- SQLite relational star schema: compounds, assays, biological entities, results (fact table with DDL constraints and indices)
- 4-step ETL pipeline (Extract → Validate → Transform → Load) with FAIR metadata, unit normalisation, and audit trail
- ChEBI compound hierarchy + Gene Ontology slim annotation; Resnik information-content semantic similarity
- Interactive **Plotly** HTML dashboard: 4-parameter Hill dose-response curves, compound × assay Z-score heatmap, selectivity scatter, FAIR compliance score

**Key results**: 50 compounds × 5 assays · FAIR score 87% · Top hit IC₅₀ 3.2 nM · Selectivity 48×

---

### 17 · MLOps Research Pipeline {#17}
**`17_mlops_research_pipeline/`**

Production-grade MLOps infrastructure for bioactivity model development:
- **MLflow** experiment tracking: RF, XGBoost, and ElasticNet runs with autolog, metric comparison, artifact storage, and model registry with Production stage transition
- **Prefect 2.x** workflow DAG: 7 tasks (ingest → validate → ETL → train → evaluate → register → report) with per-task retry logic and artifact passing; plain-Python fallback for environments without Prefect
- **PySpark** local-mode batch processing: DataFrame aggregations, window-function ranking (IC₅₀ rank within assay type), partitioned Parquet output; pandas fallback for environments without Spark

**Key results**: Best AUC-ROC 0.92 (XGBoost) · 9 MLflow runs logged · 5 Parquet partitions by assay type

---

### 18 · RNA Oligonucleotide Design {#18}
**`18_rna_oligo_design/`**

Computational design platform for antisense oligonucleotides targeting genetic medicine programs:
- Sequence tiling (ASO, siRNA, gapmer) over synthetic KRAS mRNA with nearest-neighbour Tm (SantaLucia 1998), self-complementarity scoring, and 6-mer seed off-target burden
- Nussinov dynamic programming RNA secondary structure (dot-bracket notation, coarse MFE estimation)
- Sliding-window target site accessibility; 3-mer motif enrichment (log-odds active vs. inactive)
- ML knockdown prediction: RF regressor + XGBoost classifier; permutation importance waterfall for top candidates

**Key results**: 1 980 candidates scored · Top predicted knockdown 94% · Model AUC-ROC 0.89 · 18% filtered by self-complementarity

---

---

## Data Engineering Projects — B Cell & Antibody Biology

Four self-contained projects demonstrating production-grade data engineering skills across B cell and antibody biology. Stack: Python, PostgreSQL, FastAPI, Alembic.

---

### 19 · B Cell ETL Pipeline {#19}
**`19_bioetl_pipeline/`**

Python ETL pipeline that ingests antibody and B cell data from three public APIs and loads it into PostgreSQL:
- **OAS** (Observed Antibody Space) — antibody VH sequences with VDJ gene usage and CDR3 annotations
- **ChEMBL** — approved and clinical-stage antibody therapeutics with mechanism of action
- **UniProt** — B cell surface markers (CD19, CD20, CD38, CD79A, CD22) with functional annotations
- All three sources extracted concurrently via `ThreadPoolExecutor`; each fails gracefully without blocking others
- Transformer canonicalises sequences (uppercase, strip whitespace), normalises VH gene names to `IGHV#-##` format, and computes CDR3 length
- Validator assigns a SHA-256 content hash per record; loader uses `INSERT … ON CONFLICT (content_hash) DO NOTHING` for idempotent reruns
- `structlog` with JSON and console renderers; `pydantic-settings` for all configuration; Click CLI (`run`, `init-db`, `status`)
- API fallbacks to synthetic data allow the pipeline to run end-to-end without internet access

---

### 20 · Antibody Lab LIMS Schema {#20}
**`20_lab_schema_lims/`**

PostgreSQL relational schema for a full antibody discovery laboratory, from donor enrolment to assay results:
- 9 tables in 3NF: `users`, `instruments`, `donors`, `b_cell_samples`, `flow_cytometry_runs`, `antibody_sequences`, `expression_results`, `assay_results`, `audit_log`
- SQLAlchemy 2.0 ORM (`DeclarativeBase`, `Mapped`, `mapped_column`) with all foreign keys and covering indexes
- Single Alembic migration (`001_initial_schema.py`) creates the full schema with `upgrade()` / `downgrade()`
- Four analytical SQL queries: VDJ gene usage frequency, clone frequency with Shannon diversity (H) and D50, expression yield ranking by system, assay pass rates per donor cohort
- Realistic seed data generator: 20 donors (healthy / autoimmune-lupus / CLL), 60 samples, 120 flow cytometry runs, ~480 antibody sequences with biologically plausible CDR3s, expression results, and assay KD/IC50 values

---

### 21 · Biodata REST API {#21}
**`21_biodata_api/`**

FastAPI REST API exposing the LIMS schema for upstream consumers and analysis tools:
- Full CRUD for samples (`GET`, `POST`, `PATCH`, `DELETE` with soft-delete) and sequences (immutable after creation)
- Cursor-based pagination on list endpoints; `donor_id`, `vh_gene`, `cdr3_length`, `isotype`, `clone_id` filters on search
- `GET /report` computes clone diversity (Shannon H, D50), top-10 expressing constructs, and assay pass rates server-side
- API-key authentication via `X-API-Key` header using FastAPI `Security(APIKeyHeader)`
- Async SQLAlchemy + `asyncpg` throughout; `lazy="selectin"` for nested ORM loading without N+1 queries
- Pydantic v2 response models with `ConfigDict(from_attributes=True)`; `lifespan` context manager for engine lifecycle
- Dockerfile: `python:3.11-slim`, uvicorn, ready for container deployment

---

### 22 · Data Quality Monitor {#22}
**`22_data_quality_monitor/`**

Scheduled data quality monitor for the LIMS database, designed for nightly cron execution:
- 12 checks across four domains — sequence integrity, expression outliers, flow cytometry gates, referential integrity
- `BaseCheck` ABC + `CheckResult` dataclass (`check_name`, `status`, `affected_rows`, `details`, `message`, `run_at`)
- Sequence checks: missing CDR3, malformed CDR3 (non-standard amino acid characters + length bounds), duplicate content hashes
- Expression checks: IQR × 1.5 outlier detection per `expression_system`, negative yield, missing purity above 20% threshold
- Flow checks: B cell gate out of physiological range (0.5–60%), subset percentages summing above 100%, QC flag mismatches
- Integrity checks: orphan samples (no flow runs), orphan sequences (no assay results), assay results referencing missing sequences
- JSON report with per-check details + human-readable summary; argparse CLI exits with code 1 on any `fail`-status check

---

## Tech Stack at a Glance

```
Languages        R (4.x) · Python (3.10+) · Bash · SQL
R Packages       DESeq2 · edgeR · clusterProfiler · Seurat · Monocle3
                 WGCNA · MOFA2 · limma · ggplot2 · tidyverse
Python Libs      pyDESeq2 · Scanpy · GSEApy · pandas · scikit-learn · scipy
                 networkx · matplotlib · seaborn · plotly · anndata
                 RDKit · XGBoost · SHAP · umap-learn
                 torch · transformers · fair-esm
                 mlflow · prefect · pyspark
                 anthropic (Claude API)
Data Engineering FastAPI · asyncpg · SQLAlchemy 2.0 · Alembic · psycopg2
                 pydantic-settings · structlog · Click · uvicorn
Databases        PostgreSQL · SQLite · GEO · MSigDB · ENCODE · 1000 Genomes
                 UK Biobank · ChEBI · Gene Ontology (OBO)
                 OAS (Observed Antibody Space) · ChEMBL · UniProt
Workflows        Snakemake · Nextflow (DSL2) · Docker · Conda
CLI Tools        SRA Toolkit · PLINK2 · STAR · HISAT2 · Trim Galore
                 featureCounts · samtools · bedtools · GATK4
```

---

## Reproducibility Philosophy

Every project follows these principles:
1. **Seed-locked** — all stochastic steps use fixed random seeds
2. **Environment-pinned** — `renv.lock` (R) and `environment.yml` (Python) included
3. **Documented inputs** — all datasets traced to GEO/SRA accession numbers
4. **Tested** — core functions have unit tests (`testthat` / `pytest`)
5. **Containerised** — Dockerfiles provided for HPC deployment

---

## Contact & Context

This portfolio spans ageing biology, cheminformatics, agentic AI systems, precision medicine data science,
pharma data engineering, and antibody/B cell data infrastructure — reflecting the breadth of modern computational
biology and research data science roles.
Projects 01–07 focus on ageing multi-omics; 08–13 on drug discovery and ADMET; 14–15 on precision medicine
and AI output evaluation; 16–18 on preclinical data platforms, MLOps, and RNA therapeutic design;
19–22 on production data engineering for antibody discovery (ETL, LIMS schema, REST API, DQ monitoring).

---

*All code runs on publicly available datasets. Raw data is never committed — accession numbers and download scripts are provided instead.*
