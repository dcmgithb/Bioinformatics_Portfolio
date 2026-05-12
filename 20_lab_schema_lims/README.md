# lab-schema-lims

A Laboratory Information Management System (LIMS) for an antibody discovery lab, tracking donors, B cell samples, flow cytometry runs, BCR sequencing, recombinant expression, and functional assay results.

---

## Project Overview

This LIMS stores the full antibody discovery pipeline in a relational PostgreSQL database:

1. **Donors** are enrolled with clinical metadata (age, sex, disease status, cohort).
2. **B cell samples** are collected from donors and stored under defined conditions.
3. **Flow cytometry runs** characterise B cell subsets (naive, memory, plasmablast) and QC each sample.
4. **Antibody sequences** from BCR sequencing are recorded with full VDJ gene assignments, CDR sequences, and clone IDs.
5. **Expression results** track recombinant antibody production yield and quality in CHO/HEK293/Expi293 systems.
6. **Assay results** record binding affinity (KD), neutralisation IC50, and ELISA pass/fail per sequence.
7. An **audit log** captures every INSERT/UPDATE/DELETE for regulatory compliance.

---

## Entity-Relationship Diagram

```mermaid
erDiagram
    users {
        string user_id PK
        string username
        string email
        string role
        bool is_active
        datetime created_at
    }
    instruments {
        string instrument_id PK
        string name
        string instrument_type
        string manufacturer
        string model
        string serial_number
        date calibration_date
        bool is_active
    }
    donors {
        string donor_id PK
        string donor_code
        int age
        string sex
        string disease_status
        string cohort
        date enrolled_at
    }
    b_cell_samples {
        string sample_id PK
        string donor_id FK
        string collected_by FK
        date collection_date
        float cell_count_1e6
        float viability_pct
        string storage_condition
        string notes
        datetime created_at
    }
    flow_cytometry_runs {
        string run_id PK
        string sample_id FK
        string instrument_id FK
        string operator_id FK
        date run_date
        string panel_name
        float b_cell_gate_pct
        float cd19_pct
        float cd20_pct
        float cd38_pct
        float cd138_pct
        float naive_b_pct
        float memory_b_pct
        float plasmablast_pct
        bool qc_pass
        datetime created_at
    }
    antibody_sequences {
        string seq_id PK
        string sample_id FK
        string chain_type
        string vh_gene
        string dh_gene
        string jh_gene
        string vl_gene
        string jl_gene
        string cdr1_aa
        string cdr2_aa
        string cdr3_aa
        int cdr3_length
        text full_vh_aa
        text full_vl_aa
        string isotype
        string clone_id
        int read_count
        string content_hash
        datetime created_at
    }
    expression_results {
        string result_id PK
        string seq_id FK
        string construct_name
        string expression_system
        float yield_mg_l
        float purity_pct
        float aggregation_pct
        float endotoxin_eu_ml
        date expression_date
        string expressed_by FK
        datetime created_at
    }
    assay_results {
        string assay_id PK
        string seq_id FK
        string assay_type
        string target_antigen
        float binding_kd_nm
        float ic50_nm
        float neutralisation_pct
        string pass_fail
        date assay_date
        string run_by FK
        datetime created_at
    }
    audit_log {
        string log_id PK
        string table_name
        string record_id
        string action
        string changed_by FK
        datetime changed_at
        json old_values
        json new_values
    }

    donors ||--o{ b_cell_samples : "provides"
    users ||--o{ b_cell_samples : "collects"
    b_cell_samples ||--o{ flow_cytometry_runs : "analysed_by"
    instruments ||--o{ flow_cytometry_runs : "used_in"
    users ||--o{ flow_cytometry_runs : "operates"
    b_cell_samples ||--o{ antibody_sequences : "yields"
    antibody_sequences ||--o{ expression_results : "expressed_as"
    antibody_sequences ||--o{ assay_results : "tested_in"
    users ||--o{ expression_results : "performs"
    users ||--o{ assay_results : "runs"
    users ||--o{ audit_log : "triggers"
```

---

## Schema Design Decisions

### Third Normal Form (3NF)
All tables are in 3NF: every non-key attribute depends on the whole primary key and nothing but the primary key. Donor metadata lives in `donors` only; sample metadata lives in `b_cell_samples` only. Repeating groups (CDR sequences, assay readings) are split into dedicated tables.

### UUID Primary Keys
All PKs are `UUID` (stored as `VARCHAR(36)`). Benefits:
- Globally unique — safe for distributed insertion without a sequence lock
- Opaque — no information leakage in API responses
- Compatible with PostgreSQL native `uuid` type when cast

### Content Hash for Deduplication
`antibody_sequences.content_hash` is a SHA-256 hash (first 32 hex chars) of `full_vh_aa || full_vl_aa`. A `UNIQUE` constraint on this column prevents identical sequences from being inserted twice regardless of which sample they appear in, supporting BCR repertoire deduplication workflows.

### JSONB Audit Log
`audit_log.old_values` / `new_values` use `JSON` (JSONB in PostgreSQL). This schema-free design stores complete snapshots of any row before/after mutation, which is essential for 21 CFR Part 11 compliance without requiring per-table audit tables.

---

## Setup Instructions

### 1. Create the PostgreSQL database

```bash
createdb lims_db
psql lims_db -c "CREATE USER lims_user WITH PASSWORD 'changeme';"
psql lims_db -c "GRANT ALL PRIVILEGES ON DATABASE lims_db TO lims_user;"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set DATABASE_URL
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Alembic migrations

```bash
export DATABASE_URL=postgresql://lims_user:changeme@localhost:5432/lims_db
alembic upgrade head
```

### 5. Seed synthetic data

```bash
python seed_data.py --db-url postgresql://lims_user:changeme@localhost:5432/lims_db
```

---

## Query Files

### `queries/vdj_gene_usage.sql`
Computes per-VH-gene sequence counts and frequency percentages, then groups by IGHV family. Uses `REGEXP_REPLACE` to strip the allele suffix, window functions for frequency, and `RANK()` for ordering. Useful for repertoire analysis and clonotype diversity summaries.

### `queries/clone_frequency.sql`
Identifies expanded B cell clones and computes clonal frequency as percentage of total reads. Also computes the Shannon diversity index across all clones using the per-clone `-p*ln(p)` contribution summed via a window function. Uses `MODE() WITHIN GROUP` to find the most common isotype and VH gene per clone.

### `queries/expression_yield.sql`
Aggregates expression experiment results by construct and expression system. Computes mean, median, standard deviation, min, and max yield; mean purity and aggregation. Ranks constructs both within their expression system and globally. Filtered to constructs with at least 2 runs for statistical validity.

### `queries/assay_aggregation.sql`
Joins assay results back through sequences → samples → donors to add cohort and disease status context. Computes pass rates, median and mean KD, median IC50, and mean neutralisation percentage per cohort/disease/assay type/antigen combination. Enables cross-cohort comparison of antibody functional quality.

---

## Index Strategy

| Table | Index | Rationale |
|-------|-------|-----------|
| `b_cell_samples` | `donor_id` | Fast lookup of all samples from a donor |
| `b_cell_samples` | `collection_date` | Date-range filtering for longitudinal studies |
| `antibody_sequences` | `sample_id` | Join from samples → sequences |
| `antibody_sequences` | `clone_id` | Clone-level aggregation queries |
| `antibody_sequences` | `vh_gene` | VDJ gene usage statistics |
| `antibody_sequences` | `cdr3_length` | CDR3 length distribution analysis |
| `expression_results` | `seq_id` | Join from sequences → expression |
| `expression_results` | `construct_name` | Lookup by construct |
| `assay_results` | `seq_id` | Join from sequences → assays |
| `assay_results` | `assay_type` | Filter by assay type (SPR, ELISA, …) |
| `assay_results` | `pass_fail` | QC filtering |
| `flow_cytometry_runs` | `sample_id` | Lookup runs for a sample |
| `flow_cytometry_runs` | `run_date` | Date-range filtering |
| `audit_log` | `(table_name, record_id)` | Composite: fetch audit trail for a row |
| `audit_log` | `changed_at` | Time-based audit queries |

All unique constraints (`content_hash`, `username`, `email`, `serial_number`, `donor_code`) are backed by implicit B-tree indexes in PostgreSQL.
