# bioetl-pipeline

A Python ETL pipeline that ingests antibody and B cell data from three public APIs,
validates and transforms it, and loads it into a PostgreSQL database.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │     OAS      │  │   ChEMBL     │  │       UniProt        │  │
│  │  (antibody   │  │ (antibody    │  │  (B cell surface     │  │
│  │  sequences)  │  │therapeutics) │  │      markers)        │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼──────────────────────┼─────────────┘
          │  concurrent.futures.ThreadPoolExecutor  │
          ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Extractors                               │
│  extract() → list[dict]  (raw API response, no transformation)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Transformers                              │
│  • Canonicalise sequences (uppercase, strip whitespace)         │
│  • Normalise VH gene names to IGHV#-## format                   │
│  • Compute CDR3 length                                          │
│  • Parse/standardise field names across sources                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Validator                                │
│  • Null checks on required fields                               │
│  • Amino acid alphabet validation (ACDEFGHIKLMNPQRSTVWY)        │
│  • SHA-256 content hash per record for duplicate detection      │
│  • Returns (clean_df, ValidationReport)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Loader                                 │
│  INSERT ... ON CONFLICT (content_hash) DO NOTHING               │
│  Batched upserts (default batch_size=100)                       │
│  Returns (n_inserted, n_skipped)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │                 │
                    │ ingested_       │
                    │ antibody_seqs   │
                    │ ingested_       │
                    │ therapeutics    │
                    │ ingested_       │
                    │ b_cell_markers  │
                    └─────────────────┘
```

## Data Sources

| Source | What it provides | Table loaded |
|--------|-----------------|--------------|
| **OAS** (Observed Antibody Space) | Human/mouse antibody sequences with VDJ gene usage and CDR3 annotations | `ingested_antibody_sequences` |
| **ChEMBL** | Approved and clinical-stage antibody therapeutics with mechanism of action | `ingested_therapeutics` |
| **UniProt** | B cell surface markers (CD19, CD20, CD38, CD79A, CD22, MS4A1) with function annotations | `ingested_b_cell_markers` |

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set DATABASE_URL to your PostgreSQL instance

# 4. Initialise ingestion tables
python cli.py init-db

# 5. (Optional) Seed synthetic data for demo
python seed_data.py --count 200
```

## Running the pipeline

```bash
# Run all three sources concurrently
python cli.py run --source all

# Run a single source
python cli.py run --source oas
python cli.py run --source chembl
python cli.py run --source uniprot

# Override database URL at runtime
python cli.py run --source all --db-url postgresql://user:pass@host:5432/db

# JSON-formatted logs (for log aggregation)
python cli.py run --source all --log-format json

# Check record counts per ingestion table
python cli.py status
```

### Example output

```
Running pipeline: sources=['oas', 'chembl', 'uniprot'] (concurrent)

Source    | Extracted | Passed | Inserted | Skipped
----------|-----------|--------|----------|--------
OAS       |        50 |     48 |       45 |       3
ChEMBL    |        30 |     29 |       28 |       1
UniProt   |        10 |     10 |       10 |       0
----------|-----------|--------|----------|--------
Total     |        90 |     87 |       83 |       4

Pipeline complete in 3.2s
```

## Design decisions

### Content-hash deduplication
Each record receives a `SHA-256` hash of its key content fields (stored as `content_hash`).
The loader uses `INSERT ... ON CONFLICT (content_hash) DO NOTHING`, making repeated runs
idempotent — re-ingesting the same data never creates duplicates.

### Concurrent extraction
All three API sources are fetched concurrently via `ThreadPoolExecutor`. I/O-bound HTTP
requests benefit significantly from this even in CPython (no GIL contention for network waits).
Each extractor is fully independent and fails gracefully without blocking others.

### API fallback
The OAS, ChEMBL, and UniProt APIs have network-based fallbacks to synthetic data.
This lets the pipeline run end-to-end in environments without internet access or when
APIs are temporarily unavailable, which is useful for CI and development.

### Structured logging
`structlog` is used throughout with two renderers:
- `json` — machine-readable for log aggregation (Elasticsearch, CloudWatch)
- `console` — human-readable pretty output for local development

## Ingestion schema

### `ingested_antibody_sequences`
| Column | Type | Description |
|--------|------|-------------|
| seq_id | TEXT PK | UUID |
| source | TEXT | Data source (OAS, OAS_SYNTHETIC) |
| vh_gene | TEXT | VH gene (e.g. IGHV3-23) |
| dh_gene | TEXT | DH gene |
| jh_gene | TEXT | JH gene |
| cdr3_aa | TEXT | CDR3 amino acid sequence |
| cdr3_length | INT | CDR3 length in amino acids |
| full_vh_aa | TEXT | Full VH domain amino acid sequence |
| isotype | TEXT | Antibody isotype (IgG1, IgM, etc.) |
| species | TEXT | Species of origin |
| study_id | TEXT | Source study identifier |
| content_hash | TEXT UNIQUE | SHA-256 of content for dedup |
| ingested_at | TIMESTAMPTZ | Ingest timestamp |

### `ingested_therapeutics`
| Column | Type | Description |
|--------|------|-------------|
| drug_id | TEXT PK | UUID |
| chembl_id | TEXT | ChEMBL identifier |
| name | TEXT | Drug name |
| max_phase | INT | Highest clinical trial phase |
| mechanism | TEXT | Mechanism of action |
| target_name | TEXT | Primary therapeutic target |
| sequence_or_smiles | TEXT | Sequence or SMILES if available |
| content_hash | TEXT UNIQUE | SHA-256 for dedup |
| ingested_at | TIMESTAMPTZ | Ingest timestamp |

### `ingested_b_cell_markers`
| Column | Type | Description |
|--------|------|-------------|
| marker_id | TEXT PK | UUID |
| uniprot_id | TEXT | UniProt accession |
| gene_name | TEXT | Gene symbol (CD19, CD20, etc.) |
| protein_name | TEXT | Full protein name |
| organism | TEXT | Species |
| function_text | TEXT | UniProt functional annotation |
| content_hash | TEXT UNIQUE | SHA-256 for dedup |
| ingested_at | TIMESTAMPTZ | Ingest timestamp |
