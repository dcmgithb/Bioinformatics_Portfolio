# BioData API

A production-ready REST API for an antibody discovery laboratory LIMS (Laboratory Information Management System). Built with FastAPI, SQLAlchemy 2.0 async, and PostgreSQL.

---

## Overview

The API exposes four resource groups:

| Prefix | Description |
|--------|-------------|
| `/samples` | B-cell sample management (CRUD) |
| `/sequences` | Antibody sequence management (create, read, update — immutable once created) |
| `/search/sequences` | Multi-criteria sequence search |
| `/report` | Aggregate report: clone diversity + expression + assay statistics |

All endpoints require an API key supplied in the `X-API-Key` request header.

---

## Authentication

Every request must include the header:

```
X-API-Key: your-secret-api-key-here
```

The server validates this against the `API_KEY` environment variable. Missing or incorrect keys return `HTTP 401 Unauthorized`.

---

## Full Endpoint Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | None | Health check — returns `{"status":"ok","db":"connected"}` |
| `GET` | `/samples` | Required | Paginated list of B-cell samples. Query: `page`, `size`, `donor_id` |
| `POST` | `/samples` | Required | Create a new B-cell sample |
| `GET` | `/samples/{sample_id}` | Required | Retrieve a sample with nested flow cytometry runs |
| `PATCH` | `/samples/{sample_id}` | Required | Partial update of a sample |
| `DELETE` | `/samples/{sample_id}` | Required | Hard-delete a sample (204 No Content) |
| `GET` | `/sequences` | Required | Paginated list of antibody sequences. Query: `page`, `size`, `sample_id` |
| `POST` | `/sequences` | Required | Create a new antibody sequence (auto-computes `cdr3_length` and `content_hash`) |
| `GET` | `/sequences/{seq_id}` | Required | Retrieve a sequence with nested expression and assay results |
| `PATCH` | `/sequences/{seq_id}` | Required | Partial update of a sequence |
| `DELETE` | `/sequences/{seq_id}` | Required | Returns 405 — sequences are immutable |
| `GET` | `/search/sequences` | Required | Search sequences by VH gene family, CDR3 length range, isotype, clone_id |
| `GET` | `/report` | Required | Aggregate report: Shannon diversity, D50, top constructs, assay pass rates |
| `GET` | `/docs` | None | Interactive Swagger UI |
| `GET` | `/redoc` | None | ReDoc documentation |

---

## Setup — Without Docker

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 14+ with the LIMS schema loaded
- Conda or pip virtual environment

### 2. Install dependencies

```bash
cd biodata-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your real database URL and API key
```

### 4. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit http://localhost:8000/docs for the interactive API documentation.

---

## Setup — With Docker

```bash
# Build the image
docker build -t biodata-api .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://lims_user:changeme@db:5432/lims_db" \
  -e API_KEY="your-secret-api-key-here" \
  --name biodata-api \
  biodata-api
```

Or with Docker Compose (add a `docker-compose.yml` pointing at a Postgres service).

---

## Example curl Commands

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","db":"connected"}
```

### List samples (first page, 20 items)

```bash
curl -H "X-API-Key: your-secret-api-key-here" \
     "http://localhost:8000/samples?page=1&size=20"
```

### List samples filtered by donor

```bash
curl -H "X-API-Key: your-secret-api-key-here" \
     "http://localhost:8000/samples?donor_id=550e8400-e29b-41d4-a716-446655440000"
```

### Create a sample

```bash
curl -X POST http://localhost:8000/samples \
     -H "X-API-Key: your-secret-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{
       "donor_id": "550e8400-e29b-41d4-a716-446655440000",
       "collection_date": "2024-03-15",
       "cell_count_1e6": 25.4,
       "viability_pct": 94.2,
       "storage_condition": "liquid_nitrogen",
       "notes": "Healthy donor, fasting sample"
     }'
# 201 Created — returns full SampleRead with sample_id
```

### Get a single sample

```bash
curl -H "X-API-Key: your-secret-api-key-here" \
     http://localhost:8000/samples/7c9e6679-7425-40de-944b-e07fc1f90ae7
```

### Partial update a sample

```bash
curl -X PATCH http://localhost:8000/samples/7c9e6679-7425-40de-944b-e07fc1f90ae7 \
     -H "X-API-Key: your-secret-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{"notes": "Updated: post-stimulation sample"}'
```

### Delete a sample

```bash
curl -X DELETE http://localhost:8000/samples/7c9e6679-7425-40de-944b-e07fc1f90ae7 \
     -H "X-API-Key: your-secret-api-key-here"
# 204 No Content
```

### Create a sequence

```bash
curl -X POST http://localhost:8000/sequences \
     -H "X-API-Key: your-secret-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{
       "sample_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
       "chain_type": "heavy",
       "vh_gene": "IGHV3-23*01",
       "dh_gene": "IGHD3-10*01",
       "jh_gene": "IGHJ4*02",
       "cdr3_aa": "ARDYYYYGMDV",
       "isotype": "IgG1",
       "clone_id": "clone_001",
       "read_count": 142
     }'
```

### Search sequences

```bash
# All IGHV3 family sequences with CDR3 length 10-15
curl -H "X-API-Key: your-secret-api-key-here" \
     "http://localhost:8000/search/sequences?vh_gene_family=IGHV3&cdr3_min_length=10&cdr3_max_length=15&limit=50"

# All IgG1 sequences in a specific clone
curl -H "X-API-Key: your-secret-api-key-here" \
     "http://localhost:8000/search/sequences?isotype=IgG1&clone_id=clone_001"
```

### Get the aggregate report

```bash
curl -H "X-API-Key: your-secret-api-key-here" \
     http://localhost:8000/report
```

Example response:

```json
{
  "generated_at": "2024-03-15T14:22:01.234567",
  "clone_diversity": {
    "total_sequences": 4823,
    "total_clones": 312,
    "shannon_diversity_index": 4.8721,
    "d50": 47,
    "top_clone_frequency_pct": 3.12
  },
  "top_constructs": [
    {
      "construct_name": "mAb-001-IgG1",
      "expression_system": "Expi293",
      "mean_yield_mg_l": 487.3,
      "n_runs": 6
    }
  ],
  "assay_summary": [
    {
      "assay_type": "SPR",
      "total": 89,
      "passed": 74,
      "pass_rate_pct": 83.15,
      "median_kd_nm": 2.34
    }
  ]
}
```

---

## Report Endpoint — Statistical Definitions

### Shannon Diversity Index (H)

The Shannon entropy of the clonal repertoire:

```
H = -Σ p_i × ln(p_i)
```

where `p_i` = reads for clone `i` / total reads across all clones.

- H = 0 → all reads belong to one clone (monoclonal)
- Higher H → more evenly distributed repertoire (polyclonal)
- Typical healthy donor: H ≈ 4–7

### D50

The minimum number of clones required to account for ≥ 50% of total sequencing reads, with clones sorted by abundance (highest first). A lower D50 indicates a more oligoclonal repertoire dominated by a few expanded clones.

---

## Interactive Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
