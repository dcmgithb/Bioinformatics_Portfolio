# Data Quality Monitor

A scheduled data quality monitoring tool for the antibody discovery LIMS PostgreSQL database. Runs 12 configurable checks across four categories and produces a JSON report plus a human-readable terminal summary.

---

## Overview and Purpose

Raw LIMS data accumulates errors: missing CDR3 sequences, impossible gate percentages, negative yields, orphan records. This tool catches those issues automatically before they propagate into downstream analyses or reports.

It is designed to be run:
- As a daily cron job (see `cron.example`)
- As part of a CI/CD data pipeline
- Interactively during data import

---

## Check Descriptions

| Check Name | Category | Fail Level | What It Catches |
|---|---|---|---|
| `missing_cdr3` | sequence | **fail** | `antibody_sequences` rows with `cdr3_aa IS NULL` or empty string |
| `malformed_cdr3` | sequence | **fail** | CDR3 sequences with non-standard amino acid characters, or length outside configured min/max |
| `duplicate_sequences` | sequence | **warn** | Multiple rows sharing the same `content_hash` (duplicate ingestion) |
| `outlier_expression_yield` | expression | **warn** | Yield values outside Tukey IQR fence (Q1−1.5·IQR, Q3+1.5·IQR) per expression system |
| `negative_yield` | expression | **fail** | `yield_mg_l < 0` — physically impossible, indicates data entry error |
| `missing_purity` | expression | **warn** | More than 20% of `expression_results` rows have `purity_pct IS NULL` |
| `b_cell_gate_out_of_range` | flow | **warn** | `b_cell_gate_pct` outside configured min/max range (default 0.5–60%) |
| `gate_sum_exceeds_100` | flow | **fail** | `naive_b_pct + memory_b_pct + plasmablast_pct > 100.5%` (biologically impossible) |
| `qc_flag_mismatch` | flow | **warn** | Runs with `qc_pass=TRUE` but `b_cell_gate_pct < 1.0%` or `> 50.0%` |
| `orphan_samples` | integrity | **warn** | `b_cell_samples` with no associated `flow_cytometry_runs` |
| `orphan_sequences` | integrity | **warn** | `antibody_sequences` with no associated `assay_results` |
| `assay_without_sequence` | integrity | **fail** | `assay_results` referencing a `seq_id` not in `antibody_sequences` (RI violation) |

**Status levels:**
- `pass` — no issues found; exits with code 0
- `warn` — anomaly worth investigating; exits with code 0
- `fail` — critical data integrity issue; exits with code 1

---

## Configuration via Environment Variables

Copy `.env.example` to `.env` and set values for your environment:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | *(required)* | PostgreSQL connection string `postgresql://user:pass@host:port/dbname` |
| `CDR3_MIN_LENGTH` | `6` | Minimum valid CDR3 length (amino acids) |
| `CDR3_MAX_LENGTH` | `30` | Maximum valid CDR3 length (amino acids) |
| `B_CELL_GATE_MIN_PCT` | `0.5` | Minimum expected B-cell gate % |
| `B_CELL_GATE_MAX_PCT` | `60.0` | Maximum expected B-cell gate % |
| `GATE_SUM_MAX_PCT` | `100.0` | Maximum for subset gate sum (hard tolerance applied at 100.5%) |
| `YIELD_IQR_MULTIPLIER` | `1.5` | Tukey fence multiplier for yield outlier detection |
| `OUTPUT_DIR` | `.` | Directory for JSON report output |
| `REPORT_FILENAME` | `dqm_report.json` | Filename for the JSON report |

All variables can alternatively be set directly in the shell environment.

---

## How to Run

### Install dependencies

```bash
cd data-quality-monitor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CLI examples

```bash
# Run all 12 checks (default)
python run_monitor.py

# Run only sequence checks
python run_monitor.py --checks sequence

# Run flow checks and save report to /reports/
python run_monitor.py --checks flow --output-dir /reports

# Run integrity checks — print only, no file saved
python run_monitor.py --checks integrity --no-save

# Use a custom DATABASE_URL without a .env file
DATABASE_URL=postgresql://admin:secret@prod-db:5432/lims python run_monitor.py
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All checks passed or warned (no critical failures) |
| `1` | At least one check returned `fail` status |

---

## Cron Scheduling

See `cron.example` for ready-to-use cron entries. To install:

```bash
crontab -e
# Paste the relevant lines from cron.example
```

Example schedule:
- **Daily 06:00** — full check (`--checks all`)
- **Sunday 07:00** — weekly report saved to `/reports/`
- **Weekdays 09:00, 13:00, 17:00** — sequence-only check (no file output)

---

## Example JSON Output

```json
{
  "run_at": "2024-03-15T06:00:12.341Z",
  "database": "lims_db",
  "summary": {
    "total": 12,
    "pass": 9,
    "warn": 2,
    "fail": 1
  },
  "checks": [
    {
      "check_name": "missing_cdr3",
      "status": "fail",
      "affected_rows": 3,
      "message": "3 sequence(s) are missing CDR3 (cdr3_aa IS NULL or empty).",
      "run_at": "2024-03-15T06:00:12.341123",
      "details": [
        {"seq_id": "a1b2c3d4-...", "sample_id": "e5f6g7h8-..."},
        {"seq_id": "b2c3d4e5-...", "sample_id": "f6g7h8i9-..."}
      ]
    },
    {
      "check_name": "orphan_samples",
      "status": "warn",
      "affected_rows": 7,
      "message": "7 sample(s) have no associated flow cytometry runs.",
      "run_at": "2024-03-15T06:00:12.512345",
      "details": [
        {"sample_id": "aa11bb22-...", "donor_id": "cc33dd44-...", "collection_date": "2024-02-01"}
      ]
    },
    {
      "check_name": "outlier_expression_yield",
      "status": "warn",
      "affected_rows": 2,
      "message": "2 expression yield value(s) fall outside the Tukey IQR fence (multiplier = 1.5).",
      "run_at": "2024-03-15T06:00:12.623456",
      "details": [
        {
          "result_id": "r1r2r3r4-...",
          "expression_system": "Expi293",
          "yield_mg_l": 892.3,
          "q1": 120.5,
          "q3": 340.2,
          "lower_fence": -209.4,
          "upper_fence": 670.1
        }
      ]
    },
    {
      "check_name": "negative_yield",
      "status": "pass",
      "affected_rows": 0,
      "message": "No negative yield values detected.",
      "run_at": "2024-03-15T06:00:12.734567",
      "details": []
    }
  ]
}
```

---

## Example Human-Readable Summary Output

```
======================================================================
  DATA QUALITY MONITOR — REPORT SUMMARY
======================================================================
  Run at  : 2024-03-15T06:00:12.341Z
  Database: lims_db

  Checks run  : 12
  ✓ PASS        : 9
  ⚠ WARN        : 2
  ✗ FAIL        : 1

----------------------------------------------------------------------
  Check Results
----------------------------------------------------------------------
  ✗ [FAIL] missing_cdr3
          3 sequence(s) are missing CDR3 (cdr3_aa IS NULL or empty).
          Affected rows: 3

  ✓ [PASS] malformed_cdr3
          All CDR3 sequences pass character and length validation (6–30 AA).

  ⚠ [WARN] orphan_samples
          7 sample(s) have no associated flow cytometry runs.
          Affected rows: 7

  ✓ [PASS] gate_sum_exceeds_100
          No flow runs have subset gate percentages summing above 100.5%.

======================================================================
  RESULT: ✗ FAILED — 1 critical issue(s) require attention.
======================================================================
```

---

## Extending with Custom Checks

To add a new check:

1. Create a subclass of `BaseCheck` in the appropriate `checks/` module (or a new file):

```python
from __future__ import annotations
from typing import Any
from ..database import get_cursor
from .base import BaseCheck, CheckResult

class MyCustomCheck(BaseCheck):
    """Check for some domain-specific data quality issue."""

    name: str = "my_custom_check"

    def run(self) -> CheckResult:
        """Execute the check and return a CheckResult."""
        query = "SELECT col_a, col_b FROM some_table WHERE condition_is_bad = TRUE"
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass("No issues found.")

        details: list[dict[str, Any]] = [dict(row) for row in rows]
        return self._warn(
            affected_rows=len(rows),
            details=details,
            message=f"{len(rows)} rows have the bad condition.",
        )
```

2. Register it in `monitor/checks/__init__.py`:

```python
from .my_module import MyCustomCheck

ALL_CHECKS = [..., MyCustomCheck]
CHECK_GROUPS["all"] = ALL_CHECKS
CHECK_GROUPS["custom"] = [MyCustomCheck]  # optional new group
```

3. The check will now run automatically when `--checks all` is used, and is available as `--checks custom` if you added it to a group.
