"""
schema_design.py — Relational SQLite schema for a preclinical research database.

Star schema: compounds + assays + biological_entities → results (fact table).
Demonstrates SQL DDL, data modelling, advanced queries, and synthetic data population.
"""

from __future__ import annotations

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed
except ImportError:
    def set_global_seed(s=42): np.random.seed(s)

DB_PATH = ":memory:"   # use file path e.g. "preclinical.db" to persist

# ──────────────────────────────────────────────────────────────────────────────
# DDL — table definitions
# ──────────────────────────────────────────────────────────────────────────────

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS compounds (
    compound_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    smiles          TEXT,
    mw              REAL,
    logp            REAL,
    hbd             INTEGER,
    hba             INTEGER,
    tpsa            REAL,
    chebi_id        TEXT,
    chebi_class     TEXT,
    project_id      TEXT,
    registered_by   TEXT,
    registered_date TEXT,
    status          TEXT DEFAULT 'active'   -- active | deprecated | on_hold
);

CREATE TABLE IF NOT EXISTS assays (
    assay_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    assay_type      TEXT NOT NULL,          -- biochemical | cellular | dmpk | safety
    target          TEXT,
    counter_target  TEXT,
    readout         TEXT,                   -- IC50 | % inhibition | Clint | hERG_IC50
    unit            TEXT,
    protocol_ref    TEXT,
    species         TEXT DEFAULT 'human',
    validated       INTEGER DEFAULT 1       -- 0/1
);

CREATE TABLE IF NOT EXISTS biological_entities (
    entity_id       TEXT PRIMARY KEY,
    gene_symbol     TEXT NOT NULL,
    gene_name       TEXT,
    uniprot_id      TEXT,
    organism        TEXT DEFAULT 'Homo sapiens',
    go_molecular    TEXT,                   -- JSON list of GO:XXXXXXX
    go_biological   TEXT,
    go_cellular     TEXT,
    pathway_ids     TEXT,                   -- JSON list of KEGG/Reactome IDs
    disease_assoc   TEXT                    -- JSON list
);

CREATE TABLE IF NOT EXISTS results (
    result_id       TEXT PRIMARY KEY,
    compound_id     TEXT NOT NULL REFERENCES compounds(compound_id),
    assay_id        TEXT NOT NULL REFERENCES assays(assay_id),
    value           REAL NOT NULL,
    unit            TEXT NOT NULL,
    qualifier       TEXT DEFAULT '=',       -- = < > ~
    ic50_nm         REAL,
    hill_slope      REAL,
    r_squared       REAL,
    n_replicates    INTEGER DEFAULT 1,
    cv_pct          REAL,
    operator        TEXT,
    run_date        TEXT,
    batch_id        TEXT,
    qc_flag         TEXT DEFAULT 'pass'     -- pass | fail | review
);

CREATE INDEX IF NOT EXISTS idx_results_compound ON results(compound_id);
CREATE INDEX IF NOT EXISTS idx_results_assay    ON results(assay_id);
CREATE INDEX IF NOT EXISTS idx_results_ic50     ON results(ic50_nm);
CREATE INDEX IF NOT EXISTS idx_compounds_project ON compounds(project_id);
"""

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ──────────────────────────────────────────────────────────────────────────────

CHEBI_CLASSES = [
    ("CHEBI:35222", "inhibitor"),
    ("CHEBI:38637", "kinase inhibitor"),
    ("CHEBI:33281", "antimicrobial agent"),
    ("CHEBI:35176", "antineoplastic agent"),
    ("CHEBI:23888", "drug"),
]

ASSAY_CATALOGUE = [
    ("ASY001", "CDK4/CyclinD1 Biochemical IC50",  "biochemical", "CDK4",  "CDK2",  "IC50",  "nM"),
    ("ASY002", "CDK2/CyclinA Biochemical IC50",    "biochemical", "CDK2",  None,    "IC50",  "nM"),
    ("ASY003", "MCF7 Cell Viability (72h)",        "cellular",    "CDK4",  None,    "GI50",  "nM"),
    ("ASY004", "Microsomal Stability (HLM)",       "dmpk",        None,    None,    "Clint", "µL/min/mg"),
    ("ASY005", "hERG Patch Clamp IC50",            "safety",      "hERG",  None,    "IC50",  "µM"),
]

BIOLOGICAL_ENTITIES = [
    ("ENT001", "CDK4",  "Cyclin-dependent kinase 4", "P11802",
     '["GO:0004672","GO:0016301"]', '["GO:0007049","GO:0006977"]', '["GO:0005737"]',
     '["hsa04110","R-HSA-69278"]',  '["glioblastoma","breast cancer"]'),
    ("ENT002", "CDK2",  "Cyclin-dependent kinase 2", "P24941",
     '["GO:0004672","GO:0016301"]', '["GO:0007049","GO:0045786"]', '["GO:0005737"]',
     '["hsa04110","R-HSA-69278"]',  '["ovarian cancer"]'),
    ("ENT003", "KCNH2", "Potassium voltage-gated channel (hERG)", "Q12809",
     '["GO:0005245"]',              '["GO:0006813","GO:0060307"]', '["GO:0005886"]',
     '["hsa04022"]',                '["long QT syndrome"]'),
    ("ENT004", "RB1",   "Retinoblastoma protein",    "P06400",
     '["GO:0003714"]',              '["GO:0045736","GO:0007050"]', '["GO:0005634"]',
     '["hsa04110"]',                '["retinoblastoma","lung cancer"]'),
    ("ENT005", "CCND1", "Cyclin D1",                 "P24385",
     '["GO:0016538"]',              '["GO:0007049","GO:0045786"]', '["GO:0005737"]',
     '["hsa04110","R-HSA-69278"]',  '["breast cancer","mantle cell lymphoma"]'),
]

_SMILES_POOL = [
    "CC1=NC(=CC(=O)N1)C2=CC=C(C=C2)NC(=O)C3=CC=NC=C3",
    "O=C1NC(=O)C(=C1/C=C/c2ccc(O)cc2)C(=O)O",
    "CC(C)CC1=CC(=CC(=C1)C(C)C)C(C)C",
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "C1=CC(=CC=C1N)S(=O)(=O)N",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "C(CN)c1ccc(cc1)O",
    "O=C(O)c1ccc(cc1)N",
    "CC(=O)Nc1ccc(cc1)O",
    "c1ccc(cc1)C(=O)O",
    "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)CCN1",
    "C[C@@H]1CC[C@H]2C[C@@H](/C(=C/[C@@H]3CC(=O)[C@H](C/C=C/[C@H]1OC(=O)[C@@H]2C)O3)C)OC(=O)CC(CC(=O)O)CC(=O)O",
    "O=C(O)[C@@H](N)Cc1ccc(O)cc1",
    "CC(=O)c1ccc(cc1)N",
    "NC(=O)c1cccnc1",
    "O=C(O)c1cccc(c1)O",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "CC(C)(C)c1ccc(cc1)O",
    "O=C(O)CCc1ccccc1",
    "CC1=CC(=O)c2ccccc2C1=O",
    "c1ccc(cc1)CN",
    "O=C(O)[C@@H](N)CC(=O)O",
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "CCOC(=O)c1ccc(cc1)N",
    "CC1=CC=C(C=C1)C(=O)O",
    "NC(=O)c1cccc(c1)O",
    "Clc1ccc(cc1)C(=O)O",
    "CC(=O)Nc1cccc(c1)O",
    "O=C(O)c1ccc(O)c(O)c1",
    "CC(C)c1ccc(cc1)C(C)C",
    "c1cc(ccc1N)N",
    "O=S(=O)(O)c1ccccc1",
    "CC(C)(C)OC(=O)NC(Cc1ccccc1)C(=O)O",
    "CN(C)c1ccc(cc1)N=Nc2ccc(cc2)S(=O)(=O)O",
    "O=C(O)c1ccc(cc1)Cl",
    "c1ccc(cc1)O",
    "O=C(O)C(O)(CC(=O)O)CC(=O)O",
    "OCC1OC(O)C(O)C(O)C1O",
    "O=C1CCCCC1",
    "c1ccc(cc1)CC(=O)O",
    "O=C(O)c1cccc(N)c1",
    "CC(=O)OCC1OC(OC2C(O)C(O)C(O)OC2CO)C(O)C(O)C1O",
    "NC1=NC(=O)C2=C(N1)N=CN2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
    "O=C(O)[C@H](O)[C@@H](O)C(=O)O",
    "CCCCCCCCCCCCCCCC(=O)O",
    "O=C(O)CCCCCCC(=O)O",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "c1cnc(nc1)N",
    "O=C(O)CC(=O)O",
    "OC1=CC=CC=C1C(=O)O",
]


def _lipinski(smiles: str, rng: np.random.Generator) -> Tuple[float, float, int, int, float]:
    mw   = rng.uniform(200, 600)
    logp = rng.uniform(-1, 6)
    hbd  = int(rng.integers(0, 6))
    hba  = int(rng.integers(0, 11))
    tpsa = rng.uniform(20, 160)
    return mw, logp, hbd, hba, tpsa


def generate_compounds(n: int = 50, seed: int = 42) -> List[dict]:
    rng = np.random.default_rng(seed)
    projects = ["PRJ-CDK", "PRJ-KRAS", "PRJ-EGFR", "PRJ-BRD4"]
    operators = ["analyst_01", "analyst_02", "analyst_03"]
    records = []
    for i in range(n):
        smiles = _SMILES_POOL[i % len(_SMILES_POOL)]
        mw, logp, hbd, hba, tpsa = _lipinski(smiles, rng)
        chebi_id, chebi_class = CHEBI_CLASSES[rng.integers(0, len(CHEBI_CLASSES))]
        records.append({
            "compound_id":     f"CPD{i+1:04d}",
            "name":            f"Compound-{i+1:03d}",
            "smiles":          smiles,
            "mw":              round(mw, 2),
            "logp":            round(logp, 2),
            "hbd":             hbd,
            "hba":             hba,
            "tpsa":            round(tpsa, 1),
            "chebi_id":        chebi_id,
            "chebi_class":     chebi_class,
            "project_id":      str(rng.choice(projects)),
            "registered_by":   str(rng.choice(operators)),
            "registered_date": f"2024-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}",
            "status":          "active",
        })
    return records


def generate_results(compounds: List[dict], seed: int = 42) -> List[dict]:
    rng = np.random.default_rng(seed)
    records = []
    operators = ["screener_A", "screener_B"]
    for cpd in compounds:
        for asy_id, asy_name, asy_type, target, _, readout, unit in ASSAY_CATALOGUE:
            # Not every compound is tested in every assay
            if rng.random() < 0.15:
                continue

            if asy_type == "biochemical":
                ic50_nm = float(10 ** rng.uniform(-1, 4))
                value   = ic50_nm
            elif asy_type == "cellular":
                ic50_nm = float(10 ** rng.uniform(0, 5))
                value   = ic50_nm
            elif asy_type == "dmpk":
                ic50_nm = None
                value   = float(rng.uniform(0, 200))   # Clint µL/min/mg
            else:   # safety hERG
                ic50_nm = float(10 ** rng.uniform(2, 5))   # µM range → convert
                value   = ic50_nm / 1000   # store as µM

            hill   = float(rng.uniform(0.5, 2.5))
            rsq    = float(rng.uniform(0.88, 0.999))
            cv_pct = float(rng.uniform(1, 25))
            qc     = "pass" if cv_pct < 20 else "review"

            records.append({
                "result_id":    f"RES{len(records)+1:06d}",
                "compound_id":  cpd["compound_id"],
                "assay_id":     asy_id,
                "value":        round(value, 3),
                "unit":         unit,
                "qualifier":    "=",
                "ic50_nm":      round(ic50_nm, 3) if ic50_nm else None,
                "hill_slope":   round(hill, 3),
                "r_squared":    round(rsq, 4),
                "n_replicates": int(rng.integers(2, 4)),
                "cv_pct":       round(cv_pct, 1),
                "operator":     str(rng.choice(operators)),
                "run_date":     f"2024-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}",
                "batch_id":     f"BATCH{rng.integers(1,20):03d}",
                "qc_flag":      qc,
            })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# DB operations
# ──────────────────────────────────────────────────────────────────────────────

def create_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    return conn


def populate_database(conn: sqlite3.Connection, seed: int = 42) -> dict:
    compounds = generate_compounds(n=50, seed=seed)
    results   = generate_results(compounds, seed=seed)

    conn.executemany(
        "INSERT OR IGNORE INTO compounds VALUES "
        "(:compound_id,:name,:smiles,:mw,:logp,:hbd,:hba,:tpsa,"
        ":chebi_id,:chebi_class,:project_id,:registered_by,:registered_date,:status)",
        compounds,
    )

    conn.executemany(
        "INSERT OR IGNORE INTO assays (assay_id,name,assay_type,target,"
        "counter_target,readout,unit,protocol_ref) VALUES (?,?,?,?,?,?,?,?)",
        [(a[0], a[1], a[2], a[3], a[4], a[5], a[6], f"SOP-{a[0]}")
         for a in ASSAY_CATALOGUE],
    )

    conn.executemany(
        "INSERT OR IGNORE INTO biological_entities VALUES (?,?,?,?,?,?,?,?,?,?)",
        BIOLOGICAL_ENTITIES,
    )

    conn.executemany(
        "INSERT OR IGNORE INTO results VALUES "
        "(:result_id,:compound_id,:assay_id,:value,:unit,:qualifier,"
        ":ic50_nm,:hill_slope,:r_squared,:n_replicates,:cv_pct,"
        ":operator,:run_date,:batch_id,:qc_flag)",
        results,
    )
    conn.commit()
    return {"compounds": len(compounds), "assays": len(ASSAY_CATALOGUE),
            "results": len(results), "entities": len(BIOLOGICAL_ENTITIES)}


# ──────────────────────────────────────────────────────────────────────────────
# Example analytical queries
# ──────────────────────────────────────────────────────────────────────────────

QUERIES = {
    "top10_cdk4_ic50": """
        SELECT c.compound_id, c.name, c.mw, c.logp, r.ic50_nm,
               r.hill_slope, r.r_squared, r.qc_flag
        FROM   results r
        JOIN   compounds c ON r.compound_id = c.compound_id
        WHERE  r.assay_id = 'ASY001'
          AND  r.qc_flag  = 'pass'
          AND  r.ic50_nm IS NOT NULL
        ORDER  BY r.ic50_nm ASC
        LIMIT  10
    """,
    "selectivity_ratio": """
        SELECT c.compound_id, c.name,
               r1.ic50_nm AS cdk4_ic50,
               r2.ic50_nm AS cdk2_ic50,
               ROUND(r2.ic50_nm / r1.ic50_nm, 1) AS selectivity_ratio
        FROM   results r1
        JOIN   results r2 ON r1.compound_id = r2.compound_id
        JOIN   compounds c ON r1.compound_id = c.compound_id
        WHERE  r1.assay_id = 'ASY001'
          AND  r2.assay_id = 'ASY002'
          AND  r1.ic50_nm IS NOT NULL
          AND  r2.ic50_nm IS NOT NULL
          AND  r1.qc_flag = 'pass'
          AND  r2.qc_flag = 'pass'
        ORDER  BY selectivity_ratio DESC
        LIMIT  10
    """,
    "assay_coverage": """
        SELECT a.assay_id, a.name, a.assay_type,
               COUNT(DISTINCT r.compound_id) AS compounds_tested,
               COUNT(CASE WHEN r.qc_flag='pass' THEN 1 END) AS pass_results,
               ROUND(AVG(r.ic50_nm),1) AS mean_ic50_nm,
               ROUND(MIN(r.ic50_nm),2) AS min_ic50_nm
        FROM   assays a
        LEFT   JOIN results r ON a.assay_id = r.assay_id
        GROUP  BY a.assay_id
        ORDER  BY compounds_tested DESC
    """,
    "lipinski_vs_potency": """
        SELECT c.mw, c.logp, c.tpsa,
               r.ic50_nm,
               CASE WHEN r.ic50_nm < 100 THEN 'potent'
                    WHEN r.ic50_nm < 1000 THEN 'moderate'
                    ELSE 'weak' END AS potency_class
        FROM   results r
        JOIN   compounds c ON r.compound_id = c.compound_id
        WHERE  r.assay_id = 'ASY001'
          AND  r.qc_flag  = 'pass'
          AND  r.ic50_nm IS NOT NULL
    """,
    "project_summary": """
        SELECT c.project_id,
               COUNT(DISTINCT c.compound_id)  AS n_compounds,
               COUNT(DISTINCT r.result_id)    AS n_results,
               ROUND(MIN(r.ic50_nm),2)        AS best_ic50_nm,
               ROUND(AVG(r.ic50_nm),0)        AS mean_ic50_nm
        FROM   compounds c
        LEFT   JOIN results r ON c.compound_id = r.compound_id
                              AND r.assay_id   = 'ASY001'
        GROUP  BY c.project_id
        ORDER  BY best_ic50_nm ASC NULLS LAST
    """,
}


def run_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(QUERIES[name], conn)


def print_schema_info(conn: sqlite3.Connection) -> None:
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print("\nDatabase schema:")
    for (tbl,) in tables:
        cols = conn.execute(f"PRAGMA table_info({tbl})").fetchall()
        n    = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:<25} ({n:>4} rows)  cols: {', '.join(c[1] for c in cols)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    print("Creating preclinical database …")
    conn  = create_database()
    stats = populate_database(conn)
    print(f"  Inserted: {stats}")

    print_schema_info(conn)

    print("\n── Top 10 CDK4 inhibitors ──")
    df = run_query(conn, "top10_cdk4_ic50")
    print(df.to_string(index=False))

    print("\n── Selectivity ratio (CDK4/CDK2) ──")
    df = run_query(conn, "selectivity_ratio")
    print(df.to_string(index=False))

    print("\n── Assay coverage summary ──")
    df = run_query(conn, "assay_coverage")
    print(df.to_string(index=False))

    print("\n── Project summary ──")
    df = run_query(conn, "project_summary")
    print(df.to_string(index=False))

    conn.close()
