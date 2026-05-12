"""
data_processor.py — PySpark batch processing for preclinical assay data.

Demonstrates DataFrame operations, aggregations, window functions, and
partitioned Parquet output. Falls back to pandas when PySpark is unavailable.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42): np.random.seed(s)
    PALETTES = {"young": "#2196F3", "aged": "#F44336", "accent": "#4CAF50"}

try:
    from pyspark.sql import SparkSession, DataFrame as SparkDF
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, IntegerType
    )
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic assay CSV generator (used by both Spark and pandas paths)
# ──────────────────────────────────────────────────────────────────────────────

ASSAY_TYPES = ["biochemical", "cellular", "dmpk", "safety"]
ASSAY_IDS   = ["ASY001", "ASY002", "ASY003", "ASY004", "ASY005"]
ASSAY_TYPE_MAP = {
    "ASY001": "biochemical",
    "ASY002": "biochemical",
    "ASY003": "cellular",
    "ASY004": "dmpk",
    "ASY005": "safety",
}


def generate_assay_csv(path: str, n: int = 1000, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        asy_id   = str(rng.choice(ASSAY_IDS))
        asy_type = ASSAY_TYPE_MAP[asy_id]
        ic50_nm  = float(10 ** rng.uniform(-1, 5)) if asy_type in ("biochemical", "cellular") else None
        value    = ic50_nm if ic50_nm else float(rng.uniform(0, 200))
        rows.append({
            "result_id":    f"RES{i:06d}",
            "compound_id":  f"CPD{rng.integers(1, 101):04d}",
            "assay_id":     asy_id,
            "assay_type":   asy_type,
            "value":        round(value, 3),
            "ic50_nm":      round(ic50_nm, 3) if ic50_nm else None,
            "unit":         "nM" if ic50_nm else "µL/min/mg",
            "qc_flag":      str(rng.choice(["pass", "pass", "pass", "review"], p=[0.8, 0.1, 0.05, 0.05])),
            "run_date":     f"2024-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}",
            "project_id":   str(rng.choice(["PRJ-CDK", "PRJ-KRAS", "PRJ-EGFR"])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# PySpark pipeline
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA = None  # lazy-init after HAS_SPARK check


def get_spark_schema():
    if not HAS_SPARK:
        return None
    return StructType([
        StructField("result_id",   StringType(),  False),
        StructField("compound_id", StringType(),  False),
        StructField("assay_id",    StringType(),  False),
        StructField("assay_type",  StringType(),  False),
        StructField("value",       DoubleType(),  False),
        StructField("ic50_nm",     DoubleType(),  True),
        StructField("unit",        StringType(),  False),
        StructField("qc_flag",     StringType(),  False),
        StructField("run_date",    StringType(),  False),
        StructField("project_id",  StringType(),  False),
    ])


def create_spark_session(app_name: str = "PreclinicalPipeline") -> "SparkSession":
    return (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def run_spark_pipeline(csv_path: str, out_dir: str = "spark_output") -> Dict:
    spark = create_spark_session()
    schema = get_spark_schema()

    # Load
    df = spark.read.csv(csv_path, header=True, schema=schema)
    n_total = df.count()

    # Filter: QC pass only
    df_pass = df.filter(F.col("qc_flag") == "pass")
    n_pass  = df_pass.count()

    # Log10-transform IC50
    df_pass = df_pass.withColumn(
        "log10_ic50",
        F.when(F.col("ic50_nm").isNotNull() & (F.col("ic50_nm") > 0),
               F.log10(F.col("ic50_nm"))).otherwise(F.lit(None))
    )

    # Aggregation: per compound, per assay_type
    agg_df = (
        df_pass
        .groupBy("compound_id", "assay_type")
        .agg(
            F.count("result_id").alias("n_results"),
            F.min("ic50_nm").alias("min_ic50_nm"),
            F.mean("log10_ic50").alias("mean_log10_ic50"),
            F.stddev("log10_ic50").alias("std_log10_ic50"),
        )
    )

    # Window function: rank compounds within each assay_type by min IC50
    w = Window.partitionBy("assay_type").orderBy(F.col("min_ic50_nm").asc_nulls_last())
    ranked_df = agg_df.withColumn("rank_in_assay", F.rank().over(w))

    # Project-level summary
    project_summary = (
        df_pass
        .groupBy("project_id")
        .agg(
            F.countDistinct("compound_id").alias("n_compounds"),
            F.count("result_id").alias("n_results"),
            F.min("ic50_nm").alias("best_ic50_nm"),
        )
        .orderBy("best_ic50_nm")
    )

    # Write partitioned Parquet
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, "results_by_assay_type")
    (
        df_pass
        .repartition("assay_type")
        .write
        .mode("overwrite")
        .partitionBy("assay_type")
        .parquet(parquet_path)
    )

    # Collect summaries to Python
    ranked_pd  = ranked_df.toPandas()
    project_pd = project_summary.toPandas()
    spark.stop()

    return {
        "n_total": n_total,
        "n_pass":  n_pass,
        "ranked":  ranked_pd,
        "project_summary": project_pd,
        "parquet_path": parquet_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pandas fallback (same logic, no Spark)
# ──────────────────────────────────────────────────────────────────────────────

def run_pandas_pipeline(csv_path: str, out_dir: str = "pandas_output") -> Dict:
    df = pd.read_csv(csv_path)
    n_total = len(df)

    df_pass = df[df["qc_flag"] == "pass"].copy()
    n_pass  = len(df_pass)

    df_pass["log10_ic50"] = np.where(
        df_pass["ic50_nm"].notna() & (df_pass["ic50_nm"] > 0),
        np.log10(df_pass["ic50_nm"].fillna(np.nan)),
        np.nan,
    )

    agg_df = (
        df_pass.groupby(["compound_id", "assay_type"])
        .agg(
            n_results      = ("result_id", "count"),
            min_ic50_nm    = ("ic50_nm", "min"),
            mean_log10_ic50= ("log10_ic50", "mean"),
            std_log10_ic50 = ("log10_ic50", "std"),
        )
        .reset_index()
    )

    # Rank within assay_type
    agg_df["rank_in_assay"] = (
        agg_df.groupby("assay_type")["min_ic50_nm"]
        .rank(method="min", na_option="bottom")
        .astype(int)
    )

    project_summary = (
        df_pass.groupby("project_id")
        .agg(
            n_compounds  = ("compound_id", "nunique"),
            n_results    = ("result_id", "count"),
            best_ic50_nm = ("ic50_nm", "min"),
        )
        .reset_index()
        .sort_values("best_ic50_nm")
    )

    # Parquet output (partitioned by assay_type)
    os.makedirs(out_dir, exist_ok=True)
    for at, grp in df_pass.groupby("assay_type"):
        part_dir = os.path.join(out_dir, f"assay_type={at}")
        os.makedirs(part_dir, exist_ok=True)
        grp.to_parquet(os.path.join(part_dir, "part-0.parquet"), index=False)

    return {
        "n_total": n_total,
        "n_pass":  n_pass,
        "ranked":  agg_df,
        "project_summary": project_summary,
        "parquet_path": out_dir,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_processing_summary(
    result: Dict, out_path: str = "figures/spark_summary.png"
) -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#FAFAFA")

    # Panel 1: Results per assay type
    ax = axes[0]
    ranked = result["ranked"]
    per_type = ranked.groupby("assay_type")["n_results"].sum().sort_values(ascending=False)
    colors = [PALETTES.get("young", "#2196F3"),
              PALETTES.get("aged", "#F44336"),
              PALETTES.get("accent", "#4CAF50"),
              "#FF9800"]
    ax.bar(per_type.index, per_type.values,
           color=colors[:len(per_type)], edgecolor="white")
    ax.set_xlabel("Assay Type")
    ax.set_ylabel("QC-Pass Results")
    ax.set_title("Results by Assay Type")
    ax.set_facecolor("#F5F5F5")

    # Panel 2: Project hit rates
    ax = axes[1]
    ps = result["project_summary"]
    ax.barh(ps["project_id"], ps["n_compounds"],
            color=PALETTES.get("young", "#2196F3"), edgecolor="white")
    ax.set_xlabel("Unique Compounds")
    ax.set_title("Compounds per Project")
    ax.set_facecolor("#F5F5F5")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    import tempfile
    csv_path = os.path.join(tempfile.mkdtemp(), "assay_data.csv")
    generate_assay_csv(csv_path, n=1000, seed=42)
    print(f"Generated CSV: {csv_path}  (1000 records)")

    if HAS_SPARK:
        print("Running PySpark pipeline …")
        result = run_spark_pipeline(csv_path, out_dir="spark_output")
        engine = "PySpark"
    else:
        print("PySpark not available — running pandas fallback …")
        result = run_pandas_pipeline(csv_path, out_dir="pandas_output")
        engine = "pandas"

    print(f"\n── {engine} Pipeline Results ──")
    print(f"  Total records : {result['n_total']}")
    print(f"  QC pass       : {result['n_pass']}  "
          f"({result['n_pass']/result['n_total']:.1%})")
    print(f"  Parquet output: {result['parquet_path']}")
    print("\nProject summary:")
    print(result["project_summary"].to_string(index=False))
    print(f"\nTop-ranked compounds (biochemical IC50):")
    bm = result["ranked"]
    bm_bio = bm[bm["assay_type"] == "biochemical"].sort_values("rank_in_assay")
    print(bm_bio.head(5)[["compound_id", "min_ic50_nm", "rank_in_assay"]].to_string(index=False))

    os.makedirs("figures", exist_ok=True)
    img = plot_processing_summary(result, "figures/spark_summary.png")
    print(f"\nSummary plot → {img}")
