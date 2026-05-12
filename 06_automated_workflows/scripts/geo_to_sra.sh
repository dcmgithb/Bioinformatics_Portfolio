#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# GEO → SRA Automated Download Script
# Fetches RNA-seq data from GEO using SRA Toolkit
#
# Usage:
#   ./geo_to_sra.sh GSE65907 8               # 8 parallel downloads
#   ./geo_to_sra.sh GSE174072                # default 4 parallel
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GEO_ACC="${1:?Usage: $0 <GEO_ACCESSION> [N_PARALLEL]}"
N_PARALLEL="${2:-4}"
export OUTDIR="data/raw_reads"
export GEO_ACC

echo "=================================="
echo "GEO → SRA Download Pipeline"
echo "Accession : ${GEO_ACC}"
echo "Parallel  : ${N_PARALLEL}"
echo "Output    : ${OUTDIR}"
echo "=================================="

mkdir -p "${OUTDIR}" logs

# ── Step 1: Get SRA run list from GEO ─────────────────────────────────────────

echo "[1/5] Fetching SRR list from GEO..."
# Using NCBI Entrez utilities
esearch -db gds -query "${GEO_ACC}[Accession]" | \
    elink -target sra | \
    efetch -format runinfo | \
    tail -n +2 | \
    cut -d ',' -f 1 | \
    grep -E "^SRR" > "logs/${GEO_ACC}_srr_list.txt"

N_RUNS=$(wc -l < "logs/${GEO_ACC}_srr_list.txt")
echo "Found ${N_RUNS} SRR runs"

# ── Step 2: Download SRA files ────────────────────────────────────────────────

echo "[2/5] Downloading SRA files (${N_PARALLEL} parallel)..."
cat "logs/${GEO_ACC}_srr_list.txt" | \
    xargs -I {} -P "${N_PARALLEL}" bash -c '
        SRR="{}"
        echo "  Downloading ${SRR}..."
        prefetch \
            --progress \
            --max-size 50G \
            --output-directory data/sra \
            "${SRR}" 2>> "logs/${SRR}_prefetch.log"
        echo "  ${SRR} done"
    '

# ── Step 3: Convert SRA → FASTQ (parallel-fastq-dump) ────────────────────────

echo "[3/5] Converting SRA → FASTQ..."
cat "logs/${GEO_ACC}_srr_list.txt" | \
    xargs -I {} -P "${N_PARALLEL}" bash -c '
        SRR="{}"
        echo "  Splitting ${SRR}..."
        parallel-fastq-dump \
            --sra-id "${SRR}" \
            --threads 4 \
            --split-files \
            --gzip \
            --outdir "${OUTDIR}" \
            2>> "logs/${SRR}_fastqdump.log"
        echo "  ${SRR} → FASTQ done"
    '

# ── Step 4: Rename files to sample-friendly names ─────────────────────────────

echo "[4/5] Fetching sample metadata from GEO..."
Rscript - <<'REOF'
suppressPackageStartupMessages({
    library(GEOquery)
    library(tidyverse)
})

geo_acc <- Sys.getenv("GEO_ACC", unset = commandArgs(trailingOnly=TRUE)[1])
if (is.na(geo_acc) || geo_acc == "") stop("GEO_ACC not set")

tryCatch({
    gse <- getGEO(geo_acc, GSEMatrix = TRUE)
    meta <- pData(phenoData(gse[[1]]))
    meta %>%
        select(geo_accession, title, contains("source"), contains("age"), contains("group")) %>%
        write.csv(paste0("logs/", geo_acc, "_metadata.csv"), row.names = FALSE)
    message("Metadata saved: ", nrow(meta), " samples")
}, error = function(e) {
    message("GEO metadata fetch skipped: ", e$message)
})
REOF

# ── Step 5: Validate downloads ────────────────────────────────────────────────

echo "[5/5] Validating downloaded files..."
echo "Downloaded FASTQ files:"
ls -lh "${OUTDIR}"/*.fastq.gz 2>/dev/null | awk '{print $5, $9}' || echo "  No FASTQ files found yet"

echo ""
echo "=================================="
echo "Download complete!"
echo "FASTQ files : ${OUTDIR}/"
echo "Logs        : logs/"
echo ""
echo "Next step: run the pipeline"
echo "  snakemake --cores 8 --use-conda"
echo "=================================="
