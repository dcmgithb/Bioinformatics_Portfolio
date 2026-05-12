#!/usr/bin/env bash
#SBATCH --job-name=rnaseq_aging
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=aging_proj

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PIPELINE_DIR}"

mkdir -p logs

# Load modules (adjust to your HPC module system)
module load anaconda3 2>/dev/null || true

echo "Starting RNA-seq pipeline: $(date)"
echo "Pipeline dir: ${PIPELINE_DIR}"

snakemake \
    --cores 8 \
    --use-conda \
    --cluster "sbatch \
        --time={resources.time} \
        --mem={resources.mem_mb}M \
        --cpus-per-task={threads} \
        --output=logs/slurm_%j.out \
        --error=logs/slurm_%j.err" \
    --jobs 50 \
    --latency-wait 60 \
    --rerun-incomplete \
    --printshellcmds \
    "$@"

echo "Pipeline complete: $(date)"
