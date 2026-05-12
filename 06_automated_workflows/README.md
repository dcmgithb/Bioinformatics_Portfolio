# Project 06 · Automated Sequencing Workflows

## Why Automation Matters
Raw sequencing data processing is the foundation of all downstream analyses.
Reproducible, containerised pipelines ensure:
- Results are identical regardless of who runs them
- Parameter changes are tracked in version control
- HPC/cloud deployment without code changes
- Automatic provenance and audit trails

## Workflows Included

### Snakemake (Python-based, rule-driven)
Complete RNA-seq pipeline: FASTQ → count matrix → MultiQC report
```
FastQC → Trim Galore → STAR → featureCounts → DESeq2-ready matrix
```

### Nextflow (DSL2, channel-based)
Cloud-native RNA-seq pipeline compatible with AWS/Google Cloud Batch
```
SRA download → FastQC → HISAT2/STAR → Samtools → HTSeq/featureCounts → MultiQC
```

## Files
| File | Description |
|------|-------------|
| `Snakefile` | Complete Snakemake RNA-seq workflow |
| `rnaseq_pipeline.nf` | Nextflow DSL2 RNA-seq pipeline |
| `config.yaml` | Snakemake configuration file |
| `nextflow.config` | Nextflow resource configuration |
| `envs/rnaseq.yaml` | Conda environment specification |
| `Dockerfile` | Container for reproducible deployment |
| `scripts/geo_to_sra.sh` | GEO → SRA download automation |
| `scripts/run_pipeline.sh` | HPC SLURM submission script |

## Quick Start

```bash
# Snakemake (local, 8 cores)
snakemake --cores 8 --use-conda

# Snakemake dry-run
snakemake --dryrun --printshellcmds

# Nextflow (local)
nextflow run rnaseq_pipeline.nf -profile local

# Nextflow (AWS)
nextflow run rnaseq_pipeline.nf -profile aws

# Docker
docker build -t rnaseq-pipeline .
docker run -v $(pwd):/data rnaseq-pipeline snakemake --cores 4
```

## HPC (SLURM) Submission
```bash
sbatch scripts/run_pipeline.sh
```
