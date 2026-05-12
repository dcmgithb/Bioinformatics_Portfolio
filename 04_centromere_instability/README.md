# Project 04 · Centromere Genomic Instability

## Biological Question
How do centromeric satellite DNA, CENP-A occupancy, and
α-satellite copy number change with age, and how does centromere
instability drive genomic instability hallmarks in immune cells?

This project directly addresses the **CenAGE** programme's core hypothesis.

## Why Centromeres Matter in Ageing
- Centromeres are repeat-rich regions essential for faithful chromosome segregation
- Age-associated loss of CENP-A → chromosome mis-segregation → micronuclei
- Micronuclei activate cGAS-STING → innate immune signalling → SASP/inflammageing
- α-satellite copy number variation is heritable and changes with age

## Datasets
- **WGS**: PRJNA680893 — Whole-genome sequencing, long-lived individuals vs. controls
- **ChIP-seq**: GSE153626 — CENP-A/B ChIP-seq in young vs. aged fibroblasts
- **ATAC-seq**: GSE174531 — Chromatin accessibility at centromeric regions

## Pipeline Overview
```
WGS FASTQ → BWA-MEM2 alignment → samtools sort/index
    → Satellite repeat quantification (RepeatMasker + mSMRT)
    → α-satellite CNV (mosdepth + CNVkit)
    → Centromere chromatin state (ATAC-seq signal enrichment)
    → CENP-A/B ChIP occupancy (deeptools)
    → Micronuclei integration with expression data
    → Correlation: CIN score vs. SASP, senescence markers
```

## Files
| File | Description |
|------|-------------|
| `centromere_repeat_analysis.R` | α-satellite quantification, CNV, CIN score, centenarian vs. control, SASP integration |
| `cenpa_chip_analysis.py` | CENP-A occupancy from ChIP-seq (deeptools wrapper + signal simulation) |
