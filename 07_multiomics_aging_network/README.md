# Project 07 · Multi-omics Aging Network

## Biological Question
What are the master regulatory hubs of the aging transcriptome,
and how can we integrate transcriptomics, methylomics, and proteomics
into a unified systems-level view of immunosenescence?

## Approach: Three-layer Integration
```
Layer 1: WGCNA co-expression network (RNA-seq)
    → Hub gene identification per module
    → Module-trait correlations (age, CIN score, senescence)

Layer 2: MOFA2 multi-omics factor analysis
    → Latent factors jointly explaining transcriptome + methylome + proteome
    → Factor loadings reveal biological processes

Layer 3: Network integration (igraph/networkx)
    → Weighted protein-protein interaction (PPI) network
    → Overlay GWAS, eQTL, differential expression
    → Network resilience and hub vulnerability analysis
```

## Datasets
- **Transcriptomics**: GSE65907 (PBMC RNA-seq, young vs. aged)
- **Methylomics**: GSE40279 (blood 450K array)
- **Proteomics**: PXD014945 (PBMC proteomics, aging)

## Files
| File | Description |
|------|-------------|
| `wgcna_aging_network.R` | WGCNA co-expression network + module analysis |
| `network_integration.py` | Network construction, hub analysis, visualisation |
