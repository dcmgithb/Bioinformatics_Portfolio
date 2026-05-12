# Shared Utilities

Common tools used across all portfolio projects.

## `geo_data_retrieval.R`
Standardised GEO/SRA data access:
- `download_geo_series()` — download any GEO accession with metadata
- `get_sra_metadata()` — fetch SRR accessions via NCBI Entrez
- `search_aging_geo()` — keyword search for aging datasets on GEO
- `load_msigdb_genesets()` — load any MSigDB collection as a named list
- `process_geo_for_deseq2()` — one-shot GEO → DESeq2-ready counts

## `common_functions.R`
Shared R utilities:
- `theme_aging()` — consistent ggplot2 theme
- `aging_palettes` — standardised colour palette list
- `classify_de_genes()` — Up/Down/NS classification
- `filter_low_counts()` — gene pre-filtering
- `calc_cpm() / calc_tpm()` — normalisation functions
- `annotated_heatmap()` — pheatmap wrapper
- `init_analysis()` / `save_session_info()` — reproducibility helpers

## `common_functions.py`
Python equivalents:
- `PALETTES` — shared colour palettes
- `classify_de_genes()` — pandas DE classification
- `filter_low_counts()` — count matrix filtering
- `calc_cpm() / calc_tpm()` — normalisation
- `beta_to_mvalue()` — methylation transformation
- `volcano_plot()` — publication-ready volcano
- `correlation_heatmap()` — styled correlation matrix
- `set_global_seed()` — reproducibility seed setting
- `build_sra_download_script()` — generate SRA download scripts

## Usage

```r
# R
source("utils/common_functions.R")
source("utils/geo_data_retrieval.R")

data <- download_geo_series("GSE65907")
hallmarks <- load_msigdb_genesets("Homo sapiens", "H")
```

```python
# Python
from utils.common_functions import (
    classify_de_genes, filter_low_counts, volcano_plot, set_global_seed
)
set_global_seed(42)
```
