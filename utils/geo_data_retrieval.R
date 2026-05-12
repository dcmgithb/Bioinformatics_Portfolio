################################################################################
# GEO Data Retrieval & Metadata Parsing
# Standardised tools for downloading and processing public GEO datasets
################################################################################

suppressPackageStartupMessages({
  library(GEOquery)
  library(tidyverse)
  library(rentrez)
  library(xml2)
  library(jsonlite)
})

# ─── 1. Download & Parse GEO Series ──────────────────────────────────────────

#' Download a GEO series and return standardised metadata + expression
#'
#' @param geo_acc  GEO accession (e.g. "GSE65907")
#' @param dest_dir Directory to save downloaded files
#' @param extract_counts Whether to extract count matrix from supplementary
#' @return List with $metadata and $expression
download_geo_series <- function(geo_acc, dest_dir = "data/geo",
                                 extract_counts = FALSE) {
  message("Downloading GEO series: ", geo_acc)
  dir.create(dest_dir, showWarnings = FALSE, recursive = TRUE)

  # Download series matrix
  gse <- getGEO(geo_acc,
                destdir    = dest_dir,
                GSEMatrix  = TRUE,
                AnnotGPL   = TRUE,
                getGPL     = TRUE)

  if (length(gse) == 0) stop("GSE not found: ", geo_acc)

  # If multiple platforms, take the first
  gse_obj  <- gse[[1]]
  meta     <- pData(phenoData(gse_obj))
  expr_mat <- exprs(gse_obj)

  # Parse sample characteristics columns
  char_cols <- grep("^characteristics_ch", colnames(meta), value = TRUE)
  if (length(char_cols) > 0) {
    parsed_chars <- parse_sample_characteristics(meta, char_cols)
    meta <- cbind(meta, parsed_chars)
  }

  # Download supplementary files (count matrices, etc.)
  if (extract_counts) {
    supp_files <- getGEOSuppFiles(geo_acc, baseDir = dest_dir)
    message("Supplementary files: ", paste(rownames(supp_files), collapse=", "))
  }

  message(sprintf("Downloaded: %d samples, %d features", ncol(expr_mat), nrow(expr_mat)))

  list(
    gse      = gse_obj,
    metadata = meta,
    expression = expr_mat
  )
}

#' Parse GEO sample characteristics (key:value format)
parse_sample_characteristics <- function(meta, char_cols) {
  result <- data.frame(row.names = rownames(meta))

  for (col in char_cols) {
    values <- meta[[col]]
    # Extract key
    key <- gsub(":.*", "", values[!is.na(values)][1]) %>%
           gsub("\\s+", "_", .) %>%
           tolower()
    # Extract value
    val <- gsub("^[^:]+:\\s*", "", values)
    result[[key]] <- val
  }

  result
}

# ─── 2. SRA Metadata Retrieval ────────────────────────────────────────────────

#' Fetch SRR run accessions for a given GEO accession via NCBI Entrez
#'
#' @param geo_acc GEO accession
#' @return Data frame with SRR accessions and metadata
get_sra_metadata <- function(geo_acc) {
  message("Fetching SRA metadata for: ", geo_acc)

  # Search GEO -> get linked SRA UIDs
  geo_search <- entrez_search(db = "gds", term = paste0(geo_acc, "[Accession]"))
  if (geo_search$count == 0) {
    warning("No GEO records found for: ", geo_acc)
    return(NULL)
  }

  # Link to SRA
  sra_links <- entrez_link(dbfrom = "gds", id = geo_search$ids, db = "sra")
  sra_ids   <- sra_links$links$gds_sra

  if (length(sra_ids) == 0) {
    warning("No SRA links found for: ", geo_acc)
    return(NULL)
  }

  # Fetch SRA metadata
  sra_meta_xml <- entrez_fetch(db = "sra", id = sra_ids, rettype = "runinfo")
  # Parse (simplified; in production use SRAdb or ffq)
  lines <- strsplit(sra_meta_xml, "\n")[[1]]
  message(sprintf("Found %d SRA entries", length(sra_ids)))

  data.frame(
    geo_accession = geo_acc,
    sra_ids       = sra_ids,
    n_runs        = length(sra_ids)
  )
}

# ─── 3. Batch GEO Exploration ────────────────────────────────────────────────

#' Search GEO for aging/senescence datasets
#'
#' @param query    Search query
#' @param n_results Number of results to return
#' @return Data frame of matching GEO series
search_aging_geo <- function(
  query = "aging senescence immune cells RNA-seq Homo sapiens",
  n_results = 50
) {
  message("Searching GEO: ", query)

  search <- entrez_search(
    db    = "gds",
    term  = query,
    retmax = n_results
  )

  if (search$count == 0) {
    message("No results found")
    return(NULL)
  }

  # Fetch summaries
  summaries <- entrez_summary(db = "gds", id = search$ids)

  results <- lapply(summaries, function(s) {
    data.frame(
      accession   = s$accession,
      title       = s$title,
      n_samples   = s$n_samples,
      organism    = s$taxon,
      type        = s$gdstype,
      pub_date    = s$pdat,
      stringsAsFactors = FALSE
    )
  }) %>% bind_rows()

  message(sprintf("Found %d datasets", nrow(results)))
  results
}

# ─── 4. MSigDB Gene Sets ──────────────────────────────────────────────────────

#' Load MSigDB gene sets for a given collection
#'
#' @param species  "Homo sapiens" or "Mus musculus"
#' @param category Gene set category (e.g. "H", "C2", "C5")
#' @param subcategory Optional subcategory (e.g. "CP:KEGG", "GO:BP")
#' @return Named list of gene sets (suitable for fgsea/clusterProfiler)
load_msigdb_genesets <- function(species    = "Homo sapiens",
                                  category   = "H",
                                  subcategory = NULL) {
  suppressPackageStartupMessages(library(msigdbr))

  gs <- msigdbr(species = species, category = category,
                subcategory = subcategory)
  gs_list <- split(gs$gene_symbol, gs$gs_name)
  message(sprintf("Loaded %d gene sets from MSigDB %s %s%s",
          length(gs_list), category,
          ifelse(!is.null(subcategory), ": ", ""),
          subcategory %||% ""))
  gs_list
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

# ─── 5. Standard GEO Processing Pipeline ─────────────────────────────────────

#' One-function GEO processing: download → QC → format for DESeq2
#'
#' @param geo_acc        GEO accession
#' @param condition_col  Column in metadata indicating condition/group
#' @param log2_transform Whether to reverse log2 transform (for microarray data)
process_geo_for_deseq2 <- function(geo_acc, condition_col = NULL,
                                    log2_transform = FALSE) {
  data <- download_geo_series(geo_acc)
  meta <- data$metadata
  expr <- data$expression

  # Attempt auto-detection of condition column
  if (is.null(condition_col)) {
    candidate_cols <- grep("group|condition|status|treatment|age",
                           colnames(meta), value = TRUE, ignore.case = TRUE)
    if (length(candidate_cols) > 0) {
      condition_col <- candidate_cols[1]
      message("Auto-detected condition column: ", condition_col)
    } else {
      stop("Could not auto-detect condition column. Specify condition_col.")
    }
  }

  # For RNA-seq: reverse transform if needed
  if (log2_transform) {
    expr <- round(2^expr)
    mode(expr) <- "integer"
  }

  # Filter low-count genes
  if (all(expr >= 0)) {  # count data
    keep <- rowSums(expr >= 10) >= (ncol(expr) * 0.25)
    expr <- expr[keep, ]
    message(sprintf("After filtering: %d genes", nrow(expr)))
  }

  list(
    counts      = expr,
    metadata    = meta,
    condition   = meta[[condition_col]]
  )
}

# ─── Example Usage ────────────────────────────────────────────────────────────
if (FALSE) {
  # Download GSE65907 (PBMC aging RNA-seq)
  data <- download_geo_series("GSE65907", extract_counts = TRUE)

  # Search for aging scRNA-seq datasets
  aging_datasets <- search_aging_geo(
    "aging single cell RNA-seq immune PBMC Homo sapiens",
    n_results = 30
  )

  # Load MSigDB Hallmarks
  hallmarks <- load_msigdb_genesets("Homo sapiens", "H")
  kegg      <- load_msigdb_genesets("Homo sapiens", "C2", "CP:KEGG")
  go_bp     <- load_msigdb_genesets("Homo sapiens", "C5", "GO:BP")
}

message("GEO utilities loaded. Functions: download_geo_series, get_sra_metadata,",
        " search_aging_geo, load_msigdb_genesets, process_geo_for_deseq2")
