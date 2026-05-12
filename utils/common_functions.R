################################################################################
# Common Bioinformatics Utility Functions (R)
# Shared across all portfolio projects
################################################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(scales)
  library(ggpubr)
})

# ─── Plot Aesthetics ──────────────────────────────────────────────────────────

#' Standard ggplot2 theme for publication-ready figures
theme_aging <- function(base_size = 12) {
  theme_bw(base_size = base_size) +
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 1),
      plot.subtitle    = element_text(colour = "grey40", size = base_size - 1),
      axis.title       = element_text(face = "plain"),
      legend.position  = "right",
      legend.key.size  = unit(0.8, "lines"),
      strip.background = element_rect(fill = "grey95"),
      panel.grid.minor = element_blank()
    )
}

#' Colour palettes for aging studies
aging_palettes <- list(
  age_group   = c(young = "#2196F3", aged = "#F44336"),
  age_3groups = c(young = "#4CAF50", middle = "#FF9800", aged = "#F44336"),
  regulation  = c("Up in Aged" = "#F44336", "Down in Aged" = "#2196F3", NS = "#BDBDBD"),
  sex         = c(M = "#9C27B0", F = "#FF9800"),
  species     = c(human = "#1565C0", mouse = "#2E7D32"),
  diverging   = c("#2196F3", "white", "#F44336")
)

# ─── Statistical Utilities ───────────────────────────────────────────────────

#' Multiple testing correction with FDR and Bonferroni
add_multiple_testing <- function(df, p_col = "pvalue",
                                  methods = c("BH","bonferroni")) {
  for (method in methods) {
    col_name       <- paste0("p_", method)
    df[[col_name]] <- p.adjust(df[[p_col]], method = method)
  }
  df
}

#' Volcano plot helper: classify genes by significance and fold change
classify_de_genes <- function(df, lfc_col = "log2FoldChange",
                               padj_col = "padj",
                               lfc_thresh = 1.0, padj_thresh = 0.05) {
  df %>%
    mutate(
      regulation = case_when(
        .data[[padj_col]] < padj_thresh & .data[[lfc_col]] >  lfc_thresh ~ "Up in Aged",
        .data[[padj_col]] < padj_thresh & .data[[lfc_col]] < -lfc_thresh ~ "Down in Aged",
        TRUE ~ "NS"
      )
    )
}

#' Quick summary statistics for a numeric vector
quick_stats <- function(x, label = "x") {
  x <- x[!is.na(x)]
  data.frame(
    variable = label,
    n        = length(x),
    mean     = mean(x),
    sd       = sd(x),
    median   = median(x),
    iqr      = IQR(x),
    min      = min(x),
    max      = max(x)
  )
}

# ─── Genomics Utilities ───────────────────────────────────────────────────────

#' Convert Ensembl IDs to HGNC symbols
#' Requires biomaRt (internet connection)
ensembl_to_symbol <- function(ensembl_ids, species = "human") {
  if (!requireNamespace("biomaRt", quietly = TRUE)) {
    stop("Install biomaRt: BiocManager::install('biomaRt')")
  }
  dataset <- ifelse(species == "human", "hsapiens_gene_ensembl",
                                         "mmusculus_gene_ensembl")
  mart <- biomaRt::useMart("ensembl", dataset = dataset)
  biomaRt::getBM(
    attributes  = c("ensembl_gene_id", "hgnc_symbol", "gene_biotype"),
    filters     = "ensembl_gene_id",
    values      = ensembl_ids,
    mart        = mart
  )
}

#' Map mouse gene symbols to human orthologs via biomaRt
mouse_to_human_orthologs <- function(mouse_genes) {
  if (!requireNamespace("biomaRt", quietly = TRUE))
    stop("Install biomaRt")

  human <- biomaRt::useMart("ensembl", dataset = "hsapiens_gene_ensembl")
  mouse <- biomaRt::useMart("ensembl", dataset = "mmusculus_gene_ensembl")

  biomaRt::getLDS(
    attributes  = "mgi_symbol",
    mart        = mouse,
    values      = mouse_genes,
    attributesL = "hgnc_symbol",
    martL       = human,
    uniqueRows  = TRUE
  )
}

# ─── Count Data Utilities ─────────────────────────────────────────────────────

#' Filter low-count genes using multiple criteria
filter_low_counts <- function(counts, min_count = 10, min_samples_frac = 0.25) {
  n_samples  <- ncol(counts)
  min_samples <- ceiling(n_samples * min_samples_frac)
  keep <- rowSums(counts >= min_count) >= min_samples
  message(sprintf("Filtering: %d → %d genes (%.1f%% retained)",
          nrow(counts), sum(keep), 100 * mean(keep)))
  counts[keep, ]
}

#' Calculate CPM from raw counts
calc_cpm <- function(counts) {
  lib_sizes <- colSums(counts)
  sweep(counts * 1e6, 2, lib_sizes, FUN = "/")
}

#' Calculate TPM from raw counts and gene lengths
calc_tpm <- function(counts, gene_lengths) {
  if (length(gene_lengths) != nrow(counts))
    stop("gene_lengths length must match nrow(counts)")
  # RPK (reads per kilobase)
  rpk <- counts / (gene_lengths / 1000)
  # Scale to per-million
  sweep(rpk * 1e6, 2, colSums(rpk), FUN = "/")
}

# ─── Visualisation Helpers ────────────────────────────────────────────────────

#' Combine p-value and fold change into a significance label
pval_label <- function(pval, lfc = NULL, digits = 3) {
  stars <- ifelse(pval < 0.001, "***",
           ifelse(pval < 0.01,  "**",
           ifelse(pval < 0.05,  "*", "ns")))
  if (!is.null(lfc))
    paste0(stars, " (LFC=", round(lfc, 2), ")")
  else
    stars
}

#' Create a publication-ready boxplot with significance annotations
boxplot_with_stats <- function(df, x_col, y_col, fill_col = x_col,
                                colours = NULL, title = "", subtitle = "") {
  p <- ggplot(df, aes(.data[[x_col]], .data[[y_col]], fill = .data[[fill_col]])) +
    geom_boxplot(width = 0.6, outlier.size = 0.8) +
    geom_jitter(width = 0.15, alpha = 0.3, size = 1) +
    stat_compare_means(method = "wilcox.test",
                       comparisons = list(levels(factor(df[[x_col]]))[c(1,2)]),
                       label = "p.signif") +
    labs(title = title, subtitle = subtitle, x = NULL, y = y_col) +
    theme_aging()

  if (!is.null(colours))
    p <- p + scale_fill_manual(values = colours)
  p
}

#' Heatmap wrapper: scale + annotate + pheatmap
annotated_heatmap <- function(mat, ann_col = NULL, ann_colours = NULL,
                               title = "Heatmap", scale_by = "row",
                               filename = NULL, ...) {
  if (scale_by == "row") {
    mat <- t(scale(t(mat)))
    mat[mat >  3] <-  3
    mat[mat < -3] <- -3
  }
  pheatmap::pheatmap(
    mat,
    annotation_col    = ann_col,
    annotation_colors = ann_colours,
    color  = colorRampPalette(c("#2196F3","white","#F44336"))(255),
    main   = title,
    filename = filename,
    ...
  )
}

# ─── Reproducibility Helpers ─────────────────────────────────────────────────

#' Lock random seed and print session snapshot
init_analysis <- function(seed = 42, project = "aging_analysis") {
  set.seed(seed)
  message(sprintf("[%s] Seed set to %d", project, seed))
  message(sprintf("[%s] R version: %s", project, R.version.string))
  message(sprintf("[%s] Date: %s", project, Sys.time()))
}

#' Write session info to a file for reproducibility
save_session_info <- function(outdir = ".", prefix = "session") {
  outfile <- file.path(outdir, paste0(prefix, "_sessionInfo.txt"))
  sink(outfile)
  print(sessionInfo())
  sink()
  message("Session info saved: ", outfile)
}

message("Common functions loaded. Available: theme_aging, aging_palettes,",
        " classify_de_genes, filter_low_counts, calc_cpm, calc_tpm,",
        " mouse_to_human_orthologs, boxplot_with_stats, annotated_heatmap,",
        " init_analysis, save_session_info")
