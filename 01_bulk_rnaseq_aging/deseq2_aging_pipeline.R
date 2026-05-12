################################################################################
# Bulk RNA-seq Aging Analysis — DESeq2 Pipeline
# Dataset: Human PBMCs young vs. aged (GSE65907)
#          Mouse spleen CD8+ T cells young vs. old (GSE132901)
#
# Author : Portfolio / CenAGE application
# Date   : 2024
# R      : >= 4.3
################################################################################

suppressPackageStartupMessages({
  library(DESeq2)
  library(apeglm)
  library(tidyverse)
  library(pheatmap)
  library(RColorBrewer)
  library(EnhancedVolcano)
  library(PCAtools)
  library(ggrepel)
  library(BiocParallel)
})

set.seed(42)
register(MulticoreParam(4))

# ─── 0. Configuration ──────────────────────────────────────────────────────────

CONFIG <- list(
  geo_accession  = "GSE65907",
  species        = "human",         # "human" | "mouse"
  condition_col  = "age_group",     # column in metadata
  reference_lvl  = "young",
  results_dir    = "results",
  figures_dir    = "figures",
  padj_thresh    = 0.05,
  lfc_thresh     = 1.0,             # log2 fold-change threshold
  min_count      = 10,              # minimum count filter
  n_top_genes    = 50               # for heatmaps
)

dir.create(CONFIG$results_dir, showWarnings = FALSE)
dir.create(CONFIG$figures_dir, showWarnings = FALSE)

# ─── 1. Data Loading ───────────────────────────────────────────────────────────
# In production: use GEOquery::getGEO() or load from processed counts
# Here we demonstrate the pipeline with simulated data matching real GEO dimensions

load_count_matrix <- function(geo_acc, n_genes = 15000, n_samples = 40) {
  message("Loading count matrix for ", geo_acc, " ...")

  # Simulate negative-binomial counts (matches RNA-seq distribution)
  set.seed(123)
  gene_names  <- paste0("GENE", seq_len(n_genes))
  sample_names <- paste0("SRR", sample(1e6:9e6, n_samples))

  counts <- matrix(
    rnbinom(n_genes * n_samples, mu = 100, size = 0.5),
    nrow = n_genes, ncol = n_samples,
    dimnames = list(gene_names, sample_names)
  )

  # Inject aging DE signal (500 genes up, 500 down in aged)
  aged_cols  <- seq(n_samples / 2 + 1, n_samples)
  up_genes   <- sample(seq_len(n_genes), 500)
  down_genes <- sample(setdiff(seq_len(n_genes), up_genes), 500)
  counts[up_genes,   aged_cols] <- counts[up_genes,   aged_cols] * 4
  counts[down_genes, aged_cols] <- counts[down_genes, aged_cols] %/% 4

  counts
}

load_metadata <- function(count_matrix) {
  n_samples <- ncol(count_matrix)
  n_young   <- n_samples / 2
  data.frame(
    sample_id  = colnames(count_matrix),
    age_group  = factor(rep(c("young", "aged"), each = n_young),
                        levels = c("young", "aged")),
    age_years  = c(runif(n_young, 20, 35), runif(n_young, 65, 85)),
    sex        = sample(c("M", "F"), n_samples, replace = TRUE),
    row.names  = colnames(count_matrix)
  )
}

counts   <- load_count_matrix(CONFIG$geo_accession)
metadata <- load_metadata(counts)

message(sprintf("Loaded: %d genes x %d samples", nrow(counts), ncol(counts)))
message(table(metadata$age_group))

# ─── 2. Pre-filtering ──────────────────────────────────────────────────────────

keep   <- rowSums(counts >= CONFIG$min_count) >= (ncol(counts) * 0.25)
counts <- counts[keep, ]
message(sprintf("After pre-filtering: %d genes retained", nrow(counts)))

# ─── 3. DESeq2 Object Construction ────────────────────────────────────────────

dds <- DESeqDataSetFromMatrix(
  countData = counts,
  colData   = metadata,
  design    = ~ sex + age_group        # covariate-corrected design
)
dds$age_group <- relevel(dds$age_group, ref = CONFIG$reference_lvl)

# ─── 4. Quality Control ────────────────────────────────────────────────────────

vsd <- vst(dds, blind = TRUE)

## 4a. PCA
pca_data <- plotPCA(vsd, intgroup = c("age_group", "sex"), returnData = TRUE)
pct_var  <- round(100 * attr(pca_data, "percentVar"), 1)

p_pca <- ggplot(pca_data, aes(PC1, PC2, colour = age_group, shape = sex)) +
  geom_point(size = 3.5, alpha = 0.85) +
  geom_text_repel(aes(label = name), size = 2.5, max.overlaps = 10) +
  scale_colour_manual(values = c(young = "#2196F3", aged = "#F44336")) +
  labs(
    title    = "PCA — Variance-Stabilised Counts",
    subtitle = sprintf("%s PBMCs: young vs. aged", CONFIG$geo_accession),
    x        = sprintf("PC1 (%s%%)", pct_var[1]),
    y        = sprintf("PC2 (%s%%)", pct_var[2])
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "right")

ggsave(file.path(CONFIG$figures_dir, "01_pca.pdf"), p_pca,
       width = 8, height = 6, useDingbats = FALSE)

## 4b. Sample-distance heatmap
sampleDists   <- dist(t(assay(vsd)))
distmat       <- as.matrix(sampleDists)
ann_col       <- data.frame(age_group = metadata$age_group,
                            sex       = metadata$sex,
                            row.names = rownames(metadata))
ann_colours   <- list(age_group = c(young = "#2196F3", aged = "#F44336"),
                      sex       = c(M = "#9C27B0", F = "#FF9800"))

pheatmap(
  distmat,
  clustering_distance_rows = sampleDists,
  clustering_distance_cols = sampleDists,
  annotation_col  = ann_col,
  annotation_colors = ann_colours,
  color           = colorRampPalette(rev(brewer.pal(9, "Blues")))(255),
  show_rownames   = FALSE,
  show_colnames   = FALSE,
  main            = "Sample-to-Sample Euclidean Distance (VST)",
  filename        = file.path(CONFIG$figures_dir, "02_sample_dist_heatmap.pdf"),
  width = 8, height = 7
)

# ─── 5. Differential Expression ───────────────────────────────────────────────

message("Running DESeq2 ...")
dds <- DESeq(dds, parallel = TRUE)

## Coefficient name for apeglm shrinkage
coef_name <- resultsNames(dds)[grep("age_group", resultsNames(dds))]
message("Shrinking LFC with apeglm, coefficient: ", coef_name)

res_raw <- results(dds,
                   name            = coef_name,
                   alpha           = CONFIG$padj_thresh,
                   independentFiltering = TRUE)

res_lfc <- lfcShrink(dds,
                     coef   = coef_name,
                     type   = "apeglm",
                     res    = res_raw)

## Annotate results
# Preserve Wald stat from res_raw — lfcShrink (apeglm) drops the stat column,
# but gsea_ora_pathway_analysis.R needs it for ranked-list GSEA.
stat_vec <- setNames(as.data.frame(res_raw)$stat,
                     rownames(as.data.frame(res_raw)))

res_df <- as.data.frame(res_lfc) %>%
  rownames_to_column("gene_id") %>%
  arrange(padj) %>%
  mutate(
    stat = stat_vec[gene_id],
    regulation = case_when(
      padj < CONFIG$padj_thresh & log2FoldChange >  CONFIG$lfc_thresh ~ "Up in Aged",
      padj < CONFIG$padj_thresh & log2FoldChange < -CONFIG$lfc_thresh ~ "Down in Aged",
      TRUE ~ "NS"
    )
  )

## Summary table
de_summary <- res_df %>%
  count(regulation) %>%
  mutate(pct = round(100 * n / sum(n), 1))
message("DE Summary:")
print(de_summary)

write.csv(res_df,
          file.path(CONFIG$results_dir, "DE_genes_aged_vs_young.csv"),
          row.names = FALSE)

# ─── 6. Visualisation ──────────────────────────────────────────────────────────

## 6a. Volcano plot (EnhancedVolcano)
top_labels <- res_df %>%
  filter(regulation != "NS") %>%
  slice_min(padj, n = 20) %>%
  pull(gene_id)

p_volcano <- EnhancedVolcano(res_df,
  lab            = res_df$gene_id,
  x              = "log2FoldChange",
  y              = "padj",
  selectLab      = top_labels,
  pCutoff        = CONFIG$padj_thresh,
  FCcutoff       = CONFIG$lfc_thresh,
  title          = "Aged vs. Young PBMCs",
  subtitle       = sprintf("DESeq2 + apeglm | FDR < %s, |LFC| > %s",
                           CONFIG$padj_thresh, CONFIG$lfc_thresh),
  col            = c("grey70", "#2196F3", "#FF9800", "#F44336"),
  colAlpha       = 0.7,
  pointSize      = 2.5,
  labSize        = 3.0,
  legendPosition = "bottom"
)
ggsave(file.path(CONFIG$figures_dir, "03_volcano.pdf"), p_volcano,
       width = 10, height = 9, useDingbats = FALSE)

## 6b. MA plot
res_df_ma <- res_df %>%
  mutate(mean_expr = log10(baseMean + 1))

p_ma <- ggplot(res_df_ma, aes(mean_expr, log2FoldChange, colour = regulation)) +
  geom_point(alpha = 0.4, size = 1) +
  geom_hline(yintercept = c(-CONFIG$lfc_thresh, CONFIG$lfc_thresh),
             linetype = "dashed", colour = "grey30") +
  scale_colour_manual(values = c("Up in Aged"   = "#F44336",
                                 "Down in Aged"  = "#2196F3",
                                 "NS"            = "grey70")) +
  labs(title = "MA Plot — Aged vs. Young",
       x = "Mean Expression (log10)", y = "LFC (apeglm shrunk)") +
  theme_bw(base_size = 13)

ggsave(file.path(CONFIG$figures_dir, "04_ma_plot.pdf"), p_ma,
       width = 8, height = 6, useDingbats = FALSE)

## 6c. Top DE genes heatmap
top_genes <- res_df %>%
  filter(regulation != "NS") %>%
  slice_min(padj, n = CONFIG$n_top_genes) %>%
  pull(gene_id)

heatmap_mat <- assay(vsd)[top_genes, ] %>%
  t() %>% scale() %>% t()

# Cap at ±3 SD
heatmap_mat[heatmap_mat >  3] <-  3
heatmap_mat[heatmap_mat < -3] <- -3

pheatmap(
  heatmap_mat,
  annotation_col  = ann_col,
  annotation_colors = ann_colours,
  show_colnames   = FALSE,
  fontsize_row    = 7,
  color           = colorRampPalette(c("#2196F3","white","#F44336"))(255),
  clustering_method = "ward.D2",
  main            = sprintf("Top %d DE Genes (z-scored VST)", CONFIG$n_top_genes),
  filename        = file.path(CONFIG$figures_dir, "05_top_DE_heatmap.pdf"),
  width = 10, height = 12
)

# ─── 7. Export & Session ───────────────────────────────────────────────────────

saveRDS(dds, file.path(CONFIG$results_dir, "dds_object.rds"))
saveRDS(res_lfc, file.path(CONFIG$results_dir, "res_lfc.rds"))

message("\n=== Pipeline complete ===")
message("Results: ", CONFIG$results_dir)
message("Figures: ", CONFIG$figures_dir)
message(sprintf("Significant DE genes (FDR<%s, |LFC|>%s): %d",
        CONFIG$padj_thresh, CONFIG$lfc_thresh,
        sum(res_df$regulation != "NS", na.rm = TRUE)))

sessionInfo()
