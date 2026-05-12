################################################################################
# Differential Abundance with Milo — Neighbourhood-Level Testing
# Dataset: PBMCs young vs. aged (GSE174072)
# Method : Milo (Dann et al. 2022 Nature Biotechnology)
################################################################################

suppressPackageStartupMessages({
  library(miloR)
  library(SingleCellExperiment)
  library(scater)
  library(scran)
  library(tidyverse)
  library(ggplot2)
  library(patchwork)
})

set.seed(42)

RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"
dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(FIGURES_DIR, showWarnings = FALSE)

# ─── 1. Load / Simulate SingleCellExperiment ──────────────────────────────────

message("Loading SCE (simulated from Seurat output)")

n_cells <- 5000
n_genes <- 2000

set.seed(42)
counts_mat <- matrix(
  rnbinom(n_genes * n_cells, mu = 0.8, size = 0.4),
  nrow = n_genes, ncol = n_cells,
  dimnames = list(paste0("GENE", seq_len(n_genes)),
                  paste0("CELL", seq_len(n_cells)))
)

cell_types  <- c("CD4_Naive","CD8_TEMRA","NK","B_Naive","Monocyte_Classical")
young_props <- c(0.30, 0.04, 0.15, 0.25, 0.26)
aged_props  <- c(0.15, 0.20, 0.10, 0.22, 0.33)
n_young     <- n_cells %/% 2

all_ct <- c(
  sample(cell_types, n_young,          replace = TRUE, prob = young_props),
  sample(cell_types, n_cells - n_young, replace = TRUE, prob = aged_props)
)
age_grp <- c(rep("young", n_young), rep("aged", n_cells - n_young))
donors  <- c(
  sample(paste0("Y", 1:4), n_young, replace = TRUE),
  sample(paste0("A", 1:4), n_cells - n_young, replace = TRUE)
)

sce <- SingleCellExperiment(
  assays   = list(counts = counts_mat),
  colData  = DataFrame(
    age_group = age_grp,
    donor_id  = donors,
    cell_type = all_ct
  )
)

# Normalise & dim reduce
sce <- logNormCounts(sce)
sce <- runPCA(sce, ncomponents = 20)
sce <- runUMAP(sce, dimred = "PCA")

# ─── 2. Milo Object ───────────────────────────────────────────────────────────

milo_obj <- Milo(sce)
milo_obj <- buildGraph(milo_obj, k = 30, d = 20)
milo_obj <- makeNhoods(milo_obj, prop = 0.1, k = 30, d = 20,
                        refined = TRUE)

# Size distribution
p_nhood_size <- plotNhoodSizeHist(milo_obj) +
  labs(title = "Milo Neighbourhood Size Distribution") +
  theme_bw(base_size = 11)
ggsave(file.path(FIGURES_DIR, "09_Milo_nhood_size.pdf"), p_nhood_size,
       width = 7, height = 4, useDingbats = FALSE)

# Count cells per neighbourhood per sample
milo_obj <- countCells(milo_obj,
                        meta.data = as.data.frame(colData(milo_obj)),
                        sample    = "donor_id")

# ─── 3. Experimental Design & DA Testing ─────────────────────────────────────

design_df <- data.frame(
  donor_id  = unique(colData(milo_obj)$donor_id)
) %>%
  mutate(
    age_group = ifelse(grepl("^Y", donor_id), "young", "aged"),
    age_group = factor(age_group, levels = c("young", "aged"))
  ) %>%
  column_to_rownames("donor_id")

milo_obj <- calcNhoodDistance(milo_obj, d = 20)

da_results <- testNhoods(
  milo_obj,
  design       = ~ age_group,
  design.df    = design_df,
  model.contrasts = "age_groupaged"
)

da_results <- annotateNhoods(
  milo_obj, da_results,
  coldata_col = "cell_type"
)

message(sprintf("DA testing: %d neighbourhoods | Significant (FDR<0.1): %d",
        nrow(da_results),
        sum(da_results$SpatialFDR < 0.1, na.rm = TRUE)))

write.csv(da_results,
          file.path(RESULTS_DIR, "Milo_DA_results.csv"),
          row.names = FALSE)

# ─── 4. Visualisation ────────────────────────────────────────────────────────

## DA beeswarm plot
p_beeswarm <- plotDAbeeswarm(da_results, group.by = "cell_type") +
  labs(title = "Milo DA Beeswarm — Aged vs. Young",
       subtitle = "LogFC > 0 = enriched in aged | SpatialFDR < 0.1") +
  theme_bw(base_size = 11)
ggsave(file.path(FIGURES_DIR, "10_Milo_beeswarm.pdf"), p_beeswarm,
       width = 9, height = 6, useDingbats = FALSE)

## UMAP with DA logFC
p_umap_da <- plotNhoodGraphDA(milo_obj, da_results,
                               layout = "UMAP",
                               alpha  = 0.1) +
  scale_fill_gradient2(low = "#2196F3", mid = "white", high = "#F44336",
                       midpoint = 0, name = "logFC\n(Aged/Young)") +
  labs(title = "Milo DA on UMAP — Neighbourhood LogFC") +
  theme_void(base_size = 11)
ggsave(file.path(FIGURES_DIR, "11_Milo_UMAP_DA.pdf"), p_umap_da,
       width = 8, height = 7, useDingbats = FALSE)

## Summary barplot
da_summary <- da_results %>%
  filter(!is.na(cell_type_fraction) | !is.na(cell_type)) %>%
  group_by(cell_type) %>%
  summarise(
    mean_logFC = mean(logFC, na.rm = TRUE),
    n_sig_up   = sum(SpatialFDR < 0.1 & logFC > 0, na.rm = TRUE),
    n_sig_down = sum(SpatialFDR < 0.1 & logFC < 0, na.rm = TRUE)
  ) %>%
  arrange(mean_logFC)

p_da_bar <- ggplot(da_summary, aes(reorder(cell_type, mean_logFC), mean_logFC,
                                    fill = mean_logFC > 0)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 0, colour = "black", linewidth = 0.8) +
  scale_fill_manual(values = c("FALSE" = "#2196F3", "TRUE" = "#F44336"),
                    labels = c("Depleted in Aged", "Enriched in Aged")) +
  coord_flip() +
  labs(title = "Mean Neighbourhood LogFC by Cell Type",
       subtitle = "Milo | Aged vs. Young PBMCs",
       x = NULL, y = "Mean LogFC", fill = NULL) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")
ggsave(file.path(FIGURES_DIR, "12_Milo_DA_barplot.pdf"), p_da_bar,
       width = 8, height = 6, useDingbats = FALSE)

message("\n=== Milo analysis complete ===")
sessionInfo()
