################################################################################
# scRNA-seq Immunosenescence — Seurat v5 Pipeline
# Dataset : GSE174072 — PBMCs young vs. aged (10X Chromium)
# Covers  : QC → Clustering → Cell Type Annotation → DE → Senescence Scoring
################################################################################

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(tidyverse)
  library(ggplot2)
  library(cowplot)
  library(scran)
  library(scater)
  library(DoubletFinder)
  library(UCell)
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(pheatmap)
  library(ggrepel)
  library(RColorBrewer)
  library(patchwork)
  library(BiocParallel)
})

set.seed(42)
register(MulticoreParam(4))

RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"
dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(FIGURES_DIR, showWarnings = FALSE)

# ─── 1. Data Loading / Simulation ─────────────────────────────────────────────

simulate_scrnaseq <- function(n_cells = 8000, n_genes = 3000,
                               n_young_donors = 4, n_aged_donors = 4) {
  message("Simulating scRNA-seq data: ", n_cells, " cells × ", n_genes, " genes")

  # Cell types and their marker profiles
  cell_types <- c("CD4_Naive","CD4_Memory","CD8_Naive","CD8_Memory",
                  "CD8_TEMRA","NK","B_Naive","B_Memory","Monocyte_Classical",
                  "Monocyte_NonClassical","pDC","mDC")
  # Age-biased proportions: TEMRA and Memory expand with age
  young_props <- c(0.12,0.08,0.10,0.06,0.02,0.12,0.10,0.05,0.15,0.08,0.05,0.07)
  aged_props  <- c(0.06,0.10,0.07,0.09,0.10,0.09,0.08,0.08,0.16,0.10,0.04,0.03)

  n_young_cells <- n_cells %/% 2
  n_aged_cells  <- n_cells - n_young_cells

  young_ct <- sample(cell_types, n_young_cells, replace = TRUE, prob = young_props)
  aged_ct  <- sample(cell_types, n_aged_cells,  replace = TRUE, prob = aged_props)

  all_ct      <- c(young_ct, aged_ct)
  age_group   <- c(rep("young", n_young_cells), rep("aged", n_aged_cells))
  donor_id    <- c(
    sample(paste0("young_d", 1:n_young_donors), n_young_cells, replace = TRUE),
    sample(paste0("aged_d",  1:n_aged_donors),  n_aged_cells,  replace = TRUE)
  )

  gene_names  <- c(
    # T cell markers
    "CD3E","CD3D","CD3G","CD4","CD8A","CD8B",
    # Naive T
    "CCR7","SELL","LEF1","TCF7",
    # Memory/effector T
    "GZMB","GZMK","PRF1","NKG7","IFNG",
    # TEMRA
    "KLRG1","CX3CR1","FGFBP2","TBX21",
    # NK
    "NCAM1","KLRD1","GNLY","XCL1",
    # B cells
    "MS4A1","CD19","CD79A","IGHM","IGHD",
    # Monocytes
    "LYZ","CD14","FCGR3A","CSF1R","S100A8",
    # DCs
    "CLEC4C","LILRA4","FCER1A","CLEC10A",
    # Senescence markers
    "CDKN1A","CDKN2A","TP53","RB1","MDM2",
    # SASP
    "IL6","IL8","CXCL10","MMP3","TNF",
    "IL1A","IL1B","CXCL1","CXCL2",
    paste0("GENE", 1:(n_genes - 51))
  )
  gene_names <- gene_names[1:n_genes]

  # Generate count matrix
  counts <- matrix(
    rnbinom(n_genes * n_cells, mu = 0.5, size = 0.3),
    nrow = n_genes, ncol = n_cells,
    dimnames = list(gene_names, paste0("CELL", seq_len(n_cells)))
  )

  # Add cell-type-specific marker expression
  marker_boost <- list(
    CD4_Naive  = c("CD3E","CD4","CCR7","SELL"),
    CD8_Naive  = c("CD3E","CD8A","CCR7","SELL"),
    CD8_Memory = c("CD3E","CD8A","GZMK","GZMB"),
    CD8_TEMRA  = c("CD3E","CD8A","KLRG1","CX3CR1","GZMB","PRF1"),
    NK         = c("NCAM1","GNLY","KLRD1","NKG7"),
    B_Naive    = c("MS4A1","CD19","CD79A","IGHM"),
    Monocyte_Classical = c("LYZ","CD14","S100A8"),
    Monocyte_NonClassical = c("FCGR3A","LYZ","CSF1R")
  )

  for (ct in names(marker_boost)) {
    idx_cells <- which(all_ct == ct)
    idx_genes <- which(gene_names %in% marker_boost[[ct]])
    if (length(idx_cells) > 0 && length(idx_genes) > 0) {
      counts[idx_genes, idx_cells] <-
        counts[idx_genes, idx_cells] + rpois(length(idx_genes) * length(idx_cells), 5)
    }
  }

  # Add senescence signal in aged cells
  sasp_genes <- c("IL6","IL8","CXCL10","MMP3","CDKN1A","CDKN2A")
  sasp_idx   <- which(gene_names %in% sasp_genes)
  aged_idx   <- which(age_group == "aged")
  if (length(sasp_idx) > 0)
    counts[sasp_idx, aged_idx] <- counts[sasp_idx, aged_idx] + rpois(
      length(sasp_idx) * length(aged_idx), 3)

  # Metadata
  meta <- data.frame(
    cell_barcode = colnames(counts),
    age_group    = age_group,
    donor_id     = donor_id,
    cell_type    = all_ct,
    row.names    = colnames(counts)
  )

  list(counts = counts, metadata = meta)
}

sim   <- simulate_scrnaseq()
counts <- sim$counts
meta   <- sim$metadata
message(sprintf("Simulated: %d cells × %d genes", ncol(counts), nrow(counts)))

# ─── 2. Create Seurat Object & QC ─────────────────────────────────────────────

seu <- CreateSeuratObject(counts  = counts,
                          meta.data = meta,
                          project   = "immune_aging",
                          min.cells = 3,
                          min.features = 100)

# Mitochondrial %
mito_genes      <- grep("^MT-", rownames(seu), value = TRUE)
# Simulated data has no MT genes, add synthetic %
seu$percent_mt  <- runif(ncol(seu), 0.5, 15)
seu$log10_genes <- log10(seu$nFeature_RNA)

## QC violin plots
p_qc <- VlnPlot(seu, features = c("nFeature_RNA","nCount_RNA","percent_mt"),
                group.by = "age_group", ncol = 3, pt.size = 0) &
  theme_bw(base_size = 10)
ggsave(file.path(FIGURES_DIR, "01_QC_violin.pdf"), p_qc,
       width = 14, height = 5, useDingbats = FALSE)

## Filter
seu <- subset(seu,
              subset = nFeature_RNA > 200 &
                       nFeature_RNA < 5000 &
                       percent_mt   < 20)
message(sprintf("Post-QC: %d cells", ncol(seu)))

# ─── 3. Normalisation & Feature Selection ────────────────────────────────────

seu <- NormalizeData(seu, normalization.method = "LogNormalize",
                     scale.factor = 1e4)
seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 2000)

# Top 10 HVGs
top10_hvg <- head(VariableFeatures(seu), 10)
p_hvg     <- VariableFeaturePlot(seu)
p_hvg     <- LabelPoints(plot = p_hvg, points = top10_hvg, repel = TRUE, xnudge = 0)
ggsave(file.path(FIGURES_DIR, "02_HVG_plot.pdf"), p_hvg,
       width = 8, height = 5, useDingbats = FALSE)

# ─── 4. Scaling, PCA, UMAP ────────────────────────────────────────────────────

seu <- ScaleData(seu, features = rownames(seu), vars.to.regress = "percent_mt")
seu <- RunPCA(seu, features = VariableFeatures(seu), npcs = 50, verbose = FALSE)

## Elbow plot
p_elbow <- ElbowPlot(seu, ndims = 50) + theme_bw(base_size = 11)
ggsave(file.path(FIGURES_DIR, "03_PCA_elbow.pdf"), p_elbow,
       width = 7, height = 4, useDingbats = FALSE)

## PCA coloured by age group
p_pca <- DimPlot(seu, reduction = "pca", group.by = "age_group",
                 cols = c(young = "#2196F3", aged = "#F44336")) +
  labs(title = "PCA — PBMCs Young vs. Aged") +
  theme_bw(base_size = 11)
ggsave(file.path(FIGURES_DIR, "04_PCA_age.pdf"), p_pca,
       width = 7, height = 6, useDingbats = FALSE)

# UMAP
n_dims <- 20
seu <- FindNeighbors(seu, dims = 1:n_dims, verbose = FALSE)
seu <- FindClusters(seu,  resolution = 0.5,  verbose = FALSE, algorithm = 4)
seu <- RunUMAP(seu, dims = 1:n_dims, seed.use = 42, verbose = FALSE)

# ─── 5. Cell Type Annotation ──────────────────────────────────────────────────

# Use ground-truth from simulation (in production: use reference-based annotation
# e.g. SingleR with HumanPrimaryCellAtlasData())
Idents(seu) <- "seurat_clusters"

p_umap_cluster <- DimPlot(seu, label = TRUE, repel = TRUE) +
  labs(title = "UMAP — Leiden Clusters") + NoLegend() +
  theme_bw(base_size = 11)

p_umap_celltype <- DimPlot(seu, group.by = "cell_type",
                            label = TRUE, repel = TRUE, label.size = 3) +
  labs(title = "UMAP — Cell Types (Ground Truth)") + NoLegend() +
  theme_bw(base_size = 11)

p_umap_age <- DimPlot(seu, group.by = "age_group",
                       cols = c(young = "#2196F3", aged = "#F44336")) +
  labs(title = "UMAP — Age Group") +
  theme_bw(base_size = 11)

p_umap_grid <- (p_umap_cluster | p_umap_celltype | p_umap_age)
ggsave(file.path(FIGURES_DIR, "05_UMAP_panels.pdf"), p_umap_grid,
       width = 18, height = 6, useDingbats = FALSE)

# ─── 6. Senescence Scoring (UCell) ───────────────────────────────────────────

sasp_signature <- list(
  SASP = c("IL6","CXCL10","MMP3","IL1B","CXCL1","TNF","IL8","CXCL2"),
  CellCycle_Arrest = c("CDKN1A","CDKN2A","TP53","RB1"),
  Senescence_Core  = c("CDKN1A","CDKN2A","TP53","IL6","CXCL10","MMP3","LMNB1")
)

seu <- AddModuleScore_UCell(seu, features = sasp_signature, name = "")

# Compare senescence score by age group and cell type
score_df <- seu@meta.data %>%
  select(age_group, cell_type, SASP, CellCycle_Arrest, Senescence_Core) %>%
  pivot_longer(cols = c(SASP, CellCycle_Arrest, Senescence_Core),
               names_to = "signature", values_to = "score")

p_sen <- ggplot(score_df %>% filter(signature == "Senescence_Core"),
               aes(reorder(cell_type, score, FUN = median),
                   score, fill = age_group)) +
  geom_boxplot(position = position_dodge(0.9), width = 0.7, outlier.size = 0.5) +
  scale_fill_manual(values = c(young = "#2196F3", aged = "#F44336")) +
  coord_flip() +
  labs(title = "Senescence Score by Cell Type & Age Group",
       subtitle = "UCell | Senescence Core Signature",
       x = NULL, y = "UCell Score") +
  theme_bw(base_size = 11)
ggsave(file.path(FIGURES_DIR, "06_senescence_scores.pdf"), p_sen,
       width = 10, height = 7, useDingbats = FALSE)

## UMAP coloured by senescence score
p_umap_sasp <- FeaturePlot(seu, features = "Senescence_Core",
                            order = TRUE, pt.size = 0.5) +
  scale_colour_gradientn(colours = c("grey90","#FF9800","#F44336","#B71C1C")) +
  labs(title = "Senescence Score (UMAP projection)") +
  theme_bw(base_size = 11)
ggsave(file.path(FIGURES_DIR, "07_senescence_UMAP.pdf"), p_umap_sasp,
       width = 7, height = 6, useDingbats = FALSE)

# ─── 7. Pseudobulk Differential Expression ───────────────────────────────────

# Aggregate counts per donor per cell type (pseudobulk)
pseudo_bulk_de <- function(seurat_obj, cell_type_col = "cell_type",
                            condition_col = "age_group",
                            donor_col = "donor_id", ct = "CD8_TEMRA") {
  sub <- subset(seurat_obj, !!sym(cell_type_col) == ct)
  if (ncol(sub) < 20) return(NULL)

  # Aggregate
  meta_sub  <- sub@meta.data %>% select(all_of(c(donor_col, condition_col)))
  counts_sub <- GetAssayData(sub, layer = "counts")

  agg <- lapply(unique(meta_sub[[donor_col]]), function(d) {
    cells_d <- rownames(meta_sub)[meta_sub[[donor_col]] == d]
    rowSums(counts_sub[, cells_d, drop = FALSE])
  })
  bulk_mat <- do.call(cbind, agg)
  colnames(bulk_mat) <- unique(meta_sub[[donor_col]])

  donor_meta <- meta_sub[!duplicated(meta_sub[[donor_col]]), ]
  rownames(donor_meta) <- donor_meta[[donor_col]]
  donor_meta <- donor_meta[colnames(bulk_mat), ]

  message(sprintf("Pseudobulk DE — %s: %d pseudoreplicates", ct, ncol(bulk_mat)))

  # DESeq2 (requires >=2 per group)
  if (requireNamespace("DESeq2", quietly = TRUE)) {
    dds <- DESeq2::DESeqDataSetFromMatrix(
      countData = bulk_mat,
      colData   = donor_meta,
      design    = as.formula(paste("~", condition_col))
    )
    dds[[condition_col]] <- relevel(dds[[condition_col]], ref = "young")
    dds <- DESeq2::DESeq(dds)
    res <- DESeq2::results(dds)
    as.data.frame(res) %>%
      rownames_to_column("gene") %>%
      mutate(cell_type = ct) %>%
      arrange(padj)
  }
}

de_temra <- pseudo_bulk_de(seu, ct = "CD8_TEMRA")
if (!is.null(de_temra)) {
  write.csv(de_temra,
            file.path(RESULTS_DIR, "pseudobulk_DE_CD8TEMRA.csv"),
            row.names = FALSE)
  message(sprintf("CD8 TEMRA DE: %d significant genes (FDR<0.05)",
          sum(de_temra$padj < 0.05, na.rm = TRUE)))
}

# ─── 8. Cell Proportion Analysis ─────────────────────────────────────────────

prop_df <- seu@meta.data %>%
  count(donor_id, age_group, cell_type) %>%
  group_by(donor_id) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()

prop_summary <- prop_df %>%
  group_by(age_group, cell_type) %>%
  summarise(mean_prop = mean(proportion),
            se_prop   = sd(proportion) / sqrt(n()),
            .groups   = "drop")

p_prop <- ggplot(prop_summary, aes(cell_type, mean_prop, fill = age_group)) +
  geom_col(position = position_dodge(0.9), width = 0.8) +
  geom_errorbar(aes(ymin = mean_prop - se_prop, ymax = mean_prop + se_prop),
                position = position_dodge(0.9), width = 0.3) +
  scale_fill_manual(values = c(young = "#2196F3", aged = "#F44336")) +
  coord_flip() +
  labs(title = "Immune Cell Proportions — Young vs. Aged",
       x = NULL, y = "Mean Proportion (±SE)",
       subtitle = "Note TEMRA expansion and Naive T cell contraction with age") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")
ggsave(file.path(FIGURES_DIR, "08_cell_proportions.pdf"), p_prop,
       width = 10, height = 7, useDingbats = FALSE)

# ─── 9. Save & Export ────────────────────────────────────────────────────────

saveRDS(seu, file.path(RESULTS_DIR, "seurat_immune_aging.rds"))

message("\n=== Seurat pipeline complete ===")
message("Cells: ", ncol(seu), " | Cell types: ", length(unique(seu$cell_type)))
message("Results: ", RESULTS_DIR, " | Figures: ", FIGURES_DIR)

sessionInfo()
