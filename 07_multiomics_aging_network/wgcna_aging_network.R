################################################################################
# WGCNA — Weighted Gene Co-expression Network Analysis
# Aging transcriptome: co-expression modules and hub genes
# Dataset: GSE65907 PBMCs young vs. aged
################################################################################

suppressPackageStartupMessages({
  library(WGCNA)
  library(tidyverse)
  library(ggplot2)
  library(pheatmap)
  library(igraph)
  library(cowplot)
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(msigdbr)
  library(ggraph)
  library(tidygraph)
  library(RColorBrewer)
})

set.seed(42)
options(stringsAsFactors = FALSE)
allowWGCNAThreads(nThreads = 4)

RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"
dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(FIGURES_DIR, showWarnings = FALSE)

# ─── 1. Expression Data ───────────────────────────────────────────────────────

message("Simulating VST-normalised expression matrix ...")
set.seed(42)
n_genes   <- 5000
n_samples <- 40

gene_ids    <- paste0("GENE", seq_len(n_genes))
sample_ids  <- paste0("SAMPLE_", seq_len(n_samples))
ages        <- c(runif(20, 20, 35), runif(20, 65, 85))
age_group   <- c(rep("young", 20), rep("aged", 20))

# VST-like expression (log-scale, ~Normal)
expr <- matrix(rnorm(n_genes * n_samples, 8, 2),
               nrow = n_samples, ncol = n_genes,
               dimnames = list(sample_ids, gene_ids))

# Inject module structure: 6 co-expression modules
module_sizes <- c(300, 250, 200, 150, 100, 80)
module_genes <- list()
available    <- seq_len(n_genes)

for (i in seq_along(module_sizes)) {
  selected       <- sample(available, module_sizes[i])
  module_genes[[i]] <- selected
  available      <- setdiff(available, selected)

  # Correlation within module (eigengene)
  eigengene     <- rnorm(n_samples, 0, 1)
  loading       <- rnorm(module_sizes[i], 0.7, 0.15)
  expr[, selected] <- sweep(
    matrix(rnorm(n_samples * module_sizes[i], 0, 0.5),
           nrow = n_samples),
    1, eigengene * mean(loading), "+"
  )
}

# Module 1 (M1) correlated with age (aging module)
age_z     <- (ages - mean(ages)) / sd(ages)
m1_genes  <- module_genes[[1]]
for (g in m1_genes) {
  expr[, g] <- expr[, g] + 0.8 * age_z + rnorm(n_samples, 0, 0.2)
}

metadata <- data.frame(
  sample_id = sample_ids,
  age       = ages,
  age_group = factor(age_group, levels = c("young","aged")),
  row.names = sample_ids
)

# ─── 2. Sample QC for WGCNA ──────────────────────────────────────────────────

gsg <- goodSamplesGenes(expr, verbose = 0)
if (!gsg$allOK) {
  expr <- expr[gsg$goodSamples, gsg$goodGenes]
  message(sprintf("After sample QC: %d samples × %d genes", nrow(expr), ncol(expr)))
}

## Outlier detection
sampleTree <- hclust(dist(expr), method = "average")
pdf(file.path(FIGURES_DIR, "01_sample_clustering_tree.pdf"), width = 14, height = 5)
par(cex = 0.6, mar = c(0, 4, 2, 0))
plot(sampleTree, main = "Sample Clustering (Outlier Detection)",
     sub = "", xlab = "", ylab = "Height", cex.main = 1.5)
abline(h = quantile(sampleTree$height, 0.99), col = "red", lty = 2)
dev.off()

# ─── 3. Soft-Thresholding Power Selection ────────────────────────────────────

message("Selecting soft-thresholding power ...")
powers    <- c(seq(1, 10), seq(12, 20, by = 2))
sft       <- pickSoftThreshold(expr, powerVector = powers, verbose = 0,
                                networkType = "signed hybrid")

soft_power <- sft$powerEstimate
if (is.na(soft_power)) soft_power <- 6  # fallback
message(sprintf("Selected soft power: %d", soft_power))

# Scale-free topology plot
pdf(file.path(FIGURES_DIR, "02_soft_threshold.pdf"), width = 10, height = 5)
par(mfrow = c(1,2))
plot(sft$fitIndices[,"Power"],
     -sign(sft$fitIndices[,"slope"]) * sft$fitIndices[,"SFT.R.sq"],
     xlab = "Soft Threshold (power)", ylab = "Scale Free Topology Model Fit (R²)",
     type = "n", main = "Scale Independence")
text(sft$fitIndices[,"Power"],
     -sign(sft$fitIndices[,"slope"]) * sft$fitIndices[,"SFT.R.sq"],
     labels = powers, cex = 0.9, col = "red")
abline(h = 0.90, col = "blue", lty = 2)

plot(sft$fitIndices[,"Power"], sft$fitIndices[,"mean.k."],
     xlab = "Soft Threshold (power)", ylab = "Mean Connectivity",
     type = "n", main = "Mean Connectivity")
text(sft$fitIndices[,"Power"], sft$fitIndices[,"mean.k."],
     labels = powers, cex = 0.9, col = "red")
dev.off()

# ─── 4. Network Construction & Module Detection ───────────────────────────────

message("Constructing co-expression network ...")
net <- blockwiseModules(
  expr,
  power              = soft_power,
  networkType        = "signed hybrid",
  TOMType            = "signed",
  minModuleSize      = 30,
  mergeCutHeight     = 0.25,
  numericLabels      = FALSE,
  pamRespectsDendro  = FALSE,
  saveTOMs           = FALSE,
  verbose            = 0,
  maxBlockSize       = n_genes
)

module_colors <- net$colors
module_labels <- levels(factor(module_colors))
n_modules     <- length(module_labels) - 1  # exclude grey (unassigned)

message(sprintf("Modules detected: %d (+ grey)", n_modules))
print(table(module_colors))

saveRDS(net, file.path(RESULTS_DIR, "wgcna_network.rds"))

# ─── 5. Module Eigengenes & Trait Correlation ─────────────────────────────────

MEs        <- orderMEs(net$MEs)
trait_df   <- metadata %>% select(age) %>%
              mutate(aged_binary = as.numeric(metadata$age_group == "aged"))
moduleTraitCor <- cor(MEs, trait_df, use = "pairwise.complete.obs")
moduleTraitPval <- corPvalueStudent(moduleTraitCor, nSamples = nrow(expr))

# Significance stars
textMatrix <- paste(sprintf("%.2f", moduleTraitCor), "\n(",
                   sprintf("%.3f", moduleTraitPval), ")", sep = "")
dim(textMatrix) <- dim(moduleTraitCor)

pdf(file.path(FIGURES_DIR, "03_module_trait_heatmap.pdf"), width = 8, height = 10)
par(mar = c(6, 8, 3, 3))
labeledHeatmap(
  Matrix        = moduleTraitCor,
  xLabels       = c("Age (years)", "Aged (binary)"),
  yLabels       = rownames(moduleTraitCor),
  ySymbols      = rownames(moduleTraitCor),
  colorLabels   = FALSE,
  colors        = blueWhiteRed(50),
  textMatrix    = textMatrix,
  setStdMargins = FALSE,
  cex.text      = 0.6,
  zlim          = c(-1,1),
  main          = "Module-Trait Correlation\n(PBMC aging)"
)
dev.off()

# Export module-trait correlation
as.data.frame(moduleTraitCor) %>%
  rownames_to_column("module") %>%
  write.csv(file.path(RESULTS_DIR, "module_trait_correlation.csv"), row.names = FALSE)

# ─── 6. Hub Gene Identification ───────────────────────────────────────────────

# Module membership (kME)
kME <- as.data.frame(signedKME(expr, MEs, outputColumnName = "kME"))

# Top hub genes per module
hub_genes <- lapply(setdiff(unique(module_colors), "grey"), function(mod) {
  mod_genes   <- names(module_colors)[module_colors == mod]
  kME_col     <- paste0("kME", mod)   # signedKME strips "ME" and keeps lowercase
  if (!kME_col %in% colnames(kME)) return(NULL)
  kME[mod_genes, , drop = FALSE] %>%
    rownames_to_column("gene") %>%
    arrange(desc(.data[[kME_col]])) %>%
    head(20) %>%
    mutate(module = mod)
}) %>% bind_rows()

write.csv(hub_genes,
          file.path(RESULTS_DIR, "hub_genes_per_module.csv"),
          row.names = FALSE)

message("Top hub genes written")

# ─── 7. Module Pathway Enrichment ────────────────────────────────────────────

# Enrich each significant module (|cor| > 0.5 with age)
sig_modules <- rownames(moduleTraitCor)[
  abs(moduleTraitCor[,"age"]) > 0.4 &
  moduleTraitPval[,"age"] < 0.05
]

if (length(sig_modules) == 0) {
  # Use top 2 for demonstration
  sig_modules <- rownames(moduleTraitCor)[order(abs(moduleTraitCor[,"age"]),
                                                  decreasing = TRUE)][1:2]
}
message("Enriching modules: ", paste(sig_modules, collapse=", "))

module_enrichment <- lapply(sig_modules, function(mod) {
  mod_name  <- gsub("^ME", "", mod)
  mod_genes <- names(module_colors)[module_colors == mod_name]
  if (length(mod_genes) < 5) return(NULL)

  enr <- enricher(
    gene         = mod_genes,
    TERM2GENE    = msigdbr(species = "Homo sapiens", category = "H") %>%
                   select(gs_name, gene_symbol),
    pAdjustMethod = "BH",
    pvalueCutoff  = 0.2,
    minGSSize     = 5
  )
  if (is.null(enr) || nrow(enr@result) == 0) return(NULL)
  enr@result %>% mutate(module = mod_name)
}) %>% bind_rows()

if (nrow(module_enrichment) > 0) {
  write.csv(module_enrichment,
            file.path(RESULTS_DIR, "module_pathway_enrichment.csv"),
            row.names = FALSE)
}

# ─── 8. Network Visualisation (igraph) ────────────────────────────────────────

message("Building module hub network for visualisation ...")

# Top hub genes across all modules
top_hubs <- hub_genes %>%
  group_by(module) %>%
  slice_head(n = 8) %>%
  ungroup()

# Build TOM-based graph (subset)
hub_gene_names <- top_hubs$gene
hub_expr       <- expr[, hub_gene_names]
hub_cor        <- cor(hub_expr)

# Adjacency matrix (threshold at r > 0.5)
hub_cor[hub_cor < 0.5] <- 0
diag(hub_cor) <- 0

g <- graph_from_adjacency_matrix(hub_cor,
     mode = "undirected", weighted = TRUE, diag = FALSE)

V(g)$module    <- top_hubs$module[match(V(g)$name, top_hubs$gene)]
V(g)$degree    <- degree(g)
V(g)$hub_score <- sapply(V(g)$name, function(gene) {
  i <- match(gene, hub_genes$gene)
  if (is.na(i)) return(NA_real_)
  hub_genes[[paste0("kME", hub_genes$module[i])]][i]
})

# Colour by module
mod_colours    <- setNames(brewer.pal(min(length(unique(top_hubs$module)), 9), "Set1"),
                           unique(top_hubs$module))
V(g)$colour    <- mod_colours[V(g)$module]

tg <- as_tbl_graph(g)

p_net <- ggraph(tg, layout = "fr") +
  geom_edge_link(aes(width = weight, alpha = weight),
                 colour = "grey60") +
  geom_node_point(aes(size = degree, colour = module)) +
  geom_node_text(aes(label = name), repel = TRUE, size = 2.5) +
  scale_edge_width(range = c(0.3, 2)) +
  scale_edge_alpha(range = c(0.2, 0.7)) +
  scale_size(range = c(3, 12)) +
  labs(
    title    = "Co-expression Hub Network — Aging Modules",
    subtitle = "WGCNA | Edges: Pearson r > 0.5 | Node size ∝ degree",
    colour   = "Module", size = "Degree"
  ) +
  theme_void(base_size = 11) +
  theme(plot.title    = element_text(face = "bold"),
        legend.position = "right")
ggsave(file.path(FIGURES_DIR, "04_hub_network.pdf"), p_net,
       width = 12, height = 10, useDingbats = FALSE)

# Cytoscape-ready export
igraph::write_graph(g,
  file   = file.path(RESULTS_DIR, "hub_network_cytoscape.graphml"),
  format = "graphml"
)
message("Network exported for Cytoscape: hub_network_cytoscape.graphml")

# ─── 9. Module eigengene trajectory across age ────────────────────────────────

ME_long <- as.data.frame(MEs) %>%
  bind_cols(metadata) %>%
  pivot_longer(cols = starts_with("ME"),
               names_to = "module", values_to = "eigengene")

p_eigen <- ggplot(ME_long %>% filter(module %in% sig_modules),
    aes(age, eigengene, colour = module)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
  facet_wrap(~module, scales = "free_y") +
  labs(
    title    = "Module Eigengene Trajectory across Chronological Age",
    subtitle = "Smooth curve: LOESS | Significant aging-correlated modules",
    x        = "Age (years)", y = "Module Eigengene"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none")
ggsave(file.path(FIGURES_DIR, "05_eigengene_vs_age.pdf"), p_eigen,
       width = 10, height = 6, useDingbats = FALSE)

message("\n=== WGCNA analysis complete ===")
message(sprintf("Modules: %d | Hub genes exported | Network saved", n_modules))

sessionInfo()
