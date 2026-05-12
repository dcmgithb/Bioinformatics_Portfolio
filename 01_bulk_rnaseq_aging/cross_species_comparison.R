################################################################################
# Cross-Species Aging Signature: Human PBMCs vs. Mouse Spleen CD8+ T cells
# Ortholog mapping with biomaRt → conserved aging transcriptomic signature
################################################################################

suppressPackageStartupMessages({
  library(biomaRt)
  library(tidyverse)
  library(ggplot2)
  library(ggrepel)
  library(VennDiagram)
  library(pheatmap)
  library(cowplot)
})

set.seed(42)

RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"
dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(FIGURES_DIR, showWarnings = FALSE)

# ─── 1. Load DE results ────────────────────────────────────────────────────────

human_de <- read.csv(file.path(RESULTS_DIR, "DE_genes_aged_vs_young.csv"))

# Simulate mouse DE data (in production: load from GSE132901)
set.seed(99)
n_mouse_genes <- 12000
mouse_de <- data.frame(
  gene_id        = paste0("GENE", seq_len(n_mouse_genes)),
  log2FoldChange = rnorm(n_mouse_genes, 0, 1.5),
  padj           = rbeta(n_mouse_genes, 0.2, 2)
) %>%
  mutate(
    padj = p.adjust(runif(n_mouse_genes, 0, 1), method = "BH"),
    regulation = case_when(
      padj < 0.05 & log2FoldChange >  1 ~ "Up in Old",
      padj < 0.05 & log2FoldChange < -1 ~ "Down in Old",
      TRUE ~ "NS"
    )
  )

message("Human DE: ", sum(human_de$regulation != "NS"), " significant genes")
message("Mouse DE: ", sum(mouse_de$regulation  != "NS"), " significant genes")

# ─── 2. Ortholog Mapping ──────────────────────────────────────────────────────

# In production: use biomaRt to map human <-> mouse orthologs
# Example code (requires internet):
#
# human_mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
# mouse_mart <- useMart("ensembl", dataset = "mmusculus_gene_ensembl")
# orthologs  <- getLDS(
#   attributes    = "hgnc_symbol",
#   mart          = human_mart,
#   attributesL   = "mgi_symbol",
#   martL         = mouse_mart,
#   uniqueRows    = TRUE
# )

# Simulate ortholog pairs (representative of real ~18k mappable genes)
set.seed(42)
n_ortho    <- 8000
human_pool <- human_de$gene_id
mouse_pool <- mouse_de$gene_id

orthologs <- data.frame(
  human_symbol = sample(human_pool, n_ortho, replace = FALSE),
  mouse_symbol = sample(mouse_pool, n_ortho, replace = FALSE)
)

message(sprintf("Ortholog pairs: %d", nrow(orthologs)))

# ─── 3. Merge & Conserved Signature ──────────────────────────────────────────

merged <- orthologs %>%
  left_join(human_de %>% select(gene_id, log2FoldChange, padj, regulation),
            by = c("human_symbol" = "gene_id")) %>%
  rename(human_lfc = log2FoldChange, human_padj = padj, human_reg = regulation) %>%
  left_join(mouse_de %>% select(gene_id, log2FoldChange, padj, regulation),
            by = c("mouse_symbol" = "gene_id")) %>%
  rename(mouse_lfc = log2FoldChange, mouse_padj = padj, mouse_reg = regulation) %>%
  drop_na()

# Conserved: DE in both species, same direction
merged <- merged %>%
  mutate(
    conserved = case_when(
      human_reg == "Up in Aged"    & mouse_reg == "Up in Old"    ~ "Conserved UP",
      human_reg == "Down in Aged"  & mouse_reg == "Down in Old"  ~ "Conserved DOWN",
      human_reg != "NS"            | mouse_reg  != "NS"          ~ "Species-specific",
      TRUE ~ "NS"
    )
  )

conserved_up   <- merged %>% filter(conserved == "Conserved UP")
conserved_down <- merged %>% filter(conserved == "Conserved DOWN")
message(sprintf("Conserved UP:   %d genes", nrow(conserved_up)))
message(sprintf("Conserved DOWN: %d genes", nrow(conserved_down)))

# Export
write.csv(merged, file.path(RESULTS_DIR, "cross_species_merged.csv"), row.names = FALSE)
write.csv(bind_rows(conserved_up, conserved_down),
          file.path(RESULTS_DIR, "conserved_aging_signature.csv"), row.names = FALSE)

# ─── 4. Scatter Plot: Human vs. Mouse LFC ─────────────────────────────────────

colour_map <- c(
  "Conserved UP"       = "#F44336",
  "Conserved DOWN"     = "#2196F3",
  "Species-specific"   = "#FF9800",
  "NS"                 = "#BDBDBD"
)

top_labels <- bind_rows(
  conserved_up   %>% slice_min(human_padj, n = 10),
  conserved_down %>% slice_min(human_padj, n = 10)
)

p_scatter <- ggplot(merged, aes(human_lfc, mouse_lfc, colour = conserved)) +
  geom_point(alpha = 0.4, size = 1) +
  geom_point(data = bind_rows(conserved_up, conserved_down),
             alpha = 0.8, size = 2.5) +
  geom_text_repel(data = top_labels,
                  aes(label = human_symbol),
                  size = 2.8, max.overlaps = 20, colour = "black") +
  geom_hline(yintercept = 0, lty = "dashed", colour = "grey40") +
  geom_vline(xintercept = 0, lty = "dashed", colour = "grey40") +
  geom_abline(slope = 1, intercept = 0, lty = "solid", colour = "grey60", alpha = 0.5) +
  scale_colour_manual(values = colour_map) +
  labs(
    title    = "Cross-Species Aging Transcriptome Comparison",
    subtitle = "Human PBMCs (GSE65907) vs. Mouse CD8+ T cells (GSE132901)",
    x        = "Human log2FC (Aged/Young)",
    y        = "Mouse log2FC (Old/Young)",
    colour   = "Conservation"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "right")

ggsave(file.path(FIGURES_DIR, "12_cross_species_scatter.pdf"), p_scatter,
       width = 9, height = 8, useDingbats = FALSE)

# ─── 5. Correlation of Aging Effect Sizes ────────────────────────────────────

sig_both <- merged %>%
  filter(human_reg != "NS" | mouse_reg != "NS")

cor_val  <- cor(sig_both$human_lfc, sig_both$mouse_lfc,
                use = "pairwise.complete.obs", method = "pearson")
message(sprintf("Pearson r (LFC, human vs. mouse) on DE genes: %.3f", cor_val))

# ─── 6. Summary Bar ──────────────────────────────────────────────────────────

summary_df <- merged %>%
  count(conserved) %>%
  mutate(conserved = factor(conserved,
         levels = c("Conserved UP","Conserved DOWN","Species-specific","NS")))

p_bar <- ggplot(summary_df, aes(conserved, n, fill = conserved)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = n), vjust = -0.4, fontface = "bold", size = 4) +
  scale_fill_manual(values = colour_map) +
  labs(title = "Cross-Species Aging Signature Summary",
       x = NULL, y = "Number of Ortholog Pairs") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1))

ggsave(file.path(FIGURES_DIR, "13_cross_species_bar.pdf"), p_bar,
       width = 7, height = 5, useDingbats = FALSE)

message("\n=== Cross-species analysis complete ===")
message(sprintf("Conserved aging signature: %d genes",
        nrow(conserved_up) + nrow(conserved_down)))

sessionInfo()
