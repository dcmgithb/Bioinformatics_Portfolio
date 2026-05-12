################################################################################
# Pathway Analysis: ORA + GSEA
# Inputs : DE results from deseq2_aging_pipeline.R
# Methods: clusterProfiler ORA, fgsea GSEA, MSigDB Hallmarks + KEGG + GO
################################################################################

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(enrichplot)
  library(msigdbr)
  library(fgsea)
  library(org.Hs.eg.db)
  library(org.Mm.eg.db)
  library(tidyverse)
  library(ggplot2)
  library(DOSE)
  library(cowplot)
})

set.seed(42)

SPECIES      <- "human"         # "human" | "mouse"
PADJ_THRESH  <- 0.05
LFC_THRESH   <- 1.0
RESULTS_DIR  <- "results"
FIGURES_DIR  <- "figures"

dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(FIGURES_DIR, showWarnings = FALSE)

# ─── 1. Load DE results ────────────────────────────────────────────────────────

de_file <- file.path(RESULTS_DIR, "DE_genes_aged_vs_young.csv")
if (!file.exists(de_file)) {
  stop("Run deseq2_aging_pipeline.R first to generate DE results.")
}
res_df <- read.csv(de_file)
message(sprintf("Loaded %d genes from DE results", nrow(res_df)))

# Gene lists
sig_up   <- res_df %>% filter(regulation == "Up in Aged")   %>% pull(gene_id)
sig_down <- res_df %>% filter(regulation == "Down in Aged")  %>% pull(gene_id)
sig_all  <- c(sig_up, sig_down)
bg_genes <- res_df$gene_id

message(sprintf("Significant: %d up, %d down", length(sig_up), length(sig_down)))

# Ranked gene list for GSEA (by DESeq2 stat)
ranked_genes <- setNames(res_df$stat, res_df$gene_id)
ranked_genes <- sort(ranked_genes[!is.na(ranked_genes)], decreasing = TRUE)

# ─── 2. Gene ID mapping ────────────────────────────────────────────────────────

db <- if (SPECIES == "human") org.Hs.eg.db else org.Mm.eg.db

id_map <- bitr(bg_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = db)
sig_entrez <- id_map %>% filter(SYMBOL %in% sig_all) %>% pull(ENTREZID)
bg_entrez  <- id_map$ENTREZID

# ─── 3. MSigDB Gene Sets ──────────────────────────────────────────────────────

msig_species <- if (SPECIES == "human") "Homo sapiens" else "Mus musculus"

hallmarks <- msigdbr(species = msig_species, category = "H") %>%
  dplyr::select(gs_name, gene_symbol)

c2_kegg <- msigdbr(species = msig_species, category = "C2", subcategory = "CP:KEGG") %>%
  dplyr::select(gs_name, gene_symbol)

c5_bp <- msigdbr(species = msig_species, category = "C5", subcategory = "GO:BP") %>%
  dplyr::select(gs_name, gene_symbol)

# ─── 4. ORA — Over Representation Analysis ────────────────────────────────────

run_ora <- function(gene_list, gene_sets, label, bg = NULL) {
  message("Running ORA: ", label)
  enricher(
    gene       = gene_list,
    TERM2GENE  = gene_sets,
    universe   = bg,
    pAdjustMethod = "BH",
    pvalueCutoff  = 0.05,
    qvalueCutoff  = 0.20,
    minGSSize     = 10,
    maxGSSize     = 500
  )
}

ora_hallmarks_up   <- run_ora(sig_up,   hallmarks, "Hallmarks UP",   bg_genes)
ora_hallmarks_down <- run_ora(sig_down, hallmarks, "Hallmarks DOWN", bg_genes)
ora_kegg           <- run_ora(sig_all,  c2_kegg,   "KEGG",           bg_genes)
ora_go_bp          <- run_ora(sig_all,  c5_bp,     "GO:BP",          bg_genes)

# Combine & export
extract_ora <- function(ora_res, label) {
  if (is.null(ora_res) || nrow(ora_res@result) == 0) return(NULL)
  ora_res@result %>%
    filter(p.adjust < 0.05) %>%
    mutate(source = label) %>%
    arrange(p.adjust)
}

ora_combined <- bind_rows(
  extract_ora(ora_hallmarks_up,   "Hallmarks_Up"),
  extract_ora(ora_hallmarks_down, "Hallmarks_Down"),
  extract_ora(ora_kegg,           "KEGG"),
  extract_ora(ora_go_bp,          "GO_BP")
)
write.csv(ora_combined, file.path(RESULTS_DIR, "ORA_results_all.csv"), row.names = FALSE)
message(sprintf("ORA: %d significant terms", nrow(ora_combined)))

## Dotplot — Hallmarks
if (!is.null(ora_hallmarks_up) && nrow(ora_hallmarks_up@result) > 0) {
  p_ora_hall <- dotplot(ora_hallmarks_up, showCategory = 20, font.size = 9) +
    labs(title = "ORA: MSigDB Hallmarks — Genes UP in Aged PBMCs",
         subtitle = "clusterProfiler | BH-adjusted p < 0.05") +
    theme_bw(base_size = 11)
  ggsave(file.path(FIGURES_DIR, "06_ORA_hallmarks_up.pdf"), p_ora_hall,
         width = 10, height = 8, useDingbats = FALSE)
}

## Barplot — KEGG
if (!is.null(ora_kegg) && nrow(ora_kegg@result) > 0) {
  p_ora_kegg <- barplot(ora_kegg, showCategory = 20, font.size = 9) +
    labs(title = "ORA: KEGG Pathways — Aging DE Genes",
         subtitle = "clusterProfiler | BH-adjusted p < 0.05") +
    theme_bw(base_size = 11)
  ggsave(file.path(FIGURES_DIR, "07_ORA_KEGG.pdf"), p_ora_kegg,
         width = 10, height = 8, useDingbats = FALSE)
}

# ─── 5. GSEA — Gene Set Enrichment Analysis ───────────────────────────────────

run_gsea <- function(ranked, gene_sets, label) {
  message("Running GSEA: ", label)

  # Convert to list format for fgsea
  gs_list <- split(gene_sets$gene_symbol, gene_sets$gs_name)

  fgseaRes <- fgsea(
    pathways   = gs_list,
    stats      = ranked,
    minSize    = 10,
    maxSize    = 500,
    nPermSimple = 1000,
    nproc      = 4
  )

  fgseaRes %>%
    as_tibble() %>%
    arrange(padj) %>%
    mutate(source = label,
           direction = ifelse(NES > 0, "Enriched in Aged", "Depleted in Aged"))
}

gsea_hallmarks <- run_gsea(ranked_genes, hallmarks, "Hallmarks")
gsea_kegg      <- run_gsea(ranked_genes, c2_kegg,   "KEGG")
gsea_gobp      <- run_gsea(ranked_genes, c5_bp,     "GO_BP")

gsea_combined <- bind_rows(gsea_hallmarks, gsea_kegg, gsea_gobp)

# Export (convert list column to string)
gsea_export <- gsea_combined %>%
  mutate(leadingEdge = map_chr(leadingEdge, paste, collapse = ";"))
write.csv(gsea_export, file.path(RESULTS_DIR, "GSEA_results_all.csv"), row.names = FALSE)

## Top pathways plot
top_gsea <- gsea_hallmarks %>%
  filter(padj < 0.05) %>%
  slice_max(abs(NES), n = 20)

if (nrow(top_gsea) > 0) {
  p_gsea_bar <- ggplot(top_gsea,
    aes(reorder(pathway, NES), NES, fill = direction)) +
    geom_col() +
    coord_flip() +
    scale_fill_manual(values = c("Enriched in Aged" = "#F44336",
                                 "Depleted in Aged"  = "#2196F3")) +
    geom_hline(yintercept = 0, colour = "grey30") +
    labs(
      title    = "GSEA: MSigDB Hallmarks — Aged vs. Young PBMCs",
      subtitle = "fgsea | BH-adjusted p < 0.05 | Ranked by DESeq2 Wald stat",
      x        = NULL,
      y        = "Normalised Enrichment Score (NES)",
      fill     = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom")

  ggsave(file.path(FIGURES_DIR, "08_GSEA_hallmarks_barplot.pdf"), p_gsea_bar,
         width = 10, height = 8, useDingbats = FALSE)
}

# ─── 6. Enrichment Plot for Hallmark_Inflammatory_Response ────────────────────

inflam_path <- "HALLMARK_INFLAMMATORY_RESPONSE"
if (inflam_path %in% gsea_hallmarks$pathway) {
  gs_list_hall <- split(hallmarks$gene_symbol, hallmarks$gs_name)

  pdf(file.path(FIGURES_DIR, "09_GSEA_inflammatory_enrichment.pdf"),
      width = 8, height = 5)
  plotEnrichment(gs_list_hall[[inflam_path]], ranked_genes) +
    labs(title = inflam_path,
         subtitle = sprintf("NES = %.2f | padj = %.3e",
           gsea_hallmarks$NES[gsea_hallmarks$pathway == inflam_path],
           gsea_hallmarks$padj[gsea_hallmarks$pathway == inflam_path]))
  dev.off()
}

# ─── 7. Ridgeplot — NES distribution by category ──────────────────────────────

if (nrow(filter(gsea_hallmarks, padj < 0.05)) >= 5) {
  # Use enrichplot's ridgeplot via GSEA object from clusterProfiler
  gsea_cp <- GSEA(
    geneList      = ranked_genes,
    TERM2GENE     = hallmarks,
    pvalueCutoff  = 0.05,
    pAdjustMethod = "BH",
    minGSSize     = 10,
    maxGSSize     = 500,
    by            = "fgsea",
    seed          = 42
  )

  if (!is.null(gsea_cp) && nrow(gsea_cp@result) > 0) {
    p_ridge <- ridgeplot(gsea_cp, showCategory = 20, fill = "NES") +
      labs(title = "GSEA Ridgeplot — Core Enrichment Gene Distributions",
           subtitle = "MSigDB Hallmarks | Aged vs. Young PBMCs") +
      theme_bw(base_size = 10)
    ggsave(file.path(FIGURES_DIR, "10_GSEA_ridgeplot.pdf"), p_ridge,
           width = 10, height = 9, useDingbats = FALSE)
  }
}

# ─── 8. Network plot — GO enrichment ──────────────────────────────────────────

if (!is.null(ora_go_bp) && nrow(ora_go_bp@result) >= 10) {
  ora_go_bp_simp <- simplify(ora_go_bp,
                              cutoff   = 0.7,
                              by       = "p.adjust",
                              select_fun = min)
  p_cnet <- cnetplot(ora_go_bp_simp,
                     showCategory = 10,
                     foldChange   = setNames(res_df$log2FoldChange, res_df$gene_id),
                     colorEdge    = TRUE) +
    labs(title = "GO:BP — Gene-Concept Network",
         subtitle = "Top 10 terms, coloured by LFC") +
    theme_void(base_size = 11)
  ggsave(file.path(FIGURES_DIR, "11_GO_cnetplot.pdf"), p_cnet,
         width = 12, height = 10, useDingbats = FALSE)
}

message("\n=== Pathway analysis complete ===")
message("ORA results:  ", file.path(RESULTS_DIR, "ORA_results_all.csv"))
message("GSEA results: ", file.path(RESULTS_DIR, "GSEA_results_all.csv"))
message("Figures:      ", FIGURES_DIR)

sessionInfo()
