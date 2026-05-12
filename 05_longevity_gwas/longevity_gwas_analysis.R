################################################################################
# Longevity GWAS Analysis
# - Manhattan & QQ plots
# - Locus zoom around top hits
# - MAGMA gene-level analysis (documented)
# - Gene-set enrichment on GWAS hits
# - LD-score regression (documented)
# Dataset: Timmers et al. 2019 eLife — Multivariate lifespan GWAS
################################################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(ggrepel)
  library(cowplot)
  library(data.table)
  library(TwoSampleMR)
  library(ieugwasr)
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(msigdbr)
  library(qqman)
  library(GenomicRanges)
})

set.seed(42)

RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"
dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(FIGURES_DIR, showWarnings = FALSE)

# ─── 1. Simulate GWAS Summary Statistics ─────────────────────────────────────

simulate_gwas_sumstats <- function(n_snps = 5e5) {
  message("Simulating GWAS summary statistics (n=", n_snps, " SNPs) ...")
  set.seed(42)

  chromosomes <- rep(1:22, each = round(n_snps / 22))
  chromosomes <- chromosomes[seq_len(n_snps)]

  # Known longevity loci (APOE region: chr19:45M, FOXO3: chr6:109M)
  gwas_df <- data.frame(
    rsid  = paste0("rs", sample(1e6:9e7, n_snps, replace = FALSE)),
    chr   = chromosomes,
    pos   = unlist(lapply(table(chromosomes), function(n) sample(1:250e6, n))),
    A1    = sample(c("A","T","G","C"), n_snps, replace = TRUE),
    A2    = sample(c("A","T","G","C"), n_snps, replace = TRUE),
    beta  = rnorm(n_snps, 0, 0.01),
    se    = abs(rnorm(n_snps, 0.005, 0.002)),
    p     = runif(n_snps, 0, 1),
    maf   = runif(n_snps, 0.01, 0.49),
    n     = round(rnorm(n_snps, 389166, 10000))
  )

  # Inject known longevity loci
  known_loci <- list(
    APOE  = list(chr = 19, pos_approx = 45411941, lead_p = 1e-30, window = 5e5, n_sig = 80),
    FOXO3 = list(chr =  6, pos_approx = 108989589, lead_p = 5e-12, window = 2e5, n_sig = 30),
    CHRNA = list(chr = 15, pos_approx = 78857986,  lead_p = 3e-9,  window = 3e5, n_sig = 20),
    LPA   = list(chr =  6, pos_approx = 160989101, lead_p = 1e-8,  window = 2e5, n_sig = 15),
    ATXN2 = list(chr = 12, pos_approx = 111999000, lead_p = 4e-8,  window = 1e5, n_sig = 10)
  )

  for (locus in known_loci) {
    nearby <- which(gwas_df$chr == locus$chr &
                    abs(gwas_df$pos - locus$pos_approx) < locus$window)
    if (length(nearby) >= locus$n_sig) {
      target <- sample(nearby, locus$n_sig)
      # Assign p-values from lead to edge
      p_vals <- exp(seq(log(locus$lead_p), log(5e-8), length.out = locus$n_sig))
      gwas_df$p[target] <- p_vals
      gwas_df$beta[target] <- rnorm(locus$n_sig, 0.02, 0.005)
    }
  }

  # Genomic control: lambda
  lambda <- median(qchisq(1 - gwas_df$p, 1)) / qchisq(0.5, 1)
  message(sprintf("Genomic inflation factor λ = %.3f", lambda))

  gwas_df
}

gwas <- simulate_gwas_sumstats()
gwas <- gwas %>% arrange(chr, pos)
gwas$logp <- -log10(gwas$p)

write.csv(gwas %>% slice_min(p, n = 5000),
          file.path(RESULTS_DIR, "top_gwas_hits.csv"), row.names = FALSE)
message(sprintf("GWAS: %d SNPs | GW significant (p<5e-8): %d",
        nrow(gwas), sum(gwas$p < 5e-8)))

# ─── 2. Manhattan Plot ────────────────────────────────────────────────────────

# Colour alternating chromosomes
chr_colours <- rep(c("#1565C0", "#42A5F5"), 11)

# Calculate cumulative positions
gwas_plot <- gwas %>%
  group_by(chr) %>%
  summarise(max_pos = max(pos)) %>%
  mutate(cum_offset = cumsum(as.numeric(max_pos)) - as.numeric(max_pos)) %>%
  right_join(gwas, by = "chr") %>%
  mutate(cum_pos = pos + cum_offset)

# Axis labels: midpoint per chromosome
axis_df <- gwas_plot %>%
  group_by(chr) %>%
  summarise(centre = mean(cum_pos))

# Highlight significant
gwas_sig  <- gwas_plot %>% filter(p < 5e-8)
gwas_sugg <- gwas_plot %>% filter(p >= 5e-8 & p < 1e-5)

# Known loci for labelling
known_genes <- list(
  list(chr=19, pos=45411941, gene="APOE"),
  list(chr= 6, pos=108989589, gene="FOXO3"),
  list(chr=15, pos=78857986,  gene="CHRNA3/5"),
  list(chr= 6, pos=160989101, gene="LPA"),
  list(chr=12, pos=111999000, gene="ATXN2")
)

label_df <- lapply(known_genes, function(l) {
  gwas_plot %>%
    filter(chr == l$chr & abs(pos - l$pos) < 1e5) %>%
    slice_min(p, n=1) %>%
    mutate(gene = l$gene)
}) %>% bind_rows()

p_manhattan <- ggplot(gwas_plot, aes(cum_pos, logp)) +
  geom_point(aes(colour = factor(chr)), alpha = 0.6, size = 0.8) +
  scale_colour_manual(values = chr_colours, guide = "none") +
  geom_point(data = gwas_sugg, colour = "#FF9800", alpha = 0.8, size = 1) +
  geom_point(data = gwas_sig,  colour = "#F44336", alpha = 0.9, size = 1.5) +
  geom_hline(yintercept = -log10(5e-8), colour = "#F44336",
             linetype = "dashed", linewidth = 0.8) +
  geom_hline(yintercept = -log10(1e-5), colour = "#FF9800",
             linetype = "dotted", linewidth = 0.6) +
  geom_text_repel(data = label_df,
                  aes(label = gene), colour = "black",
                  size = 3, fontface = "bold",
                  box.padding = 0.6, min.segment.length = 0) +
  scale_x_continuous(labels = axis_df$chr, breaks = axis_df$centre) +
  labs(
    title    = "Longevity GWAS — Manhattan Plot",
    subtitle = "Timmers et al. 2019 (n=389,166) | Simulated summary statistics",
    x        = "Chromosome",
    y        = "-log₁₀(p)"
  ) +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(size = 7), panel.grid.minor = element_blank())

ggsave(file.path(FIGURES_DIR, "01_manhattan.pdf"), p_manhattan,
       width = 14, height = 5, useDingbats = FALSE)

# ─── 3. QQ Plot ───────────────────────────────────────────────────────────────

qq_df <- gwas %>%
  arrange(p) %>%
  mutate(
    observed  = -log10(p),
    expected  = -log10(ppoints(n())),
    ci_lo     = -log10(qbeta(0.025, seq_len(n()), n() - seq_len(n()) + 1)),
    ci_hi     = -log10(qbeta(0.975, seq_len(n()), n() - seq_len(n()) + 1))
  )

lambda <- median(qchisq(1 - gwas$p, 1)) / qchisq(0.5, 1)

p_qq <- ggplot(qq_df %>% sample_frac(0.1), aes(expected, observed)) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.15, fill = "#2196F3") +
  geom_point(alpha = 0.4, size = 0.8, colour = "#607D8B") +
  geom_point(data = qq_df %>% filter(p < 1e-4),
             colour = "#F44336", alpha = 0.9, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, colour = "red", linewidth = 0.8) +
  annotate("text", x = 0.5, y = max(qq_df$observed) * 0.9,
           label = sprintf("λ = %.3f", lambda), hjust = 0, size = 4) +
  labs(title = "Longevity GWAS — QQ Plot",
       subtitle = "Expected vs. observed -log₁₀(p) | Red: significant SNPs",
       x = "Expected -log₁₀(p)", y = "Observed -log₁₀(p)") +
  theme_bw(base_size = 13)
ggsave(file.path(FIGURES_DIR, "02_qq_plot.pdf"), p_qq,
       width = 7, height = 6, useDingbats = FALSE)

# ─── 4. Gene Prioritisation & ORA ────────────────────────────────────────────

# Candidate genes from GW significant loci
sig_snps  <- gwas %>% filter(p < 5e-8)
message(sprintf("GW significant SNPs: %d", nrow(sig_snps)))

# In production: map to genes via positional mapping (ANNOVAR/VEP) or MAGMA
candidate_genes <- c("APOE","TOMM40","PVRL2","APOC1",    # APOE locus
                     "FOXO3","KCNK10",                    # FOXO3 locus
                     "CHRNA3","CHRNA5","CHRNB4",          # CHRNA locus
                     "LPA","PLG",                          # LPA locus
                     "ATXN2","BRAP","SH2B3")              # other

ora_longevity <- enricher(
  gene       = candidate_genes,
  TERM2GENE  = msigdbr(species = "Homo sapiens", category = "H") %>%
               select(gs_name, gene_symbol),
  pAdjustMethod = "BH",
  pvalueCutoff  = 1.0,
  minGSSize     = 2
)

if (!is.null(ora_longevity) && nrow(ora_longevity@result) > 0) {
  write.csv(ora_longevity@result,
            file.path(RESULTS_DIR, "GWAS_ORA_hallmarks.csv"),
            row.names = FALSE)

  p_ora <- dotplot(ora_longevity, showCategory = 15, font.size = 9) +
    labs(title = "Pathway Enrichment of Longevity GWAS Loci",
         subtitle = "MSigDB Hallmarks | clusterProfiler ORA") +
    theme_bw(base_size = 11)
  ggsave(file.path(FIGURES_DIR, "03_GWAS_ORA.pdf"), p_ora,
         width = 9, height = 7, useDingbats = FALSE)
}

# ─── 5. MAGMA commands (documented) ──────────────────────────────────────────

magma_cmds <- '
# MAGMA gene-level analysis
# Step 1: Annotate SNPs to genes
magma \\
  --snp-loc longevity_gwas.snp_loc \\
  --gene-annot longevity.gene_annot \\
  --annotate window=35,10 \\
  --out longevity_annot

# Step 2: Gene analysis
magma \\
  --bfile 1000G_EUR \\
  --pval longevity_gwas.sumstats ncol=N \\
  --gene-annot longevity_annot.genes.annot \\
  --out longevity_gene

# Step 3: Gene-set analysis (MSigDB)
magma \\
  --gene-results longevity_gene.genes.out \\
  --set-annot msigdb_hallmarks.txt \\
  --out longevity_geneset
'

writeLines(magma_cmds, file.path(RESULTS_DIR, "MAGMA_commands.sh"))

# ─── 6. LD-score regression (documented) ──────────────────────────────────────

ldsc_cmds <- '
#!/bin/bash
# LD-score regression — SNP heritability and genetic correlation

# Step 1: Munge sumstats
python munge_sumstats.py \\
  --sumstats longevity_gwas.txt \\
  --N 389166 \\
  --out longevity_munged \\
  --merge-alleles w_hm3.snplist

# Step 2: SNP heritability
python ldsc.py \\
  --h2 longevity_munged.sumstats.gz \\
  --ref-ld-chr eur_w_ld_chr/ \\
  --w-ld-chr   eur_w_ld_chr/ \\
  --out longevity_h2

# Step 3: Genetic correlation with disease traits
python ldsc.py \\
  --rg longevity_munged.sumstats.gz,alzheimers.sumstats.gz,heart_disease.sumstats.gz \\
  --ref-ld-chr eur_w_ld_chr/ \\
  --w-ld-chr   eur_w_ld_chr/ \\
  --out longevity_rg
'

writeLines(ldsc_cmds, file.path(RESULTS_DIR, "LDSC_commands.sh"))
message("MAGMA and LDSC command scripts written")

# ─── 7. Simulated PRS Distribution ───────────────────────────────────────────

# Simulate PRS in centenarians vs. controls
set.seed(42)
n_cent <- 200; n_ctrl <- 800
prs_df <- data.frame(
  prs   = c(rnorm(n_cent, 0.15, 0.8), rnorm(n_ctrl, 0, 1)),
  group = c(rep("Centenarian", n_cent), rep("Control", n_ctrl))
)

p_prs <- ggplot(prs_df, aes(prs, fill = group)) +
  geom_density(alpha = 0.65, colour = NA) +
  geom_vline(data = prs_df %>% group_by(group) %>%
               summarise(mean_prs = mean(prs)),
             aes(xintercept = mean_prs, colour = group),
             linetype = "dashed", linewidth = 1) +
  scale_fill_manual(values = c(Centenarian = "#4CAF50", Control = "#607D8B")) +
  scale_colour_manual(values = c(Centenarian = "#2E7D32", Control = "#37474F"),
                      guide = "none") +
  labs(
    title    = "Longevity Polygenic Risk Score (PRS)",
    subtitle = "Centenarians shifted towards positive PRS",
    x        = "Standardised PRS", y        = "Density",
    fill     = "Group"
  ) +
  theme_bw(base_size = 13)
ggsave(file.path(FIGURES_DIR, "04_PRS_distribution.pdf"), p_prs,
       width = 8, height = 5, useDingbats = FALSE)

message("\n=== Longevity GWAS analysis complete ===")
sessionInfo()
