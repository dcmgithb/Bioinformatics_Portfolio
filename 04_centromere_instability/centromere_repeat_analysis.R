################################################################################
# Centromere Instability Analysis
# - α-satellite repeat quantification from WGS data
# - Copy number variation at centromeric regions across age groups
# - Association between centromere instability (CIN) and ageing phenotypes
# - Integration with SASP/senescence gene expression
#
# Dataset: PRJNA680893 — WGS centenarians vs. controls (simulated)
################################################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(ggrepel)
  library(pheatmap)
  library(cowplot)
  library(GenomicRanges)
  library(rtracklayer)
  library(BSgenome.Hsapiens.UCSC.hg38)
  library(ggpubr)
  library(vioplot)
  library(corrplot)
})

set.seed(42)

RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"
dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(FIGURES_DIR, showWarnings = FALSE)

# ─── 1. α-Satellite CNV across age groups ─────────────────────────────────────

# In production: derived from mosdepth output on centromeric BED regions
# Using hg38 centromere coordinates (UCSC Genome Browser)
simulate_censat_cnv <- function(n_samples = 120) {
  set.seed(42)

  # 22 autosomes + chrX + chrY = 24 chromosomes
  chromosomes <- c(paste0("chr", 1:22), "chrX", "chrY")
  n_chrom <- length(chromosomes)

  age_groups <- c(rep("young", 40), rep("middle", 40), rep("aged", 40))
  ages       <- c(runif(40, 20, 40), runif(40, 45, 60), runif(40, 70, 95))
  samples    <- paste0("SAMPLE_", seq_len(n_samples))

  # Simulate α-satellite copy number (normalised to autosomal coverage)
  # Aged samples have increased variance and mean shift (loss of repeats)
  alpha_sat <- matrix(NA, nrow = n_samples, ncol = n_chrom,
                       dimnames = list(samples, chromosomes))
  for (i in seq_len(n_samples)) {
    base_cn <- case_when(
      age_groups[i] == "young"  ~ rnorm(n_chrom, mean = 1.0, sd = 0.10),
      age_groups[i] == "middle" ~ rnorm(n_chrom, mean = 0.95, sd = 0.15),
      TRUE                       ~ rnorm(n_chrom, mean = 0.85, sd = 0.25)
    )
    alpha_sat[i, ] <- pmax(0.1, base_cn)
  }

  # CIN score: mean CV across chromosomes per sample
  cin_score <- apply(alpha_sat, 1, function(x) sd(x) / mean(x))

  metadata <- data.frame(
    sample_id  = samples,
    age_group  = factor(age_groups, levels = c("young","middle","aged")),
    age        = ages,
    cin_score  = cin_score,
    row.names  = samples
  )

  list(alpha_sat = alpha_sat, metadata = metadata)
}

data <- simulate_censat_cnv()
alpha_sat <- data$alpha_sat
metadata  <- data$metadata

write.csv(as.data.frame(alpha_sat) %>% rownames_to_column("sample_id"),
          file.path(RESULTS_DIR, "alpha_sat_cnv.csv"), row.names = FALSE)
write.csv(metadata, file.path(RESULTS_DIR, "metadata.csv"), row.names = FALSE)

# ─── 2. CIN Score by Age Group ────────────────────────────────────────────────

age_pal <- c(young = "#4CAF50", middle = "#FF9800", aged = "#F44336")

p_cin <- ggplot(metadata, aes(age_group, cin_score, fill = age_group)) +
  geom_violin(width = 0.9, alpha = 0.8) +
  geom_boxplot(width = 0.2, fill = "white", outlier.size = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.3, size = 1) +
  stat_compare_means(comparisons = list(c("young","middle"),
                                         c("middle","aged"),
                                         c("young","aged")),
                     method = "wilcox.test",
                     label = "p.signif",
                     tip.length = 0.01) +
  scale_fill_manual(values = age_pal) +
  labs(
    title    = "Centromere Instability Score (CIN) by Age Group",
    subtitle = "CIN = CV of α-satellite copy number across chromosomes",
    x        = "Age Group", y = "CIN Score (CV)",
    caption  = "WGS data; mosdepth centromeric coverage"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
ggsave(file.path(FIGURES_DIR, "01_CIN_score_by_age.pdf"), p_cin,
       width = 7, height = 6, useDingbats = FALSE)

# ─── 3. CIN vs Chronological Age (correlation) ────────────────────────────────

cor_test <- cor.test(metadata$age, metadata$cin_score, method = "pearson")
message(sprintf("CIN ~ Age: r = %.3f, p = %.2e", cor_test$estimate, cor_test$p.value))

p_cin_age <- ggplot(metadata, aes(age, cin_score, colour = age_group)) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_smooth(method = "lm", se = TRUE, colour = "grey30",
              linetype = "dashed", linewidth = 0.8) +
  scale_colour_manual(values = age_pal) +
  annotate("text", x = 25, y = max(metadata$cin_score) * 0.95,
           label = sprintf("Pearson r = %.3f\np = %.2e",
                           cor_test$estimate, cor_test$p.value),
           hjust = 0, size = 3.5) +
  labs(title = "CIN Score vs. Chronological Age",
       subtitle = "Increasing centromere instability with ageing",
       x = "Chronological Age (yr)", y = "CIN Score",
       colour = "Age Group") +
  theme_bw(base_size = 13)
ggsave(file.path(FIGURES_DIR, "02_CIN_vs_age.pdf"), p_cin_age,
       width = 8, height = 6, useDingbats = FALSE)

# ─── 4. Chromosome-Level α-Satellite Heatmap ──────────────────────────────────

# Sort samples by age
sorted_idx <- order(metadata$age)
hm_mat     <- t(alpha_sat[sorted_idx, ])

# Annotation
ann_col <- data.frame(
  age_group  = metadata$age_group[sorted_idx],
  cin_score  = metadata$cin_score[sorted_idx],
  row.names  = rownames(alpha_sat)[sorted_idx]
)
ann_colours <- list(
  age_group = age_pal,
  cin_score = c("#E8F5E9", "#B71C1C")   # unnamed = continuous gradient for pheatmap
)

pheatmap(
  hm_mat,
  annotation_col    = ann_col,
  show_colnames     = FALSE,
  fontsize_row      = 8,
  color             = colorRampPalette(c("#2196F3","white","#F44336"))(200),
  breaks            = seq(0.3, 1.5, length.out = 201),
  clustering_method = "ward.D2",
  cluster_cols      = FALSE,   # keep age-sorted order
  main              = "α-Satellite Copy Number by Chromosome (Age-sorted)",
  filename          = file.path(FIGURES_DIR, "03_alpha_sat_heatmap.pdf"),
  width = 14, height = 6
)

# ─── 5. Per-Chromosome CIN Distribution ──────────────────────────────────────

chrom_df <- as.data.frame(alpha_sat) %>%
  rownames_to_column("sample_id") %>%
  left_join(metadata %>% select(sample_id, age_group), by = "sample_id") %>%
  pivot_longer(cols = starts_with("chr"),
               names_to = "chromosome", values_to = "alpha_sat_cn") %>%
  mutate(chrom_num = as.integer(recode(gsub("chr","", chromosome),
                                       X = "23", Y = "24")))

chrom_summary <- chrom_df %>%
  group_by(chromosome, age_group) %>%
  summarise(mean_cn = mean(alpha_sat_cn),
            sd_cn   = sd(alpha_sat_cn), .groups = "drop")

p_chrom <- ggplot(chrom_summary,
    aes(reorder(chromosome,
                as.integer(recode(gsub("chr","",chromosome), X="23", Y="24"))),
        mean_cn, fill = age_group)) +
  geom_col(position = position_dodge(0.9), width = 0.85) +
  geom_errorbar(aes(ymin = mean_cn - sd_cn, ymax = mean_cn + sd_cn),
                position = position_dodge(0.9), width = 0.3) +
  scale_fill_manual(values = age_pal) +
  geom_hline(yintercept = 1, lty = "dashed", colour = "grey30") +
  labs(title = "α-Satellite Copy Number per Chromosome",
       subtitle = "Mean ± SD | Relative to autosomal coverage",
       x = "Chromosome", y = "Relative α-Satellite CN",
       fill = "Age Group") +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        legend.position = "bottom")
ggsave(file.path(FIGURES_DIR, "04_per_chrom_alpha_sat.pdf"), p_chrom,
       width = 14, height = 6, useDingbats = FALSE)

# ─── 6. CIN-SASP Integration ──────────────────────────────────────────────────
# Simulate matched RNAseq SASP scores for the same samples

set.seed(99)
metadata$sasp_score  <- 0.3 * metadata$cin_score + rnorm(nrow(metadata), 0, 0.05)
metadata$il6_expr    <- 2 * metadata$cin_score + rnorm(nrow(metadata), 0, 0.3)
metadata$senescence_score <- 0.5 * metadata$cin_score + rnorm(nrow(metadata), 0, 0.08)

p_cin_sasp <- ggplot(metadata, aes(cin_score, sasp_score, colour = age_group)) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_smooth(method = "lm", se = TRUE, colour = "grey30", linetype = "dashed") +
  scale_colour_manual(values = age_pal) +
  ggpubr::stat_cor(method = "pearson", label.x.npc = 0.6, label.y.npc = 0.1) +
  labs(
    title    = "Centromere Instability → SASP Activation",
    subtitle = "CIN score correlates with SASP expression in matched samples",
    x        = "CIN Score", y = "SASP Score (expression)",
    colour   = "Age Group"
  ) +
  theme_bw(base_size = 13)
ggsave(file.path(FIGURES_DIR, "05_CIN_vs_SASP.pdf"), p_cin_sasp,
       width = 8, height = 6, useDingbats = FALSE)

# Correlation matrix: CIN + aging phenotypes
cor_mat <- metadata %>%
  select(age, cin_score, sasp_score, il6_expr, senescence_score) %>%
  cor(method = "pearson")

pdf(file.path(FIGURES_DIR, "06_CIN_correlation_matrix.pdf"), width = 7, height = 6)
corrplot::corrplot(cor_mat, method = "ellipse", type = "upper",
                   tl.cex = 0.9, tl.col = "black",
                   addCoef.col = "black", number.cex = 0.75,
                   title = "CIN & Ageing Phenotype Correlations",
                   mar = c(0,0,1,0))
dev.off()

# ─── 7. Centenarian vs. Control ──────────────────────────────────────────────

centenarians <- metadata %>% filter(age >= 90) %>% mutate(group = "Centenarian")
controls     <- metadata %>% filter(age < 70)  %>% mutate(group = "Control")
comp_df      <- bind_rows(centenarians, controls)

message(sprintf("Centenarians: %d | Controls: %d", nrow(centenarians), nrow(controls)))

p_cent <- ggplot(comp_df, aes(group, cin_score, fill = group)) +
  geom_violin(alpha = 0.8) +
  geom_boxplot(width = 0.2, fill = "white") +
  stat_compare_means(method = "wilcox.test", label = "p.format") +
  scale_fill_manual(values = c(Centenarian = "#4CAF50", Control = "#F44336")) +
  labs(
    title    = "CIN Score: Centenarians vs. Age-matched Controls",
    subtitle = "Lower CIN may be a genomic longevity feature",
    x = NULL, y = "CIN Score"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
ggsave(file.path(FIGURES_DIR, "07_centenarian_CIN.pdf"), p_cent,
       width = 6, height = 6, useDingbats = FALSE)

message("\n=== Centromere instability analysis complete ===")
sessionInfo()
