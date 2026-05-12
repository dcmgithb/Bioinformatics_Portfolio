################################################################################
# Mendelian Randomisation — Causal Risk Factors for Longevity
# Methods: IVW, MR-Egger, Weighted Median, MR-PRESSO
# Exposures: LDL-C, HDL-C, BMI, smoking, blood pressure, education
# Outcome  : Longevity (Timmers 2019 sumstats)
################################################################################

suppressPackageStartupMessages({
  library(TwoSampleMR)
  library(MendelianRandomization)
  library(tidyverse)
  library(ggplot2)
  library(cowplot)
  library(ggforestplot)
})

set.seed(42)

RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"
dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(FIGURES_DIR, showWarnings = FALSE)

# ─── 1. Exposure IVs (simulated — in production: from IEU GWAS database) ─────

# In production:
# ao <- available_outcomes()  # browse IEU GWAS database
# exposure_dat <- extract_instruments(outcomes = c("ieu-b-110",  # LDL-C
#                                                   "ieu-b-109",  # HDL-C
#                                                   "ieu-b-40",   # BMI
#                                                   "ieu-b-25"))  # Smoking

simulate_exposure_ivs <- function(exposure, n_snps = 15, beta_dir = 1) {
  set.seed(which(c("LDL_C","HDL_C","BMI","Smoking","SBP","Education") == exposure))
  data.frame(
    SNP         = paste0("rs_", exposure, "_", seq_len(n_snps)),
    beta.exposure = rnorm(n_snps, 0.08 * beta_dir, 0.02),
    se.exposure   = abs(rnorm(n_snps, 0.01, 0.003)),
    pval.exposure = runif(n_snps, 1e-10, 5e-8),
    eaf.exposure  = runif(n_snps, 0.1, 0.5),
    effect_allele.exposure = "A",
    other_allele.exposure  = "G",
    exposure      = exposure,
    id.exposure   = paste0("EXP_", exposure)
  )
}

exposures_list <- list(
  simulate_exposure_ivs("LDL_C",    n_snps = 20, beta_dir = -1),  # LDL harmful
  simulate_exposure_ivs("HDL_C",    n_snps = 15, beta_dir =  1),  # HDL protective?
  simulate_exposure_ivs("BMI",      n_snps = 25, beta_dir = -1),  # BMI harmful
  simulate_exposure_ivs("Smoking",  n_snps = 10, beta_dir = -1),  # Smoking harmful
  simulate_exposure_ivs("SBP",      n_snps = 18, beta_dir = -1),  # SBP harmful
  simulate_exposure_ivs("Education",n_snps = 12, beta_dir =  1)   # Education protective
)
exposure_dat <- bind_rows(exposures_list)

# ─── 2. Outcome: Longevity ────────────────────────────────────────────────────

# Simulate longevity effect of each IV
outcome_dat <- exposure_dat %>%
  mutate(
    id.outcome        = "LONGEVITY",
    outcome           = "Longevity",
    beta.outcome      = beta.exposure * rnorm(n(), 0.6, 0.15),
    se.outcome        = abs(rnorm(n(), 0.015, 0.005)),
    pval.outcome      = 2 * pnorm(-abs(beta.outcome / se.outcome)),
    effect_allele.outcome = effect_allele.exposure,
    other_allele.outcome  = other_allele.exposure,
    eaf.outcome           = eaf.exposure
  )

# Harmonise
dat <- harmonise_data(
  exposure_dat = exposure_dat,
  outcome_dat  = outcome_dat,
  action       = 2
)
message(sprintf("Harmonised: %d SNP-exposure-outcome pairs", nrow(dat)))

# ─── 3. MR Analysis ──────────────────────────────────────────────────────────

mr_results <- mr(dat, method_list = c("mr_ivw","mr_egger_regression",
                                       "mr_weighted_median","mr_weighted_mode"))

mr_results_sig <- mr_results %>%
  filter(method == "mr_ivw") %>%
  mutate(
    or      = exp(b),
    or_lo95 = exp(b - 1.96 * se),
    or_hi95 = exp(b + 1.96 * se),
    direction = ifelse(b > 0, "Protective", "Harmful"),
    p_fdr   = p.adjust(pval, method = "BH")
  ) %>%
  arrange(pval)

write.csv(mr_results_sig, file.path(RESULTS_DIR, "MR_results_longevity.csv"),
          row.names = FALSE)

# ─── 4. Heterogeneity & Pleiotropy Tests ─────────────────────────────────────

mr_het   <- mr_heterogeneity(dat)
mr_pleio <- mr_pleiotropy_test(dat)

write.csv(mr_het,   file.path(RESULTS_DIR, "MR_heterogeneity.csv"),  row.names = FALSE)
write.csv(mr_pleio, file.path(RESULTS_DIR, "MR_pleiotropy.csv"),     row.names = FALSE)

# ─── 5. Forest Plot ───────────────────────────────────────────────────────────

forest_df <- mr_results %>%
  filter(method == "mr_ivw") %>%
  mutate(
    lo95 = b - 1.96 * se,
    hi95 = b + 1.96 * se,
    label = sprintf("β=%.3f (95%%CI: %.3f, %.3f)\np=%.3f",
                    b, lo95, hi95, pval)
  )

p_forest <- ggplot(forest_df, aes(x = b, y = reorder(exposure, b))) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  geom_errorbarh(aes(xmin = lo95, xmax = hi95),
                 height = 0.3, colour = "grey30") +
  geom_point(aes(colour = b > 0), size = 4) +
  scale_colour_manual(values = c("FALSE" = "#F44336", "TRUE" = "#4CAF50"),
                      labels = c("FALSE" = "Harmful", "TRUE" = "Protective"),
                      name   = "Effect on Longevity") +
  geom_text(aes(label = sprintf("p=%.3f", pval)),
            hjust = -0.2, size = 3, colour = "grey30") +
  labs(
    title    = "Mendelian Randomisation — Risk Factors → Longevity",
    subtitle = "IVW method | Effect on longevity (years)",
    x        = "MR Estimate (β, IVW)",
    y        = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")
ggsave(file.path(FIGURES_DIR, "05_MR_forest_plot.pdf"), p_forest,
       width = 9, height = 6, useDingbats = FALSE)

# ─── 6. Scatter Plots (LDL-C example) ────────────────────────────────────────

ldl_dat <- dat %>% filter(exposure == "LDL_C")
if (nrow(ldl_dat) > 3) {
  p_scatter <- mr_scatter_plot(
    mr(ldl_dat, method_list = c("mr_ivw","mr_egger_regression","mr_weighted_median")),
    ldl_dat
  )
  if (!is.null(p_scatter[[1]])) {
    p_sc <- p_scatter[[1]] +
      labs(title = "MR Scatter: LDL-C → Longevity",
           subtitle = "Each point = 1 genetic instrument") +
      theme_bw(base_size = 12)
    ggsave(file.path(FIGURES_DIR, "06_MR_LDL_scatter.pdf"), p_sc,
           width = 7, height = 6, useDingbats = FALSE)
  }
}

# ─── 7. Funnel Plot ──────────────────────────────────────────────────────────

ldl_mr <- mr(ldl_dat)
p_funnel <- mr_funnel_plot(mr_singlesnp(ldl_dat))
if (!is.null(p_funnel[[1]])) {
  p_f <- p_funnel[[1]] +
    labs(title = "MR Funnel Plot: LDL-C → Longevity",
         subtitle = "Asymmetry suggests pleiotropy") +
    theme_bw(base_size = 12)
  ggsave(file.path(FIGURES_DIR, "07_MR_LDL_funnel.pdf"), p_f,
         width = 6, height = 5, useDingbats = FALSE)
}

message("\n=== Mendelian Randomisation analysis complete ===")
print(mr_results_sig %>% select(exposure, b, se, pval, p_fdr))

sessionInfo()
