# ==========================================================
# survival_OS.R
# Driver script for Overall Survival analyses
# ==========================================================

source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")

suppressPackageStartupMessages({
  library(tidyverse)
  library(survival)
  library(survminer)
  library(ggpubr)
  library(forestmodel)
  library(scales)
  library(RColorBrewer)
  library(purrr)
})

# ---- Import shared helper functions ----
source("R/helpers_survival.R")

# ==========================================================
# 1. FCR subset (Overall Survival)
# ==========================================================

km_lyfo <- read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_OS.csv") %>%
  prep_km()

# Reverse-event KM fits (prints only)
fits_rev <- fit_models(km_lyfo, use_reverse = TRUE)

# C-index + Brier at 2 years
cindex_brier(km_lyfo, time_point = 2)

# ---- Heatmaps ----
list(
  list(tp = 2,  file_prefix = "survival_OS",           legend = "Overall\nSurvival (2 years)"),
  list(tp = 5,  file_prefix = "survival_5_years_OS",   legend = "Overall\nSurvival (5 years)"),
  list(tp = 10, file_prefix = "survival_10_years_OS",  legend = "Overall\nSurvival (10 years)")
) %>% walk(function(p) {
  dfm <- age_stratified_matrix(km_lyfo, p$tp)
  export_heatmaps(dfm, p$tp, p$file_prefix, p$legend)
})

# ---- Forest plots ----
NARROW <- c(0.005, 0.001, 0.001, 0.005, 0.02, 0.90, 0.02, 0.005)
for (tp in c(2, 5, 10))
  export_forest(km_lyfo, tp, "forest_plot_OS", NARROW,
                txt_size = 4.5, point_size = 5,
                heading_base = "Overall Survival")

# ---- Combined KM ----
export_km_combine(km_lyfo,
  "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_OS.pdf",
  xlim_end = 5, size_line = 1.2,
  ylab_text = "Probability of overall survival"
)

# ==========================================================
# 2. FCR_all (Overall Survival)
# ==========================================================

km_lyfo <- read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all_OS.csv") %>%
  prep_km()

invisible(fit_models(km_lyfo, use_reverse = TRUE))
cindex_brier(km_lyfo, time_point = 2)

list(
  list(tp = 2,  file_prefix = "survival_all_OS",           legend = "Overall\nSurvival (2 years)"),
  list(tp = 5,  file_prefix = "survival_all_5_years_OS",   legend = "Overall\nSurvival (5 years)"),
  list(tp = 10, file_prefix = "survival_all_10_years_OS",  legend = "Overall\nSurvival (10 years)")
) %>% walk(function(p) {
  dfm <- age_stratified_matrix(km_lyfo, p$tp)
  export_heatmaps(dfm, p$tp, p$file_prefix, p$legend)
})

for
