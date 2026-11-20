# ==========================================================
# survival_TFS.R
# Driver script for Treatment-Free Survival analyses
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

# ---- Import shared functions ----
source("R/helpers_survival.R")

# ==========================================================
# 1. FCR subset
# ==========================================================

km_lyfo <- read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR.csv") %>% 
  prep_km()

# Reverse-event KM fits (for printouts)
fits_rev <- fit_models(km_lyfo, use_reverse = TRUE)

# C-index + Brier at 2 years
cindex_brier(km_lyfo, time_point = 2)

# ---- Heatmaps ----
list(
  list(tp = 2,  file_prefix = "survival",           legend = "Treatment-free\nSurvival (2 years)"),
  list(tp = 5,  file_prefix = "survival_5_years",   legend = "Treatment-free\nSurvival (5 years)"),
  list(tp = 10, file_prefix = "survival_10_years",  legend = "Treatment-free\nSurvival (10 years)")
) %>% walk(function(p) {
  dfm <- age_stratified_matrix(km_lyfo, p$tp)
  export_heatmaps(dfm, p$tp, p$file_prefix, p$legend)
})

# ---- Forest plots ----
NARROW <- c(0.03, 0.03, 0.03, 0.02, 0.03, 0.90, 0.03, 0.02)
for (tp in c(2, 5, 10))
  export_forest(km_lyfo, tp, "forest_plot", NARROW,
                txt_size = 5, point_size = 5,
                heading_base = "Treatment-free Survival")

# ---- Combined KM plot ----
export_km_combine(km_lyfo,
  "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS.pdf",
  xlim_end = 5, size_line = 1.2,
  ylab_text = "Probability of treatment-free survival"
)

# ==========================================================
# 2. FCR_all
# ==========================================================

km_lyfo <- read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all.csv") %>%
  prep_km()

invisible(fit_models(km_lyfo, use_reverse = TRUE))
cindex_brier(km_lyfo, time_point = 2)

list(
  list(tp = 2,  file_prefix = "survival_all",           legend = "Treatment-free\nSurvival (2 years)"),
  list(tp = 5,  file_prefix = "survival_all_5_years",   legend = "Treatment-free\nSurvival (5 years)"),
  list(tp = 10, file_prefix = "survival_all_10_years",  legend = "Treatment-free\nSurvival (10 years)")
) %>% walk(function(p) {
  dfm <- age_stratified_matrix(km_lyfo, p$tp)
  export_heatmaps(dfm, p$tp, p$file_prefix, p$legend)
})

WIDE <- c(0.03, 0.10, 0.10, 0.05, 0.03, 0.55, 0.03, 0.07)
for (tp in c(2, 5, 10))
  export_forest(km_lyfo, tp, "forest_plot_all", WIDE,
                txt_size = 4, point_size = 5,
                heading_base = "Treatment-free Survival")

export_km_combine(km_lyfo,
  "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_all.pdf",
  xlim_end = 5, size_line = 1.2,
  ylab_text = "Probability of treatment-free survival"
)

# ==========================================================
# 3. FCR_under_75
# ==========================================================

km_lyfo <- read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_under_75.csv") %>%
  prep_km()

invisible(fit_models(km_lyfo, use_reverse = TRUE))

for (tp in c(2, 5, 10))
  export_forest(km_lyfo, tp, "forest_plot_under_75", WIDE,
                txt_size = 4, point_size = 5,
                heading_base = "Treatment-free Survival")

export_km_combine(km_lyfo,
  "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_under_75.pdf",
  xlim_end = 5, size_line = 1.2,
  ylab_text = "Probability of treatment-free survival"
)

# ==========================================================
# 4. FCR_all_under_75
# ==========================================================

km_lyfo <- read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all_under_75.csv") %>%
  prep_km()

invisible(fit_models(km_lyfo, use_reverse = TRUE))

for (tp in c(2, 5, 10))
  export_forest(km_lyfo, tp, "forest_plot_all_under_75", WIDE,
                txt_size = 4, point_size = 5,
                heading_base = "Treatment-free Survival")

export_km_combine(km_lyfo,
  "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_all_under_75.pdf",
  xlim_end = 5, size_line = 1.2,
  ylab_text = "Probability of treatment-free survival"
)

# ==========================================================
# End of survival_TFS.R
# ==========================================================
