# ==========================================================
# survival_TFS_discordance_LR.R
# Clean, unified driver for ML vs Logistic Regression plots
# Generates forest + KM plots for all cohorts
# ==========================================================

source("R/helpers_concordance.R")

# ----------------------------------------------------------
# Load datasets (matching LR-based versions)
# ----------------------------------------------------------
datasets <- list(
  FCR              = "/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_LR.csv",
  FCR_all          = "/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all_LR.csv",
  FCR_under_75     = "/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_under_75_LR.csv",
  FCR_all_under_75 = "/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all_under_75_LR.csv"
)

# ----------------------------------------------------------
# Run the concordance/discordance visualizations for each
# ----------------------------------------------------------
walk(names(datasets), function(name) {
  message("Processing cohort: ", name)

  df <- read_csv(datasets[[name]], show_col_types = FALSE)

  make_concordance_plots(
    df               = df,
    comparator_col   = "lr_categorical",
    comparator_label = "LR",
    prefix           = paste0("TFS_", name, "_LR")
  )
})

message("✅ All Logistic Regression concordance plots generated successfully.")
