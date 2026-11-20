# ==========================================================
# survival_TFS_discordance.R
# Clean, unified driver for ML vs NCCN IPI concordance plots
# Generates forest + KM plots for all cohorts
# ==========================================================

source("R/helpers_concordance.R")

# ----------------------------------------------------------
# Load datasets (identical names as original workflow)
# ----------------------------------------------------------
datasets <- list(
  FCR             = "/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR.csv",
  FCR_all         = "/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all.csv",
  FCR_under_75    = "/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_under_75.csv",
  FCR_all_under_75 = "/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all_under_75.csv"
)

# ----------------------------------------------------------
# Run the concordance/discordance visualizations for each
# ----------------------------------------------------------
walk(names(datasets), function(name) {
  message("Processing cohort: ", name)

  df <- read_csv(datasets[[name]], show_col_types = FALSE)

  make_concordance_plots(
    df               = df,
    comparator_col   = "NCCN_categorical",
    comparator_label = "NCCN IPI",
    prefix           = paste0("TFS_", name)
  )
})

message("✅ All NCCN concordance plots generated successfully.")
