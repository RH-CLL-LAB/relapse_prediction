# ==========================================================
# helpers_concordance.R
# Shared utilities for NCCN–ML concordance / discordance analyses
# ==========================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(survival)
  library(survminer)
  library(ggpubr)
  library(forestmodel)
  library(RColorBrewer)
  library(scales)
})

# ---- Constants ----
RISK_LEVELS <- c("Low", "Low-Intermediate", "Intermediate-High", "High")
PALETTE_DISCORD <- c("#E41A1C", "#009E73", "#CC79A7", "#E69F00")

# ==========================================================
# Create concordance/discordance categories
# ==========================================================

prep_concordance <- function(df) {
  df %>%
    mutate(
      event = ifelse(event == 0, 0, 1),
      Predicted = factor(risk_prediction, levels = RISK_LEVELS, ordered = TRUE),
      NCCN_IPI = factor(NCCN_categorical, levels = RISK_LEVELS, ordered = TRUE),
      years_to_event = days_to_event / 365.25,
      age_category = cut(age_at_tx, breaks = c(-Inf, 45, 60, 75, Inf),
                         labels = c("<45", "45–60", "60–75", "75<"))
    ) %>%
    filter(!is.na(NCCN_IPI)) %>%
    mutate(
      risk_category = case_when(
        (NCCN_IPI %in% c("Low","Low-Intermediate") &
           Predicted %in% c("Low","Low-Intermediate")) ~ "Lower",
        (NCCN_IPI %in% c("Intermediate-High","High") &
           Predicted %in% c("Intermediate-High","High")) ~ "Higher",
        (NCCN_IPI %in% c("Intermediate-High","High") &
           Predicted %in% c("Low","Low-Intermediate")) ~ "ML: Lower, NCCN IPI: Higher",
        (NCCN_IPI %in% c("Low","Low-Intermediate") &
           Predicted %in% c("Intermediate-High","High")) ~ "ML: Higher, NCCN IPI: Lower",
        TRUE ~ NA_character_
      )
    ) %>%
    drop_na(risk_category)
}

# ==========================================================
# Export forest plots across 2/5/10-year horizons
# ==========================================================

export_concordance_forest <- function(df, prefix, time_points = c(2, 5, 10)) {
  combined <- survfit(Surv(years_to_event, event) ~ risk_category, df)

  for (time in time_points) {
    sm <- summary(combined, time = time, extend = TRUE)
    pd <- tibble(
      n = sm$n,
      level = str_remove(names(combined$strata), "risk_category="),
      estimate = sm$surv,
      conf.low = sm$lower,
      conf.high = sm$upper,
      variable = rep(c("Concordant","Discordant"), each = 2)[1:length(sm$strata)],
      reference = FALSE
    )

    label_list <- case_when(
      pd$level == "Lower" ~ "Lower",
      pd$level == "Higher" ~ "Higher",
      pd$level == "ML: Higher, NCCN IPI: Lower" ~ "ML[All]:' Higher, NCCN IPI: Lower'",
      pd$level == "ML: Lower, NCCN IPI: Higher" ~ "ML[All]:' Lower, NCCN IPI: Higher'"
    )

    category_labels <- ifelse(pd$variable == "Concordant",
                              "bold('Concordant')", "bold('Discordant')")

    panels <- list(
      forest_panel(width = 0.03),
      forest_panel(width = 0.1, display = category_labels,
                   heading = "Concordance Category", parse = TRUE),
      forest_panel(width = 0.1, display = label_list, parse = TRUE),
      forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"),
      forest_panel(width = 0.03, item = "vline", hjust = 0.5),
      forest_panel(width = 0.55, item = "forest", hjust = 0.5,
                   heading = paste0(time, "-year Treatment-free Survival")),
      forest_panel(width = 0.03, item = "vline", hjust = 0.5),
      forest_panel(width = 0.07,
                   display = sprintf("%d%% (%d%%–%d%%)",
                                     round(pd$estimate * 100),
                                     round(pd$conf.low * 100),
                                     round(pd$conf.high * 100)),
                   heading = "Percent (95% CI)")
    )

    fp <- panel_forest_plot(
      pd, limits = c(0,1), panels = panels,
      trans = function(x) 100 * x,
      format_options = list(
        colour = "black", shape = 15, banded = TRUE,
        text_size = 3, point_size = 5,
        suffix = "%", accuracy = 1
      )
    )

    ggexport(fp,
             filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/",
                               prefix, "_", time, "_years.pdf"),
             width = 15, height = 6)
  }
}

# ==========================================================
# Export KM plot for concordance/discordance groups
# ==========================================================

export_concordance_km <- function(df, prefix) {
  fit <- list(combined = survfit(Surv(years_to_event, event) ~ risk_category, df))

  km <- ggsurvplot_combine(
    fit, df,
    font.title = 40,
    font.subtitle = 20,
    legend = "bottom",
    legend.title = "Group",
    pval = TRUE,
    legend.labs.size = 6,
    legend.labs = c("Concordant: Higher", "Concordant: Lower",
                    "Discordant: (ML: Higher, NCCN IPI: Lower)",
                    "Discordant: (ML: Lower, NCCN IPI: Higher)"),
    xlim = c(0,5),
    linetype = c(1,1,5,5),
    conf.int = TRUE,
    conf.int.style = "ribbon",
    palette = PALETTE_DISCORD,
    risk.table = "percentage",
    risk.table.col = "strata",
    risk.table.y.text = FALSE,
    tables.height = 0.3,
    tables.theme = theme_cleantable(),
    ylab = "Probability of treatment-free survival",
    xlab = "Time (years)",
    break.time.by = 1,
    fontsize = 7,
    risk.table.fontsize = 10,
    size = 1.2,
    censor.size = 7
  )

  km$plot <- km$plot +
    geom_vline(xintercept = 2, linetype = "dashed", size = 0.8) +
    theme(axis.title.x = element_text(size = 25),
          axis.title.y = element_text(size = 25),
          axis.text.x = element_text(size = 30),
          axis.text.y = element_text(size = 30),
          legend.title = element_text(size = 20),
          legend.text = element_text(size = 17))

  km$table <- km$table +
    theme(plot.title = element_text(size = 30))

  ggexport(km,
           filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/",
                             prefix, "_discordance_KM.pdf"),
           width = 20, height = 14)
}
