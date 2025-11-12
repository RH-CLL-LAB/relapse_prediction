# ==========================================================
# helpers_survival.R
# Shared utilities for survival analyses (TFS + OS)
# ==========================================================

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

# ---- Global constants ----
RISK_LEVELS  <- c("Low", "Low-Intermediate", "Intermediate-High", "High")
AGE_BREAKS   <- c(-Inf, 45, 60, 75, Inf)
AGE_LABELS   <- c("<45", "45-60", "60-75", "75<")
LTYPES       <- c(1, 1, 1, 1, 5, 5, 5, 5)
PALETTE_8    <- colorRampPalette(colors = brewer.pal(4, "RdBu"))(8)
PALETTE_USE  <- c(PALETTE_8[5:8], PALETTE_8[4:1])

# ==========================================================
# Basic preprocessing and model fitting
# ==========================================================

prep_km <- function(df) {
  df %>%
    mutate(
      event          = ifelse(event == 0, 0, 1),
      Predicted      = factor(risk_prediction, levels = RISK_LEVELS, ordered = TRUE),
      NCCN_IPI       = factor(NCCN_categorical, levels = RISK_LEVELS, ordered = TRUE),
      years_to_event = days_to_event / 365.25,
      age_category   = cut(age_at_tx, breaks = AGE_BREAKS, labels = AGE_LABELS)
    )
}

reverse_events <- function(df) df %>% mutate(event = ifelse(event == 0, 1, 0))

fit_models <- function(df, use_reverse = FALSE) {
  dat <- if (use_reverse) reverse_events(df) else df
  list(
    RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, dat),
    ML   = survfit(Surv(years_to_event, event) ~ Predicted, dat)
  )
}

# ==========================================================
# Concordance index & Brier score
# ==========================================================

cindex_brier <- function(df, time_point = 2) {
  num <- df %>%
    mutate(
      Predicted_numeric = as.numeric(Predicted),
      NCCN_IPI_numeric  = as.numeric(NCCN_IPI)
    )

  surv_obj <- with(num, Surv(years_to_event, event))
  c_ml  <- survConcordance(surv_obj ~ num$Predicted_numeric)
  c_ipi <- survConcordance(surv_obj ~ num$NCCN_IPI_numeric)

  prn_cindex <- function(cfit, label) {
    c_val <- cfit$concordance
    se    <- cfit$std.err
    ci_l  <- c_val - 1.96 * se
    ci_u  <- c_val + 1.96 * se
    cat(sprintf("%s C-index: %.3f (95%% CI: %.3f–%.3f)\n",
                label, round(c_val,3), round(ci_l,3), round(ci_u,3)))
  }

  prn_cindex(c_ml,  "ML")
  prn_cindex(c_ipi, "NCCN IPI")

  # ---- Brier score bootstrap ----
  ipi_idx <- which(!is.na(num$NCCN_IPI_numeric))
  cox_ml  <- coxph(Surv(years_to_event, event) ~ Predicted_numeric,  data = num)
  cox_ipi <- coxph(Surv(years_to_event, event) ~ NCCN_IPI_numeric,   data = num[ipi_idx, ])

  sf_ml  <- survfit(cox_ml,  newdata = num)
  sf_ipi <- survfit(cox_ipi, newdata = num[ipi_idx, ])

  num$ml_pred  <- as.vector(summary(sf_ml,  times = time_point)$surv)
  num$ipi_pred <- NA_real_
  num$ipi_pred[ipi_idx] <- as.vector(summary(sf_ipi, times = time_point)$surv)
  num$brier_obs <- as.numeric(num$years_to_event > time_point)

  bootstrap_brier <- function(pred_col, obs_col, data, n_boot = 1000) {
    scores <- replicate(n_boot, {
      idx <- sample(seq_len(nrow(data)), replace = TRUE)
      mean((data[[obs_col]][idx] - data[[pred_col]][idx])^2)
    })
    list(mean = mean(scores),
         lower = quantile(scores, 0.025),
         upper = quantile(scores, 0.975))
  }

  overlap <- num[complete.cases(num$ml_pred, num$ipi_pred, num$brier_obs), ]
  b_ml  <- bootstrap_brier("ml_pred",  "brier_obs", overlap)
  b_ipi <- bootstrap_brier("ipi_pred", "brier_obs", overlap)

  cat(sprintf("Brier Score at %d years:\n", time_point))
  cat(sprintf("ML model : %.3f (95%% CI: %.3f–%.3f)\n",
              round(b_ml$mean,3), round(b_ml$lower,3), round(b_ml$upper,3)))
  cat(sprintf("NCCN IPI : %.3f (95%% CI: %.3f–%.3f)\n",
              round(b_ipi$mean,3), round(b_ipi$lower,3), round(b_ipi$upper,3)))
}

# ==========================================================
# Age-stratified survival summaries (for heatmaps)
# ==========================================================

age_stratified_matrix <- function(df, time_point) {
  map_dfr(AGE_LABELS, function(age_cat) {
    sub <- df %>% filter(age_category == age_cat)
    fits <- fit_models(sub, use_reverse = FALSE)
    ripi_s <- summary(fits$RIPI, time = time_point, extend = TRUE)
    ml_s   <- summary(fits$ML,   time = time_point, extend = TRUE)

    ripi_df <- tibble(
      surv = ripi_s$surv, strata = ripi_s$strata, age_category = age_cat,
      n    = ripi_s$n,    model  = "NCCN IPI",
      upper = ripi_s$upper, lower = ripi_s$lower
    )
    ml_df <- tibble(
      surv = ml_s$surv, strata = ml_s$strata, age_category = age_cat,
      n    = ml_s$n,    model  = "ML",
      upper = ml_s$upper, lower = ml_s$lower
    )
    bind_rows(ripi_df, ml_df)
  }) %>%
    mutate(
      strata = case_when(
        str_detect(strata, "Low-Intermediate") ~ "Low-Intermediate",
        str_detect(strata, "Intermediate-High") ~ "Intermediate-High",
        str_detect(strata, "Low")  ~ "Low",
        str_detect(strata, "High") ~ "High"
      ),
      `Risk Group` = factor(strata, RISK_LEVELS),
      `Age Group`  = age_category,
      model = recode(model, "ML" = "ML[All]", "NCCN IPI" = "NCCN~IPI")
    )
}

# ==========================================================
# Heatmap export
# ==========================================================

export_heatmaps <- function(dfm, time_point, file_prefix, legend_title) {
  heatmap_plot <- ggplot(dfm, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) +
    geom_tile(color = "black") +
    geom_text(aes(label = percent(round(surv, 2), accuracy = 1)),
              color = "white", size = 9) +
    geom_text(aes(label = paste0("(", percent(round(lower, 2), accuracy = 1), "-",
                                 percent(round(upper, 2), accuracy = 1), ")")),
              color = "white", size = 5, vjust = 3) +
    facet_wrap(~model, labeller = label_parsed) +
    scale_fill_gradient(limits = c(0,1), high = muted("blue"),
                        low = muted("red", c = 300),
                        labels = percent_format(accuracy = 1)) +
    theme_minimal(base_size = 20) +
    guides(fill = guide_legend(title = legend_title)) +
    theme(legend.position = "bottom",
          legend.box = "horizontal",
          strip.text = element_text(size = 30, face = "bold"))

  heatmap_plot_n <- ggplot(dfm, aes(x = `Risk Group`, y = `Age Group`, fill = n)) +
    geom_tile(color = "black") +
    geom_text(aes(label = round(n, 2)), color = "white", size = 9) +
    facet_wrap(~model, labeller = label_parsed) +
    theme_minimal(base_size = 20) +
    guides(fill = guide_legend(title = "Number of patients")) +
    theme(legend.position = "bottom",
          legend.box = "horizontal",
          strip.text = element_text(size = 30, face = "bold"))

  ggexport(heatmap_plot,
           filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/", file_prefix, ".pdf"),
           width = 20, height = 10)
  ggexport(heatmap_plot_n,
           filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/", file_prefix, "_n.pdf"),
           width = 20, height = 10)
}

# ==========================================================
# Forest-plot export (param: heading_base for TFS / OS)
# ==========================================================

export_forest <- function(df, time_point, file_prefix, widths,
                          txt_size, point_size,
                          heading_base = "Treatment-free Survival") {
  fits <- fit_models(df, use_reverse = FALSE)
  ripi_s <- summary(fits$RIPI, time = time_point)
  ml_s   <- summary(fits$ML,   time = time_point)

  plotting_data <- tibble(
    n = ripi_s$n, level = RISK_LEVELS,
    estimate = ripi_s$surv, conf.low = ripi_s$lower,
    conf.high = ripi_s$upper, variable = "NCCN IPI"
  ) %>%
    bind_rows(tibble(
      n = ml_s$n, level = RISK_LEVELS,
      estimate = ml_s$surv, conf.low = ml_s$lower,
      conf.high = ml_s$upper, variable = "ML"
    ))

  label_list <- ifelse(plotting_data$variable == "NCCN IPI",
                       "bold('NCCN IPI')", "bold(ML[All])")

  panels <- list(
    forest_panel(width = widths[1]),
    forest_panel(width = widths[2], display = label_list, parse = TRUE, heading = "Model"),
    forest_panel(width = widths[3], display = level),
    forest_panel(width = widths[4], display = n, hjust = 1, heading = "N"),
    forest_panel(width = widths[5], item = "vline", hjust = 0.5),
    forest_panel(width = widths[6], item = "forest", hjust = 0.5,
                 heading = paste0(time_point, "-year ", heading_base)),
    forest_panel(width = widths[7], item = "vline", hjust = 0.5),
    forest_panel(width = widths[8],
                 display = sprintf("%d%% (%d%%–%d%%)",
                                   round(estimate*100),
                                   round(conf.low*100),
                                   round(conf.high*100)),
                 heading = "Percent (95% CI)")
  )

  forest_plot <- panel_forest_plot(
    plotting_data, limits = c(0, 1), panels = panels,
    trans = function(x) 100 * x,
    format_options = list(point_size = point_size,
                          text_size = txt_size,
                          colour = "black",
                          banded = TRUE,
                          suffix = "%", accuracy = 1)
  )

  ggexport(forest_plot,
           filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/",
                             file_prefix, "_", time_point, "_years.pdf"),
           width = 15, height = 6)
}

# ==========================================================
# Combined KM plot export (param: ylab_text for TFS / OS)
# ==========================================================

export_km_combine <- function(df, file_out, xlim_end = 5, size_line = 1.2,
                              ylab_text = "Probability of treatment-free survival") {
  fits <- fit_models(df, use_reverse = FALSE)
  fit_list <- list(RIPI = fits$RIPI, ML = fits$ML)

  km <- ggsurvplot_combine(
    fit_list, df,
    font.title = 40, font.subtitle = 20,
    legend = "bottom", legend.title = "Group",
    pval = TRUE,
    legend.labs.size = 6,
    legend.labs = c("NCCN IPI: Low Risk", "NCCN IPI: Low-Intermediate Risk",
                    "NCCN IPI: Intermediate-High Risk", "NCCN IPI: High Risk",
                    "ML: Low Risk", "ML: Low-Intermediate Risk",
                    "ML: Intermediate-High Risk", "ML: High Risk"),
    xlim = c(0, xlim_end),
    linetype = LTYPES,
    conf.int = TRUE,
    conf.int.style = "ribbon",
    palette = PALETTE_USE,
    risk.table = "percentage",
    risk.table.col = "strata",
    risk.table.y.text = FALSE,
    tables.height = 0.3,
    tables.theme = theme_cleantable(),
    ylab = ylab_text,
    xlab = "Time (years)",
    break.time.by = 1,
    fontsize = 7,
    risk.table.fontsize = 10,
    size = size_line,
    censor.size = 7
  ) %++% theme(plot.title = element_text(hjust = 0, vjust = 0.6))

  km$plot <- km$plot +
    geom_vline(xintercept = 2, linetype = "dashed", size = 0.8) +
    theme(axis.title.x = element_text(size = 25),
          axis.title.y = element_text(size = 25),
          axis.text.x  = element_text(size = 30),
          axis.text.y  = element_text(size = 30),
          legend.title = element_text(size = 20),
          legend.text  = element_text(size = 17))
  km$table <- km$table + theme(plot.title = element_text(size = 30))

  ggexport(km, filename = file_out, width = 20, height = 14)
}

# ==========================================================
# End of helpers_survival.R
# ==========================================================
