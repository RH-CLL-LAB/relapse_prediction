source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R")
library(scales)
library(RColorBrewer)
library(forestmodel)

km_lyfo = read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR.csv")

km_lyfo = km_lyfo %>%
  mutate(
    event = ifelse(event == 0, 0, 1),
    Predicted = factor(risk_prediction, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High"), ordered= TRUE),
    NCCN_IPI = factor(NCCN_categorical, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High"), ordered= TRUE),
    years_to_event = days_to_event / 365.25)

km_lyfo <- km_lyfo %>% 
  mutate(
    age_category = cut(age_at_tx, breaks = c(-Inf, 45, 60, 75, Inf), labels = c("<45", "45-60","60-75", "75<")))

km_reverse_lyfo = km_lyfo %>%
  mutate(event = ifelse(event == 0, 1, 0))

RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, km_reverse_lyfo)

ML = survfit(Surv(years_to_event, event) ~ Predicted, km_reverse_lyfo)

ripi_summary = summary(RIPI, time = 2, extend = TRUE)

ml_summary = summary(ML, time = 2, extend = TRUE)

## get C-index and brier score


km_lyfo_numeric <- km_lyfo %>%
  mutate(Predicted_numeric = as.numeric(Predicted),
         NCCN_IPI_numeric = as.numeric(NCCN_IPI))

surv_obj <- Surv(km_lyfo_numeric$years_to_event, km_lyfo_numeric$event)

c_index_ml <- survConcordance(surv_obj ~ km_lyfo_numeric$Predicted_numeric)
c_index_ipi <- survConcordance(surv_obj ~ km_lyfo_numeric$NCCN_IPI_numeric)


c_val <- c_index_ml$concordance
se <- c_index_ml$std.err

ci_lower <- c_val - 1.96 * se 
ci_upper <- c_val + 1.96 * se

cat("C-index:", round(c_val, 3),
    "(95% CI:", round(ci_lower, 3), "-", round(ci_upper,3), ")")

c_val <- c_index_ipi$concordance
se <- c_index_ipi$std.err

ci_lower <- c_val - 1.96 * se 
ci_upper <- c_val + 1.96 * se

cat("C-index:", round(c_val, 3),
    "(95% CI:", round(ci_lower, 3), "-", round(ci_upper,3), ")")

## BRIER SCORE

time_col <- "years_to_event"
event_col <- "event"
time_point <- 2  # e.g., 2 years

## IPI complete cases
ipi_complete_idx <- which(!is.na(km_lyfo_numeric$NCCN_IPI_numeric))

# --- Step 1: Fit each Cox model once ---

cox_ml <- coxph(Surv(years_to_event, event) ~ Predicted_numeric, data = km_lyfo_numeric)
cox_ipi <- coxph(Surv(years_to_event, event) ~ NCCN_IPI_numeric, data = km_lyfo_numeric[ipi_complete_idx,])

# --- Step 2: Predict survival probability at time_point ---

sf_ml <- survfit(cox_ml, newdata = km_lyfo_numeric)
km_lyfo_numeric$ml_pred <- as.vector(summary(sf_ml, times = time_point)$surv)


sf_ipi <- survfit(cox_ipi, newdata = km_lyfo_numeric[ipi_complete_idx, ])
km_lyfo_numeric$ipi_pred <- NA
km_lyfo_numeric$ipi_pred[ipi_complete_idx] <- as.vector(summary(sf_ipi, times = time_point)$surv)

# --- Step 3: Observed binary outcome: 1 if survived past time_point ---
km_lyfo_numeric$brier_obs <- as.numeric(km_lyfo_numeric[[time_col]] > time_point)

# --- Step 4: Bootstrapping function for any model ---
bootstrap_brier <- function(pred_col, obs_col, data, n_boot = 1000) {
  scores <- replicate(n_boot, {
    idx <- sample(1:nrow(data), replace = TRUE)
    mean((data[[obs_col]][idx] - data[[pred_col]][idx])^2)
  })
  
  list(
    mean = mean(scores),
    lower = quantile(scores, 0.025),
    upper = quantile(scores, 0.975)
  )
}

km_lyfo_numeric_overlap <- km_lyfo_numeric[complete.cases(km_lyfo_numeric$ml_pred, km_lyfo_numeric$ipi_pred, km_lyfo_numeric$brier_obs),]

# --- Step 5: Run bootstrap for both models ---
brier_ml <- bootstrap_brier("ml_pred", "brier_obs", km_lyfo_numeric_overlap)
brier_ipi <- bootstrap_brier("ipi_pred", "brier_obs", km_lyfo_numeric_overlap)

# --- Step 6: Print results ---
cat("Brier Score at", time_point, "years:\n")
cat("ML model     :", round(brier_ml$mean, 3), 
    "(95% CI:", round(brier_ml$lower, 3), "-", round(brier_ml$upper, 3), ")\n")
cat("NCCN IPI     :", round(brier_ipi$mean, 3), 
    "(95% CI:", round(brier_ipi$lower, 3), "-", round(brier_ipi$upper, 3), ")\n")

### PUT CONFIDENCE INTERVALS IN THE AGE STRATIFIED PLOTS

master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 2, extend = TRUE)
  
  ml_summary = summary(ML, time = 2, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI", upper = ripi_summary$upper, lower = ripi_summary$lower) # , higher = ripi_summary$higher,
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, lower = ml_summary$lower, upper = ml_summary$upper,  model = "ML") # 
  
  master_df <- master_df %>% rbind(
    ripi_dataframe
  ) %>%
    rbind(ml_dataframe)
  
}

master_df <- master_df %>% 
  mutate(strata = case_when(
    str_detect(strata, "Low-Intermediate") ~ "Low-Intermediate",
    str_detect(strata, "Intermediate-High") ~ "Intermediate-High",
    str_detect(strata, "Low") ~ "Low",
    str_detect(strata, "High") ~ "High"))

master_df <- master_df %>%
  mutate(`Risk Group` = factor(strata, c("Low", "Low-Intermediate","Intermediate-High" ,"High")),
         `Age Group` = age_category)

master_df <- master_df %>%
  mutate(model = recode(model, "ML" = "ML[All]", "NCCN IPI" = "NCCN~IPI"))


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = scales::percent(round(surv, 2), accuracy = 1)), color = "white", size = 9)+
  geom_text(aes(label = paste0("(", percent(round(lower, 2), accuracy = 1), "-", percent(round(upper, 2), accuracy=1), ")")),
            color = "white", size = 5, vjust = 3) +
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300), labels = scales::percent_format(accuracy=1)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Treatment-free\nSurvival (2 years)")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",
        strip.text = element_text(size = 30, face = "bold"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Number of patients")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",
        strip.text = element_text(size = 30, face = "bold"))


heatmap_plot %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_n.pdf",
           width = 20, height = 10)



master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 5, extend = TRUE)
  
  ml_summary = summary(ML, time = 5, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI", upper = ripi_summary$upper, lower = ripi_summary$lower) # , higher = ripi_summary$higher,
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, lower = ml_summary$lower, upper = ml_summary$upper,  model = "ML") # 
  
  master_df <- master_df %>% rbind(
    ripi_dataframe
  ) %>%
    rbind(ml_dataframe)
  
}

master_df <- master_df %>% 
  mutate(strata = case_when(
    str_detect(strata, "Low-Intermediate") ~ "Low-Intermediate",
    str_detect(strata, "Intermediate-High") ~ "Intermediate-High",
    str_detect(strata, "Low") ~ "Low",
    str_detect(strata, "High") ~ "High"))

master_df <- master_df %>%
  mutate(`Risk Group` = factor(strata, c("Low", "Low-Intermediate","Intermediate-High" ,"High")),
         `Age Group` = age_category)

master_df <- master_df %>%
  mutate(model = recode(model, "ML" = "ML[All]", "NCCN IPI" = "NCCN~IPI"))


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = scales::percent(round(surv, 2), accuracy = 1)), color = "white", size = 9)+
  geom_text(aes(label = paste0("(", percent(round(lower, 2), accuracy = 1), "-", percent(round(upper, 2), accuracy=1), ")")),
            color = "white", size = 5, vjust = 3) +
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300), labels = scales::percent_format(accuracy=1)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Treatment-free\nSurvival (5 years)")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))

heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Number of patients")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))


heatmap_plot %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_5_years.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_n_5_years.pdf",
           width = 20, height = 10)


master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 10, extend = TRUE)
  
  ml_summary = summary(ML, time = 10, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI", upper = ripi_summary$upper, lower = ripi_summary$lower) # , higher = ripi_summary$higher,
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, lower = ml_summary$lower, upper = ml_summary$upper,  model = "ML") # 
  
  master_df <- master_df %>% rbind(
    ripi_dataframe
  ) %>%
    rbind(ml_dataframe)
  
}

master_df <- master_df %>% 
  mutate(strata = case_when(
    str_detect(strata, "Low-Intermediate") ~ "Low-Intermediate",
    str_detect(strata, "Intermediate-High") ~ "Intermediate-High",
    str_detect(strata, "Low") ~ "Low",
    str_detect(strata, "High") ~ "High"))

master_df <- master_df %>%
  mutate(`Risk Group` = factor(strata, c("Low", "Low-Intermediate","Intermediate-High" ,"High")),
         `Age Group` = age_category)


master_df <- master_df %>%
  mutate(model = recode(model, "ML" = "ML[All]", "NCCN IPI" = "NCCN~IPI"))


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = scales::percent(round(surv, 2), accuracy = 1)), color = "white", size = 9)+
  geom_text(aes(label = paste0("(", percent(round(lower, 2), accuracy = 1), "-", percent(round(upper, 2), accuracy=1), ")")),
            color = "white", size = 5, vjust = 3) +
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300), labels = scales::percent_format(accuracy=1)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Treatment-free\nSurvival (10 years)")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Number of patients")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))


heatmap_plot %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_10_years.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_n_10_years.pdf",
           width = 20, height = 10)




RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, km_lyfo)

ML = survfit(Surv(years_to_event, event) ~ Predicted, km_lyfo)

for (time in c(2, 5, 10)){
  
  ripi_summary <- summary(RIPI, time = time)
  
  ml_summary <- summary(ML, time = time)
  
  plotting_data = data.frame(n = ripi_summary$n, 
                             level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                             level_no = c(1,2,3,4),
                             class = c("factor", "factor", "factor", "factor"),
                             estimate = ripi_summary$surv,
                             conf.low = ripi_summary$lower,
                             conf.high = ripi_summary$upper,
                             std.error = ripi_summary$std.err,
                             p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                             variable = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             label = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             reference = c(F, F, F, F)) %>%
    rbind(data.frame(n = ml_summary$n, 
                     level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                     level_no = c(1,2,3,4),
                     class = c("factor", "factor", "factor", "factor"),
                     estimate = ml_summary$surv,
                     conf.low = ml_summary$lower,
                     conf.high = ml_summary$upper,
                     std.error = ml_summary$std.err,
                     p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                     variable = c("ML", NA_character_, NA_character_, NA_character_),
                     label = c("ML", NA_character_, NA_character_, NA_character_),
                     reference = c(F, F, F,F)))
  
  label_list <- ifelse(plotting_data$variable == "NCCN IPI",
                       "bold('NCCN IPI')", "bold(ML[All])")
  
  
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.03, display = label_list, parse = TRUE, heading = "Model"), 
                 forest_panel(width = 0.03,  display = level), 
                 forest_panel(width = 0.02,  display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.90, item = "forest", hjust = 0.5, heading = paste0(time, "-year Treatment-free Survival")), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.02, display = if_else(reference, "Reference", sprintf("%d%% (%d%%-%d%%)", round(trans(estimate)), round(trans(conf.low)), round(trans(conf.high)))), display_na = NA, heading = "Percent (95% CI)")) 
  #### SEE BELOW FOR HEADER CHANGE TO P-VALUE
  #forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
  ####
  #forest_panel(width = 0.00))
  
  forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels, 
                                   trans = function(x) 100 * x, 
                                   format_options = 
                                     list(
                                       colour = "black",
                                       color = NULL,
                                       shape = 15,
                                       text_size = 5,
                                       point_size = 5,
                                       banded = TRUE,
                                       suffix = "%",
                                       accuracy = 1
                                     ))
  
  forest_plot %>%
    ggexport(filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/forest_plot_",time,"_years.pdf"),
             width = 15, height = 6)
  
}

fit = list(RIPI = RIPI, ML = ML)

cols <- colorRampPalette( colors = brewer.pal(4,"RdBu") )
cols <- cols(8)

# cols <- cols
# 
# #custom_palette <- c(cols[1], cols[3], cols[7:8], cols[1], cols[3], cols[7:8])
# 
# custom_palette <- c(cols[1], cols[3], cols[7:8])
# 
# custom_palette_adjusted <- c("#CA0021", "#EE8D74", "#4195C4", "#0571B1")
# 
# custom_palette_combined <- c(custom_palette, custom_palette_adjusted)

cols <- c(cols[5:8], cols[4:1])

RIPI_KM = ggsurvplot_combine(fit, km_lyfo,
                             #title = "Treatment-free survival after first-line treatment",
                             #subtitle="All lymphoma patients under 75 years old", # DLBCL patients treated with R-CHOP-like treatment under 75 years old
                             font.title = 40, 
                             font.subtitle=20,
                             legend = "bottom",
                             legend.title = "Group",
                             pval=TRUE,
                             legend.labs.size = 6,
                             legend.labs = c("NCCN IPI: Low Risk","NCCN IPI: Low-Intermediate Risk","NCCN IPI: Intermediate-High Risk","NCCN IPI: High Risk", "ML: Low Risk","ML: Low-Intermediate Risk","ML: Intermediate-High Risk","ML: High Risk"),
                             xlim = c(0, 5),
                             linetype=c(1,1,1,1,5,5,5,5),
                             conf.int = TRUE,                    # Add confidence interval
                             conf.int.style = "ribbon",            # CI style, use "step" or "ribbon"
                             palette = cols,
                             risk.table = "percentage",
                             risk.table.col="strata",
                             risk.table.y.text = FALSE,
                             tables.height = 0.3,
                             tables.theme = theme_cleantable(),
                             ylab = ("Probability of treatment-free survival"),
                             xlab = "Time (years)",
                             break.time.by = 1,
                             fontsize = 7, # 9
                             risk.table.fontsize = 10,
                             linewidth = 1.2,
                             censor.size=7
) %++%
  theme(
    plot.title = element_text(hjust = 0, vjust = 0.6) ) 

# KM_OS$plot + geom_vline(xintercept = 2, linetype = "dashed")

# KM_OS$plot + geom_vline(xintercept = 2, linetype = "dashed")
RIPI_KM$plot <- RIPI_KM$plot + 
  geom_vline(xintercept = 2, linetype = "dashed", size = 0.8) + 
  #geom_segment(x = -1, xend = 2, y =ripi_summary$surv[1], yend = ripi_summary$surv[1],linetype="dotdash", color = custom_pallete[1], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ripi_summary$surv[2], yend = ripi_summary$surv[2],linetype="dotdash", color = custom_pallete[2], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ml_summary$surv[1], yend = ml_summary$surv[1],linetype="dotdash", color = custom_pallete[3], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ml_summary$surv[2], yend = ml_summary$surv[2],linetype="dotdash", color = custom_pallete[4], size = 0.8) +
  #geom_hline(yintercept=ripi_summary$surv[1],linetype="dotted", color = custom_pallete[1])+
  #geom_hline(yintercept=ripi_summary$surv[2],linetype="dotted", color = custom_pallete[2])+
  #geom_hline(yintercept=ml_summary$surv[1],linetype="dotted", color = custom_pallete[3])+
  #geom_hline(yintercept=ml_summary$surv[2],linetype="dotted", color = custom_pallete[4])+
  #annotate("segment", x=0, xend = 2, y=0.6331, yend=0.4)+ 
  theme(
    axis.title.x = element_text(size = 25),  
    axis.title.y = element_text(size = 25),
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 17)
  )

RIPI_KM$table <- RIPI_KM$table +
  theme(plot.title = element_text(size = 30)
  ) 

RIPI_KM %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS.pdf",
           width =8*2.5, height =7*2)


km_lyfo = read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all.csv")

km_lyfo = km_lyfo %>%
  mutate(
    event = ifelse(event == 0, 0, 1),
    Predicted = factor(risk_prediction, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    NCCN_IPI = factor(NCCN_categorical, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    years_to_event = days_to_event / 365.25)

km_lyfo <- km_lyfo %>% 
  mutate(
    age_category = cut(age_at_tx, breaks = c(-Inf, 45, 60, 75, Inf), labels = c("<45", "45-60","60-75", "75<")))

km_reverse_lyfo = km_lyfo %>%
  mutate(event = ifelse(event == 0, 1, 0))

RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, km_reverse_lyfo)

ML = survfit(Surv(years_to_event, event) ~ Predicted, km_reverse_lyfo)

RIPI

ML


## get C-index and brier score


km_lyfo_numeric <- km_lyfo %>%
  mutate(Predicted_numeric = as.numeric(Predicted),
         NCCN_IPI_numeric = as.numeric(NCCN_IPI))

surv_obj <- Surv(km_lyfo_numeric$years_to_event, km_lyfo_numeric$event)

c_index_ml <- survConcordance(surv_obj ~ km_lyfo_numeric$Predicted_numeric)
c_index_ipi <- survConcordance(surv_obj ~ km_lyfo_numeric$NCCN_IPI_numeric)


c_val <- c_index_ml$concordance
se <- c_index_ml$std.err

ci_lower <- c_val - 1.96 * se 
ci_upper <- c_val + 1.96 * se

cat("C-index:", round(c_val, 3),
    "(95% CI:", round(ci_lower, 3), "-", round(ci_upper,3), ")")

c_val <- c_index_ipi$concordance
se <- c_index_ipi$std.err

ci_lower <- c_val - 1.96 * se 
ci_upper <- c_val + 1.96 * se

cat("C-index:", round(c_val, 3),
    "(95% CI:", round(ci_lower, 3), "-", round(ci_upper,3), ")")

## BRIER SCORE

time_col <- "years_to_event"
event_col <- "event"
time_point <- 2  # e.g., 2 years

## IPI complete cases
ipi_complete_idx <- which(!is.na(km_lyfo_numeric$NCCN_IPI_numeric))

# --- Step 1: Fit each Cox model once ---

cox_ml <- coxph(Surv(years_to_event, event) ~ Predicted_numeric, data = km_lyfo_numeric)
cox_ipi <- coxph(Surv(years_to_event, event) ~ NCCN_IPI_numeric, data = km_lyfo_numeric[ipi_complete_idx,])

# --- Step 2: Predict survival probability at time_point ---

sf_ml <- survfit(cox_ml, newdata = km_lyfo_numeric)
km_lyfo_numeric$ml_pred <- as.vector(summary(sf_ml, times = time_point)$surv)


sf_ipi <- survfit(cox_ipi, newdata = km_lyfo_numeric[ipi_complete_idx, ])
km_lyfo_numeric$ipi_pred <- NA
km_lyfo_numeric$ipi_pred[ipi_complete_idx] <- as.vector(summary(sf_ipi, times = time_point)$surv)

# --- Step 3: Observed binary outcome: 1 if survived past time_point ---
km_lyfo_numeric$brier_obs <- as.numeric(km_lyfo_numeric[[time_col]] > time_point)

# --- Step 4: Bootstrapping function for any model ---
bootstrap_brier <- function(pred_col, obs_col, data, n_boot = 1000) {
  scores <- replicate(n_boot, {
    idx <- sample(1:nrow(data), replace = TRUE)
    mean((data[[obs_col]][idx] - data[[pred_col]][idx])^2)
  })
  
  list(
    mean = mean(scores),
    lower = quantile(scores, 0.025),
    upper = quantile(scores, 0.975)
  )
}

km_lyfo_numeric_overlap <- km_lyfo_numeric[complete.cases(km_lyfo_numeric$ml_pred, km_lyfo_numeric$ipi_pred, km_lyfo_numeric$brier_obs),]

# --- Step 5: Run bootstrap for both models ---
brier_ml <- bootstrap_brier("ml_pred", "brier_obs", km_lyfo_numeric_overlap)
brier_ipi <- bootstrap_brier("ipi_pred", "brier_obs", km_lyfo_numeric_overlap)

# --- Step 6: Print results ---
cat("Brier Score at", time_point, "years:\n")
cat("ML model     :", round(brier_ml$mean, 3), 
    "(95% CI:", round(brier_ml$lower, 3), "-", round(brier_ml$upper, 3), ")\n")
cat("NCCN IPI     :", round(brier_ipi$mean, 3), 
    "(95% CI:", round(brier_ipi$lower, 3), "-", round(brier_ipi$upper, 3), ")\n")



master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 2, extend = TRUE)
  
  ml_summary = summary(ML, time = 2, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI", upper = ripi_summary$upper, lower = ripi_summary$lower) # , higher = ripi_summary$higher,
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, lower = ml_summary$lower, upper = ml_summary$upper,  model = "ML") # 
  
  master_df <- master_df %>% rbind(
    ripi_dataframe
  ) %>%
    rbind(ml_dataframe)
  
}

master_df <- master_df %>% 
  mutate(strata = case_when(
    str_detect(strata, "Low-Intermediate") ~ "Low-Intermediate",
    str_detect(strata, "Intermediate-High") ~ "Intermediate-High",
    str_detect(strata, "Low") ~ "Low",
    str_detect(strata, "High") ~ "High"))

master_df <- master_df %>%
  mutate(`Risk Group` = factor(strata, c("Low", "Low-Intermediate","Intermediate-High" ,"High")),
         `Age Group` = age_category)

master_df <- master_df %>%
  mutate(model = recode(model, "ML" = "ML[All]", "NCCN IPI" = "NCCN~IPI"))


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = scales::percent(round(surv, 2), accuracy = 1)), color = "white", size = 9)+
  geom_text(aes(label = paste0("(", percent(round(lower, 2), accuracy = 1), "-", percent(round(upper, 2), accuracy=1), ")")),
            color = "white", size = 5, vjust = 3) +
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300), labels = scales::percent_format(accuracy=1)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Treatment-free\nSurvival (2 years)")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Number of patients")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))

heatmap_plot %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_all.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_all_n.pdf",
           width = 20, height = 10)


master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 5, extend = TRUE)
  
  ml_summary = summary(ML, time = 5, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI", upper = ripi_summary$upper, lower = ripi_summary$lower) # , higher = ripi_summary$higher,
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, lower = ml_summary$lower, upper = ml_summary$upper,  model = "ML") # 
  
  master_df <- master_df %>% rbind(
    ripi_dataframe
  ) %>%
    rbind(ml_dataframe)
  
}

master_df <- master_df %>% 
  mutate(strata = case_when(
    str_detect(strata, "Low-Intermediate") ~ "Low-Intermediate",
    str_detect(strata, "Intermediate-High") ~ "Intermediate-High",
    str_detect(strata, "Low") ~ "Low",
    str_detect(strata, "High") ~ "High"))

master_df <- master_df %>%
  mutate(`Risk Group` = factor(strata, c("Low", "Low-Intermediate","Intermediate-High" ,"High")),
         `Age Group` = age_category)

master_df <- master_df %>%
  mutate(model = recode(model, "ML" = "ML[All]", "NCCN IPI" = "NCCN~IPI"))


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = scales::percent(round(surv, 2), accuracy = 1)), color = "white", size = 9)+
  geom_text(aes(label = paste0("(", percent(round(lower, 2), accuracy = 1), "-", percent(round(upper, 2), accuracy=1), ")")),
            color = "white", size = 5, vjust = 3) +
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300), labels = scales::percent_format(accuracy=1)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Treatment-free\nSurvival (5 years)")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Number of patients")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))

heatmap_plot %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_all_5_years.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_all_n_5_years.pdf",
           width = 20, height = 10)


master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 10, extend = TRUE)
  
  ml_summary = summary(ML, time = 10, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI", upper = ripi_summary$upper, lower = ripi_summary$lower) # , higher = ripi_summary$higher,
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, lower = ml_summary$lower, upper = ml_summary$upper,  model = "ML") # 
  
  master_df <- master_df %>% rbind(
    ripi_dataframe
  ) %>%
    rbind(ml_dataframe)
  
}

master_df <- master_df %>% 
  mutate(strata = case_when(
    str_detect(strata, "Low-Intermediate") ~ "Low-Intermediate",
    str_detect(strata, "Intermediate-High") ~ "Intermediate-High",
    str_detect(strata, "Low") ~ "Low",
    str_detect(strata, "High") ~ "High"))

master_df <- master_df %>%
  mutate(`Risk Group` = factor(strata, c("Low", "Low-Intermediate","Intermediate-High" ,"High")),
         `Age Group` = age_category)

master_df <- master_df %>%
  mutate(model = recode(model, "ML" = "ML[All]", "NCCN IPI" = "NCCN~IPI"))

heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = scales::percent(round(surv, 2), accuracy = 1)), color = "white", size = 9)+
  geom_text(aes(label = paste0("(", percent(round(lower, 2), accuracy = 1), "-", percent(round(upper, 2), accuracy=1), ")")),
            color = "white", size = 5, vjust = 3) +
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300), labels = scales::percent_format(accuracy=1)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Treatment-free\nSurvival (10 years)")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model, labeller = label_parsed) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="Number of patients")) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",         strip.text = element_text(size = 30, face = "bold"))

heatmap_plot %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_all_10_years.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/survival_all_n_10_years.pdf",
           width = 20, height = 10)



RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, km_lyfo)

ML = survfit(Surv(years_to_event, event) ~ Predicted, km_lyfo)

for (time in c(2, 5, 10)){
  
  ripi_summary <- summary(RIPI, time = time)
  
  ml_summary <- summary(ML, time = time)
  
  plotting_data = data.frame(n = ripi_summary$n, 
                             level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                             level_no = c(1,2,3,4),
                             class = c("factor", "factor", "factor", "factor"),
                             estimate = ripi_summary$surv,
                             conf.low = ripi_summary$lower,
                             conf.high = ripi_summary$upper,
                             std.error = ripi_summary$std.err,
                             p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                             variable = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             label = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             reference = c(F, F, F, F)) %>%
    rbind(data.frame(n = ml_summary$n, 
                     level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                     level_no = c(1,2,3,4),
                     class = c("factor", "factor", "factor", "factor"),
                     estimate = ml_summary$surv,
                     conf.low = ml_summary$lower,
                     conf.high = ml_summary$upper,
                     std.error = ml_summary$std.err,
                     p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                     variable = c("ML", NA_character_, NA_character_, NA_character_),
                     label = c("ML", NA_character_, NA_character_, NA_character_),
                     reference = c(F, F, F,F)))
  
  label_list <- ifelse(plotting_data$variable == "NCCN IPI",
                       "bold('NCCN IPI')", "bold(ML[All])")
  
  
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = label_list, parse = TRUE, heading = "Model"), 
                 forest_panel(width = 0.1, display = level), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, heading = paste0(time, "-year Treatment-free Survival")), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.07, display = if_else(reference, "Reference", sprintf("%d%% (%d%%-%d%%)", round(trans(estimate)), round(trans(conf.low)), round(trans(conf.high)))), display_na = NA, heading = "Percent (95% CI)")) 
  #### SEE BELOW FOR HEADER CHANGE TO P-VALUE
  #forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
  ####
  #forest_panel(width = 0.00))
  
  forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels, 
                                   trans = function(x) 100 * x, 
                                   format_options = 
                                     list(
                                       colour = "black",
                                       color = NULL,
                                       shape = 15,
                                       text_size = 4,
                                       point_size = 5,
                                       banded = TRUE,
                                       suffix = "%",
                                       accuracy = 1
                                     ))

forest_plot %>%
  ggexport(filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/forest_plot_all_",time,"_years.pdf"),
           width = 15, height = 6)

}

fit = list(RIPI = RIPI, ML = ML)

cols <- colorRampPalette( colors = brewer.pal(4,"RdBu") )
cols <- cols(8)

# cols <- cols
# 
# #custom_palette <- c(cols[1], cols[3], cols[7:8], cols[1], cols[3], cols[7:8])
# 
# custom_palette <- c(cols[1], cols[3], cols[7:8])
# 
# custom_palette_adjusted <- c("#CA0021", "#EE8D74", "#4195C4", "#0571B1")
# 
# custom_palette_combined <- c(custom_palette, custom_palette_adjusted)

cols <- c(cols[5:8], cols[4:1])

RIPI_KM = ggsurvplot_combine(fit, km_lyfo,
                             #title = "Treatment-free survival after first-line treatment",
                             #subtitle="All lymphoma patients under 75 years old", # DLBCL patients treated with R-CHOP-like treatment under 75 years old
                             font.title = 40, 
                             font.subtitle=20,
                             legend = "bottom",
                             legend.title = "Group",
                             pval=TRUE,
                             legend.labs.size = 6,
                             legend.labs = c("NCCN IPI: Low Risk","NCCN IPI: Low-Intermediate Risk","NCCN IPI: Intermediate-High Risk","NCCN IPI: High Risk", "ML: Low Risk","ML: Low-Intermediate Risk","ML: Intermediate-High Risk","ML: High Risk"),
                             xlim = c(0, 5),
                             linetype=c(1,1,1,1,5,5,5,5),
                             conf.int = TRUE,                    # Add confidence interval
                             conf.int.style = "ribbon",            # CI style, use "step" or "ribbon"
                             palette = cols,
                             risk.table = "percentage",
                             risk.table.col="strata",
                             risk.table.y.text = FALSE,
                             tables.height = 0.3,
                             tables.theme = theme_cleantable(),
                             ylab = ("Probability of treatment-free survival"),
                             xlab = "Time (years)",
                             break.time.by = 1,
                             fontsize = 7, # 9
                             risk.table.fontsize = 10,
                             size = 1.2,
                             censor.size=7
) %++%
  theme(
    plot.title = element_text(hjust = 0, vjust = 0.6) ) 

# KM_OS$plot + geom_vline(xintercept = 2, linetype = "dashed")

# KM_OS$plot + geom_vline(xintercept = 2, linetype = "dashed")
RIPI_KM$plot <- RIPI_KM$plot + 
  geom_vline(xintercept = 2, linetype = "dashed", size = 0.8) + 
  #geom_segment(x = -1, xend = 2, y =ripi_summary$surv[1], yend = ripi_summary$surv[1],linetype="dotdash", color = custom_pallete[1], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ripi_summary$surv[2], yend = ripi_summary$surv[2],linetype="dotdash", color = custom_pallete[2], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ml_summary$surv[1], yend = ml_summary$surv[1],linetype="dotdash", color = custom_pallete[3], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ml_summary$surv[2], yend = ml_summary$surv[2],linetype="dotdash", color = custom_pallete[4], size = 0.8) +
  #geom_hline(yintercept=ripi_summary$surv[1],linetype="dotted", color = custom_pallete[1])+
  #geom_hline(yintercept=ripi_summary$surv[2],linetype="dotted", color = custom_pallete[2])+
  #geom_hline(yintercept=ml_summary$surv[1],linetype="dotted", color = custom_pallete[3])+
  #geom_hline(yintercept=ml_summary$surv[2],linetype="dotted", color = custom_pallete[4])+
  #annotate("segment", x=0, xend = 2, y=0.6331, yend=0.4)+ 
  theme(
    axis.title.x = element_text(size = 25),  
    axis.title.y = element_text(size = 25),
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 17)
  )

RIPI_KM$table <- RIPI_KM$table +
  theme(plot.title = element_text(size = 30)
  ) 


RIPI_KM %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_all.pdf",
           width =8*2.5, height =7*2)


km_lyfo = read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_under_75.csv")

km_lyfo = km_lyfo %>%
  mutate(
    event = ifelse(event == 0, 0, 1),
    Predicted = factor(risk_prediction, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    NCCN_IPI = factor(NCCN_categorical, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    years_to_event = days_to_event / 365.25)


km_reverse_lyfo = km_lyfo %>%
  mutate(event = ifelse(event == 0, 1, 0))

RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, km_reverse_lyfo)

ML = survfit(Surv(years_to_event, event) ~ Predicted, km_reverse_lyfo)

RIPI

ML



RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, km_lyfo)

ML = survfit(Surv(years_to_event, event) ~ Predicted, km_lyfo)

for (time in c(2, 5, 10)){
  
  ripi_summary <- summary(RIPI, time = time)
  
  ml_summary <- summary(ML, time = time)
  
  plotting_data = data.frame(n = ripi_summary$n, 
                             level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                             level_no = c(1,2,3,4),
                             class = c("factor", "factor", "factor", "factor"),
                             estimate = ripi_summary$surv,
                             conf.low = ripi_summary$lower,
                             conf.high = ripi_summary$upper,
                             std.error = ripi_summary$std.err,
                             p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                             variable = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             label = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             reference = c(F, F, F, F)) %>%
    rbind(data.frame(n = ml_summary$n, 
                     level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                     level_no = c(1,2,3,4),
                     class = c("factor", "factor", "factor", "factor"),
                     estimate = ml_summary$surv,
                     conf.low = ml_summary$lower,
                     conf.high = ml_summary$upper,
                     std.error = ml_summary$std.err,
                     p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                     variable = c("ML", NA_character_, NA_character_, NA_character_),
                     label = c("ML", NA_character_, NA_character_, NA_character_),
                     reference = c(F, F, F,F)))
  
  label_list <- ifelse(plotting_data$variable == "NCCN IPI",
                       "bold('NCCN IPI')", "bold(ML[All])")
  
  
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = label_list, parse = TRUE, heading = "Model"), 
                 forest_panel(width = 0.1, display = level), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, heading = paste0(time, "-year Treatment-free Survival")), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.07, display = if_else(reference, "Reference", sprintf("%d%% (%d%%-%d%%)", round(trans(estimate)), round(trans(conf.low)), round(trans(conf.high)))), display_na = NA, heading = "Percent (95% CI)")) 
  #### SEE BELOW FOR HEADER CHANGE TO P-VALUE
  #forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
  ####
  #forest_panel(width = 0.00))
  
  forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels, 
                                   trans = function(x) 100 * x, 
                                   format_options = 
                                     list(
                                       colour = "black",
                                       color = NULL,
                                       shape = 15,
                                       text_size = 4,
                                       point_size = 5,
                                       banded = TRUE,
                                       suffix = "%",
                                       accuracy = 1
                                     ))
  
  forest_plot %>%
    ggexport(filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/forest_plot_under_75_",time,"_years.pdf"),
             width = 15, height = 6)
  
}

fit = list(RIPI = RIPI, ML = ML)

cols <- colorRampPalette( colors = brewer.pal(4,"RdBu") )
cols <- cols(8)

# cols <- cols
# 
# #custom_palette <- c(cols[1], cols[3], cols[7:8], cols[1], cols[3], cols[7:8])
# 
# custom_palette <- c(cols[1], cols[3], cols[7:8])
# 
# custom_palette_adjusted <- c("#CA0021", "#EE8D74", "#4195C4", "#0571B1")
# 
# custom_palette_combined <- c(custom_palette, custom_palette_adjusted)

cols <- c(cols[5:8], cols[4:1])

RIPI_KM = ggsurvplot_combine(fit, km_lyfo,
                             #title = "Treatment-free survival after first-line treatment",
                             #subtitle="All lymphoma patients under 75 years old", # DLBCL patients treated with R-CHOP-like treatment under 75 years old
                             font.title = 40, 
                             font.subtitle=20,
                             legend = "bottom",
                             legend.title = "Group",
                             pval=TRUE,
                             legend.labs.size = 6,
                             legend.labs = c("NCCN IPI: Low Risk","NCCN IPI: Low-Intermediate Risk","NCCN IPI: Intermediate-High Risk","NCCN IPI: High Risk", "ML: Low Risk","ML: Low-Intermediate Risk","ML: Intermediate-High Risk","ML: High Risk"),
                             xlim = c(0, 5),
                             linetype=c(1,1,1,1,5,5,5,5),
                             conf.int = TRUE,                    # Add confidence interval
                             conf.int.style = "ribbon",            # CI style, use "step" or "ribbon"
                             palette = cols,
                             risk.table = "percentage",
                             risk.table.col="strata",
                             risk.table.y.text = FALSE,
                             tables.height = 0.3,
                             tables.theme = theme_cleantable(),
                             ylab = ("Probability of treatment-free survival"),
                             xlab = "Time (years)",
                             break.time.by = 1,
                             fontsize = 7, # 9
                             risk.table.fontsize = 10,
                             size = 1.2,
                             censor.size=7
) %++%
  theme(
    plot.title = element_text(hjust = 0, vjust = 0.6) ) 

# KM_OS$plot + geom_vline(xintercept = 2, linetype = "dashed")

# KM_OS$plot + geom_vline(xintercept = 2, linetype = "dashed")
RIPI_KM$plot <- RIPI_KM$plot + 
  geom_vline(xintercept = 2, linetype = "dashed", size = 0.8) + 
  #geom_segment(x = -1, xend = 2, y =ripi_summary$surv[1], yend = ripi_summary$surv[1],linetype="dotdash", color = custom_pallete[1], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ripi_summary$surv[2], yend = ripi_summary$surv[2],linetype="dotdash", color = custom_pallete[2], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ml_summary$surv[1], yend = ml_summary$surv[1],linetype="dotdash", color = custom_pallete[3], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ml_summary$surv[2], yend = ml_summary$surv[2],linetype="dotdash", color = custom_pallete[4], size = 0.8) +
  #geom_hline(yintercept=ripi_summary$surv[1],linetype="dotted", color = custom_pallete[1])+
  #geom_hline(yintercept=ripi_summary$surv[2],linetype="dotted", color = custom_pallete[2])+
  #geom_hline(yintercept=ml_summary$surv[1],linetype="dotted", color = custom_pallete[3])+
  #geom_hline(yintercept=ml_summary$surv[2],linetype="dotted", color = custom_pallete[4])+
  #annotate("segment", x=0, xend = 2, y=0.6331, yend=0.4)+ 
  theme(
    axis.title.x = element_text(size = 25),  
    axis.title.y = element_text(size = 25),
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 17)
  )

RIPI_KM$table <- RIPI_KM$table +
  theme(plot.title = element_text(size = 30)
  ) 


RIPI_KM %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_under_75.pdf",
           width =8*2.5, height =7*2)


km_lyfo = read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all_under_75.csv")

km_lyfo = km_lyfo %>%
  mutate(
    event = ifelse(event == 0, 0, 1),
    Predicted = factor(risk_prediction, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    NCCN_IPI = factor(NCCN_categorical, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    years_to_event = days_to_event / 365.25)



km_reverse_lyfo = km_lyfo %>%
  mutate(event = ifelse(event == 0, 1, 0))

RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, km_reverse_lyfo)

ML = survfit(Surv(years_to_event, event) ~ Predicted, km_reverse_lyfo)

RIPI

ML


km_lyfo = read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all_under_75.csv")

km_lyfo = km_lyfo %>%
  mutate(
    event = ifelse(event == 0, 0, 1),
    Predicted = factor(risk_prediction, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    NCCN_IPI = factor(NCCN_categorical, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    years_to_event = days_to_event / 365.25)

RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, km_lyfo)

ML = survfit(Surv(years_to_event, event) ~ Predicted, km_lyfo)

for (time in c(2, 5, 10)){
  
  ripi_summary <- summary(RIPI, time = time)
  
  ml_summary <- summary(ML, time = time)
  
  plotting_data = data.frame(n = ripi_summary$n, 
                             level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                             level_no = c(1,2,3,4),
                             class = c("factor", "factor", "factor", "factor"),
                             estimate = ripi_summary$surv,
                             conf.low = ripi_summary$lower,
                             conf.high = ripi_summary$upper,
                             std.error = ripi_summary$std.err,
                             p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                             variable = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             label = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             reference = c(F, F, F, F)) %>%
    rbind(data.frame(n = ml_summary$n, 
                     level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                     level_no = c(1,2,3,4),
                     class = c("factor", "factor", "factor", "factor"),
                     estimate = ml_summary$surv,
                     conf.low = ml_summary$lower,
                     conf.high = ml_summary$upper,
                     std.error = ml_summary$std.err,
                     p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                     variable = c("ML", NA_character_, NA_character_, NA_character_),
                     label = c("ML", NA_character_, NA_character_, NA_character_),
                     reference = c(F, F, F,F)))
  
  label_list <- ifelse(plotting_data$variable == "NCCN IPI",
                       "bold('NCCN IPI')", "bold(ML[All])")
  ripi_summary <- summary(RIPI, time = time)
  
  ml_summary <- summary(ML, time = time)
  
  plotting_data = data.frame(n = ripi_summary$n, 
                             level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                             level_no = c(1,2,3,4),
                             class = c("factor", "factor", "factor", "factor"),
                             estimate = ripi_summary$surv,
                             conf.low = ripi_summary$lower,
                             conf.high = ripi_summary$upper,
                             std.error = ripi_summary$std.err,
                             p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                             variable = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             label = c('NCCN IPI', NA_character_, NA_character_, NA_character_),
                             reference = c(F, F, F, F)) %>%
    rbind(data.frame(n = ml_summary$n, 
                     level = c("Low", "Low-Intermediate", "Intermediate-High", "High"), 
                     level_no = c(1,2,3,4),
                     class = c("factor", "factor", "factor", "factor"),
                     estimate = ml_summary$surv,
                     conf.low = ml_summary$lower,
                     conf.high = ml_summary$upper,
                     std.error = ml_summary$std.err,
                     p.value = c(NA_real_,NA_real_,NA_real_,NA_real_),
                     variable = c("ML", NA_character_, NA_character_, NA_character_),
                     label = c("ML", NA_character_, NA_character_, NA_character_),
                     reference = c(F, F, F,F)))
  
  label_list <- ifelse(plotting_data$variable == "NCCN IPI",
                       "bold('NCCN IPI')", "bold(ML[All])")
  
  
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = label_list, parse = TRUE, heading = "Model"), 
                 forest_panel(width = 0.1, display = level), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, heading = paste0(time, "-year Treatment-free Survival")), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.07, display = if_else(reference, "Reference", sprintf("%d%% (%d%%-%d%%)", round(trans(estimate)), round(trans(conf.low)), round(trans(conf.high)))), display_na = NA, heading = "Percent (95% CI)")) 
  #### SEE BELOW FOR HEADER CHANGE TO P-VALUE
  #forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
  ####
  #forest_panel(width = 0.00))
  
  forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels, 
                                   trans = function(x) 100 * x, 
                                   format_options = 
                                     list(
                                       colour = "black",
                                       color = NULL,
                                       shape = 15,
                                       text_size = 4,
                                       point_size = 5,
                                       banded = TRUE,
                                       suffix = "%",
                                       accuracy = 1
                                     ))
  
  forest_plot %>%
    ggexport(filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/forest_plot_all_under_75_",time,"_years.pdf"),
             width = 15, height = 6)
  
}

fit = list(RIPI = RIPI, ML = ML)

cols <- colorRampPalette( colors = brewer.pal(4,"RdBu") )
cols <- cols(8)

# cols <- cols
# 
# #custom_palette <- c(cols[1], cols[3], cols[7:8], cols[1], cols[3], cols[7:8])
# 
# custom_palette <- c(cols[1], cols[3], cols[7:8])
# 
# custom_palette_adjusted <- c("#CA0021", "#EE8D74", "#4195C4", "#0571B1")
# 
# custom_palette_combined <- c(custom_palette, custom_palette_adjusted)

fit = list(RIPI, ML)

cols <- c(cols[5:8], cols[4:1])

# labs_expr <- c(expression("NCCN IPI: Low Risk"), 
#                expression("NCCN IPI: Low-Intermediate Risk"),
#                expression("NCCN IPI: Intermediate-High Risk"),
#                expression("NCCN IPI: High Risk"),
#                expression(ML[All] ~ ": Low Risk"),
#                expression(ML[All] ~ ": Low-Intermediate Risk"),
#                expression(ML[All] ~ ": Intermediate-High Risk"),
#                expression(ML[All] ~ ": High Risk"))

RIPI_KM = ggsurvplot_combine(fit, km_lyfo,
                             #title = "Treatment-free survival after first-line treatment",
                             #subtitle="All lymphoma patients under 75 years old", # DLBCL patients treated with R-CHOP-like treatment under 75 years old
                             font.title = 40, 
                             font.subtitle=20,
                             legend = "bottom",
                             legend.title = "Group",
                             pval=TRUE,
                             legend.labs.size = 6,
                             legend.labs = c("NCCN IPI: Low Risk","NCCN IPI: Low-Intermediate Risk","NCCN IPI: Intermediate-High Risk","NCCN IPI: High Risk", "ML: Low Risk","ML: Low-Intermediate Risk","ML: Intermediate-High Risk","ML: High Risk"),
                             #legend.labs = rep("", 8),
                             xlim = c(0, 5),
                             linetype=c(1,1,1,1,5,5,5,5),
                             conf.int = TRUE,                    # Add confidence interval
                             conf.int.style = "ribbon",            # CI style, use "step" or "ribbon"
                             palette = cols,
                             risk.table = "percentage",
                             risk.table.col="strata",
                             risk.table.y.text = FALSE,
                             tables.height = 0.3,
                             tables.theme = theme_cleantable(),
                             ylab = ("Probability of treatment-free survival"),
                             xlab = "Time (years)",
                             break.time.by = 1,
                             fontsize = 7, # 9
                             risk.table.fontsize = 10,
                             size = 1.2,
                             censor.size=7
) %++%
  theme(
    plot.title = element_text(hjust = 0, vjust = 0.6) ) 
# 
# labs_expr <- c(expression(NCCN~IPI ~": Low Risk"), 
# expression("NCCN IPI: Low-Intermediate Risk"),
# expression("NCCN IPI: Intermediate-High Risk"),
# expression("NCCN IPI: High Risk"),
# expression(ML[All] ~ ": Low Risk"),
# expression(ML[All] ~ ": Low-Intermediate Risk"),
# expression(ML[All] ~ ": Intermediate-High Risk"),
# expression(ML[All] ~ ": High Risk"))
# 
# RIPI_KM$plot <- RIPI_KM$plot + 
#   scale_color_manual(values = cols, labels = labs_expr, breaks=levels(RIPI_KM$plot$data$strata)) + 
#   scale_fill_manual(values = scales::alpha(cols, 0.15), labels = labs_expr, breaks=levels(RIPI_KM$plot$data$strata))

# KM_OS$plot + geom_vline(xintercept = 2, linetype = "dashed")

# KM_OS$plot + geom_vline(xintercept = 2, linetype = "dashed")
RIPI_KM$plot <- RIPI_KM$plot + 
  geom_vline(xintercept = 2, linetype = "dashed", size = 0.8) + 
  #geom_segment(x = -1, xend = 2, y =ripi_summary$surv[1], yend = ripi_summary$surv[1],linetype="dotdash", color = custom_pallete[1], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ripi_summary$surv[2], yend = ripi_summary$surv[2],linetype="dotdash", color = custom_pallete[2], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ml_summary$surv[1], yend = ml_summary$surv[1],linetype="dotdash", color = custom_pallete[3], size = 0.8) +
  #geom_segment(x = -1, xend = 2, y =ml_summary$surv[2], yend = ml_summary$surv[2],linetype="dotdash", color = custom_pallete[4], size = 0.8) +
  #geom_hline(yintercept=ripi_summary$surv[1],linetype="dotted", color = custom_pallete[1])+
  #geom_hline(yintercept=ripi_summary$surv[2],linetype="dotted", color = custom_pallete[2])+
  #geom_hline(yintercept=ml_summary$surv[1],linetype="dotted", color = custom_pallete[3])+
  #geom_hline(yintercept=ml_summary$surv[2],linetype="dotted", color = custom_pallete[4])+
  #annotate("segment", x=0, xend = 2, y=0.6331, yend=0.4)+ 
  theme(
    axis.title.x = element_text(size = 25),  
    axis.title.y = element_text(size = 25),
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 17)
  )

RIPI_KM$table <- RIPI_KM$table +
  theme(plot.title = element_text(size = 30)
  ) 


RIPI_KM %>%
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_all_under_75.pdf",
           width =8*2.5, height =7*2)

