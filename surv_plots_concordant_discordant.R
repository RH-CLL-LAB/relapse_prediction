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

km_lyfo_concordance = km_lyfo %>%
  filter(Predicted == NCCN_IPI)

km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = paste0("NCCN IPI: ", NCCN_IPI, " | ML: ", Predicted))

km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = case_when(
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
      Predicted %in% c("Low", "Low-Intermediate"))~"Lower",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Intermediate-High", "High"))~"Higher",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"ML: Lower, NCCN IPI: Higher",
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Intermediate-High", "High"))~"ML: Higher, NCCN IPI: Lower",
    
  ))

combined = survfit(Surv(years_to_event, event) ~risk_category, km_lyfo)

for (time in c(2, 5, 10)){
  
  combined_summary = summary(combined, time = time, extend = TRUE)
  
  plotting_data = data.frame(n = combined_summary$n, 
                             level = str_remove(names(combined$strata), "risk_category="), 
                             level_no = seq(length(combined$strata)),
                             class = rep("factor", length(combined$strata)),
                             estimate = combined_summary$surv,
                             conf.low = combined_summary$lower,
                             conf.high = combined_summary$upper,
                             std.error = combined_summary$std.err,
                             p.value = rep(NA_real_, length(combined$strata)),
                             #variable = c('Risk Category', rep(NA_character_, length(combined$strata)-1)),
                             #label = c('Risk Category', rep(NA_character_, length(combined$strata)-1)),
                             variable = c("Concordant", NA_character_, "Discordant", NA_character_),
                             label = c("Concordant", NA_character_, "Discordant", NA_character_),
                             reference = rep(F, length(combined$strata)))
  
  
  label_list <- ifelse(plotting_data$variable == "Concordant",
                      "bold('Concordant')", "bold('Discordant')")

  label_list <- case_when(
    plotting_data$level == "Lower" ~ "Lower",
    plotting_data$level == "Higher" ~ "Higher",
    plotting_data$level == "ML: Higher, NCCN IPI: Lower" ~ "ML[All]:' Higher, NCCN IPI: Lower'",
    plotting_data$level == "ML: Lower, NCCN IPI: Higher" ~ "ML[All]:' Lower, NCCN IPI: Higher'"
  )
  
  category_label_list <- ifelse(plotting_data$variable == "Concordant",
                                "bold('Concordant')", "bold('Discordant')")
  
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = category_label_list, heading = "Concordance Category", parse = TRUE), 
                 forest_panel(width = 0.1, display = label_list, parse = TRUE), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, 
                              heading = paste0(time, "-year Treatment-free Survival")), 
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
                                       text_size = 3,
                                       point_size = 5,
                                       banded = TRUE,
                                       suffix = "%",
                                       accuracy = 1
                                     ))
  
  forest_plot %>%
    ggexport(filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/concordance_forest_plot_",time,"_years.pdf"),
             width = 15, height = 6)
  
}

forest_plot

fit = list(combined=combined)



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

cols = c("#E41A1C", "#009E73", "#CC79A7", "#E69F00")

RIPI_KM = ggsurvplot_combine(fit, km_lyfo,
                             #title = "Treatment-free survival after first-line treatment",
                             #subtitle="All lymphoma patients under 75 years old", # DLBCL patients treated with R-CHOP-like treatment under 75 years old
                             font.title = 40, 
                             font.subtitle=20,
                             legend = "bottom",
                             legend.title = "Group",
                             pval=TRUE,
                             legend.labs.size = 6,
                             legend.labs = c("Concordant: Higher", "Concordant: Lower", "Discordant: (ML: Higher, NCCN IPI: Lower)", "Discordant: (ML: Lower, NCCN IPI: Higher)"),
                             xlim = c(0, 5),
                             linetype=c(1,1,5,5),
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
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_discordance.pdf",
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


km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = paste0("NCCN IPI: ", NCCN_IPI, " | ML: ", Predicted))


km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = case_when(
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"Lower",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Intermediate-High", "High"))~"Higher",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"ML: Lower, NCCN IPI: Higher",
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Intermediate-High", "High"))~"ML: Higher, NCCN IPI: Lower",
    
  ))

combined = survfit(Surv(years_to_event, event) ~risk_category, km_lyfo)

for (time in c(2, 5, 10)){
  
  combined_summary = summary(combined, time = time, extend = TRUE)
  
  plotting_data = data.frame(n = combined_summary$n, 
                             level = str_remove(names(combined$strata), "risk_category="), 
                             level_no = seq(length(combined$strata)),
                             class = rep("factor", length(combined$strata)),
                             estimate = combined_summary$surv,
                             conf.low = combined_summary$lower,
                             conf.high = combined_summary$upper,
                             std.error = combined_summary$std.err,
                             p.value = rep(NA_real_, length(combined$strata)),
                             #variable = c('Risk Category', rep(NA_character_, length(combined$strata)-1)),
                             #label = c('Risk Category', rep(NA_character_, length(combined$strata)-1)),
                             variable = c("Concordant", NA_character_, "Discordant", NA_character_),
                             label = c("Concordant", NA_character_, "Discordant", NA_character_),
                             reference = rep(F, length(combined$strata)))
  
  
  label_list <- case_when(
    plotting_data$level == "Lower" ~ "Lower",
    plotting_data$level == "Higher" ~ "Higher",
    plotting_data$level == "ML: Higher, NCCN IPI: Lower" ~ "ML[All]:' Higher, NCCN IPI: Lower'",
    plotting_data$level == "ML: Lower, NCCN IPI: Higher" ~ "ML[All]:' Lower, NCCN IPI: Higher'"
  )
  
  category_label_list <- ifelse(plotting_data$variable == "Concordant",
                                "bold('Concordant')", "bold('Discordant')")
  
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = category_label_list, heading = "Concordance Category", parse = TRUE), 
                 forest_panel(width = 0.1, display = label_list, parse = TRUE), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, 
                              heading = paste0(time, "-year Treatment-free Survival")), 
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
                                       text_size = 3,
                                       point_size = 5,
                                       banded = TRUE,
                                       suffix = "%",
                                       accuracy = 1
                                     ))
  
  forest_plot %>%
    ggexport(filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/concordance_forest_plot_all_",time,"_years.pdf"),
             width = 15, height = 6)
  
}

km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = case_when(
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"Concordant Low",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Intermediate-High", "High"))~"Concordant High",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"Discordant (NCCN IPI Higher)",
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Intermediate-High", "High"))~"Discordant (ML Higher)",
    
  ))

combined = survfit(Surv(years_to_event, event) ~risk_category, km_lyfo)

fit = list(combined=combined)



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

cols = c("#E41A1C", "#009E73", "#CC79A7", "#E69F00")

RIPI_KM = ggsurvplot_combine(fit, km_lyfo,
                             #title = "Treatment-free survival after first-line treatment",
                             #subtitle="All lymphoma patients under 75 years old", # DLBCL patients treated with R-CHOP-like treatment under 75 years old
                             font.title = 40, 
                             font.subtitle=20,
                             legend = "bottom",
                             legend.title = "Group",
                             pval=TRUE,
                             legend.labs.size = 6,
                             legend.labs = c("Concordant: Higher", "Concordant: Lower", "Discordant: (ML: Higher, NCCN IPI: Lower)", "Discordant: (ML: Lower, NCCN IPI: Higher)"),
                             xlim = c(0, 5),
                             linetype=c(1,1,5,5),
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
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_discordance_all.pdf",
           width =8*2.5, height =7*2)

km_reverse_lyfo = km_lyfo %>%
  mutate(event = ifelse(event == 0, 1, 0))

RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, km_reverse_lyfo)

ML = survfit(Surv(years_to_event, event) ~ Predicted, km_reverse_lyfo)

RIPI

ML


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


km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = case_when(
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"Lower",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Intermediate-High", "High"))~"Higher",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"ML: Lower, NCCN IPI: Higher",
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Intermediate-High", "High"))~"ML: Higher, NCCN IPI: Lower",
    
  ))

combined = survfit(Surv(years_to_event, event) ~risk_category, km_lyfo)

for (time in c(2, 5, 10)){
  
  combined_summary = summary(combined, time = time, extend = TRUE)
  
  plotting_data = data.frame(n = combined_summary$n, 
                             level = str_remove(names(combined$strata), "risk_category="), 
                             level_no = seq(length(combined$strata)),
                             class = rep("factor", length(combined$strata)),
                             estimate = combined_summary$surv,
                             conf.low = combined_summary$lower,
                             conf.high = combined_summary$upper,
                             std.error = combined_summary$std.err,
                             p.value = rep(NA_real_, length(combined$strata)),
                             #variable = c('Risk Category', rep(NA_character_, length(combined$strata)-1)),
                             #label = c('Risk Category', rep(NA_character_, length(combined$strata)-1)),
                             variable = c("Concordant", NA_character_, "Discordant", NA_character_),
                             label = c("Concordant", NA_character_, "Discordant", NA_character_),
                             reference = rep(F, length(combined$strata)))
  
  
  label_list <- case_when(
    plotting_data$level == "Lower" ~ "Lower",
    plotting_data$level == "Higher" ~ "Higher",
    plotting_data$level == "ML: Higher, NCCN IPI: Lower" ~ "ML[All]:' Higher, NCCN IPI: Lower'",
    plotting_data$level == "ML: Lower, NCCN IPI: Higher" ~ "ML[All]:' Lower, NCCN IPI: Higher'"
    )
  
  category_label_list <- ifelse(plotting_data$variable == "Concordant",
                       "bold('Concordant')", "bold('Discordant')")
  
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = category_label_list, heading = "Concordance Category", parse = TRUE), 
                 forest_panel(width = 0.1, display = label_list, parse = TRUE), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, 
                              heading = paste0(time, "-year Treatment-free Survival")), 
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
                                       text_size = 3,
                                       point_size = 5,
                                       banded = TRUE,
                                       suffix = "%",
                                       accuracy = 1
                                     ))
  
  forest_plot %>%
    ggexport(filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/concordance_forest_plot_under_75_",time,"_years.pdf"),
             width = 15, height = 6)
  
}

km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = paste0("NCCN IPI: ", NCCN_IPI, " | ML: ", Predicted))

km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = case_when(
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"Concordant Low",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Intermediate-High", "High"))~"Concordant High",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"Discordant (NCCN IPI Higher)",
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Intermediate-High", "High"))~"Discordant (ML Higher)",
    
  ))

combined = survfit(Surv(years_to_event, event) ~risk_category, km_lyfo)

fit = list(combined=combined)

cols = c("#E41A1C", "#009E73", "#CC79A7", "#E69F00")


RIPI_KM = ggsurvplot_combine(fit, km_lyfo,
                             #title = "Treatment-free survival after first-line treatment",
                             #subtitle="All lymphoma patients under 75 years old", # DLBCL patients treated with R-CHOP-like treatment under 75 years old
                             font.title = 40, 
                             font.subtitle=20,
                             legend = "bottom",
                             legend.title = "Group",
                             pval=TRUE,
                             legend.labs.size = 6,
                             legend.labs = c("Concordant: Higher", "Concordant: Lower", "Discordant: (ML: Higher, NCCN IPI: Lower)", "Discordant: (ML: Lower, NCCN IPI: Higher)"),
                             xlim = c(0, 5),
                             linetype=c(1,1,5,5),
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
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_under_75_discordant.pdf",
           width =8*2.5, height =7*2)


km_lyfo = read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR_all_under_75.csv")

km_lyfo = km_lyfo %>%
  mutate(
    event = ifelse(event == 0, 0, 1),
    Predicted = factor(risk_prediction, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    NCCN_IPI = factor(NCCN_categorical, levels = c("Low", "Low-Intermediate", "Intermediate-High", "High")),
    years_to_event = days_to_event / 365.25)

km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = case_when(
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"Lower",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Intermediate-High", "High"))~"Higher",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"ML: Lower, NCCN IPI: Higher",
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Intermediate-High", "High"))~"ML: Higher, NCCN IPI: Lower",
    
  ))

combined = survfit(Surv(years_to_event, event) ~risk_category, km_lyfo)

for (time in c(2, 5, 10)){
  
  combined_summary = summary(combined, time = time, extend = TRUE)
  
  plotting_data = data.frame(n = combined_summary$n, 
                             level = str_remove(names(combined$strata), "risk_category="), 
                             level_no = seq(length(combined$strata)),
                             class = rep("factor", length(combined$strata)),
                             estimate = combined_summary$surv,
                             conf.low = combined_summary$lower,
                             conf.high = combined_summary$upper,
                             std.error = combined_summary$std.err,
                             p.value = rep(NA_real_, length(combined$strata)),
                             #variable = c('Risk Category', rep(NA_character_, length(combined$strata)-1)),
                             #label = c('Risk Category', rep(NA_character_, length(combined$strata)-1)),
                             variable = c("Concordant", NA_character_, "Discordant", NA_character_),
                             label = c("Concordant", NA_character_, "Discordant", NA_character_),
                             reference = rep(F, length(combined$strata)))
  
  
  label_list <- case_when(
    plotting_data$level == "Lower" ~ "Lower",
    plotting_data$level == "Higher" ~ "Higher",
    plotting_data$level == "ML: Higher, NCCN IPI: Lower" ~ "ML[All]:' Higher, NCCN IPI: Lower'",
    plotting_data$level == "ML: Lower, NCCN IPI: Higher" ~ "ML[All]:' Lower, NCCN IPI: Higher'"
  )
  
  category_label_list <- ifelse(plotting_data$variable == "Concordant",
                                "bold('Concordant')", "bold('Discordant')")
  
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = category_label_list, heading = "Concordance Category", parse = TRUE), 
                 forest_panel(width = 0.1, display = label_list, parse = TRUE), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, 
                              heading = paste0(time, "-year Treatment-free Survival")), 
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
                                       text_size = 3,
                                       point_size = 5,
                                       banded = TRUE,
                                       suffix = "%",
                                       accuracy = 1
                                     ))
  
  forest_plot %>%
    ggexport(filename = paste0("projects/lyfo_relapse_prediction/scripts/plots/concordance_forest_plot_all_under_75_",time,"_years.pdf"),
             width = 15, height = 6)
  
}

km_lyfo = km_lyfo %>%
  filter(!is.na(NCCN_IPI)) %>%
  mutate(risk_category = case_when(
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"Concordant Low",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Intermediate-High", "High"))~"Concordant High",
    (NCCN_IPI %in% c("Intermediate-High", "High") & 
       Predicted %in% c("Low", "Low-Intermediate"))~"Discordant (NCCN IPI Higher)",
    (NCCN_IPI %in% c("Low", "Low-Intermediate") & 
       Predicted %in% c("Intermediate-High", "High"))~"Discordant (ML Higher)",
    
  ))

combined = survfit(Surv(years_to_event, event) ~risk_category, km_lyfo)

fit = list(combined=combined)

cols = c("#E41A1C", "#009E73", "#CC79A7", "#E69F00")


RIPI_KM = ggsurvplot_combine(fit, km_lyfo,
                             #title = "Treatment-free survival after first-line treatment",
                             #subtitle="All lymphoma patients under 75 years old", # DLBCL patients treated with R-CHOP-like treatment under 75 years old
                             font.title = 40, 
                             font.subtitle=20,
                             legend = "bottom",
                             legend.title = "Group",
                             pval=TRUE,
                             legend.labs.size = 6,
                             legend.labs = c("Concordant: Higher", "Concordant: Lower", "Discordant (ML: Higher, NCCN IPI: Lower)", "Discordant (ML: Lower, NCCN IPI: Higher)"),
                             xlim = c(0, 5),
                             linetype=c(1,1,5,5),
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
  ggexport(filename = "projects/lyfo_relapse_prediction/scripts/plots/KM_TFS_all_under_75_discordant.pdf",
           width =8*2.5, height =7*2)
