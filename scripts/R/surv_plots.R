km_lyfo = read_csv("/ngc/people/mikwer_r/projects/lyfo_relapse_prediction/scripts/results/km_data_lyfo_FCR.csv")

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

ripi_summary = summary(RIPI, time = 2, extend = TRUE)

ml_summary = summary(ML, time = 2, extend = TRUE)

ripi_summary

master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 2, extend = TRUE)
  
  ml_summary = summary(ML, time = 2, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI")
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, model = "ML")
  
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


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(surv, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="TFS"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="N"))


heatmap_plot %>%
  ggexport(filename = "../mikkel_w/plots/survival.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "../mikkel_w/plots/survival_n.pdf",
           width = 20, height = 10)



master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 5, extend = TRUE)
  
  ml_summary = summary(ML, time = 5, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI")
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, model = "ML")
  
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


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(surv, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="TFS"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="N"))


heatmap_plot %>%
  ggexport(filename = "../mikkel_w/plots/survival_5_years.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "../mikkel_w/plots/survival_n_5_years.pdf",
           width = 20, height = 10)


master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 10, extend = TRUE)
  
  ml_summary = summary(ML, time = 10, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI")
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, model = "ML")
  
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


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(surv, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="TFS"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="N"))


heatmap_plot %>%
  ggexport(filename = "../mikkel_w/plots/survival_10_years.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "../mikkel_w/plots/survival_n_10_years.pdf",
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
                             variable = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                             label = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                             reference = c(F, F, F,F)) %>%
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
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = variable, fontface = "bold", heading = "Variable"), 
                 forest_panel(width = 0.1, display = level), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, heading = paste("Treatment Free Survival \n Time =", time, "years")), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.07, display = if_else(reference, "Reference", sprintf("%0.2f (%0.2f, %0.2f)", trans(estimate), trans(conf.low), trans(conf.high))), display_na = NA)) 
  #### SEE BELOW FOR HEADER CHANGE TO P-VALUE
  #forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
  ####
  #forest_panel(width = 0.00))
  
  forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels)
  
  forest_plot %>%
    ggexport(filename = paste0("../mikkel_w/plots/forest_plot_",time,"_years.pdf"),
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
  ggexport(filename = "../mikkel_w/plots/KM_TFS.pdf",
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

master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 2, extend = TRUE)
  
  ml_summary = summary(ML, time = 2, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI")
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, model = "ML")
  
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


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(surv, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="TFS"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="N"))

heatmap_plot %>%
  ggexport(filename = "../mikkel_w/plots/survival_all.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "../mikkel_w/plots/survival_all_n.pdf",
           width = 20, height = 10)


master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 5, extend = TRUE)
  
  ml_summary = summary(ML, time = 5, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI")
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, model = "ML")
  
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


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(surv, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="TFS"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="N"))

heatmap_plot %>%
  ggexport(filename = "../mikkel_w/plots/survival_all_5_years.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "../mikkel_w/plots/survival_all_n_5_years.pdf",
           width = 20, height = 10)


master_df <- data.frame()

age_categories = c("<45", "45-60","60-75", "75<")

for (age_category in c("<45", "45-60","60-75", "75<")){
  
  subset <- km_lyfo %>% filter(age_category == !!age_category)
  
  RIPI = survfit(Surv(years_to_event, event) ~ NCCN_IPI, subset)
  
  ML = survfit(Surv(years_to_event, event) ~ Predicted, subset)
  
  ripi_summary = summary(RIPI, time = 10, extend = TRUE)
  
  ml_summary = summary(ML, time = 10, extend = TRUE)
  
  ripi_dataframe = data.frame(surv = ripi_summary$surv, strata = ripi_summary$strata, age_category = age_category, n = ripi_summary$n, model = "NCCN IPI")
  
  ml_dataframe = data.frame(surv = ml_summary$surv, strata = ml_summary$strata, age_category = age_category, n = ml_summary$n, model = "ML")
  
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


heatmap_plot <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = surv)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(surv, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="TFS"))


heatmap_plot_n <- ggplot(master_df, aes(x = `Risk Group`, y = `Age Group`, fill = n)) + 
  geom_tile(color = "black") + 
  geom_text(aes(label = round(n, 2)), color = "white", size = 9)+
  facet_wrap(~model) + 
  #scale_fill_gradient2(limits = c(0, 1)) + 
  #scale_fill_gradient(limits = c(0, 1), high = muted("blue"), low=muted("red", c=300)) + 
  theme_minimal(base_size = 20) + 
  guides(fill=guide_legend(title="N"))

heatmap_plot %>%
  ggexport(filename = "../mikkel_w/plots/survival_all_10_years.pdf",
           width = 20, height = 10)

heatmap_plot_n %>%
  ggexport(filename = "../mikkel_w/plots/survival_all_n_10_years.pdf",
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
                           variable = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                           label = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                           reference = c(F, F, F,F)) %>%
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
panels <- list(forest_panel(width = 0.03), 
               forest_panel(width = 0.1, display = variable, fontface = "bold", heading = "Variable"), 
               forest_panel(width = 0.1, display = level), 
               forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
               forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
               forest_panel(width = 0.55, item = "forest", hjust = 0.5, heading = paste("Treatment Free Survival \n Time =", time, "years")), 
               forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
               forest_panel(width = 0.07, display = if_else(reference, "Reference", sprintf("%0.2f (%0.2f, %0.2f)", trans(estimate), trans(conf.low), trans(conf.high))), display_na = NA)) 
#### SEE BELOW FOR HEADER CHANGE TO P-VALUE
#forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
####
#forest_panel(width = 0.00))

forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels)

forest_plot %>%
  ggexport(filename = paste0("../mikkel_w/plots/forest_plot_all_",time,"_years.pdf"),
           width = 15, height = 6)

}


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
                           variable = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                           label = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                           reference = c(F, F, F,F)) %>%
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
panels <- list(forest_panel(width = 0.03), 
               forest_panel(width = 0.1, display = variable, fontface = "bold", heading = "Variable"), 
               forest_panel(width = 0.1, display = level), 
               forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
               forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
               forest_panel(width = 0.55, item = "forest", hjust = 0.5, heading = paste("Treatment Free Survival \n Time =", time, "years")), 
               forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
               forest_panel(width = 0.07, display = if_else(reference, "Reference", sprintf("%0.2f (%0.2f, %0.2f)", trans(estimate), trans(conf.low), trans(conf.high))), display_na = NA)) 
#### SEE BELOW FOR HEADER CHANGE TO P-VALUE
#forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
####
#forest_panel(width = 0.00))

forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels)

forest_plot %>%
  ggexport(filename = paste0("../mikkel_w/plots/forest_plot_all_",time,"_years.pdf"),
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
  ggexport(filename = "../mikkel_w/plots/KM_TFS_all.pdf",
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
                             variable = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                             label = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                             reference = c(F, F, F,F)) %>%
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
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = variable, fontface = "bold", heading = "Variable"), 
                 forest_panel(width = 0.1, display = level), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, heading = paste("Treatment Free Survival \n Time =", time, "years")), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.07, display = if_else(reference, "Reference", sprintf("%0.2f (%0.2f, %0.2f)", trans(estimate), trans(conf.low), trans(conf.high))), display_na = NA)) 
  #### SEE BELOW FOR HEADER CHANGE TO P-VALUE
  #forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
  ####
  #forest_panel(width = 0.00))
  
  forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels)
  
  forest_plot %>%
    ggexport(filename = paste0("../mikkel_w/plots/forest_plot_under_75_",time,"_years.pdf"),
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
  ggexport(filename = "../mikkel_w/plots/KM_TFS_under_75.pdf",
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
                             variable = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                             label = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                             reference = c(F, F, F,F)) %>%
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
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = variable, fontface = "bold", heading = "Variable"), 
                 forest_panel(width = 0.1, display = level), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, heading = paste("Treatment Free Survival \n Time =", time, "years")), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.07, display = if_else(reference, "Reference", sprintf("%0.2f (%0.2f, %0.2f)", trans(estimate), trans(conf.low), trans(conf.high))), display_na = NA)) 
  #### SEE BELOW FOR HEADER CHANGE TO P-VALUE
  #forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
  ####
  #forest_panel(width = 0.00))
  
  forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels)
  
  forest_plot %>%
    ggexport(filename = paste0("../mikkel_w/plots/forest_plot_under_75_",time,"_years.pdf"),
             width = 15, height = 6)
  
}


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
                             variable = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                             label = c("NCCN IPI", NA_character_, NA_character_, NA_character_),
                             reference = c(F, F, F,F)) %>%
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
  panels <- list(forest_panel(width = 0.03), 
                 forest_panel(width = 0.1, display = variable, fontface = "bold", heading = "Variable"), 
                 forest_panel(width = 0.1, display = level), 
                 forest_panel(width = 0.05, display = n, hjust = 1, heading = "N"), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.55, item = "forest", hjust = 0.5, heading = paste("Treatment Free Survival \n Time =", time, "years")), 
                 forest_panel(width = 0.03, item = "vline", hjust = 0.5), 
                 forest_panel(width = 0.07, display = if_else(reference, "Reference", sprintf("%0.2f (%0.2f, %0.2f)", trans(estimate), trans(conf.low), trans(conf.high))), display_na = NA)) 
  #### SEE BELOW FOR HEADER CHANGE TO P-VALUE
  #forest_panel(width = 0.0, display = if_else(reference, "", format.pval(p.value, digits = 1, eps = 0.001)), display_na = NA, hjust = 1, heading = "p-value"), 
  ####
  #forest_panel(width = 0.00))
  
  forest_plot <- panel_forest_plot(plotting_data, limits = c(0,1), panels = panels)
  
  forest_plot %>%
    ggexport(filename = paste0("../mikkel_w/plots/forest_plot_all_under_75_",time,"_years.pdf"),
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
  ggexport(filename = "../mikkel_w/plots/KM_TFS_all_under_75.pdf",
           width =8*2.5, height =7*2)