library(dplyr)
library(tidyr)
library(tidyverse)
library(stringr)
library(ggplot2)

### Functions
{
  # Results reading
  read_results <- function(file_path) {
    results_df <- read.csv(file_path) %>%
      separate(experiment, into = c("Dataset", "Total size", "PEFT method", "Strategy", "Lambda"), sep = "_") %>%
      rename(Run = run) %>%
      mutate(Run = as.character(Run)) %>%
      mutate(`PEFT method` = 
               case_when(`PEFT method` == "lora" ~ "LoRA",
                         `PEFT method` == "pfeiffer" ~ "Sequential Adapter",
                         `PEFT method` == "pfeifferinv" ~ "Sequential Invertible Adapter")) %>%
      mutate(Strategy = 
               case_when(Strategy == "diversity" ~ "k-means",
                         Strategy == "random" ~ "Random",
                         Strategy == "uncertainty" ~ "Margin"))
    return(results_df)
  }
  
  # Baselines reading
  read_baselines <- function(file_path) {
    results_df <- read.csv(file_path) %>%
      separate(experiment, into = c("Dataset", "Total size", "PEFT method", "tmp"), sep = "_") %>%
      select(-tmp) %>%
      rename(Run = run) %>%
      mutate(Run = as.character(Run)) %>%
      mutate(`PEFT method` = 
               case_when(`PEFT method` == "lora" ~ "LoRA",
                         `PEFT method` == "pfeiffer" ~ "Sequential Adapter",
                         `PEFT method` == "pfeifferinv" ~ "Sequential Invertible Adapter"))
    return(results_df)
  }
  
  # Baselines preparation
  prepare_baselines <- function(baselines_df) {
    baselines_df <- baselines_df %>% 
      mutate(group_id = ceiling(row_number() / 16)) %>%
      group_by(group_id) %>%
      summarise(across(starts_with("f1"), median), 
                across(c("Dataset", "Total size", "PEFT method"), first)) %>%
      rename(`F1 score` = f1) %>%
      mutate(`PEFT method` = str_wrap(`PEFT method`, 10))
    return(baselines_df)
  }
  
  # Plot saving
  save_plot <- function(results_df, baselines_df, file_path) {
    plt <- results_df %>% 
      mutate(group_id = ceiling(row_number() / 16)) %>%
      group_by(group_id) %>%
      summarise(across(starts_with("f1_"), median), 
                across(c("Dataset", "Total size", "PEFT method", "Strategy", "Lambda"), first)) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1 score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      mutate(Lambda = paste0("λ = ", Lambda)) %>%
      mutate(Lambda = factor(Lambda, levels = c("λ = 10", "λ = 50", "λ = 100", "λ = 500"))) %>%
      mutate(`PEFT method` = str_wrap(`PEFT method`, 10)) %>%
      ggplot(aes(x = Iteration, y = `F1 score`, colour = Lambda)) +
      geom_line() +
      geom_hline(
        data = baselines_df, 
        aes(yintercept = `F1 score`), 
        color = "black"
      ) +
      facet_grid(`Total size` + factor(`PEFT method`, levels = c(str_wrap("Sequential Adapter", 10),
                                                                 str_wrap("Sequential Invertible Adapter", 10),
                                                                 str_wrap("LoRA", 10))) ~ 
                   factor(Strategy, levels = c("Random", "Margin", "k-means"))) +
      scale_x_continuous(breaks = 0:10) +
      theme(legend.position = "top") +
      labs(color = "")
    ggsave(file_path, plt, width = 125, height = 140, units = "mm", device = cairo_pdf)
  }
}

### Results reading and visualisation
{
  baselines_df_sensation <- read_baselines("results/sensation/baseline/baseline_f1.csv") %>% prepare_baselines()
  baselines_df_agnews <- read_baselines("results/agnews/baseline/baseline_f1.csv") %>% prepare_baselines()
  baselines_df_yahoo <- read_baselines("results/yahoo/baseline/baseline_f1.csv") %>% prepare_baselines()
  
  results_df_sensation <- read_results("results/sensation/continual_active_learning/results_f1.csv")
  results_df_agnews <- read_results("results/agnews/continual_active_learning/results_f1.csv")
  results_df_yahoo <- read_results("results/yahoo/continual_active_learning/results_f1.csv")
  
  save_plot(results_df_sensation, baselines_df_sensation, "plots/plot_sensation.pdf")
  save_plot(results_df_agnews, baselines_df_agnews, "plots/plot_agnews.pdf")
  save_plot(results_df_yahoo, baselines_df_yahoo, "plots/plot_yahoo.pdf")
}
