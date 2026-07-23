library(dplyr)
library(tidyr)
library(tidyverse)
library(stringr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(splines)

### Tuning Step 1

## Functions
{
  # Results reading
  read_results <- function(file_path) {
    results_df <- read.csv(file_path) %>%
      separate(experiment, into = c(NA, "Dataset", "CL method", "Settings"), 
               sep = "_",  extra = "merge") %>%
      mutate(`CL method` = 
               case_when(`CL method` == "der" ~ "DER",
                         `CL method` == "sd" ~ "SD",
                         `CL method` == "sds2" ~ "SDS2"))
    return(results_df)
  }
  
  # Analysis of resulting parameters
  analyze_parameters <- function(results_df) {
    results_df_der <- results_df %>% 
      filter(`CL method` == "DER") %>%
      select(-Dataset, -`CL method`) %>%
      mutate(run = row_number()) %>%
      mutate(Settings = str_remove_all(Settings, "[a-zA-Z]")) %>%
      separate(Settings, into = c("alpha", "beta", "replay_size"), sep = "_") %>% 
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "iteration",
        values_to = "f1"
      ) %>%
      mutate(iteration = str_remove_all(iteration, "f1_")) %>% 
      mutate_all(list(function(var) as.numeric(var))) %>%
      mutate_at(c("alpha", "beta", "replay_size"), function(var) scale(var))
    results_df_sd <- results_df %>% 
      filter(`CL method` == "SD") %>%
      select(-Dataset, -`CL method`) %>%
      mutate(run = row_number()) %>%
      mutate(Settings = str_remove_all(Settings, "[a-zA-Z]")) %>%
      separate(Settings, into = c("alpha", "replay_size"), sep = "_") %>% 
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "iteration",
        values_to = "f1"
      ) %>%
      mutate(iteration = str_remove_all(iteration, "f1_")) %>% 
      mutate_all(list(function(var) as.numeric(var))) %>%
      mutate_at(c("alpha", "replay_size"), function(var) scale(var))
    results_df_sds2 <- results_df %>% 
      filter(`CL method` == "SDS2") %>%
      select(-Dataset, -`CL method`) %>%
      mutate(run = row_number()) %>%
      mutate(Settings = str_remove_all(Settings, "[a-zA-Z]")) %>%
      separate(Settings, into = c("alpha", "lambda", "kernel_width", "replay_size"), sep = "_") %>% 
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "iteration",
        values_to = "f1"
      ) %>%
      mutate(iteration = str_remove_all(iteration, "f1_")) %>% 
      mutate_all(list(function(var) as.numeric(var))) %>%
      mutate_at(c("alpha", "lambda", "kernel_width", "replay_size"), function(var) scale(var))
    
    model_der <- lmer(f1 ~ ns(iteration, 3) * (alpha + beta + replay_size) + (1 | run), 
                      data = results_df_der)
    model_sd <- lmer(f1 ~ ns(iteration, 3) * (alpha + replay_size) + (1 | run), 
                     data = results_df_sd)
    model_sds2 <- lmer(f1 ~ ns(iteration, 3) * (alpha + lambda + kernel_width + replay_size) + (1 | run), 
                       data = results_df_sds2)
    cat("DER:\n\n")
    print(summary(model_der))
    cat(paste("\n", paste(rep("-", 100), collapse = ""), "\n\n", sep = ""))
    cat("SD:\n\n")
    print(summary(model_sd))
    cat(paste("\n", paste(rep("-", 100), collapse = ""), "\n\n", sep = ""))
    cat("SDS2:\n\n")
    print(summary(model_sds2))
  }
  
  # Best settings selection
  select_best_settings <- function(results_df) {
    results_df <- results_df %>%
      rowwise() %>%
      mutate(max_val = max(c_across(f1_0:f1_9), na.rm = TRUE)) %>%
      ungroup() %>%
      group_by(`CL method`) %>%
      slice_max(order_by = max_val, n = 1, with_ties = FALSE) %>%
      ungroup() %>%
      select(-max_val)
  }
  
  # Plot creation and saving
  create_plot <- function(results_df, file_path) {
    plt <- results_df %>% 
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1 score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      ggplot(aes(x = Iteration, y = `F1 score`, colour = Settings)) +
      geom_line() +
      facet_grid(factor(`CL method`, levels = c("DER", "SD", "SDS2")) ~ .) +
      scale_x_continuous(breaks = 0:9) +
      theme(legend.position = "none") +
      labs(color = "")
    print(plt)
    ggsave(file_path, plt, width = 125, height = 140, units = "mm", device = cairo_pdf)
  }
  
  # Time reading
  read_time <- function(folder_path) {
    read_time_file <- function(file_path) {
      time_df <- read.csv(file_path) %>%
        separate(experiment, into = c(NA, "Dataset", "CL method", "Settings"), 
                 sep = "_",  extra = "merge") %>%
        mutate(`CL method` = 
                 case_when(`CL method` == "der" ~ "DER",
                           `CL method` == "sd" ~ "SD",
                           `CL method` == "sds2" ~ "SDS2"))
      return(time_df)
    }
    
    calculate_cumulative_time <- function(time_df) {
      time_df <- time_df %>%
        mutate(row_id = row_number()) %>% 
        pivot_longer(
          cols = starts_with("time_"), 
          names_to = "iter_named", 
          values_to = "time"
        ) %>%
        mutate(iter = as.numeric(gsub("time_", "", iter_named))) %>%
        arrange(row_id, iter) %>%
        group_by(row_id) %>%
        mutate(cumulative_time = cumsum(time)) %>%
        ungroup() %>%
        select(-time, -iter) %>%
        pivot_wider(
          names_from = iter_named, 
          values_from = cumulative_time
        ) %>%
        select(-row_id)
      return(time_df)
    }
    
    time_selection_df <- read_time_file(paste0(folder_path, "time_selection.csv")) %>% calculate_cumulative_time()
    time_training_df <- read_time_file(paste0(folder_path, "time_training.csv")) %>% calculate_cumulative_time()
    time_test_df <- read_time_file(paste0(folder_path, "time_test.csv")) %>% calculate_cumulative_time()
    time_df <- list(time_selection_df, time_training_df, time_test_df) %>%
      bind_rows() %>%
      group_by(across(-starts_with("time_"))) %>%
      summarise(across(starts_with("time_"), sum), .groups = "drop")
    return(time_df)
  }
  
  # Time analysis (in minutes)
  analyze_time <- function(time_df) {
    time_df <- time_df %>%
      group_by(`CL method`) %>%
      summarise(across(starts_with("time_"), function(x) mean(x) / 1000 / 60), .groups = "drop")
    return(time_df)
  }
}

results_df_agnews <- read_results("results_tuning/tuning_1/agnews/results_f1.csv")
results_df_sensation <- read_results("results_tuning/tuning_1/sensation/results_f1.csv")
results_df_trec <- read_results("results_tuning/tuning_1/trec/results_f1.csv")

analyze_parameters(results_df_agnews)
analyze_parameters(results_df_sensation)
analyze_parameters(results_df_trec)

results_df_agnews_best <- select_best_settings(results_df_agnews)
results_df_sensation_best <- select_best_settings(results_df_sensation)
results_df_trec_best <- select_best_settings(results_df_trec)

create_plot(results_df_agnews, "plots_tuning/tuning_1/agnews_all.pdf")
create_plot(results_df_agnews_best, "plots_tuning/tuning_1/agnews_best.pdf")
create_plot(results_df_sensation, "plots_tuning/tuning_1/sensation_all.pdf")
create_plot(results_df_sensation_best, "plots_tuning/tuning_1/sensation_best.pdf")
create_plot(results_df_trec, "plots_tuning/tuning_1/trec_all.pdf")
create_plot(results_df_trec_best, "plots_tuning/tuning_1/trec_best.pdf")

time_df_agnews <- read_time("time_tuning/tuning_1/agnews/")
time_df_sensation <- read_time("time_tuning/tuning_1/sensation/")
time_df_trec <- read_time("time_tuning/tuning_1/trec/")

analyze_time(time_df_agnews)
analyze_time(time_df_sensation)
analyze_time(time_df_trec)

### Tuning Step 2

## Functions
{
  # Results reading
  read_results <- function(file_path) {
    results_df <- read.csv(file_path) %>%
      separate(experiment, into = c(NA, "Dataset", "CL method", "PEFT module", "Settings"), 
               sep = "_",  extra = "merge") %>%
      mutate(`CL method` = 
               case_when(`CL method` == "der" ~ "DER",
                         `CL method` == "sd" ~ "SD",
                         `CL method` == "sds2" ~ "SDS2")) %>%
      mutate(`PEFT module` = 
               case_when(`PEFT module` == "adapter" ~ "Adapter",
                         `PEFT module` == "lora" ~ "LoRA",
                         `PEFT module` == "prefix" ~ "Prefix"))
    return(results_df)
  }
  
  # Best settings selection
  select_best_settings <- function(results_df) {
    results_df <- results_df %>%
      rowwise() %>%
      mutate(max_val = max(c_across(f1_0:f1_9), na.rm = TRUE)) %>%
      ungroup() %>%
      group_by(across(all_of(c("CL method", "PEFT module")))) %>%
      slice_max(order_by = max_val, n = 1, with_ties = FALSE) %>%
      ungroup() %>%
      select(-max_val)
  }
  
  # Plot creation and saving
  create_plot <- function(results_df, file_path) {
    plt <- results_df %>% 
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1 score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      ggplot(aes(x = Iteration, y = `F1 score`, colour = Settings)) +
      geom_line() +
      facet_grid(factor(`CL method`, levels = c("DER", "SD", "SDS2")) ~
                   factor(`PEFT module`, levels = c("Adapter", "LoRA", "Prefix"))) +
      scale_x_continuous(breaks = 0:9) +
      theme(legend.position = "none") +
      labs(color = "")
    print(plt)
    ggsave(file_path, plt, width = 250, height = 140, units = "mm", device = cairo_pdf)
  }
  
  # Time reading
  read_time <- function(folder_path) {
    read_time_file <- function(file_path) {
      time_df <- read.csv(file_path) %>%
        separate(experiment, into = c(NA, "Dataset", "CL method", "PEFT module", "Settings"), 
                 sep = "_",  extra = "merge") %>%
        mutate(`CL method` = 
                 case_when(`CL method` == "der" ~ "DER",
                           `CL method` == "sd" ~ "SD",
                           `CL method` == "sds2" ~ "SDS2")) %>%
        mutate(`PEFT module` = 
                 case_when(`PEFT module` == "adapter" ~ "Adapter",
                           `PEFT module` == "lora" ~ "LoRA",
                           `PEFT module` == "prefix" ~ "Prefix"))
      return(time_df)
    }
    
    calculate_cumulative_time <- function(time_df) {
      time_df <- time_df %>%
        mutate(row_id = row_number()) %>% 
        pivot_longer(
          cols = starts_with("time_"), 
          names_to = "iter_named", 
          values_to = "time"
        ) %>%
        mutate(iter = as.numeric(gsub("time_", "", iter_named))) %>%
        arrange(row_id, iter) %>%
        group_by(row_id) %>%
        mutate(cumulative_time = cumsum(time)) %>%
        ungroup() %>%
        select(-time, -iter) %>%
        pivot_wider(
          names_from = iter_named, 
          values_from = cumulative_time
        ) %>%
        select(-row_id)
      return(time_df)
    }
    
    time_selection_df <- read_time_file(paste0(folder_path, "time_selection.csv")) %>% calculate_cumulative_time()
    time_training_df <- read_time_file(paste0(folder_path, "time_training.csv")) %>% calculate_cumulative_time()
    time_test_df <- read_time_file(paste0(folder_path, "time_test.csv")) %>% calculate_cumulative_time()
    time_df <- list(time_selection_df, time_training_df, time_test_df) %>%
      bind_rows() %>%
      group_by(across(-starts_with("time_"))) %>%
      summarise(across(starts_with("time_"), sum), .groups = "drop")
    return(time_df)
  }
  
  # Time analysis (in minutes)
  analyze_time <- function(time_df) {
    time_df <- time_df %>%
      group_by(across(all_of(c("CL method", "PEFT module")))) %>%
      summarise(across(starts_with("time_"), function(x) mean(x) / 1000 / 60), .groups = "drop")
    return(time_df)
  }
}

# No analysis of parameters here, since there is too little data for each 
# dataset-CL-PEFT combination, and since the there are only one parameter 
# for the Adapter and Prefix-tuning modules

results_df_agnews <- read_results("results_tuning/tuning_2/agnews/results_f1.csv")
results_df_sensation <- read_results("results_tuning/tuning_2/sensation/results_f1.csv")
results_df_trec <- read_results("results_tuning/tuning_2/trec/results_f1.csv")

results_df_agnews_best <- select_best_settings(results_df_agnews)
results_df_sensation_best <- select_best_settings(results_df_sensation)
results_df_trec_best <- select_best_settings(results_df_trec)

create_plot(results_df_agnews, "plots_tuning/tuning_2/agnews_all.pdf")
create_plot(results_df_agnews_best, "plots_tuning/tuning_2/agnews_best.pdf")
create_plot(results_df_sensation, "plots_tuning/tuning_2/sensation_all.pdf")
create_plot(results_df_sensation_best, "plots_tuning/tuning_2/sensation_best.pdf")
create_plot(results_df_trec, "plots_tuning/tuning_2/trec_all.pdf")
create_plot(results_df_trec_best, "plots_tuning/tuning_2/trec_best.pdf")

time_df_agnews <- read_time("time_tuning/tuning_2/agnews/")
time_df_sensation <- read_time("time_tuning/tuning_2/sensation/")
time_df_trec <- read_time("time_tuning/tuning_2/trec/")

analyze_time(time_df_agnews)
analyze_time(time_df_sensation)
analyze_time(time_df_trec)
