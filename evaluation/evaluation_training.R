library(dplyr)
library(tidyr)
library(tidyverse)
library(stringr)
library(ggplot2)
library(scales)
library(lemon)
library(grid)
library(knitr)
library(kableExtra)

### Functions
{
  # Results reading (training)
  read_results_training <- function(file_path) {
    results_df <- read.csv(file_path) %>%
      separate(experiment, into = c(NA, "Dataset", "Total size", "PEFT module", "CL method", "AL method"), 
               sep = "_",  extra = "merge") %>%
      mutate(`CL method` = 
               case_when(`CL method` == "der" ~ "DER",
                         `CL method` == "sd" ~ "SD",
                         `CL method` == "sds2" ~ "SDS2")) %>%
      mutate(`PEFT module` = 
               case_when(`PEFT module` == "adapter" ~ "Adapter",
                         `PEFT module` == "lora" ~ "LoRA",
                         `PEFT module` == "prefix" ~ "Prefix-tuning")) %>%
      mutate(`AL method` = 
               case_when(`AL method` == "random" ~ "RND",
                         `AL method` == "entropy" ~ "ME",
                         `AL method` == "coreset" ~ "CS"))
    return(results_df)
  }
  
  # Results reading (baselines)
  read_results_baselines <- function(file_path, type) {
    if (type == "al") {
      results_df <- read.csv(file_path) %>%
        separate(experiment, into = c(NA, "Dataset", "Total size", "PEFT module", "AL method"), 
                 sep = "_",  extra = "merge") %>%
        mutate(`PEFT module` = 
                 case_when(`PEFT module` == "adapter" ~ "Adapter",
                           `PEFT module` == "lora" ~ "LoRA",
                           `PEFT module` == "prefix" ~ "Prefix-tuning"))  %>%
        mutate(`AL method` = 
                 case_when(`AL method` == "random" ~ "RND",
                           `AL method` == "entropy" ~ "ME",
                           `AL method` == "coreset" ~ "CS"))
      return(results_df)
    } else if (type == "full") {
      results_df <- read.csv(file_path) %>%
        separate(experiment, into = c(NA, "Dataset", "Total size", "PEFT module"), 
                 sep = "_",  extra = "merge") %>%
        mutate(`PEFT module` = 
                 case_when(`PEFT module` == "adapter" ~ "Adapter",
                           `PEFT module` == "lora" ~ "LoRA",
                           `PEFT module` == "prefix" ~ "Prefix-tuning"))
      return(results_df)
    }
  }
  
  # Time reading (training)
  read_time_training <- function(folder_path) {
    read_time_file <- function(file_path) {
      time_df <- read.csv(file_path) %>%
        separate(experiment, into = c(NA, "Dataset", "Total size", "PEFT module", "CL method", "AL method"), 
                 sep = "_",  extra = "merge") %>%
        mutate(`CL method` = 
                 case_when(`CL method` == "der" ~ "DER",
                           `CL method` == "sd" ~ "SD",
                           `CL method` == "sds2" ~ "SDS2")) %>%
        mutate(`PEFT module` = 
                 case_when(`PEFT module` == "adapter" ~ "Adapter",
                           `PEFT module` == "lora" ~ "LoRA",
                           `PEFT module` == "prefix" ~ "Prefix-tuning")) %>%
        mutate(`AL method` = 
                 case_when(`AL method` == "random" ~ "RND",
                           `AL method` == "entropy" ~ "ME",
                           `AL method` == "coreset" ~ "CS"))
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
  
  # Time reading (baselines)
  read_time_baselines <- function(folder_path, type) {
    if (type == "al") {
      read_time_file <- function(file_path) {
        time_df <- read.csv(file_path) %>%
          separate(experiment, into = c(NA, "Dataset", "Total size", "PEFT module", "AL method"), 
                   sep = "_",  extra = "merge") %>%
          mutate(`PEFT module` = 
                   case_when(`PEFT module` == "adapter" ~ "Adapter",
                             `PEFT module` == "lora" ~ "LoRA",
                             `PEFT module` == "prefix" ~ "Prefix-tuning")) %>%
          mutate(`AL method` = 
                   case_when(`AL method` == "random" ~ "RND",
                             `AL method` == "entropy" ~ "ME",
                             `AL method` == "coreset" ~ "CS"))
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
    } else if (type == "full") {
      read_time_file <- function(file_path) {
        time_df <- read.csv(file_path) %>%
          separate(experiment, into = c(NA, "Dataset", "Total size", "PEFT module"), 
                   sep = "_",  extra = "merge") %>%
          mutate(`PEFT module` = 
                   case_when(`PEFT module` == "adapter" ~ "Adapter",
                             `PEFT module` == "lora" ~ "LoRA",
                             `PEFT module` == "prefix" ~ "Prefix-tuning"))
        return(time_df)
      }
      
      time_selection_df <- read_time_file(paste0(folder_path, "time_selection.csv"))
      time_training_df <- read_time_file(paste0(folder_path, "time_training.csv"))
      time_test_df <- read_time_file(paste0(folder_path, "time_test.csv"))
      time_df <- list(time_selection_df, time_training_df, time_test_df) %>% 
        bind_rows() %>%
        group_by(across(-starts_with("time_"))) %>%
        summarise(across(starts_with("time_"), sum), .groups = "drop")
      return(time_df)
    }
  }
  
  # Plot creation and saving (training results)
  create_plot_results <- function(results_df_training, results_df_baselines, total_size, file_path) {
    results_df_training <- results_df_training %>% 
      filter(`Total size` == total_size) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        across(c("Dataset", "Total size", "PEFT module", "CL method", "AL method"), first),
        .groups = "drop"
      )
    
    results_df_baselines <- results_df_baselines %>% 
      filter(`Total size` == total_size) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        across(c("Dataset", "Total size", "PEFT module", "AL method"), first),
        .groups = "drop"
      ) %>%
      mutate("CL method" = "Baseline")
    
    results_df <- bind_rows(results_df_training, results_df_baselines) %>%
      mutate(Method = case_when(`CL method` == "DER" ~ "CAL-DER",
                                `CL method` == "SD" ~ "CAL-SD",
                                `CL method` == "SDS2" ~ "CAL-SDS2",
                                `CL method` == "Baseline" ~ "AL"))
    
    plt <- results_df %>%
      ggplot(aes(x = Iteration, y = mean_f1, colour = Method, fill = Method)) +
      ylab("F1 score") +
      xlab("Training data size") +
      geom_line() +
      geom_ribbon(aes(ymin = mean_f1 - sd_f1, ymax = mean_f1 + sd_f1), 
                  alpha = 0.15, color = NA) +
      facet_grid(factor(`AL method`, levels = c("CS", "ME", "RND")) ~
                   factor(`PEFT module`, levels = c("Adapter", "LoRA", "Prefix-tuning"))) +
      scale_x_continuous(breaks = 0:9, labels = paste0(total_size / 10 * 1:10, "%")) +
      scale_fill_manual(breaks = factor(c("CAL-DER", "CAL-SD", "CAL-SDS2", "AL"), 
                                        levels = c("CAL-DER", "CAL-SD", "CAL-SDS2", "AL")), 
                        values = hue_pal()(4)) +
      scale_color_manual(breaks = factor(c("CAL-DER", "CAL-SD", "CAL-SDS2", "AL"), 
                                         levels = c("CAL-DER", "CAL-SD", "CAL-SDS2", "AL")), 
                         values = hue_pal()(4)) +
      theme_bw() +
      theme(legend.position = "top") +
      labs(color = "", fill = "")
    print(plt)
    ggsave(file_path, plt, width = 250, height = 140, units = "mm", device = cairo_pdf)
  }
  
  # Plot creation and saving (speedup)
  create_plot_speedup <- function(results_df_training, results_df_baselines, 
                                  time_df_training, time_df_baselines,
                                  total_size, peft_module, type, file_path) {
    results_df_training <- results_df_training %>%
      filter(`Total size` == total_size, 
             `PEFT module` == case_when(peft_module == "adapter" ~ "Adapter",
                                        peft_module == "lora" ~ "LoRA",
                                        peft_module == "prefix" ~ "Prefix-tuning")) %>%
      select(-Dataset, -`Total size`, -`PEFT module`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("CL method", "AL method"), first),
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        .groups = "drop"
      ) %>%
      select(-group_id, -sd_f1) %>%
      rename(mean_f1_training = "mean_f1")
    
    results_df_baselines <- results_df_baselines %>%
      filter(`Total size` == total_size, 
             `PEFT module` == case_when(peft_module == "adapter" ~ "Adapter",
                                        peft_module == "lora" ~ "LoRA",
                                        peft_module == "prefix" ~ "Prefix-tuning")) %>%
      select(-Dataset, -`Total size`, -`PEFT module`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("AL method"), first),
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        .groups = "drop"
      ) %>%
      select(-group_id, -sd_f1) %>%
      rename(mean_f1_baseline = "mean_f1")
    
    results_df <- results_df_training %>%
      left_join(results_df_baselines, by = c("Iteration", "AL method")) %>%
      mutate(mean_f1_relative = mean_f1_training / mean_f1_baseline) %>%
      select(Iteration, `CL method`, `AL method`, mean_f1_training, mean_f1_baseline, mean_f1_relative)
    
    time_df_training <- time_df_training  %>%
      filter(`Total size` == total_size, 
             `PEFT module` == case_when(peft_module == "adapter" ~ "Adapter",
                                        peft_module == "lora" ~ "LoRA",
                                        peft_module == "prefix" ~ "Prefix-tuning")) %>%
      select(-Dataset, -`Total size`, -`PEFT module`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "time_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("CL method", "AL method"), first),
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
      ) %>%
      select(-group_id, -sd_time) %>%
      rename(mean_time_training = "mean_time")
    
    time_df_baselines <- time_df_baselines %>%
      filter(`Total size` == total_size, 
             `PEFT module` == case_when(peft_module == "adapter" ~ "Adapter",
                                        peft_module == "lora" ~ "LoRA",
                                        peft_module == "prefix" ~ "Prefix-tuning")) %>%
      select(-Dataset, -`Total size`, -`PEFT module`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "time_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("AL method"), first),
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
      ) %>%
      select(-group_id, -sd_time) %>%
      rename(mean_time_baseline = "mean_time")
    
    time_df <- time_df_training %>%
      left_join(time_df_baselines, by = c("Iteration", "AL method")) %>%
      mutate(mean_speedup = mean_time_baseline / mean_time_training) %>%
      select(Iteration, `CL method`, `AL method`, mean_time_training, mean_time_baseline, mean_speedup)
    
    combined_train <- results_df %>%
      left_join(time_df, by = c("Iteration", "CL method", "AL method"))
    
    combined_baseline <- results_df_baselines %>%
      left_join(time_df_baselines, by = c("Iteration", "AL method")) %>%
      mutate(
        `CL method` = "Baseline",
        mean_f1_training = mean_f1_baseline,
        mean_f1_relative = 1,
        mean_time_training = mean_time_baseline,
        mean_speedup = 1
      ) %>%
      select(Iteration, `CL method`, `AL method`, mean_f1_training, mean_f1_baseline, mean_f1_relative, 
             mean_time_training, mean_time_baseline, mean_speedup)
    
    plt <- bind_rows(combined_train, combined_baseline) %>%
      mutate(Method = case_when(`CL method` == "DER" ~ "CAL-DER",
                                `CL method` == "SD" ~ "CAL-SD",
                                `CL method` == "SDS2" ~ "CAL-SDS2",
                                `CL method` == "Baseline" ~ "AL"),
        Method = factor(Method, levels = c("CAL-DER", "CAL-SD", "CAL-SDS2", "AL")),
        numeric_size = total_size / 10 + total_size / 10 * Iteration,
        Size = fct_reorder(paste0(numeric_size, "%"), numeric_size)
      ) %>%
      rename(`Sampling method` = "AL method") %>%
      rename(`Training method` = "Method") %>%
      ggplot(aes(x = mean_speedup, 
                 y = if(type == "relative") {
                   mean_f1_relative
                 } else if (type == "absolute") {
                   mean_f1_training
                 },
                 shape = `Training method`, color = `Sampling method`)) +
      geom_point(size = 3) + 
      scale_y_continuous(expand = expansion(mult = 0.1)) +
      scale_x_continuous(breaks = seq(1, 2.5, by = 0.5), limits = c(0.95, 2.5)) +
      facet_wrap(~ Size, nrow = 2, ncol = 5) + 
      ylab(paste("F1 score", ifelse(type == "relative", "(relative)", ifelse(type == "absolute", "", NA)))) +
      xlab("Speedup") +
      theme_bw() +
      theme(legend.position = "top", legend.box = "horizontal")
    print(plt)
    ggsave(file_path, plt, width = 250, height = 140, units = "mm", device = cairo_pdf)
  }
  
  # Plot creation and saving (speedup combined)
  create_plot_speedup_combined <- function(results_df_training_list, results_df_baselines_list,
                                           time_df_training_list, time_df_baselines_list,
                                           total_size, selected_iterations, type, file_path) {
    results_df_training <- do.call(rbind, results_df_training_list)
    results_df_baselines <- do.call(rbind, results_df_baselines_list)
    time_df_training <- do.call(rbind, time_df_training_list)
    time_df_baselines <- do.call(rbind, time_df_baselines_list)
    
    results_df_training <- results_df_training %>%
      filter(`Total size` == total_size) %>%
      select(-`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("Dataset", "PEFT module", "CL method", "AL method"), first),
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        .groups = "drop"
      ) %>%
      select(-group_id, -sd_f1) %>%
      rename(mean_f1_training = "mean_f1")
    
    results_df_baselines <- results_df_baselines %>%
      filter(`Total size` == total_size) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("Dataset", "PEFT module", "AL method"), first),
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        .groups = "drop"
      ) %>%
      select(-group_id, -sd_f1) %>%
      rename(mean_f1_baseline = "mean_f1")
    
    results_df <- results_df_training %>%
      left_join(results_df_baselines, by = c("Iteration", "Dataset", "PEFT module", "AL method")) %>%
      mutate(mean_f1_relative = mean_f1_training / mean_f1_baseline) %>%
      select(Iteration, Dataset, `PEFT module`, `CL method`, `AL method`, mean_f1_training, mean_f1_baseline, mean_f1_relative)
    
    time_df_training <- time_df_training  %>%
      filter(`Total size` == total_size) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "time_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("Dataset", "PEFT module", "CL method", "AL method"), first),
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
      ) %>%
      select(-group_id, -sd_time) %>%
      rename(mean_time_training = "mean_time")
    
    time_df_baselines <- time_df_baselines %>%
      filter(`Total size` == total_size) %>%
      select(-`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "time_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("Dataset", "PEFT module", "AL method"), first),
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
      ) %>%
      select(-group_id, -sd_time) %>%
      rename(mean_time_baseline = "mean_time")
    
    time_df <- time_df_training %>%
      left_join(time_df_baselines, by = c("Iteration", "Dataset", "PEFT module", "AL method")) %>%
      mutate(mean_speedup = mean_time_baseline / mean_time_training) %>%
      select(Iteration, Dataset, `PEFT module`, `CL method`, `AL method`, mean_time_training, mean_time_baseline, mean_speedup)
    
    combined_train <- results_df %>%
      left_join(time_df, by = c("Iteration", "Dataset", "PEFT module", "CL method", "AL method")) %>%
      filter(Iteration %in% selected_iterations)
    
    combined_baseline <- results_df_baselines %>%
      left_join(time_df_baselines, by = c("Iteration", "Dataset", "PEFT module", "AL method")) %>%
      mutate(
        `CL method` = "Baseline",
        mean_f1_training = mean_f1_baseline,
        mean_f1_relative = 1,
        mean_time_training = mean_time_baseline,
        mean_speedup = 1
      ) %>%
      select(Iteration, Dataset, `PEFT module`, `CL method`, `AL method`, mean_f1_training, mean_f1_baseline, mean_f1_relative, 
             mean_time_training, mean_time_baseline, mean_speedup) %>%
      filter(Iteration %in% selected_iterations)
    
    plt <- bind_rows(combined_train, combined_baseline) %>%
      mutate(Method = case_when(`CL method` == "DER" ~ "CAL-DER",
                                `CL method` == "SD" ~ "CAL-SD",
                                `CL method` == "SDS2" ~ "CAL-SDS2",
                                `CL method` == "Baseline" ~ "AL"),
             Dataset = case_when(Dataset == "agnews" ~ "AG's news",
                                 Dataset == "sensation" ~ "Sensationalism",
                                 Dataset == "trec" ~ "TREC-6"),
             Method = factor(Method, levels = c("CAL-DER", "CAL-SD", "CAL-SDS2", "AL")),
             numeric_size = total_size / 10 + total_size / 10 * Iteration,
             Size = fct_reorder(paste0(numeric_size, "%"), numeric_size)
      ) %>%
      rename(`Sampling method` = "AL method") %>%
      rename(`Training method` = "Method") %>%
      ggplot(aes(x = mean_speedup, 
                 y = if(type == "relative") {
                   mean_f1_relative
                 } else if (type == "absolute") {
                   mean_f1_training
                 },
                 shape = `Training method`, color = `Sampling method`)) +
      geom_point(size = 3) + 
      scale_y_continuous(expand = expansion(mult = 0.1)) +
      scale_x_continuous(breaks = seq(1, 2.5, by = 0.5), limits = c(0.95, 2.5)) +
      facet_nested(Dataset ~ `PEFT module` + Size, scales = "free_y") + 
      ylab(paste("F1 score", ifelse(type == "relative", "(relative)", ifelse(type == "absolute", "", NA)))) +
      xlab("Speedup") +
      theme_bw() +
      theme(legend.position = "top", legend.box = "horizontal")
    print(plt)
    ggsave(file_path, plt, width = 250, height = 160, units = "mm", device = cairo_pdf)
  }
  
  # Short table creation and saving (results)
  create_results_table_short <- function(results_df_training, results_df_baselines, total_size, file_path) {
    dataset <- results_df_training$Dataset[1]
    
    results_df_training <- results_df_training %>%
      filter(`Total size` == total_size) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "CL method", "AL method"), first),
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        .groups = "drop"
      )
    
    results_df_baselines <- results_df_baselines %>%
      filter(`Total size` == total_size) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "AL method"), first),
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        .groups = "drop"
      ) %>% 
      mutate(`CL method` = "Baseline")
    
    results_df <- bind_rows(results_df_training, results_df_baselines) %>%
      mutate(Method = case_when(`CL method` == "DER" ~ "CAL-DER",
                                `CL method` == "SD" ~ "CAL-SD",
                                `CL method` == "SDS2" ~ "CAL-SDS2",
                                `CL method` == "Baseline" ~ "AL"))
    
    combined_max_f1 <- results_df %>%
      group_by(`PEFT module`, Iteration) %>%
      summarise(col_max = max(mean_f1), .groups = "drop")
    
    all_formatted <- results_df %>%
      left_join(combined_max_f1, by = c("PEFT module", "Iteration")) %>%
      mutate(
        is_best = round(mean_f1, 3) >= round(col_max, 3),
        Iteration_str = paste0((total_size / 10) + (Iteration * (total_size / 10)), "\\%"),
        mean_str = sub("^0\\.", ".", sprintf("%.3f", mean_f1)),
        mean_sd = ifelse(is_best, paste0("\\textbf{", mean_str, "}"), mean_str)
      ) %>%
      select(`PEFT module`, Method, `AL method`, Iteration_str, mean_sd) %>%
      mutate(Method = factor(Method, levels = c(setdiff(unique(Method), "AL"), "AL"))) %>%
      arrange(`PEFT module`, Method, `AL method`) %>%
      mutate(Method = as.character(Method)) %>%
      pivot_wider(names_from = Iteration_str, values_from = mean_sd) %>%
      rename(Training = Method) %>%
      rename(Sampling = `AL method`)
    
    peft_modules <- unique(all_formatted$`PEFT module`)
    
    tex_lines <- c(
      "\\begin{table}[htbp]",
      "\\centering"
    )
    
    for (i in seq_along(peft_modules)) {
      current_peft <- peft_modules[i]
      
      subtable_df <- all_formatted %>%
        filter(`PEFT module` == current_peft) %>%
        select(-`PEFT module`) 
      
      n_cols <- ncol(subtable_df)
      n_f1_cols <- n_cols - 2
      col_format <- paste0("c|c|", paste(rep("c", n_f1_cols), collapse = ""))
      
      latex_subtable <- subtable_df %>%
        kable(
          format = "latex", 
          escape = FALSE, 
          booktabs = TRUE,
          align = "l"
        ) %>%
        row_spec(0, bold = TRUE) %>%
        add_header_above(
          setNames(c(2, n_f1_cols), c("Method", "F1 score")), 
          bold = TRUE
        ) %>%
        collapse_rows(columns = 1, valign = "middle", latex_hline = "custom",
                      custom_latex_hline = 1) %>%
        gsub("\\\\begin\\{tabular\\}\\{[^\\}]+\\}", 
             sprintf("\\\\begin{tabular}{%s}", col_format), .) %>%
        gsub("\\\\multicolumn\\{2\\}\\{[^\\}]+\\}\\{\\\\textbf\\{Method\\}\\}", 
             "\\\\multicolumn\\{2\\}\\{c|\\}{\\\\textbf\\{Method\\}\\}", .) %>%
        gsub(sprintf("\\\\cmidrule.*?\\{1-2\\}\\s*\\\\cmidrule.*?\\{3-%d\\}", n_cols), "\\\\midrule", ., perl = TRUE) %>%
        gsub(sprintf("\\\\cline\\{1-2\\}\\s*\\\\cline\\{3-%d\\}", n_cols), "\\\\hline", .) %>%
        gsub(sprintf("\\\\cmidrule.*?\\{1-%d\\}", n_cols), "\\\\midrule", ., perl = TRUE) %>%
        gsub(sprintf("\\\\cline\\{1-%d\\}", n_cols), "\\\\hline", .)
      
      tex_lines <- c(
        tex_lines,
        "\\begin{subtable}{\\linewidth}",
        "\\centering",
        sprintf("\\caption{PEFT module -- %s}", current_peft),
        latex_subtable,
        "\\end{subtable}"
      )
      
      if (i < length(peft_modules)) {
        tex_lines <- c(tex_lines, "\\vspace{1em}")
      }
    }
    
    tex_lines <- c(
      tex_lines,
      sprintf("\\caption{Resulting macro F1 scores for %s dataset, split by the PEFT module used}",
              case_when(dataset == "agnews" ~ "AG's news",
                        dataset == "sensation" ~ "Sensationalism",
                        dataset == "trec" ~ "TREC-6")),
      sprintf("\\label{tab:results_short_%s_%s}", dataset, total_size),
      "\\end{table}"
    )
    final_tex <- paste(tex_lines, collapse = "\n")
    
    writeLines(final_tex, con = file_path)
  }
  
  # Full table creation and saving (results)
  create_results_table_full <- function(results_df_training, results_df_baselines, total_size, peft_module, file_path) {
    dataset <- results_df_training$Dataset[1]
    
    results_df_training <- results_df_training %>%
      filter(`Total size` == total_size, 
             `PEFT module` == case_when(
               peft_module == "adapter" ~ "Adapter",
               peft_module == "lora" ~ "LoRA",
               peft_module == "prefix" ~ "Prefix-tuning"
               )) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "CL method", "AL method"), first),
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        .groups = "drop"
      )
    
    results_df_baselines <- results_df_baselines %>%
      filter(`Total size` == total_size, 
             `PEFT module` == case_when(
               peft_module == "adapter" ~ "Adapter",
               peft_module == "lora" ~ "LoRA",
               peft_module == "prefix" ~ "Prefix-tuning"
             )) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "f1"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "AL method"), first),
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        .groups = "drop"
      ) %>% 
      mutate(`CL method` = "Baseline")
    
    results_df <- bind_rows(results_df_training, results_df_baselines) %>%
      mutate(Method = case_when(`CL method` == "DER" ~ "CAL-DER",
                                `CL method` == "SD" ~ "CAL-SD",
                                `CL method` == "SDS2" ~ "CAL-SDS2",
                                `CL method` == "Baseline" ~ "AL"))
    
    combined_max_f1 <- results_df %>%
      group_by(Iteration) %>%
      summarise(col_max = max(mean_f1), .groups = "drop")
    
    all_formatted <- results_df %>%
      left_join(combined_max_f1, by = "Iteration") %>%
      mutate(
        is_best = round(mean_f1, 3) >= round(col_max, 3),
        Iteration_str = paste0((total_size / 10) + (Iteration * (total_size / 10)), "\\%"),
        mean_str = sub("^0\\.", ".", sprintf("%.3f", mean_f1)),
        sd_str = ifelse(is.na(sd_f1), "", sub("^0\\.", ".", sprintf("%.3f", sd_f1))),
        mean_sd = ifelse(is.na(sd_f1), 
                         mean_str, 
                         paste0(mean_str, " $\\pm$ {\\small ", sd_str, "}")),
        mean_sd = ifelse(is_best, paste0("\\textbf{", mean_sd, "}"), mean_sd)
      ) %>%
      select(Method, `AL method`, Iteration_str, mean_sd) %>%
      mutate(Method = factor(Method, levels = c(setdiff(unique(Method), "AL"), "AL"))) %>%
      arrange(Method, `AL method`) %>%
      mutate(Method = as.character(Method)) %>%
      pivot_wider(names_from = Iteration_str, values_from = mean_sd) %>%
      rename(Training = Method) %>%
      rename(Sampling = `AL method`)
    
    table_tex_top_df <- all_formatted %>% select(1:2, 3:7)
    n_cols_top <- ncol(table_tex_top_df)
    n_f1_top <- n_cols_top - 2
    col_format_top <- paste0("c|c|@{\\\\extracolsep{\\\\fill}}", paste(rep("c", n_f1_top), collapse = ""))
    
    table_tex_top <- table_tex_top_df %>% 
      kable(
        format = "latex", 
        escape = FALSE, 
        booktabs = TRUE,
        align = "l"
      ) %>%
      row_spec(0, bold = TRUE) %>%
      add_header_above(
        setNames(c(2, n_f1_top), c("Method", "F1 score")), 
        bold = TRUE
      ) %>%
      collapse_rows(columns = 1, valign = "middle", latex_hline = "custom",
                    custom_latex_hline = 1) %>%
      gsub("\\\\begin\\{tabular\\}\\{[^\\}]+\\}", 
           sprintf("\\\\begin{tabular*}{\\\\linewidth}{%s}", col_format_top), .) %>%
      gsub("\\\\end\\{tabular\\}", "\\\\end{tabular*}", .) %>%
      gsub("\\\\multicolumn\\{2\\}\\{[^\\}]+\\}\\{\\\\textbf\\{Method\\}\\}", 
           "\\\\multicolumn\\{2\\}\\{c|\\}{\\\\textbf\\{Method\\}\\}", .) %>%
      gsub(sprintf("\\\\cmidrule.*?\\{1-2\\}\\s*\\\\cmidrule.*?\\{3-%d\\}", n_cols_top), "\\\\midrule", ., perl = TRUE) %>%
      gsub(sprintf("\\\\cline\\{1-2\\}\\s*\\\\cline\\{3-%d\\}", n_cols_top), "\\\\hline", .) %>%
      gsub(sprintf("\\\\cmidrule.*?\\{1-%d\\}", n_cols_top), "\\\\midrule", ., perl = TRUE) %>%
      gsub(sprintf("\\\\cline\\{1-%d\\}", n_cols_top), "\\\\hline", .)
    
    table_tex_bottom_df <- all_formatted %>% select(1:2, 8:12)
    n_cols_bot <- ncol(table_tex_bottom_df)
    n_f1_bot <- n_cols_bot - 2
    col_format_bot <- paste0("c|c|@{\\\\extracolsep{\\\\fill}}", paste(rep("c", n_f1_bot), collapse = ""))
    
    table_tex_bottom <- table_tex_bottom_df %>% 
      kable(
        format = "latex", 
        escape = FALSE, 
        booktabs = TRUE,
        align = "l"
      ) %>%
      row_spec(0, bold = TRUE) %>%
      add_header_above(
        setNames(c(2, n_f1_bot), c("Method", "F1 score")), 
        bold = TRUE
      ) %>%
      collapse_rows(columns = 1, valign = "middle", latex_hline = "custom",
                    custom_latex_hline = 1) %>%
      gsub("\\\\begin\\{tabular\\}\\{[^\\}]+\\}", 
           sprintf("\\\\begin{tabular*}{\\\\linewidth}{%s}", col_format_bot), .) %>%
      gsub("\\\\end\\{tabular\\}", "\\\\end{tabular*}", .) %>%
      gsub("\\\\multicolumn\\{2\\}\\{[^\\}]+\\}\\{\\\\textbf\\{Method\\}\\}", 
           "\\\\multicolumn\\{2\\}\\{c|\\}{\\\\textbf\\{Method\\}\\}", .) %>%
      gsub(sprintf("\\\\cmidrule.*?\\{1-2\\}\\s*\\\\cmidrule.*?\\{3-%d\\}", n_cols_bot), "\\\\midrule", ., perl = TRUE) %>%
      gsub(sprintf("\\\\cline\\{1-2\\}\\s*\\\\cline\\{3-%d\\}", n_cols_bot), "\\\\hline", .) %>%
      gsub(sprintf("\\\\cmidrule.*?\\{1-%d\\}", n_cols_bot), "\\\\midrule", ., perl = TRUE) %>%
      gsub(sprintf("\\\\cline\\{1-%d\\}", n_cols_bot), "\\\\hline", .)
    
    final_tex <- paste(
      "\\begin{table}[htbp]",
      "\\centering",
      table_tex_top,
      "\\vspace{0.5em}",
      table_tex_bottom,
      sprintf("\\caption{Resulting macro F1 scores for %s dataset with %s module used}",
              case_when(dataset == "agnews" ~ "AG's news",
                        dataset == "sensation" ~ "Sensationalism",
                        dataset == "trec" ~ "TREC-6"),
              case_when(peft_module == "adapter" ~ "adapter",
                        peft_module == "lora" ~ "LoRA",
                        peft_module == "prefix" ~ "prefix-tuning")),
      sprintf("\\label{tab:results_full_%s_%s_%s}", dataset, total_size, peft_module),
      "\\end{table}",
      sep = "\n"
    )
    
    writeLines(final_tex, con = file_path)
  }
  
  # Short table creation and saving (time)
  create_time_table_short <- function(time_df_training, time_df_baselines, total_size, file_path) {
    dataset <- time_df_training$Dataset[1]
    
    time_df_training <- time_df_training %>%
      filter(`Total size` == total_size) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(
        Iteration = as.numeric(str_remove(Iteration, "time_")),
        time = time / 1000 / 60
      ) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "CL method", "AL method"), first),
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
      )
    
    time_df_baselines <- time_df_baselines %>%
      filter(`Total size` == total_size) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(
        Iteration = as.numeric(str_remove(Iteration, "time_")),
        time = time / 1000 / 60
      ) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "AL method"), first),
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
      ) %>% 
      mutate(`CL method` = "Baseline")
    
    results_df <- bind_rows(time_df_training, time_df_baselines) %>%
      mutate(Method = case_when(`CL method` == "DER" ~ "CAL-DER",
                                `CL method` == "SD" ~ "CAL-SD",
                                `CL method` == "SDS2" ~ "CAL-SDS2",
                                `CL method` == "Baseline" ~ "AL"))
    
    all_formatted <- results_df %>%
      mutate(
        Iteration_str = paste0((total_size / 10) + (Iteration * (total_size / 10)), "\\%"),
        mean_formatted = sprintf("%.1f", mean_time)
      ) %>%
      select(`PEFT module`, Method, `AL method`, Iteration_str, mean_formatted) %>%
      mutate(Method = factor(Method, levels = c(setdiff(unique(Method), "AL"), "AL"))) %>%
      arrange(`PEFT module`, Method, `AL method`) %>%
      mutate(Method = as.character(Method)) %>%
      pivot_wider(names_from = Iteration_str, values_from = mean_formatted) %>%
      rename(Training = Method) %>%
      rename(Sampling = `AL method`)
    
    peft_modules <- unique(all_formatted$`PEFT module`)
    
    tex_lines <- c(
      "\\begin{table}[htbp]",
      "\\centering"
    )
    
    for (i in seq_along(peft_modules)) {
      current_peft <- peft_modules[i]
      
      subtable_df <- all_formatted %>%
        filter(`PEFT module` == current_peft) %>%
        select(-`PEFT module`) 
      
      n_cols <- ncol(subtable_df)
      n_time_cols <- n_cols - 2
      col_format <- paste0("c|c|", paste(rep("c", n_time_cols), collapse = ""))
      
      latex_subtable <- subtable_df %>%
        kable(
          format = "latex", 
          escape = FALSE, 
          booktabs = TRUE,
          align = "l"
        ) %>%
        row_spec(0, bold = TRUE) %>%
        add_header_above(
          setNames(c(2, n_time_cols), c("Method", "Time (min)")), 
          bold = TRUE
        ) %>%
        collapse_rows(columns = 1, valign = "middle", latex_hline = "custom",
                      custom_latex_hline = 1) %>%
        gsub("\\\\begin\\{tabular\\}\\{[^\\}]+\\}", 
             sprintf("\\\\begin{tabular}{%s}", col_format), .) %>%
        gsub("\\\\multicolumn\\{2\\}\\{[^\\}]+\\}\\{\\\\textbf\\{Method\\}\\}", 
             "\\\\multicolumn\\{2\\}\\{c|\\}{\\\\textbf\\{Method\\}\\}", .) %>%
        gsub(sprintf("\\\\cmidrule.*?\\{1-2\\}\\s*\\\\cmidrule.*?\\{3-%d\\}", n_cols), "\\\\midrule", ., perl = TRUE) %>%
        gsub(sprintf("\\\\cline\\{1-2\\}\\s*\\\\cline\\{3-%d\\}", n_cols), "\\\\hline", .) %>%
        gsub(sprintf("\\\\cmidrule.*?\\{1-%d\\}", n_cols), "\\\\midrule", ., perl = TRUE) %>%
        gsub(sprintf("\\\\cline\\{1-%d\\}", n_cols), "\\\\hline", .)
      
      tex_lines <- c(
        tex_lines,
        "\\begin{subtable}{\\linewidth}",
        "\\centering",
        sprintf("\\caption{PEFT module -- %s}", current_peft),
        latex_subtable,
        "\\end{subtable}"
      )
      
      if (i < length(peft_modules)) {
        tex_lines <- c(tex_lines, "\\vspace{1em}")
      }
    }
    
    tex_lines <- c(
      tex_lines,
      sprintf("\\caption{Resulting time (in minutes) for %s dataset, split by the PEFT module used}",
              case_when(dataset == "agnews" ~ "AG's news",
                        dataset == "sensation" ~ "Sensationalism",
                        dataset == "trec" ~ "TREC-6")),
      sprintf("\\label{tab:time_short_%s_%s}", dataset, total_size),
      "\\end{table}"
    )
    final_tex <- paste(tex_lines, collapse = "\n")
    
    writeLines(final_tex, con = file_path)
  }
  
  # Full table creation and saving (time)
  create_time_table_full <- function(time_df_training, time_df_baselines, total_size, peft_module, file_path) {
    dataset <- time_df_training$Dataset[1]
    
    time_df_training <- time_df_training %>%
      filter(`Total size` == total_size, 
             `PEFT module` == case_when(
               peft_module == "adapter" ~ "Adapter",
               peft_module == "lora" ~ "LoRA",
               peft_module == "prefix" ~ "Prefix-tuning"
             )) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(
        Iteration = as.numeric(str_remove(Iteration, "time_")),
        time = time / 1000 / 60
      ) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "CL method", "AL method"), first),
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
      )
    
    time_df_baselines <- time_df_baselines %>%
      filter(`Total size` == total_size, 
             `PEFT module` == case_when(
               peft_module == "adapter" ~ "Adapter",
               peft_module == "lora" ~ "LoRA",
               peft_module == "prefix" ~ "Prefix-tuning"
             )) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(
        Iteration = as.numeric(str_remove(Iteration, "time_")),
        time = time / 1000 / 60
      ) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "AL method"), first),
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
      ) %>% 
      mutate(`CL method` = "Baseline")
    
    results_df <- bind_rows(time_df_training, time_df_baselines) %>%
      mutate(Method = case_when(`CL method` == "DER" ~ "CAL-DER",
                                `CL method` == "SD" ~ "CAL-SD",
                                `CL method` == "SDS2" ~ "CAL-SDS2",
                                `CL method` == "Baseline" ~ "AL"))
    
    all_formatted <- results_df %>%
      mutate(
        Iteration_str = paste0((total_size / 10) + (Iteration * (total_size / 10)), "\\%"),
        mean_str = sprintf("%.1f", mean_time),
        sd_str = ifelse(is.na(sd_time), "", sprintf("%.2f", sd_time)),
        mean_sd = ifelse(is.na(sd_time), 
                         mean_str, 
                         paste0(mean_str, " $\\pm$ {\\small ", sd_str, "}"))
      ) %>%
      select(Method, `AL method`, Iteration_str, mean_sd) %>%
      mutate(Method = factor(Method, levels = c(setdiff(unique(Method), "AL"), "AL"))) %>%
      arrange(Method, `AL method`) %>%
      mutate(Method = as.character(Method)) %>%
      pivot_wider(names_from = Iteration_str, values_from = mean_sd) %>%
      rename(Training = Method) %>%
      rename(Sampling = `AL method`)
    
    table_tex_top_df <- all_formatted %>% select(1:2, 3:7)
    n_cols_top <- ncol(table_tex_top_df)
    n_time_top <- n_cols_top - 2
    col_format_top <- paste0("c|c|@{\\\\extracolsep{\\\\fill}}", paste(rep("c", n_time_top), collapse = ""))
    
    table_tex_top <- table_tex_top_df %>% 
      kable(
        format = "latex", 
        escape = FALSE, 
        booktabs = TRUE,
        align = "l"
      ) %>%
      row_spec(0, bold = TRUE) %>%
      add_header_above(
        setNames(c(2, n_time_top), c("Method", "Time (min)")), 
        bold = TRUE
      ) %>%
      collapse_rows(columns = 1, valign = "middle", latex_hline = "custom",
                    custom_latex_hline = 1) %>%
      gsub("\\\\begin\\{tabular\\}\\{[^\\}]+\\}", 
           sprintf("\\\\begin{tabular*}{\\\\linewidth}{%s}", col_format_top), .) %>%
      gsub("\\\\end\\{tabular\\}", "\\\\end{tabular*}", .) %>%
      gsub("\\\\multicolumn\\{2\\}\\{[^\\}]+\\}\\{\\\\textbf\\{Method\\}\\}", 
           "\\\\multicolumn\\{2\\}\\{c|\\}{\\\\textbf\\{Method\\}\\}", .) %>%
      gsub(sprintf("\\\\cmidrule.*?\\{1-2\\}\\s*\\\\cmidrule.*?\\{3-%d\\}", n_cols_top), "\\\\midrule", ., perl = TRUE) %>%
      gsub(sprintf("\\\\cline\\{1-2\\}\\s*\\\\cline\\{3-%d\\}", n_cols_top), "\\\\hline", .) %>%
      gsub(sprintf("\\\\cmidrule.*?\\{1-%d\\}", n_cols_top), "\\\\midrule", ., perl = TRUE) %>%
      gsub(sprintf("\\\\cline\\{1-%d\\}", n_cols_top), "\\\\hline", .)
    
    table_tex_bottom_df <- all_formatted %>% select(1:2, 8:12)
    n_cols_bot <- ncol(table_tex_bottom_df)
    n_time_bot <- n_cols_bot - 2
    col_format_bot <- paste0("c|c|@{\\\\extracolsep{\\\\fill}}", paste(rep("c", n_time_bot), collapse = ""))
    
    table_tex_bottom <- table_tex_bottom_df %>% 
      kable(
        format = "latex", 
        escape = FALSE, 
        booktabs = TRUE,
        align = "l"
      ) %>%
      row_spec(0, bold = TRUE) %>%
      add_header_above(
        setNames(c(2, n_time_bot), c("Method", "Time (min)")), 
        bold = TRUE
      ) %>%
      collapse_rows(columns = 1, valign = "middle", latex_hline = "custom",
                    custom_latex_hline = 1) %>%
      gsub("\\\\begin\\{tabular\\}\\{[^\\}]+\\}", 
           sprintf("\\\\begin{tabular*}{\\\\linewidth}{%s}", col_format_bot), .) %>%
      gsub("\\\\end\\{tabular\\}", "\\\\end{tabular*}", .) %>%
      gsub("\\\\multicolumn\\{2\\}\\{[^\\}]+\\}\\{\\\\textbf\\{Method\\}\\}", 
           "\\\\multicolumn\\{2\\}\\{c|\\}{\\\\textbf\\{Method\\}\\}", .) %>%
      gsub(sprintf("\\\\cmidrule.*?\\{1-2\\}\\s*\\\\cmidrule.*?\\{3-%d\\}", n_cols_bot), "\\\\midrule", ., perl = TRUE) %>%
      gsub(sprintf("\\\\cline\\{1-2\\}\\s*\\\\cline\\{3-%d\\}", n_cols_bot), "\\\\hline", .) %>%
      gsub(sprintf("\\\\cmidrule.*?\\{1-%d\\}", n_cols_bot), "\\\\midrule", ., perl = TRUE) %>%
      gsub(sprintf("\\\\cline\\{1-%d\\}", n_cols_bot), "\\\\hline", .)
    
    final_tex <- paste(
      "\\begin{table}[htbp]",
      "\\centering",
      table_tex_top,
      "\\vspace{0.5em}",
      table_tex_bottom,
      sprintf("\\caption{Resulting time (in minutes) for %s dataset with %s module used}",
              case_when(dataset == "agnews" ~ "AG's news",
                        dataset == "sensation" ~ "Sensationalism",
                        dataset == "trec" ~ "TREC-6"),
              case_when(peft_module == "adapter" ~ "adapter",
                        peft_module == "lora" ~ "LoRA",
                        peft_module == "prefix" ~ "prefix-tuning")),
      sprintf("\\label{tab:time_full_%s_%s_%s}", dataset, total_size, peft_module),
      "\\end{table}",
      sep = "\n"
    )
    
    writeLines(final_tex, con = file_path)
  }
  
  # Short table creation and saving (speedup)
  create_speedup_table <- function(time_df_training, time_df_baselines, total_size, file_path) {
    dataset <- time_df_training$Dataset[1]
    
    time_df_training_proc <- time_df_training %>%
      filter(`Total size` == total_size) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "time_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "CL method", "AL method"), first),
        mean_time_training = mean(time),
        .groups = "drop"
      )
    
    time_df_baselines_proc <- time_df_baselines %>%
      filter(`Total size` == total_size) %>%
      select(-Dataset, -`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      pivot_longer(
        cols = starts_with("time_"),
        names_to = "Iteration",
        values_to = "time"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "time_"))) %>%
      group_by(group_id, Iteration) %>%
      summarise(
        across(c("PEFT module", "AL method"), first),
        mean_time_baseline = mean(time),
        .groups = "drop"
      ) %>% 
      mutate(`CL method` = "Baseline")
    
    speedup_training <- time_df_training_proc %>%
      left_join(time_df_baselines_proc %>% select(`PEFT module`, `AL method`, Iteration, mean_time_baseline), 
                by = c("PEFT module", "AL method", "Iteration")) %>%
      mutate(
        mean_speedup = mean_time_baseline / mean_time_training,
        Method = case_when(`CL method` == "DER" ~ "CAL-DER",
                           `CL method` == "SD" ~ "CAL-SD",
                           `CL method` == "SDS2" ~ "CAL-SDS2")
      )
    
    speedup_baselines <- time_df_baselines_proc %>%
      mutate(
        mean_speedup = 1,
        Method = "AL"
      )
    
    results_df <- bind_rows(speedup_training, speedup_baselines)
    
    all_formatted <- results_df %>%
      mutate(
        Iteration_str = paste0((total_size / 10) + (Iteration * (total_size / 10)), "\\%"),
        mean_formatted = sprintf("%.1f$\\times$", mean_speedup)
      ) %>%
      select(`PEFT module`, Method, `AL method`, Iteration_str, mean_formatted) %>%
      mutate(Method = factor(Method, levels = c(setdiff(unique(Method), "AL"), "AL"))) %>%
      arrange(`PEFT module`, Method, `AL method`) %>%
      mutate(Method = as.character(Method)) %>%
      pivot_wider(names_from = Iteration_str, values_from = mean_formatted) %>%
      rename(Training = Method) %>%
      rename(Sampling = `AL method`)
    
    peft_modules <- unique(all_formatted$`PEFT module`)
    
    tex_lines <- c(
      "\\begin{table}[htbp]",
      "\\centering"
    )
    
    for (i in seq_along(peft_modules)) {
      current_peft <- peft_modules[i]
      
      subtable_df <- all_formatted %>%
        filter(`PEFT module` == current_peft) %>%
        select(-`PEFT module`) 
      
      n_cols <- ncol(subtable_df)
      n_speedup_cols <- n_cols - 2
      col_format <- paste0("c|c|", paste(rep("c", n_speedup_cols), collapse = ""))
      
      latex_subtable <- subtable_df %>%
        kable(
          format = "latex", 
          escape = FALSE, 
          booktabs = TRUE,
          align = "l"
        ) %>%
        row_spec(0, bold = TRUE) %>%
        add_header_above(
          setNames(c(2, n_speedup_cols), c("Method", "Speedup")), 
          bold = TRUE
        ) %>%
        collapse_rows(columns = 1, valign = "middle", latex_hline = "custom",
                      custom_latex_hline = 1) %>%
        gsub("\\\\begin\\{tabular\\}\\{[^\\}]+\\}", 
             sprintf("\\\\begin{tabular}{%s}", col_format), .) %>%
        gsub("\\\\multicolumn\\{2\\}\\{[^\\}]+\\}\\{\\\\textbf\\{Method\\}\\}", 
             "\\\\multicolumn\\{2\\}\\{c|\\}{\\\\textbf\\{Method\\}\\}", .) %>%
        gsub(sprintf("\\\\cmidrule.*?\\{1-2\\}\\s*\\\\cmidrule.*?\\{3-%d\\}", n_cols), "\\\\midrule", ., perl = TRUE) %>%
        gsub(sprintf("\\\\cline\\{1-2\\}\\s*\\\\cline\\{3-%d\\}", n_cols), "\\\\hline", .) %>%
        gsub(sprintf("\\\\cmidrule.*?\\{1-%d\\}", n_cols), "\\\\midrule", ., perl = TRUE) %>%
        gsub(sprintf("\\\\cline\\{1-%d\\}", n_cols), "\\\\hline", .)
      
      tex_lines <- c(
        tex_lines,
        "\\begin{subtable}{\\linewidth}",
        "\\centering",
        sprintf("\\caption{PEFT module -- %s}", current_peft),
        latex_subtable,
        "\\end{subtable}"
      )
      
      if (i < length(peft_modules)) {
        tex_lines <- c(tex_lines, "\\vspace{1em}")
      }
    }
    
    tex_lines <- c(
      tex_lines,
      sprintf("\\caption{Resulting speedup for %s dataset, split by the PEFT module used}",
              case_when(dataset == "agnews" ~ "AG's news",
                        dataset == "sensation" ~ "Sensationalism",
                        dataset == "trec" ~ "TREC-6")),
      sprintf("\\label{tab:speedup_short_%s_%s}", dataset, total_size),
      "\\end{table}"
    )
    final_tex <- paste(tex_lines, collapse = "\n")
    
    writeLines(final_tex, con = file_path)
  }
  
  # Full baselines table creation (results and time)
  create_baselines_full_table <- function(results_df_baselines_full_list, time_df_baselines_full_list, file_path) {
    results_df_baselines_full <- do.call(rbind, results_df_baselines_full_list)
    time_df_baselines_full <- do.call(rbind, time_df_baselines_full_list)
    
    results_df_baselines_full <- results_df_baselines_full %>%
      select(-`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      rename(f1 = f1_0) %>%
      group_by(group_id) %>%
      summarise(
        across(c("Dataset", "PEFT module"), first),
        mean_f1 = mean(f1),
        sd_f1 = sd(f1),
        .groups = "drop"
      ) %>%
      select(-group_id)
    
    time_df_baselines_full <- time_df_baselines_full %>%
      select(-`Total size`) %>%
      mutate(group_id = ceiling(row_number() / max(run))) %>%
      rename(time = time_0) %>%
      mutate(time = time / 1000 / 60) %>%
      group_by(group_id) %>%
      summarise(
        across(c("Dataset", "PEFT module"), first),
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
      ) %>%
      select(-group_id)
    
    all_formatted <- results_df_baselines_full %>%
      left_join(time_df_baselines_full, by = c("Dataset", "PEFT module")) %>%
      mutate(
        mean_str_f1 = sub("^0\\.", ".", sprintf("%.3f", mean_f1)),
        sd_str_f1 = ifelse(is.na(sd_f1), "", sub("^0\\.", ".", sprintf("%.3f", sd_f1))),
        mean_sd_f1 = ifelse(is.na(sd_f1), 
                            mean_str_f1, 
                            paste0(mean_str_f1, " $\\pm$ {\\small ", sd_str_f1, "}"))
      )  %>%
      mutate(
        mean_str_time = sprintf("%.1f", mean_time),
        sd_str_time = ifelse(is.na(sd_time), "", sprintf("%.2f", sd_time)),
        mean_sd_time = ifelse(is.na(sd_time), 
                              mean_str_time, 
                              paste0(mean_str_time, " $\\pm$ {\\small ", sd_str_time, "}"))
      ) %>%
      mutate(Dataset = case_when(Dataset == "agnews" ~ "AG's news",
                                 Dataset == "sensation" ~ "Sensationalism",
                                 Dataset == "trec" ~ "TREC-6")) %>%
      select(Dataset, `PEFT module`, `F1 score` = mean_sd_f1, `Time (min)` = mean_sd_time) %>%
      arrange(Dataset, `PEFT module`)
    
    n_cols <- ncol(all_formatted)
    col_format <- paste(rep("c", n_cols), collapse = "|")
    
    latex_table <- all_formatted %>%
      kable(
        format = "latex", 
        escape = FALSE, 
        booktabs = TRUE,
        align = "c"
      ) %>%
      row_spec(0, bold = TRUE) %>%
      collapse_rows(columns = 1, valign = "middle", latex_hline = "custom", custom_latex_hline = 1) %>%
      gsub("\\\\begin\\{tabular\\}\\{[^\\}]+\\}", 
           sprintf("\\\\begin{tabular}{%s}", col_format), .) %>%
      gsub(sprintf("\\\\cmidrule.*?\\{1-%d\\}", n_cols), "\\\\hline", ., perl = TRUE) %>%
      gsub(sprintf("\\\\cline\\{1-%d\\}", n_cols), "\\\\hline", .)
    
    tex_lines <- c(
      "\\begin{table}[htbp]",
      "\\centering",
      latex_table,
      "\\caption{Resulting macro F1 scores and time (in minutes) for the full dataset training}",
      "\\label{tab:baselines_full}",
      "\\end{table}"
    )
    
    final_tex <- paste(tex_lines, collapse = "\n")
    writeLines(final_tex, con = file_path)
  }
}

results_df_training_agnews <- read_results_training("results_training/agnews/results_f1.csv")
results_df_training_sensation <- read_results_training("results_training/sensation/results_f1.csv")
results_df_training_trec <- read_results_training("results_training/trec/results_f1.csv")

results_df_baselines_agnews <- read_results_baselines("results_baselines/baselines_1/agnews/results_f1.csv", type = "al")
results_df_baselines_sensation <- read_results_baselines("results_baselines/baselines_1/sensation/results_f1.csv", type = "al")
results_df_baselines_trec <- read_results_baselines("results_baselines/baselines_1/trec/results_f1.csv", type = "al")

time_df_training_agnews <- read_time_training("time_training/agnews/")
time_df_training_sensation <- read_time_training("time_training/sensation/")
time_df_training_trec <- read_time_training("time_training/trec/")

time_df_baselines_agnews <- read_time_baselines("time_baselines/baselines_1/agnews/", type = "al")
time_df_baselines_sensation <- read_time_baselines("time_baselines/baselines_1/sensation/", type = "al")
time_df_baselines_trec <- read_time_baselines("time_baselines/baselines_1/trec/", type = "al")

results_df_baselines_full_agnews <- read_results_baselines("results_baselines/baselines_2/agnews/results_f1.csv", type = "full")
results_df_baselines_full_sensation <- read_results_baselines("results_baselines/baselines_2/sensation/results_f1.csv", type = "full")
results_df_baselines_full_trec <- read_results_baselines("results_baselines/baselines_2/trec/results_f1.csv", type = "full")

time_df_baselines_full_agnews <- read_time_baselines("time_baselines/baselines_2/agnews/", type = "full")
time_df_baselines_full_sensation <- read_time_baselines("time_baselines/baselines_2/sensation/", type = "full")
time_df_baselines_full_trec <- read_time_baselines("time_baselines/baselines_2/trec/", type = "full")

results_df_training_list <- list(results_df_training_agnews, results_df_training_sensation, results_df_training_trec)
results_df_baselines_list <- list(results_df_baselines_agnews, results_df_baselines_sensation, results_df_baselines_trec)
time_df_training_list <- list(time_df_training_agnews, time_df_training_sensation, time_df_training_trec)
time_df_baselines_list <- list(time_df_baselines_agnews, time_df_baselines_sensation, time_df_baselines_trec)

results_df_baselines_full_list <- list(results_df_baselines_full_agnews, results_df_baselines_full_sensation, results_df_baselines_full_trec)
time_df_baselines_full_list <- list(time_df_baselines_full_agnews, time_df_baselines_full_sensation, time_df_baselines_full_trec)

create_plot_results(results_df_training_agnews, results_df_baselines_agnews, 
                    10, "plots_training/training/training_10_agnews.pdf")
create_plot_results(results_df_training_sensation, results_df_baselines_sensation, 
                    10, "plots_training/training/training_10_sensation.pdf")
create_plot_results(results_df_training_trec, results_df_baselines_trec, 
                    10, "plots_training/training/training_10_trec.pdf")

create_plot_speedup(results_df_training_agnews, results_df_baselines_agnews,
                    time_df_training_agnews, time_df_baselines_agnews,
                    10, "adapter", "absolute", "plots_training/speedup_absolute/speedup_10_agnews_adapter.pdf")
create_plot_speedup(results_df_training_agnews, results_df_baselines_agnews,
                    time_df_training_agnews, time_df_baselines_agnews,
                    10, "lora", "absolute", "plots_training/speedup_absolute/speedup_10_agnews_lora.pdf")
create_plot_speedup(results_df_training_agnews, results_df_baselines_agnews,
                    time_df_training_agnews, time_df_baselines_agnews,
                    10, "prefix", "absolute", "plots_training/speedup_absolute/speedup_10_agnews_prefix.pdf")
create_plot_speedup(results_df_training_sensation, results_df_baselines_sensation,
                    time_df_training_sensation, time_df_baselines_sensation,
                    10, "adapter", "absolute", "plots_training/speedup_absolute/speedup_10_sensation_adapter.pdf")
create_plot_speedup(results_df_training_sensation, results_df_baselines_sensation,
                    time_df_training_sensation, time_df_baselines_sensation,
                    10, "lora", "absolute", "plots_training/speedup_absolute/speedup_10_sensation_lora.pdf")
create_plot_speedup(results_df_training_sensation, results_df_baselines_sensation,
                    time_df_training_sensation, time_df_baselines_sensation,
                    10, "prefix", "absolute", "plots_training/speedup_absolute/speedup_10_sensation_prefix.pdf")
create_plot_speedup(results_df_training_trec, results_df_baselines_trec,
                    time_df_training_trec, time_df_baselines_trec,
                    10, "adapter", "absolute", "plots_training/speedup_absolute/speedup_10_trec_adapter.pdf")
create_plot_speedup(results_df_training_trec, results_df_baselines_trec,
                    time_df_training_trec, time_df_baselines_trec,
                    10, "lora", "absolute", "plots_training/speedup_absolute/speedup_10_trec_lora.pdf")
create_plot_speedup(results_df_training_trec, results_df_baselines_trec,
                    time_df_training_trec, time_df_baselines_trec,
                    10, "prefix", "absolute", "plots_training/speedup_absolute/speedup_10_trec_prefix.pdf")

create_plot_speedup(results_df_training_agnews, results_df_baselines_agnews,
                    time_df_training_agnews, time_df_baselines_agnews,
                    10, "adapter", "relative", "plots_training/speedup_relative/speedup_10_agnews_adapter.pdf")
create_plot_speedup(results_df_training_agnews, results_df_baselines_agnews,
                    time_df_training_agnews, time_df_baselines_agnews,
                    10, "lora", "relative", "plots_training/speedup_relative/speedup_10_agnews_lora.pdf")
create_plot_speedup(results_df_training_agnews, results_df_baselines_agnews,
                    time_df_training_agnews, time_df_baselines_agnews,
                    10, "prefix", "relative", "plots_training/speedup_relative/speedup_10_agnews_prefix.pdf")
create_plot_speedup(results_df_training_sensation, results_df_baselines_sensation,
                    time_df_training_sensation, time_df_baselines_sensation,
                    10, "adapter", "relative", "plots_training/speedup_relative/speedup_10_sensation_adapter.pdf")
create_plot_speedup(results_df_training_sensation, results_df_baselines_sensation,
                    time_df_training_sensation, time_df_baselines_sensation,
                    10, "lora", "relative", "plots_training/speedup_relative/speedup_10_sensation_lora.pdf")
create_plot_speedup(results_df_training_sensation, results_df_baselines_sensation,
                    time_df_training_sensation, time_df_baselines_sensation,
                    10, "prefix", "relative", "plots_training/speedup_relative/speedup_10_sensation_prefix.pdf")
create_plot_speedup(results_df_training_trec, results_df_baselines_trec,
                    time_df_training_trec, time_df_baselines_trec,
                    10, "adapter", "relative", "plots_training/speedup_relative/speedup_10_trec_adapter.pdf")
create_plot_speedup(results_df_training_trec, results_df_baselines_trec,
                    time_df_training_trec, time_df_baselines_trec,
                    10, "lora", "relative", "plots_training/speedup_relative/speedup_10_trec_lora.pdf")
create_plot_speedup(results_df_training_trec, results_df_baselines_trec,
                    time_df_training_trec, time_df_baselines_trec,
                    10, "prefix", "relative", "plots_training/speedup_relative/speedup_10_trec_prefix.pdf")

create_plot_speedup_combined(results_df_training_list, results_df_baselines_list,
                             time_df_training_list, time_df_baselines_list,
                             10, c(4, 9), "relative", "plots_training/speedup_relative/speedup_10_combined.pdf")

create_plot_speedup_combined(results_df_training_list, results_df_baselines_list,
                             time_df_training_list, time_df_baselines_list,
                             10, c(4, 9), "absolute", "plots_training/speedup_absolute/speedup_10_combined.pdf")

create_results_table_short(results_df_training_agnews, results_df_baselines_agnews, 
                           10, "tables_training/results_short/results_agnews_10.tex")
create_results_table_short(results_df_training_sensation, results_df_baselines_sensation,
                           10, "tables_training/results_short/results_sensation_10.tex")
create_results_table_short(results_df_training_trec, results_df_baselines_trec,
                           10, "tables_training/results_short/results_trec_10.tex")

create_results_table_full(results_df_training_agnews, results_df_baselines_agnews,
                          10, "adapter", "tables_training/results_full/results_agnews_10_adapter.tex")
create_results_table_full(results_df_training_agnews, results_df_baselines_agnews,
                          10, "lora", "tables_training/results_full/results_agnews_10_lora.tex")
create_results_table_full(results_df_training_agnews, results_df_baselines_agnews,
                          10, "prefix", "tables_training/results_full/results_agnews_10_prefix.tex")
create_results_table_full(results_df_training_sensation, results_df_baselines_sensation,
                          10, "adapter", "tables_training/results_full/results_sensation_10_adapter.tex")
create_results_table_full(results_df_training_sensation, results_df_baselines_sensation,
                          10, "lora", "tables_training/results_full/results_sensation_10_lora.tex")
create_results_table_full(results_df_training_sensation, results_df_baselines_sensation,
                          10, "prefix", "tables_training/results_full/results_sensation_10_prefix.tex")
create_results_table_full(results_df_training_trec, results_df_baselines_trec, 
                          10, "adapter", "tables_training/results_full/results_trec_10_adapter.tex")
create_results_table_full(results_df_training_trec, results_df_baselines_trec,
                          10, "lora", "tables_training/results_full/results_trec_10_lora.tex")
create_results_table_full(results_df_training_trec, results_df_baselines_trec,
                          10, "prefix", "tables_training/results_full/results_trec_10_prefix.tex")

create_time_table_short(time_df_training_agnews, time_df_baselines_agnews, 
                        10, "tables_training/time_short/time_agnews_10.tex")
create_time_table_short(time_df_training_sensation, time_df_baselines_sensation,
                        10, "tables_training/time_short/time_sensation_10.tex")
create_time_table_short(time_df_training_trec, time_df_baselines_trec,
                        10, "tables_training/time_short/time_trec_10.tex")

create_time_table_full(time_df_training_agnews, time_df_baselines_agnews,
                       10, "adapter", "tables_training/time_full/time_agnews_10_adapter.tex")
create_time_table_full(time_df_training_agnews, time_df_baselines_agnews,
                       10, "lora", "tables_training/time_full/time_agnews_10_lora.tex")
create_time_table_full(time_df_training_agnews, time_df_baselines_agnews,
                       10, "prefix", "tables_training/time_full/time_agnews_10_prefix.tex")
create_time_table_full(time_df_training_sensation, time_df_baselines_sensation,
                       10, "adapter", "tables_training/time_full/time_sensation_10_adapter.tex")
create_time_table_full(time_df_training_sensation, time_df_baselines_sensation,
                       10, "lora", "tables_training/time_full/time_sensation_10_lora.tex")
create_time_table_full(time_df_training_sensation, time_df_baselines_sensation,
                       10, "prefix", "tables_training/time_full/time_sensation_10_prefix.tex")
create_time_table_full(time_df_training_trec, time_df_baselines_trec, 
                       10, "adapter", "tables_training/time_full/time_trec_10_adapter.tex")
create_time_table_full(time_df_training_trec, time_df_baselines_trec,
                       10, "lora", "tables_training/time_full/time_trec_10_lora.tex")
create_time_table_full(time_df_training_trec, time_df_baselines_trec,
                       10, "prefix", "tables_training/time_full/time_trec_10_prefix.tex")

create_speedup_table(time_df_training_agnews, time_df_baselines_agnews, 
                     10, "tables_training/speedup/speedup_agnews_10.tex")
create_speedup_table(time_df_training_sensation, time_df_baselines_sensation,
                     10, "tables_training/speedup/speedup_sensation_10.tex")
create_speedup_table(time_df_training_trec, time_df_baselines_trec,
                     10, "tables_training/speedup/speedup_trec_10.tex")

create_baselines_full_table(results_df_baselines_full_list, time_df_baselines_full_list, "tables_training/baselines/baselines_full.tex")

