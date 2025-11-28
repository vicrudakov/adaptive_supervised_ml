library(dplyr)
library(tidyr)
library(tidyverse)
library(ggplot2)

# Plot saving function
{
  save_plot <- function(results_df, dataset, total_size, peft_method, strategy, baseline, y_lims, file_path) {
    plt <- results_df %>% 
      filter(Dataset == dataset & `Total size` == total_size & `PEFT method` == peft_method & Strategy == strategy) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      mutate(Lambda = paste0("λ = ", Lambda)) %>%
      mutate(Lambda = factor(Lambda, levels = c("λ = 10", "λ = 50", "λ = 100", "λ = 500"))) %>%
      ggplot(aes(x = factor(Iteration), y = `F1-score`)) +
      geom_boxplot(outliers = TRUE) +
      geom_hline(yintercept = baseline, color = "black", linewidth = 0.8) +
      facet_wrap(~ Lambda, ncol = 2, nrow = 2, as.table = TRUE) +
      scale_x_discrete(breaks = 0:10, labels = 0:10) +
      scale_y_continuous(breaks = seq(ceiling(y_lims[1] / 0.05) * 0.05, 
                                      floor(y_lims[2] / 0.05) * 0.05, by = 0.05), 
                         labels = function(x) format(x, nsmall = 2),
                         limits = y_lims) +
      labs(x = "Iteration", y = "F1-score") +
      theme_minimal() +
      theme(text = element_text(size = 16))
    ggsave(file_path, plt, width = 9, height = 7, units = "in", device = cairo_pdf)
  }
}

### Sensation: Results Visualization
{
  # Continual active learning results and baselines
  {
    results_df_sensation <- read.csv("results/sensation/continual_active_learning/results_f1.csv") %>%
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
    baselines_df_sensation <- read.csv("results/sensation/baseline/baseline_f1.csv")
  }
  
  # Plot limits
  {
    results_df_sensation %>% 
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      select(`F1-score`) %>%
      as.vector() %>%
      range()
    y_lims_sensation <- c(0.5, 0.75)
  }
  
  ## Results: sensation - 1000 - pfeiffer
  {
    (baseline_sensation_1000_pfeiffer <- 
       mean(baselines_df_sensation[baselines_df_sensation$experiment == 'sensation_1000_pfeiffer_baseline', 'f1']))
    save_plot(results_df_sensation, "sensation", 1000, "Sequential Adapter", "Random", baseline_sensation_1000_pfeiffer, 
              y_lims_sensation, "plots/sensation/plot_sensation_1000_pfeiffer_random.pdf")
    save_plot(results_df_sensation, "sensation", 1000, "Sequential Adapter", "Margin", baseline_sensation_1000_pfeiffer, 
              y_lims_sensation, "plots/sensation/plot_sensation_1000_pfeiffer_margin.pdf")
    save_plot(results_df_sensation, "sensation", 1000, "Sequential Adapter", "k-means", baseline_sensation_1000_pfeiffer, 
              y_lims_sensation, "plots/sensation/plot_sensation_1000_pfeiffer_kmeans.pdf")
  }
  
  ## Results: sensation - 1000 - pfeifferinv
  {
    (baseline_sensation_1000_pfeifferinv <- 
       mean(baselines_df_sensation[baselines_df_sensation$experiment == 'sensation_1000_pfeifferinv_baseline', 'f1']))
    save_plot(results_df_sensation, "sensation", 1000, "Sequential Invertible Adapter", "Random", baseline_sensation_1000_pfeifferinv, 
              y_lims_sensation, "plots/sensation/plot_sensation_1000_pfeifferinv_random.pdf")
    save_plot(results_df_sensation, "sensation", 1000, "Sequential Invertible Adapter", "Margin", baseline_sensation_1000_pfeifferinv, 
              y_lims_sensation, "plots/sensation/plot_sensation_1000_pfeifferinv_margin.pdf")
    save_plot(results_df_sensation, "sensation", 1000, "Sequential Invertible Adapter", "k-means", baseline_sensation_1000_pfeifferinv, 
              y_lims_sensation, "plots/sensation/plot_sensation_1000_pfeifferinv_kmeans.pdf")
  }
  
  ## Results: sensation - 1000 - lora
  {
    (baseline_sensation_1000_lora <- 
       mean(baselines_df_sensation[baselines_df_sensation$experiment == 'sensation_1000_lora_baseline', 'f1']))
    save_plot(results_df_sensation, "sensation", 1000, "LoRA", "Random", baseline_sensation_1000_lora, 
              y_lims_sensation, "plots/sensation/plot_sensation_1000_lora_random.pdf")
    save_plot(results_df_sensation, "sensation", 1000, "LoRA", "Margin", baseline_sensation_1000_lora, 
              y_lims_sensation, "plots/sensation/plot_sensation_1000_lora_margin.pdf")
    save_plot(results_df_sensation, "sensation", 1000, "LoRA", "k-means", baseline_sensation_1000_lora, 
              y_lims_sensation, "plots/sensation/plot_sensation_1000_lora_kmeans.pdf")
  }
  
  ## Results: sensation - 2000 - pfeiffer
  {
    (baseline_sensation_2000_pfeiffer <- 
       mean(baselines_df_sensation[baselines_df_sensation$experiment == 'sensation_2000_pfeiffer_baseline', 'f1']))
    save_plot(results_df_sensation, "sensation", 2000, "Sequential Adapter", "Random", baseline_sensation_2000_pfeiffer, 
              y_lims_sensation, "plots/sensation/plot_sensation_2000_pfeiffer_random.pdf")
    save_plot(results_df_sensation, "sensation", 2000, "Sequential Adapter", "Margin", baseline_sensation_2000_pfeiffer, 
              y_lims_sensation, "plots/sensation/plot_sensation_2000_pfeiffer_margin.pdf")
    save_plot(results_df_sensation, "sensation", 2000, "Sequential Adapter", "k-means", baseline_sensation_2000_pfeiffer, 
              y_lims_sensation, "plots/sensation/plot_sensation_2000_pfeiffer_kmeans.pdf")
  }
  
  ## Results: sensation - 2000 - pfeifferinv
  {
    (baseline_sensation_2000_pfeifferinv <- 
       mean(baselines_df_sensation[baselines_df_sensation$experiment == 'sensation_2000_pfeifferinv_baseline', 'f1']))
    save_plot(results_df_sensation, "sensation", 2000, "Sequential Invertible Adapter", "Random", baseline_sensation_2000_pfeifferinv, 
              y_lims_sensation, "plots/sensation/plot_sensation_2000_pfeifferinv_random.pdf")
    save_plot(results_df_sensation, "sensation", 2000, "Sequential Invertible Adapter", "Margin", baseline_sensation_2000_pfeifferinv, 
              y_lims_sensation, "plots/sensation/plot_sensation_2000_pfeifferinv_margin.pdf")
    save_plot(results_df_sensation, "sensation", 2000, "Sequential Invertible Adapter", "k-means", baseline_sensation_2000_pfeifferinv, 
              y_lims_sensation, "plots/sensation/plot_sensation_2000_pfeifferinv_kmeans.pdf")
  }
  
  ## Results: sensation - 2000 - lora
  {
    (baseline_sensation_2000_lora <- 
       mean(baselines_df_sensation[baselines_df_sensation$experiment == 'sensation_2000_lora_baseline', 'f1']))
    save_plot(results_df_sensation, "sensation", 2000, "LoRA", "Random", baseline_sensation_2000_lora, 
              y_lims_sensation, "plots/sensation/plot_sensation_2000_lora_random.pdf")
    save_plot(results_df_sensation, "sensation", 2000, "LoRA", "Margin", baseline_sensation_2000_lora, 
              y_lims_sensation, "plots/sensation/plot_sensation_2000_lora_margin.pdf")
    save_plot(results_df_sensation, "sensation", 2000, "LoRA", "k-means", baseline_sensation_2000_lora, 
              y_lims_sensation, "plots/sensation/plot_sensation_2000_lora_kmeans.pdf")
  }
}

### AG News: Results Visualization
{
  # Continual active learning results and baselines
  {
    results_df_agnews <- read.csv("results/agnews/continual_active_learning/results_f1.csv") %>%
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
    baselines_df_agnews <- read.csv("results/agnews/baseline/baseline_f1.csv")
  }
  
  # Plot limits
  {
    results_df_agnews %>% 
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      select(`F1-score`) %>%
      as.vector() %>%
      range()
    y_lims_agnews <- c(0.9, 1)
  }
  
  ## Results: agnews - 1000 - pfeiffer
  {
    (baseline_agnews_1000_pfeiffer <- 
       mean(baselines_df_agnews[baselines_df_agnews$experiment == 'agnews_1000_pfeiffer_baseline', 'f1']))
    save_plot(results_df_agnews, "agnews", 1000, "Sequential Adapter", "Random", baseline_agnews_1000_pfeiffer, 
              y_lims_agnews, "plots/agnews/plot_agnews_1000_pfeiffer_random.pdf")
    save_plot(results_df_agnews, "agnews", 1000, "Sequential Adapter", "Margin", baseline_agnews_1000_pfeiffer, 
              y_lims_agnews, "plots/agnews/plot_agnews_1000_pfeiffer_margin.pdf")
    save_plot(results_df_agnews, "agnews", 1000, "Sequential Adapter", "k-means", baseline_agnews_1000_pfeiffer, 
              y_lims_agnews, "plots/agnews/plot_agnews_1000_pfeiffer_kmeans.pdf")
  }
  
  ## Results: agnews - 1000 - pfeifferinv
  {
    (baseline_agnews_1000_pfeifferinv <- 
       mean(baselines_df_agnews[baselines_df_agnews$experiment == 'agnews_1000_pfeifferinv_baseline', 'f1']))
    save_plot(results_df_agnews, "agnews", 1000, "Sequential Invertible Adapter", "Random", baseline_agnews_1000_pfeifferinv, 
              y_lims_agnews, "plots/agnews/plot_agnews_1000_pfeifferinv_random.pdf")
    save_plot(results_df_agnews, "agnews", 1000, "Sequential Invertible Adapter", "Margin", baseline_agnews_1000_pfeifferinv, 
              y_lims_agnews, "plots/agnews/plot_agnews_1000_pfeifferinv_margin.pdf")
    save_plot(results_df_agnews, "agnews", 1000, "Sequential Invertible Adapter", "k-means", baseline_agnews_1000_pfeifferinv, 
              y_lims_agnews, "plots/agnews/plot_agnews_1000_pfeifferinv_kmeans.pdf")
  }
  
  ## Results: agnews - 1000 - lora
  {
    (baseline_agnews_1000_lora <- 
       mean(baselines_df_agnews[baselines_df_agnews$experiment == 'agnews_1000_lora_baseline', 'f1']))
    save_plot(results_df_agnews, "agnews", 1000, "LoRA", "Random", baseline_agnews_1000_lora, 
              y_lims_agnews, "plots/agnews/plot_agnews_1000_lora_random.pdf")
    save_plot(results_df_agnews, "agnews", 1000, "LoRA", "Margin", baseline_agnews_1000_lora, 
              y_lims_agnews, "plots/agnews/plot_agnews_1000_lora_margin.pdf")
    save_plot(results_df_agnews, "agnews", 1000, "LoRA", "k-means", baseline_agnews_1000_lora, 
              y_lims_agnews, "plots/agnews/plot_agnews_1000_lora_kmeans.pdf")
  }
  
  ## Results: agnews - 2000 - pfeiffer
  {
    (baseline_agnews_2000_pfeiffer <-
       mean(baselines_df_agnews[baselines_df_agnews$experiment == 'agnews_2000_pfeiffer_baseline', 'f1']))
    save_plot(results_df_agnews, "agnews", 2000, "Sequential Adapter", "Random", baseline_agnews_2000_pfeiffer,
              y_lims_agnews, "plots/agnews/plot_agnews_2000_pfeiffer_random.pdf")
    save_plot(results_df_agnews, "agnews", 2000, "Sequential Adapter", "Margin", baseline_agnews_2000_pfeiffer,
              y_lims_agnews, "plots/agnews/plot_agnews_2000_pfeiffer_margin.pdf")
    save_plot(results_df_agnews, "agnews", 2000, "Sequential Adapter", "k-means", baseline_agnews_2000_pfeiffer,
              y_lims_agnews, "plots/agnews/plot_agnews_2000_pfeiffer_kmeans.pdf")
  }

  ## Results: agnews - 2000 - pfeifferinv
  {
    (baseline_agnews_2000_pfeifferinv <-
       mean(baselines_df_agnews[baselines_df_agnews$experiment == 'agnews_2000_pfeifferinv_baseline', 'f1']))
    save_plot(results_df_agnews, "agnews", 2000, "Sequential Invertible Adapter", "Random", baseline_agnews_2000_pfeifferinv,
              y_lims_agnews, "plots/agnews/plot_agnews_2000_pfeifferinv_random.pdf")
    save_plot(results_df_agnews, "agnews", 2000, "Sequential Invertible Adapter", "Margin", baseline_agnews_2000_pfeifferinv,
              y_lims_agnews, "plots/agnews/plot_agnews_2000_pfeifferinv_margin.pdf")
    save_plot(results_df_agnews, "agnews", 2000, "Sequential Invertible Adapter", "k-means", baseline_agnews_2000_pfeifferinv,
              y_lims_agnews, "plots/agnews/plot_agnews_2000_pfeifferinv_kmeans.pdf")
  }

  ## Results: agnews - 2000 - lora
  {
    (baseline_agnews_2000_lora <-
       mean(baselines_df_agnews[baselines_df_agnews$experiment == 'agnews_2000_lora_baseline', 'f1']))
    save_plot(results_df_agnews, "agnews", 2000, "LoRA", "Random", baseline_agnews_2000_lora,
              y_lims_agnews, "plots/agnews/plot_agnews_2000_lora_random.pdf")
    save_plot(results_df_agnews, "agnews", 2000, "LoRA", "Margin", baseline_agnews_2000_lora,
              y_lims_agnews, "plots/agnews/plot_agnews_2000_lora_margin.pdf")
    save_plot(results_df_agnews, "agnews", 2000, "LoRA", "k-means", baseline_agnews_2000_lora,
              y_lims_agnews, "plots/agnews/plot_agnews_2000_lora_kmeans.pdf")
  }
}

### Yahoo: Results Visualization
{
  # Continual active learning results and baselines
  {
    results_df_yahoo <- read.csv("results/yahoo/continual_active_learning/results_f1.csv") %>%
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
    baselines_df_yahoo <- read.csv("results/yahoo/baseline/baseline_f1.csv")
  }
  
  # Plot limits
  {
    results_df_yahoo %>% 
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      select(`F1-score`) %>%
      as.vector() %>%
      range()
    y_lims_yahoo <- c(0.9, 1)
  }
  
  ## Results: yahoo - 1000 - pfeiffer
  {
    (baseline_yahoo_1000_pfeiffer <- 
       mean(baselines_df_yahoo[baselines_df_yahoo$experiment == 'yahoo_1000_pfeiffer_baseline', 'f1']))
    save_plot(results_df_yahoo, "yahoo", 1000, "Sequential Adapter", "Random", baseline_yahoo_1000_pfeiffer, 
              y_lims_yahoo, "plots/yahoo/plot_yahoo_1000_pfeiffer_random.pdf")
    save_plot(results_df_yahoo, "yahoo", 1000, "Sequential Adapter", "Margin", baseline_yahoo_1000_pfeiffer, 
              y_lims_yahoo, "plots/yahoo/plot_yahoo_1000_pfeiffer_margin.pdf")
    save_plot(results_df_yahoo, "yahoo", 1000, "Sequential Adapter", "k-means", baseline_yahoo_1000_pfeiffer, 
              y_lims_yahoo, "plots/yahoo/plot_yahoo_1000_pfeiffer_kmeans.pdf")
  }
  
  ## Results: yahoo - 1000 - pfeifferinv
  {
    (baseline_yahoo_1000_pfeifferinv <- 
       mean(baselines_df_yahoo[baselines_df_yahoo$experiment == 'yahoo_1000_pfeifferinv_baseline', 'f1']))
    save_plot(results_df_yahoo, "yahoo", 1000, "Sequential Invertible Adapter", "Random", baseline_yahoo_1000_pfeifferinv, 
              y_lims_yahoo, "plots/yahoo/plot_yahoo_1000_pfeifferinv_random.pdf")
    save_plot(results_df_yahoo, "yahoo", 1000, "Sequential Invertible Adapter", "Margin", baseline_yahoo_1000_pfeifferinv, 
              y_lims_yahoo, "plots/yahoo/plot_yahoo_1000_pfeifferinv_margin.pdf")
    save_plot(results_df_yahoo, "yahoo", 1000, "Sequential Invertible Adapter", "k-means", baseline_yahoo_1000_pfeifferinv, 
              y_lims_yahoo, "plots/yahoo/plot_yahoo_1000_pfeifferinv_kmeans.pdf")
  }
  
  ## Results: yahoo - 1000 - lora
  {
    (baseline_yahoo_1000_lora <- 
       mean(baselines_df_yahoo[baselines_df_yahoo$experiment == 'yahoo_1000_lora_baseline', 'f1']))
    save_plot(results_df_yahoo, "yahoo", 1000, "LoRA", "Random", baseline_yahoo_1000_lora, 
              y_lims_yahoo, "plots/yahoo/plot_yahoo_1000_lora_random.pdf")
    save_plot(results_df_yahoo, "yahoo", 1000, "LoRA", "Margin", baseline_yahoo_1000_lora, 
              y_lims_yahoo, "plots/yahoo/plot_yahoo_1000_lora_margin.pdf")
    save_plot(results_df_yahoo, "yahoo", 1000, "LoRA", "k-means", baseline_yahoo_1000_lora, 
              y_lims_yahoo, "plots/yahoo/plot_yahoo_1000_lora_kmeans.pdf")
  }
  
  ## Results: yahoo - 2000 - pfeiffer
  {
    (baseline_yahoo_2000_pfeiffer <-
       mean(baselines_df_yahoo[baselines_df_yahoo$experiment == 'yahoo_2000_pfeiffer_baseline', 'f1']))
    save_plot(results_df_yahoo, "yahoo", 2000, "Sequential Adapter", "Random", baseline_yahoo_2000_pfeiffer,
              y_lims_yahoo, "plots/yahoo/plot_yahoo_2000_pfeiffer_random.pdf")
    save_plot(results_df_yahoo, "yahoo", 2000, "Sequential Adapter", "Margin", baseline_yahoo_2000_pfeiffer,
              y_lims_yahoo, "plots/yahoo/plot_yahoo_2000_pfeiffer_margin.pdf")
    save_plot(results_df_yahoo, "yahoo", 2000, "Sequential Adapter", "k-means", baseline_yahoo_2000_pfeiffer,
              y_lims_yahoo, "plots/yahoo/plot_yahoo_2000_pfeiffer_kmeans.pdf")
  }
  
  ## Results: yahoo - 2000 - pfeifferinv
  {
    (baseline_yahoo_2000_pfeifferinv <-
       mean(baselines_df_yahoo[baselines_df_yahoo$experiment == 'yahoo_2000_pfeifferinv_baseline', 'f1']))
    save_plot(results_df_yahoo, "yahoo", 2000, "Sequential Invertible Adapter", "Random", baseline_yahoo_2000_pfeifferinv,
              y_lims_yahoo, "plots/yahoo/plot_yahoo_2000_pfeifferinv_random.pdf")
    save_plot(results_df_yahoo, "yahoo", 2000, "Sequential Invertible Adapter", "Margin", baseline_yahoo_2000_pfeifferinv,
              y_lims_yahoo, "plots/yahoo/plot_yahoo_2000_pfeifferinv_margin.pdf")
    save_plot(results_df_yahoo, "yahoo", 2000, "Sequential Invertible Adapter", "k-means", baseline_yahoo_2000_pfeifferinv,
              y_lims_yahoo, "plots/yahoo/plot_yahoo_2000_pfeifferinv_kmeans.pdf")
  }
  
  ## Results: yahoo - 2000 - lora
  {
    (baseline_yahoo_2000_lora <-
       mean(baselines_df_yahoo[baselines_df_yahoo$experiment == 'yahoo_2000_lora_baseline', 'f1']))
    save_plot(results_df_yahoo, "yahoo", 2000, "LoRA", "Random", baseline_yahoo_2000_lora,
              y_lims_yahoo, "plots/yahoo/plot_yahoo_2000_lora_random.pdf")
    save_plot(results_df_yahoo, "yahoo", 2000, "LoRA", "Margin", baseline_yahoo_2000_lora,
              y_lims_yahoo, "plots/yahoo/plot_yahoo_2000_lora_margin.pdf")
    save_plot(results_df_yahoo, "yahoo", 2000, "LoRA", "k-means", baseline_yahoo_2000_lora,
              y_lims_yahoo, "plots/yahoo/plot_yahoo_2000_lora_kmeans.pdf")
  }
}

