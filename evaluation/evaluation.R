library(dplyr)
library(tidyr)
library(tidyverse)
library(ggplot2)
library(knitr)
library(kableExtra)

### Data Description
{
  data <- read.csv("data/sensation_all_columns.csv")
  
  # General variable characteristics
  nrow(data) # 4851
  length(unique(data$annotator)) # 25
  range(as.Date(paste(data$year, data$month, data$day, sep = "-"))) # "1999-06-28" "2016-03-17"

  # Missing values
  sapply(colnames(data), function(var_name){
    data_vector <- data[[var_name]]
    n_missing <- sum(is.na(data_vector))
    paste0(var_name, ": ", n_missing, ", ", round(n_missing / length(data_vector), 4))
  }, USE.NAMES = FALSE) # "label: 405, 0.0835"
  sum(is.na(data$label) & data$answer == "ignore") # 373
  View(data[is.na(data$label) & data$answer != "ignore", ])

  # Variable document
  length(unique(data$document)) # 498
  plot_doc <- ggplot(as.data.frame(table(table(data$document))), 
         aes(x = Var1, y = Freq)) +
    geom_bar(stat = "identity", fill = "grey") +
    labs(x = "Number of sentences in the document", 
         y = "Number of documents") +
    theme_minimal() +
    theme(text = element_text(size = 16))
  ggsave("plots/plot_document.pdf", plot_doc, width = 9, height = 5.5, units = "in")
  
  # Variable year
  plot_year <- ggplot(as.data.frame(table(unique(data[, c("document", "year")])$year)), 
         aes(x = Var1, y = Freq)) +
    geom_bar(stat = "identity", fill = "grey") +
    labs(x = "Year", 
         y = "Number of documents") +
    theme_minimal() +
    theme(text = element_text(size = 16))
  ggsave("plots/plot_year.pdf", plot_year, width = 9, height = 5.5, units = "in")
  
  # Variable annotator
  plot_annotator <- ggplot(as.data.frame(table(data$annotator)), 
         aes(x = Var1, y = Freq)) +
    geom_bar(stat = "identity", fill = "grey") +
    scale_x_discrete(labels = 1:25) +
    labs(x = "Annotator", 
         y = "Number of sentences") +
    theme_minimal() +
    theme(text = element_text(size = 16))
  ggsave("plots/plot_annotator.pdf", plot_annotator, width = 9, height = 5.5, units = "in")
  
  # Variable label
  sum(table(data$label)) # 4446
  table(data$label, useNA = "always")
  
  # Variable answer
  View(data[is.na(data$label) & data$answer == "ignore", ]) # 373
  View(data[is.na(data$label) & data$answer != "ignore", ]) # 32
  View(data[!is.na(data$label) & data$answer == "ignore", ]) # 16
  View(data[!is.na(data$label) & data$answer != "ignore", ]) # 4430
}

### Results Tables Creation
{
  ## Baseline Results
  {
    read.csv("results/baseline/baseline_acc.csv") %>%
      left_join(read.csv("results/baseline/baseline_prec.csv"), by = "experiment") %>%
      left_join(read.csv("results/baseline/baseline_rec.csv"), by = "experiment") %>%
      left_join(read.csv("results/baseline/baseline_f1.csv"), by = "experiment") %>%
      separate(experiment, into = c("Total size", "PEFT method", "tmp"), sep = "_") %>%
      select(-tmp) %>%
      mutate(`PEFT method` = 
               case_when(`PEFT method` == "lora" ~ "LoRA",
                         `PEFT method` == "pfeiffer" ~ "Sequential Adapter",
                         `PEFT method` == "pfeifferinv" ~ "Sequential Invertible Adapter")) %>%
      rename("Accuracy" = "acc", "Precision" = "prec", "Recall" = "rec", "F1-score" = "f1") %>%
      print() %>%
      kable(format = "latex", booktabs = TRUE) %>%
      gsub("\\\\addlinespace\n", "", .) %>%
      collapse_rows(columns = 1) %>%
      gsub("\\\\cmidrule\\{2-6\\}\n", "", .) %>%
      gsub("\\[1\\\\dimexpr\\\\aboverulesep\\+\\\\belowrulesep\\+\\\\cmidrulewidth\\]", "", .) %>%
      gsub("\\\\raggedright\\\\arraybackslash ", "", .)
  }
  
  ## Viewing Continual Active Learning Results
  {
    # A function to view the results of application of continual active learning
    # (accuracy, precision, recall, F1-score) as a dataframe from a corresponding 
    # csv-file.
    #
    # Input:
    #   path - path to a file with results
    # 
    # Output:
    #   None. Invokes a data viewer for corresponding dataframe.
    view_results_table <- function(path) {
      read.csv(path) %>%
        separate(experiment, into = c("Total size", "PEFT method", "Strategy", "Lambda"), sep = "_") %>%
        mutate(`PEFT method` = 
                 case_when(`PEFT method` == "lora" ~ "LoRA",
                           `PEFT method` == "pfeiffer" ~ "Sequential Adapter",
                           `PEFT method` == "pfeifferinv" ~ "Sequential Invertible Adapter")) %>%
        mutate(Strategy = 
                 case_when(Strategy == "diversity" ~ "k-means",
                           Strategy == "random" ~ "Random",
                           Strategy == "uncertainty" ~ "Margin")) %>%
        View(title = str_match(path, "results_([a-zA-Z1-10]{2,4})\\.csv")[,2])
    }
    
    view_results_table("results/continual_active_learning/results_acc.csv")
    view_results_table("results/continual_active_learning/results_prec.csv")
    view_results_table("results/continual_active_learning/results_rec.csv")
    view_results_table("results/continual_active_learning/results_f1.csv")
  }

  ## Creating Tables of Continual Active Learning Results
  {
    # A function to create a LaTeX table of the results of application of 
    # continual active learning (accuracy, precision, recall, F1-score)
    # for specific configurations based on number of observations from
    # a corresponding csv-file 
    #
    # Input:
    #   path - path to a file with results
    #   data_size - number of observations in configurations to print results for
    # 
    # Output:
    #   A table with the corresponding results (knitr_kable object).
    print_results_table <- function(path, data_size) {
      results_tab <- read.csv(path) %>%
        separate(experiment, into = c("Total size", "PEFT method", "Strategy", "Lambda"), sep = "_") %>%
        mutate(`PEFT method` = 
                 case_when(`PEFT method` == "lora" ~ "LoRA",
                           `PEFT method` == "pfeiffer" ~ "Sequential Adapter",
                           `PEFT method` == "pfeifferinv" ~ "Sequential Invertible Adapter")) %>%
        mutate(Strategy = 
                 case_when(Strategy == "diversity" ~ "k-means",
                           Strategy == "random" ~ "Random",
                           Strategy == "uncertainty" ~ "Margin")) %>%
        mutate(Lambda = as.numeric(Lambda)) %>% 
        filter(`Total size` == data_size) %>%
        select(-`Total size`) %>%
        rename_with(.fn = ~ str_replace(.x, "^[^_]+_(\\d+)$", "Iter. \\1"), 
                    .cols = matches("^[^_]+_\\d+$")) %>%
        kable(format = "latex", booktabs = TRUE) %>%
        gsub("\\\\addlinespace\n", "", .) %>%
        collapse_rows(columns = c(1, 2)) %>%
        gsub("\\\\cmidrule\\{3-14\\}\n", "", .) %>%
        gsub("Lambda", "$\\\\lambda$", .) %>%
        gsub("\\[1\\\\dimexpr\\\\aboverulesep\\+\\\\belowrulesep\\+\\\\cmidrulewidth\\]", "", .) %>%
        gsub("\\\\raggedright\\\\arraybackslash ", "", .) %>%
        gsub("PEFT method", 
             "\\\\parbox\\{1.5cm\\}\\{\\\\vspace{2pt}PEFT\\\\\\\\method\\\\vspace{2pt}\\}", .) %>%
        gsub("Sequential Adapter", 
             "\\\\parbox\\{1.5cm\\}\\{Sequential\\\\\\\\Adapter\\}", .) %>%
        gsub("Sequential Invertible Adapter", 
             "\\\\parbox\\{1.5cm\\}\\{Sequential\\\\\\\\Invertible\\\\\\\\Adapter\\}", .)
      return(results_tab)
    }
    
    print_results_table("results/continual_active_learning/results_acc.csv", data_size = 1000)
    print_results_table("results/continual_active_learning/results_acc.csv", data_size = 2000)
    
    print_results_table("results/continual_active_learning/results_prec.csv", data_size = 1000)
    print_results_table("results/continual_active_learning/results_prec.csv", data_size = 2000)
    
    print_results_table("results/continual_active_learning/results_rec.csv", data_size = 1000)
    print_results_table("results/continual_active_learning/results_rec.csv", data_size = 2000)
    
    print_results_table("results/continual_active_learning/results_f1.csv", data_size = 1000)
    print_results_table("results/continual_active_learning/results_f1.csv", data_size = 2000)
  }
}

### Results Visualization
{
  # Continual active learning results and baselines
  {
    results_df <- read.csv("results/continual_active_learning/results_f1.csv") %>%
      separate(experiment, into = c("Total size", "PEFT method", "Strategy", "Lambda"), sep = "_") %>%
      mutate(`PEFT method` = 
               case_when(`PEFT method` == "lora" ~ "LoRA",
                         `PEFT method` == "pfeiffer" ~ "Sequential Adapter",
                         `PEFT method` == "pfeifferinv" ~ "Sequential Invertible Adapter")) %>%
      mutate(Strategy = 
               case_when(Strategy == "diversity" ~ "k-means",
                         Strategy == "random" ~ "Random",
                         Strategy == "uncertainty" ~ "Margin"))
    baselines_df <- read.csv("results/baseline/baseline_f1.csv")
  }
  
  # Plot limits
  {
    results_df %>% 
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      select(`F1-score`) %>%
      as.vector() %>%
      range() # 0.497 0.727
    y_lims <- c(0.49, 0.75)
  }
  
  ## Plot creation: 1000 - pfeiffer
  {
    baselines_df[baselines_df$experiment == '1000_pfeiffer_baseline', 'f1'] # 0.690
    results_df %>% 
      filter(`PEFT method` == "Sequential Adapter" & `Total size` == 1000) %>%
      view()
    results_df %>% 
      filter(`PEFT method` == "Sequential Adapter" & `Total size` == 1000) %>%
      select(where(is.numeric)) %>%
      unlist() %>%
      max()
    
    plot_1000_pfeiffer <- results_df %>% 
      filter(`PEFT method` == "Sequential Adapter" & `Total size` == 1000) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      mutate(Lambda = paste0("λ = ", Lambda)) %>%
      mutate(Lambda = factor(Lambda, levels = c("λ = 10", "λ = 50", "λ = 100", "λ = 500"))) %>%
      ggplot(aes(x = Iteration, y = `F1-score`, 
                 color = factor(Strategy, levels = c("Random", "Margin", "k-means")))) +
      geom_hline(yintercept = baselines_df[baselines_df$experiment == '1000_pfeiffer_baseline', 'f1'],
                 color = "black", size = 1) +
      facet_wrap(~ Lambda, ncol = 2, nrow = 2, as.table = TRUE) +
      geom_line(size = 1.2) +
      geom_point(size = 2) +
      scale_x_continuous(breaks = 0:10, labels = 0:10) +
      scale_y_continuous(breaks = seq(ceiling(y_lims[1] / 0.05) * 0.05, 
                                      floor(y_lims[2] / 0.05) * 0.05, by = 0.05), 
                         labels = function(x) format(x, nsmall = 2),
                         limits = y_lims) +
      labs(x = "Iteration", y = "F1-score", color = "Query\nstrategy") +
      theme_minimal() +
      theme(text = element_text(size = 16))
    ggsave("plots/plot_1000_pfeiffer.pdf", plot_1000_pfeiffer, 
           width = 9, height = 7, units = "in", device = cairo_pdf)
  }
  
  ## Plot creation: 1000 - pfeifferinv
  {
    baselines_df[baselines_df$experiment == '1000_pfeifferinv_baseline', 'f1'] # 0.683
    results_df %>% 
      filter(`PEFT method` == "Sequential Invertible Adapter" & `Total size` == 1000) %>%
      view()
    results_df %>% 
      filter(`PEFT method` == "Sequential Invertible Adapter" & `Total size` == 1000) %>%
      select(where(is.numeric)) %>%
      unlist() %>%
      max()
    
    plot_1000_pfeifferinv <- results_df %>% 
      filter(`PEFT method` == "Sequential Invertible Adapter" & `Total size` == 1000) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      mutate(Lambda = paste0("λ = ", Lambda)) %>%
      mutate(Lambda = factor(Lambda, levels = c("λ = 10", "λ = 50", "λ = 100", "λ = 500"))) %>%
      ggplot(aes(x = Iteration, y = `F1-score`, 
                 color = factor(Strategy, levels = c("Random", "Margin", "k-means")))) +
      geom_hline(yintercept = baselines_df[baselines_df$experiment == '1000_pfeifferinv_baseline', 'f1'],
                 color = "black", size = 1) +
      facet_wrap(~ Lambda, ncol = 2, nrow = 2) +
      geom_line(size = 1.2) +
      geom_point(size = 2) +
      scale_x_continuous(breaks = 0:10, labels = 0:10) +
      scale_y_continuous(breaks = seq(ceiling(y_lims[1] / 0.05) * 0.05, 
                                      floor(y_lims[2] / 0.05) * 0.05, by = 0.05), 
                         labels = function(x) format(x, nsmall = 2),
                         limits = y_lims) +
      labs(x = "Iteration", y = "F1-score", color = "Query\nstrategy") +
      theme_minimal() +
      theme(text = element_text(size = 16))
    ggsave("plots/plot_1000_pfeifferinv.pdf", plot_1000_pfeifferinv, 
           width = 9, height = 7, units = "in", device = cairo_pdf)
    
  }
  
  ## Plot creation: 1000 - lora
  {
    baselines_df[baselines_df$experiment == '1000_lora_baseline', 'f1'] # 0.674
    results_df %>% 
      filter(`PEFT method` == "LoRA" & `Total size` == 1000) %>%
      view()
    results_df %>% 
      filter(`PEFT method` == "LoRA" & `Total size` == 1000) %>%
      select(where(is.numeric)) %>%
      unlist() %>%
      max()
    
    plot_1000_lora <- results_df %>% 
      filter(`PEFT method` == "LoRA" & `Total size` == 1000) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      mutate(Lambda = paste0("λ = ", Lambda)) %>%
      mutate(Lambda = factor(Lambda, levels = c("λ = 10", "λ = 50", "λ = 100", "λ = 500"))) %>%
      ggplot(aes(x = Iteration, y = `F1-score`, 
                 color = factor(Strategy, levels = c("Random", "Margin", "k-means")))) +
      geom_hline(yintercept = baselines_df[baselines_df$experiment == '1000_lora_baseline', 'f1'],
                 color = "black", size = 1) +
      facet_wrap(~ Lambda, ncol = 2, nrow = 2) +
      geom_line(size = 1.2) +
      geom_point(size = 2) +
      scale_x_continuous(breaks = 0:10, labels = 0:10) +
      scale_y_continuous(breaks = seq(ceiling(y_lims[1] / 0.05) * 0.05, 
                                      floor(y_lims[2] / 0.05) * 0.05, by = 0.05), 
                         labels = function(x) format(x, nsmall = 2),
                         limits = y_lims) +
      labs(x = "Iteration", y = "F1-score", color = "Query\nstrategy") +
      theme_minimal() +
      theme(text = element_text(size = 16))
    ggsave("plots/plot_1000_lora.pdf", plot_1000_lora, 
           width = 9, height = 7, units = "in", device = cairo_pdf)
  }
  
  ## Plot creation: 2000 - pfeiffer
  {
    baselines_df[baselines_df$experiment == '2000_pfeiffer_baseline', 'f1'] # 0.674
    results_df %>% 
      filter(`PEFT method` == "Sequential Adapter" & `Total size` == 2000) %>%
      view()
    results_df %>% 
      filter(`PEFT method` == "Sequential Adapter" & `Total size` == 2000) %>%
      select(where(is.numeric)) %>%
      unlist() %>%
      max()
    
    plot_2000_pfeiffer <- results_df %>% 
      filter(`PEFT method` == "Sequential Adapter" & `Total size` == 2000) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      mutate(Lambda = paste0("λ = ", Lambda)) %>%
      mutate(Lambda = factor(Lambda, levels = c("λ = 10", "λ = 50", "λ = 100", "λ = 500"))) %>%
      ggplot(aes(x = Iteration, y = `F1-score`, 
                 color = factor(Strategy, levels = c("Random", "Margin", "k-means")))) +
      geom_hline(yintercept = baselines_df[baselines_df$experiment == '2000_pfeiffer_baseline', 'f1'],
                 color = "black", size = 1) +
      facet_wrap(~ Lambda, ncol = 2, nrow = 2) +
      geom_line(size = 1.2) +
      geom_point(size = 2) +
      scale_x_continuous(breaks = 0:10, labels = 0:10) +
      scale_y_continuous(breaks = seq(ceiling(y_lims[1] / 0.05) * 0.05, 
                                      floor(y_lims[2] / 0.05) * 0.05, by = 0.05), 
                         labels = function(x) format(x, nsmall = 2),
                         limits = y_lims) +
      labs(x = "Iteration", y = "F1-score", color = "Query\nstrategy") +
      theme_minimal() +
      theme(text = element_text(size = 16))
    ggsave("plots/plot_2000_pfeiffer.pdf", plot_2000_pfeiffer, 
           width = 9, height = 7, units = "in", device = cairo_pdf)
  }
  
  ## Plot creation: 2000 - pfeifferinv
  {
    baselines_df[baselines_df$experiment == '2000_pfeifferinv_baseline', 'f1'] # 0.671
    results_df %>% 
      filter(`PEFT method` == "Sequential Invertible Adapter" & `Total size` == 2000) %>%
      view()
    results_df %>% 
      filter(`PEFT method` == "Sequential Invertible Adapter" & `Total size` == 2000) %>%
      select(where(is.numeric)) %>%
      unlist() %>%
      max()
    
    plot_2000_pfeifferinv <- results_df %>% 
      filter(`PEFT method` == "Sequential Invertible Adapter" & `Total size` == 2000) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      mutate(Lambda = paste0("λ = ", Lambda)) %>%
      mutate(Lambda = factor(Lambda, levels = c("λ = 10", "λ = 50", "λ = 100", "λ = 500"))) %>%
      ggplot(aes(x = Iteration, y = `F1-score`, 
                 color = factor(Strategy, levels = c("Random", "Margin", "k-means")))) +
      geom_hline(yintercept = baselines_df[baselines_df$experiment == '2000_pfeifferinv_baseline', 'f1'],
                 color = "black", size = 1) +
      facet_wrap(~ Lambda, ncol = 2, nrow = 2) +
      geom_line(size = 1.2) +
      geom_point(size = 2) +
      scale_x_continuous(breaks = 0:10, labels = 0:10) +
      scale_y_continuous(breaks = seq(ceiling(y_lims[1] / 0.05) * 0.05, 
                                      floor(y_lims[2] / 0.05) * 0.05, by = 0.05), 
                         labels = function(x) format(x, nsmall = 2),
                         limits = y_lims) +
      labs(x = "Iteration", y = "F1-score", color = "Query\nstrategy") +
      theme_minimal() +
      theme(text = element_text(size = 16))
    ggsave("plots/plot_2000_pfeifferinv.pdf", plot_2000_pfeifferinv, 
           width = 9, height = 7, units = "in", device = cairo_pdf)
  }
  
  ## Plot creation: 2000 - lora
  {
    baselines_df[baselines_df$experiment == '2000_lora_baseline', 'f1'] # 0.696
    results_df %>% 
      filter(`PEFT method` == "LoRA" & `Total size` == 2000) %>%
      view()
    results_df %>% 
      filter(`PEFT method` == "LoRA" & `Total size` == 2000) %>%
      select(where(is.numeric)) %>%
      unlist() %>%
      max()
    
    plot_2000_lora <- results_df %>% 
      filter(`PEFT method` == "LoRA" & `Total size` == 2000) %>%
      pivot_longer(
        cols = starts_with("f1_"),
        names_to = "Iteration",
        values_to = "F1-score"
      ) %>%
      mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
      mutate(Lambda = paste0("λ = ", Lambda)) %>%
      mutate(Lambda = factor(Lambda, levels = c("λ = 10", "λ = 50", "λ = 100", "λ = 500"))) %>%
      ggplot(aes(x = Iteration, y = `F1-score`, 
                 color = factor(Strategy, levels = c("Random", "Margin", "k-means")))) +
      geom_hline(yintercept = baselines_df[baselines_df$experiment == '2000_lora_baseline', 'f1'],
                 color = "black", size = 1) +
      facet_wrap(~ Lambda, ncol = 2, nrow = 2) +
      geom_line(size = 1.2) +
      geom_point(size = 2) +
      scale_x_continuous(breaks = 0:10, labels = 0:10) +
      scale_y_continuous(breaks = seq(ceiling(y_lims[1] / 0.05) * 0.05, 
                                      floor(y_lims[2] / 0.05) * 0.05, by = 0.05), 
                         labels = function(x) format(x, nsmall = 2),
                         limits = y_lims) +
      labs(x = "Iteration", y = "F1-score", color = "Query\nstrategy") +
      theme_minimal() +
      theme(text = element_text(size = 16))
    ggsave("plots/plot_2000_lora.pdf", plot_2000_lora, 
           width = 9, height = 7, units = "in", device = cairo_pdf)
  }
}

