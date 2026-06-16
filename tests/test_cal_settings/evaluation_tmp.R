library(dplyr)
library(tidyr)
library(tidyverse)
library(stringr)
library(ggplot2)

results <- rbind(read.csv("tmp/results_old.csv"), read.csv("tmp/results_new.csv")) %>%
  filter(startsWith(experiment, "sensation_1000_lora_uncertainty") | 
           startsWith(experiment, "sensation_2000_lora_uncertainty") |
           startsWith(experiment, "sensation_300_lora_uncertainty") |
           startsWith(experiment, "sensation_300_lora_diversity")) %>%
  group_by(experiment) %>%
  summarise(across(starts_with("f1_"), mean)) %>%
  ungroup() %>%
  separate(experiment, into = c("Dataset", "Size", "PEFT", "Strategy", "Lambda"), sep = "_") %>%
  pivot_longer(
    cols = starts_with("f1_"),
    names_to = "Iteration",
    values_to = "F1"
  ) %>%
  mutate(Iteration = as.numeric(str_remove(Iteration, "f1_"))) %>%
  select(-Dataset, -PEFT, -Strategy)

results %>% 
  ggplot(aes(x = Iteration, y = F1, group = interaction(Size, Lambda), colour = Size)) +
  geom_line() +
  scale_x_continuous(breaks = 0:10)
