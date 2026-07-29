import os
import yaml
from loguru import logger
import pandas as pd

dataset_list = ["agnews", "sensation", "trec"]
parameter_efficient_fine_tuning_architecture_list = ["adapter", "lora", "prefix"]
active_learning_method_list = ["random", "entropy", "coreset"]
active_learning_start_dataset_fraction_list = [0.01]
continual_learning_method_list = ["der", "sd", "sds2"]
training_config = {
    "model": "xlm-roberta-large",
    "learning_rate": 1.0e-04,
    "num_train_epochs": 15,
    "per_device_train_batch_size": 10,
    "run_number": 3,
    "output_dir": "output"
}

for active_learning_method in active_learning_method_list:
    for active_learning_start_dataset_fraction in active_learning_start_dataset_fraction_list:
        active_learning_query_size_fraction = active_learning_start_dataset_fraction
        active_learning_config = {
            "strategy": active_learning_method,
            "start_dataset_fraction": active_learning_start_dataset_fraction,
            "query_size_fraction": active_learning_query_size_fraction,
            "al_iteration_number": 9
        }
        for dataset in dataset_list:
            if dataset == "agnews":
                pattern = "<mask> News: <TEXT>"
                verbalizer = {
                    "world": "World",
                    "sports": "Sports",
                    "business": "Business",
                    "sci_tech": "Tech",
                }
                pattern_exploiting_training_config = {
                    "pattern": pattern,
                    "verbalizer": verbalizer
                }
                for continual_learning_method in continual_learning_method_list:
                    for parameter_efficient_fine_tuning_architecture in parameter_efficient_fine_tuning_architecture_list:
                        if continual_learning_method == "der":
                            continual_learning_config = {
                                "method": "der",
                                "alpha": 0.25,
                                "beta": 0.75,
                                "replay_size_fraction": 2 * active_learning_query_size_fraction,
                                "kernel_width": 0,
                                "l": 0,
                                "c_fraction": 0
                            }
                            if parameter_efficient_fine_tuning_architecture == "adapter":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "adapter",
                                    "c_rate": 16,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "lora":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "lora",
                                    "c_rate": 0,
                                    "r": 8,
                                    "alpha": 8,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "prefix":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "prefix",
                                    "c_rate": 0,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 10
                                }
                        elif continual_learning_method == "sd":
                            continual_learning_config = {
                                "method": "sd",
                                "alpha": 0.5,
                                "beta": 0,
                                "replay_size_fraction": 2 * active_learning_query_size_fraction,
                                "kernel_width": 0,
                                "l": 0,
                                "c_fraction": 0
                            }
                            if parameter_efficient_fine_tuning_architecture == "adapter":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "adapter",
                                    "c_rate": 8,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "lora":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "lora",
                                    "c_rate": 0,
                                    "r": 4,
                                    "alpha": 16,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "prefix":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "prefix",
                                    "c_rate": 0,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 20
                                }
                        elif continual_learning_method == "sds2":
                            continual_learning_config = {
                                "method": "sds2",
                                "alpha": 0.25,
                                "beta": 0,
                                "replay_size_fraction": 1 * active_learning_query_size_fraction,
                                "kernel_width": 1,
                                "l": 0.1,
                                "c_fraction": 1 * active_learning_query_size_fraction * 1.5
                            }
                            if parameter_efficient_fine_tuning_architecture == "adapter":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "adapter",
                                    "c_rate": 8,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "lora":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "lora",
                                    "c_rate": 0,
                                    "r": 8,
                                    "alpha": 8,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "prefix":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "prefix",
                                    "c_rate": 0,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 20
                                }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                    f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}",
                                    exist_ok=True)
                        os.makedirs(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                    f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/output",
                                    exist_ok=True)
                        with open(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                  f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/config.yml",
                                  "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                    f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                            f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/train.csv",
                            index=False, header=False)
                        logger.debug(f"Created training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                     f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}")
            elif dataset == "sensation":
                pattern = "Ein <mask> Satz: <TEXT>"
                verbalizer = {
                    "neutral": "neutraler",
                    "sensationalistisch": "sensationalistischer"
                }
                pattern_exploiting_training_config = {
                    "pattern": pattern,
                    "verbalizer": verbalizer
                }
                for continual_learning_method in continual_learning_method_list:
                    for parameter_efficient_fine_tuning_architecture in parameter_efficient_fine_tuning_architecture_list:
                        if continual_learning_method == "der":
                            continual_learning_config = {
                                "method": "der",
                                "alpha": 0.1,
                                "beta": 0.75,
                                "replay_size_fraction": 1 * active_learning_query_size_fraction,
                                "kernel_width": 0,
                                "l": 0,
                                "c_fraction": 0
                            }
                            if parameter_efficient_fine_tuning_architecture == "adapter":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "adapter",
                                    "c_rate": 8,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "lora":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "lora",
                                    "c_rate": 0,
                                    "r": 8,
                                    "alpha": 8,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "prefix":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "prefix",
                                    "c_rate": 0,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 10
                                }
                        elif continual_learning_method == "sd":
                            continual_learning_config = {
                                "method": "sd",
                                "alpha": 0.1,
                                "beta": 0,
                                "replay_size_fraction": 1 * active_learning_query_size_fraction,
                                "kernel_width": 0,
                                "l": 0,
                                "c_fraction": 0
                            }
                            if parameter_efficient_fine_tuning_architecture == "adapter":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "adapter",
                                    "c_rate": 8,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "lora":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "lora",
                                    "c_rate": 0,
                                    "r": 8,
                                    "alpha": 8,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "prefix":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "prefix",
                                    "c_rate": 0,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 10
                                }
                        elif continual_learning_method == "sds2":
                            continual_learning_config = {
                                "method": "sds2",
                                "alpha": 0.5,
                                "beta": 0,
                                "replay_size_fraction": 2 * active_learning_query_size_fraction,
                                "kernel_width": 0.1,
                                "l": 0.1,
                                "c_fraction": 2 * active_learning_query_size_fraction * 1.5
                            }
                            if parameter_efficient_fine_tuning_architecture == "adapter":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "adapter",
                                    "c_rate": 8,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "lora":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "lora",
                                    "c_rate": 0,
                                    "r": 8,
                                    "alpha": 16,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "prefix":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "prefix",
                                    "c_rate": 0,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 10
                                }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                    f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}",
                                    exist_ok=True)
                        os.makedirs(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                    f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/output",
                                    exist_ok=True)
                        with open(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                  f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/config.yml",
                                  "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                    f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                            f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/train.csv",
                            index=False, header=False)
                        logger.debug(f"Created training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                     f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}")
            elif dataset == "trec":
                pattern = "Question about <mask>: <TEXT>"
                verbalizer = {
                    "abbr": "abbreviation",
                    "enty": "entity",
                    "desc": "concept",
                    "hum": "human",
                    "loc": "location",
                    "num": "number"
                }
                pattern_exploiting_training_config = {
                    "pattern": pattern,
                    "verbalizer": verbalizer
                }
                for continual_learning_method in continual_learning_method_list:
                    for parameter_efficient_fine_tuning_architecture in parameter_efficient_fine_tuning_architecture_list:
                        if continual_learning_method == "der":
                            continual_learning_config = {
                                "method": "der",
                                "alpha": 0.1,
                                "beta": 1,
                                "replay_size_fraction": 2 * active_learning_query_size_fraction,
                                "kernel_width": 0,
                                "l": 0,
                                "c_fraction": 0
                            }
                            if parameter_efficient_fine_tuning_architecture == "adapter":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "adapter",
                                    "c_rate": 8,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "lora":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "lora",
                                    "c_rate": 0,
                                    "r": 8,
                                    "alpha": 8,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "prefix":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "prefix",
                                    "c_rate": 0,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 20
                                }
                        elif continual_learning_method == "sd":
                            continual_learning_config = {
                                "method": "sd",
                                "alpha": 0.5,
                                "beta": 0,
                                "replay_size_fraction": 2 * active_learning_query_size_fraction,
                                "kernel_width": 0,
                                "l": 0,
                                "c_fraction": 0
                            }
                            if parameter_efficient_fine_tuning_architecture == "adapter":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "adapter",
                                    "c_rate": 8,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "lora":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "lora",
                                    "c_rate": 0,
                                    "r": 8,
                                    "alpha": 16,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "prefix":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "prefix",
                                    "c_rate": 0,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 10
                                }
                        elif continual_learning_method == "sds2":
                            continual_learning_config = {
                                "method": "sds2",
                                "alpha": 0.25,
                                "beta": 0,
                                "replay_size_fraction": 2 * active_learning_query_size_fraction,
                                "kernel_width": 0.1,
                                "l": 0.1,
                                "c_fraction": 2 * active_learning_query_size_fraction * 1.5
                            }
                            if parameter_efficient_fine_tuning_architecture == "adapter":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "adapter",
                                    "c_rate": 8,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "lora":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "lora",
                                    "c_rate": 0,
                                    "r": 8,
                                    "alpha": 8,
                                    "prefix_len": 0
                                }
                            elif parameter_efficient_fine_tuning_architecture == "prefix":
                                parameter_efficient_fine_tuning_config = {
                                    "architecture": "prefix",
                                    "c_rate": 0,
                                    "r": 0,
                                    "alpha": 0,
                                    "prefix_len": 10
                                }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                    f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}",
                                    exist_ok=True)
                        os.makedirs(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                    f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/output",
                                    exist_ok=True)
                        with open(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                  f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/config.yml",
                                  "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                    f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"training/training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                            f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}/train.csv",
                            index=False, header=False)
                        logger.debug(f"Created training_{dataset}_{int(active_learning_start_dataset_fraction * 1000)}_"
                                     f"{parameter_efficient_fine_tuning_architecture}_{continual_learning_method}_{active_learning_method}")

