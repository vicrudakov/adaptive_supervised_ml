import os
import yaml
from loguru import logger
import pandas as pd

## Hyperparameter Tuning Global Settings
# Fix training data size: 1% for warm starting, 1% for each AL iteration (start_dataset_fraction = 0.01, query_size_fraction = 0.01)
# Fix AL method: maximum entropy sampling (strategy = entropy)
# Tune the current step objective for every dataset

## Hyperparameter Tuning Step 1
# Fix PEFT method for all datasets: LoRA(r = 8, alpha = 8)
# Tune parameters for every CL method

## Hyperparameter Tuning Step 2
# Fix the configuration of every CL method for every dataset based on step 1
# Tune parameters for every PEFT method


# Step 1
dataset_list = ["agnews", "sensation", "trec"]
continual_learning_method_list = ["der", "sd", "sds2"]
parameter_efficient_fine_tuning_config = {
    "architecture": "lora",
    "c_rate": 0,
    "r": 8,
    "alpha": 8,
    "prefix_len": 0
}
active_learning_config = {
    "strategy": "entropy",
    "start_dataset_fraction": 0.01,
    "query_size_fraction": 0.01,
    "al_iteration_number": 9
}
training_config = {
    "model": "xlm-roberta-large",
    "learning_rate": 1.0e-04,
    "num_train_epochs": 15,
    "per_device_train_batch_size": 10,
    "run_number": 1,
    "output_dir": "output"
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
    elif dataset == "sensation":
        pattern = "Ein <mask> Satz: <TEXT>"
        verbalizer = {
            "neutral": "neutraler",
            "sensationalistisch": "sensationalistischer"
        }
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
        if continual_learning_method == "der":
            alpha_list = [0.1, 0.25, 0.5, 0.75]
            beta_list = [0.75, 1]
            replay_size_fraction_list = [0.01, 0.02]
            for alpha in alpha_list:
                for beta in beta_list:
                    for replay_size_fraction in replay_size_fraction_list:
                        continual_learning_config = {
                            "method": "der",
                            "alpha": alpha,
                            "beta": beta,
                            "replay_size_fraction": replay_size_fraction,
                            "kernel_width": 0,
                            "l": 0,
                            "c_fraction": 0
                        }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"tuning_1/tuning_{dataset}_der_a{alpha}_b{beta}_r{replay_size_fraction}", exist_ok=True)
                        os.makedirs(f"tuning_1/tuning_{dataset}_der_a{alpha}_b{beta}_r{replay_size_fraction}/output", exist_ok=True)
                        with open(f"tuning_1/tuning_{dataset}_der_a{alpha}_b{beta}_r{replay_size_fraction}/config.yml", "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"tuning_1/tuning_{dataset}_der_a{alpha}_b{beta}_r{replay_size_fraction}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"tuning_1/tuning_{dataset}_der_a{alpha}_b{beta}_r{replay_size_fraction}/train.csv",
                                     index=False, header=False)
                        logger.debug(f"Created tuning_{dataset}_der_a{alpha}_b{beta}_r{replay_size_fraction}")
        elif continual_learning_method == "sd":
            alpha_list = [0.1, 0.25, 0.5, 0.75]
            replay_size_fraction_list = [0.01, 0.02]
            for alpha in alpha_list:
                for replay_size_fraction in replay_size_fraction_list:
                    continual_learning_config = {
                        "method": "sd",
                        "alpha": alpha,
                        "beta": 0,
                        "replay_size_fraction": replay_size_fraction,
                        "kernel_width": 0,
                        "l": 0,
                        "c_fraction": 0
                    }
                    config = {
                        "pattern_exploiting_training": pattern_exploiting_training_config,
                        "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                        "active_learning": active_learning_config,
                        "continual_learning": continual_learning_config,
                        "training": training_config
                    }
                    os.makedirs(f"tuning_1/tuning_{dataset}_sd_a{alpha}_r{replay_size_fraction}", exist_ok=True)
                    os.makedirs(f"tuning_1/tuning_{dataset}_sd_a{alpha}_r{replay_size_fraction}/output", exist_ok=True)
                    with open(f"tuning_1/tuning_{dataset}_sd_a{alpha}_r{replay_size_fraction}/config.yml", "w") as f:
                        yaml_parts = []
                        for key, value in config.items():
                            part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                            yaml_parts.append(part)
                        yaml_string = "\n".join(yaml_parts)
                        f.write(yaml_string)
                    test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                    train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                    test.to_csv(f"tuning_1/tuning_{dataset}_sd_a{alpha}_r{replay_size_fraction}/test.csv",
                                index=False, header=False)
                    train.to_csv(f"tuning_1/tuning_{dataset}_sd_a{alpha}_r{replay_size_fraction}/train.csv",
                                 index=False, header=False)
                    logger.debug(f"Created tuning_{dataset}_sd_a{alpha}_r{replay_size_fraction}")
        elif continual_learning_method == "sds2":
            alpha_list = [0.1, 0.25, 0.5, 0.75]
            l_list = [0.1, 1, 10]
            kernel_width_list = [0.1, 1]
            replay_size_fraction_list = [0.01, 0.02]
            for alpha in alpha_list:
                for l in l_list:
                    for kernel_width in kernel_width_list:
                        for replay_size_fraction in replay_size_fraction_list:
                            continual_learning_config = {
                                "method": "sds2",
                                "alpha": alpha,
                                "beta": 0,
                                "replay_size_fraction": replay_size_fraction,
                                "kernel_width": kernel_width,
                                "l": l,
                                "c_fraction": replay_size_fraction * 1.5
                            }
                            config = {
                                "pattern_exploiting_training": pattern_exploiting_training_config,
                                "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                                "active_learning": active_learning_config,
                                "continual_learning": continual_learning_config,
                                "training": training_config
                            }
                            os.makedirs(f"tuning_1/tuning_{dataset}_sds2_a{alpha}_l{l}_k{kernel_width}_r{replay_size_fraction}", exist_ok=True)
                            os.makedirs(f"tuning_1/tuning_{dataset}_sds2_a{alpha}_l{l}_k{kernel_width}_r{replay_size_fraction}/output", exist_ok=True)
                            with open(f"tuning_1/tuning_{dataset}_sds2_a{alpha}_l{l}_k{kernel_width}_r{replay_size_fraction}/config.yml", "w") as f:
                                yaml_parts = []
                                for key, value in config.items():
                                    part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                    yaml_parts.append(part)
                                yaml_string = "\n".join(yaml_parts)
                                f.write(yaml_string)
                            test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                            train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                            test.to_csv(f"tuning_1/tuning_{dataset}_sds2_a{alpha}_l{l}_k{kernel_width}_r{replay_size_fraction}/test.csv",
                                        index=False, header=False)
                            train.to_csv(f"tuning_1/tuning_{dataset}_sds2_a{alpha}_l{l}_k{kernel_width}_r{replay_size_fraction}/train.csv",
                                         index=False, header=False)
                            logger.debug(f"Created tuning_{dataset}_sds2_a{alpha}_l{l}_k{kernel_width}_r{replay_size_fraction}")


# Step 2
dataset_list = ["agnews", "sensation", "trec"]
continual_learning_method_list = ["der", "sd", "sds2"]
parameter_efficient_fine_tuning_architecture_list = ["adapter", "lora", "prefix"]
active_learning_config = {
    "strategy": "entropy",
    "start_dataset_fraction": 0.01,
    "query_size_fraction": 0.01,
    "al_iteration_number": 9
}
training_config = {
    "model": "xlm-roberta-large",
    "learning_rate": 1.0e-04,
    "num_train_epochs": 15,
    "per_device_train_batch_size": 10,
    "run_number": 1,
    "output_dir": "output"
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
            if continual_learning_method == "der":
                continual_learning_config = {
                    "method": "der",
                    "alpha": 0.25,
                    "beta": 0.75,
                    "replay_size_fraction": 0.02,
                    "kernel_width": 0,
                    "l": 0,
                    "c_fraction": 0
                }
            elif continual_learning_method == "sd":
                continual_learning_config = {
                    "method": "sd",
                    "alpha": 0.5,
                    "beta": 0,
                    "replay_size_fraction": 0.02,
                    "kernel_width": 0,
                    "l": 0,
                    "c_fraction": 0
                }
            elif continual_learning_method == "sds2":
                continual_learning_config = {
                    "method": "sds2",
                    "alpha": 0.25,
                    "beta": 0,
                    "replay_size_fraction": 0.01,
                    "kernel_width": 1,
                    "l": 0.1,
                    "c_fraction": 0.01 * 1.5
                }
            for parameter_efficient_fine_tuning_architecture in parameter_efficient_fine_tuning_architecture_list:
                if parameter_efficient_fine_tuning_architecture == "adapter":
                    c_rate_list = [8, 16]
                    for c_rate in c_rate_list:
                        parameter_efficient_fine_tuning_config = {
                            "architecture": "adapter",
                            "c_rate": c_rate,
                            "r": 0,
                            "alpha": 0,
                            "prefix_len": 0
                        }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}", exist_ok=True)
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/output", exist_ok=True)
                        with open(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/config.yml", "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/train.csv",
                                     index=False, header=False)
                        logger.debug(f"Created tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}")
                elif parameter_efficient_fine_tuning_architecture == "lora":
                    r_list = [4, 8]
                    alpha_list = [8, 16]
                    for r in r_list:
                        for alpha in alpha_list:
                            parameter_efficient_fine_tuning_config = {
                                "architecture": "lora",
                                "c_rate": 0,
                                "r": r,
                                "alpha": alpha,
                                "prefix_len": 0
                            }
                            config = {
                                "pattern_exploiting_training": pattern_exploiting_training_config,
                                "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                                "active_learning": active_learning_config,
                                "continual_learning": continual_learning_config,
                                "training": training_config
                            }
                            os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}", exist_ok=True)
                            os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/output", exist_ok=True)
                            with open(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/config.yml", "w") as f:
                                yaml_parts = []
                                for key, value in config.items():
                                    part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                    yaml_parts.append(part)
                                yaml_string = "\n".join(yaml_parts)
                                f.write(yaml_string)
                            test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                            train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                            test.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/test.csv",
                                        index=False, header=False)
                            train.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/train.csv",
                                         index=False, header=False)
                            logger.debug(f"Created tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}")
                elif parameter_efficient_fine_tuning_architecture == "prefix":
                    prefix_len_list = [10, 20]
                    for prefix_len in prefix_len_list:
                        parameter_efficient_fine_tuning_config = {
                            "architecture": "prefix",
                            "c_rate": 0,
                            "r": 0,
                            "alpha": 0,
                            "prefix_len": prefix_len
                        }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}", exist_ok=True)
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/output", exist_ok=True)
                        with open(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/config.yml", "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/train.csv",
                                     index=False, header=False)
                        logger.debug(f"Created tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}")
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
            if continual_learning_method == "der":
                continual_learning_config = {
                    "method": "der",
                    "alpha": 0.1,
                    "beta": 0.75,
                    "replay_size_fraction": 0.01,
                    "kernel_width": 0,
                    "l": 0,
                    "c_fraction": 0
                }
            elif continual_learning_method == "sd":
                continual_learning_config = {
                    "method": "sd",
                    "alpha": 0.1,
                    "beta": 0,
                    "replay_size_fraction": 0.01,
                    "kernel_width": 0,
                    "l": 0,
                    "c_fraction": 0
                }
            elif continual_learning_method == "sds2":
                continual_learning_config = {
                    "method": "sds2",
                    "alpha": 0.5,
                    "beta": 0,
                    "replay_size_fraction": 0.02,
                    "kernel_width": 0.1,
                    "l": 0.1,
                    "c_fraction": 0.02 * 1.5
                }
            for parameter_efficient_fine_tuning_architecture in parameter_efficient_fine_tuning_architecture_list:
                if parameter_efficient_fine_tuning_architecture == "adapter":
                    c_rate_list = [8, 16]
                    for c_rate in c_rate_list:
                        parameter_efficient_fine_tuning_config = {
                            "architecture": "adapter",
                            "c_rate": c_rate,
                            "r": 0,
                            "alpha": 0,
                            "prefix_len": 0
                        }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}", exist_ok=True)
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/output", exist_ok=True)
                        with open(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/config.yml", "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/train.csv",
                                     index=False, header=False)
                        logger.debug(f"Created tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}")
                elif parameter_efficient_fine_tuning_architecture == "lora":
                    r_list = [4, 8]
                    alpha_list = [8, 16]
                    for r in r_list:
                        for alpha in alpha_list:
                            parameter_efficient_fine_tuning_config = {
                                "architecture": "lora",
                                "c_rate": 0,
                                "r": r,
                                "alpha": alpha,
                                "prefix_len": 0
                            }
                            config = {
                                "pattern_exploiting_training": pattern_exploiting_training_config,
                                "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                                "active_learning": active_learning_config,
                                "continual_learning": continual_learning_config,
                                "training": training_config
                            }
                            os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}", exist_ok=True)
                            os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/output", exist_ok=True)
                            with open(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/config.yml", "w") as f:
                                yaml_parts = []
                                for key, value in config.items():
                                    part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                    yaml_parts.append(part)
                                yaml_string = "\n".join(yaml_parts)
                                f.write(yaml_string)
                            test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                            train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                            test.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/test.csv",
                                        index=False, header=False)
                            train.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/train.csv",
                                         index=False, header=False)
                            logger.debug(f"Created tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}")
                elif parameter_efficient_fine_tuning_architecture == "prefix":
                    prefix_len_list = [10, 20]
                    for prefix_len in prefix_len_list:
                        parameter_efficient_fine_tuning_config = {
                            "architecture": "prefix",
                            "c_rate": 0,
                            "r": 0,
                            "alpha": 0,
                            "prefix_len": prefix_len
                        }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}", exist_ok=True)
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/output", exist_ok=True)
                        with open(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/config.yml", "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/train.csv",
                                     index=False, header=False)
                        logger.debug(f"Created tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}")
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
            if continual_learning_method == "der":
                continual_learning_config = {
                    "method": "der",
                    "alpha": 0.1,
                    "beta": 1,
                    "replay_size_fraction": 0.02,
                    "kernel_width": 0,
                    "l": 0,
                    "c_fraction": 0
                }
            elif continual_learning_method == "sd":
                continual_learning_config = {
                    "method": "sd",
                    "alpha": 0.5,
                    "beta": 0,
                    "replay_size_fraction": 0.02,
                    "kernel_width": 0,
                    "l": 0,
                    "c_fraction": 0
                }
            elif continual_learning_method == "sds2":
                continual_learning_config = {
                    "method": "sds2",
                    "alpha": 0.25,
                    "beta": 0,
                    "replay_size_fraction": 0.02,
                    "kernel_width": 0.1,
                    "l": 0.1,
                    "c_fraction": 0.02 * 1.5
                }
            for parameter_efficient_fine_tuning_architecture in parameter_efficient_fine_tuning_architecture_list:
                if parameter_efficient_fine_tuning_architecture == "adapter":
                    c_rate_list = [8, 16]
                    for c_rate in c_rate_list:
                        parameter_efficient_fine_tuning_config = {
                            "architecture": "adapter",
                            "c_rate": c_rate,
                            "r": 0,
                            "alpha": 0,
                            "prefix_len": 0
                        }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}", exist_ok=True)
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/output", exist_ok=True)
                        with open(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/config.yml", "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}/train.csv",
                                     index=False, header=False)
                        logger.debug(f"Created tuning_{dataset}_{continual_learning_method}_adapter_c{c_rate}")
                elif parameter_efficient_fine_tuning_architecture == "lora":
                    r_list = [4, 8]
                    alpha_list = [8, 16]
                    for r in r_list:
                        for alpha in alpha_list:
                            parameter_efficient_fine_tuning_config = {
                                "architecture": "lora",
                                "c_rate": 0,
                                "r": r,
                                "alpha": alpha,
                                "prefix_len": 0
                            }
                            config = {
                                "pattern_exploiting_training": pattern_exploiting_training_config,
                                "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                                "active_learning": active_learning_config,
                                "continual_learning": continual_learning_config,
                                "training": training_config
                            }
                            os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}", exist_ok=True)
                            os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/output", exist_ok=True)
                            with open(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/config.yml", "w") as f:
                                yaml_parts = []
                                for key, value in config.items():
                                    part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                    yaml_parts.append(part)
                                yaml_string = "\n".join(yaml_parts)
                                f.write(yaml_string)
                            test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                            train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                            test.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/test.csv",
                                        index=False, header=False)
                            train.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}/train.csv",
                                         index=False, header=False)
                            logger.debug(f"Created tuning_{dataset}_{continual_learning_method}_lora_r{r}_a{alpha}")
                elif parameter_efficient_fine_tuning_architecture == "prefix":
                    prefix_len_list = [10, 20]
                    for prefix_len in prefix_len_list:
                        parameter_efficient_fine_tuning_config = {
                            "architecture": "prefix",
                            "c_rate": 0,
                            "r": 0,
                            "alpha": 0,
                            "prefix_len": prefix_len
                        }
                        config = {
                            "pattern_exploiting_training": pattern_exploiting_training_config,
                            "parameter_efficient_fine_tuning": parameter_efficient_fine_tuning_config,
                            "active_learning": active_learning_config,
                            "continual_learning": continual_learning_config,
                            "training": training_config
                        }
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}", exist_ok=True)
                        os.makedirs(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/output", exist_ok=True)
                        with open(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/config.yml", "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/test.csv",
                                    index=False, header=False)
                        train.to_csv(f"tuning_2/tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}/train.csv",
                                     index=False, header=False)
                        logger.debug(f"Created tuning_{dataset}_{continual_learning_method}_prefix_p{prefix_len}")
