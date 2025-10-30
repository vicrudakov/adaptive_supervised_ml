import pandas as pd
import os
import yaml
from loguru import logger

datasets = ["agnews", "sensation", "yahoo"]
data_sizes = ["1000", "2000"]
peft_methods = ["lora", "pfeiffer", "pfeifferinv"]
al_strategies = ["baseline", "diversity", "random", "uncertainty"]
lambdas_ewc = [10, 50, 100, 500]

run_number = 16
for dataset in datasets:
    if dataset == "agnews":
        pattern = "The topic of this article is <mask>: <TEXT>"
        verbalizer = {
            "world": "World",
            "sci_tech": "Tech"
        }
    elif dataset == "sensation":
        pattern = "Das Mass an Sensationalismus in diesem Satz ist <mask>: <TEXT>"
        verbalizer = {
                "neutral": "gering",
                "sensationalistisch": "hoch"
        }
    elif dataset == "yahoo":
        pattern = "The topic of this question is <mask>: <TEXT>"
        verbalizer = {
            "society_culture": "Society",
            "science_mathematics": "Science"
        }
    data_config = {
        "api_key_file": "api_key.txt",
        "output_dir": "output",
        "emb_dir": "embeddings",
        "pattern": pattern,
        "verbalizer": verbalizer
    }
    for peft_method in peft_methods:
        adapter_config = {
            "model": "xlm-roberta-large",
            "arch": peft_method,
            "c_rate": 16,
            "learning_rate": 5.0e-5,
            "r": 8,
            "alpha": 16,
            "num_train_epochs": 15,
            "per_device_train_batch_size": 5,
        }
        for data_size in data_sizes:
            for al_strategy in al_strategies:
                if data_size == "1000":
                    start_dataset_fraction = 0.125
                    query_size_fraction = 0.0125
                elif data_size == "2000":
                    start_dataset_fraction = 0.25
                    query_size_fraction = 0.025
                if al_strategy == "baseline":
                    query_size_fraction = 0
                    al_iteration_number = 0
                    lambda_ewc = 0
                    active_learning_config = {
                        "al_strategy": "random",
                        "start_dataset_fraction": start_dataset_fraction,
                        "query_size_fraction": query_size_fraction,
                        "al_iteration_number": al_iteration_number,
                        "lambda_ewc": lambda_ewc,
                        "run_number": run_number
                    }
                    config = {
                        "data": data_config,
                        "adapter": adapter_config,
                        "active_learning": active_learning_config
                    }

                    # Create directories
                    os.makedirs(f"experiments/{dataset}_{data_size}_{peft_method}_baseline", exist_ok=True)
                    os.makedirs(f"experiments/{dataset}_{data_size}_{peft_method}_baseline/embeddings", exist_ok=True)
                    os.makedirs(f"experiments/{dataset}_{data_size}_{peft_method}_baseline/output", exist_ok=True)

                    # Create files
                    with open(f"experiments/{dataset}_{data_size}_{peft_method}_baseline/config.yml", "w") as f:
                        yaml_parts = []
                        for key, value in config.items():
                            part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                            yaml_parts.append(part)
                        yaml_string = "\n".join(yaml_parts)
                        f.write(yaml_string)
                    embeddings = pd.read_csv(f"embeddings/{dataset}/embeddings.csv")
                    embeddings.to_csv(f"experiments/{dataset}_{data_size}_{peft_method}_baseline/embeddings/embeddings.csv", index = False)
                    test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                    train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                    test.to_csv(f"experiments/{dataset}_{data_size}_{peft_method}_baseline/test.csv", index=False, header=False)
                    train.to_csv(f"experiments/{dataset}_{data_size}_{peft_method}_baseline/train.csv", index=False, header=False)

                    logger.debug(f"Created {dataset}_{data_size}_{peft_method}_baseline")
                else:
                    al_iteration_number = 10
                    for lambda_ewc in lambdas_ewc:
                        active_learning_config = {
                            "al_strategy": al_strategy,
                            "start_dataset_fraction": start_dataset_fraction,
                            "query_size_fraction": query_size_fraction,
                            "al_iteration_number": al_iteration_number,
                            "lambda_ewc": lambda_ewc,
                            "run_number": run_number
                        }
                        config = {
                            "data": data_config,
                            "adapter": adapter_config,
                            "active_learning": active_learning_config
                        }

                        # Create directories
                        os.makedirs(f"experiments/{dataset}_{data_size}_{peft_method}_{al_strategy}_{lambda_ewc}", exist_ok=True)
                        os.makedirs(f"experiments/{dataset}_{data_size}_{peft_method}_{al_strategy}_{lambda_ewc}/embeddings", exist_ok=True)
                        os.makedirs(f"experiments/{dataset}_{data_size}_{peft_method}_{al_strategy}_{lambda_ewc}/output", exist_ok=True)

                        # Create files
                        with open(f"experiments/{dataset}_{data_size}_{peft_method}_{al_strategy}_{lambda_ewc}/config.yml", "w") as f:
                            yaml_parts = []
                            for key, value in config.items():
                                part = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, indent=2)
                                yaml_parts.append(part)
                            yaml_string = "\n".join(yaml_parts)
                            f.write(yaml_string)
                        embeddings = pd.read_csv(f"embeddings/{dataset}/embeddings.csv")
                        embeddings.to_csv(f"experiments/{dataset}_{data_size}_{peft_method}_{al_strategy}_{lambda_ewc}/embeddings/embeddings.csv", index=False)
                        test = pd.read_csv(f"data/{dataset}/test.csv", header=None)
                        train = pd.read_csv(f"data/{dataset}/train.csv", header=None)
                        test.to_csv(f"experiments/{dataset}_{data_size}_{peft_method}_{al_strategy}_{lambda_ewc}/test.csv", index=False, header=False)
                        train.to_csv(f"experiments/{dataset}_{data_size}_{peft_method}_{al_strategy}_{lambda_ewc}/train.csv", index=False, header=False)

                        logger.debug(f"Created {dataset}_{data_size}_{peft_method}_{al_strategy}_{lambda_ewc}")