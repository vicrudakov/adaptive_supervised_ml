import os
import gc
import torch
import json
import time
import random
import itertools
from pathlib import Path
import pandas as pd
from loguru import logger
from transformers import TrainingArguments
from adapters import AutoAdapterModel, LoRAConfig, IA3Config, SeqBnConfig, SeqBnInvConfig
import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score
from tqdm import tqdm
import warnings
from al_functions import select_obs
from data_functions import prepare_experiment_files
from peft_functions import EWCAdapterTrainer
from pet_functions import PEThead
from utility_functions import read_config
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def evaluate_model(model, data, output_dir):
    """A function to evaluate a trained model on the test dataset, compute performance metrics, and save evaluation
    results, classification report, and predictions with probabilities.

    Parameters
    ----------
    model : transformers.AutoAdapterModel
        The model to be evaluated.
    data : dict
        A dictionary containing data as produced by `prepare_experiment_files`.
    output_dir : str or pathlib.Path
        Path to the directory where evaluation results will be saved.

    Returns
    -----
    None. Saves:
    - scores.json : computed evaluation metrics.
    - classification_report.csv : classification report.
    - predictions.csv : predicted labels and their probabilities.
    """

    # Ensure output_dir is a Path object
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Put model into evaluation mode
    model.eval()

    # Prepare placeholders for predictions and probabilities
    preds = [-100] * len(data['input_ids_test'])
    probs = [-100] * len(data['input_ids_test'])

    # Make forward pass and get predictions for all observations
    for i in tqdm(range(len(data['input_ids_test'])), desc='Predicting labels for test data'):
        with torch.no_grad():
            model_args = {key.replace('_test', ''): data[key][i] for key in data.keys() if key in ['input_ids_test', 'attention_mask_test'] or 'mask_indices_test' in key}
            res = model(**model_args)[0]
            preds[i] = np.argmax(res.cpu().detach().numpy(), axis = 1)[0]
            probs[i] = torch.round(torch.softmax(res.cpu().detach(), dim=1).squeeze(), decimals=3).tolist()

    # Create and save classification report
    report = pd.DataFrame(classification_report(data['actual_test'], preds, output_dict=True)).T
    report.to_csv(output_dir / "classification_report.csv")

    # Save predictions
    predictions = pd.DataFrame([data['labels'][x] for x in preds])
    predictions = pd.concat([predictions, pd.DataFrame(probs)], axis=1)
    predictions.to_csv(output_dir / "predictions.csv", header=False, index=False)

def stabilizing_predictions(data, al_iter, available_train_rows, config, adapter_name, output_dir, device):
    threshold = 0.99
    k = 3
    n = 1000

    if al_iter + 1 < k:
        logger.debug(f"Iteration number {al_iter}, waiting for {k} iterations to pass")
        return

    random.seed(42)
    stop_set_rows = random.sample(available_train_rows, n)
    predictions = np.zeros(shape=(k, n))
    c = 0

    for j in range(k-3, k):
        model = AutoAdapterModel.from_pretrained(config['adapter']['model'])
        model.register_custom_head("PEThead", PEThead)
        model.load_head(str(output_dir / f"{adapter_name}_{j}"))
        model.load_adapter(str(output_dir / f"{adapter_name}_{j}"))
        model.active_adapters = adapter_name
        model.heads[adapter_name].config['id2tokenid'] = {int(k): v for k, v in model.heads[adapter_name].config['id2tokenid'].items()}
        model = model.to(device)
        model.eval()
        preds = [-100] * len(data['input_ids_train'])
        for i in range(len(data['input_ids_train'])):
            with torch.no_grad():
                if i in stop_set_rows:
                    model_args = {key.replace('_train', ''): data[key][i] for key in data.keys() if
                                  key in ['input_ids_train', 'attention_mask_train'] or 'mask_indices_train' in key}
                    res = model(**model_args)[0]
                    preds[i] = np.argmax(res.cpu().detach().numpy(), axis = 1)[0]
                else:
                    preds[i] = None
        predictions[c] = [x for x in preds if x is not None]
        c += 1

    kappa = np.mean([cohen_kappa_score(x, y) for x, y in itertools.combinations(predictions, 2)])
    logger.debug(f"Iteration {al_iter}, kappa = {kappa}")
    if kappa >= threshold:
        return True
    else:
        return False

def classification_change(data, al_iter, available_train_rows, config, adapter_name, output_dir, device):
    k = 2

    if al_iter + 1 < k:
        logger.debug(f"Iteration number {al_iter}, waiting for {k} iterations to pass")
        return

    # random.seed(42)
    # stop_set_rows = random.sample(available_train_rows, 10)
    # predictions = np.zeros(shape=(k, len(stop_set_rows)))
    predictions = np.zeros(shape=(k, len(available_train_rows)))
    c = 0

    for j in range(k-2, k):
        model = AutoAdapterModel.from_pretrained(config['adapter']['model'])
        model.register_custom_head("PEThead", PEThead)
        model.load_head(str(output_dir / f"{adapter_name}_{j}"))
        model.load_adapter(str(output_dir / f"{adapter_name}_{j}"))
        model.active_adapters = adapter_name
        model.heads[adapter_name].config['id2tokenid'] = {int(k): v for k, v in model.heads[adapter_name].config['id2tokenid'].items()}
        model = model.to(device)
        model.eval()
        preds = [-100] * len(data['input_ids_train'])
        for i in range(len(data['input_ids_train'])):
            with torch.no_grad():
                if i in available_train_rows: # stop_set_rows
                    model_args = {key.replace('_train', ''): data[key][i] for key in data.keys() if
                                  key in ['input_ids_train', 'attention_mask_train'] or 'mask_indices_train' in key}
                    res = model(**model_args)[0]
                    preds[i] = np.argmax(res.cpu().detach().numpy(), axis = 1)[0]
                else:
                    preds[i] = None
        predictions[c] = [x for x in preds if x is not None]
        c += 1

    differences = predictions[0] != predictions[1]
    count = np.sum(differences)
    logger.debug(f"Iteration {al_iter}, difference count = {count}")
    if count == 0:
        return True
    else:
        return False

def oracle_acc_mcs(data, al_iter, current_train_rows, config, adapter_name, output_dir, device):
    threshold = 0.9

    if al_iter < 1:
        logger.debug(f"Iteration number {al_iter}, waiting for 1 iteration to pass")
        return

    model = AutoAdapterModel.from_pretrained(config['adapter']['model'])
    model.register_custom_head("PEThead", PEThead)
    model.load_head(str(output_dir / f"{adapter_name}_{al_iter-1}"))
    model.load_adapter(str(output_dir / f"{adapter_name}_{al_iter-1}"))
    model.active_adapters = adapter_name
    model.heads[adapter_name].config['id2tokenid'] = {int(k): v for k, v in model.heads[adapter_name].config['id2tokenid'].items()}
    model = model.to(device)
    model.eval()
    preds = [-100] * len(data['input_ids_train'])
    for i in range(len(data['input_ids_train'])):
        with torch.no_grad():
            if i in current_train_rows:
                model_args = {key.replace('_train', ''): data[key][i] for key in data.keys() if
                              key in ['input_ids_train', 'attention_mask_train'] or 'mask_indices_train' in key}
                res = model(**model_args)[0]
                preds[i] = np.argmax(res.cpu().detach().numpy(), axis = 1)[0]
            else:
                preds[i] = None

    acc = np.mean(data['actual_train'][current_train_rows] == [preds[i] for i in current_train_rows])
    logger.debug(f"Iteration {al_iter}, accuracy = {acc}")
    if acc >= threshold:
        return True
    else:
        return False

def run_adapter_training(experiment, config, data, sc, device, adapter_name="myadapter"):
    """A function to perform run PEFT module training.

    Parameters
    ----------
    experiment : Path
        The path to the experiment directory where outputs, embeddings, and adapters will be saved.
    config : dict
        A dictionary containing configuration parameters.
    data : dict
        A dictionary containing data as produced by `prepare_experiment_files`.
    sc : str
        Stopping criterion method for AL.
    device : torch.device
        Device on which to perform computation.
    adapter_name : str, optional
        The name of the PEFT module to be trained and saved during the experiment. Default is "myadapter".

    Returns
    -------
    None. Saves PEFT module files for each iteration.
    """

    for run in range(1, config['active_learning']['run_number'] + 1):
        # logger.debug(f'Training run: {run}')
        # logger.debug(f'Started training')
        logger.debug(f'Run: {run}')

        # Set adapter configuration
        arch = config['adapter']['arch']
        if arch == "pfeiffer":
            config_adapter = SeqBnConfig(reduction_factor=config['adapter']['c_rate'])
        if arch == "pfeifferinv":
            config_adapter = SeqBnInvConfig(reduction_factor=config['adapter']['c_rate'])
        if arch == "lora":
            config_adapter = LoRAConfig(r=config['adapter']['r'], alpha=config['adapter']['alpha'])
        if arch == "ia3":
            config_adapter = IA3Config()

        # Set model configuration
        output_dir = experiment / config['data']['output_dir'] / f"run_{run}"
        training_args = TrainingArguments(
            seed=int(1895 * run),
            full_determinism=True,
            learning_rate=config['adapter']['learning_rate'],
            num_train_epochs=config['adapter']['num_train_epochs'],
            logging_strategy="no",
            eval_strategy="no",
            save_strategy="no",
            output_dir=output_dir,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            per_device_train_batch_size=config['adapter']['per_device_train_batch_size']
        )
        model = AutoAdapterModel.from_pretrained(config['adapter']['model'])
        # model.add_adapter(adapter_name, config=config_adapter)
        model.register_custom_head("PEThead", PEThead)
        # model.add_custom_head(head_type="PEThead", head_name=adapter_name, id2tokenid=data['id2tokenid'])

        # Set training configuration
        al_iterations = config['active_learning']['al_iteration_number']
        al_strategy = config['active_learning']['al_strategy']
        # model.train_adapter(adapter_name)
        selection_kwargs = {}
        if al_strategy == "random":
            selection_kwargs["seed"] = 42
        elif al_strategy == "uncertainty":
            selection_kwargs["model"] = model
        elif al_strategy == "diversity":
            selection_kwargs["emb_dir"] = experiment / config['data']['emb_dir']
            selection_kwargs["seed"] = 42
        available_train_rows = list(range(0, len(data['train_dataset']["train"])))
        # trainer = EWCAdapterTrainer(
        #     model=model,
        #     lambda_ewc=config['active_learning']['lambda_ewc'],
        #     args=training_args
        # )

        # Run continual active learning or baseline training
        for al_iter in range(al_iterations + 1):
            # os.makedirs(output_dir / f"{adapter_name}_{al_iter}", exist_ok=True)
            logger.debug(f"AL iteration: {al_iter}")

            # Select observations for training
            logger.debug(f'Started selecting training observations for AL iteration {al_iter}')
            if al_iter == 0:
                current_train_rows, current_train_dataset, available_train_rows = select_obs(
                    strategy="random",
                    data=data,
                    available_train_rows=available_train_rows,
                    n=int(len(data['train_dataset']["train"]) * config['active_learning']['start_dataset_fraction']),
                    **{"seed": 42}
                )
            else:
                current_train_rows, current_train_dataset, available_train_rows = select_obs(
                    strategy=al_strategy,
                    data=data,
                    available_train_rows=available_train_rows,
                    n=int(len(data['train_dataset']["train"]) * config['active_learning']['query_size_fraction']),
                    **selection_kwargs
                )
            # trainer.train_dataset = current_train_dataset

            # # Run PEFT module training for current iteration
            # logger.debug(f'Started training for AL iteration {al_iter}, current train size: {len(current_train_rows)}')
            # training_starttime = time.time()
            # trainer.train()
            # training_endtime = time.time()
            # if arch in ("ia3", "lora"):
            #     model.merge_adapter(adapter_name)
            # logger.debug(f'Started computing Fisher information matrix for AL iteration {al_iter}')
            # trainer.save_fisher(model, current_train_dataset, device)

            model.load_head(str(output_dir / f"{adapter_name}_{al_iter}"))
            model.load_adapter(str(output_dir / f"{adapter_name}_{al_iter}"))
            model.active_adapters = adapter_name
            model.heads[adapter_name].config['id2tokenid'] = {int(k): v for k, v in model.heads[adapter_name].config['id2tokenid'].items()}
            model = model.to(device)

            if sc == "stabilizing_predictions":
                stop = stabilizing_predictions(data, al_iter, available_train_rows, config, adapter_name, output_dir, device)
            elif sc == "classification_change":
                stop = classification_change(data, al_iter, available_train_rows, config, adapter_name, output_dir, device)
            elif sc == "oracle_acc_mcs":
                stop = oracle_acc_mcs(data, al_iter, current_train_rows, config, adapter_name, output_dir, device)
            if stop:
                logger.debug(f"Stopping training, AL iteration {al_iter}")
                gc.collect()
                torch.mps.empty_cache()
                break

            # # Evaluate PEFT module
            # logger.debug(f'Started evaluation for AL iteration {al_iter}')
            # evaluate_model(model, data, output_dir / f"{adapter_name}_{al_iter}")
            # evaluate_endtime = time.time()
            # times = {"train": training_endtime - training_starttime, "test": evaluate_endtime - training_endtime}
            # with open(output_dir / f"{adapter_name}_{al_iter}" / "time.json", "w") as fp:
            #     json.dump(times, fp)

            gc.collect()

            # # Save PEFT module files
            # model.save_adapter(output_dir / f"{adapter_name}_{al_iter}", adapter_name)

            torch.mps.empty_cache()

        break

def run_experiment(path, sc):
    """A function to run a full experiment pipeline for training.

    Parameters
    ----------
    path : str or Path
        The path to the experiment directory.
    sc : str
        Stopping criterion method for AL.

    Returns
    -------
    None.
    """

    if type(path) == str:
        path = Path(path)
    handler_id = logger.add(f"{path}/logs/{sc}.log", format="{level: <8} | {name}:{function}:{line} - {message}", mode="w")
    logger.debug(f'Running experiment {path}')
    config = read_config(path)
    data = prepare_experiment_files(path, config, device='mps')
    run_adapter_training(path, config, data, sc, device='mps')
    logger.remove(handler_id)

if __name__ == '__main__':
    ### run_experiment("experiments/sensation_1000_lora_random_10", sc="stabilizing_predictions")
    ### run_experiment("experiments/sensation_1000_lora_random_10", sc="classification_change")
    # run_experiment("experiments/sensation_1000_lora_random_10", sc="oracle_acc_mcs")

    # run_experiment("experiments/sensation_1000_pfeifferinv_uncertainty_10", sc="stabilizing_predictions")
    # run_experiment("experiments/sensation_1000_pfeifferinv_uncertainty_10", sc="classification_change")
    # run_experiment("experiments/sensation_1000_pfeifferinv_uncertainty_10", sc="oracle_acc_mcs")

    # run_experiment("experiments/sensation_2000_pfeifferinv_uncertainty_10", sc="stabilizing_predictions")
    ### run_experiment("experiments/sensation_2000_pfeifferinv_uncertainty_10", sc="classification_change")
    # run_experiment("experiments/sensation_2000_pfeifferinv_uncertainty_10", sc="oracle_acc_mcs")

    ### run_experiment("experiments/sensation_2000_pfeifferinv_diversity_100", sc="stabilizing_predictions")
    ### run_experiment("experiments/sensation_2000_pfeifferinv_diversity_100", sc="classification_change")
    # run_experiment("experiments/sensation_2000_pfeifferinv_diversity_100", sc="oracle_acc_mcs")

    pass