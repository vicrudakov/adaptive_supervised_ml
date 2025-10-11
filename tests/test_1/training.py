import os
import gc
import torch
import json
import time
from pathlib import Path
import pandas as pd
from loguru import logger
from transformers import TrainingArguments
from adapters import AutoAdapterModel, LoRAConfig, IA3Config, SeqBnConfig, SeqBnInvConfig
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings
from al_functions import select_obs
from data_functions import prepare_experiment_files
from peft_functions import EWCAdapterTrainer
from pet_functions import PEThead
from utility_functions import compute_metrics, read_config
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    # Compute and save metrics
    scores = compute_metrics(data['actual_test'], preds)
    with open(output_dir / "scores.json", "w") as fp:
        json.dump(scores, fp)

    # Create and save classification report
    report = pd.DataFrame(classification_report(data['actual_test'], preds, output_dict=True)).T
    report.to_csv(f'{str(output_dir)}/classification_report.csv')

    # Save predictions
    predictions = pd.DataFrame([data['labels'][x] for x in preds])
    predictions = pd.concat([predictions, pd.DataFrame(probs)], axis=1)
    predictions.to_csv(output_dir / f"predictions.csv", header=False, index=False)

def run_adapter_training(experiment, config, data, device, adapter_name="myadapter"):
    """A function to perform run PEFT module training.

    Parameters
    ----------
    experiment : Path
        The path to the experiment directory where outputs, embeddings, and adapters will be saved.
    config : dict
        A dictionary containing configuration parameters.
    data : dict
        A dictionary containing data as produced by `prepare_experiment_files`.
    device : torch.device
        Device on which to perform computation.
    adapter_name : str, optional
        The name of the PEFT module to be trained and saved during the experiment. Default is "myadapter".

    Returns
    -------
    None. Saves PEFT module files for each iteration.
    """

    logger.debug('Started training')

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
    output_dir = experiment / config['data']['output_dir']
    training_args = TrainingArguments(
        seed=int(1895),
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
    model.add_adapter(adapter_name, config=config_adapter)
    model.register_custom_head("PEThead", PEThead)
    model.add_custom_head(head_type="PEThead", head_name=adapter_name, id2tokenid=data['id2tokenid'])

    # Set training configuration
    al_iterations = config['active_learning']['al_iteration_number']
    al_strategy = config['active_learning']['al_strategy']
    model.train_adapter(adapter_name)
    selection_kwargs = {}
    if al_strategy == "random":
        selection_kwargs["seed"] = 42
    elif al_strategy == "uncertainty":
        selection_kwargs["model"] = model
    elif al_strategy == "diversity":
        selection_kwargs["emb_dir"] = experiment / config['data']['emb_dir']
        selection_kwargs["seed"] = 42
    available_train_rows = list(range(0, len(data['train_dataset']["train"])))
    trainer = EWCAdapterTrainer(
        model=model,
        lambda_ewc=config['active_learning']['lambda_ewc'],
        args=training_args
    )

    # Run continual active learning or baseline training
    for al_iter in range(al_iterations + 1):
        os.makedirs(output_dir / f"{adapter_name}_{al_iter}", exist_ok=True)
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
        trainer.train_dataset = current_train_dataset

        # Run PEFT module training for current iteration
        logger.debug(f'Started training for AL iteration {al_iter}, current train size: {len(current_train_rows)}')
        training_starttime = time.time()
        # trainer.train()
        training_endtime = time.time()
        if arch in ("ia3", "lora"):
            model.merge_adapter(adapter_name)
        logger.debug(f'Started computing Fisher information matrix for AL iteration {al_iter}')
        trainer.save_fisher(model, current_train_dataset, device)

        # Evaluate PEFT module
        logger.debug(f'Started evaluation for AL iteration {al_iter}')
        evaluate_model(model, data, output_dir / f"{adapter_name}_{al_iter}")
        evaluate_endtime = time.time()
        times = {"train": training_endtime - training_starttime, "test": evaluate_endtime - training_endtime}
        with open(output_dir / f"{adapter_name}_{al_iter}" / "time.json", "w") as fp:
            json.dump(times, fp)

        gc.collect()

        # Save PEFT module files
        model.save_adapter(output_dir / f"{adapter_name}_{al_iter}", adapter_name)

        torch.cuda.empty_cache()

def run_experiment(path):
    """A function to run a full experiment pipeline for training.

    Parameters
    ----------
    path : str or Path
        The path to the experiment directory.

    Returns
    -------
    None.
    """

    if type(path) == str:
        path = Path(path)
    logger.debug(f'Running experiment {path}')
    config = read_config(path)
    data = prepare_experiment_files(path, config, device='cuda')
    run_adapter_training(path, config, data, device='cuda')


if __name__ == '__main__':
    pass
