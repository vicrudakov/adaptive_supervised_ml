import torch
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from loguru import logger
from transformers import AutoTokenizer
import numpy as np
from utility_functions import encode_batch, insert_list, extend_attention_mask, create_embeddings

def data_from_csv(path, pattern, verbalizer, tokenizer, split_name):
    """ A function to load data from a CSV file and process it into tokenized and padded format.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file containing the dataset. The file must have two columns: the first for text and the
        second for labels.
    pattern : dict
        A dictionary containing pattern information for pattern-exploiting training. Must include:
        - 'pattern': a pattern
        - 'n_mask_token': number of [MASK] tokens in the pattern
        - 'input_ids': list of token IDs representing the pattern
        - 'text_index': index in the pattern of [TEXT] token
        - 'n_token': number of tokens in the pattern
    verbalizer : dict
        A verbalizer for pattern-exploiting training.
    tokenizer : transformers.PreTrainedTokenizer
        A tokenizer instance used to tokenize the text.
    split_name : str
        Name of the dataset split to create, either "train" or "test".

    Returns
    -------
    tuple
        dataset_dict : datasets.DatasetDict
            DatasetDict containing tokenized, padded, and formatted dataset.
        labels : list
            Sorted list of unique labels present in the dataset.
        id2tokenid : dict
            Mapping from label IDs to their tokenized verbalizer IDs.
    """

    # Load CSV file
    dataset = pd.read_csv(path, quotechar='"', header=None, names=['text', 'label'])

    # Get labels
    labels = list(dataset.label.unique())
    labels.sort()

    # Map label IDs to their tokenized verbalizer IDs
    id2tokenid = {idx:tokenizer(verbalizer[label], add_special_tokens=False, return_attention_mask=False).input_ids[0:pattern['n_mask_token']] for idx,label in enumerate(labels)}

    # Convert labels in dataset to the IDs
    dataset['labels'] = [labels.index(x) for x in dataset['label']]

    # Set dictionary name for split
    if split_name == 'test':
        dict_name = 'test'
    elif split_name == 'train':
        dict_name = 'train'

    # Create DatasetDict
    dataset_dict = DatasetDict({dict_name: Dataset.from_pandas(dataset)})
    dataset_dict = dataset_dict.map(lambda x: encode_batch(x, tokenizer=tokenizer, max_length=512-pattern['n_token']), batched=True)
    dataset_dict = dataset_dict.map(lambda x: {'input_ids': insert_list(pattern['input_ids'], pattern['text_index'], x['input_ids'], tokenizer.pad_token_id)})
    dataset_dict = dataset_dict.map(lambda x: {'attention_mask': extend_attention_mask(x['attention_mask'], pattern['n_token'])})

    # Get positions of all [MASK] tokens, add the positions to DatasetDict
    mask_indices = [np.where(np.array(ids) == tokenizer.mask_token_id)[0] for ids in dataset_dict[dict_name]["input_ids"]]
    torch_columns = ["input_ids", "attention_mask", "labels"]
    for i in range(pattern['n_mask_token']):
        dataset_dict[dict_name] = dataset_dict[dict_name].add_column("mask_indices" + str(i + 1), [x[i] for x in mask_indices])
        torch_columns.append("mask_indices" + str(i + 1))

    dataset_dict.set_format(type="torch", columns=torch_columns)
    return dataset_dict, labels, id2tokenid

def create_experiment_data(experiment, pattern_text, verbalizer, tokenizer, device):
    """A function to create data for the experiment.

    Parameters
    ----------
    experiment : pathlib.Path
        Path to the experiment directory containing `train.csv` and `test.csv`.
    pattern_text : str
        A pattern for pattern-exploiting training.
    verbalizer : dict
        A verbalizer for pattern-exploiting training.
    tokenizer : transformers.PreTrainedTokenizer
        A tokenizer instance used to tokenize the text.
    device : torch.device
        Device on which to perform computation.

    Returns
    -------
    dict
        A dictionary containing:
            - 'labels' : label values
            - 'id2tokenid' : mapping from label IDs to token IDs
            - 'test_dataset' : DatasetDict for testing
            - 'train_dataset' : DatasetDict for training
            - 'actual_test' : test labels
            - 'actual_train' : train labels
            - 'input_ids_test' : test input IDs
            - 'input_ids_train' : train input IDs
            - 'attention_mask_test' : test attention mask
            - 'attention_mask_train' : train attention mask
            - 'mask_indices_testI' : positions of mask tokens
            - 'mask_indices_trainI' : positions of mask tokens
    """

    # Create pattern dictionary
    pattern = pattern_text.replace("<mask>", tokenizer.mask_token)
    pattern = {
        'pattern' : pattern,
        'n_mask_token' : pattern.count(tokenizer.mask_token),
        'input_ids' : tokenizer(pattern, return_attention_mask=False).input_ids,
    }
    pattern['text_index'] = np.where(np.array(pattern['input_ids']) == tokenizer.additional_special_tokens_ids)[0][0]
    pattern['input_ids'].pop(pattern['text_index'])
    pattern['n_token'] = len(pattern['input_ids'])

    # Load processed test data
    test_path = experiment / 'test.csv'
    dataset_test, labels, id2tokenid = data_from_csv(test_path, pattern, verbalizer, tokenizer, split_name='test')

    # Load processed train data
    train_path = experiment / "train.csv"
    dataset_train, _, _ = data_from_csv(train_path, pattern, verbalizer, tokenizer, split_name='train')

    # Create data for the experiment
    device = torch.device(device)
    data = {}
    data['labels'] = labels
    data['id2tokenid'] = id2tokenid
    data['test_dataset'] = dataset_test
    data['train_dataset'] = dataset_train
    data['actual_test'] = dataset_test["test"]["labels"].numpy()
    data['actual_train'] = dataset_train["train"]["labels"].numpy()
    data['input_ids_test'] = dataset_test["test"]["input_ids"].to(device)
    data['input_ids_train'] = dataset_train["train"]["input_ids"].to(device)
    data['attention_mask_test'] = dataset_test["test"]["attention_mask"].to(device)
    data['attention_mask_train'] = dataset_train["train"]["attention_mask"].to(device)
    for i in range(pattern['n_mask_token']):
        data["mask_indices_test" + str(i + 1)] = dataset_test["test"]["mask_indices" + str(i + 1)].to(device)
        data["mask_indices_train" + str(i + 1)] = dataset_train["train"]["mask_indices" + str(i + 1)].to(device)

    logger.debug(f'Created data for the experiment')
    return data

def prepare_experiment_files(experiment, config, device):
    """A function to prepare the necessary files and data for an experiment.

    Parameters
    ----------
    experiment : Path or str
        Path to the experiment directory where data and outputs are stored.
    config : dict
        Experiment configuration dictionary loaded from the configuration file.
    device : torch.device
        Device on which to perform computation.

    Returns
    -------
    dict
        Dictionary containing prepared datasets, labels, input IDs, attention masks, mask indices,
        and other data required for training and evaluation.
    """

    tokenizer = AutoTokenizer.from_pretrained(config['adapter']['model'], additional_special_tokens=["<TEXT>"])
    data = create_experiment_data(
        experiment=experiment,
        pattern_text=config['data']['pattern'],
        verbalizer=config['data']['verbalizer'],
        tokenizer=tokenizer,
        device=device
    )
    if config['active_learning']['al_strategy'] == "diversity":
        create_embeddings(
            data=data,
            api_key_file=config['data']['api_key_file'],
            emb_dir=experiment / config['data']['emb_dir']
        )
    logger.debug(f'Prepared experiment files')
    return data