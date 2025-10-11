import math
import os
import random
import gc
import torch
import json
import yaml
import sys
import time
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from loguru import logger
from transformers import AutoTokenizer, TrainingArguments
from adapters import AutoAdapterModel, AdapterTrainer, LoRAConfig, IA3Config, SeqBnConfig, SeqBnInvConfig, PredictionHead
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.cluster import KMeans
import torch.nn as nn
from tqdm import tqdm
import heapq
from collections import defaultdict
from openai import OpenAI
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def compute_metrics(a, p):
    """A function to compute classification performance measures accuracy, precision, recall and F1-score.

    Parameters
    ----------
    a : array
        Actual target values.
    p : array
        Predicted target values.

    Returns
    -------
    dict
        A dictionary containing:
        - 'accuracy': float
          The proportion of correct predictions.
        - 'f1': float
          The macro-averaged F1-score.
        - 'precision': float
          The macro-averaged precision score.
        - 'recall': float
          The macro-averaged recall score.
    """
    return {"accuracy": (p == a).mean(),
            "f1": f1_score(a, p, average='macro'),
            "precision": precision_score(a, p, average='macro'),
            "recall": recall_score(a, p, average='macro')}

def encode_batch(batch, tokenizer, max_length):
    """A function to tokenize batches of text data.

    Parameters
    ----------
    batch : dict
        A dictionary with a list of text samples to tokenize.
    tokenizer : PreTrainedTokenizer
        A tokenizer instance used to tokenize the text.
    max_length : int
        The maximum number of tokens allowed in each encoded sequence.

    Returns
    -------
    dict
        A dictionary containing the tokenized representations produced by the tokenizer.
    """
    return tokenizer(batch["text"], max_length=max_length, truncation=True, add_special_tokens=False)

def insert_list(target_list, position, new_elements, extend_with, extend_to=512):
    """A function to insert elements into a list and pad it to a fixed length.

    Parameters
    ----------
    target_list : list
        The original list into which new elements will be inserted.
    position : int
        The index at which the new elements will be inserted.
    new_elements : list
        The elements to insert into the list.
    extend_with : any
        The value used to extend the list until it reaches the desired length.
    extend_to : int, optional
        The target length of the final list after padding. Default is 512.

    Returns
    -------
    list
        A copy of the original list with new elements inserted at the specified position and padded to the specified
        length.
    """
    copied_list = target_list.copy()
    for index, item in enumerate(new_elements):
        copied_list.insert(position+index, item)
    copied_list.extend([extend_with]*(extend_to-len(copied_list)))
    return copied_list

def extend_attention_mask(target_list, pattern_length, extend_to=512):
    """A function to extend an attention mask by adding tokens for a pattern and padding to a fixed length.

    Parameters
    ----------
    target_list : list
        The original attention mask list to extend.
    pattern_length : int
        The number of tokens in the inserted pattern to mark as attended (value 1).
    extend_to : int, optional
        The target length of the final mask after padding. Default is 512.

    Returns
    -------
    list
        A copy of the original attention mask with ones appended for the pattern tokens, followed by zeroes for padding
        up to the specified length.
    """
    copied_list = target_list.copy()
    copied_list.extend([1]*pattern_length)
    copied_list.extend([0]*(extend_to-len(copied_list)))
    return copied_list

class PEThead(PredictionHead):
    """A prediction head for pattern-exploiting training.

    Attributes
    ----------
    config : dict
        A dictionary containing:
        - 'vocab_size': int
          Vocabulary size from the underlying model.
        - 'id2tokenid': dict
          Mapping from class IDs to lists of token IDs representing verbalizers.
        - 'id2tokenid_values': list
          List of all unique token IDs appearing in any verbalizer.
    """

    def __init__(self, model, head_name, id2tokenid, vocab_size=None, **kwargs):
        """A function to initialize the PEThead by storing configuration and building the prediction layers.

        Parameters
        ----------
        model : PreTrainedModel
            The model.
        head_name : str
            The name of the prediction head.
        id2tokenid : dict
            A mapping from class IDs to lists of token IDs representing the verbalizer for each class.
        vocab_size : int, optional
            Vocabulary size override. If None, uses `model.config.vocab_size`.
        **kwargs
            Additional arguments.

        Returns
        -------
        None.
        """
        super().__init__(head_name)
        self.config = {
            "vocab_size": model.config.vocab_size,
            "id2tokenid": {key:id2tokenid[key] for key in sorted(id2tokenid)}, # ensures sorted dict
            "id2tokenid_values": sorted(set([value for sublist in id2tokenid.values() for value in sublist])),
        }
        self.build(model)

    def build(self, model):
        """A function to build the PEThead layers.

        Parameters
        ----------
        model : PreTrainedModel
            The model.

        Returns
        -------
        None.
        """
        model_config = model.config

        # Create additional fully connected layers before the final classification layer
        pred_head = []
        pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
        pred_head.append(nn.GELU())
        pred_head.append(nn.LayerNorm(model_config.hidden_size, eps=1e-12))

        # Register the intermediate layers
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        # Final embedding layer
        self.add_module(
            str(len(pred_head)),
            nn.Linear(model_config.hidden_size, len(self.config["id2tokenid_values"]), bias=True),
        )

        # Initialize all weights
        self.apply(model._init_weights)

        # Ensure the training mode of head and model is consistent
        self.train(model.training)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        """A function to perform a forward pass through the PEThead.

        Parameters
        ----------
        outputs : tuple
            Model outputs.
        cls_output : torch.Tensor, optional
            Not used, reserved for compatibility.
        attention_mask : torch.Tensor, optional
            Not used, reserved for compatibility.
        return_dict : bool, optional
            If True, output a dictionary instead of a tuple. Default is False.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple
            If labels are provided: (loss, logits_for_loss, outputs).
            If labels are not provided: (logits_for_loss, outputs).
        """

        # Pass through all layers except the last embedding layer
        seq_outputs = outputs[0]
        for i in range(len(self) - 1):
            seq_outputs = self[i](seq_outputs)

        # Pass through an invertible adapter if available
        inv_adapter = kwargs.pop("invertible_adapter", None)
        if inv_adapter is not None:
            seq_outputs = inv_adapter(seq_outputs, rev=True)

        # Pass through the last embedding layer
        lm_logits = self[len(self) - 1](seq_outputs)

        # Initialize loss and cross-entropy loss function
        loss = None
        loss_fct = nn.CrossEntropyLoss()

        # Extract labels from kwargs if provided
        labels = kwargs.pop("labels", None)

        # Prepare mapping from verbalizer
        n_mask_token = max([len(self.config["id2tokenid"][i]) for i in range(len(self.config["id2tokenid"]))])
        id2newid = {i: z for i, z in zip(self.config["id2tokenid_values"], range(len(self.config["id2tokenid_values"])))}
        id2dim = {k: [id2newid[v1] for v1 in v] for k, v in self.config["id2tokenid"].items()}
        verbalizerid = list(id2dim.values())

        # Extract logits corresponding to masked positions
        mask_indices = kwargs.get("mask_indices1")
        logits_mask = lm_logits[range(lm_logits.shape[0]), mask_indices, :]
        logits_for_loss = logits_mask[:, [k[0] for k in verbalizerid]]
        for i in range(n_mask_token-1):
            mask_indices = kwargs.get("mask_indices"+str(i+2))
            logits_mask = lm_logits[range(lm_logits.shape[0]), mask_indices, :]
            logits_for_loss += logits_mask[:, [k[i+1] for k in verbalizerid]]

        # Compute cross-entropy loss if labels are provided
        if labels is not None:
            loss = loss_fct(logits_for_loss.view(-1, len(self.config["id2tokenid"])), labels.view(-1))

        # Build the outputs
        outputs = (logits_for_loss,) + outputs[1:]
        if loss is not None:
            outputs = (loss,) + outputs

        return outputs

def prepare_data_files(experiment, file, mapping, seed=42):
    """A function to map labels, split a dataset into training and test sets, and save the resulting files.

    Parameters
    ----------
    experiment : pathlib.Path
        Path to the experiment directory where the dataset file is located and where the output train/test CSV files
        will be saved.
    file : str or pathlib.Path
        Name or path of the input dataset CSV file relative to `experiment`.
    mapping : dict
        A dictionary mapping original labels to their new representations.
    seed : int, optional
        Random seed used for reproducible sampling when creating the test set. Default is 42.

    Returns
    -------
    None. Saves two CSV files in the `experiment` directory:
        - "train.csv": training set containing text and mapped labels, no header
        - "test.csv": test set containing text and mapped labels, no header
    """

    # Load dataset and map labels to new values
    data = pd.read_csv(experiment / file)
    data['label'] = [mapping[value] for value in data['label'].tolist()]

    # Get unique documents-year pairs and assign them to 3 year bins
    documents = data[['document', 'year']].drop_duplicates()
    documents['year_bin'] = pd.qcut(documents['year'], q=3, labels=False)

    # Calculate document lengths and assign them to 3 length bins
    document_lengths = data['document'].value_counts().rename('document_length')
    documents = documents.merge(document_lengths, left_on='document', right_index=True)
    documents['document_length_bin'] = pd.qcut(documents['document_length'], q=3, labels=False)

    # Sample test documents with stratification by year and length bins
    test_documents = (documents.groupby(['year_bin', 'document_length_bin'], group_keys=False)
                      .apply(lambda group: group.sample(n=5, random_state=seed) if len(group) >= 5 else group, include_groups=False))

    # Extract training and test datasets from the original dataset
    test_full = data[data['document'].isin(test_documents['document'])]
    test = test_full[['text', 'label']]
    train_full = data.drop(test.index)
    train = train_full[['text', 'label']]

    # Save training and test datasets
    train.to_csv(experiment / "train.csv", index=False, header=False)
    test.to_csv(experiment / "test.csv", index=False, header=False)
    logger.debug(f'Prepared data; train size: {len(train)}, test size: {len(test)}')

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

def create_experiment_data(experiment, pattern_text, verbalizer, tokenizer):
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
    dataset_test, labels, id2tokenid = data_from_csv(
        test_path, pattern, verbalizer, tokenizer, split_name='test'
    )

    # Load processed train data
    train_path = experiment / "train.csv"
    dataset_train, _, _ = data_from_csv(
        train_path, pattern, verbalizer, tokenizer, split_name='train'
    )

    # Create data for the experiment
    device = torch.device("cuda")
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

    # Make forward pass and get predicitions for all observations
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

def create_embeddings(data, api_key_file, emb_dir, batch_size=200, dimensionality=64):
    """A function to generate and save embeddings for the text in training dataset using the OpenAI API and the model
    'text-embedding-3-large'.

    Parameters
    ----------
    data : dict
        A dictionary containing the data produced by `create_experiment_data`.
    api_key_file : str or pathlib.Path
        Path to a text file containing the OpenAI API key.
    emb_dir : str or pathlib.Path
        Directory where the embeddings CSV file will be saved to.
    batch_size : int, optional
        Number of text samples to embed per API request. Default is 200.
    dimensionality : int, optional
        Desired dimensionality of the returned embeddings. Default is 64.

    Returns
    -----
    None. Saves embeddings.csv, the file containing the embeddings in rows. Columns are named `emb_0`, `emb_1`, ...,
    `emb_{dimensionality-1}`.
    """

    # Skip computation if embeddings already exist
    if os.path.exists(emb_dir / "embeddings.csv"):
        logger.debug('Embeddings already created')
        return

    # Read API key and create client
    with open(api_key_file, "r") as file:
        api_key = file.read()
    client = OpenAI(api_key=api_key)

    # Get texts from data and create embeddings
    df_train = data['train_dataset']["train"].to_pandas()
    texts = df_train['text'].tolist()
    text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    emb_list = []
    for i in tqdm(range(0, math.ceil(len(texts) / batch_size)), desc='Creating embeddings for training data'):
        response = client.embeddings.create(
            model="text-embedding-3-large",
            dimensions=dimensionality,
            input=text_batches[i],
            encoding_format="float"
        )
        emb_list = emb_list + [response.data[j].embedding for j in range(len(text_batches[i]))]

    # Save the embeddings
    emb_df = pd.DataFrame(emb_list, columns = [f"emb_{i}" for i in range(dimensionality)])
    emb_df.to_csv(emb_dir / "embeddings.csv", index = False)
    logger.debug('Created embeddings')

def select_obs_random(data, available_train_rows, n, seed=42):
    """A function for random sampling query strategy for active learning.

    Parameters
    ----------
    data : dict
        A dictionary containing the data produced by `create_experiment_data`.
    available_train_rows : list of int
        List of indices of rows in the training dataset that are available for selection.
    n : int
        Number of observations to select. Must be less than or equal to the length of `available_train_rows`.
    seed : int, optional
        Random seed for sampling. Default is 42.

    Returns
    -------
    tuple
        current_train_rows : list of int
            The selected row indices from the available training data.
        current_train_dataset : Dataset
            Subset of the training dataset corresponding to `current_train_rows`.
        available_train_rows : list of int
            Updated list of available row indices after removing `current_train_rows`.
    """

    if n > len(available_train_rows): sys.exit("The value of n is bigger than the number of available train rows")
    random.seed(seed)

    # Randomly select training rows
    current_train_rows = random.sample(available_train_rows, n)

    # Get subset for training
    current_train_dataset = data['train_dataset']["train"].select(current_train_rows)

    # Get the updated available rows indices
    available_train_rows = [row for row in available_train_rows if row not in current_train_rows]

    return current_train_rows, current_train_dataset, available_train_rows

def select_obs_uncertainty(data, available_train_rows, n, model):
    """A function for margin sampling query strategy for active learning.

    Parameters
    ----------
    data : dict
        A dictionary containing the data produced by `create_experiment_data`.
    available_train_rows : list of int
        List of indices of rows in the training dataset that are available for selection.
    n : int
        Number of observations to select. Must be less than or equal to the length of `available_train_rows`.
    model : transformers.AutoAdapterModel
        The model to be used to calculate predictions.

    Returns
    -------
    tuple
        current_train_rows : list of int
            The selected row indices from the available training data.
        current_train_dataset : Dataset
            Subset of the training dataset corresponding to `current_train_rows`.
        available_train_rows : list of int
            Updated list of available row indices after removing `current_train_rows`.
    """

    if n > len(available_train_rows): sys.exit("The value of n is bigger than the number of available train rows")

    # Put model into evaluation mode
    model.eval()

    # Create placeholder for probabilities
    probs = [-100] * len(data['input_ids_train'])

    # Predict label probabilities for the available rows of training data
    for i in tqdm(range(len(data['input_ids_train'])), desc='Predicting labels for unused training data'):
        with torch.no_grad():
            if i in available_train_rows:
                model_args = {key.replace('_train', ''): data[key][i] for key in data.keys() if key in ['input_ids_train', 'attention_mask_train'] or 'mask_indices_train' in key}
                res = model(**model_args)[0]
                probs[i] = torch.round(torch.softmax(res.cpu().detach(), dim=1).squeeze(), decimals=3).tolist()
            else:
                probs[i] = [None, None]

    # Calculate label probability differences for all observations
    prob_diffs = [abs(p[0] - p[1]) if p[0] is not None else sys.maxsize for p in probs]

    # Select training rows as the ones with the smallest probability difference
    current_train_rows = [i for i, _ in heapq.nsmallest(n, enumerate(prob_diffs), key=lambda x: x[1])]

    # Get subset for training
    current_train_dataset = data['train_dataset']["train"].select(current_train_rows)

    # Get the updated available rows indices
    available_train_rows = [row for row in available_train_rows if row not in current_train_rows]

    return current_train_rows, current_train_dataset, available_train_rows

def select_obs_diversity(data, available_train_rows, n, emb_dir, seed=42):
    """A function for k-means sampling query strategy for active learning.

    Parameters
    ----------
    data : dict
        A dictionary containing the data produced by `create_experiment_data`.
    available_train_rows : list of int
        List of indices of rows in the training dataset that are available for selection.
    n : int
        Number of observations to select. Must be less than or equal to the length of `available_train_rows`.
    emb_dir : str or pathlib.Path
        The directory where the embeddings file is stores.
    seed : int, optional
        Random seed for k-means algorithm. Default is 42.

    Returns
    -------
    tuple
        current_train_rows : list of int
            The selected row indices from the available training data.
        current_train_dataset : Dataset
            Subset of the training dataset corresponding to `current_train_rows`.
        available_train_rows : list of int
            Updated list of available row indices after removing `current_train_rows`.
    """

    if n > len(available_train_rows): sys.exit("The value of n is bigger than the number of available train rows")

    # Get embeddings from the file, embeddings are L2 normalized by default
    emb_df = pd.read_csv(emb_dir / "embeddings.csv")

    # Run k-means and get cluster ceneters
    kmeans = KMeans(
        n_clusters=n,
        init='k-means++',
        n_init=1,
        max_iter=300,
        tol=0.0001,
        algorithm='lloyd',
        random_state=seed
    )
    kmeans.fit(emb_df)
    _ = kmeans.predict(emb_df)
    centers = kmeans.cluster_centers_

    # Calculate the nearest observations to each cluster center
    emb_df_filtered = emb_df.loc[available_train_rows]
    emb_df_filtered_arr = emb_df_filtered.to_numpy()
    nearest_indices = []
    for center in centers:
        distances = np.linalg.norm(emb_df_filtered_arr - center, axis=1)
        nearest_id = np.argmin(distances)
        nearest_indices.append(emb_df_filtered.index[nearest_id])

    # Select training rows as the ones with the nearest observations to each cluster center
    current_train_rows = [int(id) for id in nearest_indices]

    # Get subset for training
    current_train_dataset = data['train_dataset']["train"].select(current_train_rows)

    # Get the updated available rows indices
    available_train_rows = [row for row in available_train_rows if row not in current_train_rows]

    return current_train_rows, current_train_dataset, available_train_rows

def select_obs(strategy, data, available_train_rows, n, **kwargs):
    """A function to select observations based on a specified active learning strategy.

    Parameters
    ----------
    strategy : str
        The selection strategy to use. Must be one of:
            - "random": select observations with random sampling
            - "uncertainty": select observations with margin sampling
            - "diversity": select observations with k-means sampling
    data : dict
        A dictionary containing the data produced by `create_experiment_data`.
    available_train_rows : list of int
        List of indices of rows in the training dataset that are available for selection.
    n : int
        Number of observations to select. Must be less than or equal to the length of `available_train_rows`.
    **kwargs : dict
        Additional keyword arguments to pass to the specific selection strategy function.

    Returns
    -------
    tuple
        current_train_rows : list of int
            The selected row indices from the available training data.
        current_train_dataset : Dataset
            Subset of the training dataset corresponding to `current_train_rows`.
        available_train_rows : list of int
            Updated list of available row indices after removing `current_train_rows`.
    """
    strategies = {
        "random": select_obs_random,
        "uncertainty": select_obs_uncertainty,
        "diversity": select_obs_diversity,
    }
    return strategies[strategy](data, available_train_rows, n, **kwargs)

def read_config(path):
    """A function to load and validate the experiment configuration from a YAML file.

    Parameters
    ----------
    path : Path or str
        Path to the directory containing 'config.yml'.

    Returns
    -------
    dict
        Dictionary containing the configuration.
    """
    with open (path / 'config.yml', 'r', encoding='utf8') as cfg:
        config = yaml.safe_load(cfg)
    logger.debug('Loaded configuration file')
    if 'api_key_file' not in config['data'].keys():
        config['data']['api_key_file'] = "api_key.txt"
    if 'output_dir' not in config['data'].keys():
        config['data']['output_dir'] = Path("/output")
    if 'emb_dir' not in config['data'].keys():
        config['data']['emb_dir'] = Path("/embeddings")
    if config['active_learning']['al_strategy'] not in ["random", "uncertainty", "diversity"]:
        sys.exit("Active learning parameters incorrectly specified in configuration file")
    if config['active_learning']['lambda_ewc'] < 0:
        sys.exit("EWC parameter lambda incorrectly specified in configuration file")
    return config

def prepare_experiment_files(experiment, config):
    """A function to prepare the necessary files and data for an experiment.

    Parameters
    ----------
    experiment : Path or str
        Path to the experiment directory where data and outputs are stored.
    config : dict
        Experiment configuration dictionary loaded from the configuration file.

    Returns
    -------
    dict
        Dictionary containing prepared datasets, labels, input IDs, attention masks, mask indices,
        and other data required for training and evaluation.
    """
    prepare_data_files(
        experiment=experiment,
        file=config['data']['dataset']['dataset_file'],
        mapping=config['data']['dataset']['label_mapping'],
        seed=54
    )
    tokenizer = AutoTokenizer.from_pretrained(config['adapter']['model'], additional_special_tokens=["<TEXT>"])
    data = create_experiment_data(
        experiment=experiment,
        pattern_text=config['data']['pattern'],
        verbalizer=config['data']['verbalizer'],
        tokenizer=tokenizer
    )
    if config['active_learning']['al_strategy'] == "diversity":
        create_embeddings(
            data=data,
            api_key_file=config['data']['api_key_file'],
            emb_dir=experiment / config['data']['emb_dir']
        )
    logger.debug(f'Prepared experiment files')
    return data

class EWCAdapterTrainer(AdapterTrainer):
    """A class to use AdapterTrainer with elastic weight consolidation for continual active learning.

    Attributes
    ----------
    old_parameters_list : list
        A list to save model parameters from previous continual active learning iterations.
    fisher_list : list
        A list to save fisher information from previous continual active learning iterations.
    lambda_ewc : float
        Weighting factor for the EWC regularization term.
    *args, **kwargs :
        Additional arguments passed to the base AdapterTrainer class.
    """

    def __init__(self, lambda_ewc, *args, **kwargs):
        """A function to initialize an EWCAdapterTrainer.

        Parameters
        ----------
        lambda_ewc : float
            Weighting factor for the EWC regularization term.
        *args, **kwargs :
            Additional arguments passed to the base AdapterTrainer class.

        Returns
        -------
        None.
        """
        super().__init__(*args, **kwargs)
        self.old_parameters_list = []
        self.fisher_list = []
        self.lambda_ewc = lambda_ewc

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        """A function to compute the training loss with EWC regularization.

        Parameters
        ----------
        model : transformers.AutoAdapterModel
            The model to train.
        inputs : dict
            A batch of inputs for the model.
        num_items_in_batch : int
            Number of items in the batch (for AdapterTrainer compatibility).
        return_outputs : bool, optional
            If True, also return the model outputs along with the loss.

        Returns
        -------
        tuple
            Computed loss including EWC regularization and model outputs if return_outputs=True.
        """

        outputs = model(**inputs)
        loss = outputs[0]
        for old_parameters, fisher in zip(self.old_parameters_list, self.fisher_list):
            loss += self.ewc_loss(model, old_parameters, fisher, self.lambda_ewc)
        return (loss, outputs) if return_outputs else loss

    def save_fisher(self, model, data, device):
        """A function to compute and save the Fisher information and parameters.

        Parameters
        ----------
        model : transformers.AutoAdapterModel
            The model to train.
        data : Dataset
            Data used to estimate the Fisher information.
        device : torch.device
            Device on which to perform computation.

        Returns
        -------
        None.
        """
        current_parameters = self.get_ewc_parameters(model)
        current_fisher = self.compute_fisher(model, data, device)
        self.old_parameters_list.append(current_parameters)
        self.fisher_list.append(current_fisher)

    def get_ewc_parameters(self, model):
        """A function to get model parameters suitable for EWC computation, that is, the ones which can be updated.

        Parameters
        ----------
        model : transformers.AutoAdapterModel
            The model to train.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their values.
        """
        parameters = {name: parameter.clone().detach() for name, parameter in model.named_parameters() if parameter.requires_grad}
        return parameters

    def compute_fisher(self, model, data, device):
        """A function to compute the Fisher information matrix for model parameters.

        Parameters
        ----------
        model : transformers.AutoAdapterModel
            The model to train.
        data : Dataset
            Dataset used to calculate the Fisher information.
        device : torch.device
            Device on which to perform computation.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their estimated Fisher information.
        """
        fisher = defaultdict(float)

        # Put model in the evaluation mode
        model.eval()

        # Calculate Fisher information
        for batch in data:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().clone() ** 2
            model.zero_grad()
        for name in fisher:
            fisher[name] /= len(data)

        return fisher

    def ewc_loss(self, model, old_parameters, fisher, lambda_ewc):
        """A function to compute the EWC loss.

        Parameters
        ----------
        model : transformers.AutoAdapterModel
            The model to train.
        old_params : dict
            Parameters from previous iterations.
        fisher : dict
            Fisher information from previous iterations.
        lambda_ewc : float
            Weighting factor for the EWC loss term.

        Returns
        -------
        torch.Tensor
            EWC loss.
        """
        loss = 0.0
        for name, parameter in model.named_parameters():
            if name in fisher:
                loss += (fisher[name] * (parameter - old_parameters[name])**2).sum()
        return lambda_ewc * loss

def run_adapter_training(experiment, config, data, adapter_name="myadapter"):
    """A function to perform run PEFT module training.

    Parameters
    ----------
    experiment : Path
        The path to the experiment directory where outputs, embeddings, and adapters will be saved.
    config : dict
        A dictionary containing configuration parameters.
    data : dict
        A dictionary containing data as produced by `prepare_experiment_files`.
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

    # Set training configuratin
    al_iterations = config['active_learning']['al_iteration_number']
    al_startegy = config['active_learning']['al_strategy']
    model.train_adapter(adapter_name)
    selection_kwargs = {}
    if al_startegy == "random":
        selection_kwargs["seed"] = 42
    elif al_startegy == "uncertainty":
        selection_kwargs["model"] = model
    elif al_startegy == "diversity":
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
                strategy=al_startegy,
                data=data,
                available_train_rows=available_train_rows,
                n=int(len(data['train_dataset']["train"]) * config['active_learning']['query_size_fraction']),
                **selection_kwargs
            )
        trainer.train_dataset = current_train_dataset

        # Run PEFT module training for current iteration
        logger.debug(f'Started training for AL iteration {al_iter}, current train size: {len(current_train_rows)}')
        training_starttime = time.time()
        trainer.train()
        training_endtime = time.time()
        if arch in ("ia3", "lora"):
            model.merge_adapter(adapter_name)
        logger.debug(f'Started computing Fisher information matrix for AL iteration {al_iter}')
        trainer.save_fisher(model, current_train_dataset, device='cuda')

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
    data = prepare_experiment_files(path, config)
    run_adapter_training(path, config, data)


if __name__ == '__main__':
    ### 2000
    ## LoRA
    # run_experiment("experiments/2000_lora_diversity_10")
    # run_experiment("experiments/2000_lora_diversity_50")
    # run_experiment("experiments/2000_lora_diversity_100")
    # run_experiment("experiments/2000_lora_diversity_500")
    # run_experiment("experiments/2000_lora_random_10")
    # run_experiment("experiments/2000_lora_random_50")
    # run_experiment("experiments/2000_lora_random_100")
    # run_experiment("experiments/2000_lora_random_500")
    # run_experiment("experiments/2000_lora_uncertainty_10")
    # run_experiment("experiments/2000_lora_uncertainty_50")
    # run_experiment("experiments/2000_lora_uncertainty_100")
    # run_experiment("experiments/2000_lora_uncertainty_500")
    ## Pfeiffer
    # run_experiment("experiments/2000_pfeiffer_diversity_10")
    # run_experiment("experiments/2000_pfeiffer_diversity_50")
    # run_experiment("experiments/2000_pfeiffer_diversity_100")
    # run_experiment("experiments/2000_pfeiffer_diversity_500")
    # run_experiment("experiments/2000_pfeiffer_random_10")
    # run_experiment("experiments/2000_pfeiffer_random_50")
    # run_experiment("experiments/2000_pfeiffer_random_100")
    # run_experiment("experiments/2000_pfeiffer_random_500")
    # run_experiment("experiments/2000_pfeiffer_uncertainty_10")
    # run_experiment("experiments/2000_pfeiffer_uncertainty_50")
    # run_experiment("experiments/2000_pfeiffer_uncertainty_100")
    # run_experiment("experiments/2000_pfeiffer_uncertainty_500")
    ## Pfeiffer Inv
    # run_experiment("experiments/2000_pfeifferinv_diversity_10")
    # run_experiment("experiments/2000_pfeifferinv_diversity_50")
    # run_experiment("experiments/2000_pfeifferinv_diversity_100")
    # run_experiment("experiments/2000_pfeifferinv_diversity_500")
    # run_experiment("experiments/2000_pfeifferinv_random_10")
    # run_experiment("experiments/2000_pfeifferinv_random_50")
    # run_experiment("experiments/2000_pfeifferinv_random_100")
    # run_experiment("experiments/2000_pfeifferinv_random_500")
    # run_experiment("experiments/2000_pfeifferinv_uncertainty_10")
    # run_experiment("experiments/2000_pfeifferinv_uncertainty_50")
    # run_experiment("experiments/2000_pfeifferinv_uncertainty_100")
    # run_experiment("experiments/2000_pfeifferinv_uncertainty_500")

    ### 1000
    ## LoRA
    # run_experiment("experiments/1000_lora_diversity_10")
    # run_experiment("experiments/1000_lora_diversity_50")
    # run_experiment("experiments/1000_lora_diversity_100")
    # run_experiment("experiments/1000_lora_diversity_500")
    # run_experiment("experiments/1000_lora_random_10")
    # run_experiment("experiments/1000_lora_random_50")
    # run_experiment("experiments/1000_lora_random_100")
    # run_experiment("experiments/1000_lora_random_500")
    # run_experiment("experiments/1000_lora_uncertainty_10")
    # run_experiment("experiments/1000_lora_uncertainty_50")
    # run_experiment("experiments/1000_lora_uncertainty_100")
    # run_experiment("experiments/1000_lora_uncertainty_500")
    ## Pfeiffer
    # run_experiment("experiments/1000_pfeiffer_diversity_10")
    # run_experiment("experiments/1000_pfeiffer_diversity_50")
    # run_experiment("experiments/1000_pfeiffer_diversity_100")
    # run_experiment("experiments/1000_pfeiffer_diversity_500")
    # run_experiment("experiments/1000_pfeiffer_random_10")
    # run_experiment("experiments/1000_pfeiffer_random_50")
    # run_experiment("experiments/1000_pfeiffer_random_100")
    # run_experiment("experiments/1000_pfeiffer_random_500")
    # run_experiment("experiments/1000_pfeiffer_uncertainty_10")
    # run_experiment("experiments/1000_pfeiffer_uncertainty_50")
    # run_experiment("experiments/1000_pfeiffer_uncertainty_100")
    # run_experiment("experiments/1000_pfeiffer_uncertainty_500")
    ## Pfeiffer Inv
    # run_experiment("experiments/1000_pfeifferinv_diversity_10")
    # run_experiment("experiments/1000_pfeifferinv_diversity_50")
    # run_experiment("experiments/1000_pfeifferinv_diversity_100")
    # run_experiment("experiments/1000_pfeifferinv_diversity_500")
    # run_experiment("experiments/1000_pfeifferinv_random_10")
    # run_experiment("experiments/1000_pfeifferinv_random_50")
    # run_experiment("experiments/1000_pfeifferinv_random_100")
    # run_experiment("experiments/1000_pfeifferinv_random_500")
    # run_experiment("experiments/1000_pfeifferinv_uncertainty_10")
    # run_experiment("experiments/1000_pfeifferinv_uncertainty_50")
    # run_experiment("experiments/1000_pfeifferinv_uncertainty_100")
    # run_experiment("experiments/1000_pfeifferinv_uncertainty_500")

    ### Baseline
    # run_experiment("experiments/2000_lora_baseline")
    # run_experiment("experiments/2000_pfeiffer_baseline")
    # run_experiment("experiments/2000_pfeifferinv_baseline")
    # run_experiment("experiments/1000_lora_baseline")
    # run_experiment("experiments/1000_pfeiffer_baseline")
    # run_experiment("experiments/1000_pfeifferinv_baseline")
    pass
