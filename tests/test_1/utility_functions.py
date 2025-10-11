import math
import os
import yaml
import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm
from openai import OpenAI

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