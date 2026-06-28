import yaml
import sys
from pathlib import Path
from loguru import logger

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
    copied_list.extend([1] * pattern_length)
    copied_list.extend([0] * (extend_to - len(copied_list)))
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
    if config['parameter_efficient_fine_tuning']['architecture'] not in ["adapter", "lora", "prefix_tuning", "unipelt"]:
        sys.exit("Parameter-efficient fine-tuning parameters incorrectly specified in configuration file")
    if config['active_learning']['strategy'] not in ["random", "entropy", "coreset"]:
        sys.exit("Active learning parameters incorrectly specified in configuration file")
    if config['continual_learning']['method'] not in ["der", "sd", "sds2"]:
        sys.exit("Continual learning parameters incorrectly specified in configuration file")
    return config