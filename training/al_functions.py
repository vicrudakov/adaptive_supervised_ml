import random
import torch
import sys
import numpy as np
from tqdm import tqdm

def select_obs_random(data, available_train_rows, n, seed):
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
        Random seed for sampling.

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

def select_obs_entropy(data, available_train_rows, n, model):
    """A function for maximum entropy sampling query strategy for active learning.

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

    # Create placeholder for entropies
    entropies = [-np.inf] * len(data['input_ids_train'])

    # Predict label probabilities and calculate entropy for the available rows
    for i in tqdm(range(len(data['input_ids_train'])), desc='Calculating entropy for training data'):
        with torch.no_grad():
            if i in available_train_rows:
                model_args = {key.replace('_train', ''): data[key][i] for key in data.keys()
                              if key in ['input_ids_train', 'attention_mask_train'] or 'mask_indices_train' in key}
                res = model(**model_args)[0]

                # Get probabilities
                probs = torch.softmax(res.cpu().detach(), dim=1).squeeze().numpy()
                probs = np.clip(probs, a_min=1e-6, a_max=None)

                # Calculate entropy
                entropy = np.sum(-probs * np.log(probs))
                entropies[i] = entropy

    # Select training rows as the ones with the highest entropy
    entropies_arr = np.array(entropies)
    current_train_rows = np.argpartition(entropies_arr, -n)[-n:].tolist()

    # Get subset for training
    current_train_dataset = data['train_dataset']["train"].select(current_train_rows)

    # Get the updated available rows indices
    available_train_rows = [row for row in available_train_rows if row not in current_train_rows]

    return current_train_rows, current_train_dataset, available_train_rows

def select_obs_coreset(data, available_train_rows, n, model):
    """A function for core-set sampling query strategy for active learning.

    Parameters
    ----------
    data : dict
        A dictionary containing the data produced by `create_experiment_data`.
    available_train_rows : list of int
        List of indices of rows in the training dataset that are available for selection.
    n : int
        Number of observations to select. Must be less than or equal to the length of `available_train_rows`.
    model : transformers.AutoAdapterModel
        The model to be used to get representations for the data.

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

    # Rows that have already been used for training
    unavailable_train_rows = [i for i in range(len(data['input_ids_train'])) if i not in available_train_rows]

    # Extract the penultimate representations for all training data
    representations = []
    for i in tqdm(range(len(data['input_ids_train'])), desc='Extracting representations for training data'):
        with torch.no_grad():
            model_args = {key.replace('_train', ''): data[key][i] for key in data.keys() if
                          key in ['input_ids_train', 'attention_mask_train'] or 'mask_indices_train' in key}
            outputs = model(**model_args)
            penultimate = outputs[2]

            # Get the penultimate representations for just the masked tokens
            mask_index = model_args["mask_indices1"]
            penultimate = penultimate[0, mask_index, :].squeeze().cpu()
            representations.append(penultimate)

    # Create tensor for the penultimate representations
    representations = torch.stack(representations)

    # Separate representations into unlabeled / available and labeled / used pools
    unlabeled_representations = representations[available_train_rows]
    labeled_representations = representations[unavailable_train_rows]

    # k-Center-Greedy

    # Calculate pairwise distances and find the minimum distance to any labeled observation
    dist_centers = torch.cdist(unlabeled_representations, labeled_representations, p=2)
    min_dist = torch.min(dist_centers, dim=1)[0]

    # Iteratively choose the centers
    centers_idx = []
    for _ in range(n):
        # Select the index of the unlabeled observation furthest from the labeled set
        i = torch.argmax(min_dist).item()
        centers_idx.append(i)

        # Calculate distances from all unlabeled observations to the newly selected center
        new_center = unlabeled_representations[i].unsqueeze(0)
        dist_new_center = torch.cdist(unlabeled_representations, new_center)

        # Update minimum distance array to track the shortest distance to the expanded labeled set
        min_dist = torch.minimum(min_dist, dist_new_center.squeeze(1))

    # Select train rows as the centers chosen by the algorithm
    current_train_rows = [available_train_rows[i] for i in centers_idx]

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
            - "entropy": select observations with maximum entropy sampling
            - "coreset": select observations with core-set sampling
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
        "entropy": select_obs_entropy,
        "coreset": select_obs_coreset,
    }
    return strategies[strategy](data, available_train_rows, n, **kwargs)