import random
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator


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

def select_obs_entropy(data, available_train_rows, n, model, batch_size):
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
    batch_size
        The size of the batch for computation.

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

    # Set dataloader for unused data
    dataloader = DataLoader(
        data['train_dataset']["train"].select(available_train_rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=1,
        pin_memory=True
    )

    # Create placeholder for entropies
    entropies = []

    # Predict label probabilities and calculate entropy for the available rows
    for batch in tqdm(dataloader, desc='Calculating entropy for training data'):
        with torch.no_grad():
            batch_inputs = {k: v.to(model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask'] or 'mask_indices' in k}
            res = model(**batch_inputs)[0]

            # Get probabilities
            probs = torch.softmax(res, dim=1).cpu().numpy()
            probs = np.clip(probs, a_min=1e-6, a_max=None)

            # Calculate entropy
            batch_entropies = np.sum(-probs * np.log(probs), axis=1)
            entropies.extend(batch_entropies.tolist())

    # Select training rows as the ones with the highest entropy
    entropies_arr = np.array(entropies)
    relative_top_n_indices = np.argpartition(entropies_arr, -n)[-n:].tolist()
    current_train_rows = [available_train_rows[i] for i in relative_top_n_indices]

    # Get subset for training
    current_train_dataset = data['train_dataset']["train"].select(current_train_rows)

    # Get the updated available rows indices
    available_train_rows = [row for row in available_train_rows if row not in current_train_rows]

    return current_train_rows, current_train_dataset, available_train_rows

def select_obs_coreset(data, available_train_rows, n, model, batch_size):
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
    batch_size
        The size of the batch for computation.

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

    # Set dataloader for all data
    dataloader = DataLoader(
        data['train_dataset']["train"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=1,
        pin_memory=True
    )

    # Create placeholder for penultimate representations
    representations = []

    # Extract the penultimate representations for all training data
    for batch in tqdm(dataloader, desc='Extracting representations for training data'):
        with torch.no_grad():
            batch_inputs = {k: v.to(model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask'] or 'mask_indices' in k}
            outputs = model(**batch_inputs)
            penultimate = outputs[2]

            # Get the penultimate representations for just the masked tokens
            mask_indices = batch_inputs["mask_indices1"]
            penultimate = penultimate[torch.arange(penultimate.shape[0]), mask_indices, :].squeeze().cpu()
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