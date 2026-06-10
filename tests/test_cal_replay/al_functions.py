import random
import torch
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import heapq

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

    # Run k-means and get cluster centers
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