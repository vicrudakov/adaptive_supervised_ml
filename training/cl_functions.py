import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from adapters import AdapterTrainer


def similarity(X, metric, kernel_width):
    """A function to compute the pairwise similarity matrix for a batch of features.

    Parameters
    ----------
    X : torch.Tensor
        The input features matrix.
    metric : str
        The distance metric to use. Must be one of 'rbf' or 'euclidean'.
    kernel_width : float
        The width parameter for the RBF kernel.

    Returns
    -------
    torch.Tensor
        The similarity matrix.
    """
    # Compute pairwise distance matrix
    S = torch.cdist(X, X, p=2)

    if metric == 'rbf':
        # Apply RBF kernel to calculate similarity
        S = S ** 2
        return torch.exp(-S / kernel_width)
    elif metric == 'euclidean':
        # Convert distance to similarity by subtracting from the maximum distance in the batch
        return torch.max(S) - S


def get_facility_location_submodular_order(model_features, model_scores, metric, b, l, kernel_width):
    """A function to select a subset of indices by maximizing a facility location submodular objective
    combined with a modular score using a greedy algorithm.

    Parameters
    ----------
    model_features : torch.Tensor
        The model features (e.g., the penultimate model layer).
    model_scores : torch.Tensor
        An array of scores (e.g., uncertainty) for each input.
    metric : str
        The metric for similarity calculation. Must be one of 'rbf' or 'euclidean'.
    b : int
        The number of items to select.
    l : float
        The trade-off parameter `lambda` balancing the facility location (diversity)
        and the modular score (e.g., uncertainty).
    kernel_width : float
        The width parameter for the RBF similarity kernel.

    Returns
    -------
    tuple
        The ordered array of selected indices.
        The unordered set of selected indices.
        The marginal gains recorded at each greedy selection step.
    """
    # Calculate similarity matrix
    model_features_sim = similarity(model_features, metric, kernel_width)

    # Manage inputs
    if torch.is_tensor(model_features_sim):
        model_features_sim = model_features_sim.cpu().numpy()
    if torch.is_tensor(model_scores):
        model_scores = model_scores.cpu().numpy()
    model_scores = np.ravel(model_scores)

    # Results placeholders
    sol_order = np.zeros(b, dtype=np.int32)
    sol_gains = np.zeros(b, dtype=np.float32)
    selected = []

    # Tracks the maximum similarity
    n = model_features_sim.shape[0]
    max_sim = np.zeros(n, dtype=np.float32)

    # Running sum of the modular component
    current_mod_sum = 0.0

    # Greedy selection loop
    for step in range(b):
        # Calculate marginal gain for the facility location component
        diff = model_features_sim - max_sim[:, None]
        gain_FL = np.maximum(diff, 0).sum(axis=0)

        # Calculate marginal gain for the modular (uncertainty) component in log space
        gain_M = np.log1p(current_mod_sum + model_scores) - np.log1p(current_mod_sum)

        # Total marginal gain
        total_gain = l * gain_M + gain_FL

        # Mask already selected items
        if len(selected) != 0:
            total_gain[selected] = -np.inf

        # Pick the item with the maximum marginal gain
        best_idx = np.argmax(total_gain)

        # Update results
        sol_order[step] = best_idx
        sol_gains[step] = total_gain[best_idx]
        selected.append(best_idx)

        # Update the maximum similarity state for the next iteration
        max_sim = np.maximum(max_sim, model_features_sim[:, best_idx])

        # Update the running sum of the modular component
        current_mod_sum += model_scores[best_idx]

    return sol_order, set(selected), sol_gains


class ReplayAdapterTrainer(AdapterTrainer):
    """A class to use AdapterTrainer for CAL with replay-based CL strategies CAL-DER, CAL-SD, CAL-SDS2.

    Attributes
    ----------
    method : str
        The replay method to use. Must be one of 'der', 'sd', or 'sds2'.
    alpha : float
        The weight coefficient for the replay loss. Is used in CAL-DER, CAL-SD, CAL-SDS2.
    beta : float
        The weight coefficient for the cross-entropy loss. Is used in CAL-DER.
    replay_size : int
        The number of samples to select for the active replay batch. In the case of CAL-SDS2 corresponds to
        the size of `S` in the paper.
    kernel_width : float
        The kernel width. Is used in CAL-SDS2, in this case corresponds to `sigma` in the paper.
    l : float
        Trade-off parameter `lambda`. Is used in CAL-SDS2.
    c : int
        The pool size to select observations from. Should be bigger than `replay_size`. Is used in CAL-SDS2,
        in this case corresponds to the size of `A` in the paper.
    buffer : list
        The buffer storing past inputs, true labels, and predicted logits.
    past_buffer_size : int
        Tracks the length of the buffer prior to the current update step.
    rng : numpy.random.Generator
        The random number generator for buffer sampling. Is set during the initialization with integer `seed`
        parameter.
    """

    def __init__(self, method, alpha, beta, replay_size, kernel_width, l, c, seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.replay_size = replay_size
        self.kernel_width = kernel_width
        self.l = l
        self.c = c

        self.buffer = []
        self.past_buffer_size = 0
        self.rng = np.random.default_rng(seed)

    def update_buffer(self, model, dataset, device):
        """A function to append the current inputs, true labels, predicted logits to the replay buffer.

        Parameters
        ----------
        model : transformers.AutoAdapterModel
            The model being trained.
        dataset : torch.utils.data.Dataset
            The current dataset for training.
        device : torch.device
            The device on which to perform computation.

        Returns
        -------
        None.
        """
        # The buffer length is locked before adding new elements
        self.past_buffer_size = len(self.buffer)

        model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator
        )

        # Store inputs, true labels, and predicted logits
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs[1]

                # CAL-DER stores raw logits, CAL-SD and CAL-SDS2 store probability distributions after softmax
                if self.method == 'der':
                    z = logits.detach()
                elif self.method in ['sd', 'sds2']:
                    z = F.softmax(logits, dim=-1).detach()

            # Store individual items
            batch_size = batch['input_ids'].size(0)
            for i in range(batch_size):
                item_x = {k: v[i].cpu().clone() for k, v in batch.items()}
                item_y = batch['labels'][i].cpu().clone()
                item_z = z[i].cpu().clone()
                self.buffer.append((item_x, item_y, item_z))

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        """A function to compute the loss for the current batch using CAL-DER, CAL-SD, or CAL-SDS2.

        Parameters
        ----------
        model : transformers.AutoAdapterModel
            The model being trained.
        inputs : dict
            The current batch of inputs.
        num_items_in_batch : int
            Number of items in the current batch (for AdapterTrainer compatibility).
        return_outputs : bool, optional
            If True, returns a tuple of `(loss, outputs)`. Default is False.

        Returns
        -------
        torch.Tensor or tuple
            The computed total loss, optionally with model outputs.
        """
        # Loss on the current batch
        outputs = model(**inputs)
        current_loss = outputs[0]

        # If buffer is empty (AL iteration 0), standard loss is returned
        if self.past_buffer_size == 0:
            return (current_loss, outputs) if return_outputs else current_loss

        device = inputs['input_ids'].device

        # Size of the random sample from the replay buffer, CAL-SDS2 pulls a larger candidate pool with parameter c
        pool_size = self.c if self.method == 'sds2' else self.replay_size
        past_buffer_size = self.past_buffer_size
        sample_size = min(pool_size, past_buffer_size)

        # Sample the pool from the replay buffer
        idx_pool = self.rng.choice(past_buffer_size, sample_size, replace=False)
        pool_tuples = [self.buffer[i] for i in idx_pool]

        # Construct the batch dictionary for the sampled items and get logits
        pool_items = [{**x, 'labels': y} for x, y, _ in pool_tuples]
        pool_batch = self.data_collator(pool_items)
        pool_batch = {k: v.to(device) for k, v in pool_batch.items()}
        pool_z = torch.stack([z for _, _, z in pool_tuples]).to(device)

        # Submodular subset selection for CAL-SDS2
        if self.method == 'sds2':
            model.eval()
            with torch.no_grad():
                # Outputs and logits calculated for pool data using current model
                current_pool_outputs = model(**pool_batch)
                current_pool_z = current_pool_outputs[1]
                current_pool_penultimate = current_pool_outputs[3]

                # Get the penultimate representations for just the masked tokens
                mask_indices = pool_batch["mask_indices1"]
                batch_size = current_pool_penultimate.shape[0]
                current_pool_penultimate = current_pool_penultimate[range(batch_size), mask_indices, :]

                # Reduce the dimensionality of the penultimate representations using PCA
                pca_dim = min(128, pool_batch["input_ids"].size(0))
                current_pool_penultimate_centered = current_pool_penultimate - current_pool_penultimate.mean(dim=0)
                U, S, V = torch.pca_lowrank(current_pool_penultimate_centered, q=pca_dim, center=False)
                current_pool_penultimate = torch.matmul(current_pool_penultimate_centered, V)

                # Calculate uncertainty as the modular score
                current_pool_probs = F.softmax(current_pool_z, dim=-1)
                current_uncertainty_dist = 1 - torch.max(current_pool_probs, dim=1)[0]
                current_uncertainty_dist = current_uncertainty_dist.cpu().numpy().astype(np.float32)

                selection_size = min(self.replay_size, sample_size)

                # Filter the candidate pool using submodular subset selection
                idx, _, _ = get_facility_location_submodular_order(current_pool_penultimate, current_uncertainty_dist, 'rbf',
                                                                   selection_size, self.l, self.kernel_width)

            model.train()

            # Get the subset
            pool_batch = {k: v[idx] for k, v in pool_batch.items()}
            pool_z = pool_z[idx]

        # Outputs and logits calculated for pool data using current model
        current_pool_outputs = model(**pool_batch)
        current_pool_z = current_pool_outputs[1]

        # Loss computation

        ce_loss = current_pool_outputs[0]

        if self.method == 'der':
            raw_squared_errors = F.mse_loss(current_pool_z, pool_z, reduction='none')
            mse_loss = raw_squared_errors.sum(dim=1).mean()

            total_loss = current_loss + self.alpha * mse_loss + self.beta * ce_loss
        elif self.method in ['sd', 'sds2']:
            current_al_size = len(self.train_dataset)
            previous_al_size = past_buffer_size

            new_coef = current_al_size / (current_al_size + previous_al_size)
            old_coef = 1 - new_coef

            kl_loss = F.kl_div(F.log_softmax(current_pool_z, dim=-1), pool_z, reduction='batchmean')
            replay_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

            total_loss = new_coef * current_loss + old_coef * replay_loss

        return (total_loss, outputs) if return_outputs else total_loss