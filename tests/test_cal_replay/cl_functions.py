import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from adapters import AdapterTrainer


def similarity(X, metric, kernel_width):
    S = torch.cdist(X, X, p=2)
    if metric == 'rbf':
        S = S ** 2
        return torch.exp(-S / kernel_width)
    elif metric == 'euclidean':
        return torch.max(S) - S

def get_facility_location_submodular_order(logits, model_score, metric, b, l, kernel_width):
    logits_sim = similarity(logits, metric, kernel_width)

    if torch.is_tensor(logits_sim):
        logits_sim = logits_sim.cpu().numpy()
    if torch.is_tensor(model_score):
        model_score = model_score.cpu().numpy()

    model_score = np.ravel(model_score)
    n = logits_sim.shape[0]

    l = 0.0 if l is None else float(l)

    sol_order = np.zeros(b, dtype=np.int32)
    sol_gains = np.zeros(b, dtype=np.float32)
    selected = []

    max_sim = np.zeros(n, dtype=np.float32)

    current_mod_sum = 0.0

    for step in range(b):
        diff = logits_sim - max_sim[:, None]
        gain_FL = np.maximum(diff, 0).sum(axis=0)

        gain_M = np.log1p(current_mod_sum + model_score) - np.log1p(current_mod_sum)

        total_gain = l * gain_M + gain_FL

        if selected:
            total_gain[selected] = -np.inf

        best_idx = np.argmax(total_gain)

        sol_order[step] = best_idx
        sol_gains[step] = total_gain[best_idx]
        selected.append(best_idx)

        max_sim = np.maximum(max_sim, logits_sim[:, best_idx])

        current_mod_sum += model_score[best_idx]

    return sol_order, set(selected), sol_gains

class ReplayAdapterTrainer(AdapterTrainer):
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
        # The buffer length is locked before adding new elements
        self.past_buffer_size = len(self.buffer)

        model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator
        )

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs[1]

                if self.method == 'der':
                    z = logits.detach()
                elif self.method in ['sd', 'sds2']:
                    z = F.softmax(logits, dim=-1).detach()

            batch_size = batch['input_ids'].size(0)
            for i in range(batch_size):
                item_x = {k: v[i].cpu().clone() for k, v in batch.items()}
                item_y = batch['labels'][i].cpu().clone() if 'labels' in batch else None
                item_z = z[i].cpu().clone()
                self.buffer.append((item_x, item_y, item_z))

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        outputs = model(**inputs)
        current_loss = outputs[0]

        if self.past_buffer_size == 0:
            return (current_loss, outputs) if return_outputs else current_loss

        device = inputs['input_ids'].device

        pool_size = self.c if self.method == 'sds2' else self.replay_size
        past_buffer_size = self.past_buffer_size
        sample_size = min(pool_size, past_buffer_size)

        idx_pool = self.rng.choice(past_buffer_size, sample_size, replace=False)
        pool_tuples = [self.buffer[i] for i in idx_pool]
        pool_items = []
        for x, y, _ in pool_tuples:
            item = x.copy()
            if y is not None:
                item['labels'] = y
            pool_items.append(item)
        pool_z = torch.stack([tup[2] for tup in pool_tuples]).to(device)

        pool_batch = self.data_collator(pool_items)
        pool_batch = {k: v.to(device) for k, v in pool_batch.items()}
        pool_labels = pool_batch['labels']

        if self.method == 'sds2':
            # For further manual reset because of the PEFT
            # module_states = {name: module.training for name, module in model.named_modules()}

            model.eval()
            with torch.no_grad():
                # Outputs and logits (z) calculated for pool data using current model
                current_pool_outputs = model(**pool_batch)
                current_pool_z = current_pool_outputs[1] # TODO diff with paper: not penultimate state

                current_pool_probs = F.softmax(current_pool_z, dim=-1)
                current_uncertainty_dist = 1 - torch.max(current_pool_probs, dim=1)[0] # TODO diff with paper: not entropy (mustn't be actually)
                current_uncertainty_dist = current_uncertainty_dist.cpu().numpy().astype(np.float32)

                selection_size = min(self.replay_size, sample_size)
                idx, _, _ = get_facility_location_submodular_order(current_pool_z, current_uncertainty_dist, 'rbf',
                                                                   selection_size, self.l, self.kernel_width)
            # Manual reset because of the PEFT
            # for name, module in model.named_modules():
            #     module.training = module_states[name]

            model.train()

            pool_batch = {k: v[idx] for k, v in pool_batch.items()}
            pool_labels = pool_labels[idx]
            pool_z = pool_z[idx]

        # Outputs and logits (z) calculated for pool data using current model
        current_pool_outputs = model(**pool_batch)
        current_pool_z = current_pool_outputs[1]

        ce_loss = current_pool_outputs[0]
        # loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # ce_loss = loss_fct(current_pool_z, pool_labels) # the same as current_pool_outputs[0]

        if self.method == 'der':
            # mse_loss = F.mse_loss(current_pool_z, pool_z) # also scales with class number, is false

            raw_squared_errors = F.mse_loss(current_pool_z, pool_z, reduction='none')
            mse_loss = raw_squared_errors.sum(dim=1).mean()

            total_loss = current_loss + self.alpha * mse_loss + self.beta * ce_loss
        elif self.method in ['sd', 'sds2']:
            current_al_size = len(self.train_dataset)
            previous_al_size = past_buffer_size

            new_coef = current_al_size / (current_al_size + previous_al_size)
            old_coef = 1 - new_coef # old_coef = previous_al_size / (current_al_size + previous_al_size)

            kl_loss = F.kl_div(F.log_softmax(current_pool_z, dim=-1), pool_z, reduction='batchmean')
            replay_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

            total_loss = new_coef * current_loss + old_coef * replay_loss

        return (total_loss, outputs) if return_outputs else total_loss