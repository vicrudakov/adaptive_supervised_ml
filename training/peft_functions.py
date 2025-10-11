import torch
from adapters import AdapterTrainer
from collections import defaultdict


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