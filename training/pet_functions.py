from adapters import PredictionHead
import torch.nn as nn

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