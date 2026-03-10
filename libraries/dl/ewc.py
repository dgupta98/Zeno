"""
Elastic Weight Consolidation (EWC) for Bayesian transfer in deep networks.

Extends Zeno's Bayesian prior transfer to nn.Module-based models.
EWC (Kirkpatrick et al., 2017) adds a quadratic penalty that keeps
weights close to the source model, weighted by parameter importance:

    L_total = L_target + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

where F_i is the diagonal Fisher Information (importance weight) from
the source task and theta*_i are optimal source parameters.

This is precisely a Bayesian update: using the source task's
Laplace-approximate posterior as a prior for the target task.
The Fisher diagonal is computed by averaging squared gradients
over source data — the same computation as Task2Vec.

Online EWC (Schwarz et al., 2018) maintains a running average
F_tilde = gamma * F_tilde_prev + F_current to avoid linear memory
growth with task count.
"""

import torch
import torch.nn as nn


def compute_fisher_diagonal(model, dataloader, criterion, device="cpu"):
    """
    Compute diagonal Fisher Information Matrix for all parameters.

    The Fisher diagonal measures parameter importance for the source task:
        F_i = E[(d log p(y|x,theta) / d theta_i)^2]

    Approximated by averaging squared gradients over the dataset.
    This computation is shared with Task2Vec embeddings.

    Args:
        model: trained nn.Module (source model)
        dataloader: source task DataLoader
        criterion: loss function (e.g., nn.CrossEntropyLoss)
        device: computation device

    Returns:
        fisher: dict mapping param_name -> diagonal Fisher tensor
    """
    model.eval()
    model.to(device)

    # Initialize Fisher accumulators
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)

    n_samples = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0].to(device), batch[1].to(device)
        else:
            inputs, targets = batch.to(device), None

        model.zero_grad()
        outputs = model(inputs)
        if targets is not None:
            loss = criterion(outputs, targets)
        else:
            loss = criterion(outputs)
        loss.backward()

        batch_size = inputs.size(0)
        n_samples += batch_size

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Accumulate squared gradients (scaled by batch size)
                fisher[name] += (param.grad.data ** 2) * batch_size

    # Average over all samples
    for name in fisher:
        fisher[name] /= max(n_samples, 1)

    return fisher


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation penalty term.

    Computes: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    where F_i are Fisher diagonals from the source task and
    theta*_i are the source model's optimal parameters.

    This penalty prevents catastrophic forgetting by keeping
    important parameters (high Fisher) close to their source values
    while allowing unimportant parameters to adapt freely.

    Usage:
        fisher = compute_fisher_diagonal(source_model, source_loader, criterion)
        ewc = EWCLoss(source_model, fisher, lambda_=1000)

        for epoch in range(epochs):
            for x, y in target_loader:
                loss = criterion(model(x), y) + ewc(model)
                loss.backward()
                optimizer.step()
    """

    def __init__(self, source_model, fisher_diag, lambda_=1000.0):
        """
        Args:
            source_model: trained source nn.Module (for theta* reference)
            fisher_diag: dict from compute_fisher_diagonal()
            lambda_: EWC penalty strength (higher = more conservative transfer)
        """
        super().__init__()
        self.lambda_ = lambda_

        # Store source parameters and Fisher diagonals as buffers
        # (non-trainable, move with model to device)
        self._param_names = []
        for name, param in source_model.named_parameters():
            if name in fisher_diag:
                safe_name = name.replace(".", "_")
                self._param_names.append((name, safe_name))
                self.register_buffer(
                    f"source_{safe_name}", param.data.clone()
                )
                self.register_buffer(
                    f"fisher_{safe_name}", fisher_diag[name].clone()
                )

    def forward(self, model):
        """
        Compute EWC penalty for the current model parameters.

        Args:
            model: the model being fine-tuned (same architecture as source)

        Returns:
            penalty: scalar tensor, the EWC regularization loss
        """
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        param_dict = dict(model.named_parameters())

        for orig_name, safe_name in self._param_names:
            if orig_name not in param_dict:
                continue
            param = param_dict[orig_name]
            source_param = getattr(self, f"source_{safe_name}")
            fisher = getattr(self, f"fisher_{safe_name}")

            diff = param - source_param
            penalty = penalty + (fisher * diff ** 2).sum()

        return (self.lambda_ / 2.0) * penalty


def online_ewc_update(old_fisher, new_fisher, gamma=0.9):
    """
    Online EWC: running average of Fisher diagonals across tasks.

    Avoids linear memory growth with task count:
        F_tilde = gamma * F_tilde_prev + F_current

    Args:
        old_fisher: dict of accumulated Fisher diagonals (modified in-place)
        new_fisher: dict of Fisher diagonals from the latest task
        gamma: decay factor for old Fisher (default 0.9)

    Returns:
        updated_fisher: dict of blended Fisher diagonals (same as old_fisher)
    """
    for name in new_fisher:
        if name in old_fisher:
            old_fisher[name] = gamma * old_fisher[name] + new_fisher[name]
        else:
            old_fisher[name] = new_fisher[name].clone()
    return old_fisher
