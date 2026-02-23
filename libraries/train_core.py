"""
Core training routines for linear and logistic regression via SGD.

These are intentionally simple from-scratch implementations (no sklearn)
to demonstrate that transfer learning concepts apply even to the most
basic gradient-based training loops.

Training supports warm-starting: pass w_init/b_init from a source model
to get weight transfer, or pass zeros for from-scratch training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .metrics import mse, accuracy_from_logits


def fit_linear_sgd(X, y, w_init, b_init, steps=100, lr=0.1, clip_grad=5.0):
    """
    Train a linear regression model via full-batch gradient descent.

    Minimizes MSE: L(w, b) = (1/n) Σ (xᵢᵀw + b − yᵢ)²

    Supports transfer learning via warm-starting:
      - Pass w_init=zeros for from-scratch training
      - Pass w_init=w_source for weight transfer (fine-tuning)

    Args:
        X: (n, d) feature matrix (torch tensor)
        y: (n,) target vector
        w_init: (d,) initial weights (from source model or zeros)
        b_init: (1,) initial bias
        steps: number of gradient descent iterations
        lr: learning rate (recommend 0.01 for standardized features)
        clip_grad: max gradient norm (prevents instability with
                   poor initialization or high learning rates)

    Returns:
        w: (d,) trained weights (detached)
        b: (1,) trained bias (detached)
    """
    w = nn.Parameter(w_init.clone())
    b = nn.Parameter(b_init.clone())
    opt = optim.SGD([w, b], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        yhat = X @ w + b
        loss = torch.mean((yhat - y) ** 2)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_([w, b], clip_grad)
        opt.step()
    return w.detach(), b.detach()


def fit_logistic_sgd(X, y, w_init, b_init, steps=100, lr=0.1, clip_grad=5.0):
    """
    Train a logistic regression model via full-batch gradient descent.

    Minimizes binary cross-entropy with logits:
      L(w, b) = −(1/n) Σ [yᵢ log σ(xᵢᵀw + b) + (1−yᵢ) log(1−σ(xᵢᵀw + b))]

    Supports transfer learning via warm-starting:
      - Pass w_init=zeros for from-scratch training
      - Pass w_init=w_source for weight transfer (fine-tuning)

    Args:
        X: (n, d) feature matrix (torch tensor)
        y: (n,) binary labels (0/1, torch tensor)
        w_init: (d,) initial weights (from source model or zeros)
        b_init: (1,) initial bias
        steps: number of gradient descent iterations
        lr: learning rate (recommend 0.01 for standardized features)
        clip_grad: max gradient norm (prevents instability with
                   poor initialization or high learning rates)

    Returns:
        w: (d,) trained weights (detached)
        b: (1,) trained bias (detached)
    """
    w = nn.Parameter(w_init.clone())
    b = nn.Parameter(b_init.clone())
    opt = optim.SGD([w, b], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        opt.zero_grad()
        logits = X @ w + b
        loss = loss_fn(logits, y)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_([w, b], clip_grad)
        opt.step()
    return w.detach(), b.detach()


def eval_linear(X, y, w, b):
    """Evaluate linear regression model, returning MSE."""
    yhat = X @ w + b
    return mse(yhat, y)


def eval_logistic(X, y, w, b):
    """Evaluate logistic regression model, returning accuracy."""
    logits = X @ w + b
    return accuracy_from_logits(logits, y)
