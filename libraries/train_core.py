"""
Core training routines for linear and logistic regression via SGD.

These are intentionally simple from-scratch implementations (no sklearn)
to demonstrate that transfer learning concepts apply even to the most
basic gradient-based training loops.

Training supports warm-starting: pass w_init/b_init from a source model
to get weight transfer, or pass zeros for from-scratch training.

Supports both full-batch and mini-batch SGD.  Mini-batch (the default,
batch_size=64) is realistic — it's what real-world training looks like.
Each "epoch" shuffles data and iterates through all batches.

When verbose=True, training prints per-epoch progress:
  epoch  1/30  [12 batches]  loss=0.8234
  epoch 10/30  [12 batches]  loss=0.3456
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from .metrics import mse, accuracy_from_logits


def _should_log(epoch, total_epochs):
    """Decide whether to print progress for this epoch.

    Logs epoch 1, the last epoch, and ~8 evenly-spaced epochs in between
    so that even long runs produce compact output.
    """
    if total_epochs <= 10:
        return True                         # short runs: log every epoch
    if epoch == 0 or epoch == total_epochs - 1:
        return True                         # always log first and last
    step = max(1, total_epochs // 8)
    return (epoch + 1) % step == 0


def fit_linear_sgd(X, y, w_init, b_init, epochs=10, lr=0.01,
                   clip_grad=5.0, batch_size=64, verbose=False, label=None,
                   weight_decay=0.0):
    """
    Train a linear regression model via mini-batch SGD.

    Minimizes MSE: L(w, b) = (1/n) Sigma (x_i^T w + b - y_i)^2

    Supports transfer learning via warm-starting:
      - Pass w_init=zeros for from-scratch training
      - Pass w_init=w_source for weight transfer (fine-tuning)

    Args:
        X: (n, d) feature matrix (torch tensor)
        y: (n,) target vector
        w_init: (d,) initial weights (from source model or zeros)
        b_init: (1,) initial bias
        epochs: number of full passes through the data
        lr: learning rate (recommend 0.01 for standardized features)
        clip_grad: max gradient norm (prevents instability with
                   poor initialization or high learning rates)
        batch_size: mini-batch size (default 64). Set to None or -1
                    for full-batch gradient descent.
        verbose: if True, print per-epoch progress (loss)
        label: optional name printed in progress header (e.g. "Source")
        weight_decay: L2 regularization strength (default 0). Helps
                      prevent overfitting on small datasets.

    Returns:
        w: (d,) trained weights (detached)
        b: (1,) trained bias (detached)
    """
    w = nn.Parameter(w_init.clone())
    b = nn.Parameter(b_init.clone())
    opt = optim.SGD([w, b], lr=lr, weight_decay=weight_decay)
    n = X.shape[0]

    # Full-batch mode
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        n_batches = 1
        if verbose:
            tag = f" ({label})" if label else ""
            print(f"      {epochs} epochs x 1 batch (full-batch){tag}")
        for ep in range(epochs):
            opt.zero_grad()
            yhat = X @ w + b
            loss = torch.mean((yhat - y) ** 2)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([w, b], clip_grad)
            opt.step()
            if verbose and _should_log(ep, epochs):
                print(f"      epoch {ep+1:>{len(str(epochs))}}/{epochs}  "
                      f"loss={loss.item():.4f}")
        return w.detach(), b.detach()

    # Mini-batch SGD
    n_batches = (n + batch_size - 1) // batch_size
    total_steps = epochs * n_batches

    if verbose:
        tag = f" ({label})" if label else ""
        print(f"      {epochs} epochs x {n_batches} batches "
              f"= {total_steps} steps{tag}")

    for ep in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            X_batch = X[idx]
            y_batch = y[idx]

            opt.zero_grad()
            yhat = X_batch @ w + b
            loss = torch.mean((yhat - y_batch) ** 2)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([w, b], clip_grad)
            opt.step()
            epoch_loss += loss.item()

        if verbose and _should_log(ep, epochs):
            avg_loss = epoch_loss / n_batches
            print(f"      epoch {ep+1:>{len(str(epochs))}}/{epochs}  "
                  f"[{n_batches} batches]  loss={avg_loss:.4f}")

    return w.detach(), b.detach()


def fit_logistic_sgd(X, y, w_init, b_init, epochs=10, lr=0.01,
                     clip_grad=5.0, batch_size=64, verbose=False, label=None,
                     weight_decay=0.0):
    """
    Train a logistic regression model via mini-batch SGD.

    Minimizes binary cross-entropy with logits:
      L(w, b) = -(1/n) Sigma [y_i log sigma(x_i^T w + b) + (1-y_i) log(1-sigma(x_i^T w + b))]

    Supports transfer learning via warm-starting:
      - Pass w_init=zeros for from-scratch training
      - Pass w_init=w_source for weight transfer (fine-tuning)

    Args:
        X: (n, d) feature matrix (torch tensor)
        y: (n,) binary labels (0/1, torch tensor)
        w_init: (d,) initial weights (from source model or zeros)
        b_init: (1,) initial bias
        epochs: number of full passes through the data
        lr: learning rate (recommend 0.01 for standardized features)
        clip_grad: max gradient norm (prevents instability with
                   poor initialization or high learning rates)
        batch_size: mini-batch size (default 64). Set to None or -1
                    for full-batch gradient descent.
        verbose: if True, print per-epoch progress (loss)
        label: optional name printed in progress header (e.g. "Source")
        weight_decay: L2 regularization strength (default 0). Helps
                      prevent overfitting on small datasets.

    Returns:
        w: (d,) trained weights (detached)
        b: (1,) trained bias (detached)
    """
    w = nn.Parameter(w_init.clone())
    b = nn.Parameter(b_init.clone())
    opt = optim.SGD([w, b], lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    n = X.shape[0]

    # Full-batch mode
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        if verbose:
            tag = f" ({label})" if label else ""
            print(f"      {epochs} epochs x 1 batch (full-batch){tag}")
        for ep in range(epochs):
            opt.zero_grad()
            logits = X @ w + b
            loss = loss_fn(logits, y)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([w, b], clip_grad)
            opt.step()
            if verbose and _should_log(ep, epochs):
                print(f"      epoch {ep+1:>{len(str(epochs))}}/{epochs}  "
                      f"loss={loss.item():.4f}")
        return w.detach(), b.detach()

    # Mini-batch SGD
    n_batches = (n + batch_size - 1) // batch_size
    total_steps = epochs * n_batches

    if verbose:
        tag = f" ({label})" if label else ""
        print(f"      {epochs} epochs x {n_batches} batches "
              f"= {total_steps} steps{tag}")

    for ep in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            X_batch = X[idx]
            y_batch = y[idx]

            opt.zero_grad()
            logits = X_batch @ w + b
            loss = loss_fn(logits, y_batch)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([w, b], clip_grad)
            opt.step()
            epoch_loss += loss.item()

        if verbose and _should_log(ep, epochs):
            avg_loss = epoch_loss / n_batches
            print(f"      epoch {ep+1:>{len(str(epochs))}}/{epochs}  "
                  f"[{n_batches} batches]  loss={avg_loss:.4f}")

    return w.detach(), b.detach()


def eval_linear(X, y, w, b):
    """Evaluate linear regression model, returning MSE."""
    yhat = X @ w + b
    return mse(yhat, y)


def eval_logistic(X, y, w, b):
    """Evaluate logistic regression model, returning accuracy."""
    logits = X @ w + b
    return accuracy_from_logits(logits, y)
