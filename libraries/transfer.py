"""
Core transfer learning methods for linear and logistic regression.

Implements three principled approaches:
  1. Regularized transfer — Ridge regression recentered on source weights
  2. Bayesian prior transfer — source posterior as target prior
  3. Covariance-based analytical transfer — covariance ratio mapping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .train_core import _should_log


# ---------------------------------------------------------------------------
# 1. Regularized Transfer
# ---------------------------------------------------------------------------

def regularized_transfer_linear(X_target, y_target, w_source, b_source, lam=1.0):
    """
    Closed-form regularized transfer for linear regression.

    Solves:  w* = (X'X + lambda I)^{-1} (X'y + lambda w_source)

    This is Ridge regression with the penalty centered on w_source
    instead of zero.  lambda controls trust in the source model:
      - large lambda -> heavy reliance on source weights
      - small lambda -> effectively trains from scratch

    Args:
        X_target: (n, d) target features (torch tensor)
        y_target: (n,) target labels (torch tensor)
        w_source: (d,) source model weights
        b_source: (1,) source model bias
        lam: regularization strength (trust in source)

    Returns:
        w_target: (d,) optimal target weights
        b_target: (1,) target bias
    """
    n, d = X_target.shape

    # Center y by subtracting source bias contribution, then solve for w
    # Augment X with a ones column to jointly solve for bias
    ones = torch.ones(n, 1)
    X_aug = torch.cat([X_target, ones], dim=1)  # (n, d+1)
    w_source_aug = torch.cat([w_source, b_source])  # (d+1,)

    XtX = X_aug.T @ X_aug  # (d+1, d+1)
    Xty = X_aug.T @ y_target  # (d+1,)
    I = torch.eye(d + 1)

    # Closed-form: w* = (X'X + lambda I)^{-1} (X'y + lambda w_source)
    A = XtX + lam * I
    b = Xty + lam * w_source_aug
    w_aug = torch.linalg.solve(A, b)

    return w_aug[:d], w_aug[d:d+1]


def regularized_transfer_logistic(X_target, y_target, w_source, b_source,
                                   lam=1.0, epochs=10, lr=0.01, batch_size=64,
                                   verbose=False, label=None):
    """
    Gradient-based regularized transfer for logistic regression.

    Minimizes: L(w) = BCE(w) + lambda ||w - w_source||^2

    The regularization term pulls weights toward the source model
    rather than toward zero.

    Args:
        X_target: (n, d) target features
        y_target: (n,) target labels (0/1)
        w_source: (d,) source model weights
        b_source: (1,) source model bias
        lam: regularization strength
        epochs: training epochs
        lr: learning rate (typically 10x smaller than from-scratch)
        batch_size: mini-batch size (None for full-batch)
        verbose: if True, print per-epoch progress
        label: optional name printed in progress header

    Returns:
        w_target, b_target: optimized weights and bias
    """
    w = nn.Parameter(w_source.clone())
    b = nn.Parameter(b_source.clone())
    opt = optim.SGD([w, b], lr=lr)
    bce = nn.BCEWithLogitsLoss()
    n = X_target.shape[0]

    w_src_detached = w_source.detach()
    b_src_detached = b_source.detach()

    use_minibatch = batch_size is not None and 0 < batch_size < n
    n_batches = ((n + batch_size - 1) // batch_size) if use_minibatch else 1
    total_steps = epochs * n_batches

    if verbose:
        tag = f" ({label})" if label else ""
        mode = f"{n_batches} batches" if use_minibatch else "full-batch"
        print(f"      {epochs} epochs x {mode} = {total_steps} steps{tag}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        if use_minibatch:
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                logits = X_target[idx] @ w + b
                data_loss = bce(logits, y_target[idx])
                reg_loss = lam * (torch.sum((w - w_src_detached) ** 2)
                                  + torch.sum((b - b_src_detached) ** 2))
                loss = data_loss + reg_loss
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
        else:
            opt.zero_grad()
            logits = X_target @ w + b
            data_loss = bce(logits, y_target)
            reg_loss = lam * (torch.sum((w - w_src_detached) ** 2)
                              + torch.sum((b - b_src_detached) ** 2))
            loss = data_loss + reg_loss
            loss.backward()
            opt.step()
            epoch_loss = loss.item()

        if verbose and _should_log(epoch, epochs):
            avg_loss = epoch_loss / n_batches
            print(f"      epoch {epoch+1:>{len(str(epochs))}}/{epochs}  "
                  f"[{n_batches} batches]  loss={avg_loss:.4f}")

    return w.detach(), b.detach()


# ---------------------------------------------------------------------------
# 2. Bayesian Prior Transfer
# ---------------------------------------------------------------------------

def bayesian_transfer_linear(X_target, y_target, w_source, b_source,
                              source_precision=1.0, noise_var=1.0):
    """
    Bayesian transfer for linear regression.

    Uses the source model's posterior as the target model's prior.
    The posterior mean is a precision-weighted average:

        Lambda_n = Lambda_0 + (1/sigma^2) X'X
        mu_n = Lambda_n^{-1} (Lambda_0 mu_0 + (1/sigma^2) X'y)

    where Lambda_0 = source_precision I (prior precision from source)
    and mu_0 = w_source (prior mean from source).

    The model automatically balances source knowledge against new data
    based on their relative precision.  With little target data,
    the posterior stays close to the source.  With lots of target data,
    the posterior moves toward the OLS solution.

    Args:
        X_target: (n, d) target features
        y_target: (n,) target labels
        w_source: (d,) source posterior mean
        b_source: (1,) source bias
        source_precision: scalar confidence in source (higher = more trust)
        noise_var: observation noise variance sigma^2

    Returns:
        w_posterior: (d,) posterior mean weights
        b_posterior: (1,) posterior bias
    """
    n, d = X_target.shape

    # Augment to jointly solve for bias
    ones = torch.ones(n, 1)
    X_aug = torch.cat([X_target, ones], dim=1)
    w_source_aug = torch.cat([w_source, b_source])

    # Prior precision matrix (from source posterior)
    Lambda_0 = source_precision * torch.eye(d + 1)

    # Data precision
    data_precision = (1.0 / noise_var) * (X_aug.T @ X_aug)

    # Posterior precision
    Lambda_n = Lambda_0 + data_precision

    # Posterior mean
    rhs = Lambda_0 @ w_source_aug + (1.0 / noise_var) * (X_aug.T @ y_target)
    w_posterior_aug = torch.linalg.solve(Lambda_n, rhs)

    # Also return posterior precision for downstream use
    return w_posterior_aug[:d], w_posterior_aug[d:d+1]


def bayesian_posterior_precision(X_target, source_precision=1.0, noise_var=1.0):
    """
    Compute the posterior precision matrix (useful for chaining transfers).

    Returns:
        Lambda_n: (d, d) posterior precision matrix
    """
    n, d = X_target.shape
    Lambda_0 = source_precision * torch.eye(d)
    data_precision = (1.0 / noise_var) * (X_target.T @ X_target)
    return Lambda_0 + data_precision


def bayesian_transfer_logistic(X_target, y_target, w_source, b_source,
                                source_precision=1.0, epochs=10, lr=0.01,
                                batch_size=64, verbose=False, label=None):
    """
    Bayesian-inspired transfer for logistic regression via Laplace approximation.

    No closed-form exists for logistic, so we use a Bayesian-inspired approach:
    1. Use source weights as the MAP prior mean
    2. Use source_precision to set a Gaussian prior N(w_source, (1/precision)*I)
    3. Optimize the MAP objective: -log p(y|X,w) - (precision/2)||w - w_source||^2

    This is equivalent to regularized transfer but framed as MAP estimation
    under a Gaussian prior centered on the source posterior.  The precision
    parameter encodes how concentrated the source posterior is — higher
    precision means more confident source model.

    Args:
        X_target: (n, d) target features
        y_target: (n,) target labels (0/1)
        w_source: (d,) source posterior mean
        b_source: (1,) source bias
        source_precision: prior precision (higher = more trust in source)
        epochs: training epochs
        lr: learning rate
        batch_size: mini-batch size (None for full-batch)
        verbose: if True, print per-epoch progress
        label: optional name printed in progress header

    Returns:
        w_map: (d,) MAP estimate weights
        b_map: (1,) MAP estimate bias
    """
    w = nn.Parameter(w_source.clone())
    b = nn.Parameter(b_source.clone())
    opt = optim.SGD([w, b], lr=lr)
    bce = nn.BCEWithLogitsLoss()
    n = X_target.shape[0]

    w_prior = w_source.detach()
    b_prior = b_source.detach()

    use_minibatch = batch_size is not None and 0 < batch_size < n
    n_batches = ((n + batch_size - 1) // batch_size) if use_minibatch else 1
    total_steps = epochs * n_batches

    if verbose:
        tag = f" ({label})" if label else ""
        mode = f"{n_batches} batches" if use_minibatch else "full-batch"
        print(f"      {epochs} epochs x {mode} = {total_steps} steps{tag}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        if use_minibatch:
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                logits = X_target[idx] @ w + b
                nll = bce(logits, y_target[idx])
                prior_loss = (source_precision / 2.0) * (
                    torch.sum((w - w_prior) ** 2) + torch.sum((b - b_prior) ** 2)
                )
                loss = nll + prior_loss
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
        else:
            opt.zero_grad()
            logits = X_target @ w + b
            nll = bce(logits, y_target)
            prior_loss = (source_precision / 2.0) * (
                torch.sum((w - w_prior) ** 2) + torch.sum((b - b_prior) ** 2)
            )
            loss = nll + prior_loss
            loss.backward()
            opt.step()
            epoch_loss = loss.item()

        if verbose and _should_log(epoch, epochs):
            avg_loss = epoch_loss / n_batches
            print(f"      epoch {epoch+1:>{len(str(epochs))}}/{epochs}  "
                  f"[{n_batches} batches]  loss={avg_loss:.4f}")

    return w.detach(), b.detach()


# ---------------------------------------------------------------------------
# 3. Covariance-Based Analytical Transfer
# ---------------------------------------------------------------------------

def covariance_transfer_linear(X_source, y_source, X_target, y_target,
                                blend=0.5, eps=1e-4):
    """
    Analytical transfer using covariance alignment with OLS blending.

    Under pure covariate shift (P(y|x) preserved, P(x) differs), source
    weights can be transformed via the covariance ratio:

        w_cov = Sigma_target^{-1} Sigma_source w_source

    In practice the assumption rarely holds perfectly, so this method
    blends the covariance-corrected source weights with a direct OLS
    solution on target data:

        w_final = alpha w_cov + (1 - alpha) w_ols_target

    If the covariance correction produces weights with extreme norms
    (indicating violated assumptions), the method automatically reduces
    trust in the correction and leans toward the target OLS solution.

    Args:
        X_source, y_source: source domain data (torch tensors)
        X_target, y_target: target domain data (can be small)
        blend: base blending weight for covariance-corrected source.
               0.0 = pure target OLS, 1.0 = pure covariance transfer.
               Default 0.5 = equal blend.
        eps: Tikhonov regularization for matrix inversion

    Returns:
        w_target: (d,) transferred weights
        b_target: (1,) bias adjusted for target domain
    """
    d = X_source.shape[1]
    I = torch.eye(d)
    n_s, n_t = X_source.shape[0], X_target.shape[0]

    # Adaptive regularization: when n_target is small relative to d,
    # covariance estimates are noisy and need stronger regularization.
    effective_eps = eps
    if n_t < 5 * d:
        effective_eps = max(eps, 0.1 * d / max(n_t, 1))

    reg = effective_eps * I

    # Helper: OLS with proper bias via augmented matrix
    def _ols_with_bias(X, y, lam=effective_eps):
        n = X.shape[0]
        ones = torch.ones(n, 1)
        X_aug = torch.cat([X, ones], dim=1)
        I_aug = torch.eye(d + 1) * lam
        w_aug = torch.linalg.solve(X_aug.T @ X_aug + I_aug, X_aug.T @ y)
        return w_aug[:d], w_aug[d:d+1]

    # 1. Source OLS (weights only, for covariance transform)
    w_source = torch.linalg.solve(
        X_source.T @ X_source + reg, X_source.T @ y_source
    )

    # 2. Target OLS with proper bias handling
    w_ols_target, b_ols_target = _ols_with_bias(X_target, y_target)

    # 3. Covariance ratio transform: w_cov = Sigma_t^{-1} Sigma_s w_src
    Sigma_s = X_source.T @ X_source / n_s + reg
    Sigma_t = X_target.T @ X_target / n_t + reg
    w_cov = torch.linalg.solve(Sigma_t, Sigma_s @ w_source)

    # 4. Adaptive blending — detect when covariance shift assumption fails
    #    Two checks: (a) norm ratio, (b) cosine similarity of directions.
    norm_cov = torch.norm(w_cov).item()
    norm_ols = torch.norm(w_ols_target).item() + 1e-8
    norm_ratio = norm_cov / norm_ols

    # Cosine similarity: do the two solutions point the same direction?
    cos_sim = (torch.dot(w_cov, w_ols_target)
               / (torch.norm(w_cov) * torch.norm(w_ols_target) + 1e-8)).item()

    if norm_ratio > 10.0 or norm_ratio < 0.1 or cos_sim < 0.0:
        # Extreme divergence or opposite directions — covariance
        # transfer is unreliable, return pure target OLS (safest)
        effective_blend = 0.0
    elif norm_ratio > 3.0 or norm_ratio < 0.33 or cos_sim < 0.5:
        # Moderate divergence — small covariance contribution
        effective_blend = blend * 0.15
    else:
        effective_blend = blend

    w_target = effective_blend * w_cov + (1 - effective_blend) * w_ols_target
    b_target = b_ols_target  # always use properly estimated bias

    # 5. Safety net: if blended weights are worse than pure target OLS
    #    on training data, fall back entirely to the OLS solution.
    pred_blend = X_target @ w_target + b_target
    pred_ols = X_target @ w_ols_target + b_ols_target
    mse_blend = torch.mean((pred_blend - y_target) ** 2).item()
    mse_ols = torch.mean((pred_ols - y_target) ** 2).item()
    if mse_blend > mse_ols * 1.01:  # >1% worse -> fall back
        w_target = w_ols_target
        b_target = b_ols_target

    return w_target, b_target
