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


# ---------------------------------------------------------------------------
# 1. Regularized Transfer
# ---------------------------------------------------------------------------

def regularized_transfer_linear(X_target, y_target, w_source, b_source, lam=1.0):
    """
    Closed-form regularized transfer for linear regression.

    Solves:  w* = (X'X + λI)^{-1} (X'y + λ·w_source)

    This is Ridge regression with the penalty centered on w_source
    instead of zero.  λ controls trust in the source model:
      - large λ → heavy reliance on source weights
      - small λ → effectively trains from scratch

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

    # Closed-form: w* = (X'X + λI)^{-1} (X'y + λ·w_source)
    A = XtX + lam * I
    b = Xty + lam * w_source_aug
    w_aug = torch.linalg.solve(A, b)

    return w_aug[:d], w_aug[d:d+1]


def regularized_transfer_logistic(X_target, y_target, w_source, b_source,
                                   lam=1.0, steps=100, lr=0.01):
    """
    Gradient-based regularized transfer for logistic regression.

    Minimizes: L(w) = BCE(w) + λ·||w - w_source||²

    The regularization term pulls weights toward the source model
    rather than toward zero.

    Args:
        X_target: (n, d) target features
        y_target: (n,) target labels (0/1)
        w_source: (d,) source model weights
        b_source: (1,) source model bias
        lam: regularization strength
        steps: gradient descent iterations
        lr: learning rate (typically 10x smaller than from-scratch)

    Returns:
        w_target, b_target: optimized weights and bias
    """
    w = nn.Parameter(w_source.clone())
    b = nn.Parameter(b_source.clone())
    opt = optim.SGD([w, b], lr=lr)
    bce = nn.BCEWithLogitsLoss()

    w_src_detached = w_source.detach()
    b_src_detached = b_source.detach()

    for _ in range(steps):
        opt.zero_grad()
        logits = X_target @ w + b
        data_loss = bce(logits, y_target)
        # L2 penalty pulling toward source weights
        reg_loss = lam * (torch.sum((w - w_src_detached) ** 2)
                          + torch.sum((b - b_src_detached) ** 2))
        loss = data_loss + reg_loss
        loss.backward()
        opt.step()

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

        Λ_n = Λ_0 + (1/σ²)·X'X
        μ_n = Λ_n^{-1} (Λ_0·μ_0 + (1/σ²)·X'y)

    where Λ_0 = source_precision · I (prior precision from source)
    and μ_0 = w_source (prior mean from source).

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
        noise_var: observation noise variance σ²

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
                                source_precision=1.0, steps=100, lr=0.01):
    """
    Bayesian-inspired transfer for logistic regression via Laplace approximation.

    No closed-form exists for logistic, so we use a Bayesian-inspired approach:
    1. Use source weights as the MAP prior mean
    2. Use source_precision to set a Gaussian prior N(w_source, (1/precision)*I)
    3. Optimize the MAP objective: -log p(y|X,w) - (precision/2)||w - w_source||²

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
        steps: optimization iterations
        lr: learning rate

    Returns:
        w_map: (d,) MAP estimate weights
        b_map: (1,) MAP estimate bias
    """
    w = nn.Parameter(w_source.clone())
    b = nn.Parameter(b_source.clone())
    opt = optim.SGD([w, b], lr=lr)
    bce = nn.BCEWithLogitsLoss()

    w_prior = w_source.detach()
    b_prior = b_source.detach()

    for _ in range(steps):
        opt.zero_grad()
        logits = X_target @ w + b
        nll = bce(logits, y_target)
        # Gaussian prior: -log p(w) = (precision/2) * ||w - w_source||²
        prior_loss = (source_precision / 2.0) * (
            torch.sum((w - w_prior) ** 2) + torch.sum((b - b_prior) ** 2)
        )
        loss = nll + prior_loss
        loss.backward()
        opt.step()

    return w.detach(), b.detach()


# ---------------------------------------------------------------------------
# 3. Covariance-Based Analytical Transfer
# ---------------------------------------------------------------------------

def covariance_transfer_linear(X_source, y_source, X_target, y_target,
                                eps=1e-6):
    """
    Analytical transfer using covariance structure.

    Under covariate shift (P(y|x) preserved, P(x) differs), the
    source weights can be transformed via the covariance ratio:

        w_target ≈ Σ_xx_target^{-1} · Σ_xx_source · w_source

    This method:
    1. Computes w_source via normal equation on source data
    2. Transforms using covariance ratio between domains

    Args:
        X_source, y_source: source domain data
        X_target, y_target: target domain data (can be small)
        eps: regularization for matrix inversion

    Returns:
        w_target: (d,) transferred weights
        b_target: (1,) bias adjusted for target domain
    """
    d = X_source.shape[1]

    # Source: solve normal equation w_src = (X'X)^{-1} X'y
    XsXs = X_source.T @ X_source + eps * torch.eye(d)
    Xsy = X_source.T @ y_source
    w_source = torch.linalg.solve(XsXs, Xsy)

    # Covariance matrices
    Sigma_source = X_source.T @ X_source / X_source.shape[0] + eps * torch.eye(d)
    Sigma_target = X_target.T @ X_target / X_target.shape[0] + eps * torch.eye(d)

    # Transform: w_target = Σ_target^{-1} · Σ_source · w_source
    Sigma_ratio = torch.linalg.solve(Sigma_target, Sigma_source)
    w_target = Sigma_ratio @ w_source

    # Adjust bias for target domain means
    b_target = torch.mean(y_target) - X_target.mean(dim=0) @ w_target

    return w_target, b_target.unsqueeze(0)
