"""
Negative transfer detection and prevention.

Implements three detection methods:
  1. Maximum Mean Discrepancy (MMD) — distributional distance in kernel space
  2. Proxy A-distance (PAD) — domain classifier error
  3. Feature-wise Kolmogorov-Smirnov tests — per-feature compatibility

Plus a unified `should_transfer` decision function.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats as sp_stats


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 1. Maximum Mean Discrepancy (MMD)
# ---------------------------------------------------------------------------

def _rbf_kernel(X, Y, gamma=None):
    """Compute RBF (Gaussian) kernel matrix between X and Y."""
    if gamma is None:
        # Median heuristic
        XY = np.vstack([X, Y])
        dists = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=-1)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (median_dist + 1e-8)

    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = XX + YY.T - 2 * X @ Y.T
    return np.exp(-gamma * dists)


def compute_mmd(X_source, X_target, gamma=None):
    """
    Compute Maximum Mean Discrepancy between source and target.

    MMD²(P,Q) = ||μ_P - μ_Q||²_H
              = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    Lower MMD means more similar distributions (better for transfer).

    Args:
        X_source: (n_s, d) source features
        X_target: (n_t, d) target features
        gamma: RBF kernel bandwidth (None = median heuristic)

    Returns:
        mmd_squared: float, the MMD² statistic
    """
    X_s = _to_numpy(X_source)
    X_t = _to_numpy(X_target)

    K_ss = _rbf_kernel(X_s, X_s, gamma)
    K_tt = _rbf_kernel(X_t, X_t, gamma)
    K_st = _rbf_kernel(X_s, X_t, gamma)

    n_s = X_s.shape[0]
    n_t = X_t.shape[0]

    # Unbiased estimator
    np.fill_diagonal(K_ss, 0)
    np.fill_diagonal(K_tt, 0)

    mmd_sq = (K_ss.sum() / (n_s * (n_s - 1))
              - 2 * K_st.sum() / (n_s * n_t)
              + K_tt.sum() / (n_t * (n_t - 1)))

    return float(max(0, mmd_sq))


# ---------------------------------------------------------------------------
# 2. Proxy A-distance (PAD)
# ---------------------------------------------------------------------------

def compute_proxy_a_distance(X_source, X_target, steps=200, lr=0.01, seed=42):
    """
    Compute Proxy A-distance via a domain classifier.

    Trains a linear classifier to distinguish source from target samples.
    d_A = 2(1 - 2ε) where ε is the classifier error.

    - d_A ≈ 0 → domains indistinguishable (good for transfer)
    - d_A ≈ 2 → domains perfectly separable (transfer risky)

    Args:
        X_source: (n_s, d) source features
        X_target: (n_t, d) target features
        steps: training iterations for domain classifier
        lr: learning rate
        seed: random seed for reproducibility

    Returns:
        pad: float in [0, 2], the proxy A-distance
        classifier_error: float, raw classification error
    """
    X_s = _to_numpy(X_source)
    X_t = _to_numpy(X_target)
    d = X_s.shape[1]

    # Build domain classification dataset
    X_all = np.vstack([X_s, X_t]).astype(np.float32)
    y_all = np.concatenate([np.zeros(len(X_s)), np.ones(len(X_t))]).astype(np.float32)

    # Shuffle (reproducible)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    # Train/test split (80/20)
    split = int(0.8 * len(X_all))
    X_train, y_train = torch.from_numpy(X_all[:split]), torch.from_numpy(y_all[:split])
    X_test, y_test = torch.from_numpy(X_all[split:]), torch.from_numpy(y_all[split:])

    # Linear domain classifier
    w = nn.Parameter(torch.zeros(d))
    b = nn.Parameter(torch.zeros(1))
    opt = optim.SGD([w, b], lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        opt.zero_grad()
        logits = X_train @ w + b
        loss = bce(logits, y_train)
        loss.backward()
        opt.step()

    # Evaluate
    with torch.no_grad():
        preds = (torch.sigmoid(X_test @ w + b) >= 0.5).float()
        error = (preds != y_test).float().mean().item()

    # d_A = 2(1 - 2ε), clamped to [0, 2]
    pad = max(0.0, min(2.0, 2.0 * (1.0 - 2.0 * error)))

    return pad, error


# ---------------------------------------------------------------------------
# 3. Feature-wise Kolmogorov-Smirnov Tests
# ---------------------------------------------------------------------------

def ks_feature_test(X_source, X_target, alpha=0.05):
    """
    Per-feature Kolmogorov-Smirnov test for compatibility.

    Tests whether each feature has the same distribution in source
    and target domains.  Reports the fraction of features that show
    statistically significant distributional shift.

    Args:
        X_source: (n_s, d) source features
        X_target: (n_t, d) target features
        alpha: significance level

    Returns:
        dict with:
          - fraction_shifted: float, fraction of features with p < alpha
          - max_ks_stat: float, largest KS statistic across features
          - per_feature: list of (ks_stat, p_value) tuples
    """
    X_s = _to_numpy(X_source)
    X_t = _to_numpy(X_target)
    d = X_s.shape[1]

    results = []
    n_shifted = 0
    max_stat = 0.0

    for j in range(d):
        stat, pval = sp_stats.ks_2samp(X_s[:, j], X_t[:, j])
        results.append((float(stat), float(pval)))
        if pval < alpha:
            n_shifted += 1
        max_stat = max(max_stat, stat)

    return {
        "fraction_shifted": n_shifted / d,
        "max_ks_stat": max_stat,
        "per_feature": results,
    }


# ---------------------------------------------------------------------------
# Unified Decision Function
# ---------------------------------------------------------------------------

def should_transfer(X_source, X_target, mmd_threshold=0.5, pad_threshold=1.5,
                    ks_threshold=0.5, verbose=False):
    """
    Decide whether transfer is likely to help or hurt.

    Runs all three detection methods and provides a recommendation.
    Conservative: recommends against transfer if ANY metric suggests
    high domain divergence.

    Args:
        X_source, X_target: feature matrices
        mmd_threshold: MMD² above this → domains too different
        pad_threshold: PAD above this → domains too separable
        ks_threshold: fraction of shifted features above this → risky

    Returns:
        dict with:
          - recommend: bool, True if transfer looks safe
          - mmd: float
          - pad: float
          - ks_fraction: float
          - reasons: list of strings explaining the decision
    """
    mmd_val = compute_mmd(X_source, X_target)
    pad_val, _ = compute_proxy_a_distance(X_source, X_target)
    ks_result = ks_feature_test(X_source, X_target)

    reasons = []
    recommend = True

    if mmd_val > mmd_threshold:
        recommend = False
        reasons.append(f"MMD²={mmd_val:.4f} > {mmd_threshold} (distributions differ)")

    if pad_val > pad_threshold:
        recommend = False
        reasons.append(f"PAD={pad_val:.4f} > {pad_threshold} (domains separable)")

    if ks_result["fraction_shifted"] > ks_threshold:
        recommend = False
        reasons.append(
            f"KS: {ks_result['fraction_shifted']:.0%} features shifted > {ks_threshold:.0%}"
        )

    if recommend:
        reasons.append("All metrics within safe thresholds — transfer recommended")

    if verbose:
        print(f"  MMD² = {mmd_val:.4f}  (threshold: {mmd_threshold})")
        print(f"  PAD  = {pad_val:.4f}  (threshold: {pad_threshold})")
        print(f"  KS shifted = {ks_result['fraction_shifted']:.0%}  "
              f"(threshold: {ks_threshold:.0%})")
        print(f"  → {'TRANSFER' if recommend else 'SKIP TRANSFER'}")

    return {
        "recommend": recommend,
        "mmd": mmd_val,
        "pad": pad_val,
        "ks_fraction": ks_result["fraction_shifted"],
        "ks_detail": ks_result,
        "reasons": reasons,
    }
