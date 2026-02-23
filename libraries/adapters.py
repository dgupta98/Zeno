"""
LoRA (Low-Rank Adaptation) for classical ML models.

Two variants:
  - LoRAAdapterVector: for binary classification / single-output regression
    (w is a d-vector). Note: doesn't reduce params for small d, but
    the low-rank constraint acts as implicit regularization.

  - LoRAAdapterMatrix: for multi-class classification
    (W is a d×k matrix). Genuine parameter reduction when r << min(d,k).
    e.g., d=1000, k=50, full=50,000 params → LoRA r=5 → 5,250 params (9.5×)
"""

import torch
import torch.nn as nn


class LoRAAdapterVector(nn.Module):
    """
    For a vector weight w in R^d (binary / single-output):
      w' = w_base + (alpha/r) * B @ a
    where B in R^{d x r}, a in R^{r}.
    Only B, a (and optional bias delta) are trainable.
    """
    def __init__(self, d: int, r: int = 2, alpha: float = 1.0,
                 train_bias: bool = True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        # Hu et al. (2021) init: A ~ N(0, σ²), B = 0 → ΔW = BA = 0 at start
        self.B = nn.Parameter(torch.zeros(d, r))
        self.a = nn.Parameter(torch.randn(r) * 0.01)
        self.db = nn.Parameter(torch.zeros(1)) if train_bias else None

    def delta_w(self):
        return (self.alpha / self.r) * (self.B @ self.a)  # (d,)

    def delta_b(self):
        if self.db is None:
            return 0.0
        return self.db

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoRAAdapterMatrix(nn.Module):
    """
    For a weight matrix W in R^{d x k} (multi-class):
      W' = W_base + (alpha/r) * B @ A
    where B in R^{d x r}, A in R^{r x k}, r << min(d, k).

    This is where LoRA truly shines for classical models:
      Full params:    d * k
      LoRA params:    r * (d + k)
      Reduction:      d*k / (r*(d+k))

    Example: d=1000, k=50, r=5 → 50,000 vs 5,250 = 9.5× reduction
    """
    def __init__(self, d: int, k: int, r: int = 5, alpha: float = 1.0,
                 train_bias: bool = True):
        super().__init__()
        self.d = d
        self.k = k
        self.r = r
        self.alpha = alpha

        # LoRA decomposition: ΔW = B @ A
        # Hu et al. (2021) init: A ~ N(0, σ²), B = 0 → ΔW = BA = 0 at start
        self.B = nn.Parameter(torch.zeros(d, r))
        self.A = nn.Parameter(torch.randn(r, k) * 0.01)
        self.db = nn.Parameter(torch.zeros(k)) if train_bias else None

    def delta_W(self):
        return (self.alpha / self.r) * (self.B @ self.A)  # (d, k)

    def delta_b(self):
        if self.db is None:
            return torch.zeros(self.k)
        return self.db

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def full_params(self):
        """Number of params in the equivalent full weight matrix."""
        return self.d * self.k + self.k  # W + bias

    def reduction_ratio(self):
        """How many times fewer params LoRA uses vs full."""
        return self.full_params() / self.trainable_params()