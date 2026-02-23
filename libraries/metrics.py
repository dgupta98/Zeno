"""
Evaluation metrics and utility functions for libraries.

Provides:
  - mse: Mean Squared Error for regression
  - r2_score: Coefficient of determination (R²)
  - accuracy_from_logits: Binary classification accuracy
  - estimate_energy_and_co2: Quick CO2 estimation formula
  - set_seed: Reproducibility across numpy + torch
"""

import torch
import numpy as np


def mse(yhat: torch.Tensor, y: torch.Tensor) -> float:
    """Mean Squared Error: (1/n) Σ (ŷᵢ − yᵢ)²."""
    return torch.mean((yhat - y) ** 2).item()


def r2_score(yhat: torch.Tensor, y: torch.Tensor) -> float:
    """
    Coefficient of determination (R²).

    R² = 1 − SS_res / SS_tot, where:
      SS_res = Σ (yᵢ − ŷᵢ)²
      SS_tot = Σ (yᵢ − ȳ)²

    R² = 1.0 → perfect prediction
    R² = 0.0 → predicting the mean
    R² < 0.0 → worse than predicting the mean
    """
    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2) + 1e-12
    ss_res = torch.sum((y - yhat) ** 2)
    return (1.0 - ss_res / ss_tot).item()


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Binary classification accuracy from raw logits (pre-sigmoid)."""
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return torch.mean((preds == y).float()).item()


def estimate_energy_and_co2(time_s: float, avg_power_watts: float = 30.0,
                            grid_kg_per_kwh: float = 0.45):
    """
    Quick CO2 estimation: CO2(kg) = Power(W) × Time(s) / 3,600,000 × CI.

    Args:
        time_s: elapsed training time in seconds
        avg_power_watts: estimated hardware power draw (default 30W for laptop)
        grid_kg_per_kwh: carbon intensity in kg CO2/kWh (default 0.45 = US avg)

    Returns:
        (energy_kwh, co2_kg): tuple of energy consumed and CO2 emitted
    """
    kwh = (avg_power_watts * time_s) / 3_600_000
    co2 = kwh * grid_kg_per_kwh
    return kwh, co2


def set_seed(seed: int):
    """
    Set random seeds for full reproducibility.

    Covers numpy, PyTorch CPU, and PyTorch CUDA (if available).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
