import time
import torch
import numpy as np

def mse(yhat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean((yhat - y) ** 2).item()

def r2_score(yhat: torch.Tensor, y: torch.Tensor) -> float:
    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2) + 1e-12
    ss_res = torch.sum((y - yhat) ** 2)
    return (1.0 - ss_res / ss_tot).item()

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return torch.mean((preds == y).float()).item()

def estimate_energy_and_co2(time_s: float, avg_power_watts: float = 30.0, grid_kg_per_kwh: float = 0.45):
    # E(kWh) = W * s / (3600*1000)
    kwh = (avg_power_watts * time_s) / 3600.0 / 1000.0
    co2 = kwh * grid_kg_per_kwh
    return kwh, co2

def now():
    return time.time()

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)