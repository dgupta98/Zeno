"""
libraries - Transfer Learning for Classical ML
================================================

A from-scratch Python library demonstrating that transfer learning
techniques from deep learning apply to classical linear and logistic
regression, with measurable CO2 savings.

Four core transfer approaches:
  1. Regularized weight transfer (closed-form / gradient-based)
  2. LoRA-style low-rank adaptation (vector + matrix)
  3. Statistical dataset-to-weight mapping
  4. Bayesian prior transfer (source posterior as target prior)

Plus negative transfer detection (MMD, Proxy A-distance, KS tests)
and integrated carbon emissions tracking.

Built for the ASU Principled AI Spark Challenge.
"""

from .train_core import (
    fit_linear_sgd,
    fit_logistic_sgd,
    eval_linear,
    eval_logistic,
)
from .transfer import (
    regularized_transfer_linear,
    regularized_transfer_logistic,
    bayesian_transfer_linear,
    bayesian_transfer_logistic,
    bayesian_posterior_precision,
    covariance_transfer_linear,
)
from .adapters import LoRAAdapterVector, LoRAAdapterMatrix
from .negative_transfer import (
    compute_mmd,
    compute_proxy_a_distance,
    ks_feature_test,
    should_transfer,
    validate_transfer,
)
from .stat_mapping import moment_init_linear, moment_init_logistic
from .metrics import (
    mse,
    r2_score,
    accuracy_from_logits,
    estimate_energy_and_co2,
    set_seed,
)
from .carbon import CarbonTracker, compare_emissions
from .real_datasets import (
    load_california_housing_linear,
    load_wine_linear,
    load_titanic_logistic,
    load_breast_cancer_logistic,
)

__version__ = "0.3.0"
