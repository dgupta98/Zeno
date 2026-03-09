"""
libraries - Transfer Learning for Classical & Deep Learning
============================================================

A from-scratch Python library demonstrating that transfer learning
techniques apply to both classical ML and deep neural networks,
with measurable CO2 savings.

Classical ML (linear/logistic regression):
  1. Regularized weight transfer (closed-form / gradient-based)
  2. LoRA-style low-rank adaptation (vector + matrix)
  3. Statistical dataset-to-weight mapping
  4. Bayesian prior transfer (source posterior as target prior)

Deep Learning (nn.Module-based models):
  5. LoRA injection for any pretrained model (LoRALinear, LoRAInjector)
  6. Layer-wise transfer with progressive unfreezing (BaseModel, TransferScheduler)
  7. Elastic Weight Consolidation for Bayesian transfer (EWCLoss)
  8. CKA-based negative transfer detection (compute_cka, NegativeTransferMonitor)
  9. Model merging (SLERP, Task Arithmetic, TIES, DARE, LoRA Soups)
  10. GPU-aware carbon tracking via NVML (GPUCarbonTracker)

Plus negative transfer detection (MMD, Proxy A-distance, KS, CKA)
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

# Deep learning extensions
from .dl import (
    LoRALinear,
    LoRAInjector,
    BaseModel,
    TransferScheduler,
    build_discriminative_lr_groups,
    compute_fisher_diagonal,
    EWCLoss,
    online_ewc_update,
    compute_cka,
    extract_representations,
    compute_representation_mmd,
    NegativeTransferMonitor,
    linear_merge,
    slerp_merge,
    compute_task_vector,
    apply_task_vector,
    task_arithmetic_merge,
    ties_merge,
    dare_merge,
    merge_lora_adapters,
    LoRAFlow,
    train_lora_flow,
    task_vector_stats,
    task_vector_similarity,
    GPUCarbonTracker,
    train_epoch,
    evaluate,
    fine_tune,
)

__version__ = "0.5.0"
