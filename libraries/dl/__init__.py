"""
Deep learning extensions for the Zeno transfer learning library.

Extends every classical ML concept to nn.Module-based models:
  - LoRA injection for any pretrained model (LoRALinear, LoRAInjector)
  - Layer-wise transfer with progressive unfreezing (BaseModel, TransferScheduler)
  - Elastic Weight Consolidation for Bayesian transfer (EWCLoss)
  - CKA-based negative transfer detection (compute_cka, NegativeTransferMonitor)
  - Model merging (SLERP, task arithmetic, TIES, DARE, LoRA Soups)
  - GPU-aware carbon tracking via NVML (GPUCarbonTracker)
  - From-scratch training loops (train_epoch, evaluate, fine_tune)
"""

from .lora import LoRALinear, LoRAInjector
from .transfer import BaseModel, TransferScheduler, build_discriminative_lr_groups
from .ewc import compute_fisher_diagonal, EWCLoss, online_ewc_update
from .negative_transfer import (
    compute_cka,
    extract_representations,
    compute_representation_mmd,
    NegativeTransferMonitor,
)
from .merging import (
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
)
from .carbon import GPUCarbonTracker
from .train import train_epoch, evaluate, fine_tune
