"""
Model merging algorithms for deep learning.

Extends Zeno with weight-space model combination techniques inspired by
mergekit, task arithmetic, and LoRA Soups.  All methods operate on
state_dicts — no model architecture changes needed.

Implemented algorithms:
  1. Linear merge (weighted average of state_dicts)
  2. SLERP (Spherical Linear Interpolation between two models)
  3. Task arithmetic (Ilharco et al., 2023): compute & combine task vectors
  4. TIES-Merging (Yadav et al., 2023): trim, elect sign, disjoint merge
  5. DARE (Yu et al., 2024): random drop + rescale before merging
  6. LoRA Soups (Huang et al., 2023): merge multiple LoRA adapter dicts

Key insight from the literature:
  "Model merging is transfer learning in weight space."
  Instead of training on combined data, we combine trained weights directly.
  This is zero-shot — no additional training data or compute required.

Recent findings (2024-2025):
  - TIES outperforms simple averaging by resolving sign conflicts
  - DARE + TIES is often the strongest combination for LLM merging
  - LoRA Soups enable combining 5-10 task-specific adapters with near-zero cost
  - SLERP preserves weight norms better than linear interpolation
  - Task vectors are surprisingly compositional: adding math + code vectors
    yields a model good at both, without multi-task training

Compatibility:
  All functions accept and return plain dict[str, Tensor] state_dicts.
  Works with any nn.Module — no architecture-specific code.
"""

import copy
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


# ─── Type aliases ───────────────────────────────────────────────────
StateDict = Dict[str, torch.Tensor]


# ═══════════════════════════════════════════════════════════════════
# 1. Linear Merge (Weighted Average)
# ═══════════════════════════════════════════════════════════════════

def linear_merge(state_dicts: List[StateDict],
                 weights: Optional[List[float]] = None) -> StateDict:
    """
    Weighted average of multiple model state_dicts.

    The simplest merging strategy: θ_merged = Σ w_i · θ_i
    When weights are uniform this is "model soups" (Wortsman et al., 2022).

    Works surprisingly well when models share the same pretraining base
    and were fine-tuned on related tasks.

    Args:
        state_dicts: list of state_dicts to merge
        weights: per-model weights (default: uniform 1/N)
            Weights are normalized to sum to 1.0.

    Returns:
        merged: averaged state_dict

    Example:
        >>> merged_sd = linear_merge([model_a.state_dict(), model_b.state_dict()])
        >>> model.load_state_dict(merged_sd)
    """
    if len(state_dicts) < 1:
        raise ValueError("Need at least one state_dict to merge")

    n = len(state_dicts)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(
                f"Got {len(weights)} weights for {n} state_dicts")
        # Normalize weights to sum to 1
        total = sum(weights)
        if total < 1e-12:
            raise ValueError("Weights sum to approximately zero")
        weights = [w / total for w in weights]

    # Build merged state_dict
    keys = list(state_dicts[0].keys())
    merged = {}
    for key in keys:
        merged[key] = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        ).to(state_dicts[0][key].dtype)

    return merged


# ═══════════════════════════════════════════════════════════════════
# 2. SLERP (Spherical Linear Interpolation)
# ═══════════════════════════════════════════════════════════════════

def _slerp_tensor(t: float, v0: torch.Tensor, v1: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """
    Spherical linear interpolation between two flat tensors.

    SLERP interpolates along the great circle on the hypersphere,
    preserving the norm of the interpolated vector better than
    linear interpolation (which shrinks norms at t=0.5).

    Formula:
        slerp(t, v0, v1) = sin((1-t)Ω)/sin(Ω) · v0 + sin(tΩ)/sin(Ω) · v1
        where Ω = arccos(v0·v1 / (|v0|·|v1|))

    Falls back to linear interpolation when vectors are nearly parallel
    (Ω ≈ 0) to avoid numerical instability.
    """
    v0_flat = v0.float().flatten()
    v1_flat = v1.float().flatten()

    # Compute angle between vectors
    v0_norm = torch.norm(v0_flat)
    v1_norm = torch.norm(v1_flat)

    if v0_norm < eps or v1_norm < eps:
        # Degenerate case: one vector is near-zero
        return ((1.0 - t) * v0.float() + t * v1.float()).to(v0.dtype)

    v0_unit = v0_flat / v0_norm
    v1_unit = v1_flat / v1_norm

    # Cosine of angle, clamped for numerical safety
    cos_omega = torch.clamp(torch.dot(v0_unit, v1_unit), -1.0, 1.0)
    omega = torch.acos(cos_omega)

    if omega.abs() < eps:
        # Vectors are nearly parallel — fall back to lerp
        result = (1.0 - t) * v0.float() + t * v1.float()
        return result.to(v0.dtype)

    sin_omega = torch.sin(omega)
    s0 = torch.sin((1.0 - t) * omega) / sin_omega
    s1 = torch.sin(t * omega) / sin_omega

    result_flat = s0 * v0_flat + s1 * v1_flat
    return result_flat.reshape(v0.shape).to(v0.dtype)


def slerp_merge(state_dict_a: StateDict,
                state_dict_b: StateDict,
                t: float = 0.5) -> StateDict:
    """
    Spherical linear interpolation between two model state_dicts.

    SLERP treats each parameter tensor as a point on a hypersphere
    and interpolates along the geodesic (great circle).

    Advantages over linear interpolation:
      - Preserves parameter norms at all interpolation points
      - Often yields better task performance at t=0.5
      - Standard in mergekit for 2-model merges

    Args:
        state_dict_a: first model's state_dict
        state_dict_b: second model's state_dict
        t: interpolation factor in [0, 1].
           t=0.0 returns model_a, t=1.0 returns model_b.

    Returns:
        merged: SLERP-interpolated state_dict

    Example:
        >>> merged_sd = slerp_merge(model_a.state_dict(),
        ...                          model_b.state_dict(), t=0.5)
        >>> model.load_state_dict(merged_sd)
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"Interpolation factor t must be in [0, 1], got {t}")

    merged = {}
    for key in state_dict_a:
        merged[key] = _slerp_tensor(t, state_dict_a[key], state_dict_b[key])
    return merged


# ═══════════════════════════════════════════════════════════════════
# 3. Task Arithmetic (Ilharco et al., ICLR 2023)
# ═══════════════════════════════════════════════════════════════════

def compute_task_vector(base_state_dict: StateDict,
                         finetuned_state_dict: StateDict) -> StateDict:
    """
    Compute the task vector: τ = θ_finetuned - θ_base.

    Task vectors (Ilharco et al., 2023) represent the "knowledge"
    learned during fine-tuning as a direction in weight space.

    Key properties:
      - Adding τ to a base model improves performance on that task
      - Negating τ and adding removes task knowledge (task forgetting)
      - Task vectors from different tasks can be composed:
        θ_multi = θ_base + α·τ_math + β·τ_code

    Args:
        base_state_dict: pretrained model weights (θ_base)
        finetuned_state_dict: fine-tuned model weights (θ_finetuned)

    Returns:
        task_vector: dict of parameter differences (τ = θ_ft - θ_base)
    """
    task_vec = {}
    for key in base_state_dict:
        task_vec[key] = (finetuned_state_dict[key].float()
                         - base_state_dict[key].float())
    return task_vec


def apply_task_vector(base_state_dict: StateDict,
                       task_vector: StateDict,
                       scaling: float = 1.0) -> StateDict:
    """
    Apply a task vector to a base model: θ' = θ_base + α · τ.

    Use scaling > 0 to add task knowledge, scaling < 0 to remove it.

    Args:
        base_state_dict: pretrained model weights
        task_vector: task vector (from compute_task_vector)
        scaling: coefficient α (default 1.0)

    Returns:
        new_state_dict: θ_base + α · τ
    """
    result = {}
    for key in base_state_dict:
        result[key] = (base_state_dict[key].float()
                       + scaling * task_vector[key].float()
                       ).to(base_state_dict[key].dtype)
    return result


def task_arithmetic_merge(base_state_dict: StateDict,
                           task_vectors: List[StateDict],
                           scalings: Optional[List[float]] = None
                           ) -> StateDict:
    """
    Merge multiple task vectors via task arithmetic.

    θ_merged = θ_base + Σ α_i · τ_i

    This is the core of "Editing Models with Task Arithmetic"
    (Ilharco et al., ICLR 2023).  Each task vector represents what
    a model learned during fine-tuning, and they compose additively.

    Args:
        base_state_dict: pretrained base model weights
        task_vectors: list of task vectors (each from compute_task_vector)
        scalings: per-task scaling coefficients (default: all 1.0)
            Typical range: 0.3 to 1.0

    Returns:
        merged: state_dict with all task vectors applied
    """
    if scalings is None:
        scalings = [1.0] * len(task_vectors)

    if len(scalings) != len(task_vectors):
        raise ValueError(
            f"Got {len(scalings)} scalings for {len(task_vectors)} task vectors"
        )

    result = {}
    for key in base_state_dict:
        combined = base_state_dict[key].float()
        for tv, alpha in zip(task_vectors, scalings):
            combined = combined + alpha * tv[key].float()
        result[key] = combined.to(base_state_dict[key].dtype)
    return result


# ═══════════════════════════════════════════════════════════════════
# 4. TIES-Merging (Yadav et al., NeurIPS 2023)
# ═══════════════════════════════════════════════════════════════════

def _trim_by_magnitude(task_vector: StateDict,
                        density: float) -> StateDict:
    """
    Trim: keep only the top-k% parameters by magnitude, zero the rest.

    This removes low-magnitude "noise" parameters that tend to
    cause interference during merging.

    Args:
        task_vector: parameter differences
        density: fraction of parameters to keep (e.g., 0.2 = top 20%)

    Returns:
        trimmed task vector
    """
    trimmed = {}
    for key, delta in task_vector.items():
        flat = delta.float().abs().flatten()
        k = max(1, int(density * flat.numel()))
        threshold = torch.topk(flat, k).values[-1]
        mask = delta.float().abs() >= threshold
        trimmed[key] = delta.float() * mask.float()
    return trimmed


def _elect_sign(task_vectors: List[StateDict]) -> StateDict:
    """
    Elect sign: for each parameter, choose the sign that the majority
    of task vectors agree on.

    Returns a dict of sign tensors (+1, -1, or 0).
    """
    signs = {}
    keys = list(task_vectors[0].keys())
    for key in keys:
        # Sum of signs across task vectors
        sign_sum = sum(torch.sign(tv[key].float()) for tv in task_vectors)
        # Majority sign: sign of the sum
        signs[key] = torch.sign(sign_sum)
    return signs


def _disjoint_merge(task_vectors: List[StateDict],
                     elected_signs: StateDict) -> StateDict:
    """
    Disjoint merge: for each parameter, average only the values whose
    sign agrees with the elected sign.
    """
    merged = {}
    for key in elected_signs:
        elected = elected_signs[key]
        # Sum values that agree with elected sign
        total = torch.zeros_like(elected)
        count = torch.zeros_like(elected)

        for tv in task_vectors:
            val = tv[key].float()
            agrees = (torch.sign(val) == elected).float()
            total += val * agrees
            count += agrees

        # Average where count > 0
        safe_count = torch.clamp(count, min=1.0)
        merged[key] = total / safe_count

    return merged


def ties_merge(base_state_dict: StateDict,
               task_vectors: List[StateDict],
               density: float = 0.2,
               scaling: float = 1.0) -> StateDict:
    """
    TIES-Merging: Trim, Elect sign, disjoint merge.

    Three-step process (Yadav et al., NeurIPS 2023):
    1. TRIM: For each task vector, zero out low-magnitude parameters,
       keeping only the top `density` fraction.
    2. ELECT SIGN: For each parameter position, compute the majority
       sign across task vectors.  This resolves sign conflicts that
       cause destructive interference in naive averaging.
    3. DISJOINT MERGE: Average only the values that agree with the
       elected sign, discarding conflicting updates.

    Finally: θ_merged = θ_base + scaling · merged_vector

    TIES consistently outperforms simple averaging and task arithmetic
    because it resolves the "sign disagreement" problem where different
    tasks push the same parameter in opposite directions.

    Args:
        base_state_dict: pretrained base model weights
        task_vectors: list of task vectors (from compute_task_vector)
        density: fraction of parameters to keep after trimming (default 0.2)
            Lower density = more aggressive pruning
        scaling: global scaling coefficient (default 1.0)

    Returns:
        merged: state_dict after TIES merge
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")

    # Step 1: Trim each task vector
    trimmed = [_trim_by_magnitude(tv, density) for tv in task_vectors]

    # Step 2: Elect sign (majority vote per parameter)
    elected_signs = _elect_sign(trimmed)

    # Step 3: Disjoint merge (average values agreeing with elected sign)
    merged_vector = _disjoint_merge(trimmed, elected_signs)

    # Apply to base model
    return apply_task_vector(base_state_dict, merged_vector, scaling=scaling)


# ═══════════════════════════════════════════════════════════════════
# 5. DARE (Yu et al., 2024) — Drop And REscale
# ═══════════════════════════════════════════════════════════════════

def _dare_drop_rescale(task_vector: StateDict,
                        drop_rate: float,
                        seed: Optional[int] = None) -> StateDict:
    """
    DARE: randomly drop parameters, then rescale survivors.

    For each parameter:
      - With probability `drop_rate`, set to zero
      - Rescale remaining values by 1/(1-drop_rate) to preserve expected magnitude

    This is analogous to dropout but applied to task vectors.
    The rescaling ensures the expected value of the merged vector
    is preserved despite the random zeroing.

    Args:
        task_vector: parameter differences
        drop_rate: fraction of parameters to randomly zero out (e.g., 0.9)
        seed: optional random seed for reproducibility

    Returns:
        dropped and rescaled task vector
    """
    if seed is not None:
        torch.manual_seed(seed)

    rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 0.0
    result = {}
    for key, delta in task_vector.items():
        mask = (torch.rand_like(delta.float()) >= drop_rate).float()
        result[key] = delta.float() * mask * rescale
    return result


def dare_merge(base_state_dict: StateDict,
               task_vectors: List[StateDict],
               drop_rate: float = 0.9,
               scaling: float = 1.0,
               use_ties: bool = True,
               ties_density: float = 0.2,
               seed: Optional[int] = None) -> StateDict:
    """
    DARE merging: Drop And REscale before combining task vectors.

    DARE (Yu et al., 2024) randomly zeros most of each task vector
    and rescales the survivors to preserve expected magnitude.
    This dramatically reduces parameter interference between tasks.

    Can be combined with TIES for even better results (DARE + TIES):
    1. Drop + rescale each task vector
    2. Apply TIES (trim, elect sign, disjoint merge)

    Or with simple linear merging (DARE + Linear):
    1. Drop + rescale each task vector
    2. Average the surviving values
    3. Apply to base

    Args:
        base_state_dict: pretrained base model weights
        task_vectors: list of task vectors
        drop_rate: fraction to randomly zero (default 0.9 = keep 10%)
        scaling: global scaling coefficient (default 1.0)
        use_ties: if True, apply TIES after DARE; otherwise linear average
        ties_density: density for TIES trimming (if use_ties=True)
        seed: random seed for reproducibility

    Returns:
        merged: state_dict after DARE merge
    """
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")

    # Apply DARE to each task vector
    dared = []
    for i, tv in enumerate(task_vectors):
        tv_seed = seed + i if seed is not None else None
        dared.append(_dare_drop_rescale(tv, drop_rate, seed=tv_seed))

    if use_ties:
        # DARE + TIES
        return ties_merge(base_state_dict, dared,
                          density=ties_density, scaling=scaling)
    else:
        # DARE + Linear average
        return task_arithmetic_merge(
            base_state_dict, dared,
            scalings=[scaling / len(dared)] * len(dared)
        )


# ═══════════════════════════════════════════════════════════════════
# 6. LoRA Soups (Adapter Merging)
# ═══════════════════════════════════════════════════════════════════

def merge_lora_adapters(lora_state_dicts: List[StateDict],
                         weights: Optional[List[float]] = None
                         ) -> StateDict:
    """
    Merge multiple LoRA adapter state_dicts into one (LoRA Soups).

    "LoRA Soups" (Huang et al., 2023) averages LoRA A and B matrices
    from multiple task-specific adapters.  The merged adapter can be
    loaded once for multi-task inference, avoiding the cost of
    switching adapters or running multiple forward passes.

    The merged LoRA approximates the combined effect of all adapters:
      ΔW_merged ≈ (1/N) · Σ (B_i · A_i)

    This is cheaper than merging full models because LoRA adapters
    are tiny (typically 0.1-1% of model parameters).

    Compatible with LoRAInjector.lora_state_dict() output format.

    Args:
        lora_state_dicts: list of LoRA state_dicts (from lora_state_dict())
        weights: per-adapter weights (default: uniform 1/N)

    Returns:
        merged_lora: averaged LoRA state_dict

    Example:
        >>> sd_math = LoRAInjector.lora_state_dict(model_math)
        >>> sd_code = LoRAInjector.lora_state_dict(model_code)
        >>> merged = merge_lora_adapters([sd_math, sd_code])
        >>> # Load merged adapter into a fresh model
    """
    # LoRA state_dicts are just {name: tensor} — use linear_merge
    return linear_merge(lora_state_dicts, weights=weights)


# ═══════════════════════════════════════════════════════════════════
# 7. LoRA-Flow (Learned Gating for Adapter Merging)
# ═══════════════════════════════════════════════════════════════════

class LoRAFlow(nn.Module):
    """
    LoRA-Flow: learned gating for dynamic adapter combination.

    Instead of fixed-weight merging (LoRA Soups), LoRA-Flow learns
    per-adapter gating weights that are conditioned on the input.
    This enables a single model to dynamically combine multiple
    task-specific LoRA adapters at inference time.

    Architecture:
        Given N adapters with LoRA outputs [ΔW_1·x, ΔW_2·x, ..., ΔW_N·x],
        LoRA-Flow computes:
            gate = softmax(W_gate · h + b_gate)
            output = Σ gate_i · ΔW_i · x

        where h is an input representation (e.g., pooled hidden state
        or the input x itself) and W_gate is a small learned matrix.

    The gating weights are the ONLY trainable parameters — all LoRA
    adapters and the base model remain frozen.  This makes LoRA-Flow
    extremely parameter-efficient: only N + (d × N) parameters for
    N adapters and d-dimensional input to the gate.

    Usage:
        flow = LoRAFlow(num_adapters=3, gate_input_dim=768)
        # During training:
        gate_weights = flow(hidden_states)  # (batch, num_adapters)
        combined = sum(w * out for w, out in zip(gate_weights.T, adapter_outputs))

    Reference: Wang et al., "LoRA-Flow: Dynamic LoRA Fusion for
    Large Language Models" (2024)
    """

    def __init__(self, num_adapters: int, gate_input_dim: int,
                 temperature: float = 1.0):
        """
        Args:
            num_adapters: number of LoRA adapters to gate over
            gate_input_dim: dimension of the input to the gating function
            temperature: softmax temperature (lower = sharper gating)
        """
        super().__init__()
        self.num_adapters = num_adapters
        self.temperature = temperature

        # Small gating network: linear projection → softmax
        self.gate = nn.Linear(gate_input_dim, num_adapters)
        # Initialize gate to produce uniform weights
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-adapter gating weights.

        Args:
            x: (batch, gate_input_dim) input features for gating

        Returns:
            weights: (batch, num_adapters) softmax-normalized weights
        """
        logits = self.gate(x) / self.temperature
        return torch.softmax(logits, dim=-1)

    def merge_with_gates(self, adapter_outputs: List[torch.Tensor],
                          gate_input: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple adapter outputs using learned gates.

        Args:
            adapter_outputs: list of N tensors, each (batch, out_dim)
            gate_input: (batch, gate_input_dim) features for gating

        Returns:
            combined: (batch, out_dim) gated combination
        """
        weights = self(gate_input)  # (batch, N)
        # Stack adapter outputs: (batch, N, out_dim)
        stacked = torch.stack(adapter_outputs, dim=1)
        # Weighted sum: (batch, out_dim)
        combined = torch.einsum("bn,bnd->bd", weights, stacked)
        return combined

    def get_gate_params(self):
        """Return only the gating parameters for the optimizer."""
        return list(self.gate.parameters())

    def get_current_weights(self, x: torch.Tensor) -> List[float]:
        """Get average gating weights for a batch (for monitoring)."""
        with torch.no_grad():
            weights = self(x)
            return weights.mean(dim=0).tolist()


def train_lora_flow(flow: LoRAFlow,
                     adapter_outputs_fn,
                     gate_input_fn,
                     dataloader,
                     criterion,
                     target_fn,
                     epochs: int = 5,
                     lr: float = 1e-3) -> dict:
    """
    Train LoRA-Flow gating weights on a target task.

    Only the gating parameters are trained — all adapters and the
    base model remain frozen.

    Args:
        flow: LoRAFlow module
        adapter_outputs_fn: callable(batch) -> list of adapter outputs
        gate_input_fn: callable(batch) -> gate input features
        dataloader: training data
        criterion: loss function
        target_fn: callable(batch) -> target labels
        epochs: number of training epochs
        lr: learning rate for gate parameters

    Returns:
        history: dict with 'loss' list and 'gate_weights' list
    """
    optimizer = torch.optim.Adam(flow.get_gate_params(), lr=lr)
    history = {"loss": [], "gate_weights": []}

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            adapter_outs = adapter_outputs_fn(batch)
            gate_input = gate_input_fn(batch)
            targets = target_fn(batch)

            combined = flow.merge_with_gates(adapter_outs, gate_input)
            loss = criterion(combined, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)

        # Log average gate weights
        with torch.no_grad():
            sample_input = gate_input_fn(batch)
            weights = flow.get_current_weights(sample_input)
            history["gate_weights"].append(weights)

    return history


# ═══════════════════════════════════════════════════════════════════
# 8. Merge Analysis Utilities
# ═══════════════════════════════════════════════════════════════════

def task_vector_stats(task_vector: StateDict) -> dict:
    """
    Compute statistics about a task vector for analysis.

    Useful for understanding the magnitude and distribution of
    changes from fine-tuning before deciding on a merge strategy.

    Returns:
        dict with keys:
          - total_params: total number of parameters
          - nonzero_params: number of nonzero parameters
          - sparsity: fraction of zero parameters
          - l2_norm: global L2 norm of the task vector
          - mean_magnitude: average absolute value
          - max_magnitude: maximum absolute value
          - per_layer: dict of per-layer norms
    """
    total = 0
    nonzero = 0
    sum_sq = 0.0
    sum_abs = 0.0
    max_mag = 0.0
    per_layer = {}

    for key, delta in task_vector.items():
        d = delta.float()
        n = d.numel()
        total += n
        nonzero += (d.abs() > 1e-10).sum().item()
        layer_norm = torch.norm(d).item()
        sum_sq += layer_norm ** 2
        sum_abs += d.abs().sum().item()
        max_mag = max(max_mag, d.abs().max().item())
        per_layer[key] = layer_norm

    return {
        "total_params": total,
        "nonzero_params": int(nonzero),
        "sparsity": 1.0 - nonzero / max(total, 1),
        "l2_norm": math.sqrt(sum_sq),
        "mean_magnitude": sum_abs / max(total, 1),
        "max_magnitude": max_mag,
        "per_layer": per_layer,
    }


def task_vector_similarity(tv_a: StateDict, tv_b: StateDict) -> float:
    """
    Cosine similarity between two task vectors.

    High similarity (> 0.5) suggests the tasks are related and
    merging should work well.  Low or negative similarity suggests
    the tasks may interfere.

    Args:
        tv_a, tv_b: task vectors (from compute_task_vector)

    Returns:
        cosine_similarity: float in [-1, 1]
    """
    flat_a = torch.cat([tv_a[k].float().flatten() for k in tv_a])
    flat_b = torch.cat([tv_b[k].float().flatten() for k in tv_b])

    dot = torch.dot(flat_a, flat_b)
    norm_a = torch.norm(flat_a)
    norm_b = torch.norm(flat_b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(dot / (norm_a * norm_b))
