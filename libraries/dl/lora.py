"""
LoRA (Low-Rank Adaptation) for deep learning models.

Extends Zeno's classical LoRA adapters to nn.Module-based models.
Two core components:
  - LoRALinear: wraps a single nn.Linear with low-rank adaptation
  - LoRAInjector: programmatically injects LoRA into any model

Mathematical formulation (Hu et al., ICLR 2022):
    W' = W_0 + (alpha/r) * B @ A
    where W_0 is frozen, A in R^{r x d_in}, B in R^{d_out x r}
    B initialized to zeros, A to Kaiming uniform => Delta_W = 0 at start.

Applied to Q+V projections across GPT-2's 12 layers, this yields ~147K
trainable parameters out of 124M total (0.12%).

Key 2025 finding: LoRA must target ALL linear layers (attention + FFN)
to match full fine-tuning.  "Expand layers first, then increase rank"
outperforms "high rank on few layers."
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-rank adaptation wrapper for nn.Linear.

    Wraps a frozen nn.Linear layer, adding trainable low-rank matrices
    A and B such that the effective weight becomes:

        W' = W_0 + (alpha/r) * B @ A

    At initialization, B=0 so the model produces identical outputs
    to the original pretrained model (Hu et al., 2021).

    For a 768x768 weight matrix with rank 8, LoRA requires only
    12,288 parameters versus 589,824 for full Delta_W (48x reduction).

    Args:
        linear: the nn.Linear layer to wrap (will be frozen)
        rank: rank of the low-rank decomposition (default 8)
        alpha: scaling factor (default 16, so scaling = alpha/rank = 2)
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = linear.in_features
        d_out = linear.out_features

        # A: projects input to low-rank space
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        # B: projects from low-rank space to output
        self.lora_B = nn.Linear(rank, d_out, bias=False)

        # Hu et al. (2021) initialization: A ~ Kaiming, B = 0 => Delta_W = 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

        self._merged = False

    def forward(self, x):
        """Forward pass: original output + scaled low-rank update."""
        base_out = self.linear(x)
        if self._merged:
            return base_out
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into base linear for zero-overhead inference."""
        if self._merged:
            return
        # W_merged = W_0 + (alpha/r) * B @ A
        delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.linear.weight.data += delta
        self._merged = True

    def unmerge_weights(self):
        """Reverse merge to resume training."""
        if not self._merged:
            return
        delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.linear.weight.data -= delta
        self._merged = False

    def trainable_params(self):
        """Number of trainable LoRA parameters."""
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])

    def full_params(self):
        """Number of parameters in the original linear layer."""
        n = self.linear.weight.numel()
        if self.linear.bias is not None:
            n += self.linear.bias.numel()
        return n

    def reduction_ratio(self):
        """How many times fewer params LoRA uses vs full fine-tuning."""
        return self.full_params() / self.trainable_params()


class LoRAInjector:
    """
    Programmatically inject LoRA adapters into any nn.Module.

    Walks model.named_modules(), identifies nn.Linear layers matching
    target patterns, and replaces them with LoRALinear wrappers using
    setattr() on the parent module.

    Critical implementation detail: modules are collected first, then
    replaced.  Modifying during iteration causes errors.

    Usage:
        injector = LoRAInjector()
        injector.inject(model, target_modules=["q_proj", "v_proj"], rank=8)
        optimizer = optim.AdamW(injector.get_lora_parameters(model), lr=1e-4)
        # ... training ...
        injector.merge_all(model)  # for inference
    """

    @staticmethod
    def inject(model, target_modules=None, rank=8, alpha=16.0):
        """
        Inject LoRA into matching nn.Linear layers.

        Args:
            model: any nn.Module
            target_modules: list of substrings to match layer names.
                If None, injects into ALL nn.Linear layers.
            rank: LoRA rank
            alpha: LoRA scaling factor

        Returns:
            count: number of layers injected
        """
        # Phase 1: collect targets (don't modify during iteration)
        targets = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if target_modules is None:
                    targets.append(name)
                elif any(t in name for t in target_modules):
                    targets.append(name)

        # Phase 2: replace with LoRALinear wrappers
        count = 0
        for name in targets:
            # Navigate to parent module
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]
            original = getattr(parent, child_name)

            # Replace with LoRALinear
            lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            count += 1

        return count

    @staticmethod
    def get_lora_parameters(model):
        """Return only the trainable LoRA parameters for the optimizer."""
        params = []
        for module in model.modules():
            if isinstance(module, LoRALinear):
                params.extend([module.lora_A.weight, module.lora_B.weight])
        return params

    @staticmethod
    def merge_all(model):
        """Merge all LoRA weights into base layers for inference."""
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()

    @staticmethod
    def unmerge_all(model):
        """Unmerge all LoRA weights to resume training."""
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()

    @staticmethod
    def lora_state_dict(model):
        """Extract only LoRA weights for lightweight checkpointing."""
        state = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                state[f"{name}.lora_A.weight"] = module.lora_A.weight.data.clone()
                state[f"{name}.lora_B.weight"] = module.lora_B.weight.data.clone()
        return state

    @staticmethod
    def count_lora_params(model):
        """Count total trainable LoRA parameters in the model."""
        total = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                total += module.trainable_params()
        return total
