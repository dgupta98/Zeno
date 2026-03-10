"""
Layer-wise weight transfer for deep networks.

Extends Zeno's classical transfer methods to nn.Module-based models:
  - BaseModel: wraps any nn.Module with freeze/unfreeze/layer surgery
  - TransferScheduler: progressive unfreezing with discriminative LRs
  - build_discriminative_lr_groups: standalone utility for param_groups

Key concepts (Yosinski et al., 2014):
  Early layers learn general features (edges, textures).
  Later layers learn task-specific representations.
  => Freeze general layers, fine-tune specific ones.

Progressive unfreezing (Howard & Ruder, ULMFiT 2018):
  Gradually unfreeze layers from top to bottom across epochs.
  Discriminative learning rates: eta_l = eta_base / decay^(L-l)
  where L is total depth and l is layer index.
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Transfer learning wrapper for any nn.Module.

    Provides freeze/unfreeze controls, layer group extraction for
    discriminative learning rates, parameter counting, and layer
    surgery (head replacement) for domain adaptation.

    Usage:
        base = BaseModel(resnet50(weights='IMAGENET1K_V1'))
        base.freeze_all()
        base.replace_head(num_classes=10)
        print(base.count_parameters())  # {'total': 25M, 'trainable': 5130}
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def freeze_all(self):
        """Freeze all parameters (feature extraction mode)."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters (full fine-tuning mode)."""
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_layer(self, name):
        """Freeze a specific named layer/module."""
        module = dict(self.model.named_modules()).get(name)
        if module is None:
            raise ValueError(f"Layer '{name}' not found in model")
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_layer(self, name):
        """Unfreeze a specific named layer/module."""
        module = dict(self.model.named_modules()).get(name)
        if module is None:
            raise ValueError(f"Layer '{name}' not found in model")
        for param in module.parameters():
            param.requires_grad = True

    def freeze_below(self, layer_name):
        """
        Freeze all layers that appear before the named layer.

        Useful for freezing early (general) layers while keeping
        later (task-specific) layers trainable.
        """
        found = False
        for name, module in self.model.named_children():
            if name == layer_name:
                found = True
                continue
            if not found:
                for param in module.parameters():
                    param.requires_grad = False

    def get_layer_groups(self):
        """
        Return ordered list of (name, list[param]) for discriminative LRs.

        Groups are ordered from shallowest to deepest (first = lowest LR).
        Each top-level child module is one group.
        """
        groups = []
        for name, module in self.model.named_children():
            params = list(module.parameters())
            if params:
                groups.append((name, params))
        return groups

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def replace_head(self, num_classes, head_names=None):
        """
        Replace the classification head for a new number of classes.

        Auto-detects common head names: 'fc', 'classifier', 'head'.
        Falls back to the last nn.Linear found.

        Args:
            num_classes: number of output classes for the new head
            head_names: optional list of attribute names to check
        """
        if head_names is None:
            head_names = ["fc", "classifier", "head"]

        # Try known head names first
        for name in head_names:
            head = getattr(self.model, name, None)
            if head is not None and isinstance(head, nn.Linear):
                new_head = nn.Linear(head.in_features, num_classes)
                setattr(self.model, name, new_head)
                return name

            # Handle nn.Sequential classifiers (e.g., VGG, EfficientNet)
            if head is not None and isinstance(head, nn.Sequential):
                for i in reversed(range(len(head))):
                    if isinstance(head[i], nn.Linear):
                        head[i] = nn.Linear(head[i].in_features, num_classes)
                        return f"{name}.{i}"

        # Fallback: find last nn.Linear in the entire model
        last_name = None
        last_parent = None
        last_child_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                last_name = name
                last_parent = parent
                last_child_name = parts[-1]

        if last_parent is not None:
            old = getattr(last_parent, last_child_name)
            new_head = nn.Linear(old.in_features, num_classes)
            setattr(last_parent, last_child_name, new_head)
            return last_name

        raise ValueError("Could not find a Linear head to replace")


class TransferScheduler:
    """
    Progressive unfreezing with discriminative learning rates.

    Implements ULMFiT-style training (Howard & Ruder, 2018):
    1. Start with all layers frozen except the head
    2. Each epoch (or every N epochs), unfreeze the next deeper group
    3. Newly unfrozen groups get lower learning rates

    Learning rate formula: eta_l = eta_base / decay^(L - l)
    where L is the total number of groups and l is the group index
    (0 = shallowest, L-1 = deepest/head).

    Usage:
        base = BaseModel(model)
        base.freeze_all()
        groups = base.get_layer_groups()
        scheduler = TransferScheduler(groups, base_lr=1e-3, decay=2.6)
        optimizer = scheduler.build_optimizer()

        for epoch in range(num_epochs):
            scheduler.step(epoch)
            train_epoch(model, ...)
    """

    def __init__(self, layer_groups, base_lr=1e-3, decay=2.6,
                 unfreeze_every=1):
        """
        Args:
            layer_groups: list of (name, params) from BaseModel.get_layer_groups()
            base_lr: learning rate for the topmost (head) layer
            decay: LR decay factor per layer depth (ULMFiT default: 2.6)
            unfreeze_every: unfreeze a new group every N epochs
        """
        self.layer_groups = layer_groups
        self.base_lr = base_lr
        self.decay = decay
        self.unfreeze_every = unfreeze_every
        self.num_groups = len(layer_groups)

        # Track which groups are unfrozen (start with only the last/head)
        self._unfrozen_up_to = self.num_groups  # index of first unfrozen group

        # Initially freeze all, unfreeze only the head (last group)
        for _, params in self.layer_groups:
            for p in params:
                p.requires_grad = False
        if self.layer_groups:
            for p in self.layer_groups[-1][1]:
                p.requires_grad = True
            self._unfrozen_up_to = self.num_groups - 1

    def step(self, epoch):
        """
        Unfreeze the next layer group if it's time.

        Args:
            epoch: current epoch number (0-indexed)
        """
        if epoch == 0:
            return  # head already unfrozen at init

        if epoch % self.unfreeze_every == 0 and self._unfrozen_up_to > 0:
            self._unfrozen_up_to -= 1
            idx = self._unfrozen_up_to
            for p in self.layer_groups[idx][1]:
                p.requires_grad = True

    def build_param_groups(self):
        """
        Build optimizer param_groups with discriminative learning rates.

        Returns param_groups for all currently unfrozen layers.
        Deepest layers (head) get base_lr, shallower layers get decayed LRs.

        Returns:
            list of dicts suitable for torch.optim.AdamW(param_groups)
        """
        param_groups = []
        L = self.num_groups
        for i in range(self._unfrozen_up_to, L):
            name, params = self.layer_groups[i]
            # Distance from top: L-1 is the head (highest LR)
            depth_from_top = L - 1 - i
            lr = self.base_lr / (self.decay ** depth_from_top)
            trainable = [p for p in params if p.requires_grad]
            if trainable:
                param_groups.append({"params": trainable, "lr": lr})
        return param_groups

    def build_optimizer(self, weight_decay=0.01):
        """Convenience: build AdamW with discriminative LRs."""
        param_groups = self.build_param_groups()
        if not param_groups:
            # Return optimizer with empty params to avoid error
            return torch.optim.AdamW([{"params": [], "lr": self.base_lr}],
                                     weight_decay=weight_decay)
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_discriminative_lr_groups(model, base_lr=1e-3, decay=2.6):
    """
    Standalone utility to build param_groups with discriminative LRs.

    Assigns each top-level child module a learning rate following:
        eta_l = eta_base / decay^(L - 1 - l)

    where L is the total number of layer groups and l is the group index.
    The last group (head) gets base_lr, earlier groups get progressively
    smaller learning rates.

    This is the transformer equivalent of LLRD (Layer-wise Learning Rate
    Decay), effective for BERT/RoBERTa fine-tuning.

    Args:
        model: any nn.Module (or BaseModel)
        base_lr: learning rate for the head layer
        decay: decay factor per depth level

    Returns:
        list of param_group dicts for torch.optim optimizers
    """
    # Unwrap BaseModel if needed
    inner = model.model if isinstance(model, BaseModel) else model

    groups = []
    children = list(inner.named_children())
    L = len(children)

    for i, (name, module) in enumerate(children):
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            depth_from_top = L - 1 - i
            lr = base_lr / (decay ** depth_from_top)
            groups.append({"params": params, "lr": lr})

    return groups
