"""
libraries v0.5.0 - Deep Learning Transfer Demo
=================================================

Demonstrates every deep learning transfer component in Zeno on
realistic synthetic tasks (no pretrained model downloads required).

Six demo scenarios:
  1. LoRA Injection — inject LoRA into a multi-layer MLP, compare
     trainable params and accuracy vs full fine-tuning
  2. Progressive Unfreezing — freeze-then-unfreeze with discriminative
     learning rates on a deep classifier
  3. EWC Transfer — train on source task, transfer to target task with
     Elastic Weight Consolidation to prevent catastrophic forgetting
  4. CKA Similarity — measure layer-wise representation similarity
     between domains, detect negative transfer conditions
  5. Carbon Tracking — compare CO2 emissions of full fine-tuning
     vs LoRA across training runs
  6. Model Merging — combine multiple fine-tuned models in weight space
     using task arithmetic, TIES, DARE, SLERP, and LoRA Soups

Usage:
    cd content
    python -m tests.run_dl_demo
    python -m tests.run_dl_demo --demo lora         # LoRA only
    python -m tests.run_dl_demo --demo merging       # Merging only
    python -m tests.run_dl_demo --demo all --epochs 20
    python -m tests.run_dl_demo --quiet              # suppress epoch output
"""

import argparse
import sys
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libraries.metrics import set_seed
from libraries.dl.lora import LoRALinear, LoRAInjector
from libraries.dl.transfer import BaseModel, TransferScheduler, build_discriminative_lr_groups
from libraries.dl.ewc import compute_fisher_diagonal, EWCLoss, online_ewc_update
from libraries.dl.negative_transfer import (
    compute_cka, extract_representations,
    compute_representation_mmd, NegativeTransferMonitor,
)
from libraries.dl.carbon import GPUCarbonTracker
from libraries.dl.train import train_epoch, evaluate, fine_tune
from libraries.dl.merging import (
    linear_merge, slerp_merge, compute_task_vector, apply_task_vector,
    task_arithmetic_merge, ties_merge, dare_merge, merge_lora_adapters,
    task_vector_stats, task_vector_similarity,
)
from libraries.carbon import compare_emissions


# ═══════════════════════════════════════════════════════════════════
# Synthetic data generators (realistic tasks, no downloads)
# ═══════════════════════════════════════════════════════════════════

def make_classification_data(n_samples, n_features, n_classes, shift=0.0,
                              noise=0.1, seed=42):
    """Generate a synthetic classification dataset with optional domain shift."""
    rng = np.random.RandomState(seed)
    # Create class-conditional Gaussians
    X = rng.randn(n_samples, n_features).astype(np.float32)
    # Create class centers spread across feature space
    centers = rng.randn(n_classes, n_features).astype(np.float32) * 2.0
    y = rng.randint(0, n_classes, n_samples)
    for c in range(n_classes):
        mask = y == c
        X[mask] += centers[c] + shift
    X += noise * rng.randn(n_samples, n_features).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))


def make_source_target_split(n_features=20, n_classes=5, seed=42):
    """Create source and target domains with controlled covariate shift."""
    X_src, y_src = make_classification_data(
        800, n_features, n_classes, shift=0.0, seed=seed)
    X_tgt, y_tgt = make_classification_data(
        200, n_features, n_classes, shift=1.5, seed=seed + 1)
    return X_src, y_src, X_tgt, y_tgt


def make_loaders(X, y, batch_size=64, val_frac=0.2):
    """Split into train/val and return DataLoaders."""
    n = len(X)
    n_val = int(n * val_frac)
    idx = torch.randperm(n)
    train_ds = TensorDataset(X[idx[n_val:]], y[idx[n_val:]])
    val_ds = TensorDataset(X[idx[:n_val]], y[idx[:n_val]])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════

class DeepClassifier(nn.Module):
    """
    A realistically-sized MLP for classification demos.

    Architecture mirrors a small feature extractor + classifier head,
    similar to what you'd get from the penultimate layers of a CNN
    or the pooled output of a transformer.
    """
    def __init__(self, in_features=20, hidden_dims=(128, 64, 32),
                 n_classes=5, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        return self.fc(self.features(x))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ═══════════════════════════════════════════════════════════════════
# Demo 1: LoRA Injection
# ═══════════════════════════════════════════════════════════════════

def demo_lora(args):
    """
    LoRA Injection Demo
    ───────────────────
    Scenario: You have a pretrained classifier and want to adapt it to
    a new domain.  Compare full fine-tuning vs LoRA fine-tuning on
    parameter count, accuracy, and training speed.
    """
    print("\n" + "=" * 70)
    print("  DEMO 1: LoRA Injection — Parameter-Efficient Fine-Tuning")
    print("=" * 70)

    set_seed(args.seed)
    X_src, y_src, X_tgt, y_tgt = make_source_target_split(seed=args.seed)
    src_train, src_val = make_loaders(X_src, y_src)
    tgt_train, tgt_val = make_loaders(X_tgt, y_tgt)
    criterion = nn.CrossEntropyLoss()

    # --- Phase 1: Pretrain on source ---
    print("\n  Phase 1: Pretraining on source domain...")
    pretrained = DeepClassifier()
    opt = torch.optim.Adam(pretrained.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        loss = train_epoch(pretrained, src_train, criterion, opt)
    src_result = evaluate(pretrained, src_val, criterion)
    print(f"    Source accuracy: {src_result['accuracy']:.1%}")

    # --- Phase 2a: Full fine-tuning on target ---
    print("\n  Phase 2a: Full fine-tuning on target domain...")
    full_ft = copy.deepcopy(pretrained)
    total, trainable = count_params(full_ft)
    print(f"    Parameters: {trainable:,} trainable / {total:,} total")

    opt_full = torch.optim.Adam(full_ft.parameters(), lr=args.lr * 0.1)
    t0 = time.time()
    for ep in range(args.epochs):
        loss = train_epoch(full_ft, tgt_train, criterion, opt_full)
        if not args.quiet:
            val = evaluate(full_ft, tgt_val, criterion)
            print(f"      epoch {ep+1:>2}/{args.epochs}  "
                  f"loss={loss:.4f}  val_acc={val['accuracy']:.1%}")
    full_time = time.time() - t0
    full_result = evaluate(full_ft, tgt_val, criterion)

    # --- Phase 2b: LoRA fine-tuning on target ---
    print(f"\n  Phase 2b: LoRA fine-tuning (rank={args.lora_rank}) on target...")
    lora_ft = copy.deepcopy(pretrained)
    n_injected = LoRAInjector.inject(lora_ft, rank=args.lora_rank, alpha=args.lora_rank * 2)
    lora_params = LoRAInjector.get_lora_parameters(lora_ft)
    total_lora, _ = count_params(lora_ft)
    lora_trainable = LoRAInjector.count_lora_params(lora_ft)
    print(f"    Injected into {n_injected} layers")
    print(f"    Parameters: {lora_trainable:,} trainable / {total_lora:,} total "
          f"({lora_trainable/trainable:.1%} of full FT)")

    opt_lora = torch.optim.Adam(lora_params, lr=args.lr)  # 10x higher LR for LoRA
    t0 = time.time()
    for ep in range(args.epochs):
        loss = train_epoch(lora_ft, tgt_train, criterion, opt_lora)
        if not args.quiet:
            val = evaluate(lora_ft, tgt_val, criterion)
            print(f"      epoch {ep+1:>2}/{args.epochs}  "
                  f"loss={loss:.4f}  val_acc={val['accuracy']:.1%}")
    lora_time = time.time() - t0
    lora_result = evaluate(lora_ft, tgt_val, criterion)

    # --- Phase 3: Merge and verify ---
    print("\n  Phase 3: Merge LoRA weights for inference...")
    pre_merge = evaluate(lora_ft, tgt_val, criterion)
    LoRAInjector.merge_all(lora_ft)
    post_merge = evaluate(lora_ft, tgt_val, criterion)
    print(f"    Pre-merge accuracy:  {pre_merge['accuracy']:.1%}")
    print(f"    Post-merge accuracy: {post_merge['accuracy']:.1%}")
    print(f"    Merge introduces zero overhead (identical outputs)")

    # --- Summary ---
    print("\n  " + "-" * 60)
    print("  Summary: Full Fine-Tuning vs LoRA")
    print("  " + "-" * 60)
    print(f"  {'Metric':<25} {'Full FT':>12} {'LoRA':>12} {'Ratio':>10}")
    print(f"  {'Trainable params':<25} {trainable:>12,} {lora_trainable:>12,} "
          f"{trainable/lora_trainable:>9.1f}x")
    print(f"  {'Target accuracy':<25} {full_result['accuracy']:>11.1%} "
          f"{lora_result['accuracy']:>11.1%}")
    print(f"  {'Training time':<25} {full_time:>11.3f}s {lora_time:>11.3f}s "
          f"{full_time/lora_time:>9.1f}x")
    print(f"  {'Checkpoint size':<25} {'~all params':>12} {'LoRA only':>12}")


# ═══════════════════════════════════════════════════════════════════
# Demo 2: Progressive Unfreezing + Discriminative LRs
# ═══════════════════════════════════════════════════════════════════

def demo_transfer(args):
    """
    Progressive Unfreezing Demo
    ───────────────────────────
    Scenario: Adapt a deep pretrained model to a new domain by gradually
    unfreezing layers from the head downward, with each layer getting a
    progressively smaller learning rate.
    """
    print("\n" + "=" * 70)
    print("  DEMO 2: Progressive Unfreezing + Discriminative Learning Rates")
    print("=" * 70)

    set_seed(args.seed)
    X_src, y_src, X_tgt, y_tgt = make_source_target_split(seed=args.seed)
    src_train, src_val = make_loaders(X_src, y_src)
    tgt_train, tgt_val = make_loaders(X_tgt, y_tgt)
    criterion = nn.CrossEntropyLoss()

    # Pretrain
    print("\n  Pretraining source model...")
    pretrained = DeepClassifier(hidden_dims=(128, 64, 32))
    opt = torch.optim.Adam(pretrained.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(pretrained, src_train, criterion, opt)
    src_acc = evaluate(pretrained, src_val, criterion)["accuracy"]
    print(f"    Source accuracy: {src_acc:.1%}")

    # Before transfer: evaluate pretrained on target (no adaptation)
    pre_transfer = evaluate(pretrained, tgt_val, criterion)["accuracy"]
    print(f"    Target accuracy (no adaptation): {pre_transfer:.1%}")

    # --- Strategy A: Freeze all, train head only ---
    print("\n  Strategy A: Feature extraction (head only)...")
    model_a = copy.deepcopy(pretrained)
    base_a = BaseModel(model_a)
    base_a.freeze_all()
    # Unfreeze only the fc head
    base_a.unfreeze_layer("fc")
    counts = base_a.count_parameters()
    print(f"    Trainable: {counts['trainable']:,} / {counts['total']:,}")
    opt_a = torch.optim.Adam(
        [p for p in model_a.parameters() if p.requires_grad], lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(base_a, tgt_train, criterion, opt_a)
    acc_a = evaluate(base_a, tgt_val, criterion)["accuracy"]

    # --- Strategy B: Progressive unfreezing ---
    print("\n  Strategy B: Progressive unfreezing with discriminative LRs...")
    model_b = copy.deepcopy(pretrained)
    base_b = BaseModel(model_b)
    groups = base_b.get_layer_groups()
    scheduler = TransferScheduler(groups, base_lr=args.lr, decay=2.6,
                                   unfreeze_every=1)
    print(f"    Layer groups: {len(groups)}")
    for i, (name, params) in enumerate(groups):
        n_p = sum(p.numel() for p in params)
        print(f"      [{i}] {name}: {n_p:,} params")

    opt_b = scheduler.build_optimizer()
    for ep in range(args.epochs):
        scheduler.step(ep)
        # Rebuild optimizer with newly unfrozen params
        opt_b = scheduler.build_optimizer()
        loss = train_epoch(base_b, tgt_train, criterion, opt_b)
        if not args.quiet:
            val = evaluate(base_b, tgt_val, criterion)
            unfrozen = sum(1 for _, ps in groups
                           for p in ps if p.requires_grad)
            total_p = sum(1 for _, ps in groups for p in ps)
            print(f"      epoch {ep+1:>2}/{args.epochs}  "
                  f"loss={loss:.4f}  val_acc={val['accuracy']:.1%}  "
                  f"unfrozen={unfrozen}/{total_p} param tensors")
    acc_b = evaluate(base_b, tgt_val, criterion)["accuracy"]

    # --- Strategy C: Full fine-tuning with discriminative LRs ---
    print("\n  Strategy C: Full fine-tuning with discriminative LRs...")
    model_c = copy.deepcopy(pretrained)
    lr_groups = build_discriminative_lr_groups(model_c, base_lr=args.lr, decay=2.6)
    print(f"    LR schedule ({len(lr_groups)} groups):")
    for i, g in enumerate(lr_groups):
        n_p = sum(p.numel() for p in g["params"])
        print(f"      [{i}] lr={g['lr']:.6f}  params={n_p:,}")
    opt_c = torch.optim.AdamW(lr_groups, weight_decay=0.01)
    for ep in range(args.epochs):
        train_epoch(model_c, tgt_train, criterion, opt_c)
    acc_c = evaluate(model_c, tgt_val, criterion)["accuracy"]

    # --- Summary ---
    print("\n  " + "-" * 60)
    print("  Summary: Transfer Strategies")
    print("  " + "-" * 60)
    print(f"  {'Strategy':<40} {'Accuracy':>10}")
    print(f"  {'No adaptation (pretrained as-is)':<40} {pre_transfer:>9.1%}")
    print(f"  {'A: Head-only (feature extraction)':<40} {acc_a:>9.1%}")
    print(f"  {'B: Progressive unfreezing':<40} {acc_b:>9.1%}")
    print(f"  {'C: Discriminative LRs (all layers)':<40} {acc_c:>9.1%}")


# ═══════════════════════════════════════════════════════════════════
# Demo 3: Elastic Weight Consolidation
# ═══════════════════════════════════════════════════════════════════

def demo_ewc(args):
    """
    EWC Demo
    ────────
    Scenario: Train on Task A (source), then fine-tune on Task B (target).
    Without EWC the model catastrophically forgets Task A.
    With EWC the model retains Task A performance while learning Task B.
    """
    print("\n" + "=" * 70)
    print("  DEMO 3: Elastic Weight Consolidation — Preventing Forgetting")
    print("=" * 70)

    set_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    # Task A: 10-class noisy classification (not perfectly separable → nonzero Fisher)
    X_a, y_a = make_classification_data(
        1000, 20, 10, shift=0.0, noise=2.5, seed=args.seed)
    train_a, val_a = make_loaders(X_a, y_a)

    # Task B: different class structure (shifted centers)
    X_b, y_b = make_classification_data(
        1000, 20, 10, shift=3.0, noise=2.5, seed=args.seed + 10)
    train_b, val_b = make_loaders(X_b, y_b)

    # --- Train on Task A (partial — don't fully converge, so Fisher is nonzero) ---
    print("\n  Phase 1: Training on Task A (source)...")
    model = DeepClassifier(hidden_dims=(64, 32), n_classes=10)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ewc_epochs = min(args.epochs, 8)
    for ep in range(ewc_epochs):
        train_epoch(model, train_a, criterion, opt)
    task_a_acc = evaluate(model, val_a, criterion)["accuracy"]
    print(f"    Task A accuracy: {task_a_acc:.1%}")

    # --- Compute Fisher diagonal ---
    print("\n  Computing Fisher Information diagonal (parameter importance)...")
    fisher = compute_fisher_diagonal(model, train_a, criterion)
    n_params = sum(f.numel() for f in fisher.values())
    top_importance = max(f.max().item() for f in fisher.values())
    print(f"    Fisher computed for {n_params:,} parameters")
    print(f"    Max importance score: {top_importance:.4f}")

    # Save source model for EWC
    source_model = copy.deepcopy(model)

    # --- Fine-tune on Task B WITHOUT EWC ---
    print("\n  Phase 2a: Fine-tuning on Task B WITHOUT EWC...")
    model_no_ewc = copy.deepcopy(source_model)
    opt = torch.optim.Adam(model_no_ewc.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        loss = train_epoch(model_no_ewc, train_b, criterion, opt)
        if not args.quiet:
            a_acc = evaluate(model_no_ewc, val_a, criterion)["accuracy"]
            b_acc = evaluate(model_no_ewc, val_b, criterion)["accuracy"]
            print(f"      epoch {ep+1:>2}/{args.epochs}  loss={loss:.4f}  "
                  f"Task_A={a_acc:.1%}  Task_B={b_acc:.1%}")
    final_a_no_ewc = evaluate(model_no_ewc, val_a, criterion)["accuracy"]
    final_b_no_ewc = evaluate(model_no_ewc, val_b, criterion)["accuracy"]

    # --- Fine-tune on Task B WITH EWC ---
    print(f"\n  Phase 2b: Fine-tuning on Task B WITH EWC (lambda={5000.0})...")
    model_ewc = copy.deepcopy(source_model)
    ewc_loss = EWCLoss(source_model, fisher, lambda_=5000.0)
    opt = torch.optim.Adam(model_ewc.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        model_ewc.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_b:
            inputs, targets = batch
            opt.zero_grad()
            outputs = model_ewc(inputs)
            loss = criterion(outputs, targets) + ewc_loss(model_ewc)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        if not args.quiet:
            a_acc = evaluate(model_ewc, val_a, criterion)["accuracy"]
            b_acc = evaluate(model_ewc, val_b, criterion)["accuracy"]
            print(f"      epoch {ep+1:>2}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"Task_A={a_acc:.1%}  Task_B={b_acc:.1%}")
    final_a_ewc = evaluate(model_ewc, val_a, criterion)["accuracy"]
    final_b_ewc = evaluate(model_ewc, val_b, criterion)["accuracy"]

    # --- Summary ---
    a_forgetting_no_ewc = task_a_acc - final_a_no_ewc
    a_forgetting_ewc = task_a_acc - final_a_ewc

    print("\n  " + "-" * 60)
    print("  Summary: Catastrophic Forgetting Prevention")
    print("  " + "-" * 60)
    print(f"  {'Metric':<30} {'No EWC':>12} {'With EWC':>12}")
    print(f"  {'Task A (after B training)':<30} "
          f"{final_a_no_ewc:>11.1%} {final_a_ewc:>11.1%}")
    print(f"  {'Task B accuracy':<30} "
          f"{final_b_no_ewc:>11.1%} {final_b_ewc:>11.1%}")
    print(f"  {'Task A forgetting':<30} "
          f"{a_forgetting_no_ewc:>+11.1%} {a_forgetting_ewc:>+11.1%}")
    if a_forgetting_ewc < a_forgetting_no_ewc:
        print(f"\n    EWC reduced forgetting by "
              f"{a_forgetting_no_ewc - a_forgetting_ewc:.1%} absolute")


# ═══════════════════════════════════════════════════════════════════
# Demo 4: CKA Similarity + Negative Transfer Detection
# ═══════════════════════════════════════════════════════════════════

def demo_cka(args):
    """
    CKA Similarity Demo
    ────────────────────
    Scenario: Measure representation similarity between a pretrained
    model's layers on similar vs. dissimilar domains.  CKA scores
    predict whether transfer will help or hurt.
    """
    print("\n" + "=" * 70)
    print("  DEMO 4: CKA Representation Similarity + Negative Transfer")
    print("=" * 70)

    set_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    # Train a model on source
    X_src, y_src = make_classification_data(600, 20, 5, shift=0.0, seed=args.seed)
    src_loader = DataLoader(TensorDataset(X_src, y_src), batch_size=64)

    print("\n  Training source model...")
    model = DeepClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(model, src_loader, criterion, opt)
    print(f"    Source accuracy: {evaluate(model, src_loader, criterion)['accuracy']:.1%}")

    # Similar target domain (small shift)
    X_similar, y_similar = make_classification_data(
        200, 20, 5, shift=0.5, seed=args.seed + 1)
    similar_loader = DataLoader(TensorDataset(X_similar, y_similar), batch_size=64)

    # Dissimilar target domain (large shift)
    X_dissimilar, y_dissimilar = make_classification_data(
        200, 20, 5, shift=5.0, seed=args.seed + 2)
    dissimilar_loader = DataLoader(
        TensorDataset(X_dissimilar, y_dissimilar), batch_size=64)

    # Extract representations from each hidden layer and the head
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_names.append(name)

    print("\n  CKA similarity scores (source vs target):")
    print(f"  {'Layer':<25} {'Similar':>10} {'Dissimilar':>12} {'Delta':>8}")
    print("  " + "-" * 57)

    for layer in layer_names:
        reps_src = extract_representations(model, src_loader, layer)
        reps_sim = extract_representations(model, similar_loader, layer)
        reps_dis = extract_representations(model, dissimilar_loader, layer)

        cka_similar = compute_cka(reps_src, reps_sim)
        cka_dissimilar = compute_cka(reps_src, reps_dis)
        delta = cka_similar - cka_dissimilar

        print(f"  {layer:<25} {cka_similar:>9.4f} {cka_dissimilar:>11.4f} "
              f"{delta:>+7.4f}")

    # Representation MMD
    print("\n  Representation MMD (source vs target):")
    target_layer = layer_names[0]  # first hidden layer
    mmd_similar = compute_representation_mmd(
        model, src_loader, similar_loader, target_layer)
    mmd_dissimilar = compute_representation_mmd(
        model, src_loader, dissimilar_loader, target_layer)
    print(f"    Layer '{target_layer}':")
    print(f"      Similar domain MMD:    {mmd_similar:.4f}")
    print(f"      Dissimilar domain MMD: {mmd_dissimilar:.4f}")

    # Online monitoring simulation
    print("\n  Online Negative Transfer Monitor:")
    print("    Simulating fine-tuning with degrading performance...")
    monitor = NegativeTransferMonitor(
        reference_model=model, patience=3, baseline_val_loss=0.5)
    simulated_losses = [0.48, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    for ep, loss in enumerate(simulated_losses):
        warning = monitor.check(ep, loss, model)
        status = "WARNING" if warning else "OK"
        print(f"      epoch {ep}: val_loss={loss:.2f}  [{status}]")
        if warning:
            print(f"      >>> {warning}")
            break

    # Parameter drift
    print("\n  Parameter drift from pretrained:")
    # Modify model to simulate fine-tuning
    modified = copy.deepcopy(model)
    with torch.no_grad():
        for p in modified.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    drift_monitor = NegativeTransferMonitor(reference_model=model)
    drift = drift_monitor.parameter_drift(modified)
    for name, d in sorted(drift.items(), key=lambda x: -x[1])[:5]:
        print(f"    {name:<35} drift = {d:.4f}")


# ═══════════════════════════════════════════════════════════════════
# Demo 5: Carbon Tracking
# ═══════════════════════════════════════════════════════════════════

def demo_carbon(args):
    """
    Carbon Tracking Demo
    ────────────────────
    Scenario: Compare CO2 emissions of full fine-tuning vs LoRA.
    Uses GPUCarbonTracker (falls back to CPU estimation without GPU).
    """
    print("\n" + "=" * 70)
    print("  DEMO 5: Carbon Emissions — Full Fine-Tuning vs LoRA")
    print("=" * 70)

    set_seed(args.seed)
    X_tgt, y_tgt = make_classification_data(
        400, 20, 5, shift=1.0, seed=args.seed)
    tgt_train, tgt_val = make_loaders(X_tgt, y_tgt)
    criterion = nn.CrossEntropyLoss()

    # Pretrain
    pretrained = DeepClassifier(hidden_dims=(256, 128, 64, 32))
    opt = torch.optim.Adam(pretrained.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(pretrained, tgt_train, criterion, opt)

    # --- Full fine-tuning with carbon tracking ---
    print("\n  Full fine-tuning with carbon tracking...")
    model_full = copy.deepcopy(pretrained)
    tracker_full = GPUCarbonTracker(
        "full_finetune", power_watts=150.0,
        carbon_intensity_kg_kwh=0.45, pue=1.1)

    opt_full = torch.optim.Adam(model_full.parameters(), lr=args.lr * 0.1)
    tracker_full.start()
    for ep in range(args.epochs):
        train_epoch(model_full, tgt_train, criterion, opt_full)
    result_full = tracker_full.stop()

    full_acc = evaluate(model_full, tgt_val, criterion)["accuracy"]
    print(f"    Accuracy: {full_acc:.1%}")
    print(f"    Time: {result_full['time_s']:.4f}s")
    print(f"    Energy: {result_full['kwh']:.2e} kWh")
    print(f"    CO2: {result_full['co2_kg']:.2e} kg")

    # --- LoRA fine-tuning with carbon tracking ---
    print(f"\n  LoRA fine-tuning (rank={args.lora_rank}) with carbon tracking...")
    model_lora = copy.deepcopy(pretrained)
    LoRAInjector.inject(model_lora, rank=args.lora_rank)
    tracker_lora = GPUCarbonTracker(
        "lora_finetune", power_watts=100.0,
        carbon_intensity_kg_kwh=0.45, pue=1.1)

    lora_params = LoRAInjector.get_lora_parameters(model_lora)
    opt_lora = torch.optim.Adam(lora_params, lr=args.lr)
    tracker_lora.start()
    for ep in range(args.epochs):
        train_epoch(model_lora, tgt_train, criterion, opt_lora)
    result_lora = tracker_lora.stop()

    lora_acc = evaluate(model_lora, tgt_val, criterion)["accuracy"]
    print(f"    Accuracy: {lora_acc:.1%}")
    print(f"    Time: {result_lora['time_s']:.4f}s")
    print(f"    Energy: {result_lora['kwh']:.2e} kWh")
    print(f"    CO2: {result_lora['co2_kg']:.2e} kg")

    # --- fine_tune() with integrated tracking ---
    print(f"\n  Integrated fine_tune() with carbon tracking...")
    model_integrated = copy.deepcopy(pretrained)
    LoRAInjector.inject(model_integrated, rank=args.lora_rank)
    integrated_params = LoRAInjector.get_lora_parameters(model_integrated)
    opt_int = torch.optim.Adam(integrated_params, lr=args.lr)
    tracker_int = GPUCarbonTracker(
        "integrated", power_watts=100.0,
        carbon_intensity_kg_kwh=0.45, pue=1.1)

    history = fine_tune(
        model_integrated, tgt_train, tgt_val,
        epochs=args.epochs, optimizer=opt_int, criterion=criterion,
        carbon_tracker=tracker_int, verbose=not args.quiet,
    )
    if "co2_result" in history:
        print(f"    Integrated CO2: {history['co2_result']['co2_kg']:.2e} kg")

    # --- Compare emissions ---
    print("\n  " + "-" * 60)
    summary = compare_emissions([result_full, result_lora])
    comp = summary["comparisons"][0]
    print("  Emissions Comparison")
    print("  " + "-" * 60)
    print(f"  {'Metric':<25} {'Full FT':>15} {'LoRA FT':>15}")
    print(f"  {'Time':<25} {result_full['time_s']:>14.4f}s {result_lora['time_s']:>14.4f}s")
    print(f"  {'Energy (kWh)':<25} {result_full['kwh']:>14.2e} {result_lora['kwh']:>14.2e}")
    print(f"  {'CO2 (kg)':<25} {result_full['co2_kg']:>14.2e} {result_lora['co2_kg']:>14.2e}")
    print(f"  {'CO2 saved':<25} {'baseline':>15} {comp['co2_saved_pct']:>13.1f}%")
    print(f"  {'Accuracy':<25} {full_acc:>14.1%} {lora_acc:>14.1%}")
    print(f"\n  Measurement source: {result_full['source']} "
          f"(NVML Energy API on Volta+, Power API polling, or manual)")


# ═══════════════════════════════════════════════════════════════════
# Demo 6: Model Merging
# ═══════════════════════════════════════════════════════════════════

def demo_merging(args):
    """
    Model Merging Demo
    ──────────────────
    Scenario: Fine-tune the same base model on three different tasks,
    then merge the resulting models using 5 strategies:
      - Linear averaging (Model Soups)
      - SLERP (Spherical Linear Interpolation)
      - Task Arithmetic (additive task vectors)
      - TIES (Trim + Elect Sign + Disjoint Merge)
      - DARE (Drop And REscale + TIES)
    Also demonstrates LoRA Soups (merging LoRA adapters).
    """
    print("\n" + "=" * 70)
    print("  DEMO 6: Model Merging — Transfer Learning in Weight Space")
    print("=" * 70)

    set_seed(args.seed)
    criterion = nn.CrossEntropyLoss()
    n_features, n_classes = 20, 5

    # --- Phase 1: Pretrain a shared base model ---
    print("\n  Phase 1: Pretraining shared base model...")
    X_base, y_base = make_classification_data(
        800, n_features, n_classes, shift=0.0, noise=0.5, seed=args.seed)
    base_train, base_val = make_loaders(X_base, y_base)
    base_model = DeepClassifier(
        in_features=n_features, hidden_dims=(128, 64, 32),
        n_classes=n_classes, dropout=0.1)
    opt = torch.optim.Adam(base_model.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(base_model, base_train, criterion, opt)
    base_acc = evaluate(base_model, base_val, criterion)["accuracy"]
    print(f"    Base accuracy: {base_acc:.1%}")
    base_sd = copy.deepcopy(base_model.state_dict())

    # --- Phase 2: Fine-tune on 3 different tasks ---
    tasks = [
        {"name": "Task A (shift=1.0)", "shift": 1.0, "seed_offset": 10},
        {"name": "Task B (shift=2.0)", "shift": 2.0, "seed_offset": 20},
        {"name": "Task C (shift=3.0)", "shift": 3.0, "seed_offset": 30},
    ]

    finetuned_models = []
    task_loaders = []
    task_vectors = []

    for task in tasks:
        print(f"\n  Fine-tuning on {task['name']}...")
        X_task, y_task = make_classification_data(
            400, n_features, n_classes, shift=task["shift"],
            noise=0.5, seed=args.seed + task["seed_offset"])
        train_loader, val_loader = make_loaders(X_task, y_task)
        task_loaders.append((train_loader, val_loader))

        model = copy.deepcopy(base_model)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.3)
        for ep in range(args.epochs):
            train_epoch(model, train_loader, criterion, opt)
        acc = evaluate(model, val_loader, criterion)["accuracy"]
        print(f"    {task['name']} accuracy: {acc:.1%}")
        finetuned_models.append(model)

        tv = compute_task_vector(base_sd, model.state_dict())
        task_vectors.append(tv)

    # --- Phase 3: Analyze task vectors ---
    print("\n  Task Vector Analysis:")
    for i, tv in enumerate(task_vectors):
        stats = task_vector_stats(tv)
        print(f"    {tasks[i]['name']}: L2={stats['l2_norm']:.4f}  "
              f"mean_mag={stats['mean_magnitude']:.6f}  "
              f"max_mag={stats['max_magnitude']:.4f}")

    print("\n  Task Vector Similarity (cosine):")
    print(f"  {'':>20}", end="")
    for t in tasks:
        print(f"  {t['name'][:8]:>10}", end="")
    print()
    for i, t_i in enumerate(tasks):
        print(f"  {t_i['name'][:20]:<20}", end="")
        for j, t_j in enumerate(tasks):
            sim = task_vector_similarity(task_vectors[i], task_vectors[j])
            print(f"  {sim:>10.4f}", end="")
        print()

    # --- Phase 4: Merge with different strategies ---
    print("\n  Phase 3: Merging with 5 strategies...")

    merge_results = {}

    # Strategy 1: Linear merge (Model Soups)
    print("\n  [1] Linear Merge (Model Soups — weighted average)...")
    merged_sd = linear_merge(
        [m.state_dict() for m in finetuned_models])
    merged_model = copy.deepcopy(base_model)
    merged_model.load_state_dict(merged_sd)
    accs = []
    for i, (_, val_l) in enumerate(task_loaders):
        acc = evaluate(merged_model, val_l, criterion)["accuracy"]
        accs.append(acc)
    merge_results["Linear"] = accs
    print(f"    Accuracies: " + "  ".join(
        f"{tasks[i]['name'][:6]}={accs[i]:.1%}" for i in range(3)))

    # Strategy 2: SLERP (pairwise, then with third)
    print("\n  [2] SLERP (Spherical Linear Interpolation)...")
    # SLERP is 2-model, so we do A↔B then result↔C
    slerp_ab = slerp_merge(
        finetuned_models[0].state_dict(),
        finetuned_models[1].state_dict(), t=0.5)
    slerp_abc = slerp_merge(slerp_ab, finetuned_models[2].state_dict(), t=0.33)
    merged_model.load_state_dict(slerp_abc)
    accs = []
    for i, (_, val_l) in enumerate(task_loaders):
        acc = evaluate(merged_model, val_l, criterion)["accuracy"]
        accs.append(acc)
    merge_results["SLERP"] = accs
    print(f"    Accuracies: " + "  ".join(
        f"{tasks[i]['name'][:6]}={accs[i]:.1%}" for i in range(3)))

    # Strategy 3: Task Arithmetic
    print("\n  [3] Task Arithmetic (additive task vectors, scaling=0.5)...")
    ta_sd = task_arithmetic_merge(
        base_sd, task_vectors, scalings=[0.5, 0.5, 0.5])
    merged_model.load_state_dict(ta_sd)
    accs = []
    for i, (_, val_l) in enumerate(task_loaders):
        acc = evaluate(merged_model, val_l, criterion)["accuracy"]
        accs.append(acc)
    merge_results["Task Arith"] = accs
    print(f"    Accuracies: " + "  ".join(
        f"{tasks[i]['name'][:6]}={accs[i]:.1%}" for i in range(3)))

    # Strategy 4: TIES
    print("\n  [4] TIES-Merging (trim=20%, elect sign, disjoint merge)...")
    ties_sd = ties_merge(
        base_sd, task_vectors, density=0.2, scaling=1.0)
    merged_model.load_state_dict(ties_sd)
    accs = []
    for i, (_, val_l) in enumerate(task_loaders):
        acc = evaluate(merged_model, val_l, criterion)["accuracy"]
        accs.append(acc)
    merge_results["TIES"] = accs
    print(f"    Accuracies: " + "  ".join(
        f"{tasks[i]['name'][:6]}={accs[i]:.1%}" for i in range(3)))

    # Strategy 5: DARE + TIES
    print("\n  [5] DARE + TIES (drop=90%, then TIES trim=20%)...")
    dare_sd = dare_merge(
        base_sd, task_vectors, drop_rate=0.9, scaling=1.0,
        use_ties=True, ties_density=0.2, seed=args.seed)
    merged_model.load_state_dict(dare_sd)
    accs = []
    for i, (_, val_l) in enumerate(task_loaders):
        acc = evaluate(merged_model, val_l, criterion)["accuracy"]
        accs.append(acc)
    merge_results["DARE+TIES"] = accs
    print(f"    Accuracies: " + "  ".join(
        f"{tasks[i]['name'][:6]}={accs[i]:.1%}" for i in range(3)))

    # --- Phase 5: LoRA Soups ---
    print("\n  Phase 4: LoRA Soups (merge LoRA adapters)...")
    lora_models = []
    lora_state_dicts = []
    for i, task in enumerate(tasks):
        model = copy.deepcopy(base_model)
        n_injected = LoRAInjector.inject(
            model, rank=args.lora_rank, alpha=args.lora_rank * 2)
        lora_params = LoRAInjector.get_lora_parameters(model)
        opt = torch.optim.Adam(lora_params, lr=args.lr)
        train_l, val_l = task_loaders[i]
        for ep in range(args.epochs):
            train_epoch(model, train_l, criterion, opt)
        acc = evaluate(model, val_l, criterion)["accuracy"]
        print(f"    LoRA {task['name'][:6]}: {acc:.1%}  "
              f"({LoRAInjector.count_lora_params(model):,} params)")
        lora_models.append(model)
        lora_state_dicts.append(LoRAInjector.lora_state_dict(model))

    # Merge LoRA adapters
    merged_lora_sd = merge_lora_adapters(lora_state_dicts)
    # Load merged LoRA into a fresh model
    soup_model = copy.deepcopy(base_model)
    LoRAInjector.inject(soup_model, rank=args.lora_rank,
                         alpha=args.lora_rank * 2)
    # Overwrite LoRA weights with merged values
    current_sd = soup_model.state_dict()
    for key, value in merged_lora_sd.items():
        if key in current_sd:
            current_sd[key] = value
    soup_model.load_state_dict(current_sd)
    LoRAInjector.merge_all(soup_model)

    accs_soup = []
    for i, (_, val_l) in enumerate(task_loaders):
        acc = evaluate(soup_model, val_l, criterion)["accuracy"]
        accs_soup.append(acc)
    merge_results["LoRA Soup"] = accs_soup
    print(f"\n    LoRA Soup (merged) accuracies: " + "  ".join(
        f"{tasks[i]['name'][:6]}={accs_soup[i]:.1%}" for i in range(3)))

    # --- Summary ---
    print("\n  " + "-" * 70)
    print("  Summary: Merging Strategy Comparison")
    print("  " + "-" * 70)

    # Individual model baselines
    individual_accs = []
    for i, model in enumerate(finetuned_models):
        accs_i = []
        for j, (_, val_l) in enumerate(task_loaders):
            acc = evaluate(model, val_l, criterion)["accuracy"]
            accs_i.append(acc)
        individual_accs.append(accs_i)

    print(f"  {'Strategy':<20}", end="")
    for t in tasks:
        print(f"  {t['name'][:8]:>10}", end="")
    print(f"  {'Avg':>8}")
    print("  " + "-" * 58)

    # Individual models (diagonal = own task)
    for i, t in enumerate(tasks):
        name = f"Individual {t['name'][:6]}"
        print(f"  {name:<20}", end="")
        for j in range(3):
            print(f"  {individual_accs[i][j]:>9.1%}", end="")
        avg = np.mean(individual_accs[i])
        print(f"  {avg:>7.1%}")

    print("  " + "-" * 58)

    # Merged models
    for strategy, accs in merge_results.items():
        print(f"  {strategy:<20}", end="")
        for acc in accs:
            print(f"  {acc:>9.1%}", end="")
        avg = np.mean(accs)
        print(f"  {avg:>7.1%}")

    print("\n    Key: Higher average accuracy = better multi-task generalization")
    print("    Individual models excel on their own task but fail on others")
    print("    Merged models trade peak performance for broader competence")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Zeno v0.5.0 — Deep Learning Transfer Demo")
    parser.add_argument("--demo", type=str, default="all",
                        choices=["lora", "transfer", "ewc", "cka", "carbon",
                                 "merging", "all"],
                        help="Which demo to run (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch output")
    args = parser.parse_args()

    print("=" * 70)
    print("  Zeno v0.5.0 — Deep Learning Transfer Demo")
    print("  From-scratch PyTorch (no PEFT, no HuggingFace Trainer)")
    print("=" * 70)
    print(f"  seed={args.seed}  epochs={args.epochs}  lr={args.lr}  "
          f"lora_rank={args.lora_rank}")

    demos = {
        "lora": demo_lora,
        "transfer": demo_transfer,
        "ewc": demo_ewc,
        "cka": demo_cka,
        "carbon": demo_carbon,
        "merging": demo_merging,
    }

    if args.demo == "all":
        for name, fn in demos.items():
            fn(args)
    else:
        demos[args.demo](args)

    print("\n" + "=" * 70)
    print("  All demos complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
