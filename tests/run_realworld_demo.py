"""
libraries v0.5.0 - Real-World Transfer Learning Demo
======================================================

Demonstrates the FULL Zeno DL pipeline on real-world datasets from
sklearn (no internet required).  This is the end-to-end validation
that exercises every component on genuine data with natural domain shift.

Pipeline exercised:
  1. Load real data with natural covariate shift (Breast Cancer, California Housing)
  2. Build PyTorch models on tabular features
  3. Pretrain on source domain → transfer to target domain
  4. Compare: no transfer vs LoRA vs progressive unfreezing vs EWC
  5. Measure CKA similarity between source/target representations
  6. Merge models fine-tuned on different domains
  7. Track CO2 emissions across all strategies
  8. Use LoRA-Flow for learned adapter gating

Usage:
    cd content
    python -m tests.run_realworld_demo
    python -m tests.run_realworld_demo --demo breast_cancer
    python -m tests.run_realworld_demo --demo housing
    python -m tests.run_realworld_demo --demo all --epochs 20
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
from libraries.dl.ewc import compute_fisher_diagonal, EWCLoss
from libraries.dl.negative_transfer import (
    compute_cka, extract_representations, NegativeTransferMonitor,
)
from libraries.dl.carbon import GPUCarbonTracker
from libraries.dl.train import train_epoch, evaluate, fine_tune
from libraries.dl.merging import (
    linear_merge, slerp_merge, compute_task_vector, apply_task_vector,
    task_arithmetic_merge, ties_merge, dare_merge, merge_lora_adapters,
    LoRAFlow, train_lora_flow,
    task_vector_stats, task_vector_similarity,
)
from libraries.carbon import compare_emissions


# ═══════════════════════════════════════════════════════════════════
# Data Loading Utilities
# ═══════════════════════════════════════════════════════════════════

def load_breast_cancer_pytorch(seed=42, test_frac=0.25):
    """
    Load Breast Cancer Wisconsin with natural domain split.

    Source domain: smaller tumors (mean radius ≤ median)
    Target domain: larger tumors (mean radius > median)

    Returns dict with PyTorch DataLoaders for source/target train/test.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    # Natural domain split by tumor size
    median_radius = df["mean radius"].median()
    src_df = df[df["mean radius"] <= median_radius].copy()
    tgt_df = df[df["mean radius"] > median_radius].copy()

    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)

    # Train/test split
    rng = np.random.RandomState(seed)

    def split_df(df_, frac):
        idx = rng.permutation(len(df_))
        n_test = int(frac * len(df_))
        return df_.iloc[idx[n_test:]], df_.iloc[idx[:n_test]]

    src_train, src_test = split_df(src_df, test_frac)
    tgt_train, tgt_test = split_df(tgt_df, test_frac)

    # Fit scaler on union of training data
    import pandas as pd
    union_train = pd.concat([src_train, tgt_train], axis=0)
    scaler = StandardScaler().fit(union_train[feature_cols].to_numpy())

    def to_tensors(split):
        X = scaler.transform(split[feature_cols].to_numpy()).astype(np.float32)
        y = split[target_col].to_numpy().astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(y)

    X_src_tr, y_src_tr = to_tensors(src_train)
    X_src_te, y_src_te = to_tensors(src_test)
    X_tgt_tr, y_tgt_tr = to_tensors(tgt_train)
    X_tgt_te, y_tgt_te = to_tensors(tgt_test)

    return {
        "name": "Breast Cancer Wisconsin",
        "task": "classification",
        "n_features": n_features,
        "n_classes": 2,
        "domain_split": "tumor size (mean radius ≤ / > median)",
        "src_train": DataLoader(TensorDataset(X_src_tr, y_src_tr), batch_size=32, shuffle=True),
        "src_test": DataLoader(TensorDataset(X_src_te, y_src_te), batch_size=32),
        "tgt_train": DataLoader(TensorDataset(X_tgt_tr, y_tgt_tr), batch_size=32, shuffle=True),
        "tgt_test": DataLoader(TensorDataset(X_tgt_te, y_tgt_te), batch_size=32),
        "src_train_size": len(X_src_tr),
        "src_test_size": len(X_src_te),
        "tgt_train_size": len(X_tgt_tr),
        "tgt_test_size": len(X_tgt_te),
    }


def load_california_housing_pytorch(seed=42, test_frac=0.25):
    """
    Load California Housing with natural domain split.

    Source domain: Northern CA (latitude ≥ median)
    Target domain: Southern CA (latitude < median)

    Returns dict with PyTorch DataLoaders.
    """
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler

    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"MedHouseVal": "target"})

    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)

    # Natural domain split by latitude (North vs South CA)
    median_lat = df["Latitude"].median()
    src_df = df[df["Latitude"] >= median_lat].copy()
    tgt_df = df[df["Latitude"] < median_lat].copy()

    rng = np.random.RandomState(seed)

    def split_df(df_, frac):
        idx = rng.permutation(len(df_))
        n_test = int(frac * len(df_))
        return df_.iloc[idx[n_test:]], df_.iloc[idx[:n_test]]

    src_train, src_test = split_df(src_df, test_frac)
    tgt_train, tgt_test = split_df(tgt_df, test_frac)

    # Fit scalers on union of training data
    import pandas as pd
    union_train = pd.concat([src_train, tgt_train], axis=0)
    sc_x = StandardScaler().fit(union_train[feature_cols].to_numpy())
    sc_y = StandardScaler().fit(union_train[target_col].to_numpy().reshape(-1, 1))

    def to_tensors(split):
        X = sc_x.transform(split[feature_cols].to_numpy()).astype(np.float32)
        y = sc_y.transform(
            split[target_col].to_numpy().reshape(-1, 1)
        ).ravel().astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)

    X_src_tr, y_src_tr = to_tensors(src_train)
    X_src_te, y_src_te = to_tensors(src_test)
    X_tgt_tr, y_tgt_tr = to_tensors(tgt_train)
    X_tgt_te, y_tgt_te = to_tensors(tgt_test)

    return {
        "name": "California Housing",
        "task": "regression",
        "n_features": n_features,
        "n_classes": 1,  # regression
        "domain_split": "latitude (Northern CA ≥ / Southern CA < median)",
        "src_train": DataLoader(TensorDataset(X_src_tr, y_src_tr), batch_size=64, shuffle=True),
        "src_test": DataLoader(TensorDataset(X_src_te, y_src_te), batch_size=64),
        "tgt_train": DataLoader(TensorDataset(X_tgt_tr, y_tgt_tr), batch_size=64, shuffle=True),
        "tgt_test": DataLoader(TensorDataset(X_tgt_te, y_tgt_te), batch_size=64),
        "src_train_size": len(X_src_tr),
        "src_test_size": len(X_src_te),
        "tgt_train_size": len(X_tgt_tr),
        "tgt_test_size": len(X_tgt_te),
    }


# ═══════════════════════════════════════════════════════════════════
# Model Factories
# ═══════════════════════════════════════════════════════════════════

class TabularClassifier(nn.Module):
    """MLP classifier for tabular data."""
    def __init__(self, n_features, hidden_dims=(128, 64, 32), n_classes=2,
                 dropout=0.1):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, n_classes)

    def forward(self, x):
        return self.fc(self.features(x))


class TabularRegressor(nn.Module):
    """MLP regressor for tabular data."""
    def __init__(self, n_features, hidden_dims=(128, 64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, 1)

    def forward(self, x):
        return self.fc(self.features(x)).squeeze(-1)


def build_model(data_info):
    """Build the appropriate model for the dataset."""
    if data_info["task"] == "classification":
        return TabularClassifier(
            n_features=data_info["n_features"],
            n_classes=data_info["n_classes"],
            hidden_dims=(128, 64, 32),
        )
    else:
        return TabularRegressor(
            n_features=data_info["n_features"],
            hidden_dims=(128, 64, 32),
        )


def get_criterion(task):
    if task == "classification":
        return nn.CrossEntropyLoss()
    else:
        return nn.MSELoss()


def eval_metric_name(task):
    return "accuracy" if task == "classification" else "loss"


def eval_metric_fmt(task, result):
    if task == "classification":
        return f"{result['accuracy']:.1%}"
    else:
        return f"MSE={result['loss']:.4f}"


# ═══════════════════════════════════════════════════════════════════
# Demo: Breast Cancer — Full Classification Pipeline
# ═══════════════════════════════════════════════════════════════════

def demo_breast_cancer(args):
    """
    Breast Cancer Wisconsin — Real-World Classification Transfer
    ─────────────────────────────────────────────────────────────
    Source: small tumors (mean radius ≤ median) — benign vs malignant
    Target: large tumors (mean radius > median) — harder classification

    Exercises: LoRA, progressive unfreezing, EWC, CKA, merging, carbon, LoRA-Flow
    """
    print("\n" + "=" * 72)
    print("  REAL-WORLD DEMO: Breast Cancer Wisconsin — Classification Transfer")
    print("=" * 72)

    set_seed(args.seed)
    data = load_breast_cancer_pytorch(seed=args.seed)
    criterion = get_criterion(data["task"])

    print(f"\n  Dataset: {data['name']}")
    print(f"  Task: Binary classification (benign vs malignant)")
    print(f"  Domain split: {data['domain_split']}")
    print(f"  Source: {data['src_train_size']} train / {data['src_test_size']} test")
    print(f"  Target: {data['tgt_train_size']} train / {data['tgt_test_size']} test")
    print(f"  Features: {data['n_features']}")

    # ─── Phase 1: Pretrain on source domain ───
    print("\n  ── Phase 1: Pretraining on source domain (small tumors) ──")
    pretrained = build_model(data)
    opt = torch.optim.Adam(pretrained.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(pretrained, data["src_train"], criterion, opt)
    src_result = evaluate(pretrained, data["src_test"], criterion)
    print(f"    Source test: {eval_metric_fmt(data['task'], src_result)}")
    pre_transfer = evaluate(pretrained, data["tgt_test"], criterion)
    print(f"    Target test (no adaptation): {eval_metric_fmt(data['task'], pre_transfer)}")

    results = {"No transfer": pre_transfer}

    # ─── Phase 2: CKA Similarity Analysis ───
    print("\n  ── Phase 2: CKA Representation Similarity (source ↔ target) ──")
    layer_names = []
    for name, mod in pretrained.named_modules():
        if isinstance(mod, nn.Linear):
            layer_names.append(name)

    print(f"  {'Layer':<25} {'CKA':>10}")
    print("  " + "-" * 37)
    for layer in layer_names:
        reps_src = extract_representations(pretrained, data["src_test"], layer)
        reps_tgt = extract_representations(pretrained, data["tgt_test"], layer)
        cka = compute_cka(reps_src, reps_tgt)
        print(f"  {layer:<25} {cka:>9.4f}")

    # ─── Phase 3: Transfer Strategies ───
    print("\n  ── Phase 3: Transfer Learning Strategies ──")

    # Strategy A: Full fine-tuning
    print("\n  [A] Full fine-tuning on target domain...")
    tracker_full = GPUCarbonTracker("full_ft", power_watts=100.0)
    model_full = copy.deepcopy(pretrained)
    opt_full = torch.optim.Adam(model_full.parameters(), lr=args.lr * 0.1)
    tracker_full.start()
    for ep in range(args.epochs):
        train_epoch(model_full, data["tgt_train"], criterion, opt_full)
    carbon_full = tracker_full.stop()
    result_full = evaluate(model_full, data["tgt_test"], criterion)
    print(f"    Target: {eval_metric_fmt(data['task'], result_full)}")
    total_p = sum(p.numel() for p in model_full.parameters())
    train_p = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    print(f"    Trainable: {train_p:,} / {total_p:,} params")
    results["Full FT"] = result_full

    # Strategy B: LoRA
    print(f"\n  [B] LoRA fine-tuning (rank={args.lora_rank}) on target...")
    tracker_lora = GPUCarbonTracker("lora_ft", power_watts=80.0)
    model_lora = copy.deepcopy(pretrained)
    n_injected = LoRAInjector.inject(model_lora, rank=args.lora_rank,
                                      alpha=args.lora_rank * 2)
    lora_params = LoRAInjector.get_lora_parameters(model_lora)
    lora_trainable = LoRAInjector.count_lora_params(model_lora)
    print(f"    Injected into {n_injected} layers, {lora_trainable:,} trainable params")
    opt_lora = torch.optim.Adam(lora_params, lr=args.lr)
    tracker_lora.start()
    for ep in range(args.epochs):
        train_epoch(model_lora, data["tgt_train"], criterion, opt_lora)
    carbon_lora = tracker_lora.stop()
    result_lora = evaluate(model_lora, data["tgt_test"], criterion)
    print(f"    Target: {eval_metric_fmt(data['task'], result_lora)}")
    results["LoRA"] = result_lora

    # Strategy C: Progressive Unfreezing
    print("\n  [C] Progressive unfreezing with discriminative LRs...")
    model_prog = copy.deepcopy(pretrained)
    base_prog = BaseModel(model_prog)
    groups = base_prog.get_layer_groups()
    scheduler = TransferScheduler(groups, base_lr=args.lr, decay=2.6)
    for ep in range(args.epochs):
        scheduler.step(ep)
        opt_prog = scheduler.build_optimizer()
        train_epoch(base_prog, data["tgt_train"], criterion, opt_prog)
    result_prog = evaluate(base_prog, data["tgt_test"], criterion)
    print(f"    Target: {eval_metric_fmt(data['task'], result_prog)}")
    results["Progressive"] = result_prog

    # Strategy D: EWC Transfer
    print("\n  [D] EWC-regularized fine-tuning (preventing source forgetting)...")
    model_ewc = copy.deepcopy(pretrained)
    fisher = compute_fisher_diagonal(pretrained, data["src_train"], criterion)
    ewc_loss = EWCLoss(pretrained, fisher, lambda_=1000.0)
    opt_ewc = torch.optim.Adam(model_ewc.parameters(), lr=args.lr * 0.1)
    for ep in range(args.epochs):
        model_ewc.train()
        for batch in data["tgt_train"]:
            inputs, targets = batch
            opt_ewc.zero_grad()
            loss = criterion(model_ewc(inputs), targets) + ewc_loss(model_ewc)
            loss.backward()
            opt_ewc.step()
    result_ewc = evaluate(model_ewc, data["tgt_test"], criterion)
    ewc_src_retained = evaluate(model_ewc, data["src_test"], criterion)
    print(f"    Target: {eval_metric_fmt(data['task'], result_ewc)}")
    print(f"    Source retention: {eval_metric_fmt(data['task'], ewc_src_retained)}")
    results["EWC"] = result_ewc

    # ─── Phase 4: Model Merging ───
    print("\n  ── Phase 4: Model Merging (weight-space combination) ──")

    # Create a second full fine-tuned model variant (different LR)
    model_full2 = copy.deepcopy(pretrained)
    opt_full2 = torch.optim.Adam(model_full2.parameters(), lr=args.lr * 0.3)
    for ep in range(args.epochs):
        train_epoch(model_full2, data["tgt_train"], criterion, opt_full2)

    # Task vectors from base (both models have matching state_dict keys)
    base_sd = pretrained.state_dict()
    tv_full = compute_task_vector(base_sd, model_full.state_dict())
    tv_full2 = compute_task_vector(base_sd, model_full2.state_dict())

    # Analyze
    stats_full = task_vector_stats(tv_full)
    stats_full2 = task_vector_stats(tv_full2)
    sim = task_vector_similarity(tv_full, tv_full2)
    print(f"    Task vector (Full FT, lr={args.lr*0.1:.4f}): L2={stats_full['l2_norm']:.4f}")
    print(f"    Task vector (Full FT, lr={args.lr*0.3:.4f}): L2={stats_full2['l2_norm']:.4f}")
    print(f"    Cosine similarity:     {sim:.4f}")

    # Merge strategies
    merge_strategies = {}

    # Linear merge
    merged_sd = linear_merge([model_full.state_dict(), model_full2.state_dict()])
    merged = build_model(data)
    merged.load_state_dict(merged_sd)
    merge_strategies["Linear"] = evaluate(merged, data["tgt_test"], criterion)

    # SLERP
    slerp_sd = slerp_merge(model_full.state_dict(), model_full2.state_dict(), t=0.5)
    merged.load_state_dict(slerp_sd)
    merge_strategies["SLERP"] = evaluate(merged, data["tgt_test"], criterion)

    # Task arithmetic
    ta_sd = task_arithmetic_merge(base_sd, [tv_full, tv_full2], scalings=[0.5, 0.5])
    merged.load_state_dict(ta_sd)
    merge_strategies["Task Arith"] = evaluate(merged, data["tgt_test"], criterion)

    # TIES
    ties_sd = ties_merge(base_sd, [tv_full, tv_full2], density=0.3, scaling=1.0)
    merged.load_state_dict(ties_sd)
    merge_strategies["TIES"] = evaluate(merged, data["tgt_test"], criterion)

    # DARE + TIES
    dare_sd = dare_merge(base_sd, [tv_full, tv_full2], drop_rate=0.8,
                          use_ties=True, ties_density=0.3, seed=args.seed)
    merged.load_state_dict(dare_sd)
    merge_strategies["DARE+TIES"] = evaluate(merged, data["tgt_test"], criterion)

    # LoRA Soups
    print("\n    LoRA Soups (merging LoRA adapter weights)...")
    lora_a = copy.deepcopy(pretrained)
    LoRAInjector.inject(lora_a, rank=args.lora_rank, alpha=args.lora_rank * 2)
    opt_la = torch.optim.Adam(LoRAInjector.get_lora_parameters(lora_a), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(lora_a, data["tgt_train"], criterion, opt_la)

    lora_b = copy.deepcopy(pretrained)
    LoRAInjector.inject(lora_b, rank=args.lora_rank, alpha=args.lora_rank * 2)
    opt_lb = torch.optim.Adam(LoRAInjector.get_lora_parameters(lora_b), lr=args.lr * 0.5)
    for ep in range(args.epochs):
        train_epoch(lora_b, data["tgt_train"], criterion, opt_lb)

    merged_lora_sd = merge_lora_adapters([
        LoRAInjector.lora_state_dict(lora_a),
        LoRAInjector.lora_state_dict(lora_b),
    ])
    soup_model = copy.deepcopy(pretrained)
    LoRAInjector.inject(soup_model, rank=args.lora_rank, alpha=args.lora_rank * 2)
    current_sd = soup_model.state_dict()
    for key, val in merged_lora_sd.items():
        if key in current_sd:
            current_sd[key] = val
    soup_model.load_state_dict(current_sd)
    LoRAInjector.merge_all(soup_model)
    merge_strategies["LoRA Soup"] = evaluate(soup_model, data["tgt_test"], criterion)

    print("\n  Merge Results:")
    for name, result in merge_strategies.items():
        print(f"    {name:<15} {eval_metric_fmt(data['task'], result)}")
    results.update(merge_strategies)

    # ─── Phase 5: LoRA-Flow (learned adapter gating) ───
    print("\n  ── Phase 5: LoRA-Flow (learned gating for adapter combination) ──")
    # Create two LoRA-adapted models
    model_a = copy.deepcopy(pretrained)
    LoRAInjector.inject(model_a, rank=args.lora_rank, alpha=args.lora_rank * 2)
    opt_a = torch.optim.Adam(LoRAInjector.get_lora_parameters(model_a), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(model_a, data["tgt_train"], criterion, opt_a)

    model_b = copy.deepcopy(pretrained)
    LoRAInjector.inject(model_b, rank=args.lora_rank, alpha=args.lora_rank * 2)
    opt_b = torch.optim.Adam(LoRAInjector.get_lora_parameters(model_b), lr=args.lr * 0.5)
    for ep in range(args.epochs):
        train_epoch(model_b, data["tgt_train"], criterion, opt_b)

    # LoRA-Flow: learn gating weights
    flow = LoRAFlow(num_adapters=2, gate_input_dim=data["n_features"])

    def adapter_outputs_fn(batch):
        x, _ = batch
        with torch.no_grad():
            out_a = model_a(x)
            out_b = model_b(x)
        return [out_a, out_b]

    def gate_input_fn(batch):
        x, _ = batch
        return x

    def target_fn(batch):
        _, y = batch
        return y

    flow_history = train_lora_flow(
        flow, adapter_outputs_fn, gate_input_fn,
        data["tgt_train"], criterion, target_fn,
        epochs=min(args.epochs, 10), lr=0.01
    )
    print(f"    LoRA-Flow training loss: {flow_history['loss'][0]:.4f} → {flow_history['loss'][-1]:.4f}")
    print(f"    Final gate weights: {[f'{w:.3f}' for w in flow_history['gate_weights'][-1]]}")

    # Evaluate LoRA-Flow on test set
    flow.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data["tgt_test"]:
            x, y = batch
            adapter_outs = adapter_outputs_fn(batch)
            combined = flow.merge_with_gates(adapter_outs, x)
            preds = combined.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)
    flow_acc = correct / total
    print(f"    LoRA-Flow test accuracy: {flow_acc:.1%}")
    results["LoRA-Flow"] = {"accuracy": flow_acc, "loss": 0.0}

    # ─── Phase 6: Carbon Emissions Comparison ───
    print("\n  ── Phase 6: Carbon Emissions Comparison ──")
    summary = compare_emissions([carbon_full, carbon_lora])
    comp = summary["comparisons"][0]
    print(f"    Full FT:  {carbon_full['time_s']:.4f}s  "
          f"CO2={carbon_full['co2_kg']:.2e} kg")
    print(f"    LoRA FT:  {carbon_lora['time_s']:.4f}s  "
          f"CO2={carbon_lora['co2_kg']:.2e} kg")
    print(f"    CO2 saved by LoRA: {comp['co2_saved_pct']:.1f}%")

    # ─── Summary Table ───
    print("\n  " + "=" * 72)
    print("  SUMMARY: Breast Cancer Classification Transfer Results")
    print("  " + "=" * 72)
    print(f"  {'Strategy':<20} {'Accuracy':>12} {'Notes':>30}")
    print("  " + "-" * 64)
    for name, res in results.items():
        acc = res.get("accuracy", 0)
        notes = ""
        if name == "No transfer":
            notes = "pretrained on source only"
        elif name == "Full FT":
            notes = f"{train_p:,} trainable params"
        elif name == "LoRA":
            notes = f"{lora_trainable:,} trainable params"
        elif name == "EWC":
            notes = f"src retained: {ewc_src_retained['accuracy']:.1%}"
        elif name == "LoRA-Flow":
            notes = "learned adapter gating"
        print(f"  {name:<20} {acc:>11.1%} {notes:>30}")

    print("\n  Key findings:")
    # Find best strategy
    best_name = max(results.items(), key=lambda x: x[1].get("accuracy", 0))[0]
    print(f"    • Best strategy: {best_name}")
    if "LoRA" in results and "Full FT" in results:
        lora_acc = results["LoRA"]["accuracy"]
        full_acc = results["Full FT"]["accuracy"]
        print(f"    • LoRA achieves {lora_acc:.1%} with {lora_trainable:,} params "
              f"vs {full_acc:.1%} with {train_p:,} params")
    if "EWC" in results:
        print(f"    • EWC retains {ewc_src_retained['accuracy']:.1%} on source domain")


# ═══════════════════════════════════════════════════════════════════
# Demo: California Housing — Full Regression Pipeline
# ═══════════════════════════════════════════════════════════════════

def demo_housing(args):
    """
    California Housing — Real-World Regression Transfer
    ─────────────────────────────────────────────────────
    Source: Northern CA (Bay Area, Sacramento)
    Target: Southern CA (LA, San Diego)

    Exercises: LoRA, progressive unfreezing, EWC, CKA, merging, carbon
    """
    print("\n" + "=" * 72)
    print("  REAL-WORLD DEMO: California Housing — Regression Transfer")
    print("=" * 72)

    set_seed(args.seed)
    data = load_california_housing_pytorch(seed=args.seed)
    criterion = get_criterion(data["task"])

    print(f"\n  Dataset: {data['name']}")
    print(f"  Task: Regression (predict median house value)")
    print(f"  Domain split: {data['domain_split']}")
    print(f"  Source: {data['src_train_size']} train / {data['src_test_size']} test")
    print(f"  Target: {data['tgt_train_size']} train / {data['tgt_test_size']} test")
    print(f"  Features: {data['n_features']}")

    # ─── Phase 1: Pretrain on source ───
    print("\n  ── Phase 1: Pretraining on source domain (Northern CA) ──")
    pretrained = build_model(data)
    opt = torch.optim.Adam(pretrained.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(pretrained, data["src_train"], criterion, opt)
    src_result = evaluate(pretrained, data["src_test"], criterion)
    print(f"    Source test: {eval_metric_fmt(data['task'], src_result)}")
    pre_transfer = evaluate(pretrained, data["tgt_test"], criterion)
    print(f"    Target test (no adaptation): {eval_metric_fmt(data['task'], pre_transfer)}")

    results = {"No transfer": pre_transfer}

    # ─── Phase 2: CKA Similarity ───
    print("\n  ── Phase 2: CKA Representation Similarity (Northern ↔ Southern CA) ──")
    layer_names = [n for n, m in pretrained.named_modules() if isinstance(m, nn.Linear)]
    print(f"  {'Layer':<25} {'CKA':>10}")
    print("  " + "-" * 37)
    for layer in layer_names:
        reps_src = extract_representations(pretrained, data["src_test"], layer)
        reps_tgt = extract_representations(pretrained, data["tgt_test"], layer)
        cka = compute_cka(reps_src, reps_tgt)
        print(f"  {layer:<25} {cka:>9.4f}")

    # ─── Phase 3: Transfer Strategies ───
    print("\n  ── Phase 3: Transfer Learning Strategies ──")

    # Strategy A: Full fine-tuning
    print("\n  [A] Full fine-tuning on target (Southern CA)...")
    tracker_full = GPUCarbonTracker("full_ft", power_watts=100.0)
    model_full = copy.deepcopy(pretrained)
    opt_full = torch.optim.Adam(model_full.parameters(), lr=args.lr * 0.1)
    tracker_full.start()
    for ep in range(args.epochs):
        train_epoch(model_full, data["tgt_train"], criterion, opt_full)
    carbon_full = tracker_full.stop()
    result_full = evaluate(model_full, data["tgt_test"], criterion)
    total_p = sum(p.numel() for p in model_full.parameters())
    train_p = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    print(f"    Target: {eval_metric_fmt(data['task'], result_full)}")
    results["Full FT"] = result_full

    # Strategy B: LoRA
    print(f"\n  [B] LoRA fine-tuning (rank={args.lora_rank})...")
    tracker_lora = GPUCarbonTracker("lora_ft", power_watts=80.0)
    model_lora = copy.deepcopy(pretrained)
    n_injected = LoRAInjector.inject(model_lora, rank=args.lora_rank,
                                      alpha=args.lora_rank * 2)
    lora_params = LoRAInjector.get_lora_parameters(model_lora)
    lora_trainable = LoRAInjector.count_lora_params(model_lora)
    print(f"    Injected into {n_injected} layers, {lora_trainable:,} trainable params")
    opt_lora = torch.optim.Adam(lora_params, lr=args.lr)
    tracker_lora.start()
    for ep in range(args.epochs):
        train_epoch(model_lora, data["tgt_train"], criterion, opt_lora)
    carbon_lora = tracker_lora.stop()
    result_lora = evaluate(model_lora, data["tgt_test"], criterion)
    print(f"    Target: {eval_metric_fmt(data['task'], result_lora)}")
    results["LoRA"] = result_lora

    # Strategy C: Progressive Unfreezing
    print("\n  [C] Progressive unfreezing...")
    model_prog = copy.deepcopy(pretrained)
    base_prog = BaseModel(model_prog)
    groups = base_prog.get_layer_groups()
    scheduler = TransferScheduler(groups, base_lr=args.lr, decay=2.6)
    for ep in range(args.epochs):
        scheduler.step(ep)
        opt_prog = scheduler.build_optimizer()
        train_epoch(base_prog, data["tgt_train"], criterion, opt_prog)
    result_prog = evaluate(base_prog, data["tgt_test"], criterion)
    print(f"    Target: {eval_metric_fmt(data['task'], result_prog)}")
    results["Progressive"] = result_prog

    # Strategy D: EWC
    print("\n  [D] EWC-regularized fine-tuning...")
    model_ewc = copy.deepcopy(pretrained)
    fisher = compute_fisher_diagonal(pretrained, data["src_train"], criterion)
    ewc_loss = EWCLoss(pretrained, fisher, lambda_=500.0)
    opt_ewc = torch.optim.Adam(model_ewc.parameters(), lr=args.lr * 0.1)
    for ep in range(args.epochs):
        model_ewc.train()
        for batch in data["tgt_train"]:
            inputs, targets = batch
            opt_ewc.zero_grad()
            loss = criterion(model_ewc(inputs), targets) + ewc_loss(model_ewc)
            loss.backward()
            opt_ewc.step()
    result_ewc = evaluate(model_ewc, data["tgt_test"], criterion)
    ewc_src_retained = evaluate(model_ewc, data["src_test"], criterion)
    print(f"    Target: {eval_metric_fmt(data['task'], result_ewc)}")
    print(f"    Source retention: {eval_metric_fmt(data['task'], ewc_src_retained)}")
    results["EWC"] = result_ewc

    # ─── Phase 4: Model Merging ───
    print("\n  ── Phase 4: Model Merging ──")

    # Create second full FT variant for merging (different LR)
    model_full2 = copy.deepcopy(pretrained)
    opt_full2 = torch.optim.Adam(model_full2.parameters(), lr=args.lr * 0.3)
    for ep in range(args.epochs):
        train_epoch(model_full2, data["tgt_train"], criterion, opt_full2)

    base_sd = pretrained.state_dict()
    tv_full = compute_task_vector(base_sd, model_full.state_dict())
    tv_full2 = compute_task_vector(base_sd, model_full2.state_dict())

    sim = task_vector_similarity(tv_full, tv_full2)
    print(f"    Task vector cosine similarity: {sim:.4f}")

    merge_strategies = {}

    # Linear
    m_sd = linear_merge([model_full.state_dict(), model_full2.state_dict()])
    merged = build_model(data)
    merged.load_state_dict(m_sd)
    merge_strategies["Linear"] = evaluate(merged, data["tgt_test"], criterion)

    # SLERP
    s_sd = slerp_merge(model_full.state_dict(), model_full2.state_dict(), t=0.5)
    merged.load_state_dict(s_sd)
    merge_strategies["SLERP"] = evaluate(merged, data["tgt_test"], criterion)

    # Task Arithmetic
    ta_sd = task_arithmetic_merge(base_sd, [tv_full, tv_full2], scalings=[0.5, 0.5])
    merged.load_state_dict(ta_sd)
    merge_strategies["Task Arith"] = evaluate(merged, data["tgt_test"], criterion)

    # TIES
    ties_sd = ties_merge(base_sd, [tv_full, tv_full2], density=0.3, scaling=1.0)
    merged.load_state_dict(ties_sd)
    merge_strategies["TIES"] = evaluate(merged, data["tgt_test"], criterion)

    # DARE + TIES
    dare_sd = dare_merge(base_sd, [tv_full, tv_full2], drop_rate=0.8,
                          use_ties=True, ties_density=0.3, seed=args.seed)
    merged.load_state_dict(dare_sd)
    merge_strategies["DARE+TIES"] = evaluate(merged, data["tgt_test"], criterion)

    print("\n  Merge Results:")
    for name, result in merge_strategies.items():
        print(f"    {name:<15} {eval_metric_fmt(data['task'], result)}")
    results.update(merge_strategies)

    # ─── Phase 5: Carbon Comparison ───
    print("\n  ── Phase 5: Carbon Emissions ──")
    summary = compare_emissions([carbon_full, carbon_lora])
    comp = summary["comparisons"][0]
    print(f"    Full FT:  {carbon_full['time_s']:.4f}s  "
          f"CO2={carbon_full['co2_kg']:.2e} kg")
    print(f"    LoRA FT:  {carbon_lora['time_s']:.4f}s  "
          f"CO2={carbon_lora['co2_kg']:.2e} kg")
    print(f"    CO2 saved by LoRA: {comp['co2_saved_pct']:.1f}%")

    # ─── Summary ───
    print("\n  " + "=" * 72)
    print("  SUMMARY: California Housing Regression Transfer Results")
    print("  " + "=" * 72)
    print(f"  {'Strategy':<20} {'MSE':>12} {'Notes':>30}")
    print("  " + "-" * 64)
    for name, res in results.items():
        mse_val = res.get("loss", 0)
        notes = ""
        if name == "No transfer":
            notes = "pretrained on source only"
        elif name == "Full FT":
            notes = f"{train_p:,} trainable params"
        elif name == "LoRA":
            notes = f"{lora_trainable:,} trainable params"
        elif name == "EWC":
            notes = f"src MSE: {ewc_src_retained['loss']:.4f}"
        print(f"  {name:<20} {mse_val:>11.4f} {notes:>30}")

    print("\n  Key findings:")
    best_name = min(results.items(), key=lambda x: x[1].get("loss", float("inf")))[0]
    print(f"    • Best strategy (lowest MSE): {best_name}")
    if "LoRA" in results and "Full FT" in results:
        print(f"    • LoRA: {lora_trainable:,} params vs Full FT: {train_p:,} params")
        print(f"    • CO2 saved by LoRA: {comp['co2_saved_pct']:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Zeno v0.5.0 — Real-World Transfer Learning Demo")
    parser.add_argument("--demo", type=str, default="all",
                        choices=["breast_cancer", "housing", "all"],
                        help="Which dataset demo to run (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lora_rank", type=int, default=8)
    args = parser.parse_args()

    print("=" * 72)
    print("  Zeno v0.5.0 — Real-World Transfer Learning Demo")
    print("  From-scratch PyTorch on sklearn datasets (no downloads needed)")
    print("=" * 72)
    print(f"  seed={args.seed}  epochs={args.epochs}  lr={args.lr}  "
          f"lora_rank={args.lora_rank}")

    demos = {
        "breast_cancer": demo_breast_cancer,
        "housing": demo_housing,
    }

    if args.demo == "all":
        for name, fn in demos.items():
            fn(args)
    else:
        demos[args.demo](args)

    print("\n" + "=" * 72)
    print("  All real-world demos complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
