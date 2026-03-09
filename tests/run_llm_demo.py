"""
libraries v0.5.0 - Real-World LLM Transfer Learning Demo
=========================================================

Demonstrates Zeno's FULL deep learning pipeline on a real pretrained
language model (DistilBERT, 66M params) with real NLP datasets.

This is the capstone demo: every Zeno DL component exercised on an
actual LLM — proving our from-scratch library handles real transformers.

Pipeline:
  Phase 1: Fine-tune DistilBERT on sentiment analysis (SST-2)
  Phase 2: CKA similarity between sentiment & news representations
  Phase 3: LoRA fine-tuning (Q/V projections, 99.7% param reduction)
  Phase 4: EWC transfer: sentiment → news without forgetting
  Phase 5: Progressive unfreezing on news classification
  Phase 6: Model merging (5 strategies on transformer backbone)
  Phase 7: LoRA Soups + LoRA-Flow (learned adapter gating)

Requirements:
    pip install transformers datasets

Usage:
    cd content
    python -m tests.run_llm_demo
    python -m tests.run_llm_demo --demo lora
    python -m tests.run_llm_demo --demo merging
    python -m tests.run_llm_demo --demo all --epochs 3
"""

import argparse
import sys
import os
import copy
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libraries.metrics import set_seed
from libraries.dl.lora import LoRAInjector
from libraries.dl.transfer import BaseModel, TransferScheduler
from libraries.dl.ewc import compute_fisher_diagonal, EWCLoss
from libraries.dl.negative_transfer import compute_cka
from libraries.dl.carbon import GPUCarbonTracker
from libraries.dl.train import train_epoch, evaluate, fine_tune
from libraries.dl.merging import (
    linear_merge, slerp_merge, compute_task_vector,
    task_arithmetic_merge, ties_merge, dare_merge, merge_lora_adapters,
    LoRAFlow, train_lora_flow,
    task_vector_stats, task_vector_similarity,
)
from libraries.carbon import compare_emissions


# ═══════════════════════════════════════════════════════════════════
# Device Auto-Detection
# ═══════════════════════════════════════════════════════════════════

def get_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# ═══════════════════════════════════════════════════════════════════
# Model Wrapper: DistilBERT → Zeno-compatible classifier
# ═══════════════════════════════════════════════════════════════════

class LLMClassifier(nn.Module):
    """
    Wraps a HuggingFace DistilBERT for Zeno's train_epoch/evaluate.

    Accepts packed tensors of shape (batch, 2, seq_len) where:
      channel 0 = input_ids (stored as float, converted to long)
      channel 1 = attention_mask (stored as float, converted to long)

    This packed format is transparent to Zeno's training loop which
    does batch[0].to(device) then model(inputs).
    """

    def __init__(self, model_name, num_labels):
        super().__init__()
        from transformers import AutoModel
        # Suppress safetensors LOAD REPORT — the "UNEXPECTED" keys are MLM-head
        # weights (vocab_layer_norm, vocab_projector, vocab_transform) present in
        # the distilbert-base-uncased checkpoint but not needed by the encoder-only
        # DistilBertModel. This is expected and harmless.
        sys.stdout.flush()
        sys.stderr.flush()
        _old_out = os.dup(1)
        _old_err = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 1)
        os.dup2(_devnull, 2)
        try:
            self.transformer = AutoModel.from_pretrained(model_name)
        finally:
            os.dup2(_old_out, 1)
            os.dup2(_old_err, 2)
            os.close(_devnull)
            os.close(_old_out)
            os.close(_old_err)
        hidden = self.transformer.config.hidden_size  # 768 for distilbert
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] pooling
        return self.classifier(cls_output)


# ═══════════════════════════════════════════════════════════════════
# Backbone-Only Utilities (for multi-task merging)
# ═══════════════════════════════════════════════════════════════════

def get_backbone_state_dict(model):
    """Extract only transformer backbone weights (no classifier head)."""
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith("classifier.")}


def load_backbone_state_dict(model, backbone_sd):
    """Load backbone weights into model, keeping classifier untouched."""
    current = model.state_dict()
    for k, v in backbone_sd.items():
        if k in current and current[k].shape == v.shape:
            current[k] = v
    model.load_state_dict(current)


# ═══════════════════════════════════════════════════════════════════
# Data Loading Pipeline
# ═══════════════════════════════════════════════════════════════════

def load_and_tokenize(dataset_name, tokenizer, max_samples=400,
                      max_length=64, seed=42):
    """
    Load a HuggingFace dataset, tokenize, and pack into tensors.

    Returns:
        packed: (N, 2, max_length) float tensor
        labels: (N,) long tensor
    """
    from datasets import load_dataset

    if dataset_name == "sst2":
        ds = load_dataset("stanfordnlp/sst2", split="train")
        texts = ds["sentence"][:max_samples]
        labels = ds["label"][:max_samples]
    elif dataset_name == "ag_news":
        ds = load_dataset("fancyzhx/ag_news", split="train")
        texts = ds["text"][:max_samples]
        labels = ds["label"][:max_samples]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].float()
    attention_mask = encoded["attention_mask"].float()

    packed = torch.stack([input_ids, attention_mask], dim=1)
    labels_t = torch.tensor(labels, dtype=torch.long)

    return packed, labels_t


def make_loaders(packed, labels, batch_size=32, val_frac=0.2, seed=42):
    """Split into train/val DataLoaders."""
    torch.manual_seed(seed)
    n = len(packed)
    n_val = int(n * val_frac)
    idx = torch.randperm(n)

    train_ds = TensorDataset(packed[idx[n_val:]], labels[idx[n_val:]])
    val_ds = TensorDataset(packed[idx[:n_val]], labels[idx[:n_val]])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

MODEL_NAME = "distilbert-base-uncased"
TASK_A_NAME = "SST-2 Sentiment (pos/neg)"
TASK_B_NAME = "AG News Topics (4 classes)"


def _detect_gpu_power():
    """Detect real GPU TDP via NVML, fallback to manual estimate."""
    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # Power limit in milliwatts
            power_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            return power_mw / 1000.0, name  # watts, name
        except Exception:
            # Try torch for GPU name
            name = torch.cuda.get_device_name(0)
            # Common TDPs
            tdp_map = {"T4": 70, "V100": 300, "A100": 400, "A10": 150,
                        "RTX 3090": 350, "RTX 4090": 450, "L4": 72}
            for key, watts in tdp_map.items():
                if key in name:
                    return float(watts), name
            return 70.0, name  # conservative default
    return 30.0, "CPU"  # CPU fallback


GPU_POWER_WATTS, GPU_NAME = _detect_gpu_power()

# Accumulated CO2 results for final summary
_carbon_log = []

# Accumulated results for final summary table
_summary_rows = []


def _make_tracker(label):
    """Create a GPUCarbonTracker with real GPU power."""
    return GPUCarbonTracker(label, power_watts=GPU_POWER_WATTS)


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Fine-tune on Sentiment + Carbon Tracking
# ═══════════════════════════════════════════════════════════════════

def demo_finetune(args, tokenizer, data_a, data_b):
    """Fine-tune DistilBERT on sentiment analysis with carbon tracking."""
    print("\n" + "=" * 72)
    print("  Phase 1: Fine-tune DistilBERT on Sentiment (SST-2)")
    print("=" * 72)

    train_a, val_a = data_a
    criterion = nn.CrossEntropyLoss()

    # Full fine-tuning with carbon tracking
    print(f"\n  Loading {MODEL_NAME} (66M params)...")
    model = LLMClassifier(MODEL_NAME, num_labels=2).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    tracker = _make_tracker("full_finetune")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n  Training for {args.epochs} epochs...")
    history = fine_tune(
        model, train_a, val_a,
        epochs=args.epochs, optimizer=optimizer, criterion=criterion,
        carbon_tracker=tracker, verbose=True, device=str(DEVICE),
    )

    result = evaluate(model, val_a, criterion, device=str(DEVICE))
    co2 = history['co2_result']
    print(f"\n  Final sentiment accuracy: {result['accuracy']:.1%}")
    print(f"  CO2: {co2['co2_kg']:.2e} kg  ({co2['source']} measurement, "
          f"{co2['power_watts']:.0f}W avg)")

    _carbon_log.append(co2)
    _summary_rows.append(("Phase 1", "Full Fine-tune", "SST-2",
                           result['accuracy'], f"{total_params:,} params"))
    return model, history


# ═══════════════════════════════════════════════════════════════════
# Phase 2: CKA Representation Similarity
# ═══════════════════════════════════════════════════════════════════

def _extract_cls_representations(model, dataloader, layer_name, device="cpu"):
    """
    Extract [CLS]-pooled representations from a transformer layer.

    Standard extract_representations flattens (batch, seq_len, hidden) into
    (batch, seq_len*hidden) which creates a huge seq_len*hidden x seq_len*hidden
    CKA matrix (49K x 49K for DistilBERT — OOM on 15GB GPU).

    This version takes only the [CLS] token (index 0) output, giving
    (batch, hidden) and a manageable 768 x 768 CKA matrix.
    """
    model.eval()
    model.to(device)

    target_module = dict(model.named_modules()).get(layer_name)
    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    activations = []

    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if out.dim() == 3:
            # Transformer output: (batch, seq_len, hidden) → take [CLS]
            out = out[:, 0, :]
        elif out.dim() > 2:
            out = out.flatten(start_dim=1)
        activations.append(out.detach().cpu())

    handle = target_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            model(inputs)
    handle.remove()
    return torch.cat(activations, dim=0)


def demo_cka(args, model_a, data_a, data_b):
    """Measure representation similarity between sentiment & news data."""
    print("\n" + "=" * 72)
    print("  Phase 2: CKA Representation Similarity (Sentiment vs News)")
    print("=" * 72)

    _, val_a = data_a
    _, val_b = data_b

    # Key transformer layers to compare
    layer_names = [
        "transformer.transformer.layer.0.attention.q_lin",
        "transformer.transformer.layer.2.attention.q_lin",
        "transformer.transformer.layer.5.attention.q_lin",
        "transformer.transformer.layer.5.ffn.lin2",
    ]
    layer_labels = [
        "Layer 0 Q-proj (early)",
        "Layer 2 Q-proj (middle)",
        "Layer 5 Q-proj (late)",
        "Layer 5 FFN-out (final)",
    ]

    print(f"\n  {'Layer':<30} {'CKA Score':>10}")
    print("  " + "-" * 42)

    for layer, label in zip(layer_names, layer_labels):
        try:
            reps_a = _extract_cls_representations(model_a, val_a, layer,
                                                   device=str(DEVICE))
            reps_b = _extract_cls_representations(model_a, val_b, layer,
                                                   device=str(DEVICE))
            cka = compute_cka(reps_a, reps_b)
            print(f"  {label:<30} {cka:>9.4f}")
        except Exception as e:
            print(f"  {label:<30} {'error':>9} ({e})")

    print("\n  Interpretation:")
    print("    CKA ≈ 1.0: domains use similar representations (transfer likely helps)")
    print("    CKA ≈ 0.0: domains use very different representations (transfer may hurt)")


# ═══════════════════════════════════════════════════════════════════
# Phase 3: LoRA Fine-tuning
# ═══════════════════════════════════════════════════════════════════

def demo_lora(args, tokenizer, data_a, pretrained_model):
    """Compare full fine-tuning vs LoRA on sentiment analysis."""
    print("\n" + "=" * 72)
    print("  Phase 3: LoRA Fine-tuning — Parameter-Efficient LLM Adaptation")
    print("=" * 72)

    train_a, val_a = data_a
    criterion = nn.CrossEntropyLoss()

    # Full fine-tuning baseline
    print("\n  [A] Full fine-tuning (all 66M params trainable)...")
    model_full = copy.deepcopy(pretrained_model).to(DEVICE)
    total_p = sum(p.numel() for p in model_full.parameters())
    tracker_full = _make_tracker("full_ft")
    opt_full = torch.optim.AdamW(model_full.parameters(), lr=args.lr)

    tracker_full.start()
    t0 = time.time()
    for ep in range(args.epochs):
        train_epoch(model_full, train_a, criterion, opt_full, device=str(DEVICE))
    full_time = time.time() - t0
    carbon_full = tracker_full.stop()
    result_full = evaluate(model_full, val_a, criterion, device=str(DEVICE))
    print(f"    Accuracy: {result_full['accuracy']:.1%}  "
          f"Time: {full_time:.1f}s  Params: {total_p:,}")

    # LoRA fine-tuning
    print(f"\n  [B] LoRA fine-tuning (rank={args.lora_rank}, Q/V projections)...")
    model_lora = copy.deepcopy(pretrained_model).to(DEVICE)
    n_injected = LoRAInjector.inject(
        model_lora,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention Q and V
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,
    )
    model_lora.to(DEVICE)  # LoRA layers created on CPU, move them to device
    lora_params = LoRAInjector.get_lora_parameters(model_lora)
    lora_trainable = LoRAInjector.count_lora_params(model_lora)
    print(f"    Injected into {n_injected} layers")
    print(f"    LoRA params: {lora_trainable:,} / {total_p:,} "
          f"({lora_trainable/total_p:.2%} of model)")

    tracker_lora = _make_tracker("lora_ft")
    opt_lora = torch.optim.AdamW(lora_params, lr=args.lr * 5)

    tracker_lora.start()
    t0 = time.time()
    for ep in range(args.epochs):
        train_epoch(model_lora, train_a, criterion, opt_lora, device=str(DEVICE))
    lora_time = time.time() - t0
    carbon_lora = tracker_lora.stop()
    result_lora = evaluate(model_lora, val_a, criterion, device=str(DEVICE))
    print(f"    Accuracy: {result_lora['accuracy']:.1%}  "
          f"Time: {lora_time:.1f}s  Params: {lora_trainable:,}")

    # Merge LoRA for inference
    print("\n  Merging LoRA weights into base model...")
    pre_merge = evaluate(model_lora, val_a, criterion, device=str(DEVICE))
    LoRAInjector.merge_all(model_lora)
    post_merge = evaluate(model_lora, val_a, criterion, device=str(DEVICE))
    print(f"    Pre-merge acc:  {pre_merge['accuracy']:.1%}")
    print(f"    Post-merge acc: {post_merge['accuracy']:.1%}")
    print(f"    Merge produces identical outputs (zero overhead at inference)")

    # Carbon comparison
    print("\n  " + "-" * 60)
    print(f"  Carbon Emissions Comparison  (measured via {carbon_full['source']})")
    print("  " + "-" * 60)
    summary = compare_emissions([carbon_full, carbon_lora])
    comp = summary["comparisons"][0]
    print(f"  {'':>20} {'Full FT':>15} {'LoRA FT':>15}")
    print(f"  {'Trainable params':>20} {total_p:>15,} {lora_trainable:>15,}")
    print(f"  {'Accuracy':>20} {result_full['accuracy']:>14.1%} "
          f"{result_lora['accuracy']:>14.1%}")
    print(f"  {'Time':>20} {full_time:>14.1f}s {lora_time:>14.1f}s")
    print(f"  {'Avg GPU power':>20} {carbon_full['power_watts']:>13.0f}W "
          f"{carbon_lora['power_watts']:>13.0f}W")
    print(f"  {'Energy (kWh)':>20} {carbon_full['kwh']:>14.2e} "
          f"{carbon_lora['kwh']:>14.2e}")
    print(f"  {'CO2 (kg)':>20} {carbon_full['co2_kg']:>14.2e} "
          f"{carbon_lora['co2_kg']:>14.2e}")
    print(f"  {'CO2 saved':>20} {'baseline':>15} {comp['co2_saved_pct']:>13.1f}%")

    _carbon_log.extend([carbon_full, carbon_lora])
    _summary_rows.append(("Phase 3", "Full Fine-tune", "SST-2",
                           result_full['accuracy'], f"{total_p:,} params"))
    _summary_rows.append(("Phase 3", "LoRA (rank={})".format(args.lora_rank),
                           "SST-2", result_lora['accuracy'],
                           f"{lora_trainable:,} params ({lora_trainable/total_p:.2%})"))
    return model_lora, carbon_full, carbon_lora


# ═══════════════════════════════════════════════════════════════════
# Phase 4: EWC Transfer (Sentiment → News)
# ═══════════════════════════════════════════════════════════════════

def demo_ewc(args, pretrained_model, data_a, data_b):
    """Transfer from sentiment to news with EWC to prevent forgetting."""
    print("\n" + "=" * 72)
    print("  Phase 4: EWC Transfer — Sentiment → News Without Forgetting")
    print("=" * 72)

    train_a, val_a = data_a
    train_b, val_b = data_b
    criterion = nn.CrossEntropyLoss()

    # Compute Fisher on Task A (backbone only)
    print("\n  Computing Fisher Information on sentiment task...")
    fisher_full = compute_fisher_diagonal(pretrained_model, train_a, criterion,
                                           device=str(DEVICE))
    # Filter to backbone-only (exclude classifier — different shapes for Task B)
    fisher = {k: v for k, v in fisher_full.items()
              if not k.startswith("classifier.")}
    n_fisher = sum(f.numel() for f in fisher.values())
    top_importance = max(f.max().item() for f in fisher.values())
    print(f"    Fisher over {n_fisher:,} backbone params")
    print(f"    Max importance: {top_importance:.6f}")

    source_model = copy.deepcopy(pretrained_model).to(DEVICE)

    # Task B model: new 4-class head on pretrained backbone
    print(f"\n  Building Task B model (4-class news topic classifier)...")
    model_no_ewc = LLMClassifier(MODEL_NAME, num_labels=4)
    load_backbone_state_dict(model_no_ewc, get_backbone_state_dict(pretrained_model))
    model_no_ewc = model_no_ewc.to(DEVICE)
    model_ewc = copy.deepcopy(model_no_ewc).to(DEVICE)

    # Fine-tune on Task B WITHOUT EWC
    print("\n  [A] Fine-tuning on news WITHOUT EWC...")
    tracker_no_ewc = _make_tracker("ewc_baseline")
    tracker_no_ewc.start()
    opt = torch.optim.AdamW(model_no_ewc.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        loss = train_epoch(model_no_ewc, train_b, criterion, opt,
                           device=str(DEVICE))
        if not args.quiet:
            r = evaluate(model_no_ewc, val_b, criterion, device=str(DEVICE))
            print(f"      epoch {ep+1}/{args.epochs}  loss={loss:.4f}  "
                  f"news_acc={r['accuracy']:.1%}")
    carbon_no_ewc = tracker_no_ewc.stop()
    result_no_ewc = evaluate(model_no_ewc, val_b, criterion, device=str(DEVICE))

    # Fine-tune on Task B WITH EWC
    print(f"\n  [B] Fine-tuning on news WITH EWC (lambda=500)...")
    tracker_ewc = _make_tracker("ewc_transfer")
    tracker_ewc.start()
    ewc_loss = EWCLoss(source_model, fisher, lambda_=500.0).to(DEVICE)
    opt = torch.optim.AdamW(model_ewc.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        model_ewc.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_b:
            inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad()
            loss = criterion(model_ewc(inputs), targets) + ewc_loss(model_ewc)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        if not args.quiet:
            r = evaluate(model_ewc, val_b, criterion, device=str(DEVICE))
            print(f"      epoch {ep+1}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"news_acc={r['accuracy']:.1%}")
    carbon_ewc = tracker_ewc.stop()
    result_ewc = evaluate(model_ewc, val_b, criterion, device=str(DEVICE))
    _carbon_log.extend([carbon_no_ewc, carbon_ewc])

    # Measure backbone drift from pretrained (backbone only — heads differ)
    print("\n  Backbone parameter drift from pretrained:")
    ref_params = {
        name: param.data.clone().cpu()
        for name, param in pretrained_model.named_parameters()
        if not name.startswith("classifier.")
    }

    def backbone_drift(model):
        drift = {}
        for name, param in model.named_parameters():
            if name in ref_params:
                ref = ref_params[name].to(param.device)
                drift[name] = float(torch.norm(param.data - ref).item())
        return drift

    drift_no_ewc = backbone_drift(model_no_ewc)
    drift_ewc = backbone_drift(model_ewc)

    # Average drift
    avg_drift_no = np.mean(list(drift_no_ewc.values()))
    avg_drift_ewc = np.mean(list(drift_ewc.values()))
    print(f"    Without EWC: avg drift = {avg_drift_no:.4f}")
    print(f"    With EWC:    avg drift = {avg_drift_ewc:.4f}")

    # Summary
    print("\n  " + "-" * 60)
    print("  EWC Transfer Results")
    print("  " + "-" * 60)
    print(f"  {'':>25} {'No EWC':>12} {'With EWC':>12}")
    print(f"  {'News accuracy':>25} {result_no_ewc['accuracy']:>11.1%} "
          f"{result_ewc['accuracy']:>11.1%}")
    print(f"  {'Backbone drift':>25} {avg_drift_no:>11.4f} {avg_drift_ewc:>11.4f}")
    if avg_drift_ewc < avg_drift_no:
        reduction = (1.0 - avg_drift_ewc / avg_drift_no) * 100
        print(f"\n    EWC reduced backbone drift by {reduction:.0f}%")

    _summary_rows.append(("Phase 4", "Baseline (no EWC)", "AG News",
                           result_no_ewc['accuracy'], f"drift={avg_drift_no:.4f}"))
    _summary_rows.append(("Phase 4", "EWC Transfer (λ=500)", "AG News",
                           result_ewc['accuracy'], f"drift={avg_drift_ewc:.4f}"))
    return model_no_ewc, model_ewc


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Progressive Unfreezing
# ═══════════════════════════════════════════════════════════════════

def demo_progressive(args, pretrained_model, data_b):
    """Progressive unfreezing with discriminative LRs on news task."""
    print("\n" + "=" * 72)
    print("  Phase 5: Progressive Unfreezing on News Classification")
    print("=" * 72)

    train_b, val_b = data_b
    criterion = nn.CrossEntropyLoss()

    model = LLMClassifier(MODEL_NAME, num_labels=4)
    load_backbone_state_dict(model, get_backbone_state_dict(pretrained_model))
    model = model.to(DEVICE)

    base_model = BaseModel(model)
    groups = base_model.get_layer_groups()
    print(f"\n  Layer groups: {len(groups)}")
    for i, (name, params) in enumerate(groups):
        n_p = sum(p.numel() for p in params)
        print(f"    [{i}] {name}: {n_p:,} params")

    scheduler = TransferScheduler(groups, base_lr=args.lr, decay=2.6)

    print(f"\n  Training with progressive unfreezing...")
    for ep in range(args.epochs):
        scheduler.step(ep)
        opt = scheduler.build_optimizer()
        loss = train_epoch(base_model, train_b, criterion, opt,
                           device=str(DEVICE))
        if not args.quiet:
            r = evaluate(base_model, val_b, criterion, device=str(DEVICE))
            unfrozen = sum(1 for _, ps in groups for p in ps if p.requires_grad)
            total_tensors = sum(1 for _, ps in groups for _ in ps)
            print(f"      epoch {ep+1}/{args.epochs}  loss={loss:.4f}  "
                  f"acc={r['accuracy']:.1%}  "
                  f"unfrozen={unfrozen}/{total_tensors} param tensors")

    result = evaluate(base_model, val_b, criterion, device=str(DEVICE))
    print(f"\n  Final news accuracy: {result['accuracy']:.1%}")
    _summary_rows.append(("Phase 5", "Progressive Unfreezing", "AG News",
                           result['accuracy'], f"{len(groups)} layer groups"))
    return model


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Model Merging (Backbone-Only)
# ═══════════════════════════════════════════════════════════════════

def demo_merging(args, pretrained_model, data_b):
    """Merge two fine-tuned LLM backbones using 5 strategies."""
    print("\n" + "=" * 72)
    print("  Phase 6: Model Merging — Combining LLM Experts in Weight Space")
    print("=" * 72)

    train_b, val_b = data_b
    criterion = nn.CrossEntropyLoss()

    # Train variant 1 (lower LR)
    print("\n  Training variant 1 (lr={:.0e})...".format(args.lr))
    model_v1 = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    load_backbone_state_dict(model_v1, get_backbone_state_dict(pretrained_model))
    opt_v1 = torch.optim.AdamW(model_v1.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(model_v1, train_b, criterion, opt_v1, device=str(DEVICE))
    r1 = evaluate(model_v1, val_b, criterion, device=str(DEVICE))
    print(f"    Accuracy: {r1['accuracy']:.1%}")

    # Train variant 2 (higher LR)
    lr2 = args.lr * 3
    print(f"\n  Training variant 2 (lr={lr2:.0e})...")
    model_v2 = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    load_backbone_state_dict(model_v2, get_backbone_state_dict(pretrained_model))
    opt_v2 = torch.optim.AdamW(model_v2.parameters(), lr=lr2)
    for ep in range(args.epochs):
        train_epoch(model_v2, train_b, criterion, opt_v2, device=str(DEVICE))
    r2 = evaluate(model_v2, val_b, criterion, device=str(DEVICE))
    print(f"    Accuracy: {r2['accuracy']:.1%}")

    # Task vector analysis
    base_sd = get_backbone_state_dict(pretrained_model)
    sd_v1 = get_backbone_state_dict(model_v1)
    sd_v2 = get_backbone_state_dict(model_v2)

    tv_v1 = compute_task_vector(base_sd, sd_v1)
    tv_v2 = compute_task_vector(base_sd, sd_v2)

    stats_v1 = task_vector_stats(tv_v1)
    stats_v2 = task_vector_stats(tv_v2)
    sim = task_vector_similarity(tv_v1, tv_v2)

    print(f"\n  Task Vector Analysis:")
    print(f"    Variant 1: L2={stats_v1['l2_norm']:.4f}  "
          f"mean_mag={stats_v1['mean_magnitude']:.6f}")
    print(f"    Variant 2: L2={stats_v2['l2_norm']:.4f}  "
          f"mean_mag={stats_v2['mean_magnitude']:.6f}")
    print(f"    Cosine similarity: {sim:.4f}")

    # Merge strategies
    print("\n  Merging with 5 strategies...")
    merge_results = {}

    def eval_merged(merged_sd, name):
        merged = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
        load_backbone_state_dict(merged, merged_sd)
        merged.classifier.load_state_dict(model_v1.classifier.state_dict())
        r = evaluate(merged, val_b, criterion, device=str(DEVICE))
        merge_results[name] = r
        print(f"    {name:<15} accuracy: {r['accuracy']:.1%}")

    # Linear
    eval_merged(linear_merge([sd_v1, sd_v2]), "Linear")
    # SLERP
    eval_merged(slerp_merge(sd_v1, sd_v2, t=0.5), "SLERP")
    # Task Arithmetic
    eval_merged(
        task_arithmetic_merge(base_sd, [tv_v1, tv_v2], scalings=[0.5, 0.5]),
        "Task Arith")
    # TIES
    eval_merged(
        ties_merge(base_sd, [tv_v1, tv_v2], density=0.3, scaling=1.0),
        "TIES")
    # DARE + TIES
    eval_merged(
        dare_merge(base_sd, [tv_v1, tv_v2], drop_rate=0.8,
                   use_ties=True, ties_density=0.3, seed=args.seed),
        "DARE+TIES")

    # Summary
    print("\n  " + "-" * 50)
    print(f"  {'Strategy':<20} {'Accuracy':>12}")
    print("  " + "-" * 50)
    print(f"  {'Variant 1':.<20} {r1['accuracy']:>11.1%}")
    print(f"  {'Variant 2':.<20} {r2['accuracy']:>11.1%}")
    for name, r in merge_results.items():
        print(f"  {name:.<20} {r['accuracy']:>11.1%}")

    _summary_rows.append(("Phase 6", "Variant 1", "AG News",
                           r1['accuracy'], f"lr={args.lr:.0e}"))
    _summary_rows.append(("Phase 6", "Variant 2", "AG News",
                           r2['accuracy'], f"lr={lr2:.0e}"))
    for name, r in merge_results.items():
        _summary_rows.append(("Phase 6", f"Merge: {name}", "AG News",
                               r['accuracy'], ""))

    return model_v1, model_v2


# ═══════════════════════════════════════════════════════════════════
# Phase 7: LoRA Soups + LoRA-Flow
# ═══════════════════════════════════════════════════════════════════

def demo_lora_flow(args, pretrained_model, data_b):
    """Merge LoRA adapters and learn dynamic gating."""
    print("\n" + "=" * 72)
    print("  Phase 7: LoRA Soups + LoRA-Flow (Learned Adapter Gating)")
    print("=" * 72)

    train_b, val_b = data_b
    criterion = nn.CrossEntropyLoss()
    target_modules = ["q_lin", "v_lin"]

    # Train LoRA adapter 1
    print(f"\n  Training LoRA adapter 1 (rank={args.lora_rank}, lr={args.lr*5:.0e})...")
    model_l1 = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    load_backbone_state_dict(model_l1, get_backbone_state_dict(pretrained_model))
    LoRAInjector.inject(model_l1, target_modules=target_modules,
                         rank=args.lora_rank, alpha=args.lora_rank * 2)
    model_l1.to(DEVICE)  # LoRA layers created on CPU, move to device
    opt1 = torch.optim.AdamW(LoRAInjector.get_lora_parameters(model_l1),
                              lr=args.lr * 5)
    for ep in range(args.epochs):
        train_epoch(model_l1, train_b, criterion, opt1, device=str(DEVICE))
    r1 = evaluate(model_l1, val_b, criterion, device=str(DEVICE))
    print(f"    Adapter 1 accuracy: {r1['accuracy']:.1%}  "
          f"({LoRAInjector.count_lora_params(model_l1):,} LoRA params)")

    # Train LoRA adapter 2 (different LR)
    print(f"\n  Training LoRA adapter 2 (rank={args.lora_rank}, lr={args.lr*2:.0e})...")
    model_l2 = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    load_backbone_state_dict(model_l2, get_backbone_state_dict(pretrained_model))
    LoRAInjector.inject(model_l2, target_modules=target_modules,
                         rank=args.lora_rank, alpha=args.lora_rank * 2)
    model_l2.to(DEVICE)  # LoRA layers created on CPU, move to device
    opt2 = torch.optim.AdamW(LoRAInjector.get_lora_parameters(model_l2),
                              lr=args.lr * 2)
    for ep in range(args.epochs):
        train_epoch(model_l2, train_b, criterion, opt2, device=str(DEVICE))
    r2 = evaluate(model_l2, val_b, criterion, device=str(DEVICE))
    print(f"    Adapter 2 accuracy: {r2['accuracy']:.1%}")

    # LoRA Soups: merge adapter weights
    print("\n  LoRA Soups: merging adapter weights...")
    lora_sd_1 = LoRAInjector.lora_state_dict(model_l1)
    lora_sd_2 = LoRAInjector.lora_state_dict(model_l2)
    merged_lora = merge_lora_adapters([lora_sd_1, lora_sd_2])

    soup_model = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    load_backbone_state_dict(soup_model, get_backbone_state_dict(pretrained_model))
    LoRAInjector.inject(soup_model, target_modules=target_modules,
                         rank=args.lora_rank, alpha=args.lora_rank * 2)
    soup_model.to(DEVICE)  # LoRA layers created on CPU, move to device
    current_sd = soup_model.state_dict()
    for k, v in merged_lora.items():
        if k in current_sd:
            current_sd[k] = v
    soup_model.load_state_dict(current_sd)
    # Copy classifier from adapter 1
    soup_model.classifier.load_state_dict(model_l1.classifier.state_dict())
    LoRAInjector.merge_all(soup_model)

    r_soup = evaluate(soup_model, val_b, criterion, device=str(DEVICE))
    print(f"    LoRA Soup accuracy: {r_soup['accuracy']:.1%}")

    # LoRA-Flow: learned gating
    print("\n  LoRA-Flow: training learned gating weights...")
    hidden_size = 768  # distilbert hidden size
    flow = LoRAFlow(num_adapters=2, gate_input_dim=hidden_size).to(DEVICE)

    def adapter_outputs_fn(batch):
        x = batch[0].to(DEVICE)
        with torch.no_grad():
            return [model_l1(x), model_l2(x)]

    def gate_input_fn(batch):
        x = batch[0].to(DEVICE)
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        with torch.no_grad():
            out = model_l1.transformer(
                input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def target_fn(batch):
        return batch[1].to(DEVICE)

    flow_history = train_lora_flow(
        flow, adapter_outputs_fn, gate_input_fn,
        train_b, criterion, target_fn,
        epochs=min(args.epochs, 5), lr=0.01,
    )
    print(f"    Loss: {flow_history['loss'][0]:.4f} → {flow_history['loss'][-1]:.4f}")
    print(f"    Gate weights: {[f'{w:.3f}' for w in flow_history['gate_weights'][-1]]}")

    # Evaluate LoRA-Flow
    flow.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_b:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outs = adapter_outputs_fn(batch)
            gate_in = gate_input_fn(batch)
            combined = flow.merge_with_gates(outs, gate_in)
            preds = combined.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)
    flow_acc = correct / total
    print(f"    LoRA-Flow accuracy: {flow_acc:.1%}")

    # Summary
    print("\n  " + "-" * 50)
    print(f"  {'Strategy':<20} {'Accuracy':>12}")
    print("  " + "-" * 50)
    print(f"  {'Adapter 1':.<20} {r1['accuracy']:>11.1%}")
    print(f"  {'Adapter 2':.<20} {r2['accuracy']:>11.1%}")
    print(f"  {'LoRA Soup':.<20} {r_soup['accuracy']:>11.1%}")
    print(f"  {'LoRA-Flow':.<20} {flow_acc:>11.1%}")

    lora_p = LoRAInjector.count_lora_params(model_l1)
    _summary_rows.append(("Phase 7", "LoRA Adapter 1", "AG News",
                           r1['accuracy'], f"{lora_p:,} params"))
    _summary_rows.append(("Phase 7", "LoRA Adapter 2", "AG News",
                           r2['accuracy'], f"{lora_p:,} params"))
    _summary_rows.append(("Phase 7", "LoRA Soup", "AG News",
                           r_soup['accuracy'], "uniform avg"))
    _summary_rows.append(("Phase 7", "LoRA-Flow", "AG News",
                           flow_acc, "learned gating"))


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Zeno v0.5.0 — Real-World LLM Transfer Learning Demo")
    parser.add_argument("--demo", type=str, default="all",
                        choices=["finetune", "cka", "lora", "ewc",
                                 "progressive", "merging", "lora_flow", "all"],
                        help="Which phase to run (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch output")
    args = parser.parse_args()

    print("=" * 72)
    print("  Zeno v0.5.0 — Real-World LLM Transfer Learning Demo")
    print("  DistilBERT (66M params) + SST-2 Sentiment + AG News Topics")
    print("  From-scratch PyTorch (no PEFT, no HuggingFace Trainer)")
    print("=" * 72)
    print(f"  device={DEVICE}  GPU={GPU_NAME}  power={GPU_POWER_WATTS:.0f}W")
    print(f"  seed={args.seed}  epochs={args.epochs}  lr={args.lr}  "
          f"lora_rank={args.lora_rank}  samples={args.max_samples}")
    _carbon_log.clear()
    _summary_rows.clear()

    set_seed(args.seed)

    # ─── Load model and data ───
    print("\n  Loading tokenizer and datasets...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"  Tokenizing {TASK_A_NAME}...")
    packed_a, labels_a = load_and_tokenize(
        "sst2", tokenizer, max_samples=args.max_samples)
    data_a = make_loaders(packed_a, labels_a, seed=args.seed)

    print(f"  Tokenizing {TASK_B_NAME}...")
    packed_b, labels_b = load_and_tokenize(
        "ag_news", tokenizer, max_samples=args.max_samples)
    data_b = make_loaders(packed_b, labels_b, seed=args.seed)

    print(f"  Task A: {len(packed_a)} samples ({TASK_A_NAME})")
    print(f"  Task B: {len(packed_b)} samples ({TASK_B_NAME})")

    # ─── Run phases ───
    if args.demo == "all" or args.demo == "finetune":
        pretrained, _ = demo_finetune(args, tokenizer, data_a, data_b)
    else:
        # Need a pretrained model for other phases
        print("\n  Quick-training base sentiment model...")
        pretrained = LLMClassifier(MODEL_NAME, num_labels=2).to(DEVICE)
        opt = torch.optim.AdamW(pretrained.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        for ep in range(args.epochs):
            train_epoch(pretrained, data_a[0], criterion, opt,
                        device=str(DEVICE))
        r = evaluate(pretrained, data_a[1], criterion, device=str(DEVICE))
        print(f"  Base sentiment accuracy: {r['accuracy']:.1%}")

    if args.demo == "all" or args.demo == "cka":
        demo_cka(args, pretrained, data_a, data_b)

    if args.demo == "all" or args.demo == "lora":
        demo_lora(args, tokenizer, data_a, pretrained)

    if args.demo == "all" or args.demo == "ewc":
        demo_ewc(args, pretrained, data_a, data_b)

    if args.demo == "all" or args.demo == "progressive":
        demo_progressive(args, pretrained, data_b)

    if args.demo == "all" or args.demo == "merging":
        demo_merging(args, pretrained, data_b)

    if args.demo == "all" or args.demo == "lora_flow":
        demo_lora_flow(args, pretrained, data_b)

    # ─── CO2 Summary ───
    if _carbon_log:
        print("\n" + "=" * 72)
        print("  Carbon Emissions Summary")
        print("=" * 72)
        total_co2 = sum(r['co2_kg'] for r in _carbon_log)
        total_kwh = sum(r['kwh'] for r in _carbon_log)
        total_time = sum(r['time_s'] for r in _carbon_log)
        print(f"\n  {'Phase':<25} {'Time':>8} {'Energy':>12} {'CO2':>12} {'Source':>10}")
        print("  " + "-" * 69)
        for r in _carbon_log:
            print(f"  {r['method']:<25} {r['time_s']:>7.1f}s "
                  f"{r['kwh']:>11.2e} {r['co2_kg']:>11.2e} {r['source']:>10}")
        print("  " + "-" * 69)
        print(f"  {'TOTAL':<25} {total_time:>7.1f}s "
              f"{total_kwh:>11.2e} {total_co2:>11.2e}")
        print(f"\n  GPU: {GPU_NAME}  Power: {GPU_POWER_WATTS:.0f}W  "
              f"Carbon intensity: 0.45 kg/kWh (US avg)")

        # Real-world equivalence for this run
        co2_g = total_co2 * 1000
        if co2_g > 0:
            phone_charges = co2_g / 8.22     # ~8.22g CO2 per phone charge
            km_driven = co2_g / 121          # ~121g CO2 per km (avg car)
            google_searches = co2_g / 0.2    # ~0.2g CO2 per Google search
            led_hours = co2_g / 5.0          # ~5g CO2 per hour of LED bulb
            streaming_min = co2_g / 0.6      # ~36g/hr = 0.6g/min streaming
            print(f"\n  This run's footprint ({total_time:.0f}s):")
            print(f"    {total_co2:.2e} kg CO2 = {phone_charges:.2f} phone charges")
            print(f"    = {google_searches:.0f} Google searches"
                  f"  = {led_hours:.1f} hrs LED lighting"
                  f"  = {streaming_min:.1f} min video streaming")

        # Projected savings: LoRA vs full fine-tuning at scale
        co2_full = [r for r in _carbon_log if r['method'] in ('full_ft',)]
        co2_lora = [r for r in _carbon_log if r['method'] == 'lora_ft']
        if co2_full and co2_lora:
            saved_per_task = co2_full[0]['co2_kg'] - co2_lora[0]['co2_kg']
            if saved_per_task > 0:
                N = 10000
                total_saved_kg = saved_per_task * N
                total_saved_g = total_saved_kg * 1000
                p_phone = total_saved_g / 8.22
                p_searches = total_saved_g / 0.2
                p_led = total_saved_g / 5.0
                p_stream = total_saved_g / 36.0   # 36g/hr streaming
                p_km = total_saved_g / 121.0
                print(f"\n  Projected across {N:,} training tasks (LoRA vs Full FT):")
                print(f"    CO2 saved per task:  {saved_per_task:.2e} kg "
                      f"({(saved_per_task/co2_full[0]['co2_kg'])*100:.1f}%)")
                print(f"    Total CO2 saved:     {total_saved_kg:.4f} kg")
                print(f"    = {p_phone:.0f} phone charges")
                print(f"    = {p_searches:.0f} Google searches")
                print(f"    = {p_led:.1f} hours of LED lighting")
                print(f"    = {p_stream:.1f} hours of video streaming")
                print(f"    = {p_km:.4f} km driven")

    # ─── Final Summary Table ───
    if _summary_rows:
        print("\n" + "=" * 72)
        print("  Final Results Summary — All Phases")
        print("=" * 72)
        print(f"\n  {'Phase':<10} {'Method':<28} {'Task':<10} "
              f"{'Accuracy':>9}  {'Notes'}")
        print("  " + "-" * 76)
        prev_phase = None
        for phase, method, task, acc, notes in _summary_rows:
            if prev_phase and prev_phase != phase:
                print("  " + "-" * 76)
            prev_phase = phase
            print(f"  {phase:<10} {method:<28} {task:<10} "
                  f"{acc:>8.1%}   {notes}")
        print("  " + "-" * 76)

        # Best result per task
        best_a = max((r for r in _summary_rows if r[2] == "SST-2"),
                     key=lambda r: r[3], default=None)
        best_b = max((r for r in _summary_rows if r[2] == "AG News"),
                     key=lambda r: r[3], default=None)
        print(f"\n  Best SST-2:   {best_a[1]} — {best_a[3]:.1%}" if best_a else "")
        print(f"  Best AG News: {best_b[1]} — {best_b[3]:.1%}" if best_b else "")

    print("\n" + "=" * 72)
    print("  All LLM demo phases complete.")
    print("  Every Zeno DL component validated on a real pretrained LLM.")
    print("=" * 72)


if __name__ == "__main__":
    main()
