"""
libraries v0.5.0 - Real-World LLM Transfer Learning Demo
=========================================================

Demonstrates Zeno's FULL deep learning pipeline on a real pretrained
language model (DistilBERT, 66M params) with real NLP datasets.

This is the capstone demo: every Zeno DL component exercised on an
actual LLM — proving our from-scratch library handles real transformers.

Pipeline:
  Phase 1: Full fine-tune baselines on both tasks (SST-2 + AG News)
  Phase 2: CKA similarity + representation MMD (negative transfer detection)
  Phase 3: LoRA fine-tuning (Q/V projections, 99.9% param reduction)
  Phase 4: EWC transfer + NegativeTransferMonitor: sentiment → news
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
from libraries.dl.negative_transfer import (
    compute_cka, NegativeTransferMonitor, compute_representation_mmd,
)
from libraries.negative_transfer import compute_mmd
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
                      max_length=None, seed=42):
    """
    Load a HuggingFace dataset, tokenize, and pack into tensors.

    Shuffles the dataset before slicing to avoid class-ordering bias
    (e.g. AG News is ordered by class — taking [:N] without shuffle
    would yield only class 0 samples).

    Returns:
        packed: (N, 2, max_length) float tensor
        labels: (N,) long tensor
    """
    from datasets import load_dataset

    if dataset_name == "sst2":
        ds = load_dataset("stanfordnlp/sst2", split="train")
        ds = ds.shuffle(seed=seed)
        if max_length is None:
            max_length = 64  # SST-2 sentences are short
        texts = ds["sentence"][:max_samples]
        labels = ds["label"][:max_samples]
    elif dataset_name == "ag_news":
        ds = load_dataset("fancyzhx/ag_news", split="train")
        ds = ds.shuffle(seed=seed)
        if max_length is None:
            max_length = 128  # AG News articles need more tokens
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
    """Split into train/val DataLoaders with stratified sampling."""
    torch.manual_seed(seed)
    n = len(packed)
    n_val = int(n * val_frac)

    # Stratified split: ensure each class is proportionally represented
    unique_labels = labels.unique()
    train_indices = []
    val_indices = []
    for lbl in unique_labels:
        lbl_idx = (labels == lbl).nonzero(as_tuple=True)[0]
        perm = lbl_idx[torch.randperm(len(lbl_idx))]
        n_val_lbl = max(1, int(len(perm) * val_frac))
        val_indices.append(perm[:n_val_lbl])
        train_indices.append(perm[n_val_lbl:])
    train_idx = torch.cat(train_indices)
    val_idx = torch.cat(val_indices)
    # Shuffle within splits
    train_idx = train_idx[torch.randperm(len(train_idx))]
    val_idx = val_idx[torch.randperm(len(val_idx))]

    train_ds = TensorDataset(packed[train_idx], labels[train_idx])
    val_ds = TensorDataset(packed[val_idx], labels[val_idx])

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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 30.0, "Apple MPS"  # MPS fallback
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
    """Fine-tune DistilBERT on both tasks to establish baselines."""
    print("\n" + "=" * 72)
    print("  Chapter 1: The Baselines")
    print("  \"How well can DistilBERT learn each task from scratch?\"")
    print("=" * 72)
    print("\n  We start with a pretrained DistilBERT — it understands English")
    print("  grammar and word meaning, but knows nothing about our tasks.")
    print("  Let's see how well it can learn each one with full fine-tuning.")

    criterion = nn.CrossEntropyLoss()
    train_a, val_a = data_a
    train_b, val_b = data_b

    # ── Task A: Sentiment (SST-2) ──
    print(f"\n  [A] Fine-tuning on SST-2 Sentiment (2 classes)...")
    model = LLMClassifier(MODEL_NAME, num_labels=2).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      {MODEL_NAME}: {total_params:,} params")

    tracker_a = _make_tracker("Ch1: Sentiment FT")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = fine_tune(
        model, train_a, val_a,
        epochs=args.epochs, optimizer=optimizer, criterion=criterion,
        carbon_tracker=tracker_a, verbose=True, device=str(DEVICE),
    )

    result_a = evaluate(model, val_a, criterion, device=str(DEVICE))
    co2_a = history['co2_result']
    print(f"      Sentiment accuracy: {result_a['accuracy']:.1%}")

    # ── Task B: AG News (4 classes) — the baseline for Phases 4–7 ──
    print(f"\n  [B] Fine-tuning on AG News Topics (4 classes)...")
    model_b = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    print(f"      {MODEL_NAME}: {total_params:,} params")

    tracker_b = _make_tracker("Ch1: News FT")
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=args.lr)

    history_b = fine_tune(
        model_b, train_b, val_b,
        epochs=args.epochs, optimizer=optimizer_b, criterion=criterion,
        carbon_tracker=tracker_b, verbose=True, device=str(DEVICE),
    )

    result_b = evaluate(model_b, val_b, criterion, device=str(DEVICE))
    co2_b = history_b['co2_result']
    print(f"      AG News accuracy: {result_b['accuracy']:.1%}")

    # Summary
    print(f"\n  These are our baselines. Every transfer method must beat {result_b['accuracy']:.1%}")
    print(f"  on AG News to prove it's worth the complexity.")
    print(f"\n  CO2: sentiment={co2_a['co2_kg']:.2e} kg  "
          f"news={co2_b['co2_kg']:.2e} kg  "
          f"({co2_a['source']} measurement, {co2_a['power_watts']:.0f}W avg)")

    _carbon_log.extend([co2_a, co2_b])
    _summary_rows.append(("Phase 1", "Full Fine-tune", "SST-2",
                           result_a['accuracy'], f"{total_params:,} params"))
    _summary_rows.append(("Phase 1", "Full Fine-tune (BASELINE)", "AG News",
                           result_b['accuracy'], f"{total_params:,} params"))
    return model, model_b, {
        'sst2': {'result': result_a, 'carbon': co2_a,
                 'time': co2_a.get('time_s', 0)},
        'ag_news': {'result': result_b, 'carbon': co2_b,
                    'time': co2_b.get('time_s', 0)},
    }


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
    """
    Measure how much fine-tuning changes each layer's representations.

    Standard CKA for transfer learning: pass the SAME inputs through a base
    model and a fine-tuned model, compare representations at each layer.

    High CKA = layer barely changed = universal features = safe to transfer.
    Low CKA  = layer specialized for the task = needs adaptation.

    This directly motivates which layers to freeze/transfer (Phase 5).
    """
    print("\n" + "=" * 72)
    print("  Chapter 2: The Diagnosis")
    print("  \"Before we transfer — are these tasks even related?\"")
    print("=" * 72)
    print("\n  If sentiment and news use completely different features inside")
    print("  the model, transfer will hurt. CKA measures this: we pass the")
    print("  same inputs through the base model and fine-tuned model, and")
    print("  check how much each layer changed.")

    _, val_a = data_a
    _, val_b = data_b

    # Full layer outputs — the complete contextualized representation at each depth
    layer_names = [
        "transformer.transformer.layer.0.output_layer_norm",
        "transformer.transformer.layer.1.output_layer_norm",
        "transformer.transformer.layer.2.output_layer_norm",
        "transformer.transformer.layer.3.output_layer_norm",
        "transformer.transformer.layer.4.output_layer_norm",
        "transformer.transformer.layer.5.output_layer_norm",
    ]
    layer_labels = [
        "Layer 0 (embeddings)",
        "Layer 1 (early)",
        "Layer 2 (early-mid)",
        "Layer 3 (mid)",
        "Layer 4 (deep)",
        "Layer 5 (final)",
    ]

    # Load a fresh base model for comparison
    # Note: num_labels doesn't matter for CKA — we only extract hidden states
    # before the classifier head. Using num_labels=2 to match the fine-tuned model.
    print(f"\n  Comparing base DistilBERT vs sentiment-fine-tuned model...")
    base_model = LLMClassifier(MODEL_NAME, num_labels=2).to(DEVICE)

    # ── A: Sentiment data (in-domain for fine-tuned model) ──
    print(f"\n  [A] On Sentiment data (in-domain):")
    print(f"      CKA(base, fine-tuned) — higher = layer unchanged by training")
    print(f"  {'Layer':<25} {'CKA':>8} {'Interpretation'}")
    print("  " + "-" * 60)

    sentiment_ckas = []
    for layer, label in zip(layer_names, layer_labels):
        try:
            reps_base = _extract_cls_representations(base_model, val_a, layer,
                                                      device=str(DEVICE))
            reps_ft = _extract_cls_representations(model_a, val_a, layer,
                                                    device=str(DEVICE))
            cka = compute_cka(reps_base, reps_ft)
            sentiment_ckas.append(cka)
            interp = "universal" if cka > 0.8 else "shared" if cka > 0.5 else "task-specific"
            bar = "█" * int(cka * 20) + "░" * (20 - int(cka * 20))
            print(f"  {label:<25} {cka:>7.4f} {bar} {interp}")
        except Exception as e:
            print(f"  {label:<25} {'error':>7} ({e})")

    # Free intermediate tensors before next batch of extractions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── B: News data (out-of-domain — will the backbone transfer?) ──
    print(f"\n  [B] On News data (out-of-domain):")
    print(f"      CKA(base, fine-tuned) — high means fine-tuning didn't hurt")
    print(f"  {'Layer':<25} {'CKA':>8} {'Interpretation'}")
    print("  " + "-" * 60)

    news_ckas = []
    for layer, label in zip(layer_names, layer_labels):
        try:
            reps_base = _extract_cls_representations(base_model, val_b, layer,
                                                      device=str(DEVICE))
            reps_ft = _extract_cls_representations(model_a, val_b, layer,
                                                    device=str(DEVICE))
            cka = compute_cka(reps_base, reps_ft)
            news_ckas.append(cka)
            interp = "safe to transfer" if cka > 0.8 else "transferable" if cka > 0.5 else "needs adaptation"
            bar = "█" * int(cka * 20) + "░" * (20 - int(cka * 20))
            print(f"  {label:<25} {cka:>7.4f} {bar} {interp}")
        except Exception as e:
            print(f"  {label:<25} {'error':>7} ({e})")

    # Free GPU memory before MMD computation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── C: Representation MMD (domain divergence in learned features) ──
    # NOTE: We use CLS-pooled representations (768-dim) rather than the library's
    # compute_representation_mmd which flattens to (N, seq_len*hidden) = (N, 98304).
    # The RBF kernel's median heuristic creates an (N_total, N_total, d) intermediate
    # array — with 98K dims and 1000+ samples that's hundreds of GB → instant OOM.
    # CLS pooling gives the same semantic signal in a tractable 768-dim space.
    # We also cap samples to 200/domain since the kernel still creates N²×d arrays.
    MAX_MMD_SAMPLES = 200
    print(f"\n  [C] Representation MMD — domain divergence in learned features:")
    print(f"      MMD measures how different sentiment vs news data look inside")
    print(f"      the model's layers. High MMD = domains far apart = risk.")
    print(f"      (Using [CLS]-pooled representations, capped at {MAX_MMD_SAMPLES} samples/domain)")
    mmd_layer = "transformer.transformer.layer.5.output_layer_norm"
    try:
        # Extract CLS-pooled representations (768-dim, not 98K-dim)
        reps_a = _extract_cls_representations(model_a, val_a, mmd_layer,
                                               device=str(DEVICE))
        reps_b = _extract_cls_representations(model_a, val_b, mmd_layer,
                                               device=str(DEVICE))
        # Subsample to cap memory: RBF kernel creates (n_a+n_b, n_a+n_b, d) array
        if reps_a.shape[0] > MAX_MMD_SAMPLES:
            idx = torch.randperm(reps_a.shape[0])[:MAX_MMD_SAMPLES]
            reps_a = reps_a[idx]
        if reps_b.shape[0] > MAX_MMD_SAMPLES:
            idx = torch.randperm(reps_b.shape[0])[:MAX_MMD_SAMPLES]
            reps_b = reps_b[idx]
        mmd_val = compute_mmd(reps_a.numpy(), reps_b.numpy())
        print(f"      MMD(sentiment, news) at final layer: {mmd_val:.6f}")
        if mmd_val < 0.05:
            print(f"      → Low divergence: domains similar in learned space — transfer is safe")
        elif mmd_val < 0.5:
            print(f"      → Moderate divergence: transfer should help but monitor for issues")
        else:
            print(f"      → High divergence: domains differ significantly — proceed with caution")
    except Exception as e:
        print(f"      MMD computation error: {e}")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary
    avg_sent = np.mean(sentiment_ckas) if sentiment_ckas else 0
    avg_news = np.mean(news_ckas) if news_ckas else 0
    print(f"\n  Average CKA — sentiment: {avg_sent:.4f}  news: {avg_news:.4f}")
    if sentiment_ckas and news_ckas:
        print(f"  Early layers (0-2): {np.mean(sentiment_ckas[:3]):.4f} sent / "
              f"{np.mean(news_ckas[:3]):.4f} news  → mostly preserved")
        print(f"  Late layers  (3-5): {np.mean(sentiment_ckas[3:]):.4f} sent / "
              f"{np.mean(news_ckas[3:]):.4f} news  → more specialized")
    print("\n  Key insight: early layers retain universal features (high CKA),")
    print("  making them ideal for transfer. Late layers specialize — hence")
    print("  EWC (Phase 4) and progressive unfreezing (Phase 5).")


# ═══════════════════════════════════════════════════════════════════
# Phase 3: LoRA Fine-tuning
# ═══════════════════════════════════════════════════════════════════

def demo_lora(args, tokenizer, data_a, pretrained_model, phase1_result=None):
    """Compare full fine-tuning vs LoRA on sentiment analysis."""
    print("\n" + "=" * 72)
    print("  Chapter 3: The Efficient Path")
    print("  \"Can we adapt a 66M-parameter model by training only 0.1%?\"")
    print("=" * 72)
    print("\n  Full fine-tuning updates all 66M parameters — expensive. LoRA")
    print("  freezes the model and injects tiny trainable matrices into the")
    print("  attention layers. Same model, 900x fewer trainable parameters.")

    train_a, val_a = data_a
    criterion = nn.CrossEntropyLoss()

    # Full fine-tuning baseline — reuse Phase 1 result if available
    total_p = sum(p.numel() for p in pretrained_model.parameters())
    if phase1_result is not None:
        print("\n  [A] Full fine-tuning baseline (reusing Phase 1 SST-2 result)...")
        result_full = phase1_result['result']
        carbon_full = phase1_result['carbon']
        full_time = phase1_result['time']
        print(f"    Accuracy: {result_full['accuracy']:.1%}  "
              f"Time: {full_time:.1f}s  Params: {total_p:,}  (from Phase 1)")
    else:
        print("\n  [A] Full fine-tuning (all 66M params trainable)...")
        model_full = LLMClassifier(MODEL_NAME, num_labels=2).to(DEVICE)
        tracker_full = _make_tracker("Ch3: Full FT")
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

    # LoRA fine-tuning (same fresh model — only LoRA params trained)
    print(f"\n  [B] LoRA fine-tuning (rank={args.lora_rank}, Q/V projections)...")
    model_lora = LLMClassifier(MODEL_NAME, num_labels=2).to(DEVICE)
    n_injected = LoRAInjector.inject(
        model_lora,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention Q and V
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,
    )
    model_lora.to(DEVICE)  # LoRA layers created on CPU, move them to device
    lora_params = LoRAInjector.get_lora_parameters(model_lora)
    # Include classifier head — it's freshly initialized and MUST be trained
    classifier_params = list(model_lora.classifier.parameters())
    all_trainable = list(lora_params) + classifier_params
    lora_trainable = LoRAInjector.count_lora_params(model_lora)
    clf_trainable = sum(p.numel() for p in classifier_params)
    print(f"    Injected into {n_injected} layers")
    print(f"    LoRA params: {lora_trainable:,} + classifier: {clf_trainable:,} "
          f"= {lora_trainable + clf_trainable:,} / {total_p:,} "
          f"({(lora_trainable + clf_trainable)/total_p:.2%} of model)")

    tracker_lora = _make_tracker("Ch3: LoRA FT")
    opt_lora = torch.optim.AdamW(all_trainable, lr=args.lr * 15)

    lora_epochs = max(args.epochs, 5)  # LoRA needs more epochs with fewer params
    tracker_lora.start()
    t0 = time.time()
    for ep in range(lora_epochs):
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

    if phase1_result is None:
        _carbon_log.extend([carbon_full, carbon_lora])
    else:
        _carbon_log.append(carbon_lora)
    _summary_rows.append(("Phase 3", "Full Fine-tune (=Phase 1)", "SST-2",
                           result_full['accuracy'], f"{total_p:,} params"))
    _summary_rows.append(("Phase 3", "LoRA (rank={})".format(args.lora_rank),
                           "SST-2", result_lora['accuracy'],
                           f"{lora_trainable + clf_trainable:,} trainable, "
                           f"{lora_epochs} epochs"))
    return model_lora, carbon_full, carbon_lora


# ═══════════════════════════════════════════════════════════════════
# Phase 4: EWC Transfer (Sentiment → News)
# ═══════════════════════════════════════════════════════════════════

def demo_ewc(args, pretrained_model, data_a, data_b):
    """Transfer from sentiment to news with EWC to prevent forgetting."""
    print("\n" + "=" * 72)
    print("  Chapter 4: The Transfer")
    print("  \"Can sentiment knowledge help with news classification?\"")
    print("=" * 72)
    print("\n  We take the sentiment model's backbone and attach a new 4-class")
    print("  head for news. EWC adds a penalty that protects the parameters")
    print("  most important for sentiment — so the model learns news without")
    print("  forgetting what it already knows.")

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

    # Compute pre-training baseline val loss for true negative transfer detection
    pre_train_result = evaluate(model_no_ewc, val_b, criterion, device=str(DEVICE))
    baseline_val_loss = pre_train_result['loss']
    print(f"    Pre-training baseline val_loss: {baseline_val_loss:.4f} "
          f"(acc={pre_train_result['accuracy']:.1%})")

    # Initialize Negative Transfer Monitors with true baseline
    monitor_no = NegativeTransferMonitor(
        reference_model=model_no_ewc, patience=2,
        baseline_val_loss=baseline_val_loss)
    monitor_ewc = NegativeTransferMonitor(
        reference_model=model_ewc, patience=2,
        baseline_val_loss=baseline_val_loss)
    print(f"    NegativeTransferMonitor initialized (patience=2, "
          f"baseline_loss={baseline_val_loss:.4f})")

    # Fine-tune on Task B WITHOUT EWC
    print("\n  [A] Fine-tuning on news WITHOUT EWC...")
    tracker_no_ewc = _make_tracker("Ch4: No EWC")
    tracker_no_ewc.start()
    opt = torch.optim.AdamW(model_no_ewc.parameters(), lr=args.lr)
    warnings_no = []
    for ep in range(args.epochs):
        loss = train_epoch(model_no_ewc, train_b, criterion, opt,
                           device=str(DEVICE))
        r = evaluate(model_no_ewc, val_b, criterion, device=str(DEVICE))
        warning = monitor_no.check(ep, r['loss'], model_no_ewc)
        if not args.quiet:
            print(f"      epoch {ep+1}/{args.epochs}  loss={loss:.4f}  "
                  f"news_acc={r['accuracy']:.1%}")
        if warning:
            warnings_no.append(warning)
            print(f"      >>> {warning}")
    carbon_no_ewc = tracker_no_ewc.stop()
    result_no_ewc = evaluate(model_no_ewc, val_b, criterion, device=str(DEVICE))

    # Fine-tune on Task B WITH EWC
    print(f"\n  [B] Fine-tuning on news WITH EWC (lambda=500)...")
    tracker_ewc = _make_tracker("Ch4: With EWC")
    tracker_ewc.start()
    ewc_loss = EWCLoss(source_model, fisher, lambda_=500.0).to(DEVICE)
    opt = torch.optim.AdamW(model_ewc.parameters(), lr=args.lr)
    warnings_ewc = []
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
        r = evaluate(model_ewc, val_b, criterion, device=str(DEVICE))
        warning = monitor_ewc.check(ep, r['loss'], model_ewc)
        if not args.quiet:
            print(f"      epoch {ep+1}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"news_acc={r['accuracy']:.1%}")
        if warning:
            warnings_ewc.append(warning)
            print(f"      >>> {warning}")
    carbon_ewc = tracker_ewc.stop()
    result_ewc = evaluate(model_ewc, val_b, criterion, device=str(DEVICE))
    _carbon_log.extend([carbon_no_ewc, carbon_ewc])

    # Negative Transfer Monitor Report
    print("\n  Negative Transfer Monitor Report:")
    print(f"    Tracked {len(monitor_no.history)} epochs per model")
    if not warnings_no and not warnings_ewc:
        print(f"    No negative transfer detected in either model (good!)")
        print(f"    Both models improved over their baselines within patience window")
    else:
        if warnings_no:
            print(f"    WARNING: Negative transfer in no-EWC model ({len(warnings_no)}x)")
        if warnings_ewc:
            print(f"    WARNING: Negative transfer in EWC model ({len(warnings_ewc)}x)")

    # Parameter drift from initial state (via NegativeTransferMonitor)
    print("\n  Backbone parameter drift (via NegativeTransferMonitor.parameter_drift):")
    drift_full_no = monitor_no.parameter_drift(model_no_ewc)
    drift_full_ewc = monitor_ewc.parameter_drift(model_ewc)

    # Filter to backbone only for comparison
    drift_no_ewc = {k: v for k, v in drift_full_no.items()
                    if not k.startswith("classifier.")}
    drift_ewc = {k: v for k, v in drift_full_ewc.items()
                 if not k.startswith("classifier.")}

    avg_drift_no = np.mean(list(drift_no_ewc.values()))
    avg_drift_ewc = np.mean(list(drift_ewc.values()))
    print(f"    Without EWC: avg backbone drift = {avg_drift_no:.4f}")
    print(f"    With EWC:    avg backbone drift = {avg_drift_ewc:.4f}")

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
    else:
        print(f"\n    Note: EWC constrains important parameters selectively, not")
        print(f"    all parameters. Average drift can be similar while the most")
        print(f"    critical weights are protected — accuracy tells the story.")

    _summary_rows.append(("Phase 4", "Baseline (no EWC)", "AG News",
                           result_no_ewc['accuracy'], f"drift={avg_drift_no:.4f}"))
    _summary_rows.append(("Phase 4", "EWC Transfer (λ=500)", "AG News",
                           result_ewc['accuracy'], f"drift={avg_drift_ewc:.4f}"))
    return model_no_ewc, model_ewc


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Progressive Unfreezing
# ═══════════════════════════════════════════════════════════════════

def _get_distilbert_layer_groups(model):
    """
    Build granular layer groups for DistilBERT progressive unfreezing.

    BaseModel.get_layer_groups() uses named_children() which gives only
    2 groups (transformer, classifier) — unfreezing jumps from head-only
    to everything in one step. This function splits the transformer into
    its individual layers for true gradual unfreezing:

        [0] embeddings       — word/position embeddings (shallowest)
        [1] layer_0           — transformer block 0
        ...
        [6] layer_5           — transformer block 5
        [7] classifier        — task head (deepest, unfrozen first)
    """
    groups = []

    # Embeddings
    embed_params = list(model.transformer.embeddings.parameters())
    if embed_params:
        groups.append(("embeddings", embed_params))

    # Individual transformer layers (DistilBERT has 6)
    for i, layer in enumerate(model.transformer.transformer.layer):
        params = list(layer.parameters())
        if params:
            groups.append((f"layer_{i}", params))

    # Classifier head
    clf_params = list(model.classifier.parameters())
    if clf_params:
        groups.append(("classifier", clf_params))

    return groups


def demo_progressive(args, pretrained_model, data_b):
    """Progressive unfreezing with discriminative LRs on news task."""
    print("\n" + "=" * 72)
    print("  Chapter 5: The Careful Approach")
    print("  \"What if we unfreeze the model gradually instead of all at once?\"")
    print("=" * 72)
    print("\n  Instead of training everything from the start (risky — random")
    print("  classifier gradients can corrupt pretrained features), we first")
    print("  train only the classifier head, then gradually unfreeze deeper")
    print("  layers one at a time — each with a smaller learning rate.")

    train_b, val_b = data_b
    criterion = nn.CrossEntropyLoss()

    model = LLMClassifier(MODEL_NAME, num_labels=4)
    load_backbone_state_dict(model, get_backbone_state_dict(pretrained_model))
    model = model.to(DEVICE)

    # Use granular layer groups (8 groups instead of 2)
    base_model = BaseModel(model)
    groups = _get_distilbert_layer_groups(model)
    print(f"\n  Layer groups: {len(groups)} (embeddings + 6 transformer layers + classifier)")
    for i, (name, params) in enumerate(groups):
        n_p = sum(p.numel() for p in params)
        print(f"    [{i}] {name}: {n_p:,} params")

    # Use enough epochs to see gradual unfreezing
    prog_epochs = max(args.epochs, len(groups))
    scheduler = TransferScheduler(groups, base_lr=args.lr, decay=2.6)

    tracker_prog = _make_tracker("Ch5: Progressive Unfreeze")
    tracker_prog.start()
    print(f"\n  Training with progressive unfreezing ({prog_epochs} epochs)...")
    print(f"  Epoch 0: classifier only → each epoch unfreezes the next deeper layer")

    prev_unfrozen = scheduler._unfrozen_up_to
    opt = scheduler.build_optimizer()
    for ep in range(prog_epochs):
        scheduler.step(ep)
        # Only rebuild optimizer when new parameters are unfrozen
        if scheduler._unfrozen_up_to != prev_unfrozen:
            # Preserve existing optimizer state where possible
            opt = scheduler.build_optimizer()
            prev_unfrozen = scheduler._unfrozen_up_to
        loss = train_epoch(base_model, train_b, criterion, opt,
                           device=str(DEVICE))
        if not args.quiet:
            r = evaluate(base_model, val_b, criterion, device=str(DEVICE))
            unfrozen = sum(1 for _, ps in groups for p in ps if p.requires_grad)
            total_tensors = sum(1 for _, ps in groups for _ in ps)
            # Show which groups are unfrozen
            unfrozen_names = [name for name, ps in groups
                              if any(p.requires_grad for p in ps)]
            print(f"      epoch {ep+1}/{prog_epochs}  loss={loss:.4f}  "
                  f"acc={r['accuracy']:.1%}  "
                  f"unfrozen={unfrozen}/{total_tensors}  "
                  f"[{', '.join(unfrozen_names)}]")

    carbon_prog = tracker_prog.stop()
    _carbon_log.append(carbon_prog)
    result = evaluate(base_model, val_b, criterion, device=str(DEVICE))
    print(f"\n  Final news accuracy: {result['accuracy']:.1%}")
    _summary_rows.append(("Phase 5", "Progressive Unfreezing", "AG News",
                           result['accuracy'],
                           f"{len(groups)} groups, {prog_epochs} epochs"))
    return model


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Model Merging (Backbone-Only)
# ═══════════════════════════════════════════════════════════════════

def demo_merging(args, pretrained_model, data_b):
    """Merge two fine-tuned LLM backbones using 5 strategies."""
    print("\n" + "=" * 72)
    print("  Chapter 6: The Ensemble (Without the Cost)")
    print("  \"Two models trained on different data — can we combine their strengths?\"")
    print("=" * 72)
    print("\n  Traditional ensembles run multiple models at inference (2x cost).")
    print("  Model merging combines weights directly — one model with the")
    print("  knowledge of both. Zero extra cost at inference.")
    print("\n  We train two variants on different halves of the data (model soups")
    print("  style), then merge their backbones with 5 strategies.")

    train_b, val_b = data_b
    criterion = nn.CrossEntropyLoss()

    # Split training data into two halves for genuinely different views
    all_data = list(train_b.dataset)
    mid = len(all_data) // 2
    split1 = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_b.dataset, range(0, mid)),
        batch_size=train_b.batch_size, shuffle=True)
    split2 = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_b.dataset, range(mid, len(all_data))),
        batch_size=train_b.batch_size, shuffle=True)

    tracker_merge = _make_tracker("Ch6: Model Merging")
    tracker_merge.start()

    # Train variant 1 (first half of data)
    print(f"\n  Training variant 1 (data split A, lr={args.lr:.0e})...")
    model_v1 = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    load_backbone_state_dict(model_v1, get_backbone_state_dict(pretrained_model))
    opt_v1 = torch.optim.AdamW(model_v1.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(model_v1, split1, criterion, opt_v1, device=str(DEVICE))
    r1 = evaluate(model_v1, val_b, criterion, device=str(DEVICE))
    print(f"    Accuracy: {r1['accuracy']:.1%}")

    # Train variant 2 (second half of data)
    print(f"\n  Training variant 2 (data split B, lr={args.lr:.0e})...")
    model_v2 = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    load_backbone_state_dict(model_v2, get_backbone_state_dict(pretrained_model))
    opt_v2 = torch.optim.AdamW(model_v2.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        train_epoch(model_v2, split2, criterion, opt_v2, device=str(DEVICE))
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

    # Average classifier heads from both variants for fair merging
    avg_clf_sd = {}
    for k in model_v1.classifier.state_dict():
        avg_clf_sd[k] = (model_v1.classifier.state_dict()[k] +
                         model_v2.classifier.state_dict()[k]) / 2.0

    # Merge strategies
    print("\n  Merging with 5 strategies...")
    print("  (classifier head: averaged from both variants)")
    merge_results = {}

    # Reuse a single model shell to avoid loading pretrained weights 5 times
    _merge_shell = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)

    def eval_merged(merged_sd, name):
        load_backbone_state_dict(_merge_shell, merged_sd)
        _merge_shell.classifier.load_state_dict(avg_clf_sd)
        r = evaluate(_merge_shell, val_b, criterion, device=str(DEVICE))
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

    carbon_merge = tracker_merge.stop()
    _carbon_log.append(carbon_merge)

    _summary_rows.append(("Phase 6", "Variant 1 (split A)", "AG News",
                           r1['accuracy'], f"lr={args.lr:.0e}, half data"))
    _summary_rows.append(("Phase 6", "Variant 2 (split B)", "AG News",
                           r2['accuracy'], f"lr={args.lr:.0e}, half data"))
    for name, r in merge_results.items():
        _summary_rows.append(("Phase 6", f"Merge: {name}", "AG News",
                               r['accuracy'], "avg classifier"))

    return model_v1, model_v2


# ═══════════════════════════════════════════════════════════════════
# Phase 7: LoRA Soups + LoRA-Flow
# ═══════════════════════════════════════════════════════════════════

def demo_lora_flow(args, pretrained_model, data_b):
    """Merge LoRA adapters and learn dynamic gating."""
    print("\n" + "=" * 72)
    print("  Chapter 7: The Smart Blend")
    print("  \"Can we merge lightweight LoRA adapters — and learn how?\"")
    print("=" * 72)
    print("\n  Instead of merging full 66M-param models, we merge just the")
    print("  tiny LoRA adapters. Naive averaging often fails. LoRA-Flow")
    print("  learns to dynamically weight adapters based on each input.")

    train_b, val_b = data_b
    criterion = nn.CrossEntropyLoss()
    target_modules = ["q_lin", "v_lin"]

    tracker_flow = _make_tracker("Ch7: LoRA Soup+Flow")
    tracker_flow.start()

    # Train LoRA adapter 1
    print(f"\n  Training LoRA adapter 1 (rank={args.lora_rank}, lr={args.lr*15:.0e})...")
    model_l1 = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    load_backbone_state_dict(model_l1, get_backbone_state_dict(pretrained_model))
    LoRAInjector.inject(model_l1, target_modules=target_modules,
                         rank=args.lora_rank, alpha=args.lora_rank * 2)
    model_l1.to(DEVICE)  # LoRA layers created on CPU, move to device
    # Include classifier head — it's freshly initialized and MUST be trained
    l1_trainable = list(LoRAInjector.get_lora_parameters(model_l1)) + \
                   list(model_l1.classifier.parameters())
    opt1 = torch.optim.AdamW(l1_trainable, lr=args.lr * 15)
    lora_epochs = max(args.epochs, 5)  # LoRA needs more epochs with fewer params
    for ep in range(lora_epochs):
        train_epoch(model_l1, train_b, criterion, opt1, device=str(DEVICE))
    r1 = evaluate(model_l1, val_b, criterion, device=str(DEVICE))
    print(f"    Adapter 1 accuracy: {r1['accuracy']:.1%}  "
          f"({LoRAInjector.count_lora_params(model_l1):,} LoRA params)")

    # Train LoRA adapter 2 (different LR for diversity)
    print(f"\n  Training LoRA adapter 2 (rank={args.lora_rank}, lr={args.lr*10:.0e})...")
    model_l2 = LLMClassifier(MODEL_NAME, num_labels=4).to(DEVICE)
    load_backbone_state_dict(model_l2, get_backbone_state_dict(pretrained_model))
    LoRAInjector.inject(model_l2, target_modules=target_modules,
                         rank=args.lora_rank, alpha=args.lora_rank * 2)
    model_l2.to(DEVICE)  # LoRA layers created on CPU, move to device
    # Include classifier head — it's freshly initialized and MUST be trained
    l2_trainable = list(LoRAInjector.get_lora_parameters(model_l2)) + \
                   list(model_l2.classifier.parameters())
    opt2 = torch.optim.AdamW(l2_trainable, lr=args.lr * 10)
    for ep in range(lora_epochs):
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
    # Average classifier heads from both adapters (not just adapter 1)
    avg_soup_clf = {}
    for k in model_l1.classifier.state_dict():
        avg_soup_clf[k] = (model_l1.classifier.state_dict()[k] +
                           model_l2.classifier.state_dict()[k]) / 2.0
    soup_model.classifier.load_state_dict(avg_soup_clf)
    LoRAInjector.merge_all(soup_model)

    r_soup = evaluate(soup_model, val_b, criterion, device=str(DEVICE))
    print(f"    LoRA Soup accuracy: {r_soup['accuracy']:.1%}")

    # LoRA-Flow: learned gating
    print("\n  LoRA-Flow: training learned gating weights...")
    hidden_size = 768  # distilbert hidden size
    flow = LoRAFlow(num_adapters=2, gate_input_dim=hidden_size).to(DEVICE)

    # Use the base pretrained transformer (no LoRA) for neutral gate input
    gate_backbone = copy.deepcopy(pretrained_model.transformer).to(DEVICE)
    gate_backbone.eval()

    def adapter_outputs_fn(batch):
        x = batch[0].to(DEVICE)
        with torch.no_grad():
            return [model_l1(x), model_l2(x)]

    def gate_input_fn(batch):
        x = batch[0].to(DEVICE)
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        with torch.no_grad():
            # Use base pretrained backbone (no LoRA) to avoid bias toward either adapter
            out = gate_backbone(
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

    carbon_flow = tracker_flow.stop()
    _carbon_log.append(carbon_flow)

    lora_p = LoRAInjector.count_lora_params(model_l1)
    _summary_rows.append(("Phase 7", "LoRA Adapter 1", "AG News",
                           r1['accuracy'],
                           f"{lora_p:,} LoRA + clf, {lora_epochs} epochs"))
    _summary_rows.append(("Phase 7", "LoRA Adapter 2", "AG News",
                           r2['accuracy'],
                           f"{lora_p:,} LoRA + clf, {lora_epochs} epochs"))
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
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch output")
    args = parser.parse_args()

    print("=" * 72)
    print("  Zeno v0.5.0 — A Transfer Learning Story")
    print("  \"One model, two tasks, seven ways to learn\"")
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
    phase1_info = None
    if args.demo == "all" or args.demo == "finetune":
        pretrained, _, phase1_info = demo_finetune(args, tokenizer, data_a, data_b)
    else:
        # Need a pretrained model for other phases
        print("\n  Quick-training base sentiment model...")
        set_seed(args.seed)
        pretrained = LLMClassifier(MODEL_NAME, num_labels=2).to(DEVICE)
        opt = torch.optim.AdamW(pretrained.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        for ep in range(args.epochs):
            train_epoch(pretrained, data_a[0], criterion, opt,
                        device=str(DEVICE))
        r = evaluate(pretrained, data_a[1], criterion, device=str(DEVICE))
        print(f"  Base sentiment accuracy: {r['accuracy']:.1%}")

    if args.demo == "all" or args.demo == "cka":
        set_seed(args.seed)
        demo_cka(args, pretrained, data_a, data_b)

    if args.demo == "all" or args.demo == "lora":
        set_seed(args.seed)
        # Reuse Phase 1 SST-2 result to avoid redundant full FT
        p1_sst2 = phase1_info['sst2'] if phase1_info else None
        demo_lora(args, tokenizer, data_a, pretrained, phase1_result=p1_sst2)

    if args.demo == "all" or args.demo == "ewc":
        set_seed(args.seed)
        demo_ewc(args, pretrained, data_a, data_b)

    if args.demo == "all" or args.demo == "progressive":
        set_seed(args.seed)
        demo_progressive(args, pretrained, data_b)

    if args.demo == "all" or args.demo == "merging":
        set_seed(args.seed)
        demo_merging(args, pretrained, data_b)

    if args.demo == "all" or args.demo == "lora_flow":
        set_seed(args.seed)
        demo_lora_flow(args, pretrained, data_b)

    # ─── CO2 Summary ───
    if _carbon_log:
        print("\n" + "=" * 72)
        print("  Epilogue: The Environmental Cost")
        print("  \"How much CO2 did all this learning produce?\"")
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
        # NOTE: This is a hypothetical illustration based on this toy demo
        # (small dataset, single task). Real-world savings vary greatly
        # depending on model size, dataset size, and hardware.
        co2_full = [r for r in _carbon_log if 'Full FT' in r['method']]
        co2_lora = [r for r in _carbon_log if 'LoRA FT' in r['method']]
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
                print(f"\n  Hypothetical projection across {N:,} training tasks "
                      f"(LoRA vs Full FT):")
                print(f"  ⚠ Illustrative only — based on this toy demo's "
                      f"compute profile ({args.max_samples} samples)")
                print(f"    CO2 saved per task:  {saved_per_task:.2e} kg "
                      f"({(saved_per_task/co2_full[0]['co2_kg'])*100:.1f}%)")
                print(f"    Total CO2 saved:     {total_saved_kg:.4f} kg")
                print(f"    = {p_phone:.0f} phone charge{'s' if p_phone != 1 else ''}")
                print(f"    = {p_searches:.0f} Google searches")
                print(f"    = {p_led:.1f} hours of LED lighting")
                print(f"    = {p_stream:.1f} hours of video streaming")
                print(f"    = {p_km:.4f} km driven")

    # ─── Final Summary Table ───
    if _summary_rows:
        print("\n" + "=" * 72)
        print("  The Full Story — Results Across All Chapters")
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

    # ─── Story Conclusion ───
    print("\n" + "=" * 72)
    print("  The Moral of the Story")
    print("=" * 72)
    if _summary_rows:
        baseline_b = [r for r in _summary_rows
                      if "BASELINE" in r[1] and r[2] == "AG News"]
        best_b = max((r for r in _summary_rows if r[2] == "AG News"),
                     key=lambda r: r[3], default=None)
        if baseline_b and best_b:
            base_acc = baseline_b[0][3]
            gain = (best_b[3] - base_acc) * 100
            print(f"\n  Starting point:  {base_acc:.1%} (full fine-tune from scratch)")
            print(f"  Best result:     {best_b[3]:.1%} ({best_b[1]})")
            if gain > 0:
                print(f"  Improvement:     +{gain:.1f} pp via transfer/merging/LoRA")
            else:
                print(f"  Difference:      {gain:+.1f} pp (baseline was competitive)")
    print(f"\n  Note: with only {args.max_samples} samples, accuracy differences of")
    print(f"  <5 pp are within statistical noise. Results are illustrative, not")
    print(f"  definitive — increase --max_samples for more stable comparisons.")
    print(f"\n  A pretrained model that knows English can learn new tasks faster")
    print(f"  and better by reusing what it already knows — if you're smart")
    print(f"  about how you do it. That's transfer learning.")
    print(f"\n  Every technique in this demo was built from scratch in Zeno —")
    print(f"  no PEFT, no HuggingFace Trainer, just PyTorch and our library.")
    print("=" * 72)


if __name__ == "__main__":
    main()
