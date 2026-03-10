"""
Smoke tests for deep learning extensions.

Tests all DL components using small synthetic models (no pretrained
downloads).  Run with: cd content && python -m pytest tests/test_dl_smoke.py -v
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest


# ---------- Fixtures ----------

@pytest.fixture
def small_mlp():
    """A small MLP for testing (no pretrained weights needed)."""
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    return model


@pytest.fixture
def classification_data():
    """Synthetic 4-class classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(200, 10)
    y = torch.randint(0, 4, (200,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return X, y, loader


@pytest.fixture
def source_target_loaders():
    """Source and target domain loaders for representation MMD."""
    torch.manual_seed(42)
    X_s = torch.randn(100, 10)
    y_s = torch.randint(0, 4, (100,))
    X_t = torch.randn(80, 10) + 1.0  # shifted
    y_t = torch.randint(0, 4, (80,))
    src_loader = DataLoader(TensorDataset(X_s, y_s), batch_size=32)
    tgt_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32)
    return src_loader, tgt_loader


# ---------- LoRA ----------

class TestLoRALinear:
    def test_zero_init(self):
        """LoRA should produce zero delta at initialization (Hu et al.)."""
        from libraries.dl.lora import LoRALinear
        linear = nn.Linear(20, 10)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(5, 20)
        # Output should match base linear at init (delta = 0)
        with torch.no_grad():
            base_out = linear(x)
            lora_out = lora(x)
        assert torch.allclose(base_out, lora_out, atol=1e-5), \
            "LoRALinear should produce identical output to base at init"

    def test_forward_shape(self):
        """LoRA forward should preserve output shape."""
        from libraries.dl.lora import LoRALinear
        linear = nn.Linear(20, 10)
        lora = LoRALinear(linear, rank=4)
        x = torch.randn(5, 20)
        out = lora(x)
        assert out.shape == (5, 10)

    def test_base_frozen(self):
        """Base linear weights should be frozen after LoRA wrapping."""
        from libraries.dl.lora import LoRALinear
        linear = nn.Linear(20, 10)
        lora = LoRALinear(linear, rank=4)
        assert not lora.linear.weight.requires_grad
        assert lora.lora_A.weight.requires_grad
        assert lora.lora_B.weight.requires_grad

    def test_merge_unmerge(self):
        """Merge should fuse LoRA into base; unmerge should reverse it."""
        from libraries.dl.lora import LoRALinear
        linear = nn.Linear(20, 10)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        # Modify LoRA weights so delta != 0
        with torch.no_grad():
            lora.lora_B.weight.fill_(0.1)

        x = torch.randn(5, 20)
        with torch.no_grad():
            out_before = lora(x).clone()

        lora.merge_weights()
        with torch.no_grad():
            out_merged = lora(x)
        assert torch.allclose(out_before, out_merged, atol=1e-4), \
            "Merged output should match pre-merge output"

        lora.unmerge_weights()
        with torch.no_grad():
            out_unmerged = lora(x)
        assert torch.allclose(out_before, out_unmerged, atol=1e-4), \
            "Unmerged output should match original output"

    def test_param_reduction(self):
        """LoRA should use far fewer parameters than the full layer."""
        from libraries.dl.lora import LoRALinear
        linear = nn.Linear(768, 768)
        lora = LoRALinear(linear, rank=8)
        assert lora.reduction_ratio() > 40, \
            f"Expected >40x reduction, got {lora.reduction_ratio():.1f}x"


class TestLoRAInjector:
    def test_inject_all(self, small_mlp):
        """Injecting into all Linear layers should replace 3 layers."""
        from libraries.dl.lora import LoRAInjector
        count = LoRAInjector.inject(small_mlp, target_modules=None, rank=4)
        assert count == 3, f"Expected 3 injections, got {count}"

    def test_inject_selective(self, small_mlp):
        """Selective injection should only replace matching layers."""
        from libraries.dl.lora import LoRAInjector
        # Only inject into layer named "0" (first linear)
        count = LoRAInjector.inject(small_mlp, target_modules=["0"], rank=4)
        assert count == 1, f"Expected 1 injection, got {count}"

    def test_get_lora_parameters(self, small_mlp):
        """get_lora_parameters should return only LoRA A and B weights."""
        from libraries.dl.lora import LoRAInjector
        LoRAInjector.inject(small_mlp, rank=4)
        params = LoRAInjector.get_lora_parameters(small_mlp)
        # 3 layers x 2 params (A, B) = 6
        assert len(params) == 6

    def test_lora_state_dict(self, small_mlp):
        """lora_state_dict should contain only LoRA weights."""
        from libraries.dl.lora import LoRAInjector
        LoRAInjector.inject(small_mlp, rank=4)
        state = LoRAInjector.lora_state_dict(small_mlp)
        assert len(state) == 6  # 3 layers x 2 (A, B)
        for key in state:
            assert "lora_A" in key or "lora_B" in key

    def test_count_lora_params(self, small_mlp):
        """count_lora_params should return total trainable LoRA parameters."""
        from libraries.dl.lora import LoRAInjector
        LoRAInjector.inject(small_mlp, rank=4)
        total = LoRAInjector.count_lora_params(small_mlp)
        assert total > 0


# ---------- Transfer ----------

class TestBaseModel:
    def test_freeze_all(self, small_mlp):
        """freeze_all should make all parameters non-trainable."""
        from libraries.dl.transfer import BaseModel
        base = BaseModel(small_mlp)
        base.freeze_all()
        counts = base.count_parameters()
        assert counts["trainable"] == 0

    def test_unfreeze_all(self, small_mlp):
        """unfreeze_all should make all parameters trainable."""
        from libraries.dl.transfer import BaseModel
        base = BaseModel(small_mlp)
        base.freeze_all()
        base.unfreeze_all()
        counts = base.count_parameters()
        assert counts["trainable"] == counts["total"]

    def test_count_parameters(self, small_mlp):
        """count_parameters should report correct totals."""
        from libraries.dl.transfer import BaseModel
        base = BaseModel(small_mlp)
        counts = base.count_parameters()
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]

    def test_replace_head(self):
        """replace_head should swap the classification head."""
        from libraries.dl.transfer import BaseModel
        model = nn.Module()
        model.features = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
        model.fc = nn.Linear(32, 100)
        model.forward = lambda x: model.fc(model.features(x))
        base = BaseModel(model)
        name = base.replace_head(num_classes=5)
        assert name == "fc"
        assert model.fc.out_features == 5

    def test_get_layer_groups(self, small_mlp):
        """get_layer_groups should return groups for each child module."""
        from libraries.dl.transfer import BaseModel
        base = BaseModel(small_mlp)
        groups = base.get_layer_groups()
        # Sequential has 5 children: Linear, ReLU, Linear, ReLU, Linear
        # ReLU has no parameters, so we get 3 groups with params
        assert len(groups) == 3

    def test_forward_passthrough(self, small_mlp):
        """BaseModel forward should produce same output as wrapped model."""
        from libraries.dl.transfer import BaseModel
        base = BaseModel(small_mlp)
        x = torch.randn(5, 10)
        with torch.no_grad():
            out_base = base(x)
            out_model = small_mlp(x)
        assert torch.allclose(out_base, out_model)


class TestTransferScheduler:
    def test_initial_state(self, small_mlp):
        """Only the head (last group) should be unfrozen initially."""
        from libraries.dl.transfer import BaseModel, TransferScheduler
        base = BaseModel(small_mlp)
        groups = base.get_layer_groups()
        scheduler = TransferScheduler(groups, base_lr=1e-3)

        # Only last group should have requires_grad=True
        for _, params in groups[:-1]:
            for p in params:
                assert not p.requires_grad, "Non-head layers should be frozen"
        for p in groups[-1][1]:
            assert p.requires_grad, "Head should be unfrozen"

    def test_progressive_unfreezing(self, small_mlp):
        """step() should progressively unfreeze deeper layers."""
        from libraries.dl.transfer import BaseModel, TransferScheduler
        base = BaseModel(small_mlp)
        groups = base.get_layer_groups()
        scheduler = TransferScheduler(groups, base_lr=1e-3)

        # After step(1), second-to-last group should unfreeze
        scheduler.step(1)
        for p in groups[-2][1]:
            assert p.requires_grad, "Second group should be unfrozen after step(1)"

    def test_build_param_groups(self, small_mlp):
        """build_param_groups should return groups with decayed LRs."""
        from libraries.dl.transfer import BaseModel, TransferScheduler
        base = BaseModel(small_mlp)
        groups = base.get_layer_groups()
        scheduler = TransferScheduler(groups, base_lr=1e-3, decay=2.0)

        # Unfreeze all
        for i in range(len(groups)):
            scheduler.step(i)

        param_groups = scheduler.build_param_groups()
        assert len(param_groups) > 0
        # LRs should be different (discriminative)
        lrs = [g["lr"] for g in param_groups]
        assert lrs[-1] == pytest.approx(1e-3), "Head should get base_lr"
        assert lrs[0] < lrs[-1], "Earlier layers should get lower LR"


class TestDiscriminativeLR:
    def test_build_groups(self, small_mlp):
        """build_discriminative_lr_groups should produce correct groups."""
        from libraries.dl.transfer import build_discriminative_lr_groups
        groups = build_discriminative_lr_groups(small_mlp, base_lr=1e-3, decay=2.0)
        assert len(groups) == 3  # 3 Linear layers with params
        lrs = [g["lr"] for g in groups]
        # Last group should have highest LR
        assert lrs[-1] > lrs[0]


# ---------- EWC ----------

class TestEWC:
    def test_fisher_diagonal_shapes(self, small_mlp, classification_data):
        """Fisher diagonal should have same shape as model parameters."""
        from libraries.dl.ewc import compute_fisher_diagonal
        _, _, loader = classification_data
        criterion = nn.CrossEntropyLoss()
        fisher = compute_fisher_diagonal(small_mlp, loader, criterion)
        for name, param in small_mlp.named_parameters():
            if param.requires_grad:
                assert name in fisher, f"Missing Fisher for {name}"
                assert fisher[name].shape == param.shape

    def test_fisher_nonnegative(self, small_mlp, classification_data):
        """Fisher diagonal values should be non-negative (squared grads)."""
        from libraries.dl.ewc import compute_fisher_diagonal
        _, _, loader = classification_data
        criterion = nn.CrossEntropyLoss()
        fisher = compute_fisher_diagonal(small_mlp, loader, criterion)
        for name, f in fisher.items():
            assert (f >= 0).all(), f"Fisher for {name} has negative values"

    def test_ewc_loss_penalty(self, small_mlp, classification_data):
        """EWC penalty should be > 0 when parameters have changed."""
        from libraries.dl.ewc import compute_fisher_diagonal, EWCLoss
        import copy
        _, _, loader = classification_data
        criterion = nn.CrossEntropyLoss()

        source_model = copy.deepcopy(small_mlp)
        fisher = compute_fisher_diagonal(source_model, loader, criterion)
        ewc = EWCLoss(source_model, fisher, lambda_=1000.0)

        # Modify the model's weights
        with torch.no_grad():
            for param in small_mlp.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        penalty = ewc(small_mlp)
        assert penalty.item() > 0, "EWC penalty should be > 0 when params differ"

    def test_ewc_loss_zero_at_source(self, small_mlp, classification_data):
        """EWC penalty should be 0 when model equals source."""
        from libraries.dl.ewc import compute_fisher_diagonal, EWCLoss
        import copy
        _, _, loader = classification_data
        criterion = nn.CrossEntropyLoss()

        source_model = copy.deepcopy(small_mlp)
        fisher = compute_fisher_diagonal(source_model, loader, criterion)
        ewc = EWCLoss(source_model, fisher, lambda_=1000.0)

        # Penalty on the source model itself should be 0
        penalty = ewc(source_model)
        assert penalty.item() == pytest.approx(0.0, abs=1e-6)

    def test_online_ewc_update(self):
        """online_ewc_update should blend old and new Fisher."""
        from libraries.dl.ewc import online_ewc_update
        old = {"w": torch.ones(5)}
        new = {"w": torch.ones(5) * 2}
        result = online_ewc_update(old, new, gamma=0.5)
        # Expected: 0.5 * 1 + 2 = 2.5
        assert torch.allclose(result["w"], torch.ones(5) * 2.5)


# ---------- Negative Transfer (DL) ----------

class TestCKA:
    def test_identical_representations(self):
        """CKA of identical representations should be 1.0."""
        from libraries.dl.negative_transfer import compute_cka
        X = torch.randn(50, 20)
        cka = compute_cka(X, X)
        assert cka == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_representations(self):
        """CKA of very different representations should be < 1.0."""
        from libraries.dl.negative_transfer import compute_cka
        torch.manual_seed(42)
        X = torch.randn(50, 20)
        Y = torch.randn(50, 20) * 100  # very different
        cka = compute_cka(X, Y)
        assert cka < 0.5, f"CKA between random matrices should be low, got {cka:.4f}"

    def test_cka_invariant_to_scaling(self):
        """CKA should be invariant to isotropic scaling."""
        from libraries.dl.negative_transfer import compute_cka
        X = torch.randn(50, 20)
        cka_original = compute_cka(X, X)
        cka_scaled = compute_cka(X, X * 5.0)
        assert cka_original == pytest.approx(cka_scaled, abs=1e-4)


class TestRepresentationExtraction:
    def test_extract_shape(self, small_mlp, classification_data):
        """Extracted representations should have correct shape."""
        from libraries.dl.negative_transfer import extract_representations
        _, _, loader = classification_data
        # Extract from the first linear layer (named "0")
        reps = extract_representations(small_mlp, loader, layer_name="0")
        assert reps.shape[0] == 200  # all samples
        assert reps.shape[1] == 32   # output dim of first linear

    def test_representation_mmd(self, small_mlp, source_target_loaders):
        """Representation MMD should be computable without errors."""
        from libraries.dl.negative_transfer import compute_representation_mmd
        src_loader, tgt_loader = source_target_loaders
        mmd = compute_representation_mmd(
            small_mlp, src_loader, tgt_loader, layer_name="0"
        )
        assert mmd >= 0


class TestNegativeTransferMonitor:
    def test_no_warning_initially(self):
        """No warning should be issued on the first epoch."""
        from libraries.dl.negative_transfer import NegativeTransferMonitor
        monitor = NegativeTransferMonitor(patience=3)
        warning = monitor.check(0, val_loss=0.5)
        assert warning is None

    def test_warning_after_patience(self):
        """Warning should fire after patience epochs of degradation."""
        from libraries.dl.negative_transfer import NegativeTransferMonitor
        monitor = NegativeTransferMonitor(patience=3, baseline_val_loss=0.5)
        monitor.check(0, val_loss=0.6)  # worse
        monitor.check(1, val_loss=0.7)  # worse
        warning = monitor.check(2, val_loss=0.8)  # 3rd consecutive
        assert warning is not None
        assert "Negative transfer" in warning

    def test_parameter_drift(self, small_mlp):
        """parameter_drift should detect changes from reference."""
        from libraries.dl.negative_transfer import NegativeTransferMonitor
        import copy
        reference = copy.deepcopy(small_mlp)
        monitor = NegativeTransferMonitor(reference_model=reference)

        # Modify model
        with torch.no_grad():
            for param in small_mlp.parameters():
                param.add_(1.0)

        drift = monitor.parameter_drift(small_mlp)
        assert len(drift) > 0
        assert all(d > 0 for d in drift.values())


# ---------- Carbon ----------

class TestGPUCarbonTracker:
    def test_cpu_fallback(self):
        """GPUCarbonTracker should work in CPU-only mode."""
        from libraries.dl.carbon import GPUCarbonTracker
        tracker = GPUCarbonTracker("test", power_watts=30)
        tracker.start()
        _ = sum(range(10000))
        result = tracker.stop()
        assert result["method"] == "test"
        assert result["time_s"] > 0
        assert result["co2_kg"] >= 0
        assert result["source"] == "manual"  # no GPU

    def test_compatible_with_compare_emissions(self):
        """GPUCarbonTracker output should work with compare_emissions."""
        from libraries.dl.carbon import GPUCarbonTracker
        from libraries.carbon import compare_emissions

        tracker1 = GPUCarbonTracker("full_ft", power_watts=250)
        tracker1.start()
        _ = sum(range(100000))
        r1 = tracker1.stop()

        tracker2 = GPUCarbonTracker("lora_ft", power_watts=100)
        tracker2.start()
        _ = sum(range(10000))
        r2 = tracker2.stop()

        summary = compare_emissions([r1, r2])
        assert summary["baseline"] == "full_ft"
        assert len(summary["comparisons"]) == 1


# ---------- Training ----------

class TestTraining:
    def test_train_epoch(self, small_mlp, classification_data):
        """train_epoch should return a finite loss."""
        from libraries.dl.train import train_epoch
        _, _, loader = classification_data
        optimizer = torch.optim.Adam(small_mlp.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        loss = train_epoch(small_mlp, loader, criterion, optimizer)
        assert loss > 0
        assert not np.isnan(loss)

    def test_evaluate(self, small_mlp, classification_data):
        """evaluate should return loss and accuracy."""
        from libraries.dl.train import evaluate
        _, _, loader = classification_data
        criterion = nn.CrossEntropyLoss()
        result = evaluate(small_mlp, loader, criterion)
        assert "loss" in result
        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1

    def test_training_reduces_loss(self, small_mlp, classification_data):
        """Multiple training epochs should reduce loss."""
        from libraries.dl.train import train_epoch, evaluate
        _, _, loader = classification_data
        optimizer = torch.optim.Adam(small_mlp.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        initial = evaluate(small_mlp, loader, criterion)["loss"]
        for _ in range(10):
            train_epoch(small_mlp, loader, criterion, optimizer)
        final = evaluate(small_mlp, loader, criterion)["loss"]

        assert final < initial, "Training should reduce loss"

    def test_fine_tune_integration(self, classification_data):
        """fine_tune should run end-to-end with all components."""
        from libraries.dl.train import fine_tune

        model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        X, y, _ = classification_data

        # Split into train/val
        train_ds = TensorDataset(X[:160], y[:160])
        val_ds = TensorDataset(X[160:], y[160:])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        history = fine_tune(
            model, train_loader, val_loader,
            epochs=5, optimizer=optimizer, criterion=criterion,
            verbose=False,
        )

        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5
        assert history["train_loss"][-1] < history["train_loss"][0]


# ---------- Model Merging ----------

class TestLinearMerge:
    def test_uniform_merge(self):
        """Uniform merge of identical models should return same weights."""
        from libraries.dl.merging import linear_merge
        sd = {"w": torch.ones(5), "b": torch.zeros(3)}
        merged = linear_merge([sd, sd])
        assert torch.allclose(merged["w"], sd["w"])
        assert torch.allclose(merged["b"], sd["b"])

    def test_weighted_merge(self):
        """Weighted merge should bias toward higher-weight model."""
        from libraries.dl.merging import linear_merge
        sd_a = {"w": torch.zeros(5)}
        sd_b = {"w": torch.ones(5)}
        merged = linear_merge([sd_a, sd_b], weights=[0.0, 1.0])
        assert torch.allclose(merged["w"], torch.ones(5))

    def test_three_model_merge(self):
        """Should handle merging three or more models."""
        from libraries.dl.merging import linear_merge
        sd_a = {"w": torch.ones(5) * 1.0}
        sd_b = {"w": torch.ones(5) * 2.0}
        sd_c = {"w": torch.ones(5) * 3.0}
        merged = linear_merge([sd_a, sd_b, sd_c])
        assert torch.allclose(merged["w"], torch.ones(5) * 2.0)


class TestSLERP:
    def test_endpoints(self):
        """SLERP at t=0 should return model_a, at t=1 model_b."""
        from libraries.dl.merging import slerp_merge
        sd_a = {"w": torch.randn(10)}
        sd_b = {"w": torch.randn(10)}
        merged_0 = slerp_merge(sd_a, sd_b, t=0.0)
        merged_1 = slerp_merge(sd_a, sd_b, t=1.0)
        assert torch.allclose(merged_0["w"], sd_a["w"], atol=1e-5)
        assert torch.allclose(merged_1["w"], sd_b["w"], atol=1e-5)

    def test_midpoint_norm_preservation(self):
        """SLERP at t=0.5 should preserve norms better than lerp."""
        from libraries.dl.merging import slerp_merge
        torch.manual_seed(42)
        # Two unit-norm vectors in different directions
        v0 = torch.randn(100)
        v0 = v0 / torch.norm(v0) * 10.0
        v1 = torch.randn(100)
        v1 = v1 / torch.norm(v1) * 10.0
        sd_a = {"w": v0}
        sd_b = {"w": v1}

        slerp_result = slerp_merge(sd_a, sd_b, t=0.5)
        lerp_result = 0.5 * v0 + 0.5 * v1

        # SLERP should preserve norm better than linear interpolation
        original_norm = 10.0
        slerp_norm = torch.norm(slerp_result["w"]).item()
        lerp_norm = torch.norm(lerp_result).item()

        # SLERP norm should be closer to original
        slerp_deviation = abs(slerp_norm - original_norm)
        lerp_deviation = abs(lerp_norm - original_norm)
        assert slerp_deviation <= lerp_deviation + 0.1, \
            f"SLERP deviation {slerp_deviation:.4f} > lerp {lerp_deviation:.4f}"

    def test_identical_models(self):
        """SLERP of identical models should return the same model."""
        from libraries.dl.merging import slerp_merge
        sd = {"w": torch.randn(10)}
        merged = slerp_merge(sd, sd, t=0.5)
        assert torch.allclose(merged["w"], sd["w"], atol=1e-5)


class TestTaskArithmetic:
    def test_compute_task_vector(self):
        """Task vector should be the difference between finetuned and base."""
        from libraries.dl.merging import compute_task_vector
        base = {"w": torch.ones(5)}
        finetuned = {"w": torch.ones(5) * 3.0}
        tv = compute_task_vector(base, finetuned)
        assert torch.allclose(tv["w"], torch.ones(5) * 2.0)

    def test_apply_task_vector(self):
        """Applying a task vector should add it to the base model."""
        from libraries.dl.merging import compute_task_vector, apply_task_vector
        base = {"w": torch.ones(5)}
        finetuned = {"w": torch.ones(5) * 3.0}
        tv = compute_task_vector(base, finetuned)
        result = apply_task_vector(base, tv, scaling=1.0)
        assert torch.allclose(result["w"], finetuned["w"])

    def test_negation(self):
        """Negative scaling should remove task knowledge."""
        from libraries.dl.merging import compute_task_vector, apply_task_vector
        base = {"w": torch.ones(5)}
        finetuned = {"w": torch.ones(5) * 3.0}
        tv = compute_task_vector(base, finetuned)
        result = apply_task_vector(base, tv, scaling=-1.0)
        # base + (-1) * (ft - base) = 2*base - ft
        assert torch.allclose(result["w"], torch.ones(5) * -1.0)

    def test_multi_task_merge(self):
        """Multiple task vectors should compose additively."""
        from libraries.dl.merging import (
            compute_task_vector, task_arithmetic_merge
        )
        base = {"w": torch.zeros(5)}
        ft_a = {"w": torch.ones(5) * 1.0}
        ft_b = {"w": torch.ones(5) * 2.0}
        tv_a = compute_task_vector(base, ft_a)
        tv_b = compute_task_vector(base, ft_b)
        merged = task_arithmetic_merge(base, [tv_a, tv_b], scalings=[0.5, 0.5])
        expected = torch.ones(5) * 1.5  # 0 + 0.5*1 + 0.5*2
        assert torch.allclose(merged["w"], expected)


class TestTIES:
    def test_resolves_sign_conflict(self):
        """TIES should resolve conflicting signs by majority vote."""
        from libraries.dl.merging import ties_merge
        base = {"w": torch.zeros(4)}
        # Three task vectors with sign conflicts at position 0
        tv1 = {"w": torch.tensor([1.0, 1.0, 1.0, 1.0])}
        tv2 = {"w": torch.tensor([1.0, 1.0, -1.0, 1.0])}
        tv3 = {"w": torch.tensor([1.0, -1.0, -1.0, -1.0])}
        # Position 0: all positive (3/3) → positive
        # Position 1: 2 positive, 1 negative → positive
        # Position 2: 1 positive, 2 negative → negative
        # Position 3: 2 positive, 1 negative → positive
        merged = ties_merge(base, [tv1, tv2, tv3], density=1.0, scaling=1.0)
        # The merged values should all be non-zero for density=1.0
        assert merged["w"][0] > 0, "Position 0 should be positive"
        assert merged["w"][1] > 0, "Position 1 should be positive (majority)"
        assert merged["w"][2] < 0, "Position 2 should be negative (majority)"

    def test_trimming_sparsifies(self):
        """Low density should zero out small-magnitude parameters."""
        from libraries.dl.merging import ties_merge
        base = {"w": torch.zeros(100)}
        torch.manual_seed(42)
        tv = {"w": torch.randn(100)}
        merged = ties_merge(base, [tv], density=0.1, scaling=1.0)
        # Many parameters should be zero after trimming
        nonzero = (merged["w"].abs() > 1e-10).sum().item()
        assert nonzero <= 20, \
            f"Expected sparse result, got {nonzero} nonzero params"

    def test_density_one_preserves_all(self):
        """Density=1.0 should keep all parameters (no trimming)."""
        from libraries.dl.merging import ties_merge
        base = {"w": torch.zeros(10)}
        tv = {"w": torch.ones(10)}
        merged = ties_merge(base, [tv], density=1.0, scaling=1.0)
        assert torch.allclose(merged["w"], torch.ones(10))


class TestDARE:
    def test_dare_reduces_density(self):
        """DARE with high drop rate should produce sparse task vectors."""
        from libraries.dl.merging import dare_merge
        base = {"w": torch.zeros(1000)}
        tv = {"w": torch.ones(1000)}
        merged = dare_merge(
            base, [tv], drop_rate=0.9, scaling=1.0,
            use_ties=False, seed=42
        )
        # With drop_rate=0.9, about 10% of params survive
        # But they're rescaled by 1/(1-0.9) = 10x
        # So nonzero values should be ~10.0
        nonzero = (merged["w"].abs() > 0.5).sum().item()
        assert 50 < nonzero < 200, \
            f"Expected ~100 nonzero params (10% of 1000), got {nonzero}"

    def test_dare_with_ties(self):
        """DARE + TIES should produce a valid merged model."""
        from libraries.dl.merging import dare_merge
        base = {"w": torch.zeros(100)}
        tv1 = {"w": torch.randn(100)}
        tv2 = {"w": torch.randn(100)}
        merged = dare_merge(
            base, [tv1, tv2], drop_rate=0.5, scaling=1.0,
            use_ties=True, ties_density=0.5, seed=42
        )
        assert "w" in merged
        assert merged["w"].shape == torch.Size([100])

    def test_dare_reproducible_with_seed(self):
        """Same seed should produce identical results."""
        from libraries.dl.merging import dare_merge
        base = {"w": torch.zeros(50)}
        tv = {"w": torch.randn(50)}
        m1 = dare_merge(base, [tv], drop_rate=0.5, use_ties=False, seed=123)
        m2 = dare_merge(base, [tv], drop_rate=0.5, use_ties=False, seed=123)
        assert torch.allclose(m1["w"], m2["w"])


class TestLoRASoups:
    def test_merge_two_adapters(self):
        """Merging two LoRA adapters should average their weights."""
        from libraries.dl.merging import merge_lora_adapters
        sd_a = {"layer.lora_A.weight": torch.ones(4, 8),
                "layer.lora_B.weight": torch.zeros(8, 4)}
        sd_b = {"layer.lora_A.weight": torch.ones(4, 8) * 3,
                "layer.lora_B.weight": torch.ones(8, 4) * 2}
        merged = merge_lora_adapters([sd_a, sd_b])
        assert torch.allclose(
            merged["layer.lora_A.weight"], torch.ones(4, 8) * 2.0)
        assert torch.allclose(
            merged["layer.lora_B.weight"], torch.ones(8, 4) * 1.0)

    def test_weighted_adapter_merge(self):
        """Weighted merge should bias toward higher-weight adapter."""
        from libraries.dl.merging import merge_lora_adapters
        sd_a = {"lora_A": torch.zeros(4, 8)}
        sd_b = {"lora_A": torch.ones(4, 8) * 10}
        merged = merge_lora_adapters([sd_a, sd_b], weights=[0.0, 1.0])
        assert torch.allclose(merged["lora_A"], torch.ones(4, 8) * 10)


class TestMergeAnalysis:
    def test_task_vector_stats(self):
        """task_vector_stats should return correct statistics."""
        from libraries.dl.merging import task_vector_stats
        tv = {"w": torch.tensor([1.0, 0.0, -3.0, 0.0, 2.0])}
        stats = task_vector_stats(tv)
        assert stats["total_params"] == 5
        assert stats["nonzero_params"] == 3
        assert stats["sparsity"] == pytest.approx(0.4)
        assert stats["max_magnitude"] == pytest.approx(3.0)

    def test_task_vector_similarity_identical(self):
        """Identical task vectors should have cosine similarity = 1.0."""
        from libraries.dl.merging import task_vector_similarity
        tv = {"w": torch.randn(20)}
        sim = task_vector_similarity(tv, tv)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_task_vector_similarity_opposite(self):
        """Opposite task vectors should have cosine similarity = -1.0."""
        from libraries.dl.merging import task_vector_similarity
        tv_a = {"w": torch.ones(10)}
        tv_b = {"w": -torch.ones(10)}
        sim = task_vector_similarity(tv_a, tv_b)
        assert sim == pytest.approx(-1.0, abs=1e-5)


class TestLoRAFlow:
    def test_uniform_initialization(self):
        """LoRA-Flow should produce uniform weights at initialization."""
        from libraries.dl.merging import LoRAFlow
        flow = LoRAFlow(num_adapters=3, gate_input_dim=10)
        x = torch.randn(5, 10)
        weights = flow(x)
        assert weights.shape == (5, 3)
        # Should be approximately uniform (1/3 each)
        assert torch.allclose(weights, torch.ones(5, 3) / 3.0, atol=1e-5)

    def test_merge_with_gates(self):
        """merge_with_gates should produce weighted combination."""
        from libraries.dl.merging import LoRAFlow
        flow = LoRAFlow(num_adapters=2, gate_input_dim=4)
        adapter_outs = [
            torch.ones(3, 8),  # adapter 1: all ones
            torch.ones(3, 8) * 2.0,  # adapter 2: all twos
        ]
        gate_input = torch.randn(3, 4)
        combined = flow.merge_with_gates(adapter_outs, gate_input)
        assert combined.shape == (3, 8)
        # At init, uniform weights: 0.5 * 1 + 0.5 * 2 = 1.5
        assert torch.allclose(combined, torch.ones(3, 8) * 1.5, atol=1e-4)

    def test_gate_params_trainable(self):
        """get_gate_params should return trainable parameters."""
        from libraries.dl.merging import LoRAFlow
        flow = LoRAFlow(num_adapters=3, gate_input_dim=16)
        params = flow.get_gate_params()
        assert len(params) == 2  # weight + bias
        assert all(p.requires_grad for p in params)

    def test_temperature_sharpens(self):
        """Lower temperature should produce sharper gating weights."""
        from libraries.dl.merging import LoRAFlow
        # Modify gate weights to have non-uniform logits
        flow_warm = LoRAFlow(num_adapters=3, gate_input_dim=4, temperature=5.0)
        flow_cold = LoRAFlow(num_adapters=3, gate_input_dim=4, temperature=0.1)
        # Set same gate weights
        with torch.no_grad():
            flow_warm.gate.weight.copy_(torch.randn(3, 4))
            flow_cold.gate.weight.copy_(flow_warm.gate.weight)
            flow_warm.gate.bias.copy_(torch.tensor([1.0, 0.0, -1.0]))
            flow_cold.gate.bias.copy_(flow_warm.gate.bias)

        x = torch.randn(10, 4)
        warm_weights = flow_warm(x)
        cold_weights = flow_cold(x)

        # Cold should have higher max (sharper distribution)
        assert cold_weights.max(dim=-1).values.mean() > warm_weights.max(dim=-1).values.mean()

    def test_train_lora_flow(self):
        """train_lora_flow should reduce loss and update gate weights."""
        from libraries.dl.merging import LoRAFlow, train_lora_flow

        torch.manual_seed(42)
        flow = LoRAFlow(num_adapters=2, gate_input_dim=10)

        # Create synthetic scenario: adapter 1 is better for this task
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Adapter outputs: adapter 0 = random noise, adapter 1 = closer to targets
        def adapter_outputs_fn(batch):
            x, _ = batch
            return [
                torch.randn(x.shape[0], 3),  # random
                torch.randn(x.shape[0], 3) * 0.1,  # better baseline
            ]

        def gate_input_fn(batch):
            x, _ = batch
            return x

        def target_fn(batch):
            _, y = batch
            return y

        criterion = nn.CrossEntropyLoss()
        history = train_lora_flow(
            flow, adapter_outputs_fn, gate_input_fn, loader,
            criterion, target_fn, epochs=5, lr=0.01
        )

        assert len(history["loss"]) == 5
        assert len(history["gate_weights"]) == 5
        # Gate weights should have 2 entries per epoch
        assert len(history["gate_weights"][0]) == 2

    def test_get_current_weights(self):
        """get_current_weights should return list of floats."""
        from libraries.dl.merging import LoRAFlow
        flow = LoRAFlow(num_adapters=4, gate_input_dim=8)
        x = torch.randn(10, 8)
        weights = flow.get_current_weights(x)
        assert len(weights) == 4
        assert abs(sum(weights) - 1.0) < 1e-4  # sum to 1


class TestMergingEndToEnd:
    def test_merge_finetuned_mlps(self):
        """End-to-end: fine-tune two MLPs, merge, verify merged works."""
        from libraries.dl.merging import (
            compute_task_vector, task_arithmetic_merge, linear_merge
        )
        from libraries.dl.train import train_epoch, evaluate

        torch.manual_seed(42)
        # Base model
        base = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 4))

        # Fine-tune model A on one distribution
        model_a = copy.deepcopy(base)
        X_a = torch.randn(200, 10)
        y_a = torch.randint(0, 4, (200,))
        loader_a = DataLoader(TensorDataset(X_a, y_a), batch_size=32, shuffle=True)
        opt_a = torch.optim.Adam(model_a.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        for _ in range(5):
            train_epoch(model_a, loader_a, criterion, opt_a)

        # Fine-tune model B on another distribution
        model_b = copy.deepcopy(base)
        X_b = torch.randn(200, 10) + 1.0
        y_b = torch.randint(0, 4, (200,))
        loader_b = DataLoader(TensorDataset(X_b, y_b), batch_size=32, shuffle=True)
        opt_b = torch.optim.Adam(model_b.parameters(), lr=0.01)
        for _ in range(5):
            train_epoch(model_b, loader_b, criterion, opt_b)

        # Task arithmetic merge
        tv_a = compute_task_vector(base.state_dict(), model_a.state_dict())
        tv_b = compute_task_vector(base.state_dict(), model_b.state_dict())
        merged_sd = task_arithmetic_merge(
            base.state_dict(), [tv_a, tv_b], scalings=[0.5, 0.5])

        merged_model = copy.deepcopy(base)
        merged_model.load_state_dict(merged_sd)

        # Merged model should produce valid outputs
        result_a = evaluate(merged_model, loader_a, criterion)
        result_b = evaluate(merged_model, loader_b, criterion)
        assert result_a["loss"] > 0  # finite loss
        assert result_b["loss"] > 0
        assert 0 <= result_a["accuracy"] <= 1
        assert 0 <= result_b["accuracy"] <= 1

    def test_lora_soup_injection(self, small_mlp):
        """Merge LoRA adapters from two tasks and load into model."""
        from libraries.dl.lora import LoRAInjector
        from libraries.dl.merging import merge_lora_adapters

        # Fine-tune with LoRA on two "tasks"
        model_a = copy.deepcopy(small_mlp)
        LoRAInjector.inject(model_a, rank=4)
        # Simulate training by modifying LoRA weights
        with torch.no_grad():
            for m in model_a.modules():
                if hasattr(m, 'lora_B'):
                    m.lora_B.weight.fill_(0.5)

        model_b = copy.deepcopy(small_mlp)
        LoRAInjector.inject(model_b, rank=4)
        with torch.no_grad():
            for m in model_b.modules():
                if hasattr(m, 'lora_B'):
                    m.lora_B.weight.fill_(1.5)

        # Merge LoRA adapters
        sd_a = LoRAInjector.lora_state_dict(model_a)
        sd_b = LoRAInjector.lora_state_dict(model_b)
        merged_lora = merge_lora_adapters([sd_a, sd_b])

        # Verify merged values are averages
        for key in merged_lora:
            if "lora_B" in key:
                assert torch.allclose(
                    merged_lora[key], torch.ones_like(merged_lora[key]) * 1.0,
                    atol=1e-5
                ), f"Expected avg of 0.5 and 1.5 = 1.0 for {key}"


# ---------- Package Integration ----------

class TestDLPackage:
    def test_all_dl_exports(self):
        """All DL exports should be accessible from the main package."""
        import libraries
        expected = [
            "LoRALinear", "LoRAInjector",
            "BaseModel", "TransferScheduler", "build_discriminative_lr_groups",
            "compute_fisher_diagonal", "EWCLoss", "online_ewc_update",
            "compute_cka", "extract_representations",
            "compute_representation_mmd", "NegativeTransferMonitor",
            "linear_merge", "slerp_merge",
            "compute_task_vector", "apply_task_vector",
            "task_arithmetic_merge", "ties_merge", "dare_merge",
            "merge_lora_adapters", "LoRAFlow", "train_lora_flow",
            "task_vector_stats", "task_vector_similarity",
            "GPUCarbonTracker",
            "train_epoch", "evaluate", "fine_tune",
        ]
        for name in expected:
            assert hasattr(libraries, name), f"Missing DL export: {name}"

    def test_version_bump(self):
        """Version should be bumped to 0.5.0."""
        import libraries
        assert libraries.__version__ == "0.5.0"
