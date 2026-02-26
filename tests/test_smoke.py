"""
Smoke tests for libraries.

Quick verification that all core functionality works end-to-end.
Run with: cd content && python -m pytest tests/test_smoke.py -v
"""

import numpy as np
import torch
import pytest

# ---------- Fixtures ----------

@pytest.fixture
def linear_data():
    """Simple synthetic linear regression problem."""
    rng = np.random.RandomState(42)
    d = 5
    n = 200
    w_true = rng.randn(d).astype(np.float32)
    X = rng.randn(n, d).astype(np.float32)
    y = (X @ w_true + 0.1 * rng.randn(n)).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y), d


@pytest.fixture
def logistic_data():
    """Simple synthetic binary classification problem."""
    rng = np.random.RandomState(42)
    d = 5
    n = 200
    w_true = rng.randn(d).astype(np.float32)
    X = rng.randn(n, d).astype(np.float32)
    logits = X @ w_true
    y = (logits > 0).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y), d


@pytest.fixture
def source_target_data():
    """Source and target domains with covariate shift."""
    rng = np.random.RandomState(42)
    d = 5
    w_true = rng.randn(d).astype(np.float32)
    # Source: mean=+1
    X_s = (rng.randn(300, d) + 1).astype(np.float32)
    y_s = (X_s @ w_true + 0.1 * rng.randn(300)).astype(np.float32)
    # Target: mean=-1
    X_t = (rng.randn(100, d) - 1).astype(np.float32)
    y_t = (X_t @ w_true + 0.1 * rng.randn(100)).astype(np.float32)
    return (torch.from_numpy(X_s), torch.from_numpy(y_s),
            torch.from_numpy(X_t), torch.from_numpy(y_t), d)


# ---------- Core Training ----------

class TestTrainCore:
    def test_linear_sgd_converges(self, linear_data):
        from libraries.train_core import fit_linear_sgd
        from libraries.metrics import r2_score
        X, y, d = linear_data
        w, b = fit_linear_sgd(X, y, torch.zeros(d), torch.zeros(1),
                              epochs=50, lr=0.01)
        yhat = X @ w + b
        r2 = r2_score(yhat, y)
        assert r2 > 0.9, f"Linear SGD R²={r2:.4f}, expected > 0.9"

    def test_logistic_sgd_converges(self, logistic_data):
        from libraries.train_core import fit_logistic_sgd
        from libraries.metrics import accuracy_from_logits
        X, y, d = logistic_data
        w, b = fit_logistic_sgd(X, y, torch.zeros(d), torch.zeros(1),
                                epochs=50, lr=0.1)
        logits = X @ w + b
        acc = accuracy_from_logits(logits, y)
        assert acc > 0.85, f"Logistic SGD acc={acc:.4f}, expected > 0.85"

    def test_warm_start_beats_cold(self, linear_data):
        """Transfer learning via warm-start should converge faster."""
        from libraries.train_core import fit_linear_sgd
        from libraries.metrics import mse
        X, y, d = linear_data
        # Full training
        w_full, b_full = fit_linear_sgd(X, y, torch.zeros(d), torch.zeros(1),
                                        epochs=50, lr=0.01)
        # Cold start with few epochs
        w_cold, b_cold = fit_linear_sgd(X, y, torch.zeros(d), torch.zeros(1),
                                        epochs=5, lr=0.01)
        # Warm start from full solution with few epochs
        w_warm, b_warm = fit_linear_sgd(X, y, w_full.clone(), b_full.clone(),
                                        epochs=5, lr=0.01)
        mse_cold = mse(X @ w_cold + b_cold, y)
        mse_warm = mse(X @ w_warm + b_warm, y)
        assert mse_warm < mse_cold, "Warm start should beat cold start in 5 epochs"


# ---------- Transfer Methods ----------

class TestTransfer:
    def test_regularized_linear(self, source_target_data):
        from libraries.train_core import fit_linear_sgd
        from libraries.transfer import regularized_transfer_linear
        from libraries.metrics import r2_score
        X_s, y_s, X_t, y_t, d = source_target_data
        w_src, b_src = fit_linear_sgd(X_s, y_s, torch.zeros(d), torch.zeros(1),
                                      epochs=50, lr=0.01)
        w, b = regularized_transfer_linear(X_t, y_t, w_src, b_src, lam=1.0)
        r2 = r2_score(X_t @ w + b, y_t)
        assert r2 > 0.8, f"Regularized transfer R²={r2:.4f}, expected > 0.8"

    def test_bayesian_linear(self, source_target_data):
        from libraries.train_core import fit_linear_sgd
        from libraries.transfer import bayesian_transfer_linear
        from libraries.metrics import r2_score
        X_s, y_s, X_t, y_t, d = source_target_data
        w_src, b_src = fit_linear_sgd(X_s, y_s, torch.zeros(d), torch.zeros(1),
                                      epochs=50, lr=0.01)
        w, b = bayesian_transfer_linear(X_t, y_t, w_src, b_src)
        r2 = r2_score(X_t @ w + b, y_t)
        assert r2 > 0.8, f"Bayesian transfer R²={r2:.4f}, expected > 0.8"

    def test_covariance_linear(self, source_target_data):
        from libraries.transfer import covariance_transfer_linear
        from libraries.metrics import r2_score
        X_s, y_s, X_t, y_t, d = source_target_data
        w, b = covariance_transfer_linear(X_s, y_s, X_t, y_t)
        r2 = r2_score(X_t @ w + b, y_t)
        assert r2 > 0.5, f"Covariance transfer R²={r2:.4f}, expected > 0.5"

    def test_regularized_logistic(self):
        from libraries.train_core import fit_logistic_sgd
        from libraries.transfer import regularized_transfer_logistic
        from libraries.metrics import accuracy_from_logits
        rng = np.random.RandomState(42)
        d = 5
        w_true = rng.randn(d).astype(np.float32)
        X_s = rng.randn(200, d).astype(np.float32)
        y_s = (X_s @ w_true > 0).astype(np.float32)
        X_t = (rng.randn(80, d) + 0.5).astype(np.float32)
        y_t = (X_t @ w_true > 0).astype(np.float32)

        X_s_t, y_s_t = torch.from_numpy(X_s), torch.from_numpy(y_s)
        X_t_t, y_t_t = torch.from_numpy(X_t), torch.from_numpy(y_t)

        w_src, b_src = fit_logistic_sgd(X_s_t, y_s_t, torch.zeros(d),
                                        torch.zeros(1), epochs=50, lr=0.1)
        w, b = regularized_transfer_logistic(X_t_t, y_t_t, w_src, b_src,
                                             epochs=25, lr=0.01)
        acc = accuracy_from_logits(X_t_t @ w + b, y_t_t)
        assert acc > 0.8, f"Regularized logistic acc={acc:.4f}, expected > 0.8"


# ---------- LoRA Adapters ----------

class TestLoRA:
    def test_vector_delta_w_init_zero(self):
        from libraries.adapters import LoRAAdapterVector
        adapter = LoRAAdapterVector(d=10, r=3)
        dw = adapter.delta_w()
        # B=0, so ΔW = B@a = 0 at init (Hu et al. 2021)
        assert torch.allclose(dw, torch.zeros(10), atol=1e-6), \
            "LoRA delta_w should be zero at initialization"

    def test_matrix_delta_W_init_zero(self):
        from libraries.adapters import LoRAAdapterMatrix
        adapter = LoRAAdapterMatrix(d=20, k=5, r=3)
        dW = adapter.delta_W()
        assert torch.allclose(dW, torch.zeros(20, 5), atol=1e-6), \
            "LoRA delta_W should be zero at initialization"

    def test_matrix_param_reduction(self):
        from libraries.adapters import LoRAAdapterMatrix
        adapter = LoRAAdapterMatrix(d=1000, k=50, r=5)
        assert adapter.reduction_ratio() > 5.0, \
            f"Expected >5x reduction, got {adapter.reduction_ratio():.1f}x"

    def test_lora_trainable(self, linear_data):
        """LoRA adapters can be trained via backprop."""
        from libraries.adapters import LoRAAdapterVector
        X, y, d = linear_data
        w_base = torch.randn(d)
        b_base = torch.zeros(1)
        adapter = LoRAAdapterVector(d=d, r=2)
        opt = torch.optim.SGD(adapter.parameters(), lr=0.01)
        initial_loss = torch.mean((X @ (w_base + adapter.delta_w()) + b_base - y) ** 2).item()
        for _ in range(50):
            opt.zero_grad()
            yhat = X @ (w_base + adapter.delta_w()) + (b_base + adapter.delta_b())
            loss = torch.mean((yhat - y) ** 2)
            loss.backward()
            opt.step()
        final_loss = loss.item()
        assert final_loss < initial_loss, "LoRA training should reduce loss"


# ---------- Negative Transfer Detection ----------

class TestNegativeTransfer:
    def test_mmd_similar_distributions(self):
        from libraries.negative_transfer import compute_mmd
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)
        # Same distribution, split in half
        mmd = compute_mmd(X[:100], X[100:])
        assert mmd < 0.1, f"MMD between same distribution should be small, got {mmd:.4f}"

    def test_mmd_different_distributions(self):
        from libraries.negative_transfer import compute_mmd
        rng = np.random.RandomState(42)
        X_s = rng.randn(200, 5).astype(np.float32)
        X_t = (rng.randn(200, 5) + 3).astype(np.float32)  # shifted
        mmd = compute_mmd(X_s, X_t)
        assert mmd > 0.1, f"MMD between shifted distributions should be large, got {mmd:.4f}"

    def test_should_transfer_safe(self):
        from libraries.negative_transfer import should_transfer
        rng = np.random.RandomState(42)
        X = rng.randn(400, 5).astype(np.float32)
        result = should_transfer(X[:200], X[200:])
        assert result["recommend"] is True, "Same-distribution transfer should be recommended"

    def test_ks_feature_test(self):
        from libraries.negative_transfer import ks_feature_test
        rng = np.random.RandomState(42)
        X_s = rng.randn(200, 5).astype(np.float32)
        X_t = (rng.randn(200, 5) + 3).astype(np.float32)
        result = ks_feature_test(X_s, X_t)
        assert result["fraction_shifted"] > 0.5, \
            "Shifted distributions should have many shifted features"


# ---------- Stat Mapping ----------

class TestStatMapping:
    def test_moment_init_linear(self, linear_data):
        from libraries.stat_mapping import moment_init_linear
        from libraries.metrics import r2_score
        X, y, d = linear_data
        X_np, y_np = X.numpy(), y.numpy()
        w, b = moment_init_linear(X_np, y_np)
        yhat = torch.from_numpy(X_np @ w + b)
        r2 = r2_score(yhat, y)
        assert r2 > 0.5, f"Moment init R²={r2:.4f}, should give reasonable init"

    def test_moment_init_logistic(self, logistic_data):
        from libraries.stat_mapping import moment_init_logistic
        X, y, d = logistic_data
        X_np, y_np = X.numpy(), y.numpy()
        w, b = moment_init_logistic(X_np, y_np)
        assert w.shape == (d,), f"Expected shape ({d},), got {w.shape}"
        assert isinstance(b, np.float32), f"Expected float32 bias, got {type(b)}"


# ---------- Carbon Tracker ----------

class TestCarbonTracker:
    def test_manual_tracking(self):
        from libraries.carbon import CarbonTracker
        tracker = CarbonTracker("test", power_watts=30, use_codecarbon=False)
        tracker.start()
        # Simulate work
        _ = sum(range(10000))
        result = tracker.stop()
        assert result["method"] == "test"
        assert result["time_s"] > 0
        assert result["co2_kg"] >= 0
        assert result["source"] == "manual"

    def test_compare_emissions(self):
        from libraries.carbon import compare_emissions
        results = [
            {"method": "scratch", "co2_kg": 0.01, "time_s": 10.0},
            {"method": "transfer", "co2_kg": 0.002, "time_s": 2.0},
        ]
        summary = compare_emissions(results)
        assert summary["baseline"] == "scratch"
        assert len(summary["comparisons"]) == 1
        assert summary["comparisons"][0]["co2_saved_pct"] == pytest.approx(80.0)


# ---------- Metrics ----------

class TestMetrics:
    def test_mse(self):
        from libraries.metrics import mse
        y = torch.tensor([1.0, 2.0, 3.0])
        yhat = torch.tensor([1.0, 2.0, 3.0])
        assert mse(yhat, y) == pytest.approx(0.0, abs=1e-6)

    def test_r2_perfect(self):
        from libraries.metrics import r2_score
        y = torch.tensor([1.0, 2.0, 3.0])
        assert r2_score(y, y) == pytest.approx(1.0, abs=1e-4)

    def test_r2_mean_predictor(self):
        from libraries.metrics import r2_score
        y = torch.tensor([1.0, 2.0, 3.0])
        yhat = torch.tensor([2.0, 2.0, 2.0])  # predicting mean
        assert abs(r2_score(yhat, y)) < 0.01

    def test_accuracy_from_logits(self):
        from libraries.metrics import accuracy_from_logits
        logits = torch.tensor([2.0, -2.0, 3.0, -1.0])
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert accuracy_from_logits(logits, y) == pytest.approx(1.0)

    def test_set_seed_reproducibility(self):
        from libraries.metrics import set_seed
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.allclose(a, b), "set_seed should give reproducible results"


# ---------- Package Imports ----------

class TestPackage:
    def test_version(self):
        import libraries
        assert hasattr(libraries, "__version__")

    def test_all_exports(self):
        """Verify all documented exports are accessible."""
        import libraries
        expected = [
            "fit_linear_sgd", "fit_logistic_sgd", "eval_linear", "eval_logistic",
            "regularized_transfer_linear", "regularized_transfer_logistic",
            "bayesian_transfer_linear", "bayesian_transfer_logistic",
            "bayesian_posterior_precision", "covariance_transfer_linear",
            "LoRAAdapterVector", "LoRAAdapterMatrix",
            "compute_mmd", "compute_proxy_a_distance", "ks_feature_test",
            "should_transfer", "validate_transfer",
            "moment_init_linear", "moment_init_logistic",
            "mse", "r2_score", "accuracy_from_logits", "estimate_energy_and_co2",
            "CarbonTracker", "compare_emissions", "set_seed",
        ]
        for name in expected:
            assert hasattr(libraries, name), f"Missing export: {name}"
