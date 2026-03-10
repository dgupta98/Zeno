"""
GPU-aware carbon emissions tracking for deep learning workloads.

Extends Zeno's CarbonTracker to GPU workloads via NVIDIA Management
Library (NVML).  The core formula remains identical:

    CO2 (g) = Energy (kWh) * PUE * Carbon Intensity (gCO2/kWh)

What changes is the energy measurement method:
  - Energy API (Volta+): cumulative millijoules, start/end subtraction
  - Power API (all architectures): polling thread + trapezoidal integration
  - CPU fallback: manual estimation (same as classical CarbonTracker)

Critical detail: always call torch.cuda.synchronize() before reading
energy, since GPU operations are asynchronous and NVML runs on CPU.

Published benchmarks:
  BERT fine-tuning (SST-2):  0.1-2 kWh, 40-800g CO2
  LoRA fine-tuning 7B:       0.694 kWh, 57g CO2 (Danish grid)
  BLOOM 176B pre-training:   433,195 kWh, 24.69 tonnes CO2
"""

import time
import threading
import torch


class GPUCarbonTracker:
    """
    Track GPU energy consumption and CO2 emissions.

    Follows the same start()/stop() protocol as Zeno's CarbonTracker
    for compatibility with compare_emissions().

    Measurement priority:
    1. NVML Energy API (Volta+) — most accurate, cumulative millijoules
    2. NVML Power API — polling thread, trapezoidal integration
    3. Manual estimation — CPU TDP-based fallback

    Usage:
        tracker = GPUCarbonTracker(method_name="lora_finetune")
        tracker.start()
        # ... GPU training ...
        result = tracker.stop()
        print(f"CO2: {result['co2_kg']:.6f} kg")
    """

    def __init__(self, method_name="unnamed", power_watts=30.0,
                 carbon_intensity_kg_kwh=0.45, pue=1.0,
                 gpu_index=0, poll_interval_s=1.0):
        """
        Args:
            method_name: label for this measurement
            power_watts: fallback power estimate (CPU mode)
            carbon_intensity_kg_kwh: grid CO2 intensity (default 0.45 = US avg)
                Regional values: France ~0.055, EU avg ~0.213, India ~0.7
            pue: Power Usage Effectiveness (Google Cloud ~1.09, avg DC ~1.58)
            gpu_index: which GPU to monitor (for multi-GPU setups)
            poll_interval_s: power polling interval in seconds
        """
        self.method_name = method_name
        self.power_watts = power_watts
        self.carbon_intensity = carbon_intensity_kg_kwh
        self.pue = pue
        self.gpu_index = gpu_index
        self.poll_interval_s = poll_interval_s

        self._start_time = None
        self._result = None
        self._mode = None  # "energy_api", "power_api", or "manual"

        # NVML state
        self._nvml_handle = None
        self._start_energy_mj = None
        self._poll_thread = None
        self._poll_running = False
        self._power_samples = []  # [(timestamp, watts)]

        # Detect NVML availability
        self._has_nvml = False
        self._has_energy_api = False
        self._init_nvml()

    def _init_nvml(self):
        """Try to initialize NVML for GPU power measurement."""
        if not torch.cuda.is_available():
            return

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.gpu_index
            )
            self._has_nvml = True

            # Check if Energy API is available (Volta and newer)
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml_handle)
                self._has_energy_api = True
            except pynvml.NVMLError:
                self._has_energy_api = False

        except (ImportError, Exception):
            self._has_nvml = False

    def start(self):
        """Begin tracking energy consumption."""
        self._start_time = time.time()

        if self._has_energy_api:
            import pynvml
            torch.cuda.synchronize()
            self._start_energy_mj = (
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml_handle)
            )
            self._mode = "energy_api"

        elif self._has_nvml:
            # Start power polling thread
            self._power_samples = []
            self._poll_running = True
            self._poll_thread = threading.Thread(
                target=self._poll_power, daemon=True
            )
            self._poll_thread.start()
            self._mode = "power_api"

        else:
            self._mode = "manual"

        return self

    def _poll_power(self):
        """Background thread: sample GPU power at regular intervals."""
        import pynvml
        while self._poll_running:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                power_w = power_mw / 1000.0
                self._power_samples.append((time.time(), power_w))
            except Exception:
                pass
            time.sleep(self.poll_interval_s)

    def stop(self):
        """Stop tracking and return results."""
        elapsed = time.time() - self._start_time

        if self._mode == "energy_api":
            import pynvml
            torch.cuda.synchronize()
            end_energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(
                self._nvml_handle
            )
            energy_mj = end_energy_mj - self._start_energy_mj
            energy_kwh = (energy_mj / 1e3) / 3_600_000  # mJ -> J -> kWh
            energy_kwh *= self.pue
            gpu_power_avg = (energy_mj / 1e3) / max(elapsed, 1e-6)

        elif self._mode == "power_api":
            self._poll_running = False
            if self._poll_thread is not None:
                self._poll_thread.join(timeout=2.0)

            # Trapezoidal integration of power samples
            energy_j = 0.0
            samples = self._power_samples
            for i in range(1, len(samples)):
                dt = samples[i][0] - samples[i - 1][0]
                avg_power = (samples[i][1] + samples[i - 1][1]) / 2.0
                energy_j += avg_power * dt

            energy_kwh = (energy_j / 3_600_000) * self.pue
            gpu_power_avg = energy_j / max(elapsed, 1e-6) if elapsed > 0 else 0

        else:
            # Manual CPU-based estimation
            energy_kwh = (self.power_watts * elapsed) / 3_600_000 * self.pue
            gpu_power_avg = self.power_watts

        co2_kg = energy_kwh * self.carbon_intensity

        self._result = {
            "method": self.method_name,
            "time_s": elapsed,
            "kwh": energy_kwh,
            "co2_kg": co2_kg,
            "source": self._mode,
            "power_watts": gpu_power_avg,
            "carbon_intensity": self.carbon_intensity,
        }
        return self._result

    @property
    def result(self):
        return self._result

    def __repr__(self):
        if self._result is None:
            return (f"GPUCarbonTracker(method={self.method_name!r}, "
                    f"mode={self._mode}, not started)")
        r = self._result
        return (f"GPUCarbonTracker({r['method']}: "
                f"{r['time_s']:.4f}s, "
                f"{r['co2_kg']:.2e} kg CO2, "
                f"via {r['source']})")
