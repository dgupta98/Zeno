"""
Carbon emissions tracking for transfer learning experiments.

Provides a unified CarbonTracker that:
  - Uses CodeCarbon when available (hardware-level power measurement)
  - Falls back to manual estimation: CO2(kg) = Power(W) * Time(s) / 3,600,000 * CI
"""

import time


class CarbonTracker:
    """
    Track energy consumption and CO2 emissions for a training run.

    Usage:
        tracker = CarbonTracker(method_name="weight_transfer")
        tracker.start()
        model.fit(X, y)
        result = tracker.stop()
        print(result)  # {'time_s': ..., 'kwh': ..., 'co2_kg': ..., ...}
    """

    def __init__(self, method_name="unnamed", power_watts=30.0,
                 carbon_intensity_kg_kwh=0.45, pue=1.0, use_codecarbon=True):
        """
        Args:
            method_name: label for this measurement
            power_watts: estimated hardware power draw (fallback mode)
            carbon_intensity_kg_kwh: grid CO2 intensity in kg CO2/kWh
                Default 0.45 ≈ US average.  Other values:
                  Arizona: ~0.42, EU avg: 0.213, France: ~0.07
            pue: Power Usage Effectiveness — ratio of total facility power
                to IT equipment power.  Default 1.0 (direct hardware only).
                Typical data center: 1.1–1.6.  Google: ~1.1, Average: ~1.58.
            use_codecarbon: attempt to use CodeCarbon if installed
        """
        self.method_name = method_name
        self.power_watts = power_watts
        self.carbon_intensity = carbon_intensity_kg_kwh
        self.pue = pue
        self.use_codecarbon = use_codecarbon

        self._codecarbon_tracker = None
        self._has_codecarbon = False
        self._start_time = None
        self._result = None

        if use_codecarbon:
            try:
                from codecarbon import EmissionsTracker
                self._has_codecarbon = True
            except ImportError:
                self._has_codecarbon = False

    def start(self):
        """Begin tracking."""
        self._start_time = time.time()

        if self._has_codecarbon:
            from codecarbon import EmissionsTracker
            self._codecarbon_tracker = EmissionsTracker(
                project_name=f"libraries-{self.method_name}",
                measure_power_secs=15,
                log_level="error",
                save_to_file=False,
            )
            self._codecarbon_tracker.start()

        return self

    def stop(self):
        """Stop tracking and return results."""
        elapsed = time.time() - self._start_time

        if self._has_codecarbon and self._codecarbon_tracker is not None:
            emissions_kg = self._codecarbon_tracker.stop()
            # CodeCarbon also gives energy
            energy_kwh = getattr(
                self._codecarbon_tracker, '_total_energy', None
            )
            if energy_kwh is None:
                energy_kwh = (self.power_watts * elapsed) / 3_600_000
            source = "codecarbon"
        else:
            # Manual estimation: CO2 = Power × Time / 3,600,000 × CI × PUE
            energy_kwh = (self.power_watts * elapsed) / 3_600_000 * self.pue
            emissions_kg = energy_kwh * self.carbon_intensity
            source = "manual"

        self._result = {
            "method": self.method_name,
            "time_s": elapsed,
            "kwh": energy_kwh,
            "co2_kg": emissions_kg,
            "source": source,
            "power_watts": self.power_watts,
            "carbon_intensity": self.carbon_intensity,
        }
        return self._result

    @property
    def result(self):
        return self._result

    def __repr__(self):
        if self._result is None:
            return f"CarbonTracker(method={self.method_name!r}, not started)"
        r = self._result
        return (f"CarbonTracker({r['method']}: "
                f"{r['time_s']:.4f}s, "
                f"{r['co2_kg']:.2e} kg CO2, "
                f"via {r['source']})")


def compare_emissions(results):
    """
    Compare CO2 emissions across methods.

    Args:
        results: list of dicts from CarbonTracker.stop()

    Returns:
        summary dict with savings relative to the baseline (first entry)
    """
    if not results:
        return {}

    baseline = results[0]
    summary = {
        "baseline": baseline["method"],
        "baseline_co2_kg": baseline["co2_kg"],
        "comparisons": [],
    }

    for r in results[1:]:
        saved = baseline["co2_kg"] - r["co2_kg"]
        pct = (saved / baseline["co2_kg"] * 100) if baseline["co2_kg"] > 0 else 0
        summary["comparisons"].append({
            "method": r["method"],
            "co2_kg": r["co2_kg"],
            "co2_saved_kg": saved,
            "co2_saved_pct": pct,
            "speedup": baseline["time_s"] / r["time_s"] if r["time_s"] > 0 else float("inf"),
        })

    return summary
