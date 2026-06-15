"""
generators/production.py — Simulates GTA production values second by second.
"""

from __future__ import annotations
import numpy as np


class ProductionGenerator:
    """
    Simulates realistic production values for GTA1–GTAB.
    Uses a base setpoint + daily cycle + random walk + optional fault injection.
    """

    # Nominal daily energy (MWh/day) — derived from 2022–2025 dataset means
    # GTA1: ~17000 MWh/month → ~565/day; GTA2 ~15000→500; GTA3 ~16000→533
    # Using monthly means from Excel; gtaa/gtab not present in this dataset
    _SETPOINTS = {"gta1": 565, "gta2": 480, "gta3": 540,
                  "gtaa": 0,   "gtab": 0}

    def __init__(self, noise_std: float = 15.0, rng_seed: int | None = None):
        self.rng       = np.random.default_rng(rng_seed)
        self.noise_std = noise_std
        self._state    = {k: float(v) for k, v in self._SETPOINTS.items()}
        self._fault: dict[str, float] = {}   # unit → degradation factor 0–1

    def step(self, hour_of_day: int = 12) -> dict:
        """Return one second of production readings."""
        out = {}
        daily_factor = 1.0 + 0.05 * np.sin(2 * np.pi * hour_of_day / 24)

        for unit, setpoint in self._SETPOINTS.items():
            noise     = self.rng.normal(0, self.noise_std)
            fault_deg = self._fault.get(unit, 0.0)
            val       = setpoint * daily_factor * (1 - fault_deg) + noise
            self._state[unit] = max(0.0, val)
            out[unit] = round(float(self._state[unit]), 2)

        out["production"] = round(sum(out.values()), 2)
        return out

    def inject_fault(self, unit: str, degradation: float = 0.15):
        """Simulate partial failure: degradation in [0, 1]."""
        self._fault[unit] = max(0.0, min(1.0, degradation))

    def clear_fault(self, unit: str):
        self._fault.pop(unit, None)