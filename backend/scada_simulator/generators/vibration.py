"""
generators/vibration.py — Simulates rotor vibration (mm/s RMS).
"""

from __future__ import annotations
import numpy as np


class VibrationGenerator:
    _NOMINAL: dict[str, float] = {"gta1": 1.2, "gta2": 1.4, "gta3": 1.1}

    def __init__(self, noise_std: float = 0.05, rng_seed: int | None = None):
        self.rng       = np.random.default_rng(rng_seed)
        self.noise_std = noise_std
        self._state    = dict(self._NOMINAL)
        self._fault: dict[str, float] = {}   # unit → extra vibration level

    def step(self) -> dict:
        out = {}
        for unit, nominal in self._NOMINAL.items():
            extra = self._fault.get(unit, 0.0)
            noise = self.rng.normal(0, self.noise_std)
            val   = max(0.1, nominal + extra + noise)
            self._state[unit] = val
            out[f"vibration_{unit}"] = round(float(val), 4)

        out["vibration"] = max(out.values())   # worst-case canonical alias
        return out

    def inject_vibration(self, unit: str, extra_mm_s: float = 3.5):
        """Inject an elevated vibration level on *unit*."""
        self._fault[unit] = extra_mm_s

    def clear_fault(self, unit: str):
        self._fault.pop(unit, None)