"""
generators/pressure.py — Simulates HP/MP/LP steam pressure (bar).
"""

from __future__ import annotations
import numpy as np


class PressureGenerator:
    # Real nominal values from dataset: P adm ~54.5 bar, P sout ~8.0 bar, P ext ~65 mbar
    _NOMINAL = {"hp": 54.5, "mp": 8.0, "lp": 0.065}

    def __init__(self, noise_std: float = 0.3, rng_seed: int | None = None):
        self.rng        = np.random.default_rng(rng_seed)
        self.noise_std  = noise_std
        self._state     = dict(self._NOMINAL)
        self._drop_mode = False
        self._drop_rate = 0.0

    def step(self) -> dict:
        out = {}
        for level, nominal in self._NOMINAL.items():
            drift = -self._drop_rate if self._drop_mode else 0.0
            noise = self.rng.normal(0, self.noise_std)
            self._state[level] = max(0.1, self._state[level] + drift + noise)
            out[f"pressure_{level}"] = round(float(self._state[level]), 3)

        out["pressure"] = out["pressure_hp"]   # canonical alias
        return out

    def inject_pressure_drop(self, rate: float = 0.05):
        """Start a slow pressure drop at *rate* bar/second."""
        self._drop_mode = True
        self._drop_rate = rate

    def clear_fault(self):
        self._drop_mode = False
        self._drop_rate = 0.0
        self._state     = dict(self._NOMINAL)