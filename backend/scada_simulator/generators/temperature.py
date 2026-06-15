"""
generators/temperature.py — Simulates exhaust gas temperature (°C).
"""

from __future__ import annotations
import numpy as np


class TemperatureGenerator:
    # Real nominal: T° adm GTA1~455°C, GTA2~435°C, GTA3~456°C (from dataset means)
    _NOMINAL = {"gta1": 455.0, "gta2": 435.0, "gta3": 456.0}

    def __init__(self, noise_std: float = 1.5, rng_seed: int | None = None):
        self.rng        = np.random.default_rng(rng_seed)
        self.noise_std  = noise_std
        self._state     = dict(self._NOMINAL)
        self._overtemp: dict[str, float] = {}

    def step(self) -> dict:
        out = {}
        for unit, nominal in self._NOMINAL.items():
            extra = self._overtemp.get(unit, 0.0)
            noise = self.rng.normal(0, self.noise_std)
            self._state[unit] = max(0.0, nominal + extra + noise)
            out[f"temperature_{unit}"] = round(float(self._state[unit]), 2)

        out["temperature"] = float(np.mean(list(out.values())))
        return out

    def inject_overtemperature(self, unit: str, delta: float = 30.0):
        self._overtemp[unit] = delta

    def clear_fault(self, unit: str):
        self._overtemp.pop(unit, None)