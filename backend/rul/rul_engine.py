"""
rul_engine.py — Remaining Useful Life (RUL) estimation (Phase 3 module).

Components tracked
──────────────────
Turbine_HP | Rotor | Bearings | Valves | Condenser | Generator

Health score: 0–100  (100 = new, 0 = failed)
RUL:          days

Usage
─────
engine = RULEngine()
health = engine.compute_health(reading_dict)
# → {"Rotor": 82, "Condenser": 75, "Generator": 91, ...}
rul    = engine.estimate_rul(health)
# → {"Rotor": {"rul_days": 142}, ...}
"""

from __future__ import annotations
import numpy as np
from datetime import datetime

# Degradation coefficients per component per input signal
# Each entry: (signal_name, direction, weight)
#   direction: "high" = high value degrades, "low" = low value degrades
_COMPONENT_SENSITIVITY: dict[str, list[tuple[str, str, float]]] = {
    "Turbine_HP": [
        ("vibration",    "high", 0.35),
        ("temperature",  "high", 0.25),
        ("pressure",     "low",  0.20),
        ("efficiency",   "low",  0.20),
    ],
    "Rotor": [
        ("vibration",    "high", 0.45),
        ("temperature",  "high", 0.20),
        ("pressure",     "low",  0.15),
        ("efficiency",   "low",  0.20),
    ],
    "Bearings": [
        ("vibration",    "high", 0.50),
        ("temperature",  "high", 0.30),
        ("pressure",     "low",  0.20),
    ],
    "Valves": [
        ("pressure",     "low",  0.40),
        ("temperature",  "high", 0.30),
        ("efficiency",   "low",  0.30),
    ],
    "Condenser": [
        ("temperature",  "high", 0.40),
        ("steam_hp",     "low",  0.30),
        ("efficiency",   "low",  0.30),
    ],
    "Generator": [
        ("vibration",    "high", 0.30),
        ("temperature",  "high", 0.35),
        ("efficiency",   "low",  0.35),
    ],
}

# Nominal (healthy) sensor values and their full-degradation thresholds
_NOMINAL: dict[str, float]   = {
    # Derived from 2022–2025 GTA dataset means
    "vibration":           1.3,    # SCADA-generated (not in Excel)
    "temperature":         455.0,  # avg admission temp °C
    "pressure":            54.5,   # avg admission pressure bar
    "efficiency":          0.405,  # avg rendement fraction (40.5%)
    "steam_hp":            530.0,  # total HP admission flow t/h (3 GTAs)
    "rendement_gta1":      41.5,   # %
    "rendement_gta2":      35.0,   # % (lower design)
    "rendement_gta3":      40.5,   # %
    "debit_adm_gta1":      175.0,  # t/h
    "debit_adm_gta2":      175.0,  # t/h
    "debit_adm_gta3":      175.0,  # t/h
}
_CRITICAL: dict[str, float]  = {
    "vibration":           7.0,
    "temperature":         490.0,  # overtemperature critical
    "pressure":            46.0,   # low pressure critical
    "efficiency":          0.33,   # below 33% critical
    "steam_hp":            280.0,  # low HP flow critical
    "rendement_gta1":      36.0,
    "rendement_gta2":      28.0,
    "rendement_gta3":      36.0,
    "debit_adm_gta1":      100.0,
    "debit_adm_gta2":      100.0,
    "debit_adm_gta3":      100.0,
}

# Mean time to failure at 0 health (in days) per component
_MTTF_DAYS: dict[str, float] = {
    "Turbine_HP": 365, "Rotor": 300, "Bearings": 180,
    "Valves": 120,     "Condenser": 240, "Generator": 400,
}


class RULEngine:

    def __init__(self):
        self._cumulative_stress: dict[str, float] = {c: 0.0 for c in _COMPONENT_SENSITIVITY}
        self._hours_operated: float = 0.0

    def compute_health(self, reading: dict,
                       hours_operated: float = 0.0) -> dict[str, float]:
        """
        Compute health score [0–100] for each component from a sensor reading.
        """
        self._hours_operated = hours_operated
        health: dict[str, float] = {}

        for component, signals in _COMPONENT_SENSITIVITY.items():
            stress = 0.0
            for signal, direction, weight in signals:
                val     = reading.get(signal)
                if val is None:
                    continue
                nominal  = _NOMINAL.get(signal, 1.0)
                critical = _CRITICAL.get(signal, nominal * 2)

                if direction == "high":
                    # stress increases as value approaches critical from below
                    s = (val - nominal) / (critical - nominal)
                elif direction == "low":
                    # stress increases as value drops toward critical from above
                    s = (nominal - val) / (nominal - critical)
                else:
                    s = 0.0

                stress += max(0.0, min(1.0, s)) * weight

            # age penalty: 0.1% per 1000 hours
            age_factor = min(0.30, hours_operated / 1_000_000)
            raw_health = 100.0 * (1.0 - min(1.0, stress + age_factor))
            health[component] = round(float(raw_health), 1)

        return health

    def estimate_rul(self, health: dict[str, float]) -> dict[str, dict]:
        """
        Estimate RUL in days from health scores.
        Uses a linear degradation model: rul ≈ mttf × health / 100.
        """
        rul: dict[str, dict] = {}
        for component, score in health.items():
            mttf    = _MTTF_DAYS.get(component, 200)
            rul_days = mttf * (score / 100.0)
            status  = ("critical" if score < 30 else
                       "warning"  if score < 60 else "healthy")
            rul[component] = {
                "health_score": score,
                "rul_days":     round(rul_days, 1),
                "status":       status,
                "unit":         "days",
            }
        return rul

    def update_state_with_rul(self, state: np.ndarray,
                               health: dict[str, float]) -> np.ndarray:
        """
        Append health scores to RL state vector (Phase 3 integration).
        Expands state from dim-9 to dim-9+N_components.
        """
        extra = np.array([health.get(c, 100.0) / 100.0
                          for c in _COMPONENT_SENSITIVITY], dtype=np.float32)
        return np.concatenate([state, extra])

    def get_summary(self, reading: dict,
                    hours_operated: float = 0.0) -> dict:
        """One-call convenience: health + RUL + timestamp."""
        health = self.compute_health(reading, hours_operated)
        rul    = self.estimate_rul(health)
        return {
            "timestamp":      datetime.now().isoformat(),
            "health_scores":  health,
            "rul":            rul,
            "most_critical":  min(health, key=health.get),
        }