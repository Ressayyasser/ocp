"""
simulator.py — Digital Twin: what-if scenario simulation.

Usage
─────
sim = DigitalTwinSimulator(predictor_service)
result = sim.simulate("gta3", change_percent=+15, duration_hours=24)
# → {"predicted_bilan_change": +12500, "predicted_production_change": ...}
"""

from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline.feature_engineering import add_all_features
from data_pipeline.preprocessing       import clean_dataframe
from database.database                 import query


# Sensitivity map: which state variables are affected by changing a given variable
_SENSITIVITY: dict[str, dict[str, float]] = {
    "gta1":       {"production": 0.25, "bilan_net": 0.25, "vibration": 0.05, "steam_hp": 0.10},
    "gta2":       {"production": 0.25, "bilan_net": 0.25, "vibration": 0.05, "steam_hp": 0.10},
    "gta3":       {"production": 0.25, "bilan_net": 0.25, "vibration": 0.05, "steam_hp": 0.10},
    "gtaa":       {"production": 0.20, "bilan_net": 0.20, "steam_mp": 0.08},
    "gtab":       {"production": 0.20, "bilan_net": 0.20, "steam_mp": 0.08},
    "steam_hp":   {"efficiency": 0.30, "production": 0.15, "bilan_net": 0.12},
    "steam_mp":   {"efficiency": 0.20, "production": 0.10},
    "pressure":   {"production": 0.10, "vibration": -0.05, "efficiency": 0.08},
    "vibration":  {"production": -0.10, "bilan_net": -0.08, "efficiency": -0.12},
    "temperature":{"efficiency": 0.05, "vibration": 0.03},
}

_RISK_WEIGHTS: dict[str, float] = {
    "vibration":   0.4,
    "temperature": 0.2,
    "pressure":    0.2,
    "production":  0.1,
}


class DigitalTwinSimulator:

    def __init__(self):
        self._df: pd.DataFrame | None = None

    def _get_baseline(self) -> pd.DataFrame:
        rows = query("SELECT * FROM historical_data ORDER BY timestamp DESC LIMIT 720")
        if not rows:
            raise ValueError("No historical data for simulation baseline")
        df = pd.DataFrame(rows)
        df = clean_dataframe(df)
        df = add_all_features(df)
        self._df = df
        return df

    # ── Single scenario ───────────────────────────────────────────────────────

    def simulate(self, variable: str, change_percent: float,
                 duration_hours: int = 24) -> dict:
        """
        Apply *change_percent* to *variable* and propagate effects
        through the sensitivity map.
        """
        df       = self._get_baseline()
        baseline = df.iloc[-duration_hours:].mean(numeric_only=True)

        sens = _SENSITIVITY.get(variable, {variable: 1.0})
        delta_frac = change_percent / 100.0

        changes: dict[str, float] = {}
        for affected, coeff in sens.items():
            if affected in baseline:
                base_val       = float(baseline[affected])
                changes[affected] = base_val * delta_frac * coeff

        bilan_ch = changes.get("bilan_net",    0.0)
        prod_ch  = changes.get("production",   0.0)
        eff_ch   = changes.get("efficiency",   0.0)

        # Risk score: weighted impact on hazardous variables
        risk = 0.0
        for var, w in _RISK_WEIGHTS.items():
            if var in changes:
                risk += abs(changes[var] / (float(baseline.get(var, 1)) or 1)) * w
        risk = min(1.0, risk)

        # Simple recommendation text
        if bilan_ch > 0:
            reco = f"Increasing {variable} by {change_percent:.0f}% is beneficial — proceed."
        elif risk > 0.5:
            reco = f"Caution: increasing {variable} by {change_percent:.0f}% raises risk score to {risk:.2f}."
        else:
            reco = f"Neutral or marginal impact. Monitor {variable} after change."

        return {
            "scenario":                    f"{variable} {change_percent:+.0f}%",
            "variable":                    variable,
            "change_percent":              change_percent,
            "duration_hours":             duration_hours,
            "predicted_bilan_change":      round(bilan_ch, 2),
            "predicted_production_change": round(prod_ch,  2),
            "predicted_efficiency_change": round(eff_ch,   2),
            "all_changes":                 {k: round(v, 2) for k, v in changes.items()},
            "risk_score":                  round(risk, 3),
            "recommendation":              reco,
        }

    # ── Multi-scenario comparison ─────────────────────────────────────────────

    def compare_scenarios(self, scenarios: list[dict]) -> list[dict]:
        """
        scenarios: [{"variable": "gta3", "change_percent": 15}, ...]
        """
        return [self.simulate(**s) for s in scenarios]