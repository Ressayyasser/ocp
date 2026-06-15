"""
reward_engine.py — Detailed reward breakdown + economic valuation.
"""

from __future__ import annotations
import numpy as np
from rl.environment import ACTIONS

# State indices
_PROD, _BILAN, _SHP, _SMP, _PRES, _VIB, _ANOM, _EFF, _MON = range(9)

WEIGHTS = dict(bilan=1.0, efficiency=0.5, action_cost=-0.3,
               vibration=-0.4, anomaly=-1.0, wear=-0.2)

PRICE_PER_MWH_DH = 700.0      # Moroccan DH per MWh (adjust to OCP tariff)
MWH_SCALE        = 150_000.0  # normalised unit → MWh (calibrate from data)


def breakdown(prev: np.ndarray, nxt: np.ndarray, action: int,
              wear: dict | None = None) -> dict:
    """Full reward component breakdown."""
    wear = wear or {}
    d_bilan = float(nxt[_BILAN] - prev[_BILAN])
    d_eff   = float(nxt[_EFF]   - prev[_EFF])
    cost    = ACTIONS[action]["cost"]
    vib_pen = float(max(0, nxt[_VIB] - 0.5))
    anom_pen= float(max(0, -nxt[_ANOM]))
    wear_pen= float(sum(wear.values()))

    components = {
        "bilan_reward":      WEIGHTS["bilan"]       * d_bilan,
        "efficiency_reward": WEIGHTS["efficiency"]  * d_eff,
        "action_cost":       WEIGHTS["action_cost"] * cost,
        "vibration_penalty": WEIGHTS["vibration"]   * vib_pen,
        "anomaly_penalty":   WEIGHTS["anomaly"]     * anom_pen,
        "wear_penalty":      WEIGHTS["wear"]        * wear_pen,
    }
    total = sum(components.values())
    return {
        "total":      round(total, 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "raw": {"delta_bilan": d_bilan, "delta_eff": d_eff,
                "cost": cost, "vib": vib_pen, "anom": anom_pen},
    }


def to_economic(delta_bilan_normalised: float, action_cost_normalised: float) -> dict:
    """Convert normalised reward units to Dirhams."""
    mwh       = delta_bilan_normalised * MWH_SCALE
    gain_dh   = mwh * PRICE_PER_MWH_DH
    opex_dh   = action_cost_normalised * 50_000
    return {
        "delta_mwh":       round(mwh,      2),
        "energy_gain_dh":  round(gain_dh,  2),
        "action_cost_dh":  round(opex_dh,  2),
        "net_gain_dh":     round(gain_dh - opex_dh, 2),
    }