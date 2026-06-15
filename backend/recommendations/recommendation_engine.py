"""
recommendation_engine.py — Combines DQN + SHAP to produce actionable recommendations.

Output example
──────────────
{
  "action": "Increase GTA3",
  "action_index": 3,
  "confidence": 0.91,
  "expected_gain_mwh": 12000,
  "economic_gain_dh": 350000,
  "shap_explanation": "Net balance is below threshold. ..."
}
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.dqn_agent       import DoubleDQNAgent
from rl.reward_engine   import to_economic, PRICE_PER_MWH_DH, MWH_SCALE
from rl.environment     import ACTIONS, STATE_DIM, STATE_NAMES
from database.database  import insert_many

# Cost per action in MWh-equivalent (rough heuristic, calibrate with OCP data)
_ACTION_COST_MWH: dict[int, float] = {
    0: 0,       1: 500,    2: 500,    3: 500,    4: 800,
    5: 300,     6: 200,    7: 3000,   8: 3000,   9: 3000,   10: 1000,
}

# Expected gain per action in MWh (heuristic baseline — refined by training)
_ACTION_GAIN_MWH: dict[int, float] = {
    0: 0,       1: 4000,   2: 4000,   3: 5000,   4: 3000,
    5: 2000,    6: 2500,   7: 8000,   8: 8000,   9: 8000,   10: 3500,
}


class RecommendationEngine:

    def __init__(self):
        self.agent: DoubleDQNAgent | None = None
        self._load_agent()

    def _load_agent(self):
        try:
            a = DoubleDQNAgent()
            a.load()
            self.agent = a
        except Exception:
            self.agent = DoubleDQNAgent()   # untrained — will give random advice

    # ── Main entry point ──────────────────────────────────────────────────────

    def recommend(self, state: np.ndarray | None = None,
                  df: pd.DataFrame | None = None,
                  shap_explainer=None) -> dict:
        """
        Produce a recommendation given a state vector or a DataFrame row.
        """
        if state is None:
            state = self._state_from_df(df) if df is not None else np.zeros(STATE_DIM)

        rec     = self.agent.recommend(state)
        act_idx = rec["action_index"]
        gain    = _ACTION_GAIN_MWH.get(act_idx, 0)
        cost    = _ACTION_COST_MWH.get(act_idx, 0)
        net_mwh = max(0, gain - cost)
        econ    = to_economic(net_mwh / MWH_SCALE, ACTIONS[act_idx]["cost"])

        explanation = ""
        if shap_explainer is not None and df is not None:
            try:
                shap_res    = shap_explainer.explain(df)
                explanation = shap_explainer.explain_recommendation(rec["action"], shap_res)
            except Exception:
                pass

        result = {
            "action":             rec["action"],
            "action_index":       act_idx,
            "confidence":         rec["confidence"],
            "expected_gain_mwh":  round(net_mwh, 1),
            "economic_gain_dh":   round(econ["energy_gain_dh"], 0),
            "shap_explanation":   explanation,
            "q_values":           rec.get("q_values", {}),
        }

        self._persist(result)
        return result

    def recommend_top_k(self, state: np.ndarray, k: int = 3) -> list[dict]:
        """Return the top-k actions sorted by Q-value."""
        qv   = self.agent.q_values(state)
        idxs = np.argsort(qv)[::-1][:k]
        out  = []
        for i in idxs:
            net_mwh = max(0, _ACTION_GAIN_MWH.get(int(i), 0) - _ACTION_COST_MWH.get(int(i), 0))
            out.append({
                "action":            ACTIONS[int(i)]["label"],
                "action_index":      int(i),
                "q_value":           round(float(qv[i]), 4),
                "expected_gain_mwh": round(net_mwh, 1),
                "economic_gain_dh":  round(net_mwh * PRICE_PER_MWH_DH, 0),
            })
        return out

    # ── State helpers ─────────────────────────────────────────────────────────

    def _state_from_df(self, df: pd.DataFrame) -> np.ndarray:
        row   = df.iloc[-1]
        state = np.array([float(row.get(c, 0)) for c in STATE_NAMES], dtype=np.float32)
        return state

    def build_state(self, realtime: dict, anomaly_score: float = 0.0) -> np.ndarray:
        """Build a normalised state vector from a real-time SCADA reading."""
        return np.array([
            realtime.get("production",  0),
            realtime.get("bilan_net",   0),
            realtime.get("steam_hp",    0),
            realtime.get("steam_mp",    0),
            realtime.get("pressure",    0),
            realtime.get("vibration",   0),
            anomaly_score,
            realtime.get("efficiency",  0),
            realtime.get("month",       1),
        ], dtype=np.float32)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self, result: dict):
        insert_many("recommendations", [{
            "action":             result["action"],
            "action_index":       result["action_index"],
            "expected_gain_mwh":  result["expected_gain_mwh"],
            "economic_gain_dh":   result["economic_gain_dh"],
            "confidence":         result["confidence"],
            "shap_explanation":   result.get("shap_explanation", ""),
        }])