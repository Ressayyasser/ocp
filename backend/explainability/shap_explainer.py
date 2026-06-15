"""
shap_explainer.py — SHAP TreeExplainer wrapper for XGBoost forecasts.

Output example
──────────────
{
  "feature_contributions": {"bilan_net": 52, "pressure": 18, "steam_hp": 14},
  "base_value": 120000.0,
  "predicted_value": 145320.0,
  "narrative": "Net balance is below threshold.  Pressure remains stable. ..."
}
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from forecasting.xgboost_predictor import XGBoostPredictor

# Human-readable narrative templates per feature
_NARRATIVES: dict[str, dict[str, str]] = {
    "bilan_net":    {"high": "Net balance is favourable.",
                     "low":  "Net balance is below threshold."},
    "pressure":     {"high": "Pressure exceeds normal range.",
                     "low":  "Pressure remains stable."},
    "steam_hp":     {"high": "Steam availability is high.",
                     "low":  "Steam supply is limited."},
    "steam_mp":     {"high": "Medium-pressure steam is available.",
                     "low":  "Medium-pressure steam is low."},
    "vibration":    {"high": "Vibration is elevated — maintenance may be needed.",
                     "low":  "Vibration within normal bounds."},
    "efficiency":   {"high": "Operational efficiency is good.",
                     "low":  "Efficiency is degraded."},
    "temperature":  {"high": "Temperature is elevated.",
                     "low":  "Temperature is nominal."},
    "production":   {"high": "Production is above target.",
                     "low":  "Production is below target."},
}


class SHAPExplainer:

    def __init__(self, predictor: XGBoostPredictor):
        self.predictor   = predictor
        self._explainers: dict[str, object] = {}

    # ── Build explainer ───────────────────────────────────────────────────────

    def _get_explainer(self, target: str, horizon: str):
        key = f"{target}_{horizon}"
        if key in self._explainers:
            return self._explainers[key]
        if not _HAS_SHAP:
            raise ImportError("pip install shap")
        if key not in self.predictor.models:
            self.predictor._load(key)
        if key not in self.predictor.models:
            raise ValueError(f"No trained model for {key}")
        ex = shap.TreeExplainer(self.predictor.models[key])
        self._explainers[key] = ex
        return ex

    # ── Explain one prediction ────────────────────────────────────────────────

    def explain(self, df: pd.DataFrame, target: str = "bilan_net",
                horizon: str = "24h") -> dict:
        """Compute SHAP values for the most recent data point."""
        key     = f"{target}_{horizon}"
        feats   = self.predictor.feature_names.get(key)
        if feats is None:
            self.predictor._load(key)
            feats = self.predictor.feature_names.get(key, [])

        # Prepare input row
        avail = [f for f in feats if f in df.columns]
        row   = df.iloc[-1:].copy()
        for f in feats:
            if f not in row.columns:
                row[f] = 0.0
        X = row[feats].fillna(0).values

        explainer   = self._get_explainer(target, horizon)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            sv = shap_values[0][0]
        else:
            sv = shap_values[0]

        base_val   = float(explainer.expected_value
                           if not hasattr(explainer.expected_value, "__len__")
                           else explainer.expected_value[0])
        pred_val   = float(base_val + sv.sum())

        contributions = {f: round(float(v), 4) for f, v in zip(feats, sv)}
        top           = dict(sorted(contributions.items(),
                                    key=lambda x: abs(x[1]), reverse=True)[:10])

        # Convert to percentage importance
        total_abs = sum(abs(v) for v in top.values()) or 1
        pct       = {f: round(abs(v) / total_abs * 100, 1) for f, v in top.items()}

        return {
            "feature_contributions": contributions,
            "top_features":          top,
            "importance_pct":        pct,
            "base_value":            round(base_val, 2),
            "predicted_value":       round(pred_val, 2),
            "narrative":             self._narrative(top, contributions),
            "target":                target,
            "horizon":               horizon,
        }

    # ── Explain an RL recommendation ─────────────────────────────────────────

    def explain_recommendation(self, action_label: str,
                                shap_result: dict) -> str:
        """
        Convert a SHAP result dict into a human-readable recommendation text.
        """
        top = list(shap_result.get("top_features", {}).keys())[:3]
        pct = shap_result.get("importance_pct", {})
        parts = [f"Recommendation: {action_label}", "", "Reason:"]
        for feat in top:
            val = shap_result["feature_contributions"].get(feat, 0)
            direction = "high" if val > 0 else "low"
            tmpl = _NARRATIVES.get(feat, {}).get(direction, f"{feat} is a key driver.")
            parts.append(f"  • {tmpl}  (contribution {pct.get(feat, 0):.0f}%)")
        return "\n".join(parts)

    # ── Narrative builder ─────────────────────────────────────────────────────

    def _narrative(self, top: dict, contributions: dict) -> str:
        lines: list[str] = []
        for feat, val in list(top.items())[:4]:
            direction = "high" if val > 0 else "low"
            tmpl = _NARRATIVES.get(feat, {}).get(direction, f"{feat} is a key driver.")
            lines.append(tmpl)
        return "  ".join(lines)

    # ── Waterfall data for Dash ───────────────────────────────────────────────

    def waterfall_data(self, shap_result: dict) -> dict:
        """Return data structure ready for Plotly waterfall chart."""
        contrib = shap_result.get("top_features", {})
        feats   = list(contrib.keys())
        vals    = list(contrib.values())
        base    = shap_result.get("base_value", 0)
        return {
            "features":   feats,
            "values":     vals,
            "base_value": base,
            "measure":    ["relative"] * len(feats) + ["total"],
            "x":          feats + ["Final"],
            "y":          vals  + [shap_result.get("predicted_value", base + sum(vals))],
        }