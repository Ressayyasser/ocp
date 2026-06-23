"""
shap_explainer.py — SHAP TreeExplainer wrapper for XGBoost forecasts.
Returns data formatted specifically for the Dash frontend charts.
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

    # ── Explain predictions (returns data formatted for Dash frontend) ────────

    def explain(self, df: pd.DataFrame, target: str = "bilan_net",
                horizon: str = "24h") -> dict:
        """Compute SHAP values for a sample of data to support frontend charts."""
        key     = f"{target}_{horizon}"
        feats   = self.predictor.feature_names.get(key)
        if feats is None:
            self.predictor._load(key)
            feats = self.predictor.feature_names.get(key, [])
            
        if not feats:
            raise ValueError(f"No features found for model {key}")

        # Prepare input data: take the last 100 rows for charts, or all if less
        n_samples = min(100, len(df))
        X_sample = df.tail(n_samples)[feats].fillna(0).values

        explainer   = self._get_explainer(target, horizon)
        
        # Compute SHAP values for the sample (returns a 2D matrix)
        shap_values_matrix = explainer.shap_values(X_sample)
        if isinstance(shap_values_matrix, list):
            shap_values_matrix = shap_values_matrix[0]
            
        # Base value
        base_val   = float(explainer.expected_value
                           if not hasattr(explainer.expected_value, "__len__")
                           else explainer.expected_value[0])
                           
        # Last row prediction
        sv_last = shap_values_matrix[-1]
        pred_val = float(base_val + sv_last.sum())
        
        # Compute mean absolute SHAP for the bar chart
        mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0).tolist()
        
        # Top features for narrative
        contributions = {f: round(float(v), 4) for f, v in zip(feats, sv_last)}
        top = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

        # Return EXACTLY the keys the Dash frontend expects
        return {
            "feature_names": feats,
            "mean_abs_shap": mean_abs_shap,
            "shap_values": shap_values_matrix.tolist(), # 2D list for heatmap
            "base_value": round(base_val, 2),
            "predicted_value": round(pred_val, 2),
            "explanation": self._narrative(top, contributions),
            "target": target,
            "horizon": horizon,
        }

    # ── Explain an RL recommendation ─────────────────────────────────────────

    def explain_recommendation(self, action_label: str,
                                shap_result: dict) -> str:
        """Convert a SHAP result dict into a human-readable recommendation text."""
        top = list(shap_result.get("top_features", {}).keys())[:3]
        if not top and "feature_names" in shap_result:
            mean_abs = shap_result.get("mean_abs_shap", [])
            if mean_abs:
                top_idx = sorted(range(len(mean_abs)), key=lambda i: abs(mean_abs[i]), reverse=True)[:3]
                top = [shap_result["feature_names"][i] for i in top_idx]
                
        parts = [f"Recommendation: {action_label}", "", "Reason:"]
        
        contributions = shap_result.get("feature_contributions", {})
        if not contributions and "feature_names" in shap_result and "shap_values" in shap_result:
            feats = shap_result["feature_names"]
            sv_last = shap_result["shap_values"][-1]
            contributions = {f: v for f, v in zip(feats, sv_last)}
            
        for feat in top:
            val = contributions.get(feat, 0)
            direction = "high" if val > 0 else "low"
            tmpl = _NARRATIVES.get(feat, {}).get(direction, f"{feat} is a key driver.")
            parts.append(f"  • {tmpl}")
        return "\n".join(parts)

    # ── Narrative builder ─────────────────────────────────────────────────────

    def _narrative(self, top: dict, contributions: dict) -> str:
        lines: list[str] = []
        for feat, val in list(top.items())[:4]:
            direction = "high" if val > 0 else "low"
            tmpl = _NARRATIVES.get(feat, {}).get(direction, f"{feat} is a key driver.")
            lines.append(tmpl)
        return "  ".join(lines)