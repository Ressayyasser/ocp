"""
xgboost_predictor.py — Multi-target, multi-horizon XGBoost forecaster.

Targets  : bilan_net | production | efficiency
Horizons : 1h | 24h | 7d | 30d

Output example
──────────────
{"variable":"bilan_net","horizon":"24h","predicted_value":145320.0,"confidence":0.92}
"""

from __future__ import annotations
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    _HAS_SKL = True
except ImportError:
    _HAS_SKL = False

MODEL_DIR = os.environ.get("MODEL_DIR", "models/forecasting")

TARGETS: list[str] = ["bilan_net", "production", "efficiency"]

HORIZONS: dict[str, int] = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}

_BASE_FEATS: dict[str, list[str]] = {
    "bilan_net": [
        "production", "consumption", "steam_hp", "steam_mp", "steam_bp",
        "steam_ratio", "efficiency", "gta_balance",
        "gta1", "gta2", "gta3",
        "debit_adm_gta1", "debit_adm_gta2", "debit_adm_gta3",
        "rendement_gta1", "rendement_gta2", "rendement_gta3",
        "pression_adm_gta1", "temp_adm_gta1",
        "pressure", "temperature",
        "delta_production", "delta_gta1", "delta_gta2", "delta_gta3",
        "production_roll_7d", "bilan_net_roll_7d", "production_roll_30d",
        "month_sin", "month_cos", "is_weekend", "day_of_week",
    ],
    "production": [
        "gta1", "gta2", "gta3",
        "debit_adm_gta1", "debit_adm_gta2", "debit_adm_gta3",
        "debit_sout_gta1", "debit_sout_gta2", "debit_sout_gta3",
        "rendement_gta1", "rendement_gta2", "rendement_gta3",
        "steam_hp", "steam_mp", "pression_adm_gta1", "pression_adm_gta2", "pression_adm_gta3",
        "efficiency", "gta_balance",
        "delta_gta1", "delta_gta2", "delta_gta3",
        "production_roll_7d", "production_roll_30d",
        "month_sin", "month_cos", "day_of_week",
    ],
    "efficiency": [
        "production", "steam_hp", "steam_mp", "steam_bp", "steam_ratio",
        "gta1", "gta2", "gta3",
        "debit_adm_gta1", "debit_adm_gta2", "debit_adm_gta3",
        "rendement_gta1", "rendement_gta2", "rendement_gta3",
        "pression_adm_gta1", "pression_adm_gta2", "pression_adm_gta3",
        "temp_adm_gta1", "temp_adm_gta2", "temp_adm_gta3",
        "gta_balance", "efficiency_roll_7d",
        "month_sin", "month_cos", "day_of_week",
    ],
}

_XGB_PARAMS: dict = {
    "objective":        "reg:squarederror",
    "max_depth":        6,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
}


class XGBoostPredictor:

    def __init__(self):
        self.models: dict[str, object]       = {}
        self.feature_names: dict[str, list]  = {}
        self.metrics: dict[str, dict]        = {}

    # ── Feature helpers ───────────────────────────────────────────────────────

    def _feats(self, df: pd.DataFrame, target: str) -> list[str]:
        desired = _BASE_FEATS.get(target, _BASE_FEATS["bilan_net"])
        return [f for f in desired if f in df.columns and f != target]

    def _supervised(self, df: pd.DataFrame, target: str, horizon: int):
        feats = self._feats(df, target)
        if not feats:
            raise ValueError(f"No features for target='{target}'")
        X = df[feats].values
        y = df[target].shift(-horizon).values
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]
        row_ok = ~np.isnan(X).any(axis=1)
        return X[row_ok], y[row_ok], feats

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, target: str = "bilan_net",
              horizon_key: str = "24h") -> dict:
        if not _HAS_XGB:
            raise ImportError("pip install xgboost")
        horizon = HORIZONS[horizon_key]
        X, y, feats = self._supervised(df, target, horizon)
        if len(X) < 100:
            raise ValueError(f"Only {len(X)} samples (need ≥100)")

        split  = int(len(X) * 0.8)
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]

        model = xgb.XGBRegressor(**_XGB_PARAMS)
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

        ypred = model.predict(Xte)
        metrics = {
            "mae":   float(mean_absolute_error(yte, ypred)) if _HAS_SKL else None,
            "rmse":  float(np.sqrt(mean_squared_error(yte, ypred))) if _HAS_SKL else None,
            "r2":    float(r2_score(yte, ypred)) if _HAS_SKL else None,
            "train": split,
            "test":  len(Xte),
        }

        key = f"{target}_{horizon_key}"
        self.models[key]       = model
        self.feature_names[key] = feats
        self.metrics[key]      = metrics
        self._save(key)
        return metrics

    def train_all(self, df: pd.DataFrame) -> dict:
        results = {}
        for target in TARGETS:
            if target not in df.columns:
                continue
            for hk in HORIZONS:
                key = f"{target}_{hk}"
                try:
                    m = self.train(df, target, hk)
                    results[key] = m
                    print(f"  {key}: MAE={m['mae']:.2f}  R²={m['r2']:.3f}")
                except Exception as exc:
                    results[key] = {"error": str(exc)}
                    print(f"  {key}: FAILED — {exc}")
        return results

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, target: str = "bilan_net",
                horizon_key: str = "24h") -> dict:
        key = f"{target}_{horizon_key}"
        if key not in self.models:
            self._load(key)
        if key not in self.models:
            raise ValueError(f"No trained model for {key}")

        feats   = self.feature_names[key]
        row     = df.iloc[-1:].copy()
        for f in feats:
            if f not in row.columns:
                row[f] = 0.0
        X       = row[feats].fillna(0).values
        value   = float(self.models[key].predict(X)[0])
        r2      = self.metrics.get(key, {}).get("r2") or 0.5
        conf    = max(0.0, min(1.0, r2))

        return {
            "variable":        target,
            "horizon":         horizon_key,
            "predicted_value": round(value, 2),
            "confidence":      round(conf,  3),
            "timestamp":       datetime.now().isoformat(),
        }

    def predict_all(self, df: pd.DataFrame) -> list[dict]:
        out = []
        for target in TARGETS:
            for hk in HORIZONS:
                key = f"{target}_{hk}"
                if self._exists(key) or key in self.models:
                    try:
                        out.append(self.predict(df, target, hk))
                    except Exception:
                        pass
        return out

    def feature_importance(self, target: str, horizon_key: str) -> dict:
        key = f"{target}_{horizon_key}"
        if key not in self.models:
            self._load(key)
        imp = self.models[key].feature_importances_
        return dict(sorted(
            zip(self.feature_names[key], map(float, imp)),
            key=lambda x: x[1], reverse=True
        ))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, key: str):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, f"{key}.pkl"), "wb") as f:
            pickle.dump(self.models[key], f)
        with open(os.path.join(MODEL_DIR, f"{key}_meta.json"), "w") as f:
            json.dump({"features": self.feature_names[key],
                       "metrics":  self.metrics.get(key, {}),
                       "trained_at": datetime.now().isoformat()}, f, indent=2)

    def _load(self, key: str):
        pkl  = os.path.join(MODEL_DIR, f"{key}.pkl")
        meta = os.path.join(MODEL_DIR, f"{key}_meta.json")
        if os.path.exists(pkl) and os.path.exists(meta):
            with open(pkl,  "rb") as f: self.models[key] = pickle.load(f)
            with open(meta)        as f: d = json.load(f)
            self.feature_names[key] = d["features"]
            self.metrics[key]       = d.get("metrics", {})

    def _exists(self, key: str) -> bool:
        return os.path.exists(os.path.join(MODEL_DIR, f"{key}.pkl"))