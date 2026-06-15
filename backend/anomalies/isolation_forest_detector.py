"""
isolation_forest_detector.py — Multi-group anomaly detection.

Detection groups
────────────────
production  — production / GTA balance drop
pressure    — pressure / steam anomalies
vibration   — vibration spikes
steam       — steam network faults
overall     — system-wide composite score

Output example
──────────────
{"is_anomaly": true, "score": 0.95, "severity": "critical",
 "cause": "vibration", "group": "vibration"}
"""

from __future__ import annotations
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _HAS_SKL = True
except ImportError:
    _HAS_SKL = False

MODEL_DIR = os.environ.get("MODEL_DIR", "models/anomaly")

GROUPS: dict[str, dict] = {
    "production": {
        "features": [
            "production", "gta1", "gta2", "gta3",
            "delta_production", "delta_gta1", "delta_gta2", "delta_gta3",
            "production_roll_7d", "efficiency",
        ],
        "label": "Production anomaly",
    },
    "rendement": {
        "features": [
            "rendement_gta1", "rendement_gta2", "rendement_gta3",
            "efficiency", "gta_balance",
        ],
        "label": "GTA efficiency anomaly",
    },
    "steam": {
        "features": [
            "steam_hp", "steam_mp", "steam_bp", "steam_ratio",
            "debit_adm_gta1", "debit_adm_gta2", "debit_adm_gta3",
            "debit_sout_gta1", "debit_sout_gta2", "debit_sout_gta3",
        ],
        "label": "Steam network anomaly",
    },
    "thermodynamic": {
        "features": [
            "pression_adm_gta1", "pression_adm_gta2", "pression_adm_gta3",
            "temp_adm_gta1", "temp_adm_gta2", "temp_adm_gta3",
            "pressure", "temperature",
        ],
        "label": "Thermodynamic anomaly",
    },
    "overall": {
        "features": [
            "production", "bilan_net", "efficiency", "steam_hp", "steam_mp",
            "gta1", "gta2", "gta3", "gta_balance",
            "debit_adm_gta1", "debit_adm_gta2", "debit_adm_gta3",
            "rendement_gta1", "rendement_gta2", "rendement_gta3",
        ],
        "label": "System-wide anomaly",
    },
}

# More negative score → more anomalous
_SEV: dict[str, float] = {"critical": -0.5, "warning": -0.3, "info": -0.1}


class IsolationForestDetector:

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.models: dict[str, IsolationForest] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self._features: dict[str, list[str]]   = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, group: str = "overall") -> dict:
        if not _HAS_SKL:
            raise ImportError("pip install scikit-learn")
        cfg   = GROUPS[group]
        feats = [f for f in cfg["features"] if f in df.columns]
        if len(feats) < 2:
            raise ValueError(f"Insufficient features for group '{group}'")
        X = df[feats].dropna().values
        if len(X) < 50:
            raise ValueError(f"Only {len(X)} samples for '{group}'")

        scaler  = StandardScaler()
        Xs      = scaler.fit_transform(X)
        model   = IsolationForest(contamination=self.contamination,
                                  n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(Xs)

        preds  = model.predict(Xs)
        n_anom = int((preds == -1).sum())

        self.models[group]   = model
        self.scalers[group]  = scaler
        self._features[group] = feats
        self._save(group, feats)

        return {"group": group, "n_samples": len(X), "n_anomalies": n_anom,
                "anomaly_rate": n_anom / len(X), "features": feats}

    def train_all(self, df: pd.DataFrame) -> dict:
        results = {}
        for group in GROUPS:
            try:
                m = self.train(df, group)
                results[group] = m
                print(f"  {group}: {m['n_anomalies']} anomalies ({m['anomaly_rate']:.1%})")
            except Exception as exc:
                results[group] = {"error": str(exc)}
                print(f"  {group}: FAILED — {exc}")
        return results

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, data: dict, group: str = "overall") -> dict:
        """Detect anomaly in a single real-time reading."""
        if group not in self.models:
            self._load(group)
        if group not in self.models:
            raise ValueError(f"No model for group '{group}'")

        feats = self._features[group]
        X     = np.array([float(data.get(f, 0)) for f in feats]).reshape(1, -1)
        Xs    = self.scalers[group].transform(X)

        score    = float(self.models[group].decision_function(Xs)[0])
        is_anom  = self.models[group].predict(Xs)[0] == -1
        severity = "normal"
        if is_anom:
            for sev, threshold in _SEV.items():
                if score < threshold:
                    severity = sev
                    break
            if severity == "normal":
                severity = "info"

        cause = feats[int(np.argmax(np.abs(Xs[0])))] if is_anom else None

        return {
            "is_anomaly": is_anom,
            "score":      round(score, 4),
            "severity":   severity,
            "group":      group,
            "cause":      cause,
            "label":      GROUPS[group]["label"],
            "timestamp":  datetime.now().isoformat(),
        }

    def detect_all(self, data: dict) -> list[dict]:
        return [r for group in GROUPS
                for r in [self._safe_detect(data, group)] if r and r["is_anomaly"]]

    def detect_batch(self, df: pd.DataFrame, group: str = "overall") -> pd.DataFrame:
        if group not in self.models:
            self._load(group)
        feats = self._features[group]
        avail = [f for f in feats if f in df.columns]
        X     = df[avail].fillna(0).values
        Xs    = self.scalers[group].transform(X)
        scores = self.models[group].decision_function(Xs)
        preds  = self.models[group].predict(Xs)
        out    = df.copy()
        out["anomaly_score"] = scores
        out["is_anomaly"]    = preds == -1
        out["severity"]      = "normal"
        for sev, thr in _SEV.items():
            out.loc[out["is_anomaly"] & (out["anomaly_score"] < thr), "severity"] = sev
        return out

    def _safe_detect(self, data: dict, group: str):
        try:
            return self.detect(data, group)
        except Exception:
            return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, group: str, feats: list[str]):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, f"iforest_{group}.pkl"), "wb") as f:
            pickle.dump({"model": self.models[group], "scaler": self.scalers[group]}, f)
        with open(os.path.join(MODEL_DIR, f"iforest_{group}_meta.json"), "w") as f:
            json.dump({"features": feats, "trained_at": datetime.now().isoformat()}, f)

    def _load(self, group: str):
        pkl  = os.path.join(MODEL_DIR, f"iforest_{group}.pkl")
        meta = os.path.join(MODEL_DIR, f"iforest_{group}_meta.json")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                d = pickle.load(f)
            self.models[group]  = d["model"]
            self.scalers[group] = d["scaler"]
        if os.path.exists(meta):
            with open(meta) as f:
                self._features[group] = json.load(f).get("features", [])