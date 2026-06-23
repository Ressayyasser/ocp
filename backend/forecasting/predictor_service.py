"""
predictor_service.py — High-level orchestration for training and serving forecasts.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from forecasting.xgboost_predictor import XGBoostPredictor
from data_pipeline.feature_engineering import add_all_features
from data_pipeline.preprocessing import clean_dataframe
from database.database import query, insert_many


class PredictorService:

    def __init__(self):
        self.predictor = XGBoostPredictor()

    def _load_df(self) -> pd.DataFrame:
        rows = query("SELECT * FROM historical_data ORDER BY timestamp")
        if not rows:
            raise ValueError("No historical data in DB — run excel_loader first")
        df = pd.DataFrame(rows)
        df = clean_dataframe(df)
        df = add_all_features(df)
        return df

    def train_all(self) -> dict:
        df = self._load_df()
        print(f"[Forecasting] Training on {len(df):,} rows × {len(df.columns)} features")
        return self.predictor.train_all(df)

    def predict(self, target: str = "bilan_net", horizon: str = "24h") -> dict:
        df     = self._load_df()
        result = self.predictor.predict(df, target, horizon)
        insert_many("predictions", [{
            "timestamp":       result["timestamp"],
            "variable":        result["variable"],
            "horizon":         result["horizon"],
            "predicted_value": result["predicted_value"],
            "confidence":      result["confidence"],
            "model_version":   "xgboost_v1",  # Ajustement : Renseigne la colonne model_version
        }])
        return result

    def predict_all(self) -> list[dict]:
        df      = self._load_df()
        results = self.predictor.predict_all(df)
        if results:
            insert_many("predictions", [{
                "timestamp":       r["timestamp"], 
                "variable":        r["variable"],
                "horizon":         r["horizon"], 
                "predicted_value": r["predicted_value"],
                "confidence":      r["confidence"],
                "model_version":   "xgboost_v1",  # Ajustement : Renseigne la colonne model_version
            } for r in results])
        return results

    def get_recent(self, limit: int = 50) -> list[dict]:
        return query("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", [limit])

    def feature_importance(self, target="bilan_net", horizon="24h") -> dict:
        return self.predictor.feature_importance(target, horizon)