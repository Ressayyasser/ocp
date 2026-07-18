"""Tests for F04b — XGBoost multi-target / multi-horizon forecasting."""

import numpy as np
import pytest

from forecasting.xgboost_predictor import XGBoostPredictor, TARGETS, HORIZONS
from data_pipeline.feature_engineering import add_all_features
from tests.conftest import make_synthetic_df


@pytest.fixture(scope="module")
def featured_df():
    return add_all_features(make_synthetic_df(300))


class TestXGBoostPredictor:
    def test_report_targets_and_horizons(self):
        """F04b — multi-target (bilan_net, production, efficiency, steam_hp),
        multi-horizon (24h, 7d, 30d)."""
        assert set(TARGETS) == {"bilan_net", "production", "efficiency", "steam_hp"}
        assert {"24h", "7d", "30d"}.issubset(HORIZONS.keys())

    def test_train_returns_metrics(self, featured_df):
        pred = XGBoostPredictor()
        metrics = pred.train(featured_df, target="bilan_net", horizon_key="1d")
        assert metrics["train"] > 0 and metrics["test"] > 0
        assert np.isfinite(metrics["mae"]) and metrics["mae"] >= 0
        assert np.isfinite(metrics["r2"])

    def test_predict_returns_confidence(self, featured_df):
        pred = XGBoostPredictor()
        pred.train(featured_df, target="production", horizon_key="1d")
        out = pred.predict(featured_df, target="production", horizon_key="1d")
        assert out["variable"] == "production"
        assert np.isfinite(out["predicted_value"])
        assert 0.0 <= out["confidence"] <= 1.0

    def test_train_rejects_tiny_dataset(self):
        small = add_all_features(make_synthetic_df(60))
        with pytest.raises(ValueError):
            XGBoostPredictor().train(small, target="bilan_net", horizon_key="1d")

    def test_reproducible_with_seed_42(self, featured_df):
        """KPI C.1 — reproducibility (random_state=42)."""
        m1 = XGBoostPredictor().train(featured_df, "bilan_net", "1d")
        m2 = XGBoostPredictor().train(featured_df, "bilan_net", "1d")
        assert m1["mae"] == pytest.approx(m2["mae"])
        assert m1["r2"] == pytest.approx(m2["r2"])
