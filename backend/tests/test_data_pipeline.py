"""Tests for Layer 1 — ingestion, cleaning and feature engineering (F02, F03)."""

import numpy as np
import pandas as pd

from data_pipeline.preprocessing import (
    clean_dataframe, resample_daily, normalize_minmax, denormalize,
)
from data_pipeline.feature_engineering import generate_features, add_all_features
from data_pipeline.data_validator import validate
from tests.conftest import make_synthetic_df


class TestCleanDataframe:
    def test_removes_nan_and_negatives(self):
        df = make_synthetic_df(60)
        df.loc[5, "production"] = np.nan
        df.loc[10, "steam_hp"] = -50.0        # physically impossible
        out = clean_dataframe(df)
        num = out.select_dtypes(include=[np.number])
        assert not num.isna().any().any(), "NaN values must be imputed"
        assert (out["steam_hp"] >= 0).all(), "negative steam flow must be removed"

    def test_deduplicates_timestamps(self):
        df = make_synthetic_df(30)
        df.loc[29, "timestamp"] = df.loc[28, "timestamp"]  # duplicate
        out = clean_dataframe(df)
        assert out["timestamp"].is_unique

    def test_clips_extreme_outliers(self):
        df = make_synthetic_df(100)
        df.loc[50, "vibration"] = 45.0        # extreme outlier (within 0-50 guard)
        out = clean_dataframe(df)
        mu, sd = df["vibration"].mean(), df["vibration"].std()
        assert out["vibration"].max() <= mu + 4 * sd + 1e-6, "±4σ clip must apply"
        assert out["vibration"].max() < 45.0, "outlier must be reduced"


class TestResampleNormalize:
    def test_resample_daily_keeps_rows(self, synth_df):
        out = resample_daily(synth_df)
        assert len(out) > 0
        assert "timestamp" in out.columns

    def test_normalize_denormalize_roundtrip(self, synth_df):
        norm, params = normalize_minmax(synth_df[["production", "steam_hp"]].copy())
        assert norm["production"].min() >= 0.0 and norm["production"].max() <= 1.0
        restored = denormalize(norm["production"].values, params["production"])
        np.testing.assert_allclose(restored, synth_df["production"].values, rtol=1e-6)


class TestFeatureEngineering:
    def test_generate_features_adds_temporal_and_lags(self, synth_df):
        out = generate_features(synth_df)
        for col in ("month", "day_of_week", "is_weekend",
                    "production_lag_1", "steam_per_mwh", "active_gta_count"):
            assert col in out.columns, f"missing engineered feature: {col}"
        assert not out["production_lag_1"].isna().any()

    def test_add_all_features_adds_derived_columns(self, synth_df):
        out = add_all_features(synth_df)
        assert out.shape[1] > synth_df.shape[1]
        for col in ("gta_balance", "delta_production", "production_roll_7d"):
            assert col in out.columns, f"missing engineered feature: {col}"


class TestValidator:
    def test_validate_returns_report(self, synth_df):
        report = validate(synth_df)
        summary = report.summary()
        assert isinstance(summary, dict) and summary, "validator must produce a summary"
