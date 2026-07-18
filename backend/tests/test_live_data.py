"""Tests for data_pipeline/live_data.py — historical + simulation_data bridge."""

import pandas as pd
import pytest

from data_pipeline.live_data import (
    backfill_simulation_gap, aggregate_simulation_daily,
    get_combined_daily_df, GTA_KEYS, BACKFILL_SIM_ID,
)
from database.database import query


TODAY = pd.Timestamp.now().normalize()


class TestBackfill:
    def test_fills_gap_up_to_today(self, seeded_db):
        # NB: other tests (e.g. the API suite) may already have triggered the
        # backfill on the shared session DB — assert coverage, not row count.
        backfill_simulation_gap()
        last_day = query(
            "SELECT MAX(substr(timestamp,1,10)) AS d FROM simulation_data")[0]["d"]
        assert last_day == TODAY.strftime("%Y-%m-%d")
        gtas = {r["gta_type"] for r in query(
            "SELECT DISTINCT gta_type FROM simulation_data WHERE simulation_id = ?",
            [BACKFILL_SIM_ID])}
        assert gtas == set(GTA_KEYS)

    def test_idempotent(self, seeded_db):
        backfill_simulation_gap()
        assert backfill_simulation_gap() == 0, "already-covered days must be skipped"


class TestAggregation:
    def test_daily_schema_mapping(self, seeded_db):
        backfill_simulation_gap()
        agg = aggregate_simulation_daily()
        assert not agg.empty
        row = agg.iloc[-1]
        # plant production is the sum of the per-GTA daily energies
        assert row["production"] == pytest.approx(
            row["gta1"] + row["gta2"] + row["gta3"], rel=1e-3)
        assert row["bilan_net"] == pytest.approx(
            row["production"] - row["consumption"], rel=1e-3)
        assert 0.2 <= row["efficiency"] <= 0.55          # fraction, not percent
        assert row["source"] == "live_sim"
        for col in ("steam_hp", "steam_mp", "pressure", "temperature",
                    "rendement_gta1", "debit_adm_gta2", "vibration"):
            assert col in agg.columns


class TestCombinedSeries:
    def test_extends_history_to_today(self, seeded_db):
        df = get_combined_daily_df()
        assert not df.empty
        assert str(df["timestamp"].iloc[-1])[:10] == TODAY.strftime("%Y-%m-%d")
        assert set(df["source"].unique()) == {"historical", "live_sim"}
        # continuity: no simulated day may precede the last historical day
        last_hist = df.loc[df["source"] == "historical", "timestamp"].max()
        first_sim = df.loc[df["source"] == "live_sim", "timestamp"].min()
        assert str(first_sim) > str(last_hist)

    def test_sorted_and_deduplicated_days(self, seeded_db):
        df = get_combined_daily_df()
        ts = df["timestamp"].astype(str).tolist()
        assert ts == sorted(ts)
        sim_days = df.loc[df["source"] == "live_sim", "timestamp"].astype(str).str[:10]
        assert sim_days.is_unique
