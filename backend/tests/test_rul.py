"""Tests for F09b — RUL engine (6 components, health 0–100, linear degradation)."""

import numpy as np
import pytest

from rul.rul_engine import RULEngine, _MTTF_DAYS

NOMINAL_READING = {
    "vibration": 1.3, "temperature": 455.0, "pressure": 54.5,
    "efficiency": 0.405, "steam_hp": 530.0,
}

COMPONENTS = {"Turbine_HP", "Rotor", "Bearings", "Valves", "Condenser", "Generator"}


class TestRULEngine:
    def test_tracks_six_report_components(self):
        health = RULEngine().compute_health(NOMINAL_READING)
        assert set(health.keys()) == COMPONENTS

    def test_nominal_reading_is_healthy(self):
        health = RULEngine().compute_health(NOMINAL_READING)
        assert all(score >= 95 for score in health.values()), health

    def test_critical_vibration_degrades_bearings_most(self):
        reading = dict(NOMINAL_READING, vibration=7.0)   # critical threshold
        health = RULEngine().compute_health(reading)
        assert health["Bearings"] < 60
        assert health["Bearings"] < health["Valves"]     # Valves ignore vibration

    def test_rul_is_linear_in_health(self):
        engine = RULEngine()
        rul = engine.estimate_rul({"Bearings": 50.0, "Condenser": 100.0})
        assert rul["Bearings"]["rul_days"] == pytest.approx(_MTTF_DAYS["Bearings"] * 0.5)
        assert rul["Condenser"]["rul_days"] == pytest.approx(_MTTF_DAYS["Condenser"])

    def test_status_thresholds(self):
        rul = RULEngine().estimate_rul({"Rotor": 20.0, "Valves": 45.0, "Generator": 90.0})
        assert rul["Rotor"]["status"] == "critical"
        assert rul["Valves"]["status"] == "warning"
        assert rul["Generator"]["status"] == "healthy"

    def test_summary_identifies_most_critical(self):
        reading = dict(NOMINAL_READING, temperature=489.0)  # near-critical overtemp
        summary = RULEngine().get_summary(reading, hours_operated=10_000)
        assert summary["most_critical"] in COMPONENTS
        assert set(summary["rul"].keys()) == COMPONENTS
        assert "timestamp" in summary

    def test_state_extension_for_rl(self):
        engine = RULEngine()
        health = engine.compute_health(NOMINAL_READING)
        extended = engine.update_state_with_rul(np.zeros(9, dtype=np.float32), health)
        assert extended.shape == (9 + len(COMPONENTS),)
        assert np.all((extended[9:] >= 0) & (extended[9:] <= 1))
