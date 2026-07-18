"""Tests for Layer 5 — Digital Twin What-If simulator + Monte Carlo (F19)."""

import pytest

from digital_twin.simulator import DigitalTwinSimulator
from digital_twin.monte_carlo import MonteCarloSimulator


class TestWhatIfSimulator:
    def test_gta_increase_improves_bilan(self, seeded_db):
        sim = DigitalTwinSimulator()
        res = sim.simulate("gta1", change_percent=15, duration_hours=24)
        assert res["variable"] == "gta1"
        assert res["predicted_bilan_change"] > 0, "raising GTA1 output must raise bilan"
        assert res["predicted_production_change"] > 0
        assert 0.0 <= res["risk_score"] <= 1.0
        assert "recommendation" in res and res["recommendation"]

    def test_negative_change_reduces_bilan(self, seeded_db):
        res = DigitalTwinSimulator().simulate("gta3", change_percent=-20)
        assert res["predicted_bilan_change"] < 0

    def test_all_changes_follow_sensitivity_map(self, seeded_db):
        res = DigitalTwinSimulator().simulate("steam_hp", change_percent=10)
        assert "efficiency" in res["all_changes"], "steam_hp must impact efficiency"

    def test_compare_scenarios(self, seeded_db):
        out = DigitalTwinSimulator().compare_scenarios([
            {"variable": "gta1", "change_percent": 10},
            {"variable": "gta2", "change_percent": -10},
        ])
        assert len(out) == 2
        assert out[0]["scenario"].startswith("gta1")


class TestMonteCarlo:
    def test_defaults_match_report(self):
        """F19 — N = 1000 iterations by default, seed = 42."""
        mc = MonteCarloSimulator()
        assert mc.n_trials == 1000

    def test_statistics_converge(self):
        mc = MonteCarloSimulator(n_trials=500, seed=42)
        out = mc.run(lambda: {"gain": 10.0}, noise_std={"gain": 1.0})
        stats = out["summary"]["gain"]
        assert out["n_trials"] == 500
        assert stats["mean"] == pytest.approx(10.0, abs=0.2)
        assert stats["std"] == pytest.approx(1.0, abs=0.2)
        assert stats["p5"] < stats["median"] < stats["p95"]
        assert len(out["raw"]) == 20

    def test_reproducible_with_seed(self):
        run = lambda: MonteCarloSimulator(n_trials=100, seed=42).run(
            lambda: {"x": 5.0}, noise_std={"x": 2.0})
        assert run()["summary"] == run()["summary"], "seed=42 must give identical results"

    def test_noise_estimated_from_dataframe(self, synth_df):
        mc = MonteCarloSimulator(n_trials=50, seed=1)
        out = mc.run(lambda: {"production": 1500.0}, df=synth_df)
        assert out["summary"]["production"]["std"] > 0
