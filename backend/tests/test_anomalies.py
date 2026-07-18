"""Tests for Layer 3 — anomaly detection: Isolation Forest (F08) + CUSUM (F09)."""

import numpy as np
import pytest

from anomalies.cusum_detector import CUSUMDetector
from anomalies.isolation_forest_detector import IsolationForestDetector
from data_pipeline.feature_engineering import add_all_features
from tests.conftest import make_synthetic_df


class TestCUSUM:
    def test_detects_upward_shift(self):
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(0, 1, 80), rng.normal(6, 1, 40)])
        cps = CUSUMDetector(threshold=5.0, drift=0.5).detect(data)
        assert cps, "CUSUM must flag an obvious upward level shift"
        assert any(c.direction == "increase" and c.idx >= 75 for c in cps)

    def test_detects_downward_drift(self):
        rng = np.random.default_rng(1)
        base = rng.normal(10, 0.5, 60)
        drifting = 10 - np.linspace(0, 6, 60) + rng.normal(0, 0.5, 60)
        cps = CUSUMDetector(threshold=5.0, drift=0.5).detect(np.concatenate([base, drifting]))
        assert any(c.direction == "decrease" for c in cps), "slow degradation must be caught"

    def test_stable_signal_has_no_changepoint(self):
        rng = np.random.default_rng(2)
        cps = CUSUMDetector(threshold=8.0, drift=0.5).detect(rng.normal(0, 1, 150))
        assert cps == []

    def test_short_series_returns_empty(self):
        assert CUSUMDetector().detect(np.array([1.0, 2.0, 3.0])) == []

    def test_detect_dataframe(self, synth_df):
        df = synth_df.copy()
        df.loc[150:, "efficiency"] = 0.30      # injected efficiency collapse
        res = CUSUMDetector(threshold=4.0).detect_dataframe(df, ["efficiency", "pressure"])
        assert "efficiency" in res


class TestIsolationForest:
    @pytest.fixture(scope="class")
    def trained(self):
        det = IsolationForestDetector(contamination=0.05)
        df = add_all_features(make_synthetic_df(300))
        det.train_all(df)
        return det

    def test_train_reports_metrics(self, trained):
        det = IsolationForestDetector(contamination=0.05)
        df = add_all_features(make_synthetic_df(300))
        info = det.train(df, "overall")
        assert info["n_samples"] >= 50
        assert 0.0 <= info["anomaly_rate"] <= 0.2
        assert set(info["features"]).issubset(set(df.columns))

    def test_normal_reading_scores_higher_than_extreme(self, trained):
        normal = {"production": 1560, "bilan_net": 1260, "efficiency": 0.405,
                  "steam_hp": 530, "steam_mp": 400, "gta1": 520, "gta2": 500,
                  "gta3": 540, "gta_balance": 0.02,
                  "debit_adm_gta1": 175, "debit_adm_gta2": 175, "debit_adm_gta3": 175,
                  "rendement_gta1": 41.5, "rendement_gta2": 35.0, "rendement_gta3": 40.5}
        extreme = dict(normal, production=200, bilan_net=-500, efficiency=0.10,
                       steam_hp=100, gta1=50, gta2=50, gta3=100)
        r_norm = trained.detect(normal, "overall")
        r_ext = trained.detect(extreme, "overall")
        assert r_ext["score"] < r_norm["score"], "extreme state must be more anomalous"
        assert r_ext["is_anomaly"], "extreme state must be flagged"
        assert r_ext["severity"] in ("info", "warning", "critical")
        assert r_ext["cause"] is not None

    def test_detect_all_returns_only_anomalies(self, trained):
        collapse = {"production": 100, "bilan_net": -800, "efficiency": 0.05,
                    "steam_hp": 50, "steam_mp": 20, "gta1": 30, "gta2": 30, "gta3": 40,
                    "gta_balance": 0.5, "rendement_gta1": 20, "rendement_gta2": 15,
                    "rendement_gta3": 18, "debit_adm_gta1": 40, "debit_adm_gta2": 40,
                    "debit_adm_gta3": 40}
        results = trained.detect_all(collapse)
        assert all(r["is_anomaly"] for r in results)

    def test_detect_batch_adds_columns(self, trained, synth_df):
        out = trained.detect_batch(add_all_features(synth_df), "overall")
        for col in ("anomaly_score", "is_anomaly", "severity", "anomaly_cause"):
            assert col in out.columns


class TestAnomalyScanService:
    def test_scan_and_store_populates_table(self, seeded_db):
        from anomalies.anomaly_service import AnomalyScanService
        from database.database import query

        det = IsolationForestDetector(contamination=0.05)
        df = add_all_features(make_synthetic_df(300))
        det.train_all(df)

        # inject a slow efficiency drift so CUSUM has something to find
        df.loc[df.index[-60]:, "efficiency"] -= 0.04

        svc = AnomalyScanService(detector=det)
        summary = svc.scan_and_store(df, window_days=300)

        assert summary["count"] > 0
        assert summary["by_detector"]["isolation_forest"] > 0, \
            "contamination=5% must flag some points"
        assert summary["by_detector"]["cusum"] > 0, \
            "injected efficiency drift must be caught"

        rows = query("SELECT * FROM anomalies ORDER BY timestamp")
        assert len(rows) == summary["count"]
        assert all(r["severity"] in ("info", "warning", "critical") for r in rows)
        assert all(r["score"] is not None and r["score"] <= 0 for r in rows)
        # uniform timestamp format (mixed formats break the dashboard parsing)
        assert all(":" in str(r["timestamp"]) for r in rows if r["timestamp"])

        # rescan replaces (not duplicates) the derived table
        svc.scan_and_store(df, window_days=300)
        assert query("SELECT COUNT(*) AS c FROM anomalies")[0]["c"] == summary["count"]
