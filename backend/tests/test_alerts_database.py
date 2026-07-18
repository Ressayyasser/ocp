"""Tests for AlertService (F25) and SQLite persistence (F01 — WAL mode)."""

import os
import sqlite3

from alerts.alert_service import AlertService, _THRESHOLDS, _SCADA_THRESHOLDS
from database.database import get_connection, insert_many, query


class TestDatabase:
    def test_wal_mode_enabled(self):
        conn = get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode.lower() == "wal", "report requires SQLite WAL mode"

    def test_historical_data_seeded(self, seeded_db):
        rows = query("SELECT COUNT(*) AS n FROM historical_data")
        assert rows[0]["n"] >= 200

    def test_insert_and_query_roundtrip(self, seeded_db):
        before = query("SELECT COUNT(*) AS n FROM historical_data")[0]["n"]
        insert_many("historical_data",
                    [{"timestamp": "2026-01-01 00:00:00", "production": 1500.0,
                      "bilan_net": 1200.0, "year": 2026}])
        after = query("SELECT COUNT(*) AS n FROM historical_data")[0]["n"]
        assert after == before + 1


class TestAlertService:
    def test_threshold_check_flags_critical_vibration(self):
        svc = AlertService()
        hits = svc._check_thresholds({"vibration_gta2": 5.0}, _SCADA_THRESHOLDS)
        assert hits, "vibration 5.0 mm/s must trip the 4.5 mm/s CRITICAL threshold"
        assert any(h["level"] == "CRITICAL" for h in hits)

    def test_threshold_check_flags_low_production(self):
        svc = AlertService()
        hits = svc._check_thresholds({"production": 800.0}, _THRESHOLDS)
        assert any(h["level"] == "CRITICAL" for h in hits)

    def test_nominal_reading_raises_nothing(self):
        svc = AlertService()
        nominal = {"production": 1560.0, "efficiency": 0.405, "bilan_net": 1200.0,
                   "steam_hp": 530.0, "rendement_gta1": 41.5}
        assert svc._check_thresholds(nominal, _THRESHOLDS) == []

    def test_manual_alert_lifecycle(self, seeded_db):
        svc = AlertService()
        created = svc.create_manual("WARNING", "test alert — pytest", source="pytest")
        assert created["level"] == "WARNING"
        recent = svc.get_recent(limit=10)
        assert any(a["message"] == "test alert — pytest" for a in recent)
        target = next(a for a in recent if a["message"] == "test alert — pytest")
        svc.acknowledge(target["id"])
        rows = query("SELECT acknowledged FROM alerts WHERE id = ?", [target["id"]])
        assert rows and rows[0]["acknowledged"]

    def test_subscribe_broadcasts_new_alerts(self, seeded_db):
        svc = AlertService()
        received = []
        svc.subscribe(received.append)
        svc.create_manual("INFO", "broadcast check", source="pytest")
        assert received and received[0]["message"] == "broadcast check"
