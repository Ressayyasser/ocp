"""Smoke tests for the FastAPI application (F14) — endpoints from Annexe D."""

import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

try:
    from api.main import app
    _IMPORT_ERROR = None
except Exception as exc:          # pragma: no cover — env-specific
    app = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    app is None, reason=f"api.main import failed: {_IMPORT_ERROR}")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:    # runs lifespan (DB init + SCADA simulator)
        yield c


class TestEndpoints:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_alerts_listing(self, client):
        r = client.get("/alerts")
        assert r.status_code == 200
        assert isinstance(r.json(), (list, dict))

    def test_rul_summary_under_500ms(self, client):
        """NFR — /rul/summary must answer well below the 500 ms SLA."""
        t0 = time.perf_counter()
        r = client.get("/rul/summary")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200
        assert elapsed_ms < 500, f"/rul/summary took {elapsed_ms:.0f} ms"

    def test_simulate_endpoint(self, client):
        r = client.post("/simulate", json={
            "variable": "gta1", "change_percent": 15, "duration_hours": 24})
        assert r.status_code == 200
        body = r.json()
        assert "predicted_bilan_change" in body
        assert "risk_score" in body

    def test_historical_data(self, client):
        r = client.get("/data/historical")
        assert r.status_code == 200

    def test_daily_gta_telemetry(self, client):
        r = client.get("/data/daily/GTA1")
        assert r.status_code == 200
        body = r.json()
        assert body["gta"] == "GTA1"
        assert body["count"] > 0, "daily per-GTA fallback must return records"
        last = body["data"][-1]
        for key in ("date", "adm_debit", "rendement", "puissance_mw", "energie_mwh"):
            assert key in last
        assert last["adm_debit"] is not None
        assert client.get("/data/daily/GTA9").status_code == 400
