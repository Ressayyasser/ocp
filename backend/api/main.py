"""
main.py — FastAPI application entry point.

Run:  uvicorn backend.api.main:app --reload --port 8000
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database.database               import init_db
from api.routes                      import router
from api.websocket                   import ws_router
from scada_simulator.simulator       import SCADASimulator
from anomalies.isolation_forest_detector import IsolationForestDetector
from alerts.alert_service            import AlertService

# Shared singletons (accessible via app.state in routes)
scada     = SCADASimulator()
detector  = IsolationForestDetector()
alerts    = AlertService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    init_db()

    # Wire SCADA → anomaly detector → alert service
    def on_reading(reading: dict):
        app.state.latest_reading = reading
        try:
            # Isolation Forest anomaly detection → alerts
            anomalies = detector.detect_all(reading)
            for anom in anomalies:
                alerts.alert_from_anomaly(anom)
            # Threshold-based checks on live SCADA parameters
            # (HP steam flow/pressure, vibration, per-GTA admission, ...)
            alerts.check_scada_reading(reading)
        except Exception:
            pass

    scada.subscribe(on_reading)
    alerts.subscribe(lambda a: None)   # placeholder — replaced by WS in routes
    scada.start()

    app.state.scada    = scada
    app.state.detector = detector
    app.state.alerts   = alerts
    app.state.latest_reading = {}

    yield
    # ── Shutdown ───────────────────────────────────────────────────────────
    scada.stop()


app = FastAPI(
    title="OCP Cogeneration AI Platform",
    description="Predictive + Causal + Prescriptive AI for GTA cogeneration optimization",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ws_router)