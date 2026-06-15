"""
routes.py — All REST endpoints.

GET  /health
GET  /realtime
GET  /predictions
GET  /anomalies
GET  /causal-graph
POST /recommend
GET  /shap
POST /simulate
POST /chat
POST /train
GET  /alerts
POST /alerts/{id}/acknowledge
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime

from database.database                       import query
from database.models                         import (
    RecommendationRequest, SimulationRequest,
    ChatRequest, TrainingStatus,
)
from forecasting.predictor_service           import PredictorService
from causal.pcmci_engine                     import PCMCIEngine
from causal.dag_builder                      import to_cytoscape_elements, build_dag_from_links
from recommendations.recommendation_engine  import RecommendationEngine
from explainability.shap_explainer           import SHAPExplainer
from digital_twin.simulator                  import DigitalTwinSimulator
    # from chatbot.rag_agent                       import RAGAgent
from data_pipeline.preprocessing             import clean_dataframe
from data_pipeline.feature_engineering       import add_all_features

import pandas as pd
import numpy as np

router = APIRouter()

# ── Lazy singletons (instantiated on first request) ────────────────────────────
_predictor   : PredictorService      | None = None
_rec_engine  : RecommendationEngine  | None = None
_digital_twin: DigitalTwinSimulator  | None = None
# _rag         : RAGAgent              | None = None
_pcmci       : PCMCIEngine           | None = None
_shap        : SHAPExplainer         | None = None
_training_status: dict = {"status": "idle", "progress": 0, "phase": "none"}


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = PredictorService()
    return _predictor


def _get_rec():
    global _rec_engine
    if _rec_engine is None:
        _rec_engine = RecommendationEngine()
    return _rec_engine


def _get_dt():
    global _digital_twin
    if _digital_twin is None:
        _digital_twin = DigitalTwinSimulator()
    return _digital_twin


# def _get_rag():
#     global _rag
#     if _rag is None:
#         _rag = RAGAgent()
#     return _rag


def _get_df():
    rows = query("SELECT * FROM historical_data ORDER BY timestamp DESC LIMIT 2000")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = clean_dataframe(df)
    df = add_all_features(df)
    return df


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"}


# ── Realtime ───────────────────────────────────────────────────────────────────

@router.get("/realtime")
def realtime(request: Request):
    reading = getattr(request.app.state, "latest_reading", {})
    if not reading:
        return {"status": "no_data", "reading": {}}
    return {"status": "ok", "reading": reading}


# ── Predictions ────────────────────────────────────────────────────────────────

@router.get("/predictions")
def predictions(target: str = "bilan_net", horizon: str = "24h"):
    try:
        return _get_predictor().predict(target, horizon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/predictions/all")
def predictions_all():
    return {"predictions": _get_predictor().get_recent(50)}


# ── Anomalies ──────────────────────────────────────────────────────────────────

@router.get("/anomalies")
def anomalies(limit: int = 50, severity: str | None = None):
    if severity:
        rows = query("SELECT * FROM anomalies WHERE severity=? ORDER BY timestamp DESC LIMIT ?",
                     [severity, limit])
    else:
        rows = query("SELECT * FROM anomalies ORDER BY timestamp DESC LIMIT ?", [limit])
    return {"anomalies": rows, "count": len(rows)}


# ── Causal graph ───────────────────────────────────────────────────────────────

@router.get("/causal-graph")
def causal_graph():
    global _pcmci
    if _pcmci is None:
        _pcmci = PCMCIEngine()
    df = _get_df()
    if df.empty:
        raise HTTPException(400, "No historical data")
    result = _pcmci.run(df)
    G      = build_dag_from_links(result)
    return {
        "nodes":    result["nodes"],
        "edges":    result["edges"],
        "cytoscape": to_cytoscape_elements(G),
    }


# ── Recommendations ────────────────────────────────────────────────────────────

@router.post("/recommend")
def recommend(req: RecommendationRequest, request: Request):
    state  = np.array(req.state, dtype=np.float32) if req.state else None
    df     = _get_df()
    engine = _get_rec()
    return engine.recommend(state=state, df=df if not df.empty else None)


# ── SHAP ───────────────────────────────────────────────────────────────────────

@router.get("/shap")
def shap_explain(target: str = "bilan_net", horizon: str = "24h"):
    global _shap
    predictor = _get_predictor().predictor
    if _shap is None:
        _shap = SHAPExplainer(predictor)
    df = _get_df()
    if df.empty:
        raise HTTPException(400, "No historical data")
    try:
        result = _shap.explain(df, target, horizon)
        return result
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Digital Twin simulation ────────────────────────────────────────────────────

@router.post("/simulate")
def simulate(req: SimulationRequest):
    try:
        return _get_dt().simulate(req.variable, req.change_percent, req.duration_hours)
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Chat ───────────────────────────────────────────────────────────────────────

@router.post("/chat")
def chat(req: ChatRequest):
    # return _get_rag().answer(req.question, req.context_window)
    pass


# ── Alerts ─────────────────────────────────────────────────────────────────────

@router.get("/alerts")
def get_alerts(limit: int = 50, level: str | None = None):
    alerts_svc = AlertService if False else None   # imported lazily below
    rows = query("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", [limit])
    return {"alerts": rows, "count": len(rows)}


@router.post("/alerts/{alert_id}/acknowledge")
def ack_alert(alert_id: int, request: Request):
    from database.database import execute
    execute("UPDATE alerts SET acknowledged=1 WHERE id=?", [alert_id])
    return {"status": "acknowledged", "id": alert_id}


# ── Training ───────────────────────────────────────────────────────────────────

@router.post("/train/phase1")
def train_phase1(bg: BackgroundTasks):
    """Kick off Phase-1 training (XGBoost + Isolation Forest) in background."""
    def _run():
        global _training_status
        _training_status = {"status": "running", "progress": 0, "phase": "phase1"}
        try:
            _get_predictor().train_all()
            _training_status = {"status": "done", "progress": 100, "phase": "phase1"}
        except Exception as exc:
            _training_status = {"status": "error", "progress": 0, "phase": "phase1",
                                 "detail": str(exc)}
    bg.add_task(_run)
    return {"status": "started", "phase": "phase1"}


@router.post("/train/phase2")
def train_phase2(episodes: int = 500, bg: BackgroundTasks = None):
    """Kick off Phase-2 DQN training in background."""
    def _run():
        global _training_status
        _training_status = {"status": "running", "progress": 0, "phase": "phase2"}
        try:
            from rl.trainer import Trainer
            df  = _get_df()
            t   = Trainer(data=df if not df.empty else None, episodes=episodes)
            t.train(callback=lambda ep, r, l: None)
            _training_status = {"status": "done", "progress": 100, "phase": "phase2"}
        except Exception as exc:
            _training_status = {"status": "error", "phase": "phase2", "detail": str(exc)}
    if bg:
        bg.add_task(_run)
    return {"status": "started", "phase": "phase2", "episodes": episodes}


@router.get("/train/status")
def train_status():
    return _training_status