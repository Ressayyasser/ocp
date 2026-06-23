# routes.py
from __future__ import annotations
import sys, os
import sqlite3
import warnings
# Suppress the annoying sklearn parallel delayed warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from datetime import datetime

from database.database                       import query, get_connection
from database.models                         import (
    RecommendationRequest, SimulationRequest,
    ChatRequest, TrainingStatus,
)
from forecasting.predictor_service              import PredictorService
from causal.pcmci_engine                        import PCMCIEngine
from causal.dag_builder                         import to_cytoscape_elements, build_dag_from_links
from recommendations.recommendation_engine      import RecommendationEngine
from explainability.shap_explainer              import SHAPExplainer
from digital_twin.simulator                     import DigitalTwinSimulator
from digital_twin.monte_carlo                   import MonteCarloSimulator
from data_pipeline.preprocessing                import clean_dataframe
from data_pipeline.feature_engineering          import add_all_features
from rul.rul_engine                             import RULEngine
from chatbot.rag_agent                          import RAGAgent


import pandas as pd
import numpy as np

router = APIRouter()
rul_engine = RULEngine()

# ── Pydantic Models for new Digital Twin endpoints ─────────────────────────────
class MultiScenarioRequest(BaseModel):
    scenarios: list[dict]  # e.g., [{"variable": "gta3", "change_percent": 15}, ...]

# ── Lazy singletons (instantiated on first request) ────────────────────────────
_predictor   : PredictorService      | None = None
_rec_engine  : RecommendationEngine  | None = None
_digital_twin: DigitalTwinSimulator  | None = None
_pcmci       : PCMCIEngine           | None = None
_shap        : SHAPExplainer         | None = None
_mc_sim      : MonteCarloSimulator  | None = None 
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


def _get_df():
    rows = query("SELECT * FROM historical_data ORDER BY timestamp DESC LIMIT 2000")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = clean_dataframe(df)
    df = add_all_features(df)
    return df


def sanitize_for_json(obj):
    """Replaces NaN, Inf with None for JSON compatibility safely across DataFrames/Dicts"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, (float, np.floating)) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj

def _get_mc():
    global _mc_sim
    if _mc_sim is None:
        _mc_sim = MonteCarloSimulator(n_trials=500, seed=42)
    return _mc_sim

# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "version": "1.0.0"}

# ── Realtime ───────────────────────────────────────────────────────────────────

@router.get("/realtime")
def realtime(request: Request):
    reading = getattr(request.app.state, "latest_reading", {})
    if not reading:
        return {"status": "no_data", "reading": {}}
    return {"status": "ok", "reading": reading}

# ── Digital Twin: What-If & Monte Carlo Simulations ────────────────────────────

@router.post("/simulate")
def simulate(req: SimulationRequest):
    """Single what-if scenario simulation."""
    try:
        return sanitize_for_json(_get_dt().simulate(req.variable, req.change_percent, req.duration_hours))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/simulate/compare")
def compare_scenarios(req: MultiScenarioRequest):
    """Compare multiple what-if scenarios side-by-side."""
    try:
        dt = _get_dt()
        results = dt.compare_scenarios(req.scenarios)
        return sanitize_for_json(results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/simulate/monte-carlo")
def simulate_monte_carlo(req: SimulationRequest, n_trials: int = Query(500, ge=10, le=5000)):
    """
    Runs Monte Carlo simulations for a given what-if scenario to quantify uncertainty.
    Returns mean, std, P5, P95, and median for all numeric outputs.
    """
    try:
        dt = _get_dt()
        mc = MonteCarloSimulator(n_trials=n_trials)
        
        # Define the scenario function that MC will run N times
        def scenario_fn():
            return dt.simulate(req.variable, req.change_percent, req.duration_hours)
            
        # Get baseline df to estimate noise from historical variance
        df = _get_df()
        
        # Run Monte Carlo
        result = mc.run(scenario_fn, df=df)
        return sanitize_for_json(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# In your FastAPI backend
from fastapi import BackgroundTasks
from concurrent.futures import ThreadPoolExecutor
import uuid

# Store for job results
jobs = {}

def perform_monte_carlo_simulation(variable: str, change_percent: float, duration_hours: int, n_trials: int):
    """
    Core logic to run Monte Carlo simulation using the Digital Twin and MonteCarloSimulator.
    """
    # 1. Use the existing lazy-loaded Digital Twin instance
    dt = _get_dt()
    
    # 2. Initialize Monte Carlo simulator with the requested number of trials
    mc = MonteCarloSimulator(n_trials=n_trials)
    
    # 3. Define the scenario function that MC will run N times
    def scenario_fn():
        return dt.simulate(variable, change_percent, duration_hours)
        
    # 4. Get baseline historical dataframe to estimate noise from historical variance
    df = _get_df()
    
    # 5. Run Monte Carlo simulation
    result = mc.run(scenario_fn, df=df)
    return result

def run_monte_carlo_task(job_id: str, variable: str, change_percent: float, duration_hours: int, n_trials: int):
    """Background task to run Monte Carlo simulation."""
    try:
        # Your Monte Carlo simulation logic here
        result = perform_monte_carlo_simulation(variable, change_percent, duration_hours, n_trials)
        
        jobs[job_id] = {
            "status": "completed",
            "result": result
        }
    except Exception as e:
        jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }

@router.post("/simulate/monte-carlo/start")
async def start_monte_carlo(
    payload: dict,
    background_tasks: BackgroundTasks
):
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {"status": "running"}
    
    background_tasks.add_task(
        run_monte_carlo_task,
        job_id,
        payload["variable"],
        payload["change_percent"],
        payload["duration_hours"],
        payload.get("n_trials", 500)
    )
    
    return {"job_id": job_id, "status": "started"}

@router.get("/simulate/job/{job_id}/status")
async def get_job_status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})

@router.get("/simulate/job/{job_id}/result")
async def get_job_result(job_id: str):
    return jobs.get(job_id, {})

# ── Predictions ────────────────────────────────────────────────────────────────

@router.get("/predictions")
def predictions(target: str = "bilan_net", horizon: str = "24h"):
    try:
        return sanitize_for_json(_get_predictor().predict(target, horizon))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/predictions/all")
def predictions_all():
    return {"predictions": sanitize_for_json(_get_predictor().get_recent(50))}

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
        raise HTTPException(400, "No historical data available to compute graph")
    
    result = _pcmci.run(df)
    G = build_dag_from_links(result)
    return {
        "nodes": result["nodes"],
        "edges": result["edges"],
        "cytoscape": to_cytoscape_elements(G),
    }

# ── Recommendations ────────────────────────────────────────────────────────────

@router.post("/recommend")
def recommend(req: RecommendationRequest):
    """DQN prescriptive engine interface supporting state vector evaluation."""
    state = np.array(req.state, dtype=np.float32) if req.state else None
    df = _get_df()
    engine = _get_rec()
    try:
        res = engine.recommend(state=state, df=df if not df.empty else None)
        return sanitize_for_json(res)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/recommendations")
def get_recommendations():
    """Fetches recommendation mapping driven by the latest SCADA state row."""
    engine = _get_rec()
    df = _get_df()
    if df.empty:
        return {"recommendation": None}
    
    last_row = df.iloc[-1]
    # FIXED: Replaced literal '...' placeholder with dynamic columns filtering to protect runtime
    features = ['production', 'bilan_net', 'efficiency']
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        valid_features = df.select_dtypes(include=[np.number]).columns.tolist()[:3]
        
    state = last_row[valid_features].values
    return sanitize_for_json(engine.recommend(state=state.tolist(), df=df))

# ── SHAP ───────────────────────────────────────────────────────────────────────

@router.get("/shap")
def shap_explain(target: str = "bilan_net", horizon: str = "24h"):
    """Evaluates XGBoost feature vectors using TreeExplainer matrix calculations."""
    global _shap
    predictor_instance = _get_predictor().predictor
    if _shap is None:
        _shap = SHAPExplainer(predictor_instance)
    df = _get_df()
    if df.empty:
        raise HTTPException(400, "No historical data found for tracking explainer targets")
    try:
        result = _shap.explain(df, target, horizon)
        return sanitize_for_json(result)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


# FIXED: Renamed function signature to prevent namespace overwriting and infinite loop crash
@router.get("/explain/shap")
def shap_explain_alias(variable: str = "bilan_net", horizon: str = "24h"):
    """Dash Frontend compatibility route wrapper mapping variable parameters to targets."""
    return shap_explain(target=variable, horizon=horizon)

# ── Digital Twin simulation ────────────────────────────────────────────────────

@router.post("/simulate")
def simulate(req: SimulationRequest):
    try:
        return _get_dt().simulate(req.variable, req.change_percent, req.duration_hours)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


# @router.post("/chat")
# def chat(req: ChatRequest):
#     # Fallback response until rag_agent dependency chain is un-commented out
#     return {"response": "L'agent RAG est temporairement indisponible.", "context": []}

# ── Alerts ─────────────────────────────────────────────────────────────────────

@router.get("/alerts")
def get_alerts(limit: int = 50):
    rows = query("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", [limit])
    return {"alerts": rows, "count": len(rows)}


@router.post("/alerts/{alert_id}/acknowledge")
def ack_alert(alert_id: int):
    from database.database import execute
    execute("UPDATE alerts SET acknowledged=1 WHERE id=?", [alert_id])
    return {"status": "acknowledged", "id": alert_id}

# ── Training Pipelines ──────────────────────────────────────────────────────────

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
            _training_status = {"status": "error", "progress": 0, "phase": "phase1", "detail": str(exc)}
    bg.add_task(_run)
    return {"status": "started", "phase": "phase1"}


@router.post("/train/phase2")
def train_phase2(episodes: int = 50, bg: BackgroundTasks = None):
    """Kick off Phase-2 DQN training loops via background thread pools."""
    if bg is None:
        raise HTTPException(status_code=400, detail="BackgroundTasks manager context required")
        
    def _run():
        global _training_status
        _training_status = {"status": "running", "progress": 0, "phase": "phase2"}
        try:
            from rl.trainer import Trainer
            df = _get_df()
            t = Trainer(data=df if not df.empty else None, episodes=episodes)
            t.train(callback=lambda ep, r, l: None)
            _training_status = {"status": "done", "progress": 100, "phase": "phase2"}
        except Exception as exc:
            _training_status = {"status": "error", "phase": "phase2", "detail": str(exc)}
    bg.add_task(_run)
    return {"status": "started", "phase": "phase2", "episodes": episodes}


@router.get("/train/status")
def train_status():
    rows = query(
        "SELECT episode, total_reward, steps, avg_bilan FROM rl_episodes ORDER BY episode ASC"
    )
    episodes = [
        {
            "episode": r["episode"],
            "total_reward": r["total_reward"],
            "steps": r["steps"],
            "avg_bilan": r["avg_bilan"],
        }
        for r in rows
    ]
    return {
        **_training_status,
        "episodes": episodes,
    }

@router.get("/rl/status")
def rl_status():
    return _training_status


@router.get("/data/historical")
def get_historical(days: int = 30):
    """Provides cleaned historical matrices to Dash telemetry scopes without serialization faults."""
    try:
        df = _get_df()
        if df.empty:
            return {"data": [], "count": 0}
        
        n_rows = min(len(df), days * 48)  # Calculates estimation limits assuming 30-min entries
        data = df.tail(n_rows).replace([np.inf, -np.inf], np.nan).to_dict(orient="records")
        return sanitize_for_json({"data": data, "count": len(data)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Temporary test data
@router.get("/rl/episodes")
def get_rl_episodes(limit: int = 100):
    rows = query(
        "SELECT episode, total_reward, steps, avg_bilan FROM rl_episodes ORDER BY episode ASC LIMIT ?",
        [limit]
    )
    episodes = [
        {
            "episode": r["episode"],
            "total_reward": r["total_reward"],
            "steps": r["steps"],
            "avg_bilan": r["avg_bilan"],
        }
        for r in rows
    ]
    return {"episodes": episodes, "count": len(episodes)}

def get_latest_scada_reading(gta_type: str = "GTA1") -> dict:
    """
    Fetches the latest sensor reading from the simulation_data table 
    and maps it to the variables expected by the RUL Engine.
    """
    try:
        
        row = query("""
            SELECT vib1, vib2, adm_temp, adm_pression, rendement, adm_debit 
            FROM simulation_data 
            WHERE gta_type = ? 
            ORDER BY id DESC LIMIT 1
        """, (gta_type,))
        
        if not row:
            return None
            
        # Map DB columns to RUL Engine expected inputs
        # We use the max of vib1 and vib2 to be conservative for vibration stress
        return {
            "vibration": max(row["vib1"] or 0, row["vib2"] or 0),
            "temperature": row["adm_temp"],
            "pressure": row["adm_pression"],
            "efficiency": row["rendement"],
            "steam_hp": row["adm_debit"]  # HP Admission flow acts as the steam_hp proxy
        }
    except Exception as e:
        print(f"Database error fetching RUL reading: {e}")
        return None


@router.get("/rul/summary")
def get_rul_summary(hours_operated: float = Query(10000.0)):
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Try to get live data from simulation_data
    cursor.execute("""
        SELECT adm_temp, adm_pression, adm_debit, rendement, vib1, vib2
        FROM simulation_data 
        ORDER BY id DESC LIMIT 1
    """)
    row = cursor.fetchone()
    
    # 2. Fallback to historical_data if simulation_data is empty
    if not row:
        cursor.execute("""
            SELECT temp_adm_gta1 as adm_temp, pression_adm_gta1 as adm_pression, 
                   debit_adm_gta1 as adm_debit, rendement_gta1 as rendement,
                   vibration
            FROM historical_data 
            ORDER BY id DESC LIMIT 1
        """)
        row = cursor.fetchone()
        
    conn.close()
    
    if not row:
        return {"error": "No data found in simulation_data or historical_data tables."}
    
    # Convert sqlite3.Row to dict for safe .get() access
    row_dict = dict(row)
    
    # Safely extract vibration, falling back to 0.5 if missing
    vib1 = row_dict.get("vib1") or row_dict.get("vibration") or 0.5
    vib2 = row_dict.get("vib2") or 0.5
    
    # Map DB columns to the keys expected by RULEngine
    reading = {
        "vibration": max(float(vib1), float(vib2)),
        "temperature": float(row_dict.get("adm_temp") or 455.0),
        "pressure": float(row_dict.get("adm_pression") or 54.5),
        "steam_hp": float(row_dict.get("adm_debit") or 175.0),
        "efficiency": float(row_dict.get("rendement") or 40.5) / 100.0  # Convert % to fraction
    }
    
    # Calculate Health and RUL
    engine = RULEngine()
    health = engine.compute_health(reading, hours_operated)
    rul = engine.estimate_rul(health)
    
    most_critical = min(health, key=health.get)
    
    # Return EXACTLY the keys the frontend expects
    return {
        "status": "ok",
        "most_critical": most_critical,
        "health_scores": health,  # <--- FIXED KEY (was "health")
        "rul": rul,
        "reading": reading
    }

@router.get("/rul/debug")
def rul_debug():
    rows = query("SELECT COUNT(*) as count, gta_type FROM simulation_data GROUP BY gta_type")
    return {"simulation_data_count": rows}

# ── Chat ───────────────────────────────────────────────────────────────────────

@router.post("/chat")
def chat(req: ChatRequest):
    """Routes the user query to the local Ollama RAG Agent."""
    try:
        agent = RAGAgent()
        result = agent.answer(req.question, context_hours=req.context_hours)
        return {
            "response": result.get("answer", ""),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0),
            "model": result.get("model", "local")
        }
    except Exception as e:
        return {"response": f"Erreur lors de la génération: {str(e)}", "sources": [], "confidence": 0}