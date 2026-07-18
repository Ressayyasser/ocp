import os
import requests
import warnings
# Suppress the annoying sklearn parallel delayed warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
# 🔥 FIX: Increased timeout from 10s to 120s. 
# Running 500 trials of DB queries + feature engineering takes time!
TIMEOUT = 120  # seconds
SCADA_API_URL = os.environ.get("SCADA_API_URL", "http://127.0.0.1:8051")


def _get(path: str, params: dict = None) -> dict | list:
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def _post(path: str, payload: dict = None) -> dict:
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=payload or {}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}

def _get_scada(path: str, params: dict = None) -> dict | list:
    """Appelle l'API du simulateur SCADA sur le port 8051."""
    try:
        r = requests.get(f"{SCADA_API_URL}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}
    
# ── Endpoints ─────────────────────────────────────────────────────────────────

def get_health():
    return _get("/health")

def get_current_data():
    return _get("/realtime")

def get_historical(days: int = 30):
    return _get("/data/historical", params={"days": days})

def get_predictions(target: str = "bilan_net", horizon: str = "24h"):
    return _get("/predictions", params={"target": target, "horizon": horizon})

def get_all_predictions():
    return _get("/predictions/all")

def get_anomalies(limit: int = 50, severity: str = None):
    params = {"limit": limit}
    if severity:
        params["severity"] = severity
    return _get("/anomalies", params=params)

def rescan_anomalies(window_days: int = 365):
    """Re-run the batch Isolation Forest + CUSUM sweep on the backend."""
    return _post(f"/anomalies/scan?window_days={window_days}")

def get_alerts(limit: int = 50):
    return _get("/alerts", params={"limit": limit})

def acknowledge_alert(alert_id: int):
    return _post(f"/alerts/{alert_id}/acknowledge")

def get_causal_graph():
    return _get("/causal-graph")

def get_recommendations(state: list = None):
    return _post("/recommend", payload={"state": state})

def get_shap_explanation(target: str = "bilan_net", horizon: str = "24h"):
    return _get("/shap", params={"target": target, "horizon": horizon})

def get_train_status():
    return _get("/train/status")

def trigger_training_phase1():
    return _post("/train/phase1")

def trigger_training_phase2(episodes: int = 500):
    return _post(f"/train/phase2?episodes={episodes}")

def run_simulation(variable: str, change_percent: float, duration_hours: int):
    payload = {
        "variable": variable,
        "change_percent": change_percent,
        "duration_hours": duration_hours
    }
    return _post("/simulate", payload=payload)

def compare_scenarios(scenarios: list):
    """Compare multiple what-if scenarios."""
    return _post("/simulate/compare", payload={"scenarios": scenarios})

# 🔥 FIX: Simplified to use the working synchronous backend endpoint
def run_monte_carlo(variable: str, change_percent: float, duration_hours: int, n_trials: int = 500):
    """Run Monte Carlo simulation for uncertainty quantification."""
    return _post(f"/simulate/monte-carlo?n_trials={n_trials}", payload={
        "variable": variable,
        "change_percent": change_percent,
        "duration_hours": duration_hours
    })

# ── Dash Page Compatibility Wrappers ──────────────────────────────────────────

def get_rl_status():
    return get_train_status()

def trigger_training():
    return trigger_training_phase1()

def get_rl_episodes(limit: int = 200):
    return _get("/rl/episodes", params={"limit": limit})

def ask_chatbot(question: str, context_hours: int = 24):
    """Send a question to the RAG chatbot backend."""
    return _post("/chat", payload={"question": question, "context_hours": context_hours})

# ── NOUVEAU : Données live par GTA depuis le simulateur (port 8051) ───────────

def get_live_gta_data(gta: str) -> dict:
    """Récupère les données live d'un GTA depuis le simulateur (port 8051)."""
    return _get_scada(f"/api/live/{gta}")

def get_all_live_gta_data() -> dict:
    """Récupère les données live des 3 GTA en une seule fois."""
    results = {}
    for gta in ["GTA1", "GTA2", "GTA3"]:
        results[gta] = get_live_gta_data(gta)
    return results