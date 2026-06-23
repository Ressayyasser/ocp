"""
models.py — Pydantic v2 schemas for all API request / response bodies.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Historical ────────────────────────────────────────────────────────────────
class HistoricalRecord(BaseModel):
    timestamp:   str
    gta1:        Optional[float] = None
    gta2:        Optional[float] = None
    gta3:        Optional[float] = None
    gtaa:        Optional[float] = None
    gtab:        Optional[float] = None
    steam_hp:    Optional[float] = None
    steam_mp:    Optional[float] = None
    steam_bp:    Optional[float] = None
    production:  Optional[float] = None
    consumption: Optional[float] = None
    bilan_net:   Optional[float] = None
    efficiency:  Optional[float] = None
    pressure:    Optional[float] = None
    vibration:   Optional[float] = None
    temperature: Optional[float] = None
    
    # Ajustement : Intégration complète des variables physiques des turbines (GTA 1, 2, 3)
    debit_adm_gta1:    Optional[float] = None
    debit_adm_gta2:    Optional[float] = None
    debit_adm_gta3:    Optional[float] = None
    debit_sout_gta1:   Optional[float] = None
    debit_sout_gta2:   Optional[float] = None
    debit_sout_gta3:   Optional[float] = None
    debit_ext_gta1:    Optional[float] = None
    debit_ext_gta2:    Optional[float] = None
    debit_ext_gta3:    Optional[float] = None
    rendement_gta1:    Optional[float] = None
    rendement_gta2:    Optional[float] = None
    rendement_gta3:    Optional[float] = None
    pression_adm_gta1: Optional[float] = None
    pression_adm_gta2: Optional[float] = None
    pression_adm_gta3: Optional[float] = None
    temp_adm_gta1:     Optional[float] = None
    temp_adm_gta2:     Optional[float] = None
    temp_adm_gta3:     Optional[float] = None
    h_adm_gta1:        Optional[float] = None
    h_adm_gta2:        Optional[float] = None
    h_adm_gta3:        Optional[float] = None
    h_sout_gta1:       Optional[float] = None
    h_sout_gta2:       Optional[float] = None
    h_sout_gta3:       Optional[float] = None
    h_ext_gta1:        Optional[float] = None
    h_ext_gta2:        Optional[float] = None
    h_ext_gta3:        Optional[float] = None
    year:              Optional[int]   = None


# ── Realtime ──────────────────────────────────────────────────────────────────
class RealtimeData(BaseModel):
    timestamp:     str
    gta1:          float = 0
    gta2:          float = 0
    gta3:          float = 0
    gtaa:          float = 0
    gtab:          float = 0
    steam_hp:      float = 0
    steam_mp:      float = 0
    steam_bp:      float = 0
    production:    float = 0
    consumption:   float = 0
    bilan_net:     float = 0
    pressure:      float = 0
    vibration:     float = 0
    temperature:   float = 0
    efficiency:    float = 0
    anomaly_score: float = 0


# ── Predictions ───────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    variable:        str
    horizon:         str
    predicted_value: float
    confidence:      float
    timestamp:       str


# ── Anomalies ─────────────────────────────────────────────────────────────────
class AnomalyResponse(BaseModel):
    id:         int
    timestamp:  str
    severity:   str
    score:      float
    cause:      Optional[str] = None
    variable:   Optional[str] = None
    raw_value:  Optional[float] = None


# ── Recommendations ───────────────────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    state: Optional[List[float]] = None   # 9-dim state vector (optional)


class RecommendationResponse(BaseModel):
    action:            str
    action_index:      int
    confidence:        float
    expected_gain_mwh: float
    economic_gain_dh:  float
    shap_explanation:  Optional[str] = None


# ── Alerts ────────────────────────────────────────────────────────────────────
class AlertResponse(BaseModel):
    id:        int
    timestamp: str
    message:   str
    level:     str
    source:    Optional[str] = None


# ── Causal graph ──────────────────────────────────────────────────────────────
class CausalLink(BaseModel):
    source_var: str
    target_var: str
    lag:        int
    strength:   float
    p_value:    Optional[float] = None


class CausalGraphResponse(BaseModel):
    nodes: List[str]
    edges: List[CausalLink]


# ── Simulation / Digital Twin ─────────────────────────────────────────────────
class SimulationRequest(BaseModel):
    variable:       str
    change_percent: float
    duration_hours: int = 24


class SimulationResponse(BaseModel):
    scenario:                    str
    predicted_bilan_change:      float
    predicted_production_change: float
    predicted_efficiency_change: float
    risk_score:                  float
    recommendation:              str


# ── Chat / RAG ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question:       str
    history: List[dict] = []
    context_hours: int = 24


class ChatResponse(BaseModel):
    answer:     str
    sources:    List[str] = []
    confidence: float = 0.0


# ── Health / RUL ──────────────────────────────────────────────────────────────
class HealthScore(BaseModel):
    component:    str
    health_score: float
    rul_days:     Optional[float] = None


# ── Training status ───────────────────────────────────────────────────────────
class TrainingStatus(BaseModel):
    phase:              str
    progress:           float
    episodes_completed: int   = 0
    current_reward:     float = 0.0
    status:             str   = "idle"