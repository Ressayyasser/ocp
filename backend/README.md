# OCP Cogeneration AI Backend

## Architecture
```
FastAPI  →  SCADA Simulator  →  Anomaly Detection  →  Alerts
                            →  Forecasting (XGBoost)
                            →  PCMCI Causal Analysis
                            →  Double DQN Agent
                            →  SHAP Explainability
                            →  Digital Twin
                            →  RAG Chatbot
```

## Quick Start

```bash
cd backend
pip install -r requirements.txt

# 1. Put your Excel files in backend/data/
# 2. Run full training pipeline
python training_pipeline.py --phase all --data_dir data

# 3. Start the API
uvicorn api.main:app --reload --port 8000
```

## Tests

```bash
cd backend
python -m pytest tests                    # full suite (unit + API smoke tests)
python -m pytest tests --cov=. -q         # with coverage report (target ≥ 65 %)
```

Tests run against a temporary SQLite database and a temporary model
directory (`DB_PATH` / `MODEL_DIR` env vars) — they never touch
`cogen_ocp.db` or `models/`.

## Docker

```bash
# From the repository root
docker compose up --build
# Backend  → http://localhost:8000   (OpenAPI docs at /docs)
# Frontend → http://localhost:8050
# Optional local LLM (chatbot): docker compose exec ollama ollama pull qwen2.5:7b
```

## API Endpoints

| Method | Path               | Description                        |
|--------|--------------------|------------------------------------|
| GET    | /health            | Health check                       |
| GET    | /realtime          | Latest SCADA reading               |
| GET    | /predictions       | XGBoost forecast                   |
| GET    | /anomalies         | Recent anomalies                   |
| GET    | /causal-graph      | PCMCI DAG (Cytoscape format)       |
| POST   | /recommend         | DQN recommendation                 |
| GET    | /shap              | SHAP feature importance            |
| POST   | /simulate          | Digital Twin what-if               |
| POST   | /chat              | RAG chatbot                        |
| GET    | /alerts            | Alert history                      |
| POST   | /train/phase1      | Train forecasters + detectors      |
| POST   | /train/phase2      | Train DQN agent                    |
| GET    | /train/status      | Training progress                  |
| WS     | /ws/realtime       | Live SCADA stream                  |
| WS     | /ws/alerts         | Live alert stream                  |

## Development Order (PFE Roadmap)

1. `excel_loader.py` → load GTA data into SQLite
2. `scada_simulator/` → real-time data feed
3. `pcmci_engine.py` → causal discovery
4. `isolation_forest_detector.py` → anomaly detection
5. `xgboost_predictor.py` → forecasting
6. `rl/` → Double DQN agent
7. `recommendation_engine.py` → prescriptive output
8. `shap_explainer.py` → explainability
9. Dash frontend (separate `frontend/` folder)
10. `rag_agent.py` → chatbot
11. `digital_twin/` → what-if simulator
12. `rul/rul_engine.py` → RUL (Phase 3 enhancement)

## Directory Structure
```
backend/
├── api/               FastAPI routes, WebSocket
├── data_pipeline/     Excel loader, cleaning, feature engineering
├── forecasting/       XGBoost multi-target predictor
├── anomalies/         Isolation Forest + CUSUM
├── causal/            PCMCI + Granger + DAG builder
├── rl/                Environment, DQN, trainer, replay buffer
├── explainability/    SHAP TreeExplainer
├── recommendations/   Recommendation engine
├── digital_twin/      What-if simulator + Monte Carlo
├── chatbot/           RAG agent (Ollama qwen2.5:7b / rule-based fallback)
├── alerts/            Alert service with WebSocket broadcast
├── rul/               RUL engine (Phase 3)
├── scada_simulator/   Real-time sensor simulator + fault injection
├── database/          SQLite schema + Pydantic models
└── training_pipeline.py  End-to-end training orchestrator
``