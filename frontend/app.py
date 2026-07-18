"""
app.py — Dash multi-page application entry point.
Connects to the FastAPI backend at API_BASE_URL.
"""

import os

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from utils.alert_toast import register_alert_toast

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="OCP Cogeneration AI Platform",
)
server = app.server  # for gunicorn

# ── Sidebar navigation ────────────────────────────────────────────────────────
sidebar = dbc.Nav(
    [
        html.Div([
            html.Img(src="/assets/ocp_logo.svg", height="56px",
                     style={"marginBottom": "8px"},
                     className="d-block mx-auto"),
            html.H6("OCP Cogénération", className="text-center text-light fw-bold mb-0"),
            html.Small("AI Platform v1.0", className="text-center text-muted d-block mb-3"),
        ], className="p-3 border-bottom border-secondary"),

        dbc.NavLink([html.I(className="bi bi-speedometer2 me-2"), "Monitoring"],
                    href="/", active="exact"),
        dbc.NavLink([html.I(className="bi bi-fan me-2"), "GTA"],
                    href="/gta", active="exact"),
        dbc.NavLink([html.I(className="bi bi-cpu me-2"), "Digital Twin"],
                    href="/digital-twin", active="exact"),
        dbc.NavLink([html.I(className="bi bi-graph-up-arrow me-2"), "Prédiction"],
                    href="/prediction", active="exact"),
        dbc.NavLink([html.I(className="bi bi-exclamation-triangle me-2"), "Anomalies"],
                    href="/anomalies", active="exact"),
        dbc.NavLink([html.I(className="bi bi-diagram-3 me-2"), "DAG Causal"],
                    href="/dag", active="exact"),
        dbc.NavLink([html.I(className="bi bi-gear me-2"), "RUL & Health"],
                    href="/rul", active="exact"),
        dbc.NavLink([html.I(className="bi bi-robot me-2"), "Agent RL"],
                    href="/rl", active="exact"),
        dbc.NavLink([html.I(className="bi bi-bar-chart-steps me-2"), "SHAP"],
                    href="/shap", active="exact"),
        dbc.NavLink([html.I(className="bi bi-chat-dots me-2"), "Assistant IA"],
                    href="/chatbot", active="exact"),
    ],
    vertical=True,
    pills=True,
    className="flex-column px-2 py-2",
    style={"gap": "4px"},
)

app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [sidebar],
                    width=2,
                    className="min-vh-100 bg-dark border-end border-secondary",
                    style={"position": "sticky", "top": 0, "height": "100vh",
                           "overflowY": "auto"},
                ),
                # Page content
                dbc.Col(
                    dash.page_container,
                    width=10,
                    className="p-4",
                    style={"minHeight": "100vh", "backgroundColor": "#1a1a2e"},
                ),
            ],
            className="g-0",
        ),
    ],
    fluid=True,
    className="p-0",
)

# Mount the app-wide real-time alert toast (WebSocket driven)
register_alert_toast(app, api_base=os.environ.get("API_BASE_URL", "http://localhost:8000"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)