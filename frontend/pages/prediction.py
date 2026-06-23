"""
pages/prediction.py — Multi-horizon forecasting results page.
Shows XGBoost predictions for production, bilan, efficiency.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import get_predictions, get_historical, get_all_predictions

dash.register_page(__name__, path="/prediction", name="Prédiction", title="Prédiction")

VARIABLES = ["production", "bilan_net", "efficiency", "steam_hp"]
HORIZONS  = ["1d", "7d", "30d"]

layout = html.Div([
    html.H3("📈 Prédictions Multi-Horizons — XGBoost", className="text-light fw-bold mb-4"),

    # Controls
    dbc.Row([
        dbc.Col([
            html.Label("Variable", className="text-muted small"),
            dcc.Dropdown(
                id="pred-variable",
                options=[{"label": v.replace("_", " ").title(), "value": v} for v in VARIABLES],
                value="production",
                clearable=False,
                className="dash-dark-dropdown",
                style={"backgroundColor": "#2a2a3e", "color": "#fff"},
            ),
        ], width=4),
        dbc.Col([
            html.Label("Horizon", className="text-muted small"),
            dcc.Dropdown(
                id="pred-horizon",
                options=[{"label": h, "value": h} for h in HORIZONS],
                value="7d",
                clearable=False,
                style={"backgroundColor": "#2a2a3e", "color": "#fff"},
            ),
        ], width=3),
        dbc.Col([
            html.Label("Historique (jours)", className="text-muted small"),
            dcc.Slider(id="hist-days", min=7, max=90, step=7, value=30,
                       marks={7:"7j", 30:"30j", 60:"60j", 90:"90j"},
                       className="mt-2"),
        ], width=5),
    ], className="mb-4"),

    dbc.Button("🔄 Actualiser", id="pred-refresh", color="warning",
               size="sm", className="mb-3"),

    # Main chart
    dbc.Card([
        dbc.CardHeader("Historique + Prévision", className="text-light"),
        dbc.CardBody(dcc.Graph(id="pred-chart", style={"height": "380px"})),
    ], className="bg-dark border-secondary mb-4"),

    # Summary cards
    dbc.Row(id="pred-summary-row"),

    # All predictions table
    dbc.Card([
        dbc.CardHeader("Toutes les prévisions actives", className="text-light"),
        dbc.CardBody(html.Div(id="pred-table")),
    ], className="bg-dark border-secondary mt-4"),
])


@callback(
    Output("pred-chart", "figure"),
    Output("pred-summary-row", "children"),
    Input("pred-variable",  "value"),
    Input("pred-horizon",   "value"),
    Input("hist-days",      "value"),
    Input("pred-refresh",   "n_clicks"),
)
def update_prediction(variable, horizon, days, _):
    dark = dict(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                font=dict(color="#e0e0e0"), margin=dict(l=50, r=30, t=30, b=40))

    hist_resp = get_historical(days or 30)
    pred = get_predictions(variable, horizon)

    fig = go.Figure()

    # FIX: The API returns {"data": [...], "count": N}, so we must extract the "data" key
    hist_data = hist_resp.get("data", []) if isinstance(hist_resp, dict) else hist_resp

    # Historical trace
    if isinstance(hist_data, list) and hist_data:
        df = pd.DataFrame(hist_data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if variable in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df[variable],
                    mode="lines", name="Historique",
                    line=dict(color="#6bcb77", width=2),
                ))

    # Prediction trace
    cards = []
    if isinstance(pred, dict) and "error" not in pred:
        pred_val = pred.get("predicted_value")
        conf     = pred.get("confidence", 0)
        ts       = pred.get("timestamp", "")

        if pred_val is not None and isinstance(hist_data, list) and hist_data:
            last_ts = pd.to_datetime(hist_data[-1]["timestamp"])
            
            # Safely parse horizon string (e.g., "1d", "7d") to Timedelta
            try:
                delta = pd.Timedelta(horizon)
            except ValueError:
                delta = pd.Timedelta(days=7)

            fig.add_trace(go.Scatter(
                x=[last_ts, last_ts + delta],
                y=[hist_data[-1].get(variable, pred_val), pred_val],
                mode="lines+markers", name=f"Prévision {horizon}",
                line=dict(color="#f0c040", width=2, dash="dash"),
                marker=dict(size=10, symbol="diamond"),
            ))

        cards = [
            dbc.Col(dbc.Card([dbc.CardBody([
                html.Small("Valeur prédite", className="text-muted"),
                html.H4(f"{pred_val:.2f}" if pred_val is not None else "—",
                        className="text-warning fw-bold mb-0"),
            ])], className="bg-dark border-warning"), width=4),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.Small("Confiance", className="text-muted"),
                html.H4(f"{conf*100:.1f}%" if conf else "—",
                        className="text-info fw-bold mb-0"),
            ])], className="bg-dark border-info"), width=4),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.Small("Horizon", className="text-muted"),
                html.H4(horizon, className="text-success fw-bold mb-0"),
            ])], className="bg-dark border-success"), width=4),
        ]

    fig.update_layout(**dark, legend=dict(orientation="h", y=1.1),
                      yaxis_title=variable.replace("_", " ").title())
    return fig, cards


@callback(
    Output("pred-table", "children"),
    Input("pred-refresh", "n_clicks"),
)
def update_all_preds(_):
    all_preds_resp = get_all_predictions()
    if isinstance(all_preds_resp, dict) and "error" in all_preds_resp:
        return html.P(f"Erreur: {all_preds_resp['error']}", className="text-danger")
        
    # FIX: The API returns {"predictions": [...]}, so we extract the key
    all_preds = all_preds_resp.get("predictions", []) if isinstance(all_preds_resp, dict) else all_preds_resp
    
    if not all_preds:
        return html.P("Aucune prévision disponible. (Assurez-vous d'avoir lancé l'entraînement)", className="text-muted")

    rows = []
    for p in (all_preds if isinstance(all_preds, list) else []):
        rows.append(html.Tr([
            html.Td(p.get("variable", "—")),
            html.Td(p.get("horizon",  "—")),
            html.Td(f"{p.get('predicted_value', 0):.2f}"),
            html.Td(f"{p.get('confidence', 0)*100:.1f}%"),
            html.Td(str(p.get("timestamp", "—"))[:16]),
        ]))

    return dbc.Table(
        [html.Thead(html.Tr([html.Th(c) for c in
                             ["Variable", "Horizon", "Valeur", "Confiance", "Horodatage"]])),
         html.Tbody(rows)],
        bordered=True, hover=True, size="sm",
        className="table-dark"
    )