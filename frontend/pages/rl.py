"""
pages/rl.py — Reinforcement Learning agent monitoring page.
Shows training curves, reward history, episode stats, and recommendations.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import (
    get_train_status,
    trigger_training_phase1,
    trigger_training_phase2,
    get_recommendations,
    get_rl_episodes,
)

dash.register_page(__name__, path="/rl", name="Agent RL", title="Agent RL")

layout = html.Div([
    html.H3("🤖 Agent RL — Double DQN & Jumeau Numérique", className="text-light fw-bold mb-4"),

    dcc.Interval(id="rl-interval", interval=10_000, n_intervals=0),

    # Status + Actions row
    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.Small("Statut de l'Entraînement", className="text-muted d-block"),
                html.H5(id="rl-status-text", className="text-info fw-bold mb-0"),
            ])], className="bg-dark border-info"),
        ], width=3),
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.Small("Épisodes entraînés", className="text-muted d-block"),
                html.H5(id="rl-episodes-count", className="text-warning fw-bold mb-0"),
            ])], className="bg-dark border-warning"),
        ], width=3),
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.Small("Récompense moyenne", className="text-muted d-block"),
                html.H5(id="rl-avg-reward", className="text-success fw-bold mb-0"),
            ])], className="bg-dark border-success"),
        ], width=3),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("▶ Phase 1 (Causalité)", id="rl-btn-p1", color="info", size="sm", className="w-100"),
                dbc.Button("🚀 Phase 2 (RL Deep)", id="rl-btn-p2", color="danger", size="sm", className="w-100"),
            ], vertical=True, className="w-100"),
            html.Div(id="rl-train-msg", className="small mt-1 text-center"),
        ], width=3),
    ], className="mb-4"),

    # Training curves
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Courbe de récompense par épisode", className="text-light"),
                dbc.CardBody(dcc.Graph(id="rl-reward-chart", style={"height": "300px"})),
            ], className="bg-dark border-secondary"),
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Bilan moyen par épisode (MWh)", className="text-light"),
                dbc.CardBody(dcc.Graph(id="rl-bilan-chart", style={"height": "300px"})),
            ], className="bg-dark border-secondary"),
        ], width=5),
    ], className="mb-4"),

    # Recommendations
    dbc.Card([
        dbc.CardHeader("💡 Recommandations prescriptives (Agent RL + SHAP)", className="text-light"),
        dbc.CardBody(html.Div(id="rl-recommendations")),
    ], className="bg-dark border-secondary mb-4"),

    # Episode table
    dbc.Card([
        dbc.CardHeader("Journal des épisodes d'entraînement", className="text-light"),
        dbc.CardBody(html.Div(id="rl-episode-table", style={"maxHeight": "300px", "overflowY": "auto"})),
    ], className="bg-dark border-secondary"),
])


@callback(
    Output("rl-status-text",     "children"),
    Output("rl-episodes-count", "children"),
    Output("rl-avg-reward",      "children"),
    Output("rl-reward-chart",   "figure"),
    Output("rl-bilan-chart",    "figure"),
    Output("rl-episode-table",  "children"),
    Output("rl-recommendations","children"),
    Input("rl-interval", "n_intervals"),
)
def update_rl(n):
    dark = dict(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#2a2a4e"),
        yaxis=dict(gridcolor="#2a2a4e"),
    )

    # Get training status
    train_status = get_train_status()
    if isinstance(train_status, dict):
        status_text = train_status.get("status", "Inactif").upper()
    else:
        status_text = "DÉCONNECTÉ"

    # Get episodes from dedicated endpoint
    episodes_response = get_rl_episodes(limit=200)
    if isinstance(episodes_response, dict) and "episodes" in episodes_response:
        episodes = episodes_response["episodes"]
    else:
        episodes = []

    # Get recommendations
    recs_response = get_recommendations()
    if isinstance(recs_response, dict):
        # Backend returns single recommendation dict
        if "recommendation" in recs_response and recs_response["recommendation"]:
            recs = [recs_response["recommendation"]]
        elif "action" in recs_response:
            recs = [recs_response]
        else:
            recs = []
    elif isinstance(recs_response, list):
        recs = recs_response
    else:
        recs = []

    ep_count = "—"
    avg_reward = "—"
    fig_reward = go.Figure()
    fig_bilan = go.Figure()

    if isinstance(episodes, list) and len(episodes) > 0:
        df = pd.DataFrame(episodes).sort_values("episode")
        ep_count = str(len(df))

        if "total_reward" in df.columns:
            avg_reward = f"{df['total_reward'].mean():.1f}"
        else:
            avg_reward = "—"

        # Reward chart
        if "total_reward" in df.columns:
            fig_reward.add_trace(go.Scatter(
                x=df["episode"],
                y=df["total_reward"],
                mode="lines",
                name="Récompense totale",
                line=dict(color="#f0c040", width=2),
            ))
            if len(df) >= 5:
                fig_reward.add_trace(go.Scatter(
                    x=df["episode"],
                    y=df["total_reward"].rolling(10, min_periods=1).mean(),
                    mode="lines",
                    name="Moyenne mobile (10 ep)",
                    line=dict(color="#ff6b6b", width=2, dash="dash"),
                ))

        fig_reward.update_layout(
            **dark,
            yaxis_title="Récompense",
            xaxis_title="Épisode",
            legend=dict(orientation="h", y=1.1),
            height=300,
        )

        # Bilan chart
        if "avg_bilan" in df.columns:
            fig_bilan.add_trace(go.Scatter(
                x=df["episode"],
                y=df["avg_bilan"],
                mode="lines+markers",
                name="Bilan moyen",
                line=dict(color="#6bcb77", width=2),
                marker=dict(size=3),
            ))

        fig_bilan.update_layout(
            **dark,
            yaxis_title="Bilan Net (MWh)",
            xaxis_title="Épisode",
            height=300,
        )
    else:
        fig_reward.update_layout(**dark, title="En attente de données d'entraînement...", height=300)
        fig_bilan.update_layout(**dark, title="En attente de données d'entraînement...", height=300)

    # Episode table
    if isinstance(episodes, list) and len(episodes) > 0:
        rows = [
            html.Tr([
                html.Td(ep.get("episode", "—")),
                html.Td(
                    f"{ep.get('total_reward', 0):.2f}"
                    if ep.get("total_reward") is not None
                    else "—"
                ),
                html.Td(ep.get("steps", "—")),
                html.Td(
                    f"{ep.get('avg_bilan', 0):.1f}"
                    if ep.get("avg_bilan") is not None
                    else "—"
                ),
                html.Td(str(ep.get("trained_at", ""))[:16] if ep.get("trained_at") else "—"),
            ])
            for ep in episodes[-50:]
        ]

        ep_table = dbc.Table(
            [
                html.Thead(html.Tr([html.Th(c) for c in ["Épisode", "Récompense", "Steps", "Bilan Moy.", "Date"]])),
                html.Tbody(rows),
            ],
            bordered=True,
            hover=True,
            size="sm",
            className="table-dark table-sm",
        )
    else:
        ep_table = html.P(
            "Aucun épisode enregistré — En attente du démarrage du processus.",
            className="text-muted text-center py-3",
        )

    # Recommendations section
    rec_items = []
    if isinstance(recs, list) and len(recs) > 0:
        for r in recs[:5]:
            gain = r.get("expected_gain_mwh", 0)
            eco = r.get("economic_gain_dh", 0)
            conf = r.get("confidence")
            if conf is None:
                conf = 0
            
            rec_items.append(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Strong(r.get("action", "—"), className="text-warning"),
                                                html.P(
                                                    r.get("shap_explanation", "Aucune explication SHAP disponible.")[:120]
                                                    + "...",
                                                    className="text-muted small mt-1 mb-0",
                                                ),
                                            ],
                                            width=7,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [html.Strong(f"+{gain:.1f} MWh"), html.Small(" gain")],
                                                    className="text-success",
                                                ),
                                                html.Div(
                                                    [html.Strong(f"{eco:.0f} DH"), html.Small(" économie")],
                                                    className="text-info",
                                                ),
                                                dbc.Badge(
                                                    f"Conf: {conf*100:.0f}%", color="secondary", className="mt-1"
                                                ),
                                            ],
                                            width=5,
                                            className="text-end",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="bg-dark border-warning mb-2",
                )
            )
    else:
        rec_items = [
            html.P("Aucune recommandation prescriptive calculée pour le moment.", className="text-muted")
        ]

    return (status_text, ep_count, avg_reward, fig_reward, fig_bilan, ep_table, rec_items)


@callback(
    Output("rl-train-msg", "children"),
    Input("rl-btn-p1", "n_clicks"),
    Input("rl-btn-p2", "n_clicks"),
    prevent_initial_call=True,
)
def trigger_rl_training(n_p1, n_p2):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "rl-btn-p1":
        result = trigger_training_phase1()
        phase_label = "Phase 1 (Causalité)"
    else:
        result = trigger_training_phase2(episodes=500)
        phase_label = "Phase 2 (RL Deep)"

    if isinstance(result, dict) and "error" in result:
        return dbc.Badge(f"Erreur {phase_label}: {result['error']}", color="danger")

    return dbc.Badge(f"{phase_label} lancé avec succès ✓", color="success")