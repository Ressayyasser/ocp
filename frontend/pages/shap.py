"""
pages/shap.py — SHAP Explainability page.
Shows feature importance, waterfall charts, and natural-language explanations.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import get_shap_explanation

dash.register_page(__name__, path="/shap", name="SHAP", title="SHAP Explainabilité")

VARIABLES = ["production", "bilan_net", "efficiency", "steam_hp"]

layout = html.Div([
    html.H3("🔍 Explicabilité — SHAP TreeExplainer", className="text-light fw-bold mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Variable cible", className="text-muted small"),
            dcc.Dropdown(
                id="shap-variable",
                options=[{"label": v.replace("_"," ").title(), "value": v} for v in VARIABLES],
                value="production",
                clearable=False,
                style={"backgroundColor": "#2a2a3e", "color": "#fff"},
            ),
        ], width=4),
        dbc.Col([
            dbc.Button("🔄 Calculer SHAP", id="shap-refresh", color="warning",
                       size="sm", className="mt-3"),
        ], width=3),
    ], className="mb-4"),

    # Main charts row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Importance globale des features (SHAP mean |value|)",
                               className="text-light"),
                dbc.CardBody(dcc.Graph(id="shap-importance-bar", style={"height": "380px"})),
            ], className="bg-dark border-secondary"),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Waterfall — contribution par feature (dernière prédiction)",
                               className="text-light"),
                dbc.CardBody(dcc.Graph(id="shap-waterfall", style={"height": "380px"})),
            ], className="bg-dark border-secondary"),
        ], width=6),
    ], className="mb-4"),

    # Feature interaction heatmap
    dbc.Card([
        dbc.CardHeader("Top 10 features — valeurs SHAP (heatmap)", className="text-light"),
        dbc.CardBody(dcc.Graph(id="shap-heatmap", style={"height": "320px"})),
    ], className="bg-dark border-secondary mb-4"),

    # Natural language explanation
    dbc.Card([
        dbc.CardHeader("📖 Explication en langage naturel", className="text-light"),
        dbc.CardBody(html.Div(id="shap-nlp-explanation")),
    ], className="bg-dark border-secondary"),
])


@callback(
    Output("shap-importance-bar", "figure"),
    Output("shap-waterfall",      "figure"),
    Output("shap-heatmap",        "figure"),
    Output("shap-nlp-explanation","children"),
    Input("shap-refresh",  "n_clicks"),
    Input("shap-variable", "value"),
)
def update_shap(_, variable):
    dark = dict(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                font=dict(color="#e0e0e0"), margin=dict(l=160, r=20, t=30, b=40))

    data = get_shap_explanation(variable)

    # ── Empty figures on error ─────────────────────────────────────────────────
    def empty_fig(msg="Aucune donnée SHAP disponible."):
        fig = go.Figure()
        fig.update_layout(**dark)
        fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=14, color="#aaa"))
        return fig

    if isinstance(data, dict) and "error" in data:
        msg = f"Erreur: {data['error']} — Entraînez le modèle XGBoost d'abord."
        return empty_fig(msg), empty_fig(msg), empty_fig(msg), \
               html.P(msg, className="text-danger")

    # Extract SHAP values
    feature_names  = data.get("feature_names", [])
    mean_abs_shap  = data.get("mean_abs_shap", [])   # list of floats
    shap_values    = data.get("shap_values", [])      # 2D list [samples x features]
    base_value     = data.get("base_value", 0)
    explanation    = data.get("explanation", "")

    if not feature_names:
        return empty_fig(), empty_fig(), empty_fig(), \
               html.P("Aucune donnée SHAP — lancez l'entraînement.", className="text-muted")

    # ── Importance bar ─────────────────────────────────────────────────────────
    sorted_pairs = sorted(zip(mean_abs_shap, feature_names), reverse=True)[:20]
    vals, feats  = zip(*sorted_pairs) if sorted_pairs else ([], [])
    colors       = ["#f0c040" if v > 0.2 else "#4d9de0" for v in vals]

    fig_bar = go.Figure(go.Bar(
        x=list(vals), y=list(feats),
        orientation="h",
        marker_color=colors,
    ))
    fig_bar.update_layout(**dark, xaxis_title="SHAP mean |value|",
                          yaxis=dict(autorange="reversed"))

    # ── Waterfall (last sample) ────────────────────────────────────────────────
    fig_wf = go.Figure()
    if shap_values:
        last = shap_values[-1]  # last sample
        top_idx  = sorted(range(len(last)), key=lambda i: abs(last[i]), reverse=True)[:10]
        wf_feats = [feature_names[i] for i in top_idx]
        wf_vals  = [last[i] for i in top_idx]
        wf_colors = ["#6bcb77" if v > 0 else "#ff6b6b" for v in wf_vals]
        fig_wf.add_trace(go.Bar(
            x=wf_vals, y=wf_feats, orientation="h",
            marker_color=wf_colors, name="SHAP contribution",
        ))
        fig_wf.add_vline(x=0, line_color="#fff", line_width=1)
        fig_wf.update_layout(**dark, xaxis_title="Contribution SHAP",
                              yaxis=dict(autorange="reversed"))
    else:
        fig_wf = empty_fig("Pas de valeurs SHAP individuelles.")

    # ── Heatmap (top 10 features x last 30 samples) ────────────────────────────
    fig_heat = go.Figure()
    if shap_values and len(shap_values) > 1:
        import numpy as np
        arr  = [[row[i] for i in range(len(feature_names))] for row in shap_values[-30:]]
        top10 = sorted(range(len(feature_names)),
                       key=lambda i: abs(mean_abs_shap[i]), reverse=True)[:10]
        z    = [[row[i] for i in top10] for row in arr]
        feat_labels = [feature_names[i] for i in top10]
        fig_heat.add_trace(go.Heatmap(
            z=list(map(list, zip(*z))),  # transpose: features x samples
            y=feat_labels,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="SHAP"),
        ))
        fig_heat.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font=dict(color="#e0e0e0"),
            margin=dict(l=160, r=20, t=20, b=40),
            xaxis_title="Derniers 30 échantillons",
        )
    else:
        fig_heat = empty_fig("Pas assez d'échantillons pour la heatmap.")

    # ── NLP explanation ────────────────────────────────────────────────────────
    if explanation:
        nlp = [html.P(explanation, className="text-light")]
    elif sorted_pairs:
        top_feat = feats[0] if feats else "—"
        top_val  = vals[0]  if vals  else 0
        nlp = [
            html.P([
                "Pour la variable ", html.Strong(variable), ", la feature la plus influente est ",
                html.Strong(top_feat),
                f" avec une contribution SHAP moyenne de {top_val:.4f}. ",
                "Les barres vertes (waterfall) indiquent un effet positif sur la prédiction, "
                "les rouges un effet négatif.",
            ], className="text-light"),
            html.P(f"Base value (prédiction sans features): {base_value:.2f}",
                   className="text-muted small"),
        ]
    else:
        nlp = [html.P("Aucune explication disponible.", className="text-muted")]

    return fig_bar, fig_wf, fig_heat, nlp