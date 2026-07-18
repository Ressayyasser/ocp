"""
pages/anomalies.py — Anomaly detection results page.
Shows Isolation Forest scores, CUSUM alerts, and historical anomaly log.
"""

import dash
from dash import dcc, html, Input, Output, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import get_anomalies, get_historical, get_alerts, rescan_anomalies

dash.register_page(__name__, path="/anomalies", name="Anomalies", title="Anomalies")

SEVERITY_COLOR = {"critical": "#ff4d4d", "warning": "#ffd93d",
                  "info": "#6bcb77", "normal": "#4d9de0"}

layout = html.Div([
    html.H3("🚨 Détection d'Anomalies — Isolation Forest + CUSUM",
            className="text-light fw-bold mb-4"),

    dcc.Interval(id="anom-interval", interval=15_000, n_intervals=0),

    # Scan controls
    html.Div([
        dbc.Button("🔍 Relancer l'analyse (Isolation Forest + CUSUM)",
                   id="anom-rescan-btn", color="warning", size="sm",
                   className="me-3"),
        html.Span(id="anom-scan-status", className="text-muted small"),
    ], className="mb-3 d-flex align-items-center"),

    # Summary badges
    dbc.Row(id="anom-summary", className="mb-4"),

    # Score timeline
    dbc.Card([
        dbc.CardHeader("Score d'anomalie dans le temps (30 derniers jours)",
                       className="text-light"),
        dbc.CardBody(dcc.Graph(id="anom-score-chart", style={"height": "300px"})),
    ], className="bg-dark border-secondary mb-4"),

    dbc.Row([
        # Group distribution pie
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribution par groupe", className="text-light"),
                dbc.CardBody(dcc.Graph(id="anom-pie", style={"height": "280px"})),
            ], className="bg-dark border-secondary"),
        ], width=5),

        # Severity bar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribution par sévérité", className="text-light"),
                dbc.CardBody(dcc.Graph(id="anom-severity-bar", style={"height": "280px"})),
            ], className="bg-dark border-secondary"),
        ], width=7),
    ], className="mb-4"),

    # Anomaly table
    dbc.Card([
        dbc.CardHeader([
            "📋 Journal des anomalies ",
            dbc.Badge(id="anom-count-badge", color="danger", className="ms-2"),
        ], className="text-light"),
        dbc.CardBody(html.Div(id="anom-table", style={"maxHeight": "350px", "overflowY": "auto"})),
    ], className="bg-dark border-secondary"),
])


@callback(
    Output("anom-summary",       "children"),
    Output("anom-score-chart",   "figure"),
    Output("anom-pie",           "figure"),
    Output("anom-severity-bar",  "figure"),
    Output("anom-table",         "children"),
    Output("anom-count-badge",   "children"),
    Output("anom-scan-status",   "children"),
    Input("anom-interval", "n_intervals"),
    Input("anom-rescan-btn", "n_clicks"),
)
def update_anomalies(n, n_rescan):
    dark = dict(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                font=dict(color="#e0e0e0"), margin=dict(l=40, r=20, t=20, b=40))

    scan_status = ""
    if ctx.triggered_id == "anom-rescan-btn" and n_rescan:
        res = rescan_anomalies()
        if isinstance(res, dict) and "error" not in res:
            det = res.get("by_detector", {})
            scan_status = (f"Analyse terminée : {res.get('count', 0)} anomalies "
                           f"(Isolation Forest {det.get('isolation_forest', 0)} · "
                           f"CUSUM {det.get('cusum', 0)}) "
                           f"sur {res.get('window_days', '—')} jours.")
        else:
            scan_status = f"Échec de l'analyse : {res.get('error', '?')}"

    anom_resp = get_anomalies(500)
    hist_resp = get_historical(30)

    # 🔥 FIX: Extract lists safely from the dictionary responses returned by the API
    anomalies = anom_resp.get("anomalies", []) if isinstance(anom_resp, dict) else anom_resp
    if not isinstance(anomalies, list):
        anomalies = []
        
    hist = hist_resp.get("data", []) if isinstance(hist_resp, dict) else hist_resp
    if not isinstance(hist, list):
        hist = []

    # ── Summary badges ─────────────────────────────────────────────────────────
    n_critical = sum(1 for a in anomalies if a.get("severity") == "critical")
    n_warning  = sum(1 for a in anomalies if a.get("severity") == "warning")
    n_total    = len(anomalies)

    summary = [
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H3(str(n_total),    className="text-danger fw-bold mb-0"),
            html.Small("Total anomalies", className="text-muted"),
        ])], className="bg-dark border-danger text-center"), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H3(str(n_critical), className="text-danger fw-bold mb-0"),
            html.Small("Critiques", className="text-muted"),
        ])], className="bg-dark border-danger text-center"), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H3(str(n_warning),  className="text-warning fw-bold mb-0"),
            html.Small("Avertissements", className="text-muted"),
        ])], className="bg-dark border-warning text-center"), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H3(str(n_total - n_critical - n_warning),
                    className="text-info fw-bold mb-0"),
            html.Small("Informations", className="text-muted"),
        ])], className="bg-dark border-info text-center"), width=3),
    ]

    # ── Score timeline (Isolation Forest vs CUSUM) ─────────────────────────────
    fig_score = go.Figure()
    if anomalies:
        df_a = pd.DataFrame(anomalies)
        if "timestamp" in df_a.columns and "score" in df_a.columns:
            df_a["timestamp"] = pd.to_datetime(df_a["timestamp"], errors="coerce",
                                               format="mixed")
            df_a = df_a.dropna(subset=["timestamp"]).sort_values("timestamp")
            cause = df_a.get("cause", pd.Series("", index=df_a.index)).astype(str)
            df_if    = df_a[cause.str.startswith("Isolation")]
            df_cusum = df_a[cause.str.startswith("CUSUM")]

            if not df_if.empty:
                fig_score.add_trace(go.Scatter(
                    x=df_if["timestamp"], y=df_if["score"],
                    mode="markers", name="Isolation Forest",
                    marker=dict(size=7, color="#ff6b6b", symbol="circle"),
                ))
            if not df_cusum.empty:
                fig_score.add_trace(go.Scatter(
                    x=df_cusum["timestamp"], y=df_cusum["score"],
                    mode="markers", name="CUSUM (dérive)",
                    marker=dict(size=9, color="#ffb84d", symbol="diamond"),
                ))
            # Threshold reference
            fig_score.add_hline(y=-0.3, line_dash="dash",
                                line_color="#ffd93d", annotation_text="Seuil warning")
            fig_score.add_hline(y=-0.5, line_dash="dash",
                                line_color="#ff4d4d", annotation_text="Seuil critique")
    fig_score.update_layout(**dark, yaxis_title="Score (plus négatif = plus anormal)",
                            legend=dict(orientation="h", y=1.12))

    # ── Pie by group ───────────────────────────────────────────────────────────
    fig_pie = go.Figure()
    if anomalies:
        df_a = pd.DataFrame(anomalies)
        # Check for 'variable' or 'cause' column
        col_to_count = "variable" if "variable" in df_a.columns else "cause"
        if col_to_count in df_a.columns:
            grp = df_a[col_to_count].value_counts()
            fig_pie.add_trace(go.Pie(
                labels=grp.index.tolist(), values=grp.values.tolist(),
                hole=0.4, marker=dict(colors=["#ff6b6b","#ffd93d","#6bcb77","#4d9de0","#c77dff"]),
            ))
    fig_pie.update_layout(**dark, showlegend=True,
                          legend=dict(orientation="v", x=1, y=0.5))

    # ── Severity bar ───────────────────────────────────────────────────────────
    fig_bar = go.Figure()
    if anomalies:
        df_a  = pd.DataFrame(anomalies)
        if "severity" in df_a.columns:
            sev  = df_a["severity"].value_counts()
            cols = [SEVERITY_COLOR.get(s, "#888") for s in sev.index]
            fig_bar.add_trace(go.Bar(x=sev.index.tolist(), y=sev.values.tolist(),
                                     marker_color=cols, name="Sévérité"))
    fig_bar.update_layout(**dark, yaxis_title="Nombre", showlegend=False)

    # ── Table ─────────────────────────────────────────────────────────────────
    if anomalies:
        rows = []
        for a in anomalies[:50]:
            sev   = a.get("severity", "info")
            color = SEVERITY_COLOR.get(sev, "#888")
            rows.append(html.Tr([
                html.Td(str(a.get("timestamp","—"))[:16], style={"fontSize":"0.8rem"}),
                html.Td(dbc.Badge(sev.upper(), style={"backgroundColor": color, "color": "white"})),
                html.Td(f"{float(a.get('score', 0) or 0):.3f}", style={"fontSize":"0.8rem"}),
                html.Td(a.get("cause",    "—"), style={"fontSize":"0.8rem"}),
                html.Td(a.get("variable", "—"), style={"fontSize":"0.8rem"}),
            ]))
            
        # Note: using color="dark" for Bootstrap 5 compatibility
        table = dbc.Table(
            [html.Thead(html.Tr([html.Th(c) for c in
                                 ["Horodatage","Sévérité","Score","Cause","Variable"]])),
             html.Tbody(rows)],
            bordered=True, color="dark", hover=True, size="sm",
        )
    else:
        table = html.P("Aucune anomalie détectée — cliquez sur « Relancer l'analyse » "
                       "pour lancer le balayage Isolation Forest + CUSUM.",
                       className="text-success text-center py-3")

    if not scan_status and anomalies:
        n_if    = sum(1 for a in anomalies
                      if str(a.get("cause", "")).startswith("Isolation"))
        n_cusum = sum(1 for a in anomalies
                      if str(a.get("cause", "")).startswith("CUSUM"))
        scan_status = (f"{n_total} anomalies en base — "
                       f"Isolation Forest {n_if} · CUSUM {n_cusum}")

    return summary, fig_score, fig_pie, fig_bar, table, str(n_total), scan_status