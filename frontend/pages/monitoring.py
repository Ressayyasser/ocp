import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import get_live_gta_data, get_all_live_gta_data, get_historical, get_alerts

dash.register_page(__name__, path="/", name="Monitoring", title="Monitoring")


def kpi_card(title, value_id, unit, icon, color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"bi {icon} fs-2 text-{color}"),
                html.Div([
                    html.Small(title, className="text-muted"),
                    html.H4(id=value_id, className=f"text-{color} mb-0 fw-bold"),
                    html.Small(unit, className="text-muted"),
                ], className="ms-3"),
            ], className="d-flex align-items-center"),
        ])
    ], className="mb-3 bg-dark border-secondary")


layout = html.Div([
    dbc.Row([
        dbc.Col(html.H3("⚡ Tableau de Bord — Monitoring Temps Réel", className="text-light fw-bold"), width=9),
        dbc.Col([dbc.Badge(id="status-badge", color="secondary", className="fs-6 p-2")], width=3, className="text-end"),
    ], className="mb-4"),

    dcc.Interval(id="monitor-interval", interval=3_000, n_intervals=0),

    dbc.Row([
        dbc.Col(kpi_card("Production Totale", "kpi-production", "MW", "bi-lightning-charge-fill", "warning"), width=3),
        dbc.Col(kpi_card("Bilan Net",         "kpi-bilan",      "MWh/j", "bi-arrow-up-right-circle", "success"), width=3),
        dbc.Col(kpi_card("Rendement Moyen",   "kpi-efficiency", "%",     "bi-gear-fill",             "info"),    width=3),
        dbc.Col(kpi_card("Pression Adm.",     "kpi-pressure",   "bar",   "bi-thermometer-half",      "danger"),  width=3),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("GTA 1", className="text-warning fw-bold"),
                dbc.CardBody([
                    html.H5(id="gta1-val", className="text-warning"),
                    html.Small("MW", className="text-muted"),
                    html.Small(id="gta1-rend", className="text-muted d-block"),
                ]),
            ], className="bg-dark border-warning text-center"),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("GTA 2", className="text-info fw-bold"),
                dbc.CardBody([
                    html.H5(id="gta2-val", className="text-info"),
                    html.Small("MW", className="text-muted"),
                    html.Small(id="gta2-rend", className="text-muted d-block"),
                ]),
            ], className="bg-dark border-info text-center"),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("GTA 3", className="text-success fw-bold"),
                dbc.CardBody([
                    html.H5(id="gta3-val", className="text-success"),
                    html.Small("MW", className="text-muted"),
                    html.Small(id="gta3-rend", className="text-muted d-block"),
                ]),
            ], className="bg-dark border-success text-center"),
        ], width=4),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Production journalière", className="text-light"),
                dbc.CardBody(dcc.Graph(id="production-chart", style={"height": "300px"})),
            ], className="bg-dark border-secondary"),
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Débits vapeur (t/h)", className="text-light"),
                dbc.CardBody(dcc.Graph(id="steam-gauge-chart", style={"height": "300px"})),
            ], className="bg-dark border-secondary"),
        ], width=4),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rendement GTA", className="text-light"),
                dbc.CardBody(dcc.Graph(id="rendement-chart", style={"height": "280px"})),
            ], className="bg-dark border-secondary"),
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔔 Alertes récentes", className="text-light"),
                dbc.CardBody(html.Div(id="alerts-list", style={"maxHeight": "260px", "overflowY": "auto"})),
            ], className="bg-dark border-secondary"),
        ], width=4),
    ]),
])


# ── Callback KPIs — Données live depuis le simulateur (port 8051) ─────────────

@callback(
    Output("kpi-production", "children"),
    Output("kpi-bilan",      "children"),
    Output("kpi-efficiency", "children"),
    Output("kpi-pressure",   "children"),
    Output("gta1-val", "children"),
    Output("gta2-val", "children"),
    Output("gta3-val", "children"),
    Output("gta1-rend", "children"),
    Output("gta2-rend", "children"),
    Output("gta3-rend", "children"),
    Output("status-badge", "children"),
    Output("status-badge", "color"),
    Input("monitor-interval", "n_intervals"),
)
def update_kpis(n):
    # Récupérer les données live des 3 GTA depuis le simulateur (port 8051)
    gta1_data = get_live_gta_data("GTA1")
    gta2_data = get_live_gta_data("GTA2")
    gta3_data = get_live_gta_data("GTA3")

    # Extraire les données de chaque GTA
    def extract_gta(data):
        if isinstance(data, dict) and "data" in data and data["data"]:
            row = data["data"][0]
            return {
                "puissance_mw": float(row.get("puissance_mw", 0)),
                "rendement": float(row.get("rendement", 0)),
                "adm_pression": float(row.get("adm_pression", 0)),
                "adm_temp": float(row.get("adm_temp", 0)),
                "adm_debit": float(row.get("adm_debit", 0)),
            }
        return {"puissance_mw": 0, "rendement": 0, "adm_pression": 0, "adm_temp": 0, "adm_debit": 0}

    g1 = extract_gta(gta1_data)
    g2 = extract_gta(gta2_data)
    g3 = extract_gta(gta3_data)

    # Calculer les totaux
    total_prod = g1["puissance_mw"] + g2["puissance_mw"] + g3["puissance_mw"]
    avg_rendement = (g1["rendement"] + g2["rendement"] + g3["rendement"]) / 3
    avg_pressure = (g1["adm_pression"] + g2["adm_pression"] + g3["adm_pression"]) / 3
    # Bilan net approximatif (somme des productions - estimation conso)
    bilan_net = total_prod * 0.85  # estimation simplifiée

    # Statut
    is_live = all(
        isinstance(d, dict) and "data" in d and d["data"]
        for d in [gta1_data, gta2_data, gta3_data]
    )

    return (
        f"{total_prod:.1f}",
        f"{bilan_net:.1f}",
        f"{avg_rendement:.1f}",
        f"{avg_pressure:.1f}",
        f"{g1['puissance_mw']:.1f}",
        f"{g2['puissance_mw']:.1f}",
        f"{g3['puissance_mw']:.1f}",
        f"Rend: {g1['rendement']:.1f}%",
        f"Rend: {g2['rendement']:.1f}%",
        f"Rend: {g3['rendement']:.1f}%",
        "● LIVE" if is_live else "● OFFLINE",
        "success" if is_live else "danger",
    )


# ── Callback Charts — Données historiques ─────────────────────────────────────

@callback(
    Output("production-chart", "figure"),
    Output("rendement-chart",  "figure"),
    Input("monitor-interval", "n_intervals"),
)
def update_charts(n):
    res = get_historical(30)
    dark = dict(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                font=dict(color="#e0e0e0"), margin=dict(l=40, r=20, t=20, b=40))

    hist_list = res.get("data", []) if isinstance(res, dict) else []

    if hist_list:
        df = pd.DataFrame(hist_list)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Production chart
        fig_prod = go.Figure()
        if "production" in df.columns:
            fig_prod.add_trace(go.Scatter(
                x=df["timestamp"], y=df["production"],
                mode="lines+markers", name="Production",
                line=dict(color="#f0c040", width=2)
            ))
        if "bilan_net" in df.columns:
            fig_prod.add_trace(go.Scatter(
                x=df["timestamp"], y=df["bilan_net"],
                mode="lines", name="Bilan Net",
                line=dict(color="#40c0f0", width=1.5, dash="dot")
            ))
        fig_prod.update_layout(**dark, legend=dict(orientation="h", y=1.1))

        # Rendement chart
        fig_rend = go.Figure()
        for col, color, name in [
            ("rendement_gta1", "#f0c040", "GTA1"),
            ("rendement_gta2", "#40c0f0", "GTA2"),
            ("rendement_gta3", "#40f0a0", "GTA3"),
        ]:
            if col in df.columns:
                fig_rend.add_trace(go.Scatter(
                    x=df["timestamp"], y=df[col],
                    mode="lines", name=name,
                    line=dict(color=color, width=2)
                ))
        fig_rend.update_layout(**dark, legend=dict(orientation="h", y=1.1), yaxis_title="Rendement (%)")
    else:
        fig_prod = go.Figure().update_layout(**dark)
        fig_rend = go.Figure().update_layout(**dark)

    return fig_prod, fig_rend


# ── Callback Steam Gauge ──────────────────────────────────────────────────────

@callback(
    Output("steam-gauge-chart", "figure"),
    Input("monitor-interval", "n_intervals"),
)
def update_steam(n):
    dark = dict(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                font=dict(color="#e0e0e0"), margin=dict(l=20, r=20, t=30, b=20))
    fig = go.Figure()

    # Récupérer les données live du GTA1 pour les débits vapeur
    gta1_data = get_live_gta_data("GTA1")
    if isinstance(gta1_data, dict) and "data" in gta1_data and gta1_data["data"]:
        row = gta1_data["data"][0]
        steam_hp = float(row.get("sout_debit", 0))  # sout_debit = vapeur HP soutirée
        # Pour MP et BP, on utilise des valeurs approximatives basées sur les ratios
        steam_mp = float(row.get("sout_debit", 0)) * 0.6  # estimation
        steam_bp = float(row.get("ext_debit", 0))  # ext_debit = extraction BP

        for label, value, color in [
            ("HP (t/h)", steam_hp, "#ff6b6b"),
            ("MP (t/h)", steam_mp, "#ffd93d"),
            ("BP (t/h)", steam_bp, "#6bcb77"),
        ]:
            fig.add_trace(go.Bar(x=[label], y=[value], marker_color=color, name=label))

    fig.update_layout(**dark, barmode="group", showlegend=False)
    return fig


# ── Callback Alertes ──────────────────────────────────────────────────────────

@callback(
    Output("alerts-list", "children"),
    Input("monitor-interval", "n_intervals"),
)
def update_alerts(n):
    res = get_alerts(10)

    if "error" in res or not isinstance(res, dict):
        return html.P("Aucune alerte", className="text-muted small")

    alerts_list = res.get("alerts", [])
    if not alerts_list:
        return html.P("Aucune alerte", className="text-muted small")

    color_map = {"critical": "danger", "warning": "warning", "info": "info"}
    items = []
    for a in alerts_list[:8]:
        level   = a.get("level", "info")
        color   = color_map.get(level, "secondary")
        ts      = a.get("timestamp", "")[:16]
        message = a.get("message", "—")
        items.append(
            dbc.Alert([
                html.Strong(f"[{level.upper()}] "),
                html.Small(ts, className="text-muted me-1"),
                html.Br(),
                html.Small(message),
            ], color=color, className="py-1 px-2 mb-1", style={"fontSize": "0.75rem"})
        )
    return items