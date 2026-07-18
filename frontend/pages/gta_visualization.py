"""
pages/gta_visualization.py — GTA Digital Twin synoptic.

Process-flow architecture (not a floating-KPI dashboard):
    Source HP -> Turbine (HP -> MP -> BP) -> Alternateur -> Reseau MT
                              |
                              v  (echappement BP)
                         Condenseur <-> Refroidissement

Each zone is its own SVG component with its own internal layout. Zones are
connected exclusively through coloured, animated pipes/links (no floating
cards, no dashed diagonals). Every zone is clickable (via a transparent
HTML overlay positioned over its SVG bounding box) and opens a detail
panel with the full telemetry for that equipment, including fields that
no longer have a permanent on-canvas slot (vibration, dilatation,
servo-valve position, running hours, alarms — all folded into the
Turbine detail panel).
"""

from pathlib import Path
import sys, os
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
import math
import dash
from dash import dcc, html, Input, Output, State, ALL, ctx, no_update
import dash_bootstrap_components as dbc
from utils.styles import COLORS, card_style
from utils.api_client import get
import requests
import pandas as pd
import plotly.graph_objects as go

dash.register_page(__name__, path="/gta", name="GTA", title="GTA")

# Shared SVG synoptic builder (extracted to utils so non-page modules,
# e.g. the Digital Twin's Virtual GTA sandbox, can reuse it without
# re-importing this page module and duplicating its callbacks).
from utils.gta_svg import (
    _build_gta_svg, _parse_record, _rend_color, _GTA_COLORS,
    ZONES, ZONE_LABELS, VB_W, VB_H,
    _MUTED, _YELLOW, _ORANGE, _CYAN, _GREEN, _RED,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CLICK OVERLAYS  (transparent HTML divs positioned over each SVG zone)
# ─────────────────────────────────────────────────────────────────────────────

def _click_overlays():
    divs = []
    for key, (zx, zy, zw, zh) in ZONES.items():
        divs.append(html.Div(
            id={"type": "gta-equip-click", "index": key},
            n_clicks=0,
            title=f"{ZONE_LABELS[key]} — cliquer pour le détail",
            className="gta-equip-overlay",
            style={
                "position": "absolute",
                "left":   f"{zx / VB_W * 100:.3f}%",
                "top":    f"{zy / VB_H * 100:.3f}%",
                "width":  f"{zw / VB_W * 100:.3f}%",
                "height": f"{zh / VB_H * 100:.3f}%",
                "cursor": "pointer",
                "zIndex": 5,
            },
        ))
    return divs


# ─────────────────────────────────────────────────────────────────────────────
#  EQUIPMENT DETAIL PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _detail_row(label, value, color=None):
    return html.Div([
        html.Span(label, style={"color": _MUTED, "fontSize": "13px"}),
        html.Span(value, style={"color": color or COLORS["text_primary"],
                                 "fontWeight": "600", "fontSize": "14px"}),
    ], style={"display": "flex", "justifyContent": "space-between",
              "padding": "7px 0", "borderBottom": f"1px solid {COLORS['border']}"})


def _detail_section(title, rows):
    return html.Div([
        html.H6(title, style={"color": COLORS["text_primary"], "marginTop": "16px",
                               "marginBottom": "2px", "fontSize": "13px",
                               "textTransform": "uppercase", "letterSpacing": "0.05em",
                               "opacity": 0.7}),
        html.Div(rows),
    ])


def _hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _mini_trend(records, col, color, fallback_col=None):
    df = pd.DataFrame(records or [])
    use_col = col if col in df.columns else (fallback_col if fallback_col in df.columns else None)
    if df.empty or use_col is None:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, xref="paper",
                           yref="paper", showarrow=False,
                           font=dict(size=11, color="#8b949e"))
    else:
        df = df.dropna(subset=[use_col])
        x = df["date"] if "date" in df.columns else df.index
        # a single point would draw an invisible line — show a marker instead
        mode = "lines" if len(df) >= 2 else "markers"
        fig = go.Figure(go.Scatter(x=x, y=df[use_col], mode=mode,
                                    line=dict(color=color, width=2),
                                    marker=dict(color=color, size=8),
                                    fill="tozeroy",
                                    fillcolor=_hex_to_rgba(color)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=30, r=10, t=6, b=20),
                       height=110, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return fig


def _fetch_daily_history(gta_name: str, days: int = 60) -> list:
    """Daily per-GTA history from the main API — used for the trend charts.
    The live store often holds a single instantaneous reading, which cannot
    make a trend; the /data/daily/{gta} endpoint provides the daily series
    (historical + simulation days, up to today)."""
    try:
        resp = get(f"/data/daily/{gta_name}", params={"days": days})
        if isinstance(resp, dict) and "data" in resp:
            return resp["data"] or []
        if isinstance(resp, list):
            return resp
    except Exception:
        pass
    return []


def _build_equip_detail(key, last, records, gta_name, trend_records=None):
    # Trends need a time series: prefer the daily history endpoint; fall back
    # to whatever the live store holds (may be a single reading).
    trends = trend_records if trend_records else records
    d = _parse_record(last)
    gta_col = _GTA_COLORS.get(gta_name, "#1f6feb")

    if key == "source_hp":
        return "Source HP — Vapeur Haute Pression", html.Div([
            _detail_section("Caractéristiques", [
                _detail_row("Pression",    f"{d['adm_pression']} bar", _YELLOW),
                _detail_row("Température", f"{d['adm_temp']} °C",      _ORANGE),
                _detail_row("Débit",       f"{d['adm_debit']} t/h",    COLORS["text_primary"]),
            ]),
        ])

    if key == "source_bp":
        return "Source BP — Vapeur Basse Pression", html.Div([
            _detail_section("Caractéristiques", [
                _detail_row("Pression",       f"{d['bp_pression']} bar", _YELLOW),
                _detail_row("Débit",          f"{d['bp_debit']} t/h",    COLORS["text_primary"]),
                _detail_row("Désamorçage",    "S-18 min"),
            ]),
        ])

    if key == "soutirage_mp":
        return "Soutirage MP", html.Div([
            _detail_section("Vapeur MP soutirée", [
                _detail_row("Débit",    f"{d['sout_debit']} t/h",    COLORS["text_primary"]),
                _detail_row("Pression", f"{d['sout_pression']} bar", _YELLOW),
            ]),
        ])

    if key == "turbine":
        rend_col = _rend_color(d["rendement"])
        alarms = [
            ("TRIP",        d["rendement"] < 35),
            ("VIBRATION",   d["vib2"] > 0.8),
            ("LOW VACUUM",  d["ext_pression"] > 0.12),
            ("TEMP. ADM.",  d["adm_temp"] > 460),
        ]
        return "Turbine à vapeur", html.Div([
            _detail_section("Étages", [
                _detail_row("Admission HP", f"{d['vap_inlet']} bar · {d['adm_temp']} °C · {d['adm_debit']} t/h"),
                _detail_row("Admission MP", f"{d['sout_pression']} bar · {d['temp_mp']} °C · {d['sout_debit']} t/h"),
                _detail_row("Admission BP", f"{d['bp_pression']} bar · {d['temp_bp']} °C · {d['bp_debit']} t/h"),
            ]),
            _detail_section("Mesures complémentaires", [
                _detail_row("Vitesse",            f"{d['vitesse']} RPM"),
                _detail_row("Rendement",           f"{d['rendement']} %", rend_col),
                _detail_row("Position servo HP",   f"{d['posit_hp']} %", _YELLOW),
                _detail_row("Position servo BP",   f"{d['posit_bp']} %", _YELLOW),
                _detail_row("Dilatation axiale",   "+0.12 mm", _YELLOW),
                _detail_row("Dilatation corps",    "4.5 mm", _YELLOW),
                _detail_row("Vibration palier 1",  f"{d['vib1']} μm"),
                _detail_row("Vibration palier 2",  f"{d['vib2']} μm"),
                _detail_row("DD3",                 f"{d['dd3']} mm"),
                _detail_row("Huile — pression",    f"{d['oil_pression']} bar", _YELLOW),
                _detail_row("Huile — température", f"{d['oil_temp']} °C", _ORANGE),
                _detail_row("Heures de fonctionnement", "32767 h"),
            ]),
            _detail_section("Alarmes", [
                _detail_row(name, "ACTIVE" if active else "ok",
                            _RED if active else _GREEN)
                for name, active in alarms
            ]),
            _detail_section("Tendances (historique journalier)", [
                html.Div("Débit admission (t/h)", style={"color": _MUTED, "fontSize": "11px", "marginTop": "8px"}),
                dcc.Graph(figure=_mini_trend(trends, "adm_debit", gta_col), config={"displayModeBar": False}),
                html.Div("Énergie produite (MWh)", style={"color": _MUTED, "fontSize": "11px"}),
                dcc.Graph(figure=_mini_trend(trends, "puissance_mw", _YELLOW, "energie_mwh"),
                          config={"displayModeBar": False}),
                html.Div("Rendement (%)", style={"color": _MUTED, "fontSize": "11px"}),
                dcc.Graph(figure=_mini_trend(trends, "rendement", _GREEN), config={"displayModeBar": False}),
            ]),
        ])

    if key == "alternateur":
        return "Alternateur — 47 MVA", html.Div([
            _detail_section("Électrique", [
                _detail_row("Puissance active",   f"{d['p_active']} MW", _GREEN),
                _detail_row("Puissance réactive",  f"{d['p_reactive']} Mvar"),
                _detail_row("Cos φ",               str(d['cos_phi'])),
                _detail_row("Tension",             f"{d['tension']} kV", _YELLOW),
                _detail_row("Courant",             "1369 A"),
                _detail_row("Fréquence",           "50 Hz"),
                _detail_row("Vibration alternateur", f"{d['vib1']} μm"),
            ]),
        ])

    if key == "reseau_mt":
        return "Réseau MT", html.Div([
            _detail_section("Réseau", [
                _detail_row("Puissance active", f"{d['p_active']} MW", _GREEN),
                _detail_row("Fréquence réseau",  "50 Hz"),
                _detail_row("Tension bus",       "6.3 kV", _YELLOW),
                _detail_row("Charge site",       "14.0 %"),
            ]),
        ])

    if key == "condenseur":
        return "Condenseur principal", html.Div([
            _detail_section("Condensation", [
                _detail_row("Pression de vide",   f"{d['ext_pression']} bar", _CYAN),
                _detail_row("Température sortie BP", f"{d['cond_temp']} °C", _ORANGE),
                _detail_row("Débit d'eau",         f"{d['cond_eau']} t/h"),
                _detail_row("Niveau",              f"{d['level_pct']} %"),
            ]),
        ])

    if key == "refroidissement":
        return "Circuit de refroidissement", html.Div([
            _detail_section("Pompes", [
                _detail_row("Pompe A — 20CC01", "0.1 A"),
                _detail_row("Pompe B — 20MC01", "0.0 A"),
            ]),
            _detail_section("Eau de circulation", [
                _detail_row("Débit",          f"{d['cond_eau']} t/h"),
                _detail_row("Temp. entrée",    f"{d['cond_temp']} °C", _ORANGE),
                _detail_row("Temp. sortie",    f"{round(d['cond_temp'] - 8, 1)} °C", _CYAN),
                _detail_row("Conductivité (ATI)", "6.2 μS"),
            ]),
        ])

    return "Détail", html.Div("Aucune donnée.")


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def layout_gta():
    return html.Div([
        # ── GTA selector ──────────────────────────────────────────────────
        html.Div(style=card_style, children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "24px",
                       "flexWrap": "wrap"},
                children=[
                    html.H5("🏭 GTA — Synoptique Procédé", style={"color": COLORS["text_primary"], "margin": 0}),
                    dcc.RadioItems(
                        id="gta-selector",
                        options=[
                            {"label": "  GTA 1", "value": "GTA1"},
                            {"label": "  GTA 3", "value": "GTA3"},
                            {"label": "  GTA 2 (Sept–Déc)", "value": "GTA2"},
                        ],
                        value="GTA1", inline=True,
                        style={"color": COLORS["text_primary"]},
                        inputStyle={"marginRight": "5px", "marginLeft": "16px"},
                    ),
                ],
            ),
        ]),

        # ── SVG interactive synoptic + click overlays ─────────────────────
        html.Div(
            style={**card_style, "padding": "0", "overflow": "hidden", "position": "relative"},
            children=[
                html.Div(id="gta-svg-container", style={"width": "100%", "position": "relative"}),
                *_click_overlays(),
            ],
        ),

        # ── Daily trend charts ─────────────────────────────────────────────
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)",
                   "gap": "16px", "marginTop": "16px"},
            children=[
                html.Div(style=card_style, children=[
                    html.H6("Débit Admission (t/h)", style={"color": COLORS["text_muted"]}),
                    dcc.Graph(id="gta-chart-debit", style={"height": "220px"}),
                ]),
                html.Div(style=card_style, children=[
                    html.H6("Energie Produite (MWh)", style={"color": COLORS["text_muted"]}),
                    dcc.Graph(id="gta-chart-energie", style={"height": "220px"}),
                ]),
                html.Div(style=card_style, children=[
                    html.H6("Rendement (%)", style={"color": COLORS["text_muted"]}),
                    dcc.Graph(id="gta-chart-rendement", style={"height": "220px"}),
                ]),
            ],
        ),
        dcc.Interval(id="gta-live-interval", interval=1000, n_intervals=0),
        dcc.Store(id="gta-daily-data"),

        # ── Equipment detail panel ─────────────────────────────────────────
        dbc.Offcanvas(
            id="gta-equip-offcanvas", title="Détail équipement", is_open=False,
            placement="end", style={"width": "420px", "backgroundColor": "#0d1117"},
            children=html.Div(id="gta-equip-detail-content"),
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

from dash import callback


@callback(
    Output("gta-daily-data", "data"),
    Input("gta-selector", "value"),
    Input("gta-live-interval", "n_intervals"),
)
def fetch_gta_data(gta, n):
    try:
        resp = requests.get(f"http://127.0.0.1:8051/api/live/{gta}", timeout=1)
        if resp.status_code == 200:
            live_data = resp.json()
            if isinstance(live_data, list):
                live_data = {"data": live_data}
            live_data["is_live"] = True
            return live_data
    except Exception as e:
        print(f"[Live API Error] {e}")

    try:
        hist_data = get(f"/data/daily/{gta}")
        if isinstance(hist_data, list):
            hist_data = {"data": hist_data, "is_live": False}
        elif isinstance(hist_data, dict):
            hist_data["is_live"] = False
        return hist_data
    except Exception as e:
        print(f"[Historical API Error] {e}")
        return {"data": [], "is_live": False}


@callback(
    Output("gta-svg-container", "children"),
    Input("gta-daily-data", "data"),
    Input("gta-selector", "value"),
)
def update_svg(data, gta):
    if not data:
        return html.Div("⏳ En attente de données...", style={"color": "white", "padding": "20px"})
    records = data.get("data", []) if isinstance(data, dict) else data
    if not records:
        return html.Div("📭 Aucune donnée disponible.", style={"color": "red", "padding": "20px"})
    return _build_gta_svg({"data": records, "is_live": data.get("is_live", False)}, gta)


@callback(
    Output("gta-equip-offcanvas", "is_open"),
    Output("gta-equip-detail-content", "children"),
    Output("gta-equip-offcanvas", "title"),
    Input({"type": "gta-equip-click", "index": ALL}, "n_clicks"),
    State("gta-daily-data", "data"),
    State("gta-selector", "value"),
    prevent_initial_call=True,
)
def open_equip_detail(n_clicks_list, data, gta):
    if not n_clicks_list or not any(n_clicks_list):
        return no_update, no_update, no_update
    triggered = ctx.triggered_id
    if not triggered:
        return no_update, no_update, no_update
    key = triggered["index"]
    records = (data or {}).get("data", []) if isinstance(data, dict) else (data or [])
    last = records[-1] if records else {}
    # Daily history for the trend charts (the live store may hold a single
    # instantaneous reading — useless for a trend line)
    trend_records = _fetch_daily_history(gta) if key == "turbine" else None
    title, content = _build_equip_detail(key, last, records, gta,
                                         trend_records=trend_records)
    return True, content, title


@callback(
    [Output("gta-chart-debit", "figure"),
     Output("gta-chart-energie", "figure"),
     Output("gta-chart-rendement", "figure")],
    Input("gta-daily-data", "data"),
)
def update_gta_charts(data):
    if not data:
        return _empty_fig(), _empty_fig(), _empty_fig()

    records = data.get("data", []) if isinstance(data, dict) else data
    is_live = data.get("is_live", False) if isinstance(data, dict) else False

    if not records:
        return _empty_fig(), _empty_fig(), _empty_fig()

    df = pd.DataFrame(records)
    if df.empty:
        return _empty_fig(), _empty_fig(), _empty_fig()

    gta_name  = data.get("gta", "GTA") if isinstance(data, dict) else "GTA"
    color_map = {"GTA1": COLORS["accent_blue"], "GTA2": COLORS["accent_orange"],
                 "GTA3": COLORS["accent_green"]}
    color = color_map.get(gta_name, COLORS["accent_blue"])
    bg    = COLORS["bg_card"]

    if is_live:
        last_row = df.iloc[-1]
        adm_debit = float(last_row.get("adm_debit", 0))
        energie = float(last_row.get("puissance_mw", last_row.get("energie_mwh", 0)))
        rendement = float(last_row.get("rendement", 0))

        def _live_gauge(value, title, color, max_val=100):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=value,
                title={'text': f"<b>{title}</b>", 'font': {'size': 14, 'color': COLORS["text_muted"]}},
                number={'font': {'size': 24, 'color': color}},
                gauge={'axis': {'range': [0, max_val], 'tickcolor': COLORS["text_muted"]},
                       'bar': {'color': color}, 'bgcolor': bg, 'borderwidth': 2,
                       'bordercolor': COLORS["border"]},
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                               margin=dict(l=20, r=20, t=40, b=20), height=220)
            return fig

        return (
            _live_gauge(adm_debit, "Débit Admission (t/h)", color, max_val=250),
            _live_gauge(energie, "Puissance (MW)", COLORS["accent_yellow"], max_val=50),
            _live_gauge(rendement, "Rendement (%)", COLORS["accent_green"], max_val=50),
        )
    else:
        x = df["date"] if "date" in df.columns else df.index

        def _line_fig(col, y_title, c=color):
            fig = go.Figure(go.Scatter(x=x, y=df[col], mode="lines", line=dict(color=c, width=2)))
            fig.update_layout(template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                               margin=dict(l=50, r=20, t=30, b=40), height=220,
                               xaxis=dict(title="Date", showgrid=True, gridcolor=COLORS["border"]),
                               yaxis=dict(title=y_title, showgrid=True, gridcolor=COLORS["border"]),
                               showlegend=False)
            return fig

        energie_col = "puissance_mw" if "puissance_mw" in df.columns else "energie_mwh"
        return (
            _line_fig("adm_debit", "Débit (t/h)"),
            _line_fig(energie_col, "Energie (MWh)"),
            _line_fig("rendement", "Rendement (%)", COLORS["accent_green"]),
        )


def _empty_fig():
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg_card"],
                       plot_bgcolor=COLORS["bg_card"], margin=dict(l=0, r=0, t=0, b=0),
                       height=220)
    return fig


layout = layout_gta