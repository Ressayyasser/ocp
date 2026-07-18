from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import json
import uuid
import math
import copy
from datetime import datetime
import threading
import time

from database.database import get_connection, init_db

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                title="OCP - GTA SCADA Simulator", suppress_callback_exceptions=True)

# ====================== GLOBAL STATE ======================
current_sim_id = str(uuid.uuid4())[:12]
simulation_running = True
current_scenario = "normal"
active_gta = "GTA1"

# Real base parameters calculated from historical datasets (2022-2025)
base_params = {
    "GTA1": {
        "adm_debit": 195.3, "adm_temp": 460.1, "adm_pression": 54.8,
        "sout_debit": 147.8, "sout_pression": 8.04,
        "ext_debit": 46.7, "ext_pression": 76.3,
        "bp_pression": 0.9, "bp_debit": 8.7,
        "puissance_mw": 25.0, "rendement": 41.7,
        "vitesse": 6400.0, "vib1": 0.22, "vib2": 0.41, "dd3": 0.61,
        "oil_pression": 1.52, "oil_temp": 40.4,
        "cos_phi": 0.855, "p_active": 25.0, "p_reactive": 15.1, "tension": 10.5,
        "posit_hp": 85.2, "posit_bp": 74.8, "vap_inlet": 55.4,
        "cond_temp": 245.0, "cond_eau": 87.0, "level_pct": 78.1
    },
    "GTA2": {
        "adm_debit": 185.0, "adm_temp": 435.0, "adm_pression": 54.3,
        "sout_debit": 110.0, "sout_pression": 8.20,
        "ext_debit": 70.0, "ext_pression": 65.0,
        "bp_pression": 0.9, "bp_debit": 8.7,
        "puissance_mw": 18.0, "rendement": 35.0,
        "vitesse": 6350.0, "vib1": 0.25, "vib2": 0.45, "dd3": 0.65,
        "oil_pression": 1.45, "oil_temp": 42.0,
        "cos_phi": 0.840, "p_active": 18.0, "p_reactive": 11.5, "tension": 10.5,
        "posit_hp": 82.0, "posit_bp": 72.0, "vap_inlet": 54.3,
        "cond_temp": 240.0, "cond_eau": 85.0, "level_pct": 75.5
    },
    "GTA3": {
        "adm_debit": 194.3, "adm_temp": 453.3, "adm_pression": 56.1,
        "sout_debit": 149.9, "sout_pression": 8.00,
        "ext_debit": 44.4, "ext_pression": 68.8,
        "bp_pression": 0.9, "bp_debit": 8.7,
        "puissance_mw": 24.0, "rendement": 39.7,
        "vitesse": 6380.0, "vib1": 0.20, "vib2": 0.38, "dd3": 0.58,
        "oil_pression": 1.50, "oil_temp": 39.0,
        "cos_phi": 0.860, "p_active": 24.0, "p_reactive": 14.5, "tension": 10.5,
        "posit_hp": 84.5, "posit_bp": 74.0, "vap_inlet": 56.1,
        "cond_temp": 248.0, "cond_eau": 88.0, "level_pct": 77.0
    }
}

drift_factors = {"GTA1": 0.0, "GTA2": 0.0, "GTA3": 0.0}

# Immutable design-point reference (base_params gets mutated by Manual Tuning)
NOMINAL_PARAMS = copy.deepcopy(base_params)

# Condenser conditions shared by the 3 GTAs
_T_COND_K   = 45.2 + 273.15     # condensate temperature (K)
_P_EXHAUST  = 0.09              # condenser vacuum (bar abs)


def compute_rendement(gta: str, adm_debit: float, adm_temp: float,
                      adm_pression: float) -> float:
    """
    Thermodynamic efficiency model — rendement responds to the admission
    conditions instead of being a static value.

        η = η₀ · f_charge(ṁ/ṁ₀) · f_T(T_adm) · f_P(P_adm)

    • f_charge — part-load (Willans) penalty: a turbine is designed for its
      rated steam flow; running below (or above) it degrades efficiency
      quadratically:  f = 1 − 0.25·max(0, 1−L)² − 0.08·max(0, L−1)²
    • f_T — Carnot-like temperature factor: the ideal cycle efficiency is
      1 − T_cond/T_adm (in Kelvin); normalised by its value at the design
      temperature, so hotter admission steam ⇒ better rendement.
    • f_P — pressure factor: the isentropic enthalpy drop grows with the
      expansion ratio ≈ ln(P_adm/P_exhaust); normalised at design pressure,
      so higher admission pressure ⇒ more recoverable energy per tonne.

    η₀, ṁ₀, T₀, P₀ are each GTA's design point (NOMINAL_PARAMS).
    Result clipped to the physical plausibility band [20 %, 55 %].
    """
    ref = NOMINAL_PARAMS[gta]

    # Charge (steam flow) factor
    load = max(0.1, float(adm_debit)) / ref["adm_debit"]
    f_charge = 1.0 - 0.25 * max(0.0, 1.0 - load) ** 2 \
                   - 0.08 * max(0.0, load - 1.0) ** 2

    # Carnot temperature factor (Kelvin)
    t_k, t0_k = float(adm_temp) + 273.15, ref["adm_temp"] + 273.15
    f_temp = (1.0 - _T_COND_K / t_k) / (1.0 - _T_COND_K / t0_k)

    # Expansion-ratio pressure factor
    p = max(1.0, float(adm_pression))
    f_pres = math.log(p / _P_EXHAUST) / math.log(ref["adm_pression"] / _P_EXHAUST)

    eta = ref["rendement"] * f_charge * f_temp * f_pres
    return round(max(20.0, min(55.0, eta)), 2)


def generate_data_point(scenario: str, gta: str, params: dict):
    global drift_factors
    data = params.copy()
    noise = np.random.normal(0, 0.35)

    # Scenario-driven mechanical degradation on top of the thermodynamics
    # (fouling, bearing wear, ...) expressed as an efficiency penalty in points.
    rend_penalty = 0.0
    if scenario != "normal":
        drift_factors[gta] = min(1.0, drift_factors[gta] + 0.012)
        intensity = drift_factors[gta]

        if scenario == "mild_anomaly":
            rend_penalty = 2.5 * intensity
            data["adm_debit"] *= (1 - 0.035 * intensity)
            data["vib1"] += 0.8 * intensity
            data["vib2"] += 0.6 * intensity
            data["oil_temp"] += 4.2 * intensity

        elif scenario == "severe_anomaly":
            rend_penalty = 8.0 * intensity
            data["adm_debit"] *= (1 - 0.085 * intensity)
            data["vib1"] += 3.2 * intensity
            data["vib2"] += 2.8 * intensity
            data["oil_temp"] += 15.5 * intensity
            data["oil_pression"] -= 0.35 * intensity
            data["vitesse"] -= 210 * intensity
            data["level_pct"] -= 12 * intensity

        elif scenario == "fouling":
            rend_penalty = 3.5 * intensity
            data["adm_temp"] += 12 * intensity
            data["oil_pression"] -= 0.15 * intensity
            data["posit_hp"] += 5.0 * intensity

    # Apply dynamic noises to simulated metrics
    data["adm_temp"] += noise
    data["adm_pression"] += noise * 0.05
    data["vitesse"] = max(0.0, data["vitesse"] + np.random.normal(0, 4))
    data["vib1"] = max(0.0, data["vib1"] + np.random.normal(0, 0.015))
    data["vib2"] = max(0.0, data["vib2"] + np.random.normal(0, 0.015))
    data["oil_temp"] += np.random.normal(0, 0.08)

    # ── Rendement from the thermodynamic model ────────────────────────────────
    # Manual Tuning (débit / T° / P admission) and scenario drifts flow through
    # the efficiency formula instead of a static baseline value.
    eta = compute_rendement(gta, data["adm_debit"], data["adm_temp"],
                            data["adm_pression"])
    eta = eta - rend_penalty + float(np.random.normal(0, 0.12))
    data["rendement"] = round(max(20.0, min(55.0, eta)), 2)

    # ── Power output: P = ṁ · k · (η / η₀) ────────────────────────────────────
    # Base MW/(t/h) ratios from the historical datasets, scaled by the current
    # relative efficiency so power reacts to both steam flow AND rendement.
    ratios = {"GTA1": 0.128, "GTA2": 0.097, "GTA3": 0.123}
    ratio = ratios.get(gta, 0.128)
    eta_rel = data["rendement"] / NOMINAL_PARAMS[gta]["rendement"]

    data["puissance_mw"] = round(data["adm_debit"] * ratio * eta_rel, 2)
    data["p_active"] = round(data["puissance_mw"], 2)
    data["p_reactive"] = round(data["p_active"] * 0.605, 2)

    data["anomaly_flag"] = 1 if scenario != "normal" else 0
    data["timestamp"] = datetime.now().isoformat()

    return data


def simulation_loop(simulation_id):
    global simulation_running, current_scenario, base_params, drift_factors
    for gta in drift_factors:
        drift_factors[gta] = 0.0
        
    init_db()
    
    conn = get_connection()
    conn.execute("INSERT INTO simulations VALUES (?, ?, ?, ?, ?)",
                 (simulation_id, current_scenario, json.dumps(base_params), 
                  datetime.now().isoformat(), "running"))
    conn.commit()
    
    columns = [
        "simulation_id", "timestamp", "gta_type", "adm_debit", "adm_temp", "adm_pression",
        "sout_debit", "sout_pression", "ext_debit", "ext_pression", "bp_pression", "bp_debit",
        "puissance_mw", "rendement", "vitesse", "vib1", "vib2", "dd3", "oil_pression", "oil_temp",
        "cos_phi", "p_active", "p_reactive", "tension", "posit_hp", "posit_bp", "vap_inlet",
        "cond_temp", "cond_eau", "level_pct", "anomaly_flag"
    ]
    placeholders = ", ".join(["?"] * len(columns))
    insert_sql = f"INSERT INTO simulation_data ({', '.join(columns)}) VALUES ({placeholders})"
    
    while simulation_running:
        for gta in ["GTA1", "GTA2", "GTA3"]:
            point = generate_data_point(current_scenario, gta, base_params[gta])
            
            values = (
                simulation_id, point["timestamp"], gta, point["adm_debit"], point["adm_temp"], point["adm_pression"],
                point["sout_debit"], point["sout_pression"], point["ext_debit"], point["ext_pression"], 
                point["bp_pression"], point["bp_debit"], point["puissance_mw"], point["rendement"], point["vitesse"], 
                point["vib1"], point["vib2"], point["dd3"], point["oil_pression"], point["oil_temp"], point["cos_phi"], 
                point["p_active"], point["p_reactive"], point["tension"], point["posit_hp"], point["posit_bp"], 
                point["vap_inlet"], point["cond_temp"], point["cond_eau"], point["level_pct"], point["anomaly_flag"]
                )
            conn.execute(insert_sql, values)
        conn.commit()
        time.sleep(1)
        
    conn.close()


# ====================== LAYOUT ======================
app.layout = dbc.Container([
    html.H1("🛠️ OCP Jorf Lasfar - GTA SCADA Live Simulator", className="text-center my-4"),
    
    # Live Summary for all 3 GTAs
    dbc.Row([
        dbc.Col(dbc.Card(id="card-gta1", className="bg-dark text-white border-secondary mb-3"), width=4),
        dbc.Col(dbc.Card(id="card-gta2", className="bg-dark text-white border-secondary mb-3"), width=4),
        dbc.Col(dbc.Card(id="card-gta3", className="bg-dark text-white border-secondary mb-3"), width=4),
    ]),
    
    dbc.Tabs(active_tab="tab-live", children=[
        dbc.Tab(label="🎛️ Live SCADA Panel", tab_id="tab-live", children=[
            dbc.Row([
                dbc.Col(width=3, children=[
                    dbc.Card([
                        dbc.CardHeader("Control Panel"),
                        dbc.CardBody([
                            dbc.Label("Active GTA"),
                            dcc.Dropdown(id="gta-dropdown", 
                                options=[
                                    {"label": "GTA 1", "value": "GTA1"},
                                    {"label": "GTA 2", "value": "GTA2"},
                                    {"label": "GTA 3", "value": "GTA3"},
                                ],
                                value="GTA1", className="mb-3"),
                            
                            dbc.Label("Scenario"),
                            dcc.Dropdown(id="scenario-dropdown", 
                                options=[
                                    {"label": "🟢 Normal", "value": "normal"},
                                    {"label": "🟡 Mild Anomaly", "value": "mild_anomaly"},
                                    {"label": "🔴 Severe Anomaly", "value": "severe_anomaly"},
                                    {"label": "🔧 Fouling", "value": "fouling"},
                                ],
                                value="normal", className="mb-3"),
                            
                            html.Hr(),
                            html.H5("Manual Tuning"),
                            dbc.Label("Debit Admission (t/h)"),
                            dcc.Slider(id="slider-debit", min=100, max=250, value=195, step=1,
                                       marks={100: "100", 175: "175", 250: "250"},
                                       tooltip={"placement": "bottom", "always_visible": True}),

                            dbc.Label("T° Admission (°C)", className="mt-2"),
                            dcc.Slider(id="slider-t", min=400, max=480, value=460, step=1,
                                       marks={400: "400", 440: "440", 480: "480"},
                                       tooltip={"placement": "bottom", "always_visible": True}),

                            dbc.Label("P Admission (bar)", className="mt-2"),
                            dcc.Slider(id="slider-p", min=50, max=60, value=54.6, step=0.1,
                                       marks={50: "50", 55: "55", 60: "60"},
                                       tooltip={"placement": "bottom", "always_visible": True}),

                            # Live preview of the thermodynamic efficiency formula
                            html.Div(id="rendement-preview", className="mt-3"),

                            html.Button("Apply Changes", id="apply-btn", className="btn btn-warning w-100 mt-3"),
                        ])
                    ], className="mb-3"),
                ]),
                
                dbc.Col(width=9, children=[
                    dcc.Graph(id="rendement-graph", style={"height": "45vh"}),
                    dcc.Graph(id="debit-graph", style={"height": "45vh"}),
                ])
            ])
        ]),
        
        dbc.Tab(label="📊 History", tab_id="tab-history",
                children=[html.Div(id="history-div", className="p-3")])
    ]),
    
    dcc.Interval(id='interval', interval=1000, n_intervals=0),
    dcc.Store(id='store', data={"sim_id": current_sim_id, "active_gta": active_gta})
], fluid=True)


# ====================== CALLBACKS ======================

@app.callback(
    [Output('card-gta1', 'children'), Output('card-gta2', 'children'), Output('card-gta3', 'children')],
    Input('interval', 'n_intervals')
)
def update_summary_cards(n):
    conn = get_connection()
    results = []
    for gta in ["GTA1", "GTA2", "GTA3"]:
        df = pd.read_sql(f"""
            SELECT rendement, puissance_mw, anomaly_flag 
            FROM simulation_data 
            WHERE simulation_id = '{current_sim_id}' AND gta_type = '{gta}'
            ORDER BY timestamp DESC LIMIT 1
        """, conn)
        if not df.empty:
            rend = df.iloc[0]['rendement']
            p = df.iloc[0]['puissance_mw']
            anomaly = df.iloc[0]['anomaly_flag']
            
            # Thresholds adjusted per GTA baseline (GTA2 is ~35%, GTA1/3 are ~40-42%)
            baseline_rend = 35.0 if gta == "GTA2" else 41.0
            warning_rend = baseline_rend - 3.0
            danger_rend = baseline_rend - 6.0
            
            color = "success" if rend >= warning_rend else "warning" if rend >= danger_rend else "danger"
            status_icon = "⚠️" if anomaly else "✅"
            
            card_content = [
                dbc.CardHeader(f"{status_icon} {gta} Status"),
                dbc.CardBody([
                    html.H4(f"Rendement: {rend:.2f}%", className=f"text-{color} fw-bold"),
                    html.H5(f"Puissance: {p:.2f} MW", className="text-info"),
                ])
            ]
        else:
            card_content = [dbc.CardBody(html.P("Initializing...", className="text-muted"))]
        results.append(dbc.Card(card_content, className="bg-dark text-white border-secondary"))
    conn.close()
    return results


@app.callback(
    Output('store', 'data'),
    Input('scenario-dropdown', 'value'),
    prevent_initial_call=True
)
def update_scenario(value):
    global current_scenario, drift_factors, active_gta
    current_scenario = value
    if value == "normal":
        for gta in drift_factors:
            drift_factors[gta] = 0.0
    return {"sim_id": current_sim_id, "active_gta": active_gta}


@app.callback(
    Output('store', 'data', allow_duplicate=True),
    Input('gta-dropdown', 'value'),
    prevent_initial_call=True
)
def update_active_gta(value):
    global active_gta
    active_gta = value
    return {"sim_id": current_sim_id, "active_gta": active_gta}


@app.callback(
    Output('store', 'data', allow_duplicate=True),
    Input('apply-btn', 'n_clicks'),
    [State('gta-dropdown', 'value'), State('slider-debit', 'value'), State('slider-t', 'value'), State('slider-p', 'value')],
    prevent_initial_call=True
)
def apply_parameters(n, gta, debit, t, p):
    global base_params, active_gta
    if n:
        active_gta = gta
        base_params[gta]["adm_debit"] = float(debit)
        base_params[gta]["adm_temp"] = float(t)
        base_params[gta]["adm_pression"] = float(p)
    return {"sim_id": current_sim_id, "active_gta": active_gta}


@app.callback(
    [Output('slider-debit', 'value'), Output('slider-t', 'value'), Output('slider-p', 'value')],
    Input('gta-dropdown', 'value'),
    prevent_initial_call=True
)
def update_sliders(gta):
    global active_gta
    active_gta = gta
    return base_params[gta]["adm_debit"], base_params[gta]["adm_temp"], base_params[gta]["adm_pression"]


@app.callback(
    Output('rendement-preview', 'children'),
    [Input('slider-debit', 'value'), Input('slider-t', 'value'),
     Input('slider-p', 'value'), Input('gta-dropdown', 'value')]
)
def preview_rendement(debit, t, p, gta):
    """Live evaluation of the efficiency formula while tuning the sliders,
    before the changes are applied to the running simulation."""
    if None in (debit, t, p) or gta not in NOMINAL_PARAMS:
        return ""
    eta = compute_rendement(gta, float(debit), float(t), float(p))
    eta0 = NOMINAL_PARAMS[gta]["rendement"]
    delta = eta - eta0
    color = "success" if delta >= 0 else ("warning" if delta > -3 else "danger")
    return dbc.Alert([
        html.Strong(f"η calculé ({gta}) : {eta:.2f} % "),
        html.Span(f"({delta:+.2f} pt vs design {eta0:.1f} %)",
                  className="small"),
        html.Br(),
        
    ], color=color, className="py-2 px-3 mb-0")


@app.callback(
    [Output('rendement-graph', 'figure'), Output('debit-graph', 'figure')],
    [Input('interval', 'n_intervals'), Input('store', 'data')]
)
def update_graphs(n, store_data):
    if not store_data:
        raise PreventUpdate
    gta = store_data.get("active_gta", "GTA1")
    
    conn = get_connection()
    df = pd.read_sql(f"""
        SELECT timestamp, rendement, adm_debit 
        FROM simulation_data 
        WHERE simulation_id = '{current_sim_id}' AND gta_type = '{gta}'
        ORDER BY timestamp DESC LIMIT 150
    """, conn)
    conn.close()
    
    if df.empty:
        raise PreventUpdate
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig_r = go.Figure(go.Scatter(x=df['timestamp'], y=df['rendement'], mode='lines+markers', line=dict(color='orange')))
    fig_r.update_layout(title=f"Rendement (%) - Live ({gta})", template="plotly_dark", height=380)
    
    fig_d = go.Figure(go.Scatter(x=df['timestamp'], y=df['adm_debit'], mode='lines+markers', line=dict(color='#00ffcc')))
    fig_d.update_layout(title=f"Débit Admission (t/h) - Live ({gta})", template="plotly_dark", height=380)
    
    return fig_r, fig_d


@app.callback(Output('history-div', 'children'), Input('interval', 'n_intervals'))
def show_history(n):
    conn = get_connection()
    df = pd.read_sql("SELECT simulation_id, scenario_type, start_time FROM simulations ORDER BY start_time DESC LIMIT 8", conn)
    conn.close()
    
    if df.empty:
        return html.P("No simulations yet.", className="text-muted")
    
    table = dbc.Table.from_dataframe(
        df, 
        striped=True, 
        bordered=True, 
        hover=True,
        className="table-dark"
    )
    return table

# ====================== LIVE API ENDPOINT ======================
from flask import jsonify

@app.server.route('/api/live/<gta>')
def get_live_data(gta):
    """
    API endpoint to fetch the latest live simulation data for a specific GTA.
    Used by the GTA Visualization module to render the live SVG diagram.
    """
    try:
        conn = get_connection()
        query = """
            SELECT * FROM simulation_data 
            WHERE gta_type = ? 
            ORDER BY id DESC LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(gta,))
        conn.close()
        
        if df.empty:
            return jsonify({"gta": gta, "data": []})
        
        # Convert the row to a dictionary
        row = df.iloc[0].to_dict()
        return jsonify({"gta": gta, "data": [row]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====================== START ======================
if __name__ == '__main__':
    import os
    init_db()
    # Stable single-process service. Set SCADA_DEBUG=1 for the dev reloader —
    # in that mode __main__ runs in BOTH the watcher parent and the serving
    # child, so the simulation loop must only start in the child (otherwise a
    # zombie parent thread keeps feeding the DB with stale base_params that
    # Manual Tuning can never reach).
    DEBUG = os.environ.get("SCADA_DEBUG", "0") == "1"
    if (not DEBUG) or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=simulation_loop, args=(current_sim_id,),
                         daemon=True).start()
    app.run(host='0.0.0.0', debug=DEBUG, port=8051, threaded=True)