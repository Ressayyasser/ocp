"""
pages/digital_twin.py — Digital Twin What-If & Monte Carlo Simulation Dashboard.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import run_simulation, compare_scenarios, run_monte_carlo

dash.register_page(__name__, path="/digital-twin", name="Digital Twin", title="Digital Twin")

# Available variables for simulation
SIMULATION_VARIABLES = [
    {"label": "GTA1 Production", "value": "gta1"},
    {"label": "GTA2 Production", "value": "gta2"},
    {"label": "GTA3 Production", "value": "gta3"},
    {"label": "Steam HP", "value": "steam_hp"},
    {"label": "Steam MP", "value": "steam_mp"},
    {"label": "Pressure", "value": "pressure"},
    {"label": "Temperature", "value": "temperature"},
    {"label": "Vibration", "value": "vibration"},
]

def layout():
    return dbc.Container([
        html.H2("🔮 Digital Twin — What-If Scenario Simulation", className="text-light my-4"),
        
        # Tabbed interface
        dcc.Tabs(id="sim-tabs", value="tab-whatif", className="mb-4", children=[
            dcc.Tab(label="📊 What-If Simulation", value="tab-whatif", 
                    className="text-light", selected_className="text-warning"),
            dcc.Tab(label="📈 Scenario Comparison", value="tab-compare",
                    className="text-light", selected_className="text-warning"),
            dcc.Tab(label="🎲 Monte Carlo Analysis", value="tab-monte-carlo",
                    className="text-light", selected_className="text-warning"),
        ]),
        
        # What-If Simulation Tab
        html.Div(id="tab-whatif-content", className="mb-4"),
        
        # Comparison Tab
        html.Div(id="tab-compare-content", className="mb-4"),
        
        # Monte Carlo Tab
        html.Div(id="tab-monte-carlo-content", className="mb-4"),
        
        # Results display
        dbc.Card([
            dbc.CardHeader("📊 Simulation Results", className="text-light"),
            dbc.CardBody([
                html.Div(id="simulation-results"),
            ]),
        ], className="bg-dark border-secondary mb-4"),
    ], fluid=True, className="bg-dark text-light min-vh-100 p-4")


# What-If Tab Content
@callback(
    Output("tab-whatif-content", "children"),
    Input("sim-tabs", "value")
)
def render_whatif_tab(active_tab):
    if active_tab != "tab-whatif":
        return html.Div()
    
    return dbc.Card([
        dbc.CardHeader("🔧 Single Scenario Simulation", className="text-light"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Variable to Modify", className="text-light"),
                    dcc.Dropdown(
                        id="whatif-variable",
                        options=SIMULATION_VARIABLES,
                        value="gta3",
                        clearable=False,
                        className="mb-3",
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Change Percentage (%)", className="text-light"),
                    dcc.Slider(
                        id="whatif-change",
                        min=-50, max=50, step=5, value=15,
                        marks={-50: "-50%", 0: "0%", 50: "+50%"},
                        className="mb-3",
                    ),
                    html.Div(id="whatif-change-value", className="text-warning fw-bold"),
                ], width=4),
                dbc.Col([
                    html.Label("Duration (hours)", className="text-light"),
                    dcc.Slider(
                        id="whatif-duration",
                        min=1, max=168, step=1, value=24,
                        marks={1: "1h", 24: "24h", 72: "72h", 168: "168h"},
                        className="mb-3",
                    ),
                    html.Div(id="whatif-duration-value", className="text-warning fw-bold"),
                ], width=4),
            ]),
            dbc.Button("🚀 Run Simulation", id="run-whatif-btn", color="warning", size="lg", className="w-100"),
            html.Div(id="whatif-loading", children=dcc.Loading(id="loading-whatif", type="circle", children=[]), style={"display": "none"}),
        ]),
    ], className="bg-dark border-secondary")


# Comparison Tab Content
@callback(
    Output("tab-compare-content", "children"),
    Input("sim-tabs", "value")
)
def render_compare_tab(active_tab):
    if active_tab != "tab-compare":
        return html.Div()
    
    # Create 3 scenario input groups
    scenario_inputs = []
    for i in range(1, 4):
        scenario_inputs.append(
            dbc.Card([
                dbc.CardHeader(f"Scenario {i}", className="text-light bg-secondary"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Variable", className="text-light small"),
                            dcc.Dropdown(
                                id=f"compare-var-{i}",
                                options=SIMULATION_VARIABLES,
                                value="gta3" if i == 1 else "steam_hp",
                                clearable=False,
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Change %", className="text-light small"),
                            dcc.Input(
                                id=f"compare-change-{i}",
                                type="number",
                                value=15 if i == 1 else (-10 if i == 2 else 5),
                                className="form-control bg-dark text-light",
                            ),
                        ], width=6),
                    ]),
                ]),
            ], className="bg-dark border-secondary mb-3")
        )
    
    return dbc.Card([
        dbc.CardHeader("📊 Multi-Scenario Comparison", className="text-light"),
        dbc.CardBody([
            *scenario_inputs,
            dbc.Button("🔍 Compare Scenarios", id="run-compare-btn", color="info", size="lg", className="w-100"),
        ]),
    ], className="bg-dark border-secondary")


# Monte Carlo Tab Content
@callback(
    Output("tab-monte-carlo-content", "children"),
    Input("sim-tabs", "value")
)
def render_monte_carlo_tab(active_tab):
    if active_tab != "tab-monte-carlo":
        return html.Div()
    
    return dbc.Card([
        dbc.CardHeader("🎲 Monte Carlo Uncertainty Analysis", className="text-light"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Variable to Modify", className="text-light"),
                    dcc.Dropdown(
                        id="mc-variable",
                        options=SIMULATION_VARIABLES,
                        value="steam_hp",
                        clearable=False,
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Change Percentage (%)", className="text-light"),
                    dcc.Input(
                        id="mc-change",
                        type="number",
                        value=-10,
                        className="form-control bg-dark text-light",
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Duration (hours)", className="text-light"),
                    dcc.Input(
                        id="mc-duration",
                        type="number",
                        value=48,
                        className="form-control bg-dark text-light",
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Number of Trials", className="text-light"),
                    dcc.Input(
                        id="mc-trials",
                        type="number",
                        value=500,
                        min=100, max=5000, step=100,
                        className="form-control bg-dark text-light",
                    ),
                ], width=2),
            ], className="mb-3"),
            dbc.Button("🎯 Run Monte Carlo", id="run-mc-btn", color="success", size="lg", className="w-100"),
            html.Div(id="mc-loading", children=dcc.Loading(id="loading-mc", type="circle", children=[]), style={"display": "none"}),
        ]),
    ], className="bg-dark border-secondary")


# Callback for What-If Simulation
@callback(
    Output("simulation-results", "children"),
    Output("whatif-loading", "style"),
    Input("run-whatif-btn", "n_clicks"),
    State("whatif-variable", "value"),
    State("whatif-change", "value"),
    State("whatif-duration", "value"),
    prevent_initial_call=True,
)
def run_whatif_simulation(n_clicks, variable, change_percent, duration_hours):
    if not n_clicks:
        return [], {"display": "none"}
    
    # Show loading
    loading_style = {"display": "block"}
    
    # Run simulation
    result = run_simulation(variable, change_percent, duration_hours)
    
    if "error" in result:
        return dbc.Alert(f"Error: {result['error']}", color="danger"), {"display": "none"}
    
    # Create visualizations
    figs = create_simulation_visualizations(result)
    
    # Create results cards
    results_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Predicted Bilan Change", className="text-muted"),
                    html.H4(f"{result['predicted_bilan_change']:+,.2f} MWh", 
                           className="text-success fw-bold"),
                ])
            ], className="bg-dark border-success"),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Predicted Production Change", className="text-muted"),
                    html.H4(f"{result['predicted_production_change']:+,.2f} MWh", 
                           className="text-info fw-bold"),
                ])
            ], className="bg-dark border-info"),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Risk Score", className="text-muted"),
                    html.H4(f"{result['risk_score']:.2%}", 
                           className="text-warning fw-bold"),
                ])
            ], className="bg-dark border-warning"),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Efficiency Change", className="text-muted"),
                    html.H4(f"{result['predicted_efficiency_change']:+,.2f}%", 
                           className="text-primary fw-bold"),
                ])
            ], className="bg-dark border-primary"),
        ], width=3),
    ], className="mb-4")
    
    return html.Div([
        html.H4(f"📋 Scenario: {result['scenario']}", className="text-light mb-3"),
        results_cards,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=figs['bar'], className="bg-dark"), width=6),
            dbc.Col(dcc.Graph(figure=figs['radar'], className="bg-dark"), width=6),
        ]),
        dbc.Alert(result['recommendation'], color="success" if result['predicted_bilan_change'] > 0 else "warning", className="mt-3"),
    ]), {"display": "none"}


# Callback for Scenario Comparison
@callback(
    Output("simulation-results", "children", allow_duplicate=True),
    Input("run-compare-btn", "n_clicks"),
    [State(f"compare-var-{i}", "value") for i in range(1, 4)],
    [State(f"compare-change-{i}", "value") for i in range(1, 4)],
    prevent_initial_call=True,
)
def run_comparison(n_clicks, *args):
    if not n_clicks:
        return []
    
    # Parse arguments
    variables = args[:3]
    changes = args[3:6]
    
    scenarios = [
        {"variable": var, "change_percent": change, "duration_hours": 24}
        for var, change in zip(variables, changes)
    ]
    
    results = compare_scenarios(scenarios)
    
    if "error" in results:
        return dbc.Alert(f"Error: {results['error']}", color="danger")
    
    # Create comparison visualizations
    fig = create_comparison_chart(results)
    
    # Create comparison table
    table_rows = []
    for i, result in enumerate(results, 1):
        table_rows.append(html.Tr([
            html.Td(f"Scenario {i}: {result['scenario']}"),
            html.Td(f"{result['predicted_bilan_change']:+,.2f} MWh", 
                   className="text-success" if result['predicted_bilan_change'] > 0 else "text-danger"),
            html.Td(f"{result['predicted_production_change']:+,.2f} MWh"),
            html.Td(f"{result['risk_score']:.2%}"),
            html.Td(result['recommendation'][:50] + "..."),
        ]))
    
    return html.Div([
        html.H4("📊 Scenario Comparison Results", className="text-light mb-3"),
        dcc.Graph(figure=fig, className="bg-dark mb-4"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Scenario"),
                html.Th("Bilan Change"),
                html.Th("Production Change"),
                html.Th("Risk Score"),
                html.Th("Recommendation"),
            ])),
            html.Tbody(table_rows),
        ], bordered=True, hover=True, className="table-dark"),
    ])


# Callback for Monte Carlo Simulation
@callback(
    Output("simulation-results", "children", allow_duplicate=True),
    Output("mc-loading", "style"),
    Input("run-mc-btn", "n_clicks"),
    State("mc-variable", "value"),
    State("mc-change", "value"),
    State("mc-duration", "value"),
    State("mc-trials", "value"),
    prevent_initial_call=True,
)
def run_monte_carlo_sim(n_clicks, variable, change_percent, duration_hours, n_trials):
    if not n_clicks:
        return [], {"display": "none"}
    
    loading_style = {"display": "block"}
    
    result = run_monte_carlo(variable, change_percent, duration_hours, n_trials)
    
    if "error" in result:
        return dbc.Alert(f"Error: {result['error']}", color="danger"), {"display": "none"}
    
    # Create Monte Carlo visualizations
    figs = create_monte_carlo_visualizations(result)
    
    # Create summary statistics cards
    summary = result.get('summary', {})
    
    cards = []
    for metric, stats in list(summary.items())[:4]:  # Show top 4 metrics
        cards.append(dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6(metric.replace('_', ' ').title(), className="text-muted small"),
                    html.H5(f"Mean: {stats['mean']:,.2f}", className="text-info"),
                    html.P(f"Std: {stats['std']:,.2f}", className="text-muted mb-0"),
                    html.Small(f"P5: {stats['p5']:,.2f} | P95: {stats['p95']:,.2f}", className="text-warning"),
                ])
            ], className="bg-dark border-secondary")
        ], width=3))
    
    return html.Div([
        html.H4(f"🎲 Monte Carlo Analysis ({n_trials} trials)", className="text-light mb-3"),
        dbc.Row(cards, className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=figs['histogram'], className="bg-dark"), width=6),
            dbc.Col(dcc.Graph(figure=figs['box'], className="bg-dark"), width=6),
        ]),
        html.P(f"💡 The confidence intervals show the range of possible outcomes. "
               f"There is a 90% probability the actual result will fall between P5 and P95.", 
               className="text-info mt-3"),
    ]), {"display": "none"}


# Visualization Helper Functions
def create_simulation_visualizations(result):
    """Create bar chart and radar chart for single simulation."""
    
    # Extract the changes dictionary
    changes = result.get('all_changes', {})
    
    # 🔥 FIX: Filter out None values and ensure all values are floats
    # This prevents the TypeError and ensures Plotly can plot the y-axis correctly
    changes = {k: float(v) for k, v in changes.items() if v is not None}
    
    # Handle the edge case where the simulator returns no valid changes
    if not changes:
        fig_empty = go.Figure()
        fig_empty.update_layout(
            title="No variable changes calculated by the simulator.",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        return {'bar': fig_empty, 'radar': fig_empty}
    
    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=list(changes.keys()),
        y=list(changes.values()),
        # Now safe to compare because all values are guaranteed to be floats
        marker_color=['green' if v > 0 else 'red' for v in changes.values()],
    ))
    fig_bar.update_layout(
        title="Impact on All Variables",
        xaxis_title="Variable",
        yaxis_title="Change",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )
    
    # ── Radar chart ───────────────────────────────────────────────────────────
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[abs(v) for v in changes.values()],
        theta=list(changes.keys()),
        fill='toself',
        line_color='#00ffcc',
    ))
    
    # Calculate max value for radial axis safely (prevents error if all values are 0)
    max_val = max(abs(v) for v in changes.values())
    radial_range = [0, max_val * 1.2] if max_val > 0 else [0, 1]
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=radial_range)),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )
    
    return {'bar': fig_bar, 'radar': fig_radar}


def create_comparison_chart(results):
    """Create comparison bar chart."""
    df = pd.DataFrame([
        {
            'Scenario': f"Scenario {i+1}: {r['scenario']}",
            'Bilan Change': r['predicted_bilan_change'],
            'Production Change': r['predicted_production_change'],
            'Risk Score': r['risk_score'] * 100,
        }
        for i, r in enumerate(results)
    ])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Bilan Change', x=df['Scenario'], y=df['Bilan Change'], marker_color='green'))
    fig.add_trace(go.Bar(name='Production Change', x=df['Scenario'], y=df['Production Change'], marker_color='blue'))
    fig.update_layout(
        title="Scenario Comparison",
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )
    return fig


def create_monte_carlo_visualizations(result):
    """Create histogram and box plot for Monte Carlo results."""
    summary = result.get('summary', {})
    
    # Get the main metric (bilan change)
    if 'predicted_bilan_change' in summary:
        metric = 'predicted_bilan_change'
        stats = summary[metric]
        
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=[stats['p5'], stats['median'], stats['p95']],
            marker_color='rgba(0, 255, 204, 0.7)',
            name='Distribution',
        ))
        fig_hist.add_vline(x=stats['mean'], line_dash="dash", line_color="red", annotation_text="Mean")
        fig_hist.update_layout(
            title=f"Distribution of {metric.replace('_', ' ').title()}",
            xaxis_title="Value",
            yaxis_title="Frequency",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=[stats['p5'], stats['p95']],
            name=metric,
            marker_color='rgba(0, 255, 204, 0.7)',
        ))
        fig_box.update_layout(
            title="Confidence Intervals",
            yaxis_title="Value",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        
        return {'histogram': fig_hist, 'box': fig_box}
    
    return {'histogram': go.Figure(), 'box': go.Figure()}


# Update slider value displays
@callback(
    Output("whatif-change-value", "children"),
    Input("whatif-change", "value")
)
def update_change_value(value):
    return f"{value:+d}%"

@callback(
    Output("whatif-duration-value", "children"),
    Input("whatif-duration", "value")
)
def update_duration_value(value):
    return f"{value} hours"