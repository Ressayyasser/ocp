import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import requests
import os

dash.register_page(__name__, path="/rul", name="RUL & Health", title="RUL")

# Adjust this to match your FastAPI backend URL
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

def layout():
    return dbc.Container([
        html.H2("🔧 Remaining Useful Life (RUL) & Health Monitoring", className="text-light my-4"),
        
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Total Hours Operated:", className="text-light"),
                dcc.Input(id="hours-input", type="number", value=10000, className="form-control bg-dark text-light"),
            ], width=3),
            dbc.Col([
                html.Button("Refresh Data", id="refresh-btn", n_clicks=0, className="btn btn-primary mt-4"),
            ], width=2),
        ], className="mb-4"),
        
        # Most Critical Component Alert
        html.Div(id="critical-alert", className="mb-4"),
        
        # Component Cards Grid
        html.Div(id="components-grid"),
        
        # Interval for auto-refresh (every 10 seconds)
        dcc.Interval(id="interval-component", interval=10*1000, n_intervals=0),
        dcc.Store(id="rul-data-store")
    ], fluid=True, className="bg-dark text-light min-vh-100 p-4")

@callback(
    Output("rul-data-store", "data"),
    Input("refresh-btn", "n_clicks"),
    Input("interval-component", "n_intervals"),
    Input("hours-input", "value"),
    prevent_initial_call=False
)
def fetch_rul_data(n_clicks, n_intervals, hours):
    try:
        response = requests.get(f"{API_BASE_URL}/rul/summary", params={"hours_operated": hours or 10000})
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching RUL data: {e}")
    return None

# ... [Keep all your imports and the layout() function exactly as they are] ...

@callback(
    Output("critical-alert", "children"),
    Output("components-grid", "children"),
    Input("rul-data-store", "data")
)
def render_rul_dashboard(data):
    if not data:
        return dbc.Alert("Could not fetch RUL data. Check backend connection.", color="danger"), []
    
    # Handle backend errors gracefully (e.g., if both DB tables are empty)
    if "error" in data:
        return dbc.Alert(f"⚠️ {data['error']}", color="warning"), []
    
    most_critical = data.get("most_critical", "Unknown")
    health_scores = data.get("health_scores", {})  # <--- FIXED KEY
    rul_data = data.get("rul", {})
    
    if not health_scores or not rul_data:
        return dbc.Alert("No health data available to display.", color="warning"), []
    
    # 1. Critical Alert Banner
    critical_health = health_scores.get(most_critical, 100)
    critical_alert = dbc.Alert(
        f"⚠️ Most Critical Component: {most_critical.upper()} (Health: {critical_health:.1f}%)",
        color="warning" if critical_health > 30 else "danger",
        className="fw-bold"
    )
    
    # 2. Component Health Cards
    cards = []
    for component, rul_info in rul_data.items():
        health = rul_info.get("health_score", 100)
        rul_days = rul_info.get("rul_days", 0)
        status = rul_info.get("status", "healthy")
        
        # Color mapping based on status
        if status == "healthy":
            border_color = "success"
            text_color = "text-success"
            progress_color = "success"
        elif status == "warning":
            border_color = "warning"
            text_color = "text-warning"
            progress_color = "warning"
        else:
            border_color = "danger"
            text_color = "text-danger"
            progress_color = "danger"
            
        card = dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5(component, className="text-center mb-0 text-light")),
                dbc.CardBody([
                    html.H3(f"{health:.1f}%", className=f"{text_color} text-center fw-bold"),
                    dbc.Progress(value=health, color=progress_color, className="mb-3"),
                    html.Hr(className="bg-secondary"),
                    html.P([
                        html.Strong("RUL: "),
                        html.Span(f"{rul_days:.0f} days", className="text-light")
                    ], className="text-center"),
                    html.P([
                        html.Strong("Status: "),
                        html.Span(status.upper(), className=f"{text_color} fw-bold")
                    ], className="text-center text-uppercase")
                ])
            ], className=f"border-{border_color} shadow-sm h-100 bg-dark")
        ], md=4, sm=6, xs=12, className="mb-4")
        cards.append(card)
        
    return critical_alert, dbc.Row(cards)