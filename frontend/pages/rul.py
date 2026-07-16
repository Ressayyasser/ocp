import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import requests
import os
import numpy as np

dash.register_page(__name__, path="/rul", name="RUL & Health", title="RUL")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


# ── 3D Component Geometry Generators ──────────────────────────────────────────

def _health_color(health: float) -> str:
    """Returns a color based on health score (red → yellow → green)."""
    if health >= 70:
        return "#28a745"  # green
    elif health >= 40:
        return "#ffc107"  # yellow
    else:
        return "#dc3545"  # red


def _create_turbine_3d(health: float) -> go.Figure:
    """3D visualization of a Turbine (HP) with blades."""
    color = _health_color(health)
    
    # Generate turbine body (cylinder)
    theta = np.linspace(0, 2*np.pi, 30)
    z = np.linspace(-1, 1, 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    R = 0.8  # radius
    x_body = R * np.cos(theta_grid)
    y_body = R * np.sin(theta_grid)
    z_body = z_grid
    
    # Generate blades (6 radial blades)
    n_blades = 6
    blade_x, blade_y, blade_z = [], [], []
    for i in range(n_blades):
        angle = i * 2 * np.pi / n_blades
        for r in np.linspace(0.2, 0.9, 8):
            for zz in np.linspace(-0.3, 0.3, 4):
                blade_x.append(r * np.cos(angle))
                blade_y.append(r * np.sin(angle))
                blade_z.append(zz)
    
    fig = go.Figure()
    
    # Turbine body surface
    fig.add_trace(go.Surface(
        x=x_body, y=y_body, z=z_body,
        colorscale=[[0, color], [1, color]],
        opacity=0.6,
        showscale=False,
        name="Body"
    ))
    
    # Blades as scatter points
    fig.add_trace(go.Scatter3d(
        x=blade_x, y=blade_y, z=blade_z,
        mode='markers',
        marker=dict(size=4, color=color, opacity=0.9),
        name="Blades"
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(visible=False, range=[-1.2, 1.2]),
            zaxis=dict(visible=False, range=[-1.2, 1.2]),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            bgcolor="rgba(0,0,0,0)",
            aspectmode='cube'
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Turbine HP - Health: {health:.0f}%", 
                   font=dict(color="white", size=14), x=0.5)
    )
    return fig


def _create_rotor_3d(health: float) -> go.Figure:
    """3D visualization of a Rotor (elongated cylinder)."""
    color = _health_color(health)
    
    theta = np.linspace(0, 2*np.pi, 30)
    z = np.linspace(-2, 2, 15)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    R = 0.4
    x = R * np.cos(theta_grid)
    y = R * np.sin(theta_grid)
    z_rotor = z_grid
    
    # Add disks along the rotor
    disk_x, disk_y, disk_z = [], [], []
    for zz in [-1.5, -0.5, 0.5, 1.5]:
        for t in np.linspace(0, 2*np.pi, 20):
            for r in np.linspace(0, 0.5, 5):
                disk_x.append(r * np.cos(t))
                disk_y.append(r * np.sin(t))
                disk_z.append(zz)
    
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z_rotor,
        colorscale=[[0, color], [1, color]],
        opacity=0.7, showscale=False, name="Rotor"
    ))
    fig.add_trace(go.Scatter3d(
        x=disk_x, y=disk_y, z=disk_z,
        mode='markers',
        marker=dict(size=2, color="white", opacity=0.5),
        name="Disks"
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-0.8, 0.8]),
            yaxis=dict(visible=False, range=[-0.8, 0.8]),
            zaxis=dict(visible=False, range=[-2.5, 2.5]),
            camera=dict(eye=dict(x=2.0, y=1.5, z=1.0)),
            bgcolor="rgba(0,0,0,0)",
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=2.5)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Rotor - Health: {health:.0f}%", 
                   font=dict(color="white", size=14), x=0.5)
    )
    return fig


def _create_bearing_3d(health: float) -> go.Figure:
    """3D visualization of a Bearing (torus/ring)."""
    color = _health_color(health)
    
    # Torus parametric equations
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, 2*np.pi, 20)
    u_grid, v_grid = np.meshgrid(u, v)
    
    R = 0.7  # major radius
    r = 0.25  # minor radius
    
    x = (R + r * np.cos(v_grid)) * np.cos(u_grid)
    y = (R + r * np.cos(v_grid)) * np.sin(u_grid)
    z = r * np.sin(v_grid)
    
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        opacity=0.85, showscale=False, name="Bearing"
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(visible=False, range=[-1.2, 1.2]),
            zaxis=dict(visible=False, range=[-0.5, 0.5]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Bearings - Health: {health:.0f}%", 
                   font=dict(color="white", size=14), x=0.5)
    )
    return fig


def _create_valve_3d(health: float) -> go.Figure:
    """3D visualization of a Valve."""
    color = _health_color(health)
    
    # Valve body (cylinder)
    theta = np.linspace(0, 2*np.pi, 30)
    z = np.linspace(-1, 1, 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    R = 0.5
    x_body = R * np.cos(theta_grid)
    y_body = R * np.sin(theta_grid)
    z_body = z_grid
    
    # Valve disk (perpendicular plate)
    disk_theta = np.linspace(0, 2*np.pi, 30)
    disk_r = np.linspace(0, 0.45, 10)
    disk_theta_grid, disk_r_grid = np.meshgrid(disk_theta, disk_r)
    
    disk_x = disk_r_grid * np.cos(disk_theta_grid)
    disk_y = np.full_like(disk_x, 0)
    disk_z = disk_r_grid * np.sin(disk_theta_grid)
    
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x_body, y=y_body, z=z_body,
        colorscale=[[0, color], [1, color]],
        opacity=0.5, showscale=False, name="Body"
    ))
    fig.add_trace(go.Surface(
        x=disk_x, y=disk_y, z=disk_z,
        colorscale=[[0, "#ffffff"], [1, "#cccccc"]],
        opacity=0.9, showscale=False, name="Disk"
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-0.8, 0.8]),
            yaxis=dict(visible=False, range=[-0.8, 0.8]),
            zaxis=dict(visible=False, range=[-1.2, 1.2]),
            camera=dict(eye=dict(x=1.8, y=1.5, z=1.2)),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Valves - Health: {health:.0f}%", 
                   font=dict(color="white", size=14), x=0.5)
    )
    return fig


def _create_condenser_3d(health: float) -> go.Figure:
    """3D visualization of a Condenser (tube bundle in shell)."""
    color = _health_color(health)
    
    # Outer shell (cylinder)
    theta = np.linspace(0, 2*np.pi, 30)
    z = np.linspace(-1.2, 1.2, 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    R = 0.9
    x_shell = R * np.cos(theta_grid)
    y_shell = R * np.sin(theta_grid)
    z_shell = z_grid
    
    # Inner tubes (multiple small cylinders)
    tube_x, tube_y, tube_z = [], [], []
    n_tubes = 12
    for i in range(n_tubes):
        angle = i * 2 * np.pi / n_tubes
        r_center = 0.5
        cx = r_center * np.cos(angle)
        cy = r_center * np.sin(angle)
        
        for zz in np.linspace(-1.0, 1.0, 15):
            tube_x.append(cx)
            tube_y.append(cy)
            tube_z.append(zz)
    
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x_shell, y=y_shell, z=z_shell,
        colorscale=[[0, color], [1, color]],
        opacity=0.3, showscale=False, name="Shell"
    ))
    fig.add_trace(go.Scatter3d(
        x=tube_x, y=tube_y, z=tube_z,
        mode='markers',
        marker=dict(size=5, color="#4da6ff", opacity=0.8),
        name="Tubes"
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(visible=False, range=[-1.2, 1.2]),
            zaxis=dict(visible=False, range=[-1.5, 1.5]),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Condenser - Health: {health:.0f}%", 
                   font=dict(color="white", size=14), x=0.5)
    )
    return fig


def _create_generator_3d(health: float) -> go.Figure:
    """3D visualization of a Generator (cylinder with coils)."""
    color = _health_color(health)
    
    # Main cylinder body
    theta = np.linspace(0, 2*np.pi, 30)
    z = np.linspace(-1.2, 1.2, 12)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    R = 0.7
    x_body = R * np.cos(theta_grid)
    y_body = R * np.sin(theta_grid)
    z_body = z_grid
    
    # Coil windings (helical pattern)
    coil_x, coil_y, coil_z = [], [], []
    t = np.linspace(0, 6*np.pi, 100)
    coil_r = 0.75
    coil_x = coil_r * np.cos(t)
    coil_y = coil_r * np.sin(t)
    coil_z = np.linspace(-1.1, 1.1, 100)
    
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x_body, y=y_body, z=z_body,
        colorscale=[[0, color], [1, color]],
        opacity=0.5, showscale=False, name="Stator"
    ))
    fig.add_trace(go.Scatter3d(
        x=coil_x, y=coil_y, z=coil_z,
        mode='lines',
        line=dict(color="#ffcc00", width=4),
        name="Coils"
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.0, 1.0]),
            yaxis=dict(visible=False, range=[-1.0, 1.0]),
            zaxis=dict(visible=False, range=[-1.5, 1.5]),
            camera=dict(eye=dict(x=1.8, y=1.5, z=1.2)),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Generator - Health: {health:.0f}%", 
                   font=dict(color="white", size=14), x=0.5)
    )
    return fig


def _get_3d_figure(component: str, health: float) -> go.Figure:
    """Returns the appropriate 3D figure for a component."""
    generators = {
        "Turbine_HP": _create_turbine_3d,
        "Rotor": _create_rotor_3d,
        "Bearings": _create_bearing_3d,
        "Valves": _create_valve_3d,
        "Condenser": _create_condenser_3d,
        "Generator": _create_generator_3d,
    }
    generator = generators.get(component, _create_turbine_3d)
    return generator(health)


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    return dbc.Container([
        html.H2("🔧 Remaining Useful Life (RUL) & Health Monitoring", className="text-light my-4"),
        
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Total Hours Operated:", className="text-light"),
                dcc.Input(id="hours-input", type="number", value=10000, 
                         className="form-control bg-dark text-light"),
            ], width=3),
            dbc.Col([
                dbc.Button("Refresh Data", id="refresh-btn", n_clicks=0, 
                          className="btn btn-primary mt-4"),
            ], width=2),
        ], className="mb-4"),
        
        # Most Critical Component Alert
        html.Div(id="critical-alert", className="mb-4"),
        
        # Component Cards Grid with 3D visualizations
        html.Div(id="components-grid"),
        
        # Interval for auto-refresh (every 10 seconds)
        dcc.Interval(id="interval-component", interval=10*1000, n_intervals=0),
        dcc.Store(id="rul-data-store")
    ], fluid=True, className="bg-dark text-light min-vh-100 p-4")


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("rul-data-store", "data"),
    Input("refresh-btn", "n_clicks"),
    Input("interval-component", "n_intervals"),
    Input("hours-input", "value"),
    prevent_initial_call=False
)
def fetch_rul_data(n_clicks, n_intervals, hours):
    try:
        response = requests.get(f"{API_BASE_URL}/rul/summary", 
                               params={"hours_operated": hours or 10000})
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching RUL data: {e}")
    return None


@callback(
    Output("critical-alert", "children"),
    Output("components-grid", "children"),
    Input("rul-data-store", "data")
)
def render_rul_dashboard(data):
    if not data:
        return dbc.Alert("Could not fetch RUL data. Check backend connection.", 
                        color="danger"), []
    
    if "error" in data:
        return dbc.Alert(f"⚠️ {data['error']}", color="warning"), []
    
    most_critical = data.get("most_critical", "Unknown")
    health_scores = data.get("health_scores", {})
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
    
    # 2. Component Health Cards with 3D visualizations
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
        
        # Generate 3D figure for this component
        fig_3d = _get_3d_figure(component, health)
        
        card = dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5(component, className="text-center mb-0 text-light")),
                dbc.CardBody([
                    # 3D Visualization
                    dcc.Graph(
                        figure=fig_3d,
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': ['resetCameraDefault3D'],
                            'displaylogo': False,
                        },
                        style={"height": "280px"}
                    ),
                    html.Hr(className="bg-secondary"),
                    # Health Score
                    html.H3(f"{health:.1f}%", className=f"{text_color} text-center fw-bold"),
                    dbc.Progress(value=health, color=progress_color, className="mb-2"),
                    # RUL and Status
                    dbc.Row([
                        dbc.Col([
                            html.P([
                                html.Strong("RUL: "),
                                html.Span(f"{rul_days:.0f} days", className="text-light")
                            ], className="text-center mb-0")
                        ], width=6),
                        dbc.Col([
                            html.P([
                                html.Strong("Status: "),
                                html.Span(status.upper(), className=f"{text_color} fw-bold")
                            ], className="text-center mb-0")
                        ], width=6),
                    ]),
                ])
            ], className=f"border-{border_color} shadow-sm h-100 bg-dark")
        ], md=6, lg=4, sm=12, className="mb-4")
        cards.append(card)
    
    return critical_alert, dbc.Row(cards)