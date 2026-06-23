"""
pages/dag.py — Causal DAG visualization page.
Displays PCMCI+ causal graph using Plotly network graph.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import get_causal_graph

dash.register_page(__name__, path="/dag", name="DAG Causal", title="DAG Causal")

layout = html.Div([
    html.H3("🔗 Graphe Causal — PCMCI+", className="text-light fw-bold mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Force min. du lien", className="text-muted small"),
            dcc.Slider(id="dag-strength-slider", min=0, max=1, step=0.05, value=0.1,
                       marks={0:"0", 0.25:"0.25", 0.5:"0.5", 0.75:"0.75", 1:"1"}),
        ], width=5),
        dbc.Col([
            dbc.Button("🔄 Recharger le graphe", id="dag-refresh", color="warning",
                       size="sm", className="mt-3"),
        ], width=3),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Graphe orienté des causes", className="text-light"),
                dbc.CardBody(dcc.Graph(id="dag-graph", style={"height": "500px"})),
            ], className="bg-dark border-secondary"),
        ], width=8),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top liens causaux", className="text-light"),
                dbc.CardBody(html.Div(id="dag-links-table",
                                      style={"maxHeight":"460px","overflowY":"auto"})),
            ], className="bg-dark border-secondary"),
        ], width=4),
    ], className="mb-4"),

    # Interpretation panel
    dbc.Card([
        dbc.CardHeader("📖 Interprétation automatique", className="text-light"),
        dbc.CardBody(html.Div(id="dag-interpretation")),
    ], className="bg-dark border-secondary"),
])


def _build_network(edges: list, min_strength: float = 0.1) -> go.Figure:
    """Build a Plotly scatter network from causal links."""
    dark = dict(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                font=dict(color="#e0e0e0"), margin=dict(l=20, r=20, t=30, b=20))

    if not edges:
        fig = go.Figure()
        fig.update_layout(**dark)
        fig.add_annotation(text="Aucun lien causal disponible — lancez l'entraînement.",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=16, color="#aaa"))
        return fig

    # Filter edges by strength
    filtered = [e for e in edges if abs(e.get("strength", 0)) >= min_strength]
    
    # Gather unique nodes using the correct keys 'source' and 'target'
    nodes = list({e["source"] for e in filtered} | {e["target"] for e in filtered})
    n = len(nodes)
    if n == 0:
        return go.Figure().update_layout(**dark)

    import math
    # Arrange in a circle
    pos = {node: (math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n))
           for i, node in enumerate(nodes)}

    # Edge traces
    edge_x, edge_y = [], []
    for lnk in filtered:
        x0, y0 = pos[lnk["source"]]
        x1, y1 = pos[lnk["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=1.5, color="#4d9de0"), hoverinfo="none")

    # Node trace
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=nodes, textposition="top center",
        marker=dict(size=20, color="#f0c040",
                    line=dict(width=2, color="#fff")),
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(**dark,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig


@callback(
    Output("dag-graph",          "figure"),
    Output("dag-links-table",    "children"),
    Output("dag-interpretation", "children"),
    Input("dag-refresh",         "n_clicks"),
    Input("dag-strength-slider", "value"),
)
def update_dag(_, min_strength):
    response = get_causal_graph()
    
    # Safely extract edges from the API response dictionary
    if isinstance(response, dict):
        if "error" in response:
            edges = []
        else:
            edges = response.get("edges", [])
    elif isinstance(response, list):
        edges = response
    else:
        edges = []

    # Network figure
    fig = _build_network(edges, min_strength)

    # Table
    filtered = sorted(
        [e for e in edges if abs(e.get("strength", 0)) >= min_strength],
        key=lambda x: abs(x.get("strength", 0)), reverse=True,
    )
    
    # Use the correct keys 'source' and 'target' matching the backend schema
    rows = [html.Tr([
        html.Td(e.get("source", "—"), style={"fontSize": "0.8rem"}),
        html.Td("→"),
        html.Td(e.get("target", "—"), style={"fontSize": "0.8rem"}),
        html.Td(f"lag {e.get('lag', 0)}", style={"fontSize": "0.8rem"}),
        html.Td(f"{e.get('strength', 0):.3f}",
                style={"fontSize": "0.8rem",
                       "color": "#f0c040" if abs(e.get("strength", 0)) > 0.5 else "#aaa"}),
    ]) for e in filtered[:30]]

    table = dbc.Table(
        [html.Thead(html.Tr([html.Th(c) for c in ["Source","","Cible","Lag","Force"]])),
         html.Tbody(rows)],
        bordered=True, color="dark", hover=True, size="sm", # <--- FIXED
    ) if rows else html.P("Aucun lien au-dessus du seuil.", className="text-muted")

    # Interpretation
    if filtered:
        top = filtered[0]
        interp = [
            html.P([
                html.Strong(f"{top['source']} → {top['target']} "),
                f"est le lien causal le plus fort (force={top['strength']:.3f}, lag={top.get('lag', 0)}). "
                "Cela signifie que les variations de ",
                html.Strong(top['source']),
                " précèdent et influencent statistiquement ",
                html.Strong(top['target']),
                f" avec un décalage de {top.get('lag', 0)} pas de temps.",
            ], className="text-light"),
            html.P(f"Nombre total de liens détectés: {len(edges)} | "
                   f"Liens au-dessus du seuil {min_strength}: {len(filtered)}",
                   className="text-muted small"),
        ]
    else:
        interp = [html.P("Entraînez le module causal (PCMCI+) pour voir l'interprétation.",
                         className="text-muted")]

    return fig, table, interp