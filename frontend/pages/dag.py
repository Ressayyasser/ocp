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
            html.Label("Force min. de mise en évidence", className="text-muted small"),
            dcc.Slider(id="dag-strength-slider", min=0, max=1, step=0.05, value=0.1,
                       marks={0:"0", 0.25:"0.25", 0.5:"0.5", 0.75:"0.75", 1:"1"}),
            html.Small("Le graphe affiche TOUS les liens détectés — ceux sous le seuil "
                       "apparaissent estompés, ceux au-dessus sont surlignés.",
                       className="text-muted"),
        ], width=6),
        dbc.Col([
            dbc.Button("🔄 Recharger le graphe", id="dag-refresh", color="warning",
                       size="sm", className="mt-3"),
        ], width=3),
    ], className="mb-3"),

    # Graph statistics (nodes / links counts)
    dbc.Row(id="dag-stats", className="mb-4"),

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


def _build_network(all_nodes: list, edges: list, min_strength: float = 0.1) -> go.Figure:
    """Full causal network: every node and every link is drawn. Links below the
    highlight threshold are faded; links above are coloured with width ∝ force.
    Self-dependencies (autocorrelation links) are drawn as loops around nodes."""
    dark = dict(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                font=dict(color="#e0e0e0"), margin=dict(l=20, r=20, t=30, b=20))

    if not edges and not all_nodes:
        fig = go.Figure()
        fig.update_layout(**dark)
        fig.add_annotation(text="Aucun lien causal disponible — lancez l'entraînement.",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=16, color="#aaa"))
        return fig

    import math
    # All nodes: those declared by the engine plus any appearing in a link
    nodes = list(dict.fromkeys(
        list(all_nodes or [])
        + [e["source"] for e in edges] + [e["target"] for e in edges]))
    n = max(len(nodes), 1)
    pos = {node: (math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n))
           for i, node in enumerate(nodes)}

    self_edges  = [e for e in edges if e["source"] == e["target"]]
    cross_edges = [e for e in edges if e["source"] != e["target"]]

    fig = go.Figure()

    # ── Weak cross links (below threshold): one faded trace ───────────────────
    weak_x, weak_y = [], []
    for e in cross_edges:
        if abs(e.get("strength", 0)) >= min_strength:
            continue
        x0, y0 = pos[e["source"]]; x1, y1 = pos[e["target"]]
        weak_x += [x0, x1, None]; weak_y += [y0, y1, None]
    if weak_x:
        fig.add_trace(go.Scatter(x=weak_x, y=weak_y, mode="lines",
                                 line=dict(width=1, color="#4d9de0"),
                                 opacity=0.25, hoverinfo="none",
                                 name="Liens < seuil", showlegend=True))

    # ── Strong cross links: individual traces (width ∝ strength) + arrows ─────
    strong = [e for e in cross_edges if abs(e.get("strength", 0)) >= min_strength]
    for i, e in enumerate(strong):
        x0, y0 = pos[e["source"]]; x1, y1 = pos[e["target"]]
        s = abs(e.get("strength", 0))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=1.5 + 5 * s, color="#f0c040"),
            opacity=0.9, hoverinfo="none",
            name="Liens ≥ seuil", legendgroup="strong", showlegend=(i == 0)))
        # arrowhead + invisible hover marker at midpoint
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.4,
                           arrowwidth=1.5, arrowcolor="#f0c040", opacity=0.9,
                           standoff=16, text="")
        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="markers",
            marker=dict(size=14, opacity=0), showlegend=False,
            hoverinfo="text",
            hovertext=f"{e['source']} → {e['target']}<br>"
                      f"lag {e.get('lag', 0)} · force {e.get('strength', 0):.3f}"))

    # ── Self-loops (autocorrelation links, e.g. steam_hp → steam_hp) ──────────
    for e in self_edges:
        cx, cy = pos[e["source"]]
        s = abs(e.get("strength", 0))
        is_strong = s >= min_strength
        # small circle drawn just outside the node, away from the centre
        ox, oy = cx * 1.22, cy * 1.22
        t = [2 * math.pi * k / 24 for k in range(25)]
        fig.add_trace(go.Scatter(
            x=[ox + 0.11 * math.cos(a) for a in t],
            y=[oy + 0.11 * math.sin(a) for a in t],
            mode="lines",
            line=dict(width=1.5 + (4 * s if is_strong else 0),
                      color="#ff9f43" if is_strong else "#4d9de0"),
            opacity=0.9 if is_strong else 0.25,
            hoverinfo="text", showlegend=False,
            hovertext=f"↻ {e['source']} (auto-dépendance)<br>"
                      f"lag {e.get('lag', 0)} · force {s:.3f}"))
        if is_strong:
            fig.add_annotation(x=ox, y=oy + 0.19, text=f"↻ lag {e.get('lag', 0)}",
                               showarrow=False,
                               font=dict(size=9, color="#ff9f43"))

    # ── Nodes (size ∝ number of strong connections) ───────────────────────────
    degree = {node: 0 for node in nodes}
    for e in edges:
        if abs(e.get("strength", 0)) >= min_strength:
            degree[e["source"]] = degree.get(e["source"], 0) + 1
            degree[e["target"]] = degree.get(e["target"], 0) + 1
    fig.add_trace(go.Scatter(
        x=[pos[node][0] for node in nodes], y=[pos[node][1] for node in nodes],
        mode="markers+text", text=nodes, textposition="top center",
        marker=dict(size=[16 + 4 * degree[node] for node in nodes],
                    color="#f0c040", line=dict(width=2, color="#fff")),
        hoverinfo="text", showlegend=False,
        hovertext=[f"{node} — {degree[node]} lien(s) ≥ seuil" for node in nodes]))

    fig.update_layout(**dark,
                      legend=dict(orientation="h", y=1.08),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                                 range=[-1.6, 1.6]),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                                 range=[-1.6, 1.6]))
    return fig


def _stat_card(value, label, color):
    return dbc.Col(dbc.Card([dbc.CardBody([
        html.H3(str(value), className=f"text-{color} fw-bold mb-0"),
        html.Small(label, className="text-muted"),
    ])], className=f"bg-dark border-{color} text-center"), width=2)


@callback(
    Output("dag-stats",          "children"),
    Output("dag-graph",          "figure"),
    Output("dag-links-table",    "children"),
    Output("dag-interpretation", "children"),
    Input("dag-refresh",         "n_clicks"),
    Input("dag-strength-slider", "value"),
)
def update_dag(_, min_strength):
    response = get_causal_graph()

    # Safely extract nodes/edges from the API response dictionary
    all_nodes = []
    if isinstance(response, dict):
        if "error" in response:
            edges = []
        else:
            edges = response.get("edges", [])
            all_nodes = response.get("nodes", [])
    elif isinstance(response, list):
        edges = response
    else:
        edges = []

    self_edges = [e for e in edges if e.get("source") == e.get("target")]
    cross_edges = [e for e in edges if e.get("source") != e.get("target")]
    filtered = sorted(
        [e for e in edges if abs(e.get("strength", 0)) >= min_strength],
        key=lambda x: abs(x.get("strength", 0)), reverse=True,
    )
    n_nodes = len(set(all_nodes)
                  | {e["source"] for e in edges} | {e["target"] for e in edges})

    # ── Stats cards: how many nodes and links the DAG contains ────────────────
    stats = [
        _stat_card(n_nodes, "Nœuds (variables)", "warning"),
        _stat_card(len(edges), "Liens causaux détectés", "info"),
        _stat_card(len(cross_edges), "Liens inter-variables", "primary"),
        _stat_card(len(self_edges), "Auto-dépendances (↻)", "danger"),
        _stat_card(len(filtered), f"Liens ≥ seuil {min_strength}", "success"),
        _stat_card((max((e.get("lag", 0) for e in edges), default=0)),
                   "Lag max (jours)", "light"),
    ]

    # ── Network figure: the FULL DAG (all nodes, all links) ───────────────────
    fig = _build_network(all_nodes, edges, min_strength)

    # ── Table: every detected link, above-threshold ones highlighted ──────────
    all_sorted = sorted(edges, key=lambda x: abs(x.get("strength", 0)), reverse=True)
    rows = []
    for e in all_sorted:
        above = abs(e.get("strength", 0)) >= min_strength
        is_self = e.get("source") == e.get("target")
        rows.append(html.Tr([
            html.Td(("↻ " if is_self else "") + e.get("source", "—"),
                    style={"fontSize": "0.8rem"}),
            html.Td("→"),
            html.Td(e.get("target", "—"), style={"fontSize": "0.8rem"}),
            html.Td(f"lag {e.get('lag', 0)}", style={"fontSize": "0.8rem"}),
            html.Td(f"{e.get('strength', 0):.3f}",
                    style={"fontSize": "0.8rem",
                           "color": "#f0c040" if above else "#777",
                           "fontWeight": "700" if above else "400"}),
        ], style={} if above else {"opacity": "0.55"}))

    table = html.Div([
        html.P(f"{len(edges)} lien(s) au total — les liens ≥ {min_strength} sont "
               "surlignés en or.", className="text-muted small mb-2"),
        dbc.Table(
            [html.Thead(html.Tr([html.Th(c) for c in ["Source", "", "Cible", "Lag", "Force"]])),
             html.Tbody(rows)],
            bordered=True, color="dark", hover=True, size="sm",
        ),
    ]) if rows else html.P("Aucun lien détecté.", className="text-muted")

    # ── Interpretation ────────────────────────────────────────────────────────
    if edges:
        lag_counts: dict[int, int] = {}
        for e in edges:
            lag_counts[e.get("lag", 0)] = lag_counts.get(e.get("lag", 0), 0) + 1
        lag_txt = " · ".join(f"lag {k} : {v} lien(s)"
                             for k, v in sorted(lag_counts.items()))

        interp = [
            html.P([
                html.Strong("Résumé du graphe : "),
                f"{n_nodes} nœuds (variables énergétiques) reliés par {len(edges)} liens "
                f"causaux temporels, dont {len(self_edges)} auto-dépendance(s) "
                f"(inertie d'une variable sur elle-même) et {len(cross_edges)} liens "
                f"inter-variables. {len(filtered)} lien(s) dépassent le seuil de "
                f"{min_strength}.",
            ], className="text-light"),
            html.P([html.Strong("Répartition par décalage temporel : "), lag_txt],
                   className="text-light small"),
        ]
        if filtered:
            top = filtered[0]
            if top["source"] == top["target"]:
                interp.append(html.P([
                    html.Strong(f"↻ {top['source']} "),
                    f"présente la plus forte dépendance causale (force={top['strength']:.3f}, "
                    f"lag={top.get('lag', 0)}) : c'est une auto-dépendance — sa valeur "
                    f"d'il y a {top.get('lag', 0)} jour(s) détermine fortement sa valeur "
                    "actuelle (inertie thermique du réseau vapeur). Elle en fait la "
                    "variable exogène maîtresse du système.",
                ], className="text-light"))
            else:
                interp.append(html.P([
                    html.Strong(f"{top['source']} → {top['target']} "),
                    f"est le lien causal inter-variables le plus fort "
                    f"(force={top['strength']:.3f}, lag={top.get('lag', 0)}) : les "
                    f"variations de {top['source']} précèdent et influencent "
                    f"statistiquement {top['target']} avec un décalage de "
                    f"{top.get('lag', 0)} pas de temps.",
                ], className="text-light"))
    else:
        interp = [html.P("Entraînez le module causal (PCMCI+) pour voir l'interprétation.",
                         className="text-muted")]

    return stats, fig, table, interp