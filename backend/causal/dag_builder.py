"""
dag_builder.py — Build, serialise, and query the causal DAG.
"""

from __future__ import annotations
import networkx as nx
from networkx.readwrite import json_graph


def build_dag_from_links(result: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in result.get("nodes", []):
        G.add_node(node, label=node)
    for e in result.get("edges", []):
        if e.get("direction") == "causal":
            G.add_edge(e["source"], e["target"],
                       lag=e["lag"], strength=e["strength"],
                       p_value=e.get("p_value", 0.0))
    return G


def dag_to_json(G: nx.DiGraph) -> dict:
    return json_graph.node_link_data(G)


def dag_from_json(data: dict) -> nx.DiGraph:
    return json_graph.node_link_graph(data)


def get_root_causes(G: nx.DiGraph, target: str) -> list[str]:
    if target not in G:
        return []
    ancestors = nx.ancestors(G, target)
    return [n for n in ancestors if G.in_degree(n) == 0]


def get_causal_path(G: nx.DiGraph, source: str, target: str) -> list[str]:
    try:
        return nx.shortest_path(G, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_effect_chain(G: nx.DiGraph, source: str) -> dict:
    if source not in G:
        return {"source": source, "effects": []}
    effects = []
    for desc in nx.descendants(G, source):
        for path in nx.all_simple_paths(G, source, desc):
            lag  = sum(G.edges[path[i], path[i+1]].get("lag", 0)      for i in range(len(path)-1))
            strength = 1.0
            for i in range(len(path)-1):
                strength *= G.edges[path[i], path[i+1]].get("strength", 1.0)
            effects.append({"target": desc, "path": path,
                             "total_lag": lag, "cumulative_strength": strength})
    effects.sort(key=lambda x: x["cumulative_strength"], reverse=True)
    return {"source": source, "effects": effects}


def to_cytoscape_elements(G: nx.DiGraph) -> list[dict]:
    elements = [{"data": {"id": n, "label": n,
                           "in_degree": G.in_degree(n),
                           "out_degree": G.out_degree(n)}} for n in G.nodes()]
    elements += [{"data": {"source": u, "target": v,
                            "strength": d.get("strength", 0),
                            "lag": d.get("lag", 0),
                            "label": f"τ={d.get('lag',0)}"}}
                 for u, v, d in G.edges(data=True)]
    return elements