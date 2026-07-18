"""Tests for Layer 2 — temporal causality: Granger (F06), DAG (F07), PCMCI+ (F05)."""

import numpy as np
import pandas as pd

from causal.granger_validator import GrangerValidator
from causal.dag_builder import (
    build_dag_from_links, dag_to_json, get_root_causes,
    get_causal_path, get_effect_chain, to_cytoscape_elements,
)
from causal.pcmci_engine import PCMCIEngine


def _lagged_pair(n=300, seed=7):
    """x drives y with lag 1: y_t = 0.8·x_{t-1} + ε."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.8 * x[t - 1] + rng.normal(0, 0.3)
    return x, y


class TestGranger:
    def test_detects_true_causal_link(self):
        x, y = _lagged_pair()
        res = GrangerValidator(max_lag=3).test_pair(x, y, lag=1)
        assert res["significant"], "x→y (lag 1) must be Granger-significant"
        assert res["p_value"] < 0.05

    def test_independent_series_not_significant(self):
        rng = np.random.default_rng(11)
        res = GrangerValidator().test_pair(rng.normal(0, 1, 300),
                                           rng.normal(0, 1, 300), lag=2)
        assert not res["significant"]

    def test_short_series_is_safe(self):
        res = GrangerValidator().test_pair(np.ones(5), np.ones(5), lag=3)
        assert res == {"f_stat": 0.0, "p_value": 1.0, "significant": False}

    def test_run_full_finds_directed_edge(self):
        x, y = _lagged_pair()
        df = pd.DataFrame({"steam_hp": x, "production": y})
        graph = GrangerValidator(max_lag=3).run_full(df, ["steam_hp", "production"])
        assert graph["nodes"] == ["steam_hp", "production"]
        assert any(e["source"] == "steam_hp" and e["target"] == "production"
                   for e in graph["edges"])


class TestDAGBuilder:
    RESULT = {
        "nodes": ["steam_hp", "production", "bilan_net"],
        "edges": [
            {"source": "steam_hp", "target": "production",
             "lag": 1, "strength": 0.8, "p_value": 0.001, "direction": "causal"},
            {"source": "production", "target": "bilan_net",
             "lag": 0, "strength": 0.9, "p_value": 0.001, "direction": "causal"},
        ],
    }

    def test_build_and_query(self):
        G = build_dag_from_links(self.RESULT)
        assert G.number_of_nodes() == 3 and G.number_of_edges() == 2
        assert get_root_causes(G, "bilan_net") == ["steam_hp"]
        assert get_causal_path(G, "steam_hp", "bilan_net") == \
            ["steam_hp", "production", "bilan_net"]

    def test_effect_chain_accumulates_lag_and_strength(self):
        G = build_dag_from_links(self.RESULT)
        chain = get_effect_chain(G, "steam_hp")
        effects = {e["target"]: e for e in chain["effects"]}
        assert effects["bilan_net"]["total_lag"] == 1
        assert abs(effects["bilan_net"]["cumulative_strength"] - 0.72) < 1e-9

    def test_serialisation_roundtrip_and_cytoscape(self):
        G = build_dag_from_links(self.RESULT)
        assert dag_to_json(G)["directed"] is True
        elements = to_cytoscape_elements(G)
        node_el = [e for e in elements if "source" not in e["data"]]
        edge_el = [e for e in elements if "source" in e["data"]]
        assert len(node_el) == 3 and len(edge_el) == 2
        assert edge_el[0]["data"]["label"].startswith("τ=")


class TestPCMCIEngine:
    def test_run_returns_causal_graph(self, synth_df):
        engine = PCMCIEngine(max_lag=3)
        result = engine.run(synth_df)
        assert "nodes" in result and "edges" in result
        assert len(result["nodes"]) >= 2
        for e in result["edges"]:
            assert {"source", "target", "lag", "strength"}.issubset(e.keys())
