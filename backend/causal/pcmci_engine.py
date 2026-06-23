"""
pcmci_engine.py — PCMCI+ causal discovery (tigramite).
Falls back to Granger causality if tigramite is not installed.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .granger_validator import GrangerValidator
try:
    import tigramite
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    _HAS_TIGRAMITE = True
except ImportError:
    _HAS_TIGRAMITE = False

DEFAULT_VARS = [
    "production", "bilan_net", "steam_hp", "steam_mp",
    "pressure",   "vibration", "temperature", "efficiency",
]


class PCMCIEngine:
    def __init__(self, max_lag: int = 5, alpha: float = 0.05):
        self.max_lag  = max_lag
        self.alpha    = alpha
        self.results  = None
        self.var_names: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, variables: list[str] | None = None) -> dict:
        """Run PCMCI+ (or Granger fallback). Returns serialisable dict."""
        self.var_names = [v for v in (variables or DEFAULT_VARS) if v in df.columns]
        data = self._prepare(df)

        if _HAS_TIGRAMITE:
            return self._run_tigramite(data)
        else:
            print("[PCMCI] tigramite not found — using Granger fallback")
            from backend.causal.granger_validator import GrangerValidator
            result = GrangerValidator(self.max_lag, self.alpha).run_full(df, self.var_names)
            self.results = result
            return result

    def get_networkx_graph(self):
        import networkx as nx
        from backend.causal.dag_builder import build_dag_from_links
        if self.results is None:
            raise RuntimeError("Run .run() first")
        return build_dag_from_links(self.results)

    def get_cytoscape_elements(self) -> list:
        from backend.causal.dag_builder import to_cytoscape_elements
        return to_cytoscape_elements(self.get_networkx_graph())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        data = df[self.var_names].values.astype(np.float32)
        for i in range(data.shape[1]):
            col = data[:, i]
            nan_mask = np.isnan(col)
            col[nan_mask] = np.nanmean(col) if not nan_mask.all() else 0.0
        means = data.mean(axis=0)
        stds  = data.std(axis=0)
        stds[stds == 0] = 1.0
        return (data - means) / stds

    def _run_tigramite(self, data: np.ndarray) -> dict:
        dataframe = pp.DataFrame(data, var_names=self.var_names,
                                 datatime=np.arange(len(data)))
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"),
                      verbosity=0)
        self.results_raw = pcmci.run_pcmciplus(tau_min=0, tau_max=self.max_lag,
                                               pc_alpha=self.alpha)
        return self._extract(self.results_raw)

    def _extract(self, res: dict) -> dict:
        graph      = res["graph"]
        val_matrix = res["val_matrix"]
        p_matrix   = res["p_matrix"]
        n = len(self.var_names)
        edges = []
        for i in range(n):
            for j in range(n):
                for tau in range(self.max_lag + 1):
                    if graph[i, j, tau] == "-->":
                        edges.append({
                            "source":    self.var_names[i],
                            "target":    self.var_names[j],
                            "lag":       tau,
                            "strength":  float(abs(val_matrix[i, j, tau])),
                            "p_value":   float(p_matrix[i, j, tau]),
                            "direction": "causal",
                        })
        edges.sort(key=lambda x: x["strength"], reverse=True)
        self.results = {"nodes": self.var_names, "edges": edges}
        return self.results