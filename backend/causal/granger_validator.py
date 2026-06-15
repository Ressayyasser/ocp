"""
granger_validator.py — Pairwise Granger causality (OLS F-test).
Used as the PCMCI fallback and as an independent cross-check.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


class GrangerValidator:
    def __init__(self, max_lag: int = 5, alpha: float = 0.05):
        self.max_lag = max_lag
        self.alpha   = alpha

    def test_pair(self, x: np.ndarray, y: np.ndarray, lag: int) -> dict:
        """F-test: does x(t-lag…) help predict y(t) beyond y's own lags?"""
        n = len(y)
        if n < lag * 4:
            return {"f_stat": 0.0, "p_value": 1.0, "significant": False}

        Y   = y[lag:]
        Y_lags = np.column_stack([y[lag-i-1: n-i-1] for i in range(lag)])
        X_lags = np.column_stack([x[lag-i-1: n-i-1] for i in range(lag)])

        Xr = np.column_stack([np.ones(len(Y)), Y_lags])
        Xu = np.column_stack([np.ones(len(Y)), Y_lags, X_lags])

        try:
            br = np.linalg.lstsq(Xr, Y, rcond=None)[0]
            bu = np.linalg.lstsq(Xu, Y, rcond=None)[0]
            ssr_r = float(np.sum((Y - Xr @ br) ** 2))
            ssr_u = float(np.sum((Y - Xu @ bu) ** 2))
            df1, df2 = lag, len(Y) - Xu.shape[1]
            if ssr_u <= 0 or df2 <= 0:
                return {"f_stat": 0.0, "p_value": 1.0, "significant": False}
            f  = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            pv = 1 - stats.f.cdf(f, df1, df2)
            return {"f_stat": float(f), "p_value": float(pv), "significant": pv < self.alpha}
        except Exception:
            return {"f_stat": 0.0, "p_value": 1.0, "significant": False}

    def run_full(self, df: pd.DataFrame, variables: list[str]) -> dict:
        """All pairwise Granger tests. Returns edges list."""
        existing = [v for v in variables if v in df.columns]
        edges = []
        for src in existing:
            for tgt in existing:
                if src == tgt:
                    continue
                x, y = df[src].values.astype(float), df[tgt].values.astype(float)
                best = {"f_stat": 0.0, "p_value": 1.0, "significant": False, "lag": 0}
                for lag in range(1, self.max_lag + 1):
                    r = self.test_pair(x, y, lag)
                    if r["significant"] and r["f_stat"] > best["f_stat"]:
                        best = {**r, "lag": lag}
                if best["significant"]:
                    edges.append({
                        "source":    src,
                        "target":    tgt,
                        "lag":       best["lag"],
                        "strength":  float(min(best["f_stat"] / 20, 1.0)),
                        "p_value":   best["p_value"],
                        "direction": "causal",
                    })
        edges.sort(key=lambda e: e["strength"], reverse=True)
        return {"nodes": existing, "edges": edges}