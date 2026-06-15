"""
cusum_detector.py — Tabular CUSUM for slow drift / change-point detection.
Complements Isolation Forest for gradual degradation (e.g. bearing wear).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ChangePoint:
    idx:       int
    direction: str   # "increase" | "decrease"
    cusum:     float
    threshold: float


class CUSUMDetector:
    """Two-sided CUSUM with automatic reference estimation."""

    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift     = drift

    def detect(self, data: np.ndarray) -> list[ChangePoint]:
        data = np.asarray(data, dtype=float)
        n    = len(data)
        if n < 10:
            return []

        ref_n = max(10, int(n * 0.2))
        mu0   = np.mean(data[:ref_n])
        sigma = np.std(data[:ref_n]) or 1.0
        z     = (data - mu0) / sigma

        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        cps: list[ChangePoint] = []

        for i in range(1, n):
            s_pos[i] = max(0, s_pos[i-1] + z[i] - self.drift)
            s_neg[i] = max(0, s_neg[i-1] - z[i] - self.drift)

            if s_pos[i] > self.threshold:
                cps.append(ChangePoint(i, "increase", float(s_pos[i]), self.threshold))
                s_pos[i] = 0
            if s_neg[i] > self.threshold:
                cps.append(ChangePoint(i, "decrease", float(s_neg[i]), self.threshold))
                s_neg[i] = 0

        return cps

    def detect_dataframe(self, df: pd.DataFrame, columns: list[str]) -> dict:
        results: dict[str, list] = {}
        for col in columns:
            if col not in df.columns:
                continue
            cps = self.detect(df[col].dropna().values)
            if cps:
                results[col] = [{"idx": c.idx, "direction": c.direction,
                                  "cusum": c.cusum} for c in cps]
        return results