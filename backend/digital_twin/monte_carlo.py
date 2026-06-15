"""
monte_carlo.py — Monte Carlo uncertainty quantification for Digital Twin scenarios.
Samples from historical variability to produce confidence intervals.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable


class MonteCarloSimulator:
    """
    Runs N Monte Carlo trials of a scenario function with
    Gaussian noise sampled from historical data statistics.
    """

    def __init__(self, n_trials: int = 1000, seed: int = 42):
        self.n_trials = n_trials
        self.rng      = np.random.default_rng(seed)

    def run(self, scenario_fn: Callable[[], dict],
            noise_std: dict[str, float] | None = None,
            df: pd.DataFrame | None = None) -> dict:
        """
        scenario_fn: callable that returns a dict with numeric outputs.
        noise_std:   per-key standard deviations for perturbation.
        df:          used to estimate noise_std from data if not provided.
        """
        if noise_std is None and df is not None:
            num = df.select_dtypes(include=[np.number])
            noise_std = {c: float(num[c].std()) * 0.02 for c in num.columns}
        elif noise_std is None:
            noise_std = {}

        results: list[dict] = []
        for _ in range(self.n_trials):
            trial = scenario_fn()
            for k, std in noise_std.items():
                if k in trial and std > 0:
                    trial[k] += float(self.rng.normal(0, std))
            results.append(trial)

        # Aggregate
        keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
        summary: dict = {}
        for k in keys:
            vals = np.array([r[k] for r in results if isinstance(r.get(k), (int, float))])
            summary[k] = {
                "mean":   round(float(vals.mean()),  2),
                "std":    round(float(vals.std()),   2),
                "p5":     round(float(np.percentile(vals,  5)), 2),
                "p95":    round(float(np.percentile(vals, 95)), 2),
                "median": round(float(np.median(vals)),       2),
            }

        return {
            "n_trials": self.n_trials,
            "summary":  summary,
            "raw":      results[:20],   # first 20 trials for diagnostics
        }