"""
anomaly_service.py — Batch anomaly scan: Isolation Forest + CUSUM → anomalies table.

The live pipeline (SCADA reading → IsolationForestDetector.detect_all →
AlertService) only feeds the *alerts* table. This service is the batch/
historical counterpart required by the anomalies dashboard: it sweeps the
combined daily series (historical + live-sim days, i.e. up to today) with

  • Isolation Forest per variable group  → point anomalies (F08)
  • two-sided tabular CUSUM              → slow drifts / trend breaks (F09)

and persists every detection into the `anomalies` table read by
GET /anomalies. Re-scanning replaces previous auto-scan results (the table
is derived data, deterministic given the models and the series).
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from anomalies.isolation_forest_detector import IsolationForestDetector, GROUPS
from anomalies.cusum_detector import CUSUMDetector
from database.database import execute, insert_many, query

# Columns monitored for slow drift (report §5.8: rendement, usure, vapeur)
CUSUM_COLUMNS = [
    "production", "bilan_net", "efficiency", "steam_hp", "vibration",
    "rendement_gta1", "rendement_gta2", "rendement_gta3",
]


class AnomalyScanService:

    # After a change point, ignore re-triggers on the same column/direction
    # for this many samples (days) — we only care about the drift onset.
    CUSUM_COOLDOWN = 21
    # Keep at most this many (most recent) drift onsets per column
    CUSUM_MAX_PER_COLUMN = 5

    def __init__(self,
                 detector: IsolationForestDetector | None = None,
                 cusum: CUSUMDetector | None = None):
        self.detector = detector or IsolationForestDetector()
        self.cusum = cusum or CUSUMDetector(threshold=5.0, drift=0.5)

    # ── Isolation Forest sweep ────────────────────────────────────────────────

    def _scan_iforest(self, df: pd.DataFrame, window_days: int) -> list[dict]:
        rows: list[dict] = []
        recent = df.tail(window_days)
        for group in GROUPS:
            try:
                out = self.detector.detect_batch(recent, group)
            except Exception:
                continue                       # group model missing → skip
            flagged = out[out["is_anomaly"]]
            for _, r in flagged.iterrows():
                severity = r["severity"] if r["severity"] != "normal" else "info"
                cause_var = r.get("anomaly_cause")
                rows.append({
                    "timestamp": str(r.get("timestamp", "")),
                    "severity":  severity,
                    "score":     round(float(r["anomaly_score"]), 4),
                    "cause":     f"Isolation Forest — {GROUPS[group]['label']}",
                    "variable":  cause_var,
                    "raw_value": (round(float(r[cause_var]), 3)
                                  if cause_var in r and pd.notna(r[cause_var]) else None),
                    "threshold": -0.1,
                    "acknowledged": 0,
                })
        return rows

    # ── CUSUM drift sweep ─────────────────────────────────────────────────────

    def _scan_cusum(self, df: pd.DataFrame, window_days: int) -> list[dict]:
        rows: list[dict] = []
        recent = df.tail(window_days).reset_index(drop=True)
        timestamps = recent["timestamp"].astype(str) if "timestamp" in recent else None
        for col in CUSUM_COLUMNS:
            if col not in recent.columns:
                continue
            series = pd.to_numeric(recent[col], errors="coerce").dropna()
            cps = self.cusum.detect(series.values)

            # Keep drift *onsets* only: after a detection, skip re-triggers of
            # the same column/direction inside the cooldown window.
            kept, last_idx = [], {}
            for cp in cps:
                prev = last_idx.get(cp.direction)
                if prev is not None and cp.idx - prev < self.CUSUM_COOLDOWN:
                    last_idx[cp.direction] = cp.idx
                    continue
                last_idx[cp.direction] = cp.idx
                kept.append(cp)
            kept = kept[-self.CUSUM_MAX_PER_COLUMN:]

            for cp in kept:
                # excess over the detection threshold sets the severity;
                # score mapped to the Isolation-Forest scale
                # (-0.3 = warning, -0.5 = critical)
                excess = (cp.cusum - self.cusum.threshold) / self.cusum.threshold
                severity = "critical" if excess >= 1.0 else "warning"
                score = round(-(0.3 + 0.2 * min(1.0, excess)), 4)
                idx = series.index[cp.idx] if cp.idx < len(series.index) else None
                ts = (timestamps.iloc[idx] if timestamps is not None
                      and idx is not None else "")
                # normalise to the same 'YYYY-MM-DD HH:MM:SS' format as the
                # Isolation Forest rows (mixed formats break datetime parsing)
                if ts:
                    try:
                        ts = str(pd.Timestamp(ts))
                    except Exception:
                        pass
                arrow = "hausse" if cp.direction == "increase" else "baisse"
                rows.append({
                    "timestamp": str(ts),
                    "severity":  severity,
                    "score":     score,
                    "cause":     f"CUSUM — dérive lente ({arrow})",
                    "variable":  col,
                    "raw_value": (round(float(series.iloc[cp.idx]), 3)
                                  if cp.idx < len(series) else None),
                    "threshold": self.cusum.threshold,
                    "acknowledged": 0,
                })
        return rows

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self, df: pd.DataFrame, window_days: int = 365) -> list[dict]:
        """Run both detectors over the last *window_days* of the series."""
        if df is None or df.empty:
            return []
        rows = self._scan_iforest(df, window_days) + self._scan_cusum(df, window_days)
        rows.sort(key=lambda r: str(r["timestamp"]))
        return rows

    def scan_and_store(self, df: pd.DataFrame, window_days: int = 365) -> dict:
        """Scan and replace the (derived) contents of the anomalies table."""
        rows = self.scan(df, window_days)
        execute("DELETE FROM anomalies")
        if rows:
            insert_many("anomalies", rows)
        by_sev: dict[str, int] = {}
        for r in rows:
            by_sev[r["severity"]] = by_sev.get(r["severity"], 0) + 1
        by_detector = {
            "isolation_forest": sum(1 for r in rows if r["cause"].startswith("Isolation")),
            "cusum":            sum(1 for r in rows if r["cause"].startswith("CUSUM")),
        }
        return {"count": len(rows), "by_severity": by_sev,
                "by_detector": by_detector, "window_days": window_days}
