"""
data_validator.py — Quality-gate checks run after loading and before training.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class DataValidationReport:
    def __init__(self):
        self.checks: list[dict] = []
        self.passed = True

    def add(self, name: str, ok: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": ok, "detail": detail})
        if not ok:
            self.passed = False

    def summary(self) -> dict:
        return {
            "overall_passed": self.passed,
            "total":  len(self.checks),
            "failed": [c for c in self.checks if not c["passed"]],
            "all":    self.checks,
        }


def validate(df: pd.DataFrame) -> DataValidationReport:
    r = DataValidationReport()

    r.add("non_empty", len(df) > 0, f"{len(df)} rows")
    if len(df) == 0:
        return r

    # Timestamp
    has_ts = "timestamp" in df.columns
    r.add("has_timestamp", has_ts)
    if has_ts:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        null_ts = ts.isna().sum()
        r.add("ts_parseable",  null_ts < len(df) * 0.05, f"{null_ts} unparseable")
        r.add("ts_sorted",     ts.dropna().is_monotonic_increasing)
        span = (ts.max() - ts.min()).days
        r.add("ts_span_30d",   span >= 30, f"{span} days")

    # Key columns
    for col in ("production", "bilan_net"):
        exists = col in df.columns
        r.add(f"has_{col}", exists)

    # Missing-value ratio per numeric column
    for col in df.select_dtypes(include=[np.number]).columns:
        pct = df[col].isna().mean()
        r.add(f"missing_{col}", pct < 0.30, f"{pct:.1%} missing")

    # Physical sanity
    if "pressure"  in df.columns:
        r.add("pressure_positive",  (df["pressure"]  >= 0).all())
    if "vibration" in df.columns:
        extreme = (df["vibration"] > 50).sum()
        r.add("vibration_sane", extreme < len(df) * 0.01, f"{extreme} extreme")

    return r