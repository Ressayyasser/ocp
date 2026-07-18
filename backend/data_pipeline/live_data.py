"""
live_data.py — Bridges historical Excel data with live SCADA simulation data.

historical_data (Excel ETL) typically lags behind the present day, while the
SCADA simulator writes 1 Hz per-GTA records into simulation_data — but only
while it is running. To keep the platform aligned with "today":

1. backfill_simulation_gap()    — synthesises one daily per-GTA record into
   simulation_data for every missing day between the end of historical_data
   and today. Values follow the statistical profile of the recent history and
   are deterministic per calendar day (seeded by the date), so repeated calls
   are idempotent and reproducible.
2. aggregate_simulation_daily() — aggregates simulation_data (whatever its
   native cadence: 1 Hz live rows or daily backfill rows) into daily
   plant-level records mapped onto the historical_data schema.
3. get_combined_daily_df()      — continuous daily series: historical_data
   followed by the simulation-derived days, with a 'source' column
   ('historical' | 'live_sim') so consumers can tell them apart.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from database.database import query, insert_many, execute

GTA_KEYS = ["GTA1", "GTA2", "GTA3"]

# Never synthesise more than this many days back (guards against a DB whose
# history is years stale — filling years of synthetic data would be misleading).
BACKFILL_MAX_DAYS = 90

BACKFILL_SIM_ID = "backfill_daily"

# Per-GTA fallback profile (used when a column is absent from the history)
_DEFAULTS = {
    "GTA1": {"mwh": 520.0, "debit": 175.0, "rendement": 41.5, "pression": 54.5, "temp": 455.0},
    "GTA2": {"mwh": 500.0, "debit": 175.0, "rendement": 35.0, "pression": 54.5, "temp": 455.0},
    "GTA3": {"mwh": 540.0, "debit": 175.0, "rendement": 40.5, "pression": 54.5, "temp": 455.0},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Historical statistics (mean/std of the recent past)
# ─────────────────────────────────────────────────────────────────────────────

def _hist_stats(n_days: int = 90) -> dict:
    rows = query("SELECT * FROM historical_data ORDER BY timestamp DESC LIMIT ?", [n_days])
    df = pd.DataFrame(rows)

    def ms(col: str, d_mean: float, d_std: float) -> tuple[float, float]:
        if not df.empty and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) >= 5:
                return float(s.mean()), float(max(s.std(), d_std * 0.05))
        return d_mean, d_std

    stats: dict = {}
    for i, gta in enumerate(GTA_KEYS, 1):
        d = _DEFAULTS[gta]
        stats[gta] = {
            "mwh":       ms(f"gta{i}",             d["mwh"],       30.0),
            "debit":     ms(f"debit_adm_gta{i}",   d["debit"],     8.0),
            "rendement": ms(f"rendement_gta{i}",   d["rendement"], 0.8),
            "pression":  ms(f"pression_adm_gta{i}", d["pression"], 0.8),
            "temp":      ms(f"temp_adm_gta{i}",    d["temp"],      5.0),
            "sout":      ms(f"debit_sout_gta{i}",  60.0,           5.0),
        }
    # consumption/production ratio for the bilan_net estimate
    cons_m, _ = ms("consumption", 300.0, 20.0)
    prod_m, _ = ms("production", 1560.0, 50.0)
    stats["cons_ratio"] = (cons_m / prod_m) if prod_m else 0.2
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  1. Backfill of the gap up to today
# ─────────────────────────────────────────────────────────────────────────────

def _synth_row(day: pd.Timestamp, gta: str, stats: dict, rng: np.random.Generator) -> dict:
    def draw(key: str) -> float:
        mean, std = stats[gta][key]
        return float(rng.normal(mean, std))

    mwh       = max(0.0, draw("mwh"))
    debit     = max(50.0, draw("debit"))
    rendement = float(np.clip(draw("rendement"), 20.0, 55.0))
    pression  = max(40.0, draw("pression"))
    temp      = float(np.clip(draw("temp"), 380.0, 495.0))
    sout      = max(0.0, draw("sout"))
    p_mw      = round(mwh / 24.0, 2)

    return {
        "simulation_id": BACKFILL_SIM_ID,
        "timestamp":     day.strftime("%Y-%m-%dT12:00:00"),
        "gta_type":      gta,
        "adm_debit":     round(debit, 1),
        "adm_temp":      round(temp, 1),
        "adm_pression":  round(pression, 2),
        "sout_debit":    round(sout, 1),
        "sout_pression": 8.9,
        "ext_debit":     90.0,
        "ext_pression":  0.09,
        "bp_pression":   0.9,
        "bp_debit":      8.7,
        "puissance_mw":  p_mw,
        "rendement":     round(rendement, 2),
        "vitesse":       3000.0,
        "vib1":          round(max(0.0, float(rng.normal(0.20, 0.03))), 3),
        "vib2":          round(max(0.0, float(rng.normal(0.40, 0.05))), 3),
        "dd3":           0.61,
        "oil_pression":  1.52,
        "oil_temp":      round(float(rng.normal(40.4, 0.6)), 1),
        "cos_phi":       0.855,
        "p_active":      p_mw,
        "p_reactive":    round(p_mw * 0.605, 2),
        "tension":       10.5,
        "posit_hp":      82.0,
        "posit_bp":      64.0,
        "vap_inlet":     round(pression + 0.6, 2),
        "cond_temp":     45.2,
        "cond_eau":      87.0,
        "level_pct":     78.1,
        "anomaly_flag":  0,
    }


def backfill_simulation_gap(today: str | None = None) -> int:
    """Insert synthetic daily per-GTA rows into simulation_data for every day
    missing between the end of historical_data and *today*. Idempotent
    (days that already have simulation rows — live or backfilled — are kept
    untouched). Returns the number of rows inserted."""
    today_ts = pd.Timestamp(today) if today else pd.Timestamp.now()
    today_ts = today_ts.normalize()

    last = query("SELECT MAX(timestamp) AS mx FROM historical_data")[0]["mx"]
    if not last:
        return 0
    start = pd.Timestamp(str(last)[:10]) + pd.Timedelta(days=1)
    start = max(start, today_ts - pd.Timedelta(days=BACKFILL_MAX_DAYS))
    if start > today_ts:
        return 0

    existing = {r["d"] for r in
                query("SELECT DISTINCT substr(timestamp,1,10) AS d FROM simulation_data")}

    stats = _hist_stats()
    rows: list[dict] = []
    for day in pd.date_range(start, today_ts, freq="D"):
        key = day.strftime("%Y-%m-%d")
        if key in existing:
            continue
        rng = np.random.default_rng(day.toordinal())   # deterministic per date
        for gta in GTA_KEYS:
            rows.append(_synth_row(day, gta, stats, rng))
    if rows:
        # parent record for the FK simulation_data.simulation_id → simulations
        execute(
            "INSERT OR IGNORE INTO simulations "
            "(simulation_id, scenario_type, base_params, start_time, status) "
            "VALUES (?, 'backfill', '{}', ?, 'completed')",
            [BACKFILL_SIM_ID, pd.Timestamp.now().isoformat()],
        )
        insert_many("simulation_data", rows)
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Daily aggregation of simulation_data → historical schema
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_simulation_daily(after: str | None = None) -> pd.DataFrame:
    """Aggregate simulation_data to one plant-level record per day, mapped
    onto the historical_data schema. *after* (YYYY-MM-DD) keeps only strictly
    later days — pass the last historical date to avoid overlaps."""
    sql, params = "SELECT * FROM simulation_data", []
    if after:
        sql += " WHERE substr(timestamp,1,10) > ?"
        params = [str(after)[:10]]
    rows = query(sql, params)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = df["timestamp"].astype(str).str[:10]
    grouped = df.groupby(["date", "gta_type"]).mean(numeric_only=True).reset_index()

    cons_ratio = _hist_stats().get("cons_ratio", 0.2)

    records: list[dict] = []
    for date, day in grouped.groupby("date"):
        per = {r["gta_type"]: r for _, r in day.iterrows()}
        rec: dict = {"timestamp": date, "source": "live_sim", "year": int(str(date)[:4])}

        production = steam_hp = steam_mp = steam_bp = 0.0
        rendements, pressions, temps, vibs = [], [], [], []
        for i, gta in enumerate(GTA_KEYS, 1):
            r = per.get(gta)
            if r is None:
                continue
            mwh = float(r["puissance_mw"]) * 24.0
            rec[f"gta{i}"]             = round(mwh, 1)
            rec[f"debit_adm_gta{i}"]   = round(float(r["adm_debit"]), 1)
            rec[f"debit_sout_gta{i}"]  = round(float(r["sout_debit"]), 1)
            rec[f"rendement_gta{i}"]   = round(float(r["rendement"]), 2)
            rec[f"pression_adm_gta{i}"] = round(float(r["adm_pression"]), 2)
            rec[f"temp_adm_gta{i}"]    = round(float(r["adm_temp"]), 1)
            production += mwh
            steam_hp   += float(r["adm_debit"])
            steam_mp   += float(r["sout_debit"])
            steam_bp   += float(r["bp_debit"])
            rendements.append(float(r["rendement"]))
            pressions.append(float(r["adm_pression"]))
            temps.append(float(r["adm_temp"]))
            vibs.append(float(r["vib2"]))

        consumption = production * cons_ratio
        rec.update({
            "gtaa": 0.0, "gtab": 0.0,
            "production":  round(production, 1),
            "consumption": round(consumption, 1),
            "bilan_net":   round(production - consumption, 1),
            "steam_hp":    round(steam_hp, 1),
            "steam_mp":    round(steam_mp, 1),
            "steam_bp":    round(steam_bp, 1),
            "steam_ratio": round(steam_hp / production, 4) if production else 0.0,
            "efficiency":  round(float(np.mean(rendements)) / 100.0, 4) if rendements else None,
            "pressure":    round(float(np.mean(pressions)), 2) if pressions else None,
            "temperature": round(float(np.mean(temps)), 1) if temps else None,
            "vibration":   round(float(np.mean(vibs)), 3) if vibs else None,
        })
        records.append(rec)

    out = pd.DataFrame(records)
    return out.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Combined continuous series
# ─────────────────────────────────────────────────────────────────────────────

def get_combined_daily_df(limit: int = 2000, backfill: bool = True) -> pd.DataFrame:
    """historical_data + daily simulation aggregates, as one continuous daily
    series ending today. Raw (uncleaned) — callers apply clean_dataframe /
    add_all_features as usual."""
    if backfill:
        try:
            backfill_simulation_gap()
        except Exception as exc:               # never break serving on backfill issues
            print(f"[live_data] backfill skipped: {exc}")

    hist_rows = query("SELECT * FROM historical_data ORDER BY timestamp DESC LIMIT ?", [limit])
    hist_df = pd.DataFrame(hist_rows)
    last_date = None
    if not hist_df.empty:
        hist_df = hist_df.drop(columns=["id"], errors="ignore")
        hist_df["source"] = "historical"
        last_date = str(hist_df["timestamp"].max())[:10]

    sim_df = aggregate_simulation_daily(after=last_date)

    if hist_df.empty:
        combined = sim_df
    elif sim_df.empty:
        combined = hist_df
    else:
        combined = pd.concat([hist_df, sim_df], ignore_index=True)

    if combined.empty:
        return combined
    combined["timestamp"] = combined["timestamp"].astype(str)
    return combined.sort_values("timestamp").reset_index(drop=True)
