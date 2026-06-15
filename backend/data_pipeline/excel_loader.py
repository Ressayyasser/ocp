"""
excel_loader.py — Loads GTA_Donnees_Completes_20XX_GENERATED.xlsx files.

Real file structure (confirmed from 2022–2025 datasets):
  Sheet "Journalier GTA1" → 365/366 rows, columns:
    Date, Débit adm (t/h), T° adm (°C), P adm (bar), H adm (kJ/kg),
    Débit sout. (t/h), T° sout. (°C), P sout. (bar), H sout. (kJ/kg),
    Débit ext. (t/h), T° ext. (°C), P ext. (mbar), H ext. (kJ/kg),
    Énergie (MWh), Rendement (%)
  Same structure for Journalier GTA2 and Journalier GTA3.

  Sheet "Données Mensuelles" → monthly summary (skipped for daily pipeline).

Output canonical columns (per daily merged row):
  timestamp, gta1, gta2, gta3,
  debit_adm_gta1, temp_adm_gta1, pression_adm_gta1,
  debit_adm_gta2, temp_adm_gta2, pression_adm_gta2,
  debit_adm_gta3, temp_adm_gta3, pression_adm_gta3,
  debit_sout_gta1, debit_sout_gta2, debit_sout_gta3,
  debit_ext_gta1,  debit_ext_gta2,  debit_ext_gta3,
  production, efficiency, pressure, temperature,
  steam_hp, steam_mp, steam_bp, bilan_net
"""

from __future__ import annotations
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Sheet parsers ─────────────────────────────────────────────────────────────

def _parse_journalier(path: str, sheet: str, prefix: str) -> pd.DataFrame:
    """
    Parse one 'Journalier GTAx' sheet.
    Returns a DataFrame indexed by Date with prefixed columns.
    Rows with Énergie == 0 are kept but flagged (GTA2 was offline in 2025).
    """
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

    # Rename to canonical internal names
    rename = {
        "Date":             "timestamp",
        "Débit adm (t/h)":  f"debit_adm_{prefix}",
        "T° adm (°C)":      f"temp_adm_{prefix}",
        "P adm (bar)":      f"pression_adm_{prefix}",
        "H adm (kJ/kg)":    f"h_adm_{prefix}",
        "Débit sout. (t/h)":f"debit_sout_{prefix}",
        "T° sout. (°C)":    f"temp_sout_{prefix}",
        "P sout. (bar)":    f"p_sout_{prefix}",
        "H sout. (kJ/kg)":  f"h_sout_{prefix}",
        "Débit ext. (t/h)": f"debit_ext_{prefix}",
        "T° ext. (°C)":     f"temp_ext_{prefix}",
        "P ext. (mbar)":    f"p_ext_{prefix}",
        "H ext. (kJ/kg)":   f"h_ext_{prefix}",
        "Énergie (MWh)":    f"energie_{prefix}",
        "Rendement (%)":    f"rendement_{prefix}",
    }
    df = df.rename(columns=rename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df


# ── Workbook loader ───────────────────────────────────────────────────────────

def _load_workbook(path: str, year: int) -> pd.DataFrame:
    """
    Merge the three Journalier sheets from one workbook into a daily DataFrame.
    """
    xl   = pd.ExcelFile(path, engine="openpyxl")
    sheets = {s.lower(): s for s in xl.sheet_names}

    # Try to find each GTA sheet (case-insensitive, partial match)
    def _find(keyword: str) -> str | None:
        for k, v in sheets.items():
            if keyword in k:
                return v
        return None

    s1 = _find("gta1") or _find("gta 1")
    s2 = _find("gta2") or _find("gta 2")
    s3 = _find("gta3") or _find("gta 3")

    if not s1:
        raise ValueError(f"No GTA1 sheet found in {path}")

    df1 = _parse_journalier(path, s1, "gta1")
    base = df1[["timestamp"]].copy()

    if s2:
        df2 = _parse_journalier(path, s2, "gta2")
        base = base.merge(df2, on="timestamp", how="left")
    if s3:
        df3 = _parse_journalier(path, s3, "gta3")
        base = base.merge(df3, on="timestamp", how="left")

    # Re-attach GTA1 columns
    base = base.merge(df1.drop(columns=["timestamp"]), left_index=True, right_index=True, how="left")

    # ── Derived canonical columns ────────────────────────────────────────────
    e1 = base.get("energie_gta1", pd.Series(0.0, index=base.index)).fillna(0)
    e2 = base.get("energie_gta2", pd.Series(0.0, index=base.index)).fillna(0)
    e3 = base.get("energie_gta3", pd.Series(0.0, index=base.index)).fillna(0)

    base["gta1"]       = e1
    base["gta2"]       = e2
    base["gta3"]       = e3
    base["production"] = e1 + e2 + e3

    # Efficiency: average of operating GTAs (non-zero)
    rend_cols = [c for c in ["rendement_gta1", "rendement_gta2", "rendement_gta3"]
                 if c in base.columns]
    if rend_cols:
        rend_df = base[rend_cols].replace(0, np.nan)
        base["efficiency"] = rend_df.mean(axis=1) / 100.0   # convert % → fraction
    else:
        base["efficiency"] = np.nan

    # Pressure: average HP admission pressure across operating GTAs
    p_cols = [c for c in ["pression_adm_gta1", "pression_adm_gta2", "pression_adm_gta3"]
              if c in base.columns]
    if p_cols:
        p_df = base[p_cols].replace(0, np.nan)
        base["pressure"] = p_df.mean(axis=1)
    else:
        base["pressure"] = np.nan

    # Temperature: average admission temperature
    t_cols = [c for c in ["temp_adm_gta1", "temp_adm_gta2", "temp_adm_gta3"]
              if c in base.columns]
    if t_cols:
        t_df = base[t_cols].replace(0, np.nan)
        base["temperature"] = t_df.mean(axis=1)
    else:
        base["temperature"] = np.nan

    # Steam proxies: HP = total admission flow (t/h), MP = soutirage flow, BP = extraction flow
    d_adm_cols  = [c for c in ["debit_adm_gta1",  "debit_adm_gta2",  "debit_adm_gta3"]  if c in base.columns]
    d_sout_cols = [c for c in ["debit_sout_gta1", "debit_sout_gta2", "debit_sout_gta3"] if c in base.columns]
    d_ext_cols  = [c for c in ["debit_ext_gta1",  "debit_ext_gta2",  "debit_ext_gta3"]  if c in base.columns]

    base["steam_hp"] = base[d_adm_cols].fillna(0).sum(axis=1)  if d_adm_cols  else np.nan
    base["steam_mp"] = base[d_sout_cols].fillna(0).sum(axis=1) if d_sout_cols else np.nan
    base["steam_bp"] = base[d_ext_cols].fillna(0).sum(axis=1)  if d_ext_cols  else np.nan

    # Steam ratio
    base["steam_ratio"] = np.where(
        base["steam_hp"] > 0, base["steam_mp"] / base["steam_hp"], np.nan
    )

    # Bilan net: production - ~14% auto-consumption estimate (no separate conso column in data)
    base["consumption"] = base["production"] * 0.14
    base["bilan_net"]   = base["production"] - base["consumption"]

    # Vibration: not in original data → simulate placeholder (will be replaced by SCADA)
    base["vibration"] = np.nan

    base["year"] = year
    base = base.sort_values("timestamp").reset_index(drop=True)
    return base


# ── Public API ────────────────────────────────────────────────────────────────

def load_all_gta_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load all GTA Excel workbooks from *data_dir*.
    Accepts filenames like GTA_Donnees_Completes_2022_GENERATED.xlsx
    Returns a single sorted canonical DataFrame.
    """
    pattern = os.path.join(data_dir, "*.xlsx")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Excel files found in {data_dir!r}")

    frames: list[pd.DataFrame] = []
    for f in files:
        stem = Path(f).stem
        # Extract 4-digit year from filename
        year = next((int(t) for t in stem.split("_") if t.isdigit() and len(t) == 4), 2023)
        print(f"  Loading {Path(f).name}  (year={year})")
        try:
            df = _load_workbook(f, year)
            frames.append(df)
            print(f"    → {len(df)} days, production range "
                  f"[{df['production'].min():.0f}–{df['production'].max():.0f}] MWh")
        except Exception as exc:
            print(f"    ⚠  Skipped: {exc}")

    if not frames:
        raise ValueError("No data could be loaded from Excel files")

    merged = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    # Remove exact duplicate dates (keep first)
    merged = merged.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    print(f"  → Total: {len(merged):,} rows, {len(merged.columns)} columns")
    return merged


# ── DB persistence ────────────────────────────────────────────────────────────

DB_COLS = [
    "timestamp", "gta1", "gta2", "gta3", "gtaa", "gtab",
    "steam_hp", "steam_mp", "steam_bp", "steam_ratio",
    "production", "consumption", "bilan_net",
    "efficiency", "pressure", "temperature", "vibration",
    "debit_adm_gta1", "debit_adm_gta2", "debit_adm_gta3",
    "debit_sout_gta1", "debit_sout_gta2", "debit_sout_gta3",
    "debit_ext_gta1", "debit_ext_gta2", "debit_ext_gta3",
    "rendement_gta1", "rendement_gta2", "rendement_gta3",
    "pression_adm_gta1", "pression_adm_gta2", "pression_adm_gta3",
    "temp_adm_gta1", "temp_adm_gta2", "temp_adm_gta3",
    "year",
]


def load_to_database(data_dir: str = "data"):
    """Load Excel data and persist to SQLite historical_data table."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from database.database import init_db, insert_many

    init_db()
    df = load_all_gta_data(data_dir)

    df["timestamp"] = df["timestamp"].astype(str)
    for c in DB_COLS:
        if c not in df.columns:
            df[c] = None

    rows = df[DB_COLS].replace({float("nan"): None}).to_dict(orient="records")
    insert_many("historical_data", rows)
    print(f"  → Inserted {len(rows):,} rows into historical_data")


if __name__ == "__main__":
    load_to_database()