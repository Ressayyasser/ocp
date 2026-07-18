"""
preprocessing.py — Clean, interpolate, clip, and resample GTA data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def clean_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage des données historiques après chargement et validation.
    """
    df_clean = df.copy()
    
    # 1. Tri et gestion de l'index temporel
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)
    
    # 2. Traitement des valeurs aberrantes / Physiquement impossibles
    # Si le rendement ou les débits sont négatifs à cause d'un bruit de capteur
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if "debit" in col or "rendement" in col or "energie" in col:
            df_clean[col] = df_clean[col].clip(lower=0)

    # 3. Imputation des valeurs manquantes (Ex: Vibration manquante dans l'Excel)
    # Pour la vibration, on peut définir une valeur de référence stable (ex: 15 mm/s)
    if "vibration" in df_clean.columns:
        df_clean["vibration"] = df_clean["vibration"].fillna(15.0)
        
    # Pour les variables thermodynamiques, interpolation linéaire si trous mineurs
    cols_to_interpolate = ["pressure", "temperature", "efficiency", "steam_ratio"]
    for col in cols_to_interpolate:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].interpolate(method="linear").bfill()

    return df_clean

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Parse & sort timestamps
      2. Drop duplicate timestamps
      3. Force columns to numeric and convert None/strings to NaN (FIX)
      4. Replace physically impossible sensor values with NaN
      5. Linear interpolation (max 6-step gap)
      6. Fill remaining NaN with column median
      7. Clip extreme outliers beyond ±4σ
    """
    df = df.copy()
    # 'source' tags the row origin (historical | live_sim) — keep it textual
    exclude_cols = ['Date', 'timestamp', 'Mois', 'source']
    
    # ── Timestamp ─────────────────────────────────────────────────────────────
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = (df.dropna(subset=["timestamp"])
                .drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True))
        df = df.set_index("timestamp")

    # ── 🛠️ FIX: Force all sensor/feature columns to numeric type ──────────────
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Now select_dtypes will successfully capture all your numeric data columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Physical sanity ───────────────────────────────────────────────────────
    neg_zero_cols = [c for c in num_cols if any(k in c for k in
                     ["gta", "prod", "steam", "vapeur", "consumption"])]
    for c in neg_zero_cols:
        df.loc[df[c] < 0, c] = np.nan
        
    for guard, lo, hi in [("pressure", 0, 200), ("vibration", 0, 50), ("temperature", 0, 600)]:
        if guard in df.columns:
            df.loc[(df[guard] < lo) | (df[guard] > hi), guard] = np.nan

    # ── Interpolate small gaps ────────────────────────────────────────────────
    for c in num_cols:
        df[c] = df[c].interpolate(method="linear", limit=6)

    # ── Median fill ───────────────────────────────────────────────────────────
    for c in num_cols:
        med = df[c].median()
        if not np.isnan(med):
            df[c] = df[c].fillna(med)

    # ── 4-sigma clip ──────────────────────────────────────────────────────────
    for c in num_cols:
        mu, sigma = df[c].mean(), df[c].std()
        if sigma > 0:
            df[c] = df[c].clip(mu - 4 * sigma, mu + 4 * sigma)
    

    if df.index.name == "timestamp":
        df = df.reset_index()

    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    out = df.select_dtypes(include=[np.number]).resample("1h").mean()
    out = out.interpolate(method="linear", limit=3).reset_index()
    return out


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    num = df.select_dtypes(include=[np.number])
    sum_cols  = [c for c in num if any(k in c for k in ["gta","prod","steam","bilan","consumption"])]
    mean_cols = [c for c in num if c not in sum_cols]
    parts = []
    if sum_cols:  parts.append(num[sum_cols].resample("1D").sum())
    if mean_cols: parts.append(num[mean_cols].resample("1D").mean())
    return pd.concat(parts, axis=1).reset_index() if parts else num.resample("1D").mean().reset_index()


def normalize_minmax(df: pd.DataFrame, columns: list | None = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    params: dict = {}
    for c in cols:
        if c not in df.columns:
            continue
        lo, hi = df[c].min(), df[c].max()
        rng = hi - lo
        df[c] = (df[c] - lo) / rng if rng > 0 else 0.0
        params[c] = {"min": float(lo), "max": float(hi)}
    return df, params


def denormalize(values: np.ndarray, params: dict) -> np.ndarray:
    return values * (params["max"] - params["min"]) + params["min"]