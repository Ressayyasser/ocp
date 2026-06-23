"""
feature_engineering.py — Derives all secondary features needed by
forecasting, anomaly detection, and the RL agent.

New columns added
─────────────────
efficiency, steam_ratio, gta_balance, import_export_ratio,
*_roll_7d, *_roll_30d, *_std_7d,
delta_pressure, delta_vibration, delta_production, delta_bilan, delta_efficiency,
hour, day_of_week, month, quarter, is_weekend,
hour_sin, hour_cos, month_sin, month_cos,
*_lag_1 … *_lag_24
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère des caractéristiques (features) pour l'entraînement du modèle.
    """
    df_feat = df.copy()
    df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"])
    
    # 1. Features Temporelles (Saisonnalité de la demande/température extérieure)
    df_feat["month"] = df_feat["timestamp"].dt.month
    df_feat["day_of_week"] = df_feat["timestamp"].dt.dayofweek
    df_feat["is_weekend"] = df_feat["day_of_week"].isin([5, 6]).astype(int)
    
    # 2. Feature d'état opérationnel (Combien de GTA sont actifs simultanément ?)
    df_feat["gta1_active"] = (df_feat["gta1"] > 0.5).astype(int)
    df_feat["gta2_active"] = (df_feat["gta2"] > 0.5).astype(int)
    df_feat["gta3_active"] = (df_feat["gta3"] > 0.5).astype(int)
    df_feat["active_gta_count"] = df_feat["gta1_active"] + df_feat["gta2_active"] + df_feat["gta3_active"]
    
    # 3. Features Glissantes / Historiques (Lags et Moyennes mobiles)
    # Très important pour capturer l'inertie thermique et les tendances
    lag_features = ["production", "efficiency", "steam_hp"]
    for col in lag_features:
        if col in df_feat.columns:
            # Valeur de la veille (Lag 1)
            df_feat[f"{col}_lag_1"] = df_feat[col].shift(1)
            # Moyenne mobile sur 3 jours
            df_feat[f"{col}_roll_mean_3d"] = df_feat[col].shift(1).rolling(window=3, min_periods=1).mean()
    
    # 4. Indicateur d'intensité énergétique spécifique (Tonnes de vapeur par MWh produit)
    df_feat["steam_per_mwh"] = np.where(
        df_feat["production"] > 0, 
        df_feat["steam_hp"] / df_feat["production"], 
        0
    )
    
    # Suppression des lignes contenant des NaN créés par les Lags
    df_feat = df_feat.dropna(subset=[f"{c}_lag_1" for c in lag_features if c in df_feat.columns])
    
    return df_feat.reset_index(drop=True)

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _efficiency(df)
    df = _steam_ratio(df)
    df = _gta_balance(df)
    df = _import_export_ratio(df)
    df = _rolling(df)
    df = _deltas(df)
    df = _temporal(df)
    df = _lags(df)
    return df


# ── Individual feature builders ───────────────────────────────────────────────

def _efficiency(df: pd.DataFrame) -> pd.DataFrame:
    # "efficiency" already comes from the Excel loader as avg rendement/100
    # Only recompute if missing
    if "efficiency" not in df.columns or df["efficiency"].isna().all():
        prod  = _first(df, "production")
        steam = _first(df, "steam_hp")
        if prod and steam:
            df["efficiency"] = np.where(df[steam] > 0, df[prod] / df[steam], 0.0)
    return df


def _steam_ratio(df: pd.DataFrame) -> pd.DataFrame:
    # Already computed in loader; only fill if missing
    if "steam_ratio" not in df.columns or df["steam_ratio"].isna().all():
        mp = _first(df, "steam_mp")
        hp = _first(df, "steam_hp")
        if mp and hp:
            df["steam_ratio"] = np.where(df[hp] > 0, df[mp] / df[hp], 0.0)
    return df


def _gta_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Difference between best and worst performing GTA (efficiency spread)."""
    rend_cols = [c for c in ["rendement_gta1", "rendement_gta2", "rendement_gta3"]
                 if c in df.columns]
    if len(rend_cols) >= 2:
        active = df[rend_cols].replace(0, np.nan)
        df["gta_balance"] = active.max(axis=1) - active.min(axis=1)
    elif _first(df, "gta1", "gta2", "gta3"):
        ecols = [c for c in ["gta1","gta2","gta3"] if c in df.columns]
        df["gta_balance"] = df[ecols].sum(axis=1)
    return df


def _import_export_ratio(df: pd.DataFrame) -> pd.DataFrame:
    prod = _first(df, "production")
    cons = _first(df, "consumption")
    if prod and cons:
        df["import_export_ratio"] = np.where(df[prod] > 0, df[cons] / df[prod], 1.0)
    return df


def _rolling(df: pd.DataFrame, windows=(7, 30)) -> pd.DataFrame:
    targets = [c for c in [
        "production", "bilan_net", "efficiency", "steam_hp", "pressure", "vibration",
        "gta1", "gta2", "gta3",
        "debit_adm_gta1", "debit_adm_gta2", "debit_adm_gta3",
        "rendement_gta1", "rendement_gta2", "rendement_gta3",
    ] if c in df.columns]
    for col in targets:
        for w in windows:
            df[f"{col}_roll_{w}d"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_std_7d"] = df[col].rolling(7, min_periods=1).std().fillna(0)
    return df


def _deltas(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "pressure", "vibration", "production", "bilan_net", "efficiency",
        "gta1", "gta2", "gta3", "steam_hp", "steam_mp",
        "debit_adm_gta1", "debit_adm_gta2", "debit_adm_gta3",
    ]:
        if col in df.columns:
            df[f"delta_{col}"] = df[col].diff().fillna(0)
    return df


def _temporal(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["timestamp"])
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"]       = ts.dt.month
    df["quarter"]     = ts.dt.quarter
    df["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _lags(df: pd.DataFrame, lags=(1, 3, 6, 12, 24)) -> pd.DataFrame:
    for col in ["production","bilan_net","efficiency"]:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    df = df.bfill(limit=max(lags))
    return df


def _first(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first candidate column that exists in df."""
    return next((c for c in candidates if c in df.columns), None)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"timestamp","source_file","year","id"}
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]