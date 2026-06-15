"""
scada_loader.py — Adapters for SCADA real-time feed and CSV/JSON snapshots.
"""

from __future__ import annotations
import json
import pandas as pd


def load_scada_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return df


def load_scada_json(filepath: str) -> pd.DataFrame:
    with open(filepath) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def parse_scada_message(msg: dict) -> dict:
    """
    Normalise a single SCADA WebSocket / MQTT message into our canonical schema.
    """
    gta1 = float(msg.get("gta1", msg.get("GTA1_prod", 0)))
    gta2 = float(msg.get("gta2", msg.get("GTA2_prod", 0)))
    gta3 = float(msg.get("gta3", msg.get("GTA3_prod", 0)))
    gtaa = float(msg.get("gtaa", msg.get("GTAA_prod", 0)))
    gtab = float(msg.get("gtab", msg.get("GTAB_prod", 0)))

    steam_hp = float(msg.get("steam_hp", msg.get("vapeur_hp", 0)))
    steam_mp = float(msg.get("steam_mp", msg.get("vapeur_mp", 0)))
    steam_bp = float(msg.get("steam_bp", msg.get("vapeur_bp", 0)))

    production  = gta1 + gta2 + gta3 + gtaa + gtab
    consumption = float(msg.get("consumption", msg.get("conso", production * 0.15)))
    bilan_net   = production - consumption
    efficiency  = production / steam_hp if steam_hp > 0 else 0.0

    return {
        "timestamp":   msg.get("timestamp", msg.get("ts")),
        "gta1": gta1, "gta2": gta2, "gta3": gta3,
        "gtaa": gtaa, "gtab": gtab,
        "steam_hp": steam_hp, "steam_mp": steam_mp, "steam_bp": steam_bp,
        "pressure":    float(msg.get("pressure",    msg.get("pression", 0))),
        "vibration":   float(msg.get("vibration",   0)),
        "temperature": float(msg.get("temperature", 0)),
        "production":  production,
        "consumption": consumption,
        "bilan_net":   bilan_net,
        "efficiency":  efficiency,
    }