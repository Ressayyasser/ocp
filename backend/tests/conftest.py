"""
conftest.py — Shared fixtures for the backend test suite.

Isolation strategy
──────────────────
* DB_PATH    → temporary SQLite file  (never touches cogen_ocp.db)
* MODEL_DIR  → temporary directory    (never touches models/anomaly)
Both env vars are read at import time by the modules under test, so they
are set here BEFORE any backend import happens.
"""

import os
import sys
import tempfile

# ── Isolate environment BEFORE importing backend modules ─────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

_TMP = tempfile.mkdtemp(prefix="ocp_tests_")
os.environ["DB_PATH"] = os.path.join(_TMP, "test_cogen.db")
os.environ["MODEL_DIR"] = os.path.join(_TMP, "models_anomaly")

import numpy as np
import pandas as pd
import pytest

from database.database import init_db, insert_many


def make_synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Physically plausible daily GTA dataset (values match plant nominals)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D")

    gta1 = rng.normal(520, 30, n)
    gta2 = rng.normal(500, 30, n)
    gta3 = rng.normal(540, 30, n)
    production = gta1 + gta2 + gta3
    consumption = rng.normal(300, 20, n)
    steam_hp = rng.normal(530, 25, n)

    df = pd.DataFrame({
        "timestamp": ts,
        "gta1": gta1, "gta2": gta2, "gta3": gta3,
        "gtaa": np.zeros(n), "gtab": np.zeros(n),
        "production": production,
        "consumption": consumption,
        "bilan_net": production - consumption,
        "steam_hp": steam_hp,
        "steam_mp": rng.normal(400, 20, n),
        "steam_bp": rng.normal(120, 10, n),
        "steam_ratio": steam_hp / production,
        "efficiency": rng.normal(0.405, 0.01, n),
        "pressure": rng.normal(54.5, 0.8, n),
        "temperature": rng.normal(455, 5, n),
        "vibration": rng.normal(1.3, 0.2, n),
        "rendement_gta1": rng.normal(41.5, 0.8, n),
        "rendement_gta2": rng.normal(35.0, 0.8, n),
        "rendement_gta3": rng.normal(40.5, 0.8, n),
        "debit_adm_gta1": rng.normal(175, 8, n),
        "debit_adm_gta2": rng.normal(175, 8, n),
        "debit_adm_gta3": rng.normal(175, 8, n),
        "debit_sout_gta1": rng.normal(60, 5, n),
        "debit_sout_gta2": rng.normal(60, 5, n),
        "debit_sout_gta3": rng.normal(60, 5, n),
        "pression_adm_gta1": rng.normal(54.5, 0.8, n),
        "pression_adm_gta2": rng.normal(54.5, 0.8, n),
        "pression_adm_gta3": rng.normal(54.5, 0.8, n),
        "temp_adm_gta1": rng.normal(455, 5, n),
        "temp_adm_gta2": rng.normal(455, 5, n),
        "temp_adm_gta3": rng.normal(455, 5, n),
        "year": np.full(n, 2024),
    })
    return df


@pytest.fixture(scope="session")
def synth_df() -> pd.DataFrame:
    return make_synthetic_df()


@pytest.fixture(scope="session", autouse=True)
def seeded_db(synth_df):
    """Initialise the temp DB and seed historical_data once per session."""
    init_db()
    rows = []
    for _, r in synth_df.iterrows():
        row = r.to_dict()
        row["timestamp"] = str(row["timestamp"])
        rows.append(row)
    insert_many("historical_data", rows)
    yield
