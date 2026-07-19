"""
database.py — SQLite core (swap to PostgreSQL via DB_URL env var).
"""

import sqlite3
import os

DB_PATH = os.environ.get("DB_PATH", "cogen_ocp.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS historical_data (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            -- GTA energy production (MWh/day)
            gta1      REAL, gta2 REAL, gta3 REAL, gtaa REAL, gtab REAL,
            -- Steam flows (t/h averages)
            steam_hp  REAL, steam_mp REAL, steam_bp REAL, steam_ratio REAL,
            -- Energy totals
            production REAL, consumption REAL, bilan_net REAL,
            -- Performance
            efficiency REAL, pressure REAL, temperature REAL, vibration REAL,
            -- Per-GTA thermodynamic columns (from Journalier sheets)
            debit_adm_gta1  REAL, debit_adm_gta2  REAL, debit_adm_gta3  REAL,
            debit_sout_gta1 REAL, debit_sout_gta2 REAL, debit_sout_gta3 REAL,
            debit_ext_gta1  REAL, debit_ext_gta2  REAL, debit_ext_gta3  REAL,
            rendement_gta1  REAL, rendement_gta2  REAL, rendement_gta3  REAL,
            pression_adm_gta1 REAL, pression_adm_gta2 REAL, pression_adm_gta3 REAL,
            temp_adm_gta1   REAL, temp_adm_gta2   REAL, temp_adm_gta3   REAL,
            -- Ajustement : Ajout des colonnes d'Enthalpie manquantes
            h_adm_gta1 REAL, h_adm_gta2 REAL, h_adm_gta3 REAL,
            h_sout_gta1 REAL, h_sout_gta2 REAL, h_sout_gta3 REAL,
            h_ext_gta1 REAL, h_ext_gta2 REAL, h_ext_gta3 REAL,
            year            INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_hist_ts ON historical_data(timestamp);

        CREATE TABLE IF NOT EXISTS simulations (
            simulation_id TEXT PRIMARY KEY,
            scenario_type TEXT,
            base_params TEXT,
            start_time TEXT,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS simulation_data (
            id INTEGER PRIMARY KEY,
            simulation_id TEXT,
            timestamp TEXT,
            gta_type TEXT,

            -- High Pressure (Admission)
            adm_debit REAL, adm_temp REAL, adm_pression REAL,

            -- Medium Pressure (Soutirage)
            sout_debit REAL, sout_pression REAL,

            -- External / Extraction
            ext_debit REAL, ext_pression REAL,

            -- Low Pressure (Basse Pression)
            bp_pression REAL, bp_debit REAL,

            -- Performance & Energy
            puissance_mw REAL, rendement REAL,

            -- Mechanical & Vibrations
            vitesse REAL, vib1 REAL, vib2 REAL, dd3 REAL,

            -- Lubrication Oil System
            oil_pression REAL, oil_temp REAL,

            -- Electrical Parameters
            cos_phi REAL, p_active REAL, p_reactive REAL, tension REAL,

            -- Actuators & Targets
            posit_hp REAL, posit_bp REAL, vap_inlet REAL,

            -- Condenser Loop
            cond_temp REAL, cond_eau REAL, level_pct REAL,

            anomaly_flag INTEGER,
            FOREIGN KEY(simulation_id) REFERENCES simulations(simulation_id)
        );
                      
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
            variable        TEXT NOT NULL,
            horizon         TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            confidence      REAL,
            model_version   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp);

        CREATE TABLE IF NOT EXISTS anomalies (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
            severity     TEXT NOT NULL,
            score        REAL NOT NULL,
            cause        TEXT,
            variable     TEXT,
            raw_value    REAL,
            threshold    REAL,
            acknowledged INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_anom_ts ON anomalies(timestamp);

        CREATE TABLE IF NOT EXISTS recommendations (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL DEFAULT (datetime('now')),
            action            TEXT NOT NULL,
            action_index      INTEGER,
            expected_gain_mwh REAL,
            economic_gain_dh  REAL,
            confidence        REAL,
            shap_explanation  TEXT,
            applied           INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
            message      TEXT NOT NULL,
            level        TEXT NOT NULL,
            source       TEXT,
            acknowledged INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS causal_links (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_var  TEXT NOT NULL,
            target_var  TEXT NOT NULL,
            lag         INTEGER NOT NULL,
            strength    REAL NOT NULL,
            p_value     REAL,
            computed_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rl_episodes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            episode      INTEGER NOT NULL,
            total_reward REAL,
            steps        INTEGER,
            avg_bilan    REAL,
            trained_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS component_health (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
            component       TEXT NOT NULL,
            health_score    REAL NOT NULL,
            rul_days        REAL,
            vibration       REAL,
            pressure        REAL,
            temperature     REAL,
            hours_operation REAL
        );

        -- Hot-path indexes (alerts feed polls + live SCADA queries)
        CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
        CREATE INDEX IF NOT EXISTS idx_alerts_level_ts ON alerts(level, timestamp);
        CREATE INDEX IF NOT EXISTS idx_simdata_gta_id
            ON simulation_data(gta_type, id);
        CREATE INDEX IF NOT EXISTS idx_simdata_sim_gta_ts
            ON simulation_data(simulation_id, gta_type, timestamp);
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialized at {DB_PATH}")


def insert_many(table: str, rows: list):
    if not rows:
        return
    cols = list(rows[0].keys())
    ph   = ", ".join(["?"] * len(cols))
    sql  = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({ph})"
    conn = get_connection()
    conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    conn.commit()
    conn.close()


def query(sql: str, params=None) -> list:
    conn = get_connection()
    cur  = conn.execute(sql, params or [])
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def execute(sql: str, params=None):
    conn = get_connection()
    conn.execute(sql, params or [])
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()