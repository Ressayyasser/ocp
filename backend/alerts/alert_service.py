"""
alert_service.py — Generates, stores, and broadcasts alerts.

Levels : INFO | WARNING | CRITICAL

Example output
──────────────
CRITICAL: GTA2 vibration exceeded 5 mm/s
Failure probability: 83%
Recommended action: Maintenance GTA2
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from database.database import insert_many, query

# Thresholds for automatic alert generation
_THRESHOLDS: list[dict] = [
    # Real data ranges: rendement ~38–44%, pression_adm ~53–57 bar, production ~1200–2000 MWh/day per GTA

    # Daily production (MWh/day total, 3 GTAs, nominal ~1400–1800 MWh)
    {"var": "production",       "op": "<", "thr": 900,   "level": "CRITICAL",
     "msg": "Daily production critically low (<900 MWh) — check GTA availability"},
    {"var": "production",       "op": "<", "thr": 1100,  "level": "WARNING",
     "msg": "Daily production below normal range (<1100 MWh)"},

    # GTA efficiency (rendement, %)
    {"var": "rendement_gta1",   "op": "<", "thr": 36.0,  "level": "WARNING",
     "msg": "GTA1 efficiency below 36% — check admission parameters"},
    {"var": "rendement_gta2",   "op": "<", "thr": 30.0,  "level": "WARNING",
     "msg": "GTA2 efficiency below 30% — check admission parameters"},
    {"var": "rendement_gta3",   "op": "<", "thr": 37.0,  "level": "WARNING",
     "msg": "GTA3 efficiency below 37% — check admission parameters"},

    # Admission pressure (bar, nominal ~54–56)
    {"var": "pression_adm_gta1","op": "<", "thr": 48.0,  "level": "CRITICAL",
     "msg": "GTA1 admission pressure critically low (<48 bar)"},
    {"var": "pression_adm_gta1","op": "<", "thr": 51.0,  "level": "WARNING",
     "msg": "GTA1 admission pressure below normal range"},

    # Admission temperature (°C, nominal ~420–470)
    {"var": "temp_adm_gta1",    "op": ">", "thr": 480.0, "level": "WARNING",
     "msg": "GTA1 admission temperature above 480°C — monitor turbine stress"},
    {"var": "temp_adm_gta1",    "op": "<", "thr": 400.0, "level": "WARNING",
     "msg": "GTA1 admission temperature below 400°C — reduced thermal efficiency"},

    # Steam HP flow (t/h total of 3 GTAs, nominal 450–630)
    {"var": "steam_hp",         "op": "<", "thr": 300.0, "level": "CRITICAL",
     "msg": "Total HP steam flow critically low (<300 t/h) — check steam network"},
    {"var": "steam_hp",         "op": "<", "thr": 380.0, "level": "WARNING",
     "msg": "Total HP steam flow below normal (<380 t/h)"},

    # Bilan net
    {"var": "bilan_net",        "op": "<", "thr": 0,     "level": "WARNING",
     "msg": "Net energy balance is negative — plant is net importer"},

    # Efficiency (fraction, nominal ~0.38–0.43)
    {"var": "efficiency",       "op": "<", "thr": 0.33,  "level": "CRITICAL",
     "msg": "Overall efficiency critically low (<33%) — investigate all GTAs"},
    {"var": "efficiency",       "op": "<", "thr": 0.36,  "level": "WARNING",
     "msg": "Overall efficiency below normal range (<36%)"},

    # Anomaly score from Isolation Forest
    {"var": "anomaly_score",    "op": "<", "thr": -0.5,  "level": "CRITICAL",
     "msg": "Critical anomaly detected by Isolation Forest"},
    {"var": "anomaly_score",    "op": "<", "thr": -0.3,  "level": "WARNING",
     "msg": "Anomaly detected — review sensor readings"},
]

_SUGGESTED_ACTIONS: dict[str, str] = {
    "production":        "Check GTA availability and steam supply",
    "rendement_gta1":    "Inspect GTA1 admission nozzles and blade fouling",
    "rendement_gta2":    "Inspect GTA2 admission nozzles and blade fouling",
    "rendement_gta3":    "Inspect GTA3 admission nozzles and blade fouling",
    "pression_adm_gta1": "Check HP steam supply valves for GTA1",
    "temp_adm_gta1":     "Monitor GTA1 turbine thermal stress — adjust load",
    "steam_hp":          "Check HP steam network — valve positions and leaks",
    "bilan_net":         "Increase GTA production or reduce plant consumption",
    "efficiency":        "Run efficiency diagnostic on all active GTAs",
    "anomaly_score":     "Review anomaly dashboard and run PCMCI causal analysis",
}


class AlertService:

    def __init__(self):
        self._subscribers: list = []   # callbacks for WebSocket broadcast

    # ── Auto-check ────────────────────────────────────────────────────────────

    def check_reading(self, reading: dict) -> list[dict]:
        """Evaluate a sensor reading against all thresholds. Returns new alerts."""
        new_alerts: list[dict] = []
        for rule in _THRESHOLDS:
            val = reading.get(rule["var"])
            if val is None:
                continue
            triggered = (rule["op"] == ">" and val > rule["thr"]) or \
                        (rule["op"] == "<" and val < rule["thr"])
            if triggered:
                action = _SUGGESTED_ACTIONS.get(rule["var"], "Inspect system")
                alert  = self._create_alert(
                    level=rule["level"],
                    message=rule["msg"],
                    source=rule["var"],
                    reading_value=val,
                    suggested_action=action,
                )
                new_alerts.append(alert)

        if new_alerts:
            insert_many("alerts", [_to_db(a) for a in new_alerts])
            for a in new_alerts:
                self._broadcast(a)

        return new_alerts

    # ── Manual alert ──────────────────────────────────────────────────────────

    def create_manual(self, level: str, message: str, source: str = "") -> dict:
        alert = self._create_alert(level=level.upper(), message=message, source=source)
        insert_many("alerts", [_to_db(alert)])
        self._broadcast(alert)
        return alert

    # ── Anomaly-derived alert ─────────────────────────────────────────────────

    def alert_from_anomaly(self, anomaly: dict) -> dict:
        sev_map = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}
        level   = sev_map.get(anomaly.get("severity", "info"), "INFO")
        cause   = anomaly.get("cause", "unknown variable")
        score   = anomaly.get("score", 0)
        msg     = (f"Anomaly detected in '{cause}' "
                   f"(score={score:.3f}) — {anomaly.get('label', '')}")
        return self.create_manual(level, msg, source=cause)

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 50, level: str | None = None) -> list[dict]:
        if level:
            return query("SELECT * FROM alerts WHERE level=? ORDER BY timestamp DESC LIMIT ?",
                         [level.upper(), limit])
        return query("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", [limit])

    def acknowledge(self, alert_id: int):
        from database.database import execute
        execute("UPDATE alerts SET acknowledged=1 WHERE id=?", [alert_id])

    # ── WebSocket broadcast ───────────────────────────────────────────────────

    def subscribe(self, callback):
        """Register a callback for real-time alert broadcast."""
        self._subscribers.append(callback)

    def _broadcast(self, alert: dict):
        for cb in self._subscribers:
            try:
                cb(alert)
            except Exception:
                pass

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _create_alert(level: str, message: str, source: str = "",
                      reading_value: float | None = None,
                      suggested_action: str = "") -> dict:
        return {
            "timestamp":        datetime.now().isoformat(),
            "level":            level,
            "message":          message,
            "source":           source,
            "reading_value":    reading_value,
            "suggested_action": suggested_action,
        }


def _to_db(a: dict) -> dict:
    return {
        "timestamp": a["timestamp"],
        "message":   a["message"],
        "level":     a["level"],
        "source":    a.get("source", ""),
    }