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

# Thresholds for the *live SCADA* reading schema emitted by the simulator
# (pressure_hp ~54.5 bar, steam_hp ~197 t/h, vibration ~0.x–x mm/s per GTA,
#  per-GTA adm_pression / rendement). These mirror the injection scenarios
# (pressure_drop, steam_loss, gta2_vibration, production_drop, overtemperature)
# so a parameter falling below/above its nominal band raises an alert in real time.
_SCADA_THRESHOLDS: list[dict] = [
    # High-pressure steam flow (sum of 3 GTA admission flows), nominal ~197 t/h
    {"var": "steam_hp",      "op": "<", "thr": 150.0, "level": "CRITICAL",
     "msg": "Débit vapeur HP critique (<150 t/h) — fuite ou perte vapeur détectée"},
    {"var": "steam_hp",      "op": "<", "thr": 170.0, "level": "WARNING",
     "msg": "Débit vapeur HP anormalement bas (<170 t/h)"},

    # HP steam pressure (bar), nominal ~54.5 — drops during pressure_drop scenario
    {"var": "pressure_hp",   "op": "<", "thr": 40.0,  "level": "CRITICAL",
     "msg": "Pression vapeur HP critique (<40 bar) — chute de pression détectée"},
    {"var": "pressure_hp",   "op": "<", "thr": 48.0,  "level": "WARNING",
     "msg": "Pression vapeur HP sous le seuil normal (<48 bar)"},

    # Medium-pressure steam pressure (bar), nominal ~8.0
    {"var": "pressure_mp",   "op": "<", "thr": 6.0,   "level": "WARNING",
     "msg": "Pression vapeur MP basse (<6 bar)"},
    {"var": "pressure_mp",   "op": ">", "thr": 11.0,  "level": "WARNING",
     "msg": "Pression vapeur MP élevée (>11 bar)"},

    # Vibration per GTA (mm/s), nominal <0.6 — spikes during gta2_vibration scenario
    {"var": "vibration_gta1","op": ">", "thr": 4.5,   "level": "CRITICAL",
     "msg": "Vibration GTA1 critique (>4.5 mm/s) — défaut roulement probable"},
    {"var": "vibration_gta2","op": ">", "thr": 4.5,   "level": "CRITICAL",
     "msg": "Vibration GTA2 critique (>4.5 mm/s) — défaut roulement probable"},
    {"var": "vibration_gta3","op": ">", "thr": 4.5,   "level": "CRITICAL",
     "msg": "Vibration GTA3 critique (>4.5 mm/s) — défaut roulement probable"},
    {"var": "vibration_gta1","op": ">", "thr": 3.0,   "level": "WARNING",
     "msg": "Vibration GTA1 élevée (>3.0 mm/s)"},
    {"var": "vibration_gta2","op": ">", "thr": 3.0,   "level": "WARNING",
     "msg": "Vibration GTA2 élevée (>3.0 mm/s)"},
    {"var": "vibration_gta3","op": ">", "thr": 3.0,   "level": "WARNING",
     "msg": "Vibration GTA3 élevée (>3.0 mm/s)"},

    # Admission pressure per GTA (bar), nominal ~54.5
    {"var": "adm_pression_gta1", "op": "<", "thr": 48.0, "level": "CRITICAL",
     "msg": "Pression admission GTA1 critique (<48 bar)"},
    {"var": "adm_pression_gta1", "op": "<", "thr": 51.0, "level": "WARNING",
     "msg": "Pression admission GTA1 sous le seuil normal (<51 bar)"},

    # Admission temperature per GTA (°C), nominal ~455 — rises during overtemperature
    {"var": "adm_temp_gta1", "op": ">", "thr": 490.0, "level": "CRITICAL",
     "msg": "Température admission GTA1 critique (>490°C) — surchaufe turbine"},
    {"var": "adm_temp_gta1", "op": ">", "thr": 475.0, "level": "WARNING",
     "msg": "Température admission GTA1 élevée (>475°C)"},

    # Per-GTA efficiency (rendement, %), nominal 35–42
    {"var": "rendement_gta1", "op": "<", "thr": 36.0, "level": "WARNING",
     "msg": "Rendement GTA1 sous 36% — vérifier paramètres d'admission"},
    {"var": "rendement_gta2", "op": "<", "thr": 30.0, "level": "WARNING",
     "msg": "Rendement GTA2 sous 30% — vérifier paramètres d'admission"},
    {"var": "rendement_gta3", "op": "<", "thr": 37.0, "level": "WARNING",
     "msg": "Rendement GTA3 sous 37% — vérifier paramètres d'admission"},
]

_SUGGESTED_ACTIONS: dict[str, str] = {
    "production":        "Check GTA availability and steam supply",
    "rendement_gta1":    "Inspect GTA1 admission nozzles and blade fouling",
    "rendement_gta2":    "Inspect GTA2 admission nozzles and blade fouling",
    "rendement_gta3":    "Inspect GTA3 admission nozzles and blade fouling",
    "pression_adm_gta1": "Check HP steam supply valves for GTA1",
    "temp_adm_gta1":     "Monitor GTA1 turbine thermal stress — adjust load",
    "steam_hp":          "Check HP steam network — valve positions and leaks",
    "pressure_hp":       "Check HP steam network — possible leak or trip",
    "pressure_mp":       "Check MP steam extraction valves",
    "vibration_gta1":    "Inspect GTA1 bearings and balance",
    "vibration_gta2":    "Inspect GTA2 bearings and balance",
    "vibration_gta3":    "Inspect GTA3 bearings and balance",
    "adm_pression_gta1": "Check HP steam supply valves for GTA1",
    "adm_temp_gta1":     "Reduce GTA1 load and monitor turbine thermal stress",
    "bilan_net":         "Increase GTA production or reduce plant consumption",
    "efficiency":        "Run efficiency diagnostic on all active GTAs",
    "anomaly_score":     "Review anomaly dashboard and run PCMCI causal analysis",
}


class AlertService:

    # A (variable, level) pair re-fires at most once per cooldown window —
    # live SCADA readings arrive at 1 Hz, and a breached threshold must raise
    # ONE loud alarm, not one per second.
    ALERT_COOLDOWN_S = 60.0

    def __init__(self):
        self._subscribers: list = []   # callbacks for WebSocket broadcast
        self._last_fired: dict[tuple, float] = {}   # (var, level) → monotonic ts

    # ── Threshold evaluation helper ─────────────────────────────────────────────

    def _check_thresholds(self, reading: dict, thresholds: list[dict]) -> list[dict]:
        """Evaluate a reading against a threshold list. Returns new alerts
        (deduplicated: each (variable, level) fires once per cooldown)."""
        import time as _time
        now = _time.monotonic()
        new_alerts: list[dict] = []
        for rule in thresholds:
            val = reading.get(rule["var"])
            if val is None:
                continue
            triggered = (rule["op"] == ">" and float(val) > float(rule["thr"])) or \
                        (rule["op"] == "<" and float(val) < float(rule["thr"]))
            if triggered:
                key = (rule["var"], rule["level"])
                last = self._last_fired.get(key)
                if last is not None and now - last < self.ALERT_COOLDOWN_S:
                    continue
                self._last_fired[key] = now
                action = _SUGGESTED_ACTIONS.get(rule["var"], "Inspect system")
                alert  = self._create_alert(
                    level=rule["level"],
                    message=rule["msg"],
                    source=rule["var"],
                    reading_value=float(val),
                    suggested_action=action,
                )
                new_alerts.append(alert)
        return new_alerts

    # ── Auto-check (historical / canonical schema) ──────────────────────────────

    def check_reading(self, reading: dict) -> list[dict]:
        """Evaluate a sensor reading (canonical/historical schema) against all thresholds."""
        new_alerts = self._check_thresholds(reading, _THRESHOLDS)
        if new_alerts:
            insert_many("alerts", [_to_db(a) for a in new_alerts])
            for a in new_alerts:
                self._broadcast(a)
        return new_alerts

    # ── Auto-check (live SCADA simulator schema) ────────────────────────────────

    def check_scada_reading(self, reading: dict) -> list[dict]:
        """Evaluate a live SCADA reading (pressure_hp, steam_hp, per-GTA
        vibration/adm_*) against the SCADA-specific threshold set so that a
        parameter falling under/over its nominal band raises an alert in real time.
        Per-GTA fields are flattened from the ``gta`` sub-dict if present."""
        flat = dict(reading)
        # Flatten nested per-GTA data emitted by the SCADA simulator, e.g.
        # reading["gta"]["GTA1"]["vib1"] -> vibration_gta1, adm_pression_gta1, ...
        gta = reading.get("gta") if isinstance(reading.get("gta"), dict) else None
        if gta:
            for gta_id, prefix in (("GTA1", "gta1"), ("GTA2", "gta2"), ("GTA3", "gta3")):
                sub = gta.get(gta_id)
                if isinstance(sub, dict):
                    flat[f"vibration_{prefix}"]   = max(
                        float(sub.get("vib1", 0) or 0), float(sub.get("vib2", 0) or 0))
                    if sub.get("adm_pression") is not None:
                        flat[f"adm_pression_{prefix}"] = float(sub["adm_pression"])
                    if sub.get("adm_temp") is not None:
                        flat[f"adm_temp_{prefix}"] = float(sub["adm_temp"])
                    if sub.get("rendement") is not None:
                        flat[f"rendement_{prefix}"] = float(sub["rendement"])

        new_alerts = self._check_thresholds(flat, _SCADA_THRESHOLDS)
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

    def alert_from_anomaly(self, anomaly: dict) -> dict | None:
        sev_map = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}
        level   = sev_map.get(anomaly.get("severity", "info"), "INFO")
        cause   = anomaly.get("cause", "unknown variable")
        group   = anomaly.get("group", cause)
        # Cooldown: live readings are evaluated at 1 Hz — the same anomaly
        # group must not spam an alert every second.
        import time as _time
        now = _time.monotonic()
        key = ("anomaly:" + str(group), level)
        last = self._last_fired.get(key)
        if last is not None and now - last < self.ALERT_COOLDOWN_S:
            return None
        self._last_fired[key] = now
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