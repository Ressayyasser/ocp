"""
utils/virtual_gta.py — Independent Virtual GTA twin model.

A self-contained simulation sandbox: it owns its OWN nominal state (no SCADA,
no live feed, no backend) and applies fault / operating scenarios on it.
Effects are propagated through a physics-inspired sensitivity model so the
impact can be shown per GTA component (Source HP, Turbine, Alternateur,
Soutirage MP, Condenseur, Refroidissement, Réseau MT), together with
rule-based anomaly detection and DCS-style alarms.

The record keys follow the telemetry schema of pages/gta_visualization.py
(_parse_record) so the same SVG synoptic can render the virtual state.
"""

from __future__ import annotations
import copy

# ─────────────────────────────────────────────────────────────────────────────
#  NOMINAL BASELINE (healthy GTA at ~25 MW — values from the DCS synoptic)
# ─────────────────────────────────────────────────────────────────────────────
# NB: gta_visualization._parse_record divides puissance_mw by 24
# (daily-energy → MW), so the stored value is MW × 24.
BASELINE: dict[str, float] = {
    "adm_pression":  54.8,    # bar   — HP admission pressure
    "adm_temp":      460.1,   # °C    — HP admission temperature
    "adm_debit":     195.3,   # t/h   — HP admission flow
    "vap_inlet":     55.4,    # bar   — turbine inlet pressure
    "sout_debit":    47.8,    # t/h   — MP extraction flow
    "sout_pression": 8.9,     # bar   — MP extraction pressure
    "ext_debit":     90.0,    # t/h   — exhaust flow
    "ext_pression":  0.09,    # bar   — condenser vacuum
    "puissance_mw":  600.0,   # MWh/day (=> 25.0 MW after /24)
    "rendement":     88.0,    # %     — isentropic efficiency
    "bp_pression":   0.9,     # bar
    "bp_debit":      8.7,     # t/h
    "vib1":          0.20,    # mm/s  — bearing 1 vibration
    "vib2":          0.40,    # mm/s  — bearing 2 vibration
    "dd3":           0.61,    # mm
    "oil_pression":  1.52,    # bar
    "oil_temp":      40.4,    # °C
    "cos_phi":       0.855,
    "p_active":      25.0,    # MW
    "p_reactive":    12.9,    # Mvar
    "tension":       10.5,    # kV
    "vitesse":       3000,    # RPM
    "posit_hp":      82.0,    # % servo-valve
    "posit_bp":      64.0,    # %
    "cond_temp":     45.2,    # °C  — condensate temperature
    "cond_eau":      87.0,    # t/h — cooling water flow
    "level_pct":     78.1,    # %   — condenser hotwell level
}

# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO CATALOGUE
#  Each effect: var → (delta_at_full_intensity, mode)   mode: "mul" | "add"
#  "mul": value × (1 + delta·i)     "add": value + delta·i
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS: dict[str, dict] = {
    "pressure_drop": {
        "label": "💨 Chute de pression vapeur HP",
        "description": "Perte de pression sur le collecteur HP (fuite réseau / "
                       "déclenchement chaudière). La détente disponible chute, la "
                       "turbine perd de la puissance et du rendement.",
        "effects": {
            "adm_pression": (-0.25, "mul"), "vap_inlet": (-0.25, "mul"),
            "adm_debit":    (-0.10, "mul"), "puissance_mw": (-0.30, "mul"),
            "p_active":     (-0.30, "mul"), "rendement": (-9.0, "add"),
            "ext_pression": (+0.45, "mul"), "sout_pression": (-0.12, "mul"),
            "posit_hp":     (+0.15, "mul"),
        },
    },
    "overtemperature": {
        "label": "🌡️ Surchauffe admission",
        "description": "Dérive de la température de vapeur vive (désurchauffeur "
                       "défaillant). Contraintes thermiques sur le corps HP, "
                       "dilatations et vibrations en hausse.",
        "effects": {
            "adm_temp": (+42.0, "add"), "rendement": (-4.0, "add"),
            "vib2": (+0.35, "add"), "vib1": (+0.15, "add"),
            "oil_temp": (+12.0, "add"), "dd3": (+0.25, "add"),
        },
    },
    "vibration_spike": {
        "label": "📳 Pic de vibration paliers",
        "description": "Balourd / défaut de palier n°2 (usure roulement, "
                       "désalignement). Risque mécanique direct sur la ligne "
                       "d'arbre turbine–alternateur.",
        "effects": {
            "vib2": (+0.65, "add"), "vib1": (+0.30, "add"),
            "rendement": (-2.0, "add"), "dd3": (+0.35, "add"),
            "oil_temp": (+6.0, "add"),
        },
    },
    "steam_loss": {
        "label": "♨️ Perte vapeur / encrassement",
        "description": "Encrassement des aubages ou fuite vapeur interne : le "
                       "débit utile baisse, la puissance et le rendement chutent.",
        "effects": {
            "adm_debit": (-0.20, "mul"), "sout_debit": (-0.15, "mul"),
            "bp_debit": (-0.15, "mul"), "puissance_mw": (-0.22, "mul"),
            "p_active": (-0.22, "mul"), "rendement": (-6.5, "add"),
            "posit_hp": (+0.10, "mul"),
        },
    },
    "condenser_fouling": {
        "label": "🧊 Dégradation vide condenseur",
        "description": "Encrassement des tubes / entrée d'air : le vide se dégrade, "
                       "la contre-pression augmente et le cycle perd du rendement.",
        "effects": {
            "ext_pression": (+1.20, "mul"), "cond_temp": (+14.0, "add"),
            "rendement": (-5.0, "add"), "puissance_mw": (-0.08, "mul"),
            "p_active": (-0.08, "mul"), "level_pct": (+10.0, "add"),
            "cond_eau": (-0.10, "mul"),
        },
    },
    "load_increase": {
        "label": "⚡ Augmentation de charge",
        "description": "Montée en charge du groupe (+débit vapeur admis). Gain de "
                       "production mais sollicitation mécanique et thermique accrue.",
        "effects": {
            "adm_debit": (+0.30, "mul"), "puissance_mw": (+0.28, "mul"),
            "p_active": (+0.28, "mul"), "sout_debit": (+0.10, "mul"),
            "adm_temp": (+8.0, "add"), "vib2": (+0.28, "add"),
            "ext_pression": (+0.20, "mul"), "cond_eau": (+0.08, "mul"),
            "posit_hp": (+0.18, "mul"),
        },
    },
    "load_decrease": {
        "label": "🔻 Baisse de charge",
        "description": "Réduction de la charge du groupe (moins de vapeur admise). "
                       "Production en baisse, la machine se détend.",
        "effects": {
            "adm_debit": (-0.25, "mul"), "puissance_mw": (-0.25, "mul"),
            "p_active": (-0.25, "mul"), "sout_debit": (-0.12, "mul"),
            "vib2": (-0.10, "add"), "adm_temp": (-6.0, "add"),
            "ext_pression": (-0.10, "mul"),
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  COMPONENT (ZONE) → VARIABLES MAP  (keys match gta_visualization.ZONES)
# ─────────────────────────────────────────────────────────────────────────────
ZONE_VARS: dict[str, list[str]] = {
    "source_hp":       ["adm_pression", "adm_temp", "adm_debit"],
    "turbine":         ["rendement", "vib1", "vib2", "vap_inlet", "posit_hp",
                        "oil_temp", "dd3", "vitesse"],
    "alternateur":     ["p_active", "cos_phi", "tension"],
    "reseau_mt":       ["p_active"],
    "source_bp":       ["bp_pression", "bp_debit"],
    "soutirage_mp":    ["sout_debit", "sout_pression"],
    "condenseur":      ["ext_pression", "cond_temp", "level_pct"],
    "refroidissement": ["cond_eau", "cond_temp"],
}

VAR_META: dict[str, tuple[str, str, int]] = {
    # var → (label, unit, decimals)
    "adm_pression":  ("Pression admission", "bar", 1),
    "adm_temp":      ("Température admission", "°C", 1),
    "adm_debit":     ("Débit admission", "t/h", 1),
    "vap_inlet":     ("Pression entrée turbine", "bar", 1),
    "sout_debit":    ("Débit soutirage MP", "t/h", 1),
    "sout_pression": ("Pression soutirage MP", "bar", 2),
    "ext_pression":  ("Vide condenseur", "bar", 3),
    "puissance_mw":  ("Énergie journalière", "MWh", 0),
    "rendement":     ("Rendement", "%", 1),
    "bp_pression":   ("Pression BP", "bar", 2),
    "bp_debit":      ("Débit BP", "t/h", 1),
    "vib1":          ("Vibration palier 1", "mm/s", 2),
    "vib2":          ("Vibration palier 2", "mm/s", 2),
    "dd3":           ("Dilatation DD3", "mm", 2),
    "oil_temp":      ("Température huile", "°C", 1),
    "oil_pression":  ("Pression huile", "bar", 2),
    "cos_phi":       ("Cos φ", "", 3),
    "p_active":      ("Puissance active", "MW", 1),
    "tension":       ("Tension", "kV", 1),
    "vitesse":       ("Vitesse", "RPM", 0),
    "posit_hp":      ("Servo-valve HP", "%", 1),
    "cond_temp":     ("Température condensat", "°C", 1),
    "cond_eau":      ("Débit eau refroid.", "t/h", 1),
    "level_pct":     ("Niveau condenseur", "%", 1),
}

# ─────────────────────────────────────────────────────────────────────────────
#  ANOMALY RULES  (independent, threshold-based — mirrors backend AlertService)
#  (var, op, threshold, level, message, zone)
# ─────────────────────────────────────────────────────────────────────────────
ANOMALY_RULES: list[tuple] = [
    ("adm_pression", "<", 44.0,  "CRITICAL",
     "Pression admission HP critique (<44 bar) — risque de déclenchement", "source_hp"),
    ("adm_pression", "<", 50.0,  "WARNING",
     "Pression admission HP sous la plage normale (<50 bar)", "source_hp"),
    ("adm_temp", ">", 490.0, "CRITICAL",
     "Température admission critique (>490 °C) — surchauffe turbine", "source_hp"),
    ("adm_temp", ">", 475.0, "WARNING",
     "Température admission élevée (>475 °C)", "source_hp"),
    ("adm_temp", "<", 400.0, "WARNING",
     "Température admission basse (<400 °C) — rendement dégradé", "source_hp"),
    ("adm_debit", "<", 150.0, "CRITICAL",
     "Débit vapeur HP critique (<150 t/h) — perte vapeur majeure", "source_hp"),
    ("adm_debit", "<", 170.0, "WARNING",
     "Débit vapeur HP anormalement bas (<170 t/h)", "source_hp"),
    ("vib2", ">", 0.80, "CRITICAL",
     "Vibration palier n°2 critique (>0.80 mm/s) — défaut roulement probable", "turbine"),
    ("vib2", ">", 0.60, "WARNING",
     "Vibration palier n°2 élevée (>0.60 mm/s)", "turbine"),
    ("rendement", "<", 60.0, "CRITICAL",
     "Rendement isentropique critique (<60 %) — dégradation majeure", "turbine"),
    ("rendement", "<", 78.0, "WARNING",
     "Rendement isentropique dégradé (<78 %)", "turbine"),
    ("oil_temp", ">", 55.0, "WARNING",
     "Température huile de graissage élevée (>55 °C)", "turbine"),
    ("ext_pression", ">", 0.120, "CRITICAL",
     "Perte de vide condenseur (>0.120 bar) — LOW VACUUM", "condenseur"),
    ("ext_pression", ">", 0.100, "WARNING",
     "Vide condenseur dégradé (>0.100 bar)", "condenseur"),
    ("level_pct", ">", 92.0, "WARNING",
     "Niveau condenseur haut (>92 %)", "condenseur"),
    ("p_active", ">", 34.0, "CRITICAL",
     "Surcharge alternateur (>34 MW) — limite 37 MW", "alternateur"),
    ("p_active", ">", 30.0, "WARNING",
     "Charge alternateur élevée (>30 MW)", "alternateur"),
    ("sout_debit", "<", 38.0, "WARNING",
     "Débit de soutirage MP bas (<38 t/h) — procédés aval impactés", "soutirage_mp"),
    ("cond_eau", "<", 75.0, "WARNING",
     "Débit d'eau de refroidissement bas (<75 t/h)", "refroidissement"),
]

# DCS alarm lamps shown on the panel (name → predicate on the record)
ALARM_LAMPS: list[tuple] = [
    ("TRIP",       lambda d: d["rendement"] < 60 or d["p_active"] > 35),
    ("VIBRATION",  lambda d: d["vib2"] > 0.80),
    ("LOW VACUUM", lambda d: d["ext_pression"] > 0.120),
    ("TEMP. ADM.", lambda d: d["adm_temp"] > 475),
    ("SURCHARGE",  lambda d: d["p_active"] > 30),
]

_PHYSICAL_MIN: dict[str, float] = {"rendement": 0.0, "level_pct": 0.0,
                                   "vib1": 0.0, "vib2": 0.0, "ext_pression": 0.005}


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL API
# ─────────────────────────────────────────────────────────────────────────────

def apply_scenario(scenario_key: str, intensity_pct: float,
                   base: dict | None = None) -> dict:
    """Return the virtual record after applying *scenario_key* at
    *intensity_pct* (0–100 %) on the baseline (or a given state)."""
    record = copy.deepcopy(base or BASELINE)
    scenario = SCENARIOS.get(scenario_key)
    if not scenario:
        return record
    i = max(0.0, min(1.0, (intensity_pct or 0) / 100.0))
    for var, (delta, mode) in scenario["effects"].items():
        if var not in record:
            continue
        if mode == "mul":
            record[var] = record[var] * (1.0 + delta * i)
        else:
            record[var] = record[var] + delta * i
        if var in _PHYSICAL_MIN:
            record[var] = max(_PHYSICAL_MIN[var], record[var])
    # level_pct capped at 100 %
    record["level_pct"] = min(100.0, record["level_pct"])
    return record


def detect_anomalies(record: dict) -> list[dict]:
    """Threshold-based anomaly detection on the virtual state.
    Only the most severe rule per variable is kept (rules are ordered
    most-severe-first per variable)."""
    seen: set[str] = set()
    out: list[dict] = []
    for var, op, thr, level, msg, zone in ANOMALY_RULES:
        if var in seen:
            continue
        val = record.get(var)
        if val is None:
            continue
        hit = val < thr if op == "<" else val > thr
        if hit:
            seen.add(var)
            out.append({"var": var, "level": level, "message": msg,
                        "zone": zone, "value": round(float(val), 3),
                        "threshold": thr})
    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    out.sort(key=lambda a: order.get(a["level"], 3))
    return out


def get_alarms(record: dict) -> list[dict]:
    """DCS alarm lamp states."""
    return [{"name": name, "active": bool(pred(record))}
            for name, pred in ALARM_LAMPS]


def compute_zone_impacts(base: dict, new: dict,
                         anomalies: list[dict] | None = None) -> dict:
    """Per-component impact: changed variables + a severity classification.

    Severity: 'critical' ≥15 % relative change (or CRITICAL anomaly in zone),
              'warning'  ≥5 %  (or WARNING anomaly), 'ok' otherwise.
    """
    anomalies = anomalies or []
    zone_anom_level: dict[str, str] = {}
    for a in anomalies:
        cur = zone_anom_level.get(a["zone"])
        if a["level"] == "CRITICAL" or cur is None:
            zone_anom_level[a["zone"]] = a["level"]

    impacts: dict[str, dict] = {}
    for zone, variables in ZONE_VARS.items():
        changes = []
        max_pct = 0.0
        for var in variables:
            b, n = base.get(var), new.get(var)
            if b is None or n is None:
                continue
            pct = 0.0 if b == 0 else (n - b) / abs(b) * 100.0
            if abs(pct) >= 0.5:
                label, unit, nd = VAR_META.get(var, (var, "", 2))
                changes.append({
                    "var": var, "label": label, "unit": unit,
                    "base": round(float(b), nd), "new": round(float(n), nd),
                    "pct": round(pct, 1),
                })
                max_pct = max(max_pct, abs(pct))
        severity = "ok"
        if max_pct >= 15:
            severity = "critical"
        elif max_pct >= 5:
            severity = "warning"
        anom = zone_anom_level.get(zone)
        if anom == "CRITICAL":
            severity = "critical"
        elif anom == "WARNING" and severity == "ok":
            severity = "warning"
        changes.sort(key=lambda c: abs(c["pct"]), reverse=True)
        impacts[zone] = {"severity": severity, "changes": changes,
                         "max_pct": round(max_pct, 1)}
    return impacts


def global_status(anomalies: list[dict]) -> str:
    """Overall virtual-GTA status: NORMAL | DÉGRADÉ | CRITIQUE."""
    if any(a["level"] == "CRITICAL" for a in anomalies):
        return "CRITIQUE"
    if any(a["level"] == "WARNING" for a in anomalies):
        return "DÉGRADÉ"
    return "NORMAL"
