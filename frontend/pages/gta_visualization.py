"""

===========
Drop-in replacement for the original GTA Dash module.
Requires: dash, dash-svg (pip install dash dash-svg)

Usage (unchanged from original):
    from gta_view import layout_gta, register_gta_callbacks
"""

from pathlib import Path
import sys, os
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
import math
import dash
from dash import dcc, html, Input, Output
from utils.styles import COLORS, card_style
from utils.api_client import get
import requests

dash.register_page(__name__, path="/gta", name="GTA", title="GTA")

_SVG_STYLE_ATTRS = frozenset([
    'filter', 'clip-path', 'offset',
    'marker-end', 'marker-start', 'marker-mid',
    'flood-color', 'flood-opacity',
    'color-interpolation-filters',
])

def _svg_kwargs(kw: dict) -> dict:
    """
    Déplace les attributs SVG non supportés par dash_svg v0.0.12
    depuis les kwargs vers style={...}.
    """
    kw = dict(kw)  # copie défensive
    style = dict(kw.pop('style', {}) or {})

    for attr in list(kw):
        if attr in _SVG_STYLE_ATTRS:
            style[attr] = kw.pop(attr)

    if style:
        kw['style'] = style
    return kw
def _stop(offset, stop_color, stop_opacity="1"):
    return svg.Stop(style={
        "offset":       offset,
        "stopColor":    stop_color,
        "stopOpacity":  stop_opacity,
    })
# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_GTA_COLORS = {"GTA1": "#1f6feb", "GTA2": "#f0883e", "GTA3": "#3fb950"}
_BG         = "#0a0e17"
_CARD       = "#0d1117"
_BORDER     = "#21262d"
_MUTED      = "#8b949e"
_TEXT       = "#e6edf3"
_ORANGE     = "#f0883e"
_YELLOW     = "#d29922"
_CYAN       = "#39c5cf"
_GREEN      = "#3fb950"
_RED        = "#f85149"
_GRAY       = "#30363d"


def _rend_color(r):
    if r > 85:  return _GREEN
    if r > 75:  return _ORANGE
    return _RED


# ─────────────────────────────────────────────────────────────────────────────
#  SVG PRIMITIVE HELPERS  (pure Python / Dash-SVG)
# ─────────────────────────────────────────────────────────────────────────────
import dash_svg as svg
# ─── PATCH dash_svg v0.0.12 bug (missing 'offset' prop) ─────────────────────


def _circle(cx, cy, r, **kw):
    return svg.Circle(cx=str(cx), cy=str(cy), r=str(r), **_svg_kwargs(kw))

def _rect(x, y, w, h, **kw):
    return svg.Rect(x=str(x), y=str(y), width=str(w), height=str(h), **_svg_kwargs(kw))

def _text(x, y, content, **kw):
    return svg.Text(content, x=str(x), y=str(y), **_svg_kwargs(kw))

def _line(x1, y1, x2, y2, **kw):
    return svg.Line(x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), **_svg_kwargs(kw))

def _path(d, **kw):
    return svg.Path(d=d, **_svg_kwargs(kw))

def _g(**kw):
    return svg.G(**_svg_kwargs(kw))


def _poly(points, **kw):
    return svg.Polygon(points=points, **kw)


# ─────────────────────────────────────────────────────────────────────────────
#  REUSABLE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────


def _pipe(x1, y1, x2, y2, color="#39c5cf", width=8, dashed=False, animated=True):
    dash_arr = "10,6" if dashed else None
    line = _line(x1, y1, x2, y2,
                 stroke=color, strokeWidth=str(width),
                 strokeDasharray=dash_arr, strokeLinecap="round", opacity="0.9")
    children = [line]
    if animated:
        # Very simplified: animate cx along the line
        cx = x1
        cy = y1
        circle = _circle(cx, cy, 4,
                         fill=color, filter="url(#glow)",
                         children=[
                             svg.Animate(
                                 attributeName="cx",
                                 values=f"{x1};{x2}",
                                 dur="1.4s",
                                 repeatCount="indefinite"
                             ),
                             svg.Animate(
                                 attributeName="cy",
                                 values=f"{y1};{y2}",
                                 dur="1.4s",
                                 repeatCount="indefinite"
                             )
                         ])
        children.append(circle)
    return svg.G(children=children)

def _dashed_flow(x1, y1, x2, y2, color="#39c5cf", width=3):
    lne = _line(x1, y1, x2, y2,
                stroke=color, strokeWidth=str(width),
                strokeDasharray="8,5", strokeLinecap="round", opacity="0.75",
                children=[
                    svg.Animate(
                        attributeName="stroke-dashoffset",
                        values="0;-30",
                        dur="1s",
                        repeatCount="indefinite",
                    )
                ])
    dot = _circle(0, 0, 3,
                  fill=color, opacity="0.9",
                  children=[
                      svg.Animate(
                          attributeName="cx",
                          values=f"{x1};{x2}",
                          dur="1.8s",
                          repeatCount="indefinite",
                      ),
                      svg.Animate(
                          attributeName="cy",
                          values=f"{y1};{y2}",
                          dur="1.8s",
                          repeatCount="indefinite",
                      ),
                  ])
    return svg.G(children=[lne, dot])


def _gauge(cx, cy, tag, val, color):
    """Circular instrument (PT / FT / TT …) with readout below."""
    return svg.G(children=[
        _circle(cx, cy, 16,
                fill=_CARD, stroke=color, strokeWidth="2", filter="url(#glow)"),
        _text(cx, cy - 3, tag,
              fill=color, fontSize="8", fontWeight="bold"),
        _rect(cx - 25, cy + 19, 50, 16, rx="3",
              fill="rgba(13,17,23,0.9)", stroke=_GRAY),
        _text(cx, cy + 27, str(val),
              fill=_TEXT, fontSize="8"),
    ])


def _alarm_dot(x, y, label, active):
    color = _RED if active else _GREEN
    children = [_circle(x, y, 7, fill=color, filter="url(#glow)")]
    if active:
        children[0] = _circle(x, y, 7, fill=color, filter="url(#glow)",
                               children=[
                                   svg.Animate(attributeName="opacity",
                                               values="1;0.3;1", dur="0.8s",
                                               repeatCount="indefinite")
                               ])
    children.append(_text(x + 22, y, label,
                           fill=_TEXT, fontSize="9", textAnchor="start"))
    return svg.G(children=children)


def _metric(x, y, label, value, color):
    return svg.G(children=[
        _text(x, y,       label, fill=_MUTED, fontSize="8", textAnchor="start"),
        _text(x, y + 13,  value, fill=color,  fontSize="11",
              fontWeight="bold", textAnchor="start"),
    ])


def _sys_row(x, y, label, value, color):
    return svg.G(children=[
        _text(x,       y, f"{label}:", fill=_MUTED, fontSize="9", textAnchor="start"),
        _text(x + 120, y, value,       fill=color,  fontSize="10",
              fontWeight="bold", textAnchor="end"),
        _line(x, y + 4, x + 122, y + 4, stroke="#21262d", strokeWidth="1"),
    ])


def _turbine_stage(x, y, label, color, pression, debit):
    cx, cy = x + 45, y + 55
    blades = [
        _line(cx, cy,
              cx + 24 * math.cos(i * math.pi / 3),
              cy + 24 * math.sin(i * math.pi / 3),
              stroke=color, strokeWidth="2", strokeLinecap="round")
        for i in range(6)
    ]
    wheel = svg.G(
        children=[
            _circle(cx, cy, 30, fill="none", stroke=color, strokeWidth="1.8"),
            _circle(cx, cy, 11, fill=color, fillOpacity="0.28"),
            *blades,
            # Use Animate to drive the transform attribute
            svg.Animate(
                attributeName="transform",
                values=f"rotate(0,{cx},{cy});rotate(360,{cx},{cy})",
                dur="3s",
                repeatCount="indefinite"
            ),
        ]
    )
    return svg.G(children=[
        _rect(x, y, 90, 110, rx="6",
              fill=_CARD, stroke=color, strokeWidth="1.5"),
        _text(cx, y + 16, label,
              fill=color, fontSize="12", fontWeight="bold"),
        wheel,
        _text(cx, y + 93,  f"{pression}b",   fill=_YELLOW, fontSize="9"),
        _text(cx, y + 106, f"{debit} t/h",   fill=_MUTED,  fontSize="8"),
    ])



# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SVG BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_gta_svg(data: dict, gta_name: str) -> html.Div:
    """
    Builds the full interactive SCADA-style SVG for one GTA unit.
    All real values come from `data['data'][-1]`.
    """
    records = data.get("data", [])
    if records:
        last          = records[-1]
        adm_debit     = round(float(last.get("adm_debit")     or 0), 1)
        adm_temp      = round(float(last.get("adm_temp")      or 0), 1)
        adm_pression  = round(float(last.get("adm_pression")  or 0), 2)
        sout_debit    = round(float(last.get("sout_debit")    or 0), 1)
        sout_pression = round(float(last.get("sout_pression") or 0), 2)
        ext_debit     = round(float(last.get("ext_debit")     or 0), 1)
        ext_pression  = round(float(last.get("ext_pression")  or 0), 4)
        puissance_mw  = round(float(last.get("puissance_mw")  or 0), 2)
        rendement     = round(float(last.get("rendement")     or 0), 2)
        bp_pression   = round(float(last.get("bp_pression")   or 0.9), 2)
        bp_debit      = round(float(last.get("bp_debit")      or 8.7), 1)
        vib1          = round(float(last.get("vib1")          or 0.2), 2)
        vib2          = round(float(last.get("vib2")          or 0.4), 2)
        dd3           = round(float(last.get("dd3")           or 0.61), 2)
        oil_pression  = round(float(last.get("oil_pression")  or 1.52), 2)
        oil_temp      = round(float(last.get("oil_temp")      or 40.4), 1)
        cos_phi       = round(float(last.get("cos_phi")       or 0.855), 3)
        p_active      = round(float(last.get("p_active")      or 21.3), 1)
        p_reactive    = round(float(last.get("p_reactive")    or 12.9), 1)
        tension       = round(float(last.get("tension")       or 10.5), 1)
        vitesse       = int(float(last.get("vitesse")         or 6398))
        posit_hp      = round(float(last.get("posit_hp")      or -1.0), 1)
        posit_bp      = round(float(last.get("posit_bp")      or -0.9), 1)
        vap_inlet     = round(float(last.get("vap_inlet")     or 55.4), 1)
        cond_temp     = round(float(last.get("cond_temp")     or 245), 1)
        cond_eau      = round(float(last.get("cond_eau")      or 87), 1)
        level_pct     = round(float(last.get("level_pct")     or 78.1), 1)
    else:
        adm_debit = adm_temp = adm_pression = sout_debit = sout_pression = 0
        ext_debit = ext_pression = puissance_mw = rendement = 0
        bp_pression = bp_debit = vib1 = vib2 = dd3 = 0
        oil_pression = oil_temp = cos_phi = 0
        p_active = p_reactive = tension = 0
        vitesse = 0; posit_hp = posit_bp = vap_inlet = 0
        cond_temp = cond_eau = level_pct = 0

    gta_col   = _GTA_COLORS.get(gta_name, "#1f6feb")
    rend_col  = _rend_color(rendement)
    puissance_mw  = round(puissance_mw / 24, 2) if puissance_mw > 0 else 0
    statut    = "NORMAL" if rendement > 85 else "DÉGRADÉ"
    stat_col  = _GREEN   if rendement > 85 else _RED

    # ── DEFS ──────────────────────────────────────────────────────────────────
    defs = svg.Defs(children=[
        svg.Filter(id="glow", children=[
            svg.FeGaussianBlur(stdDeviation="3", result="blur"),
            svg.FeMerge(children=[
                svg.FeMergeNode(**{"in": "blur"}),
                svg.FeMergeNode(**{"in": "SourceGraphic"}),
            ]),
        ]),
        svg.LinearGradient(
            id="tgrad", x1="0%", y1="0%", x2="100%", y2="100%",
            children=[
                _stop("0%",   "#1a3a5c"),
                _stop("100%", "#0a1628"),
            ]
        ),
        svg.Marker(id="arr",  markerWidth="10", markerHeight="7",
                   refX="9",  refY="3.5", orient="auto",
                   children=[_poly("0 0,10 3.5,0 7", fill=_ORANGE)]),
        svg.Marker(id="arrc", markerWidth="10", markerHeight="7",
                   refX="9",  refY="3.5", orient="auto",
                   children=[_poly("0 0,10 3.5,0 7", fill=_CYAN)]),
        svg.Marker(id="arrg", markerWidth="10", markerHeight="7",
                   refX="9",  refY="3.5", orient="auto",
                   children=[_poly("0 0,10 3.5,0 7", fill=gta_col)]),
    ])

    # ── BACKGROUND ────────────────────────────────────────────────────────────
    bg_grid = []
    for x in range(0, 1361, 80):
        bg_grid.append(_line(x, 0, x, 700, stroke="#161c28", strokeWidth="1"))
    for y in range(0, 701, 60):
        bg_grid.append(_line(0, y, 1360, y, stroke="#161c28", strokeWidth="1"))

    # ── TITLE BAR ─────────────────────────────────────────────────────────────
    title_bar = [
        _rect(0, 0, 1360, 44, fill="rgba(13,17,23,0.9)", stroke=_BORDER, strokeWidth="1"),
        _text(680, 22, gta_name,
              fill=gta_col, fontSize="20", fontWeight="bold", filter="url(#glow)"),
        _text(680, 36, "DIGITAL TWIN — OCP DPSC Jorf Lasfar",
              fill=_MUTED, fontSize="8.5"),
    ]

    # ── PRODUCTION / RENDEMENT BADGES ─────────────────────────────────────────
    badges = [
        _rect(14, 54, 200, 50, rx="7", fill=_CARD, stroke=gta_col, strokeWidth="1.5"),
        _text(114, 68,  "PRODUCTION",       fill=_MUTED,  fontSize="9"),
        _text(114, 88,  f"{puissance_mw} MWh",   fill=gta_col, fontSize="18",
              fontWeight="bold", filter="url(#glow)"),
        _rect(228, 54, 160, 50, rx="7", fill=_CARD, stroke=rend_col, strokeWidth="1.5"),
        _text(308, 68, "RENDEMENT",         fill=_MUTED,  fontSize="9"),
        _text(308, 88, f"{rendement} %",    fill=rend_col, fontSize="18",
              fontWeight="bold", filter="url(#glow)"),
    ]

    # ── DILATATION THERMIQUE ───────────────────────────────────────────────────
    dilatation = [
        _rect(510, 58, 160, 56, rx="6",
              fill="rgba(13,17,23,0.7)", stroke=_GRAY, strokeWidth="1.2"),
        _text(590, 73, "DILATATION THERMIQUE", fill=_MUTED, fontSize="8"),
        _text(524, 90, "Déplac. Axial",  fill=_MUTED,  fontSize="8", textAnchor="start"),
        _text(524, 103,"+0.12 mm",       fill=_YELLOW, fontSize="9",
              fontWeight="bold", textAnchor="start"),
        _text(622, 90, "Corps",          fill=_MUTED,  fontSize="8", textAnchor="start"),
        _text(622, 103,"4.5 mm",         fill=_YELLOW, fontSize="9",
              fontWeight="bold", textAnchor="start"),
    ]

    # ── CONSIGNE HP (top) + PT gauge ──────────────────────────────────────────
    consigne = [
        _rect(480, 50, 130, 36, rx="5", fill="#161b22", stroke=_GRAY),
        _text(545, 63, "Consigne HP",           fill=_MUTED,  fontSize="8"),
        _text(545, 76, f"{vap_inlet} bar",      fill=_YELLOW, fontSize="10",
              fontWeight="bold"),
        _gauge(590, 108, "PT", f"{adm_pression}b", _YELLOW),
        _dashed_flow(590, 155, 590, 188, gta_col, 2.5),
    ]

    # ── V1 / V2 / V3 EQUILIBRAGE CIRCLES ─────────────────────────────────────
    valve_data = [(427, 238, "V1"), (390, 288, "V2"), (390, 352, "V3")]
    valves = []
    for vx, vy, vl in valve_data:
        valves += [
            _circle(vx, vy, 18, fill=_CARD, stroke=gta_col, strokeWidth="2"),
            _text(vx, vy - 5, vl,    fill=gta_col, fontSize="9", fontWeight="bold"),
            _text(vx, vy + 6, "100%", fill=_MUTED,  fontSize="7"),
        ]
    valves += [
        _dashed_flow(340, 238, 427, 238, gta_col, 1.5),
        _dashed_flow(340, 288, 390, 288, gta_col, 1.5),
        _dashed_flow(340, 350, 390, 350, gta_col, 1.5),
        _text(425, 216, "Equilibrage HP",  fill=_MUTED, fontSize="8"),
        _text(382, 266, "Cible:100%",      fill=_MUTED, fontSize="7"),
        _text(382, 330, "Cible:100%",      fill=_MUTED, fontSize="7"),
    ]

    # ── SOURCE HP (left box) ──────────────────────────────────────────────────
    src_hp = svg.G(children=[
        _rect(14, 168, 155, 175, rx="8", fill="none",
              stroke=_ORANGE, strokeWidth="2"),
        _text(91, 190, "SOURCE HP",             fill=_ORANGE, fontSize="11", fontWeight="bold"),
        _text(91, 206, "Vapeur Haute Pression",  fill=_MUTED,  fontSize="7.5"),
        _text(50, 248, "🔥",                     fill=_ORANGE, fontSize="18",
              textAnchor="start"),
        _text(91, 270, "Pression",               fill=_MUTED,  fontSize="8"),
        _text(91, 284, f"{adm_pression} bar",    fill=_YELLOW, fontSize="12",
              fontWeight="bold"),
        _text(91, 304, "Température",            fill=_MUTED,  fontSize="8"),
        _text(91, 318, f"{adm_temp} °C",         fill=_ORANGE, fontSize="12",
              fontWeight="bold"),
        _text(91, 176, f"Débit HP: {adm_debit} t/h",
              fill=_TEXT, fontSize="9", fontWeight="bold"),
    ])

    # ── HP MAIN PIPE ──────────────────────────────────────────────────────────
    hp_pipe = [
        _pipe(169, 258, 338, 258, _ORANGE, 10),
        _gauge(255, 238, "FT", str(adm_debit), gta_col),
    ]

    # ── SERVOMOTEURS (top of turbine) ─────────────────────────────────────────
    serv = []
    for sx, sl, sv in [(380, "SERV. HP", f"POSIT_HP: {posit_hp}%"),
                       (710, "SERV. BP", f"POSIT_BP: {posit_bp}%")]:
        serv += [
            _rect(sx, 58, 115, 36, rx="5", fill="#161b22", stroke=_GRAY),
            _text(sx + 57, 71, sl, fill=_MUTED,  fontSize="8"),
            _text(sx + 57, 84, sv, fill=_YELLOW, fontSize="9", fontWeight="bold"),
            _line(sx + 57, 94, sx + 57, 112, stroke=_GRAY, strokeWidth="2",
                  strokeDasharray="3,2"),
        ]

    # ── TURBINE MAIN BOX ──────────────────────────────────────────────────────
    turbine_box = [
        _rect(336, 193, 490, 292, rx="12", fill="none",
              stroke=gta_col, strokeWidth="2.5", filter="url(#glow)"),
        _rect(337, 194, 488, 290, rx="11", fill="url(#tgrad)"),
        _text(581, 216, "TURBINE À VAPEUR",
              fill=gta_col, fontSize="14", fontWeight="bold", filter="url(#glow)"),
        _text(581, 231, "Réduction multi-étages HP → MP → BP",
              fill=_MUTED, fontSize="8"),
        _turbine_stage(350, 252, "HP", "#1f6feb",    vap_inlet,    adm_debit),
        _turbine_stage(490, 252, "MP", _YELLOW,      sout_pression, sout_debit),
        _turbine_stage(630, 252, "BP", _CYAN,         bp_pression,  bp_debit),
    ]

    # Turbine bottom metrics bar
    turb_metrics = [
        _rect(350, 398, 490, 78, rx="5", fill=_BG, stroke=_BORDER),
    ]
    for lbl, val, col, tx, ty in [
        ("VITESSE",   f"{vitesse} RPM",       _TEXT,    410, 418),
        ("RENDEMENT", f"{rendement}%",         rend_col, 530, 418),
        ("P adm.",    f"{adm_pression} bar",   _YELLOW,  660, 418),
        ("T° adm.",   f"{adm_temp}°C",         _ORANGE,  780, 418),
        ("POSIT. HP", f"{posit_hp}%",           _YELLOW,  410, 450),
        ("DÉBIT ADM.",f"{adm_debit} t/h",      _TEXT,    530, 450),
        ("COS Φ",     str(cos_phi),             _TEXT,    660, 450),
        ("P active",  f"{puissance_mw} MW",         gta_col,  780, 450),
    ]:
        turb_metrics.append(_metric(tx, ty, lbl, val, col))

    # ── VIBRATION SHAFT LINE ──────────────────────────────────────────────────
    vib_sensors = [
        _line(350, 488, 820, 488, stroke=_GRAY,  strokeWidth="8", strokeLinecap="round"),
        _line(350, 488, 820, 488, stroke=_MUTED, strokeWidth="3",
              strokeLinecap="round", opacity="0.5"),
    ]
    for vx, vy, vl, vv in [
        (405, 490, "VIB1-TV",  f"{vib1} μm"),
        (560, 490, "20ST10C",  "-2.1 mm/s"),
        (690, 490, "VIB2-TV",  f"{vib2} μm"),
        (760, 490, "DD3",      f"{dd3} mm"),
    ]:
        col2 = _RED if vl == "20ST10C" else _TEXT
        vib_sensors += [
            _rect(vx - 42, vy - 11, 84, 22, rx="3",
                  fill="rgba(13,17,23,0.7)", stroke=_GRAY),
            _text(vx - 10, vy, vl,  fill=_MUTED, fontSize="7.5", textAnchor="start"),
            _text(vx + 40, vy, vv,  fill=col2,   fontSize="8",
                  fontWeight="bold", textAnchor="end"),
        ]

    # Temperature tags on turbine body
    temp_tags = []
    for tx2, ty2, tn, tv in [
        (363, 245, "20TE173", f"{round(adm_temp - 14, 1)}°C"),
        (440, 245, "20TE175", "55.7°C"),
        (540, 245, "20TE181", "35.9°C"),
        (720, 245, "20TE183", "36.8°C"),
        (820, 245, "20TE185", "35.5°C"),
    ]:
        temp_tags += [
            _text(tx2, ty2 - 8, tn, fill=_MUTED, fontSize="6.5"),
            _rect(tx2 - 24, ty2 - 2, 48, 14, rx="3",
                  fill="rgba(13,17,23,0.7)", stroke=_GRAY),
            _text(tx2, ty2 + 5, tv, fill=_ORANGE, fontSize="8", fontWeight="bold"),
        ]

    # ── ALTERNATEUR ───────────────────────────────────────────────────────────
    alt = [
        _line(826, 488, 870, 313,
              stroke=_MUTED, strokeWidth="6", strokeLinecap="round"),
        _rect(870, 168, 200, 230, rx="10", fill="none",
              stroke=_GREEN, strokeWidth="2.5"),
        _text(970, 191, "ALTERNATEUR",              fill=_GREEN, fontSize="12",
              fontWeight="bold"),
        _text(970, 206, "47 MVA · Topologie Froide", fill=_MUTED, fontSize="8"),
        _circle(970, 278, 44, fill="#0d2137", stroke=_GREEN, strokeWidth="2",
                filter="url(#glow)"),
        _text(970, 283, "~", fill=_GREEN, fontSize="36",
              fontFamily="serif"),
    ]
    for lbl, val, c, tx, ty in [
        ("P active",  f"{p_active} MW",   _GREEN, 900, 340),
        ("P réact.",  f"{p_reactive} Mvar", _TEXT, 900, 356),
        ("Tension",   f"{tension} kV",     _YELLOW,900, 372),
        ("Cosφ",      str(cos_phi),         _TEXT, 1010, 340),
        ("I",         "1369 A",             _TEXT, 1010, 356),
        ("VIB-ALT",   f"{vib1} μm",         _TEXT, 1010, 372),
    ]:
        alt.append(_metric(tx, ty, lbl, val, c))

    # ── BREAKER + TRANSFORMER ─────────────────────────────────────────────────
    elec = [
        _line(1070, 313, 1090, 313, stroke=_GREEN, strokeWidth="5",
              strokeLinecap="round"),
        _rect(1082, 301, 18, 24, fill=_BG, stroke=_GREEN),
        _circle(1112, 313, 14, stroke=_YELLOW, fill="none", strokeWidth="2"),
        _circle(1136, 313, 14, stroke=_YELLOW, fill="none", strokeWidth="2"),
        _pipe(1030, 313, 1070, 313, _GREEN, 4, dashed=True),
    ]

    # ── RÉSEAU MT ─────────────────────────────────────────────────────────────
    reseau = [
        _rect(1150, 188, 185, 175, rx="10", fill="none",
              stroke=_GREEN, strokeWidth="2", strokeDasharray="7,4"),
        _text(1242, 211, "RÉSEAU MT",
              fill=_GREEN, fontSize="11", fontWeight="bold"),
    ]
    for rx2 in [1185, 1230, 1275]:
        reseau += [
            _line(rx2, 228, rx2, 253, stroke=_GREEN, strokeWidth="2"),
            _line(rx2 - 10, 253, rx2 + 10, 253, stroke=_GREEN, strokeWidth="2"),
            _line(rx2 -  6, 260, rx2 +  6, 260, stroke=_GREEN, strokeWidth="1.5"),
            _line(rx2 -  2, 266, rx2 +  2, 266, stroke=_GREEN, strokeWidth="1"),
        ]
    for lbl, val, c, tx, ty in [
        ("P active",   f"{p_active} MW",  _GREEN,  1165, 288),
        ("f réseau",   "50 Hz",            _TEXT,   1165, 306),
        ("U bus",      "6.3 kV",           _YELLOW, 1165, 324),
        ("Charge site","14.0 %",           _TEXT,   1165, 342),
    ]:
        reseau.append(_metric(tx, ty, lbl, val, c))
    reseau.append(_pipe(1070, 313, 1150, 313, _GREEN, 4))

    # ── SOUTIRAGE MP ──────────────────────────────────────────────────────────
    sout_mp = [
        _line(550, 483, 550, 538,
              stroke=_YELLOW, strokeWidth="6",
              strokeLinecap="round", markerEnd="url(#arr)"),
        _gauge(550, 508, "FT", str(sout_debit),   _YELLOW),
        _gauge(620, 530, "TT", f"{adm_temp}°",    _ORANGE),
        _rect(460, 553, 230, 80, rx="8",
              fill="none", stroke=_YELLOW, strokeWidth="1.8"),
        _text(575, 571, "SOUTIRAGE MP",
              fill=_YELLOW, fontSize="11", fontWeight="bold"),
        _text(575, 588, f"{sout_debit} t/h",
              fill=_TEXT, fontSize="14", fontWeight="bold"),
        _text(575, 605, f"{sout_pression} bar",
              fill=_MUTED, fontSize="10"),
        _text(575, 620, "Vapeur MP soutirée",
              fill=_MUTED, fontSize="8"),
    ]

    # ── CONDENSEUR PRINCIPAL ──────────────────────────────────────────────────
    level_h = 58 * (level_pct / 100)
    cond = [
        _line(690, 483, 690, 548,
              stroke=_CYAN, strokeWidth="5",
              strokeLinecap="round", markerEnd="url(#arrc)"),
        _gauge(690, 518, "VMP", f"{bp_pression}b", _CYAN),
        _rect(700, 553, 250, 125, rx="8",
              fill="none", stroke=_CYAN, strokeWidth="2.2", filter="url(#glow)"),
        _text(825, 573, "CONDENSEUR PRINCIPAL",
              fill=_CYAN, fontSize="11", fontWeight="bold"),
        _text(825, 589, "Pression quasi-vide (absolue)",
              fill=_MUTED, fontSize="8"),
        # level indicator
        _rect(712, 598, 80, 62, rx="3", fill="#0d2137"),
        _rect(712, 598 + 62 - level_h, 80, level_h,
              rx="2", fill="#1f6feb", opacity="0.6"),
        _text(752, 610, "20LT201",  fill=_MUTED, fontSize="7"),
        _text(752, 650, f"{level_pct} %",
              fill=_TEXT, fontSize="10", fontWeight="bold"),
        _text(807, 606, "P vide:",   fill=_MUTED, fontSize="9", textAnchor="start"),
        _text(947, 606, f"{ext_pression} bar",
              fill=_CYAN, fontSize="10", fontWeight="bold", textAnchor="end"),
        _text(807, 624, "T bp sort.:", fill=_MUTED, fontSize="9", textAnchor="start"),
        _text(947, 624, f"{cond_temp}°C",
              fill=_ORANGE, fontSize="10", fontWeight="bold", textAnchor="end"),
        _text(807, 642, "D eau:",     fill=_MUTED, fontSize="9", textAnchor="start"),
        _text(947, 642, f"{cond_eau} t/h",
              fill=_TEXT, fontSize="10", fontWeight="bold", textAnchor="end"),
    ]

    # ── EAU DE REFROIDISSEMENT + POMPES ───────────────────────────────────────
    refroid = [
        _rect(14, 508, 300, 170, rx="8",
              fill="rgba(13,17,23,0.6)", stroke=_BORDER, strokeWidth="1.5"),
        _text(164, 526, "EAU DE REFROIDISSEMENT",
              fill=_CYAN, fontSize="10", fontWeight="bold"),
        # bac à vide
        _rect(24, 538, 90, 100, rx="5", fill="#0d1f37", stroke="#1f6feb",
              strokeWidth="1.5"),
        _text(69, 556, "Bac Vide",  fill="#1f6feb", fontSize="9"),
        _text(69, 568, "20CR04",    fill=_MUTED,    fontSize="8"),
        _rect(32, 575, 74, 56, rx="3", fill="#0d2137"),
        _rect(32, 575 + 56 * 0.22, 74, 56 * 0.78,
              rx="2", fill="#1f6feb", opacity="0.5"),
        _text(69, 606, "Eau de mer", fill=_CYAN, fontSize="7.5"),
    ]
    for px, pl, pa, pv in [(148, "20CC01", "A_20CC01", "0.1 A"),
                            (220, "20MC01", "A_20MC01", "0.0 A")]:
        py = 546
        refroid += [
            _circle(px, py + 28, 18,
                    fill=_BG, stroke=_MUTED, strokeWidth="1.5"),
            _text(px, py + 28, "M",        fill=_MUTED, fontSize="10"),
            _text(px, py + 54, pl,         fill=_MUTED, fontSize="8"),
            _rect(px - 28, py + 60, 56, 18, rx="3", fill=_BG, stroke=_GRAY),
            _text(px, py + 72, pv,         fill=_YELLOW, fontSize="8"),
        ]
    refroid += [
        _pipe(320, 613, 460, 613, _CYAN, 4),
        _text(390, 602, "Condensats", fill=_CYAN, fontSize="8"),
        _gauge(400, 638, "ATI", "6.2 μS", _CYAN),
    ]

    # ── SOURCE BP ─────────────────────────────────────────────────────────────
    src_bp = [
        _rect(14, 418, 155, 82, rx="8",
              fill="none", stroke=_CYAN, strokeWidth="1.8"),
        _text(91, 438, "SOURCE BP",               fill=_CYAN,  fontSize="11",
              fontWeight="bold"),
        _text(91, 454, f"{bp_pression} bar",       fill=_CYAN,  fontSize="12",
              fontWeight="bold"),
        _text(91, 469, "Désamorçage S-18 min",     fill=_MUTED, fontSize="7"),
        _text(91, 482, f"Go: {bp_debit} t/h",      fill=_TEXT,  fontSize="9"),
        _dashed_flow(169, 459, 338, 378, _CYAN, 2.5),
        _text(255, 413, "(Désamorçage autoconsent.)", fill=_MUTED, fontSize="7.5"),
    ]

    # Vapeur MP pipe (left side)
    vap_mp = [
        _pipe(169, 343, 338, 343, _YELLOW, 5),
        _text(255, 332, "Vapeur MP",       fill=_YELLOW, fontSize="8"),
        _text(255, 348, f"{sout_pression} bar",
              fill=_YELLOW, fontSize="9", fontWeight="bold"),
    ]

    # ── HUILE GRAISSAGE ───────────────────────────────────────────────────────
    huile = [
        _rect(380, 543, 200, 60, rx="6",
              fill="rgba(210,153,34,0.06)", stroke=_YELLOW, strokeWidth="1.8"),
        _text(480, 560, "HUILE GRAISSAGE",
              fill=_YELLOW, fontSize="10", fontWeight="bold"),
        _text(480, 574, "Lubrification Bain de Borne",
              fill=_MUTED, fontSize="7.5"),
        _text(408, 592, "Pression",         fill=_MUTED,   fontSize="8",
              textAnchor="start"),
        _text(408, 604, f"{oil_pression} bar",
              fill=_YELLOW, fontSize="11", fontWeight="bold", textAnchor="start"),
        _text(515, 592, "Temp.",            fill=_MUTED,   fontSize="8",
              textAnchor="start"),
        _text(515, 604, f"{oil_temp}°C",   fill=_ORANGE,  fontSize="11",
              fontWeight="bold", textAnchor="start"),
    ]

    # ── ALARMES ───────────────────────────────────────────────────────────────
    alarmes = [
        _rect(1148, 48, 185, 110, rx="7",
              fill="rgba(13,17,23,0.7)", stroke=_GRAY),
        _text(1238, 64, "ALARMES", fill=_MUTED, fontSize="10"),
        _alarm_dot(1163, 86,  "TRIP",        rendement < 35),
        _alarm_dot(1163, 106, "VIBRATION",   vib2 > 0.8),
        _alarm_dot(1163, 126, "LOW VACUUM",  ext_pression > 0.12),
        _alarm_dot(1163, 146, "TEMP. ADM.",  adm_temp > 460),
    ]

    # ── ÉTAT SYSTÈME ──────────────────────────────────────────────────────────
    etat = [
        _rect(1010, 513, 330, 165, rx="9",
              fill="rgba(13,17,23,0.9)", stroke=_GRAY, strokeWidth="1.5"),
        _text(1175, 535, "ÉTAT SYSTÈME",
              fill=_GREEN, fontSize="12", fontWeight="bold"),
        _text(1175, 552, statut, fill=stat_col, fontSize="10"),
    ]
    for lbl, val, c, tx, ty in [
        ("V1",  "100%",          _GREEN,  1022, 568),
        ("V2",  "100%",          _GREEN,  1022, 588),
        ("V3",  "100%",          _GREEN,  1022, 608),
        ("HP",  "50%",           _YELLOW, 1022, 628),
        ("BP",  "79%",           _CYAN,   1022, 648),
        ("P active",    f"{p_active} MW",   gta_col, 1145, 568),
        ("Vitesse",     f"{vitesse} RPM",    _TEXT,   1145, 588),
        ("Rendement",   f"{rendement} %",    rend_col,1145, 608),
        ("P barillet",  f"{sout_pression} bar", _YELLOW, 1145, 628),
        ("Cos φ",       str(cos_phi),         _TEXT,   1145, 648),
    ]:
        etat.append(_sys_row(tx, ty, lbl, val, c))

    # HEURES badge
    etat += [
        _rect(1148, 168, 180, 28, rx="5", fill=_BG, stroke=_GRAY),
        _text(1238, 182, "HEURES GTA",  fill=_MUTED,  fontSize="9"),
        _text(1316, 182, "32767 h",     fill=_YELLOW, fontSize="9",
              fontWeight="bold", textAnchor="end"),
    ]

    # A-VIREUR
    a_vireur = [
        _rect(830, 278, 40, 50, rx="5", fill="#161b22", stroke=_MUTED),
        _text(850, 293, "A_V", fill=_MUTED, fontSize="8"),
        _text(850, 308, "IRR", fill=_MUTED, fontSize="7"),
        _text(850, 321, "0.0 A", fill=_YELLOW, fontSize="7"),
    ]

    # ── ASSEMBLE ALL ──────────────────────────────────────────────────────────
    children = (
        [defs,
         _rect(0, 0, 1360, 700, fill=_BG)]
        + bg_grid
        + title_bar
        + badges
        + dilatation
        + consigne
        + valves
        + [src_hp]
        + hp_pipe
        + serv
        + turbine_box
        + turb_metrics
        + vib_sensors
        + temp_tags
        + alt
        + elec
        + reseau
        + sout_mp
        + cond
        + refroid
        + src_bp
        + vap_mp
        + huile
        + alarmes
        + etat
        + a_vireur
    )

    diagram = svg.Svg(
        viewBox="0 0 1360 700",
        style={"width": "100%", "height": "auto",
               "display": "block", "backgroundColor": _BG},
        children=children,
    )
    return html.Div(diagram,
                    style={"padding": "8px", "backgroundColor": _BG,
                           "borderRadius": "8px"})


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def layout_gta():
    return html.Div([
        # ── GTA selector ──────────────────────────────────────────────────────
        html.Div(style=card_style, children=[
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "gap": "24px", "flexWrap": "wrap"},
                children=[
                    html.H5("🏭 GTA — Vue Procédé Interactive",
                            style={"color": COLORS["text_primary"], "margin": 0}),
                    dcc.RadioItems(
                        id="gta-selector",
                        options=[
                            {"label": "  GTA 1",           "value": "GTA1"},
                            {"label": "  GTA 3",           "value": "GTA3"},
                            {"label": "  GTA 2 (Sept–Déc)","value": "GTA2"},
                        ],
                        value="GTA1",
                        inline=True,
                        style={"color": COLORS["text_primary"]},
                        inputStyle={"marginRight": "5px", "marginLeft": "16px"},
                    ),
                ]
            ),
        ]),

        # ── SVG interactive diagram ────────────────────────────────────────────
        html.Div(
            style={**card_style, "padding": "0", "overflow": "hidden"},
            children=[
                html.Div(id="gta-svg-container",
                         style={"width": "100%", "position": "relative"}),
            ]
        ),

        # ── Daily trend charts ─────────────────────────────────────────────────
        html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "repeat(3,1fr)",
                   "gap": "16px", "marginTop": "16px"},
            children=[
                html.Div(style=card_style, children=[
                    html.H6("Débit Admission (t/h)",
                            style={"color": COLORS["text_muted"]}),
                    dcc.Graph(id="gta-chart-debit", style={"height": "220px"}),
                ]),
                html.Div(style=card_style, children=[
                    html.H6("Energie Produite (MWh)",
                            style={"color": COLORS["text_muted"]}),
                    dcc.Graph(id="gta-chart-energie", style={"height": "220px"}),
                ]),
                html.Div(style=card_style, children=[
                    html.H6("Rendement (%)",
                            style={"color": COLORS["text_muted"]}),
                    dcc.Graph(id="gta-chart-rendement", style={"height": "220px"}),
                ]),
            ]
        ),
        dcc.Interval(id="gta-live-interval", interval=1000, n_intervals=0),
        dcc.Store(id="gta-daily-data"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

import requests
from dash import callback, no_update

# IMPORTANT: When using dash.register_page, DO NOT wrap callbacks in a function!
# Use the @callback decorator directly so Dash Pages can auto-register them.

@callback(
    Output("gta-daily-data", "data"),
    Input("gta-selector", "value"),
    Input("gta-live-interval", "n_intervals"),
)
def fetch_gta_data(gta, n):
    # 1. Try Live API first (Use 127.0.0.1 since you are NOT in Docker)
    try:
        resp = requests.get(f"http://127.0.0.1:8051/api/live/{gta}", timeout=1)
        if resp.status_code == 200:
            live_data = resp.json()
            if isinstance(live_data, list):
                live_data = {"data": live_data}
            live_data["is_live"] = True
            return live_data
    except Exception as e:
        print(f"[Live API Error] {e}")
        pass  # Fallback to historical
            
    # 2. Fallback to Historical Data
    try:
        hist_data = get(f"/data/daily/{gta}")
        if isinstance(hist_data, list):
            hist_data = {"data": hist_data, "is_live": False}
        elif isinstance(hist_data, dict):
            hist_data["is_live"] = False
        return hist_data
    except Exception as e:
        print(f"[Historical API Error] {e}")
        return {"data": [], "is_live": False}

@callback(
    Output("gta-svg-container", "children"),
    Input("gta-daily-data", "data"),
    Input("gta-selector", "value"),
)
def update_svg(data, gta):
    if not data:
        return html.Div("⏳ En attente de données...", style={"color": "white", "padding": "20px"})
    
    # Safe extraction
    records = data.get("data", []) if isinstance(data, dict) else data
    
    if not records:
        return html.Div("📭 Aucune donnée disponible.", style={"color": "red", "padding": "20px"})
        
    # Pass normalized data to the SVG builder
    return _build_gta_svg({"data": records, "is_live": data.get("is_live", False)}, gta)

@callback(
    [Output("gta-chart-debit", "figure"),
     Output("gta-chart-energie", "figure"),
     Output("gta-chart-rendement", "figure")],
    Input("gta-daily-data", "data"),
)
def update_gta_charts(data):
    import plotly.graph_objects as go
    import pandas as pd

    if not data:
        return _empty_fig(), _empty_fig(), _empty_fig()

    # Safe extraction
    records = data.get("data", []) if isinstance(data, dict) else data
    is_live = data.get("is_live", False) if isinstance(data, dict) else False

    if not records:
        return _empty_fig(), _empty_fig(), _empty_fig()

    df = pd.DataFrame(records)
    if df.empty:
        return _empty_fig(), _empty_fig(), _empty_fig()

    gta_name  = data.get("gta", "GTA") if isinstance(data, dict) else "GTA"
    color_map = {"GTA1": COLORS["accent_blue"],
                 "GTA2": COLORS["accent_orange"],
                 "GTA3": COLORS["accent_green"]}
    color = color_map.get(gta_name, COLORS["accent_blue"])
    bg    = COLORS["bg_card"]

    if is_live:
        # --- LIVE MODE: Show current values as Gauges ---
        last_row = df.iloc[-1]
        adm_debit = float(last_row.get("adm_debit", 0))
        energie = float(last_row.get("puissance_mw", last_row.get("energie_mwh", 0))) 
        rendement = float(last_row.get("rendement", 0))
        
        def _live_gauge(value, title, color, max_val=100):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': f"<b>{title}</b>", 'font': {'size': 14, 'color': COLORS["text_muted"]}},
                number={'font': {'size': 24, 'color': color}},
                gauge={
                    'axis': {'range': [0, max_val], 'tickcolor': COLORS["text_muted"]},
                    'bar': {'color': color},
                    'bgcolor': bg,
                    'borderwidth': 2,
                    'bordercolor': COLORS["border"],
                }
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                margin=dict(l=20, r=20, t=40, b=20), height=220,
            )
            return fig
        
        return (
            _live_gauge(adm_debit, "Débit Admission (t/h)", color, max_val=250),
            _live_gauge(energie, "Puissance (MW)", COLORS["accent_yellow"], max_val=50),
            _live_gauge(rendement, "Rendement (%)", COLORS["accent_green"], max_val=50),
        )
    else:
        # --- HISTORICAL MODE: Show Line Charts ---
        x = df["date"] if "date" in df.columns else df.index
        
        def _line_fig(col, y_title, c=color):
            fig = go.Figure(go.Scatter(
                x=x, y=df[col], mode="lines",
                line=dict(color=c, width=2),
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg,
                margin=dict(l=50, r=20, t=30, b=40), height=220,
                xaxis=dict(title="Date", showgrid=True, gridcolor=COLORS["border"]),
                yaxis=dict(title=y_title, showgrid=True, gridcolor=COLORS["border"]),
                showlegend=False
            )
            return fig

        energie_col = "puissance_mw" if "puissance_mw" in df.columns else "energie_mwh"

        return (
            _line_fig("adm_debit", "Débit (t/h)"),
            _line_fig(energie_col, "Energie (MWh)"),
            _line_fig("rendement", "Rendement (%)", COLORS["accent_green"]),
        )

def _empty_fig():
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_card"],
        margin=dict(l=0, r=0, t=0, b=0),
        height=220,
    )
    return fig

# This is required for Dash Pages to find the layout!
layout = layout_gta