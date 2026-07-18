"""
utils/gta_svg.py — Shared GTA process-flow SVG synoptic builder.

Extracted from pages/gta_visualization.py so that other pages (e.g. the
Digital Twin's Virtual GTA sandbox) can render the same schematic without
importing a Dash *page* module (which would re-register its callbacks).

Process-flow architecture:
    Source HP -> Turbine (HP -> MP -> BP) -> Alternateur -> Reseau MT
                              |
                              v  (echappement BP)
                         Condenseur <-> Refroidissement
"""

from __future__ import annotations
import math

from dash import html
import dash_svg as svg

_SVG_STYLE_ATTRS = frozenset([
    'filter', 'clip-path', 'offset',
    'marker-end', 'marker-start', 'marker-mid',
    'flood-color', 'flood-opacity',
    'color-interpolation-filters',
])

def _svg_kwargs(kw: dict) -> dict:
    """
    Deplace les attributs SVG non supportes par dash_svg v0.0.12
    depuis les kwargs vers style={...}.
    """
    kw = dict(kw)
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
#  COLOUR SYSTEM
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

# Liaison colours (procede) — imposees par le cahier des charges
_HP_FLOW   = "#ff7b54"   # vapeur HP
_MP_FLOW   = "#ffb84d"   # vapeur MP
_BP_FLOW   = "#ffd166"   # vapeur BP
_COND_FLOW = "#4cc9f0"   # condensat
_COOL_FLOW = "#4895ef"   # eau de refroidissement
_ELEC_FLOW = "#2ec27e"   # liaison electrique


def _rend_color(r):
    if r > 85:  return _GREEN
    if r > 75:  return _ORANGE
    return _RED


# ─────────────────────────────────────────────────────────────────────────────
#  SVG PRIMITIVE HELPERS  (pure Python / Dash-SVG)
# ─────────────────────────────────────────────────────────────────────────────

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
#  FLOW / LINK PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _flow_pipe(x1, y1, x2, y2, color, width=9, dur="1.1s", marker=None):
    """Pipe with a continuously animated dashed overlay (stroke-dashoffset)."""
    base = _line(x1, y1, x2, y2, stroke=color, strokeWidth=str(width),
                 strokeLinecap="round", opacity="0.18")
    flow_kw = dict(stroke=color, strokeWidth=str(max(width - 4, 3)),
                   strokeLinecap="round", strokeDasharray="14,9", opacity="0.95")
    if marker:
        flow_kw["markerEnd"] = f"url(#{marker})"
    flow = _line(x1, y1, x2, y2,
                 children=[svg.Animate(attributeName="stroke-dashoffset", values="0;-46",
                                        dur=dur, repeatCount="indefinite")],
                 **flow_kw)
    return svg.G(children=[base, flow])


def _elec_link(x1, y1, x2, y2, color=_ELEC_FLOW, width=5):
    """Pulsed electrical link (turbine shaft -> alternateur -> reseau).

    NB: no SVG filter here — a percentage-based filter region (objectBoundingBox,
    the default) collapses to zero on an axis-aligned line because the line's
    own geometric bbox has zero height/width, which clips the element to
    nothing. The glow halo is faked instead with a wider, low-opacity duplicate
    line underneath the crisp animated one.
    """
    glow = _line(x1, y1, x2, y2, stroke=color, strokeWidth=str(width + 5),
                 strokeLinecap="round", opacity="0.35")
    core = _line(x1, y1, x2, y2, stroke=color, strokeWidth=str(width),
                 strokeLinecap="round",
                 children=[svg.Animate(attributeName="stroke-opacity", values="1;0.4;1",
                                        dur="1.6s", repeatCount="indefinite")])
    return svg.G(children=[glow, core])


def _wheel(cx, cy, color, r=26):
    blades = [
        _line(cx, cy,
              cx + (r - 6) * math.cos(i * math.pi / 3),
              cy + (r - 6) * math.sin(i * math.pi / 3),
              stroke=color, strokeWidth="2.2", strokeLinecap="round")
        for i in range(6)
    ]
    return svg.G(children=[
        _circle(cx, cy, r, fill="none", stroke=color, strokeWidth="1.8"),
        _circle(cx, cy, r * 0.38, fill=color, fillOpacity="0.3"),
        *blades,
        svg.Animate(attributeName="transform",
                    values=f"rotate(0,{cx},{cy});rotate(360,{cx},{cy})",
                    dur="3s", repeatCount="indefinite"),
    ])


def _stat_row(x, y, w, label, value, color):
    return svg.G(children=[
        _text(x, y, label, fill=_MUTED, fontSize="9.5", textAnchor="start"),
        _text(x + w, y, value, fill=color, fontSize="12.5", fontWeight="bold", textAnchor="end"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  ZONE GEOMETRY  (single source of truth — used for drawing AND click overlays)
# ─────────────────────────────────────────────────────────────────────────────
VB_W, VB_H = 1500, 760

ZONES = {
    "source_hp":       (30,   70,  220, 230),
    "turbine":         (400,  60,  600, 260),
    "alternateur":     (1060, 70,  210, 230),
    "reseau_mt":       (1330, 70,  170, 230),
    "source_bp":       (30,   380, 220, 200),
    "soutirage_mp":    (420,  380, 200, 200),
    "condenseur":      (650,  380, 220, 200),
    "refroidissement": (900,  380, 220, 200),
}

ZONE_LABELS = {
    "source_hp": "Source HP", "turbine": "Turbine à vapeur",
    "alternateur": "Alternateur", "reseau_mt": "Réseau MT",
    "source_bp": "Source BP", "soutirage_mp": "Soutirage MP",
    "condenseur": "Condenseur principal", "refroidissement": "Refroidissement",
}


# ─────────────────────────────────────────────────────────────────────────────
#  TELEMETRY PARSING (shared by the synoptic and the detail panels)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_record(last: dict) -> dict:
    def f(key, default, nd=2):
        v = last.get(key)
        return round(float(v) if v is not None else default, nd)

    d = dict(
        adm_debit=f("adm_debit", 0, 1), adm_temp=f("adm_temp", 0, 1),
        adm_pression=f("adm_pression", 0, 2), sout_debit=f("sout_debit", 0, 1),
        sout_pression=f("sout_pression", 0, 2), ext_debit=f("ext_debit", 0, 1),
        ext_pression=f("ext_pression", 0, 4), puissance_mw=f("puissance_mw", 0, 2),
        rendement=f("rendement", 0, 2), bp_pression=f("bp_pression", 0.9, 2),
        bp_debit=f("bp_debit", 8.7, 1), vib1=f("vib1", 0.2, 2), vib2=f("vib2", 0.4, 2),
        dd3=f("dd3", 0.61, 2), oil_pression=f("oil_pression", 1.52, 2),
        oil_temp=f("oil_temp", 40.4, 1), cos_phi=f("cos_phi", 0.855, 3),
        p_active=f("p_active", 21.3, 1), p_reactive=f("p_reactive", 12.9, 1),
        tension=f("tension", 10.5, 1), vitesse=int(f("vitesse", 6398, 0)),
        posit_hp=f("posit_hp", -1.0, 1), posit_bp=f("posit_bp", -0.9, 1),
        vap_inlet=f("vap_inlet", 55.4, 1), cond_temp=f("cond_temp", 245, 1),
        cond_eau=f("cond_eau", 87, 1), level_pct=f("level_pct", 78.1, 1),
    )
    d["puissance_mw"] = round(d["puissance_mw"] / 24, 2) if d["puissance_mw"] > 0 else 0
    d["temp_mp"] = round(d["adm_temp"] * 0.55, 1)
    d["temp_bp"] = round(d["adm_temp"] * 0.32, 1)
    return d


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SVG BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_gta_svg(data: dict, gta_name: str) -> html.Div:
    records = data.get("data", [])
    last = records[-1] if records else {}
    d = _parse_record(last)

    gta_col  = _GTA_COLORS.get(gta_name, "#1f6feb")
    rend_col = _rend_color(d["rendement"])
    statut   = "NORMAL" if d["rendement"] > 85 else "DÉGRADÉ"
    alarm_active = d["rendement"] < 35 or d["vib2"] > 0.8 or d["ext_pression"] > 0.12

    # ── DEFS ──────────────────────────────────────────────────────────────
    defs = svg.Defs(children=[
        svg.Filter(id="glow", children=[
            svg.FeGaussianBlur(stdDeviation="3", result="blur"),
            svg.FeMerge(children=[
                svg.FeMergeNode(**{"in": "blur"}),
                svg.FeMergeNode(**{"in": "SourceGraphic"}),
            ]),
        ]),
        svg.Filter(id="shadow", x="-30%", y="-30%", width="160%", height="160%",
                   children=[
                       svg.FeGaussianBlur(**{"in": "SourceAlpha"}, stdDeviation="3",
                                          result="blur"),
                       svg.FeOffset(**{"in": "blur"}, dx="0", dy="2", result="offsetBlur"),
                       svg.FeMerge(children=[
                           svg.FeMergeNode(**{"in": "offsetBlur"}),
                           svg.FeMergeNode(**{"in": "SourceGraphic"}),
                       ]),
                   ]),
        svg.LinearGradient(
            id="tgrad", x1="0%", y1="0%", x2="100%", y2="100%",
            children=[_stop("0%", "#1a3a5c"), _stop("100%", "#0a1628")],
        ),
        svg.RadialGradient(
            id="vignette", cx="50%", cy="28%", r="80%",
            children=[_stop("0%", "#121b2d", "0"), _stop("100%", "#05080f", "0.6")],
        ),
        svg.Marker(id="arr_hp", markerWidth="10", markerHeight="7", refX="9", refY="3.5",
                   orient="auto", children=[_poly("0 0,10 3.5,0 7", fill=_HP_FLOW)]),
        svg.Marker(id="arr_mp", markerWidth="10", markerHeight="7", refX="9", refY="3.5",
                   orient="auto", children=[_poly("0 0,10 3.5,0 7", fill=_MP_FLOW)]),
        svg.Marker(id="arr_bp", markerWidth="10", markerHeight="7", refX="9", refY="3.5",
                   orient="auto", children=[_poly("0 0,10 3.5,0 7", fill=_BP_FLOW)]),
        svg.Marker(id="arr_cond", markerWidth="10", markerHeight="7", refX="9", refY="3.5",
                   orient="auto", children=[_poly("0 0,10 3.5,0 7", fill=_COND_FLOW)]),
        svg.Marker(id="arr_cool", markerWidth="10", markerHeight="7", refX="9", refY="3.5",
                   orient="auto", children=[_poly("0 0,10 3.5,0 7", fill=_COOL_FLOW)]),
    ])

    # ── BACKGROUND ────────────────────────────────────────────────────────
    bg = [_rect(0, 0, VB_W, VB_H, fill=_BG), _rect(0, 0, VB_W, VB_H, fill="url(#vignette)")]
    for x in range(0, VB_W + 1, 120):
        bg.append(_line(x, 0, x, VB_H, stroke="#171f30", strokeWidth="1", opacity="0.5"))
    for y in range(0, VB_H + 1, 100):
        bg.append(_line(0, y, VB_W, y, stroke="#171f30", strokeWidth="1", opacity="0.5"))

    # ── TITLE BAR ─────────────────────────────────────────────────────────
    title_bar = [
        _rect(0, 0, VB_W, 46, fill="rgba(13,17,23,0.95)", stroke=_BORDER, strokeWidth="1"),
        _rect(0, 45, VB_W, 2, fill=gta_col, opacity="0.8"),
        _text(VB_W / 2, 25, gta_name, fill=gta_col, fontSize="24", fontWeight="bold",
              textAnchor="middle", filter="url(#glow)"),
        _text(VB_W / 2, 39, "DIGITAL TWIN — OCP DPSC Jorf Lasfar", fill=_MUTED,
              fontSize="10", textAnchor="middle"),
        _circle(VB_W - 192, 22, 4, fill=_GREEN, filter="url(#glow)",
                children=[svg.Animate(attributeName="opacity", values="1;0.35;1",
                                       dur="1.6s", repeatCount="indefinite")]),
        _text(VB_W - 182, 26, "LIVE", fill=_GREEN, fontSize="9.5", fontWeight="bold",
              textAnchor="start"),
    ]

    # ── SOURCE HP ─────────────────────────────────────────────────────────
    zx, zy, zw, zh = ZONES["source_hp"]
    source_hp = svg.G(children=[
        _rect(zx, zy, zw, zh, rx="14", fill="rgba(255,123,84,0.05)",
              stroke=_HP_FLOW, strokeWidth="2", filter="url(#shadow)"),
        _text(zx + zw / 2, zy + 26, "SOURCE HP", fill=_HP_FLOW, fontSize="13.5",
              fontWeight="bold", textAnchor="middle"),
        _text(zx + zw / 2, zy + 42, "Vapeur Haute Pression", fill=_MUTED, fontSize="9",
              textAnchor="middle"),
        _line(zx + 18, zy + 52, zx + zw - 18, zy + 52, stroke=_HP_FLOW, strokeWidth="1",
              opacity="0.35"),
        _stat_row(zx + 20, zy + 90,  zw - 40, "Pression",    f"{d['adm_pression']} bar", _YELLOW),
        _stat_row(zx + 20, zy + 115, zw - 40, "Température", f"{d['adm_temp']} °C",      _ORANGE),
        _stat_row(zx + 20, zy + 140, zw - 40, "Débit",       f"{d['adm_debit']} t/h",    _TEXT),
        _text(zx + zw / 2, zy + zh - 16, "🔥", fontSize="20", textAnchor="middle"),
    ])
    hp_inlet_pipe = _flow_pipe(zx + zw, zy + zh / 2, 400, zy + zh / 2, _HP_FLOW,
                                width=10, dur="0.9s", marker="arr_hp")

    # ── TURBINE ───────────────────────────────────────────────────────────
    tx, ty, tw, th = ZONES["turbine"]
    pad, gap = 20, 14
    stage_y, stage_h = ty + 70, th - 70 - 16
    stage_w = (tw - 2 * pad - 2 * gap) / 3
    stage_specs = [
        ("ADM. HP", "#5b9dff", d["vap_inlet"],     d["adm_temp"],  d["adm_debit"]),
        ("ADM. MP", _YELLOW,   d["sout_pression"], d["temp_mp"],   d["sout_debit"]),
        ("ADM. BP", _CYAN,     d["bp_pression"],   d["temp_bp"],   d["bp_debit"]),
    ]
    turbine_children = [
        _rect(tx, ty, tw, th, rx="18", fill="url(#tgrad)", stroke=gta_col,
              strokeWidth="2.5", filter="url(#glow)"),
        _text(tx + tw / 2, ty + 28, "TURBINE À VAPEUR", fill=gta_col, fontSize="17",
              fontWeight="bold", textAnchor="middle", filter="url(#glow)"),
        _text(tx + tw / 2 - 110, ty + 48, f"Rendement {d['rendement']} %", fill=rend_col,
              fontSize="11", fontWeight="bold", textAnchor="middle"),
        _text(tx + tw / 2 + 110, ty + 48, f"{d['vitesse']} RPM", fill=_TEXT, fontSize="11",
              fontWeight="bold", textAnchor="middle"),
        _circle(tx + tw - 24, ty + 24, 7, fill=_RED if alarm_active else _GREEN,
                filter="url(#glow)",
                children=([svg.Animate(attributeName="opacity", values="1;0.3;1",
                                        dur="0.8s", repeatCount="indefinite")]
                          if alarm_active else [])),
    ]
    stage_centers = {}
    for i, (label, color, p, t, deb) in enumerate(stage_specs):
        sx = tx + pad + i * (stage_w + gap)
        cx = sx + stage_w / 2
        wheel_cy = stage_y + 48
        turbine_children += [
            _rect(sx, stage_y, stage_w, stage_h, rx="11", fill="rgba(255,255,255,0.02)",
                  stroke=color, strokeWidth="1.5"),
            _text(cx, stage_y + 18, label, fill=color, fontSize="11", fontWeight="bold",
                  textAnchor="middle"),
            _wheel(cx, wheel_cy, color),
            _stat_row(sx + 12, stage_y + 96,  stage_w - 24, "Pression", f"{p} bar", _YELLOW),
            _stat_row(sx + 12, stage_y + 116, stage_w - 24, "Temp.",    f"{t} °C",  _ORANGE),
            _stat_row(sx + 12, stage_y + 136, stage_w - 24, "Débit",    f"{deb} t/h", _TEXT),
        ]
        stage_centers[label] = (cx, sx, sx + stage_w, stage_y + stage_h)
    turbine = svg.G(children=turbine_children)

    hp_cx, hp_x0, hp_x1, stage_bottom = stage_centers["ADM. HP"]
    mp_cx, mp_x0, mp_x1, _            = stage_centers["ADM. MP"]
    bp_cx, bp_x0, bp_x1, _            = stage_centers["ADM. BP"]

    turbine_bottom = ty + th  # outer casing edge

    # ── SOURCE BP ─────────────────────────────────────────────────────────
    bx0, by0, bw0, bh0 = ZONES["source_bp"]
    source_bp = svg.G(children=[
        _rect(bx0, by0, bw0, bh0, rx="14", fill="rgba(255,209,102,0.05)",
              stroke=_BP_FLOW, strokeWidth="2", filter="url(#shadow)"),
        _text(bx0 + bw0 / 2, by0 + 26, "SOURCE BP", fill=_BP_FLOW, fontSize="13.5",
              fontWeight="bold", textAnchor="middle"),
        _text(bx0 + bw0 / 2, by0 + 42, "Désamorçage S-18 min", fill=_MUTED, fontSize="9",
              textAnchor="middle"),
        _line(bx0 + 18, by0 + 52, bx0 + bw0 - 18, by0 + 52, stroke=_BP_FLOW, strokeWidth="1",
              opacity="0.35"),
        _stat_row(bx0 + 20, by0 + 92,  bw0 - 40, "Pression", f"{d['bp_pression']} bar", _YELLOW),
        _stat_row(bx0 + 20, by0 + 118, bw0 - 40, "Débit",    f"{d['bp_debit']} t/h",     _TEXT),
    ])
    src_bp_pipe = _flow_pipe(bx0 + bw0, by0 + 60, 420, turbine_bottom, _BP_FLOW,
                              width=7, dur="1.3s", marker="arr_bp")

    # ── SOUTIRAGE MP ──────────────────────────────────────────────────────
    mx0, my0, mw0, mh0 = ZONES["soutirage_mp"]
    soutirage_mp = svg.G(children=[
        _rect(mx0, my0, mw0, mh0, rx="14", fill="rgba(255,184,77,0.05)",
              stroke=_MP_FLOW, strokeWidth="2", filter="url(#shadow)"),
        _text(mx0 + mw0 / 2, my0 + 26, "SOUTIRAGE MP", fill=_MP_FLOW, fontSize="13.5",
              fontWeight="bold", textAnchor="middle"),
        _text(mx0 + mw0 / 2, my0 + 42, "Vapeur MP soutirée", fill=_MUTED, fontSize="9",
              textAnchor="middle"),
        _line(mx0 + 18, my0 + 52, mx0 + mw0 - 18, my0 + 52, stroke=_MP_FLOW, strokeWidth="1",
              opacity="0.35"),
        _stat_row(mx0 + 20, my0 + 92,  mw0 - 40, "Débit",    f"{d['sout_debit']} t/h",    _TEXT),
        _stat_row(mx0 + 20, my0 + 118, mw0 - 40, "Pression", f"{d['sout_pression']} bar", _YELLOW),
    ])
    sout_mp_pipe = _flow_pipe(mp_cx, turbine_bottom, mx0 + mw0 / 2, my0, _MP_FLOW,
                               width=8, dur="1.1s", marker="arr_mp")

    # ── ALTERNATEUR ───────────────────────────────────────────────────────
    ax, ay, aw, ah = ZONES["alternateur"]
    alt_cx = ax + 55
    alt_cy = ay + ah / 2
    alternateur = svg.G(children=[
        _rect(ax, ay, aw, ah, rx="14", fill="rgba(63,185,80,0.05)", stroke=_GREEN,
              strokeWidth="2.5", filter="url(#shadow)"),
        _text(ax + aw / 2, ay + 24, "ALTERNATEUR", fill=_GREEN, fontSize="13",
              fontWeight="bold", textAnchor="middle"),
        _text(ax + aw / 2, ay + 39, "47 MVA · Topologie Froide", fill=_MUTED, fontSize="8.5",
              textAnchor="middle"),
        _circle(alt_cx, alt_cy + 10, 36, fill="#0d2137", stroke=_GREEN, strokeWidth="2",
                filter="url(#glow)"),
        _text(alt_cx, alt_cy + 18, "~", fill=_GREEN, fontSize="30", fontFamily="serif",
              textAnchor="middle"),
        _stat_row(ax + 14, ay + ah - 60, aw - 28, "P active", f"{d['p_active']} MW", _GREEN),
        _stat_row(ax + 14, ay + ah - 42, aw - 28, "Cos φ",    str(d['cos_phi']),      _TEXT),
        _stat_row(ax + 14, ay + ah - 24, aw - 28, "Tension",  f"{d['tension']} kV",   _YELLOW),
    ])
    shaft_link = _line(tx + tw, ty + th / 2, ax, ay + ah / 2, stroke=_MUTED, strokeWidth="6",
                        strokeLinecap="round")
    elec_link_1 = _elec_link(ax + aw, ay + ah / 2, ZONES["reseau_mt"][0], ay + ah / 2)

    # ── RÉSEAU MT ─────────────────────────────────────────────────────────
    rx0, ry0, rw0, rh0 = ZONES["reseau_mt"]
    reseau = [
        _rect(rx0, ry0, rw0, rh0, rx="14", fill="rgba(63,185,80,0.04)", stroke=_GREEN,
              strokeWidth="2", strokeDasharray="7,4"),
        _text(rx0 + rw0 / 2, ry0 + 24, "RÉSEAU MT", fill=_GREEN, fontSize="12.5",
              fontWeight="bold", textAnchor="middle"),
    ]
    for px in [rx0 + 35, rx0 + 85, rx0 + 135]:
        py = ry0 + 46
        reseau += [
            _line(px, py, px, py + 22, stroke=_GREEN, strokeWidth="2"),
            _line(px - 9, py + 22, px + 9, py + 22, stroke=_GREEN, strokeWidth="2"),
            _line(px - 5, py + 28, px + 5, py + 28, stroke=_GREEN, strokeWidth="1.5"),
            _line(px - 2, py + 33, px + 2, py + 33, stroke=_GREEN, strokeWidth="1"),
        ]
    reseau += [
        _stat_row(rx0 + 16, ry0 + 118, rw0 - 32, "P active", f"{d['p_active']} MW", _GREEN),
        _stat_row(rx0 + 16, ry0 + 138, rw0 - 32, "f réseau", "50 Hz", _TEXT),
        _stat_row(rx0 + 16, ry0 + 158, rw0 - 32, "U bus",    "6.3 kV", _YELLOW),
        _stat_row(rx0 + 16, ry0 + 178, rw0 - 32, "Charge",   "14.0 %", _TEXT),
    ]
    reseau_g = svg.G(children=reseau)

    # ── CONDENSEUR PRINCIPAL ──────────────────────────────────────────────
    cx0, cy0, cw0, ch0 = ZONES["condenseur"]
    level_h = 70 * (d["level_pct"] / 100)
    condenseur = svg.G(children=[
        _rect(cx0, cy0, cw0, ch0, rx="14", fill="rgba(76,201,240,0.05)", stroke=_COND_FLOW,
              strokeWidth="2.2", filter="url(#glow)"),
        _text(cx0 + cw0 / 2, cy0 + 24, "CONDENSEUR PRINCIPAL", fill=_COND_FLOW, fontSize="12",
              fontWeight="bold", textAnchor="middle"),
        _rect(cx0 + 16, cy0 + 40, 64, 92, rx="4", fill="#0d2137"),
        _rect(cx0 + 16, cy0 + 40 + 92 - level_h, 64, level_h, rx="3", fill="#1f6feb",
              opacity="0.6"),
        _text(cx0 + 48, cy0 + 142, f"{d['level_pct']} %", fill=_TEXT, fontSize="10.5",
              fontWeight="bold", textAnchor="middle"),
        _text(cx0 + 48, cy0 + 36, "Niveau", fill=_MUTED, fontSize="8",
              textAnchor="middle"),
        _stat_row(cx0 + 96, cy0 + 70,  cw0 - 112, "Vide",        f"{d['ext_pression']} bar", _CYAN),
        _stat_row(cx0 + 96, cy0 + 95,  cw0 - 112, "Température", f"{d['cond_temp']} °C",      _ORANGE),
        _stat_row(cx0 + 96, cy0 + 120, cw0 - 112, "Débit eau",   f"{d['cond_eau']} t/h",       _TEXT),
    ])
    bp_exhaust_pipe = _flow_pipe(bp_cx, stage_bottom, cx0 + cw0 * 0.5, cy0, _BP_FLOW,
                                  width=9, dur="1s", marker="arr_bp")

    # ── REFROIDISSEMENT ───────────────────────────────────────────────────
    fx0, fy0, fw0, fh0 = ZONES["refroidissement"]
    refroidissement_children = [
        _rect(fx0, fy0, fw0, fh0, rx="14", fill="rgba(13,17,23,0.65)", stroke=_BORDER,
              strokeWidth="1.5", filter="url(#shadow)"),
        _text(fx0 + fw0 / 2, fy0 + 24, "REFROIDISSEMENT", fill=_COOL_FLOW, fontSize="12",
              fontWeight="bold", textAnchor="middle"),
    ]
    for i, (px_off, pl, pv) in enumerate([(60, "Pompe A · 20CC01", "0.1 A"),
                                           (160, "Pompe B · 20MC01", "0.0 A")]):
        pcx = fx0 + px_off
        refroidissement_children += [
            _circle(pcx, fy0 + 56, 19, fill=_BG, stroke=_MUTED, strokeWidth="1.5"),
            _text(pcx, fy0 + 61, "M", fill=_MUTED, fontSize="11", textAnchor="middle"),
            _text(pcx, fy0 + 84, pl, fill=_MUTED, fontSize="8", textAnchor="middle"),
            _rect(pcx - 26, fy0 + 90, 52, 17, rx="3", fill=_BG, stroke=_GRAY),
            _text(pcx, fy0 + 102, pv, fill=_YELLOW, fontSize="9", fontWeight="bold",
                  textAnchor="middle"),
        ]
    refroidissement_children += [
        _stat_row(fx0 + 16, fy0 + 132, fw0 - 32, "Débit",       f"{d['cond_eau']} t/h",   _CYAN),
        _stat_row(fx0 + 16, fy0 + 152, fw0 - 32, "Temp. entrée", f"{d['cond_temp']} °C",  _ORANGE),
        _stat_row(fx0 + 16, fy0 + 172, fw0 - 32, "Temp. sortie", f"{round(d['cond_temp'] - 8, 1)} °C", _CYAN),
    ]
    refroidissement = svg.G(children=refroidissement_children)

    condensat_pipe = _flow_pipe(cx0 + cw0, cy0 + 70, fx0, fy0 + 70, _COND_FLOW,
                                 width=8, dur="1.2s", marker="arr_cond")
    cooling_return_pipe = _flow_pipe(fx0, fy0 + 110, cx0 + cw0, cy0 + 110, _COOL_FLOW,
                                      width=8, dur="1.2s", marker="arr_cool")

    # ── ASSEMBLE ──────────────────────────────────────────────────────────
    children = (
        [defs] + bg + title_bar
        + [hp_inlet_pipe, source_hp, turbine,
           shaft_link, alternateur, elec_link_1, reseau_g,
           src_bp_pipe, source_bp, sout_mp_pipe, soutirage_mp,
           bp_exhaust_pipe, condenseur, condensat_pipe, cooling_return_pipe,
           refroidissement]
    )

    diagram = svg.Svg(
        viewBox=f"0 0 {VB_W} {VB_H}",
        style={"width": "100%", "height": "auto", "display": "block",
               "backgroundColor": _BG},
        children=children,
    )
    return html.Div(diagram, style={"padding": "0", "backgroundColor": _BG,
                                     "borderRadius": "8px", "position": "relative"})
