"""Reusable alert-toast popup driven by the backend /ws/alerts WebSocket.

Shows a Bootstrap-style toast in the top-right corner the moment the backend
raises an alert (threshold breach, anomaly, etc.). Works on every Dash page
because it is mounted once in app.py.
"""

import json
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

# The toast client runs entirely in the BROWSER (native WebSocket) — no Python
# websocket package is required. The JS is therefore always injected.
_HAS_WS = True

LEVEL_STYLE = {
    "CRITICAL": {"bg": "#ff4d4d", "icon": "bi-exclamation-octagon-fill"},
    "WARNING":  {"bg": "#ffd93d", "icon": "bi-exclamation-triangle-fill"},
    "INFO":     {"bg": "#4d9de0", "icon": "bi-info-circle-fill"},
}

# Whether to play an audible alarm (WebAudio, no asset needed). The operator can
# mute it from the toast itself (persisted in localStorage).
ALARM_ENABLED = True



def register_alert_toast(app, api_base: str = "http://localhost:8000"):
    """Mount the alert-toast store + container and its live-update logic."""
    ws_url = api_base.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = ws_url.rstrip("/") + "/ws/alerts"

    # Hidden store that holds the most recent alert payload
    app.layout.children.append(dcc.Store(id="alert-toast-store", data=None))
    app.layout.children.append(
        html.Div(
            [
                html.Div(id="alert-toast-container",
                         style={"position": "fixed", "top": "16px", "right": "16px",
                                "zIndex": 9999, "width": "360px", "maxWidth": "92vw",
                                "display": "flex", "flexDirection": "column",
                                "gap": "10px"}),
            ]
        )
    )

    if not _HAS_WS:
        # Fallback: poll /alerts every 5s and surface the newest unacked alert
        app.layout.children.append(
            dcc.Interval(id="alert-toast-poll", interval=5_000, n_intervals=0)
        )

        @app.callback(
            Output("alert-toast-store", "data"),
            Input("alert-toast-poll", "n_intervals"),
            State("alert-toast-store", "data"),
            prevent_initial_call=True,
        )
        def _poll_alerts(n, current):
            try:
                import requests
                r = requests.get(f"{api_base}/alerts", params={"limit": 1},
                                 timeout=5)
                rows = r.json().get("alerts", [])
            except Exception:
                return current
            if not rows:
                return current
            last = rows[0]
            if current and current.get("id") == last.get("id"):
                return current
            return last

    else:
        # Live: receive alerts over the WebSocket via a hidden JS client
        app.index_string = app.index_string.replace(
            "</body>",
            f"""
            <script>
            (function() {{
                var MAX_ALERTS = 5;
                var container = null;

                // Dash renders the layout AFTER this script runs — wait for the
                // toast container to exist before wiring everything up.
                var bootTimer = setInterval(function() {{
                    container = document.getElementById('alert-toast-container');
                    if (!container) return;
                    clearInterval(bootTimer);
                    boot();
                }}, 300);

                function boot() {{

                // ── Audible alarm (WebAudio, generated — no asset needed) ──────
                var audioCtx = null;
                function getCtx() {{
                    if (!audioCtx) {{
                        try {{ audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }}
                        catch(e) {{ audioCtx = null; }}
                    }}
                    return audioCtx;
                }}
                function playAlarm(critical) {{
                    try {{
                        var muted = localStorage.getItem('ocp_alarm_muted') === '1';
                        if (muted) return;
                        var ctx = getCtx(); if (!ctx) return;
                        if (ctx.state === 'suspended') ctx.resume();
                        var now = ctx.currentTime;
                        var beeps = critical ? 3 : 2;
                        for (var i = 0; i < beeps; i++) {{
                            var t0 = now + i * 0.45;
                            var osc = ctx.createOscillator();
                            var gain = ctx.createGain();
                            osc.type = critical ? 'square' : 'triangle';
                            osc.frequency.setValueAtTime(critical ? 880 : 660, t0);
                            osc.frequency.setValueAtTime(critical ? 660 : 520, t0 + 0.18);
                            gain.gain.setValueAtTime(0.0001, t0);
                            gain.gain.exponentialRampToValueAtTime(0.35, t0 + 0.02);
                            gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.30);
                            osc.connect(gain); gain.connect(ctx.destination);
                            osc.start(t0); osc.stop(t0 + 0.32);
                        }}
                    }} catch(e) {{}}
                }}

                function showAlert(a) {{
                    var lvl = (a.level || 'INFO').toUpperCase();
                    // Toasts + alarm are reserved for threshold breaches
                    // (WARNING/CRITICAL). INFO stays in the alert panels.
                    if (lvl === 'INFO') return;
                    var critical = lvl === 'CRITICAL';
                    var warning  = lvl === 'WARNING';
                    var colors = {{CRITICAL:'#ff4d4d', WARNING:'#ffd93d', INFO:'#4d9de0'}};
                    var icons = {{CRITICAL:'bi-exclamation-octagon-fill', WARNING:'bi-exclamation-triangle-fill', INFO:'bi-info-circle-fill'}};
                    var bg = colors[lvl] || '#4d9de0';
                    var icon = icons[lvl] || 'bi-info-circle-fill';

                    var t = document.createElement('div');
                    t.className = 'alert-toast' + (critical ? ' alert-toast-critical' : '');
                    t.style.background = bg;
                    t.innerHTML = '<div class="d-flex align-items-start">'
                        + '<i class="bi ' + icon + ' alert-toast-icon me-2"></i>'
                        + '<div style="flex:1;min-width:0">'
                        + '<div class="fw-bold" style="font-size:.82rem;line-height:1.15">'
                        + (a.source ? a.source.toUpperCase() + ' — ' : '') + lvl + '</div>'
                        + '<div style="font-size:.82rem;margin-top:2px">' + (a.message || '') + '</div>'
                        + '</div>'
                        + '<i class="bi bi-x-circle-fill alert-toast-close ms-2" title="Fermer"></i>'
                        + '</div>';

                    // close on click of the X
                    t.querySelector('.alert-toast-close').addEventListener('click', function() {{
                        t.classList.remove('show');
                        setTimeout(function(){{ if (t.parentNode) t.parentNode.removeChild(t); }}, 400);
                    }});

                    container.prepend(t);
                    requestAnimationFrame(function(){{ t.classList.add('show'); }});
                    // force a "pop" + shake so it is immediately noticed
                    t.classList.add('alert-toast-pop');
                    if (critical) t.classList.add('alert-toast-flash');

                    // audible alarm (skipped for plain INFO to avoid noise)
                    if ({str(ALARM_ENABLED).lower()} && lvl !== 'INFO') playAlarm(critical);

                    // auto-dismiss: critical stays longer and re-flashes
                    var life = critical ? 12000 : (warning ? 9000 : 6000);
                    var timer = setTimeout(function() {{
                        t.classList.remove('show');
                        setTimeout(function(){{ if (t.parentNode) t.parentNode.removeChild(t); }}, 400);
                    }}, life);
                    t.addEventListener('mouseenter', function(){{ clearTimeout(timer); }});

                    while (container.children.length > MAX_ALERTS) {{
                        container.removeChild(container.lastChild);
                    }}
                }}

                // ── Mute toggle, persisted ──────────────────────────────────────
                var muteBtn = document.createElement('button');
                muteBtn.id = 'alert-mute-btn';
                muteBtn.className = 'btn btn-sm';
                function renderMute() {{
                    var muted = localStorage.getItem('ocp_alarm_muted') === '1';
                    muteBtn.innerHTML = muted ? '🔇 Alarme OFF' : '🔊 Alarme ON';
                    muteBtn.style.background = muted ? '#444' : '#222';
                    muteBtn.style.color = muted ? '#bbb' : '#fff';
                }}
                muteBtn.onclick = function() {{
                    var muted = localStorage.getItem('ocp_alarm_muted') === '1';
                    localStorage.setItem('ocp_alarm_muted', muted ? '0' : '1');
                    renderMute();
                    if (!muted) {{ try {{ getCtx().resume(); }} catch(e) {{}} }}
                }};
                renderMute();
                container.parentNode.insertBefore(muteBtn, container);

                function connect() {{
                    try {{
                        var proto = location.protocol === 'https:' ? 'wss' : 'ws';
                        var ws = new WebSocket('{ws_url}');
                        ws.onmessage = function(ev) {{
                            try {{
                                var msg = JSON.parse(ev.data);
                                if (msg && (msg.type === 'alert' || msg.data)) {{
                                    var a = msg.data || msg;
                                    if (a && (a.message || a.level)) showAlert(a);
                                }}
                            }} catch(e) {{}}
                        }};
                        ws.onclose = function() {{ setTimeout(connect, 3000); }};
                        ws.onerror = function() {{ try {{ ws.close(); }} catch(e) {{}} }};
                    }} catch(e) {{ setTimeout(connect, 3000); }}
                }}
                connect();
                }}  // end boot()
            }})();
            </script>
            </body>"""
        )
