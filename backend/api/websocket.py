from __future__ import annotations
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, default=str)
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                if ws in self.active:
                    self.active.remove(ws)


_realtime_mgr = ConnectionManager()
_alert_mgr    = ConnectionManager()


# FIXED: Removed 'request: Request' signature to protect signature compilation patterns.
# Accessing state blocks safely using 'websocket.app.state' instead.
@ws_router.websocket("/ws/realtime")
async def ws_realtime(websocket: WebSocket):
    """Streams live SCADA readings down to Dash frontends every second."""
    await _realtime_mgr.connect(websocket)
    try:
        while True:
            reading = getattr(websocket.app.state, "latest_reading", {})
            if reading:
                await websocket.send_text(json.dumps(reading, default=str))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        _realtime_mgr.disconnect(websocket)


_alert_bridge_installed = False


def _install_alert_bridge(app, loop: asyncio.AbstractEventLoop):
    """Subscribe ONCE to the AlertService and bridge its worker-thread
    callbacks onto the asyncio loop. AlertService fires from the SCADA
    simulator thread, where `asyncio.create_task` would raise ("no running
    event loop") and alerts would silently never reach the browser —
    `run_coroutine_threadsafe` is the correct hand-off."""
    global _alert_bridge_installed
    if _alert_bridge_installed:
        return
    alert_svc = getattr(app.state, "alerts", None)
    if alert_svc is None:
        return

    def _thread_cb(alert: dict):
        try:
            asyncio.run_coroutine_threadsafe(
                _alert_mgr.broadcast({"type": "alert", "data": alert}), loop)
        except Exception:
            pass

    alert_svc.subscribe(_thread_cb)
    _alert_bridge_installed = True


@ws_router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """Broadcasts newly compiled anomaly alerts down live pipelines."""
    await _alert_mgr.connect(websocket)
    _install_alert_bridge(websocket.app, asyncio.get_running_loop())

    try:
        while True:
            await asyncio.sleep(30)   # Keep-alive frame window
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        _alert_mgr.disconnect(websocket)


async def broadcast_reading(reading: dict):
    """Pushes a fresh SCADA sensor package out to all active WebSocket clients."""
    await _realtime_mgr.broadcast({"type": "reading", "data": reading})


async def broadcast_alert(alert: dict):
    """Pushes an engine alert object out to all active alert channels."""
    await _alert_mgr.broadcast({"type": "alert", "data": alert})