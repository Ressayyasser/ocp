"""
websocket.py — WebSocket endpoint for real-time SCADA data + alerts.

Connect: ws://localhost:8000/ws/realtime
Connect: ws://localhost:8000/ws/alerts
"""

from __future__ import annotations
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

ws_router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, default=str)
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                self.active.remove(ws)


_realtime_mgr = ConnectionManager()
_alert_mgr    = ConnectionManager()


@ws_router.websocket("/ws/realtime")
async def ws_realtime(websocket: WebSocket, request: Request):
    """Streams SCADA readings every second."""
    await _realtime_mgr.connect(websocket)
    try:
        while True:
            reading = getattr(request.app.state, "latest_reading", {})
            if reading:
                await websocket.send_text(json.dumps(reading, default=str))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        _realtime_mgr.disconnect(websocket)


@ws_router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket, request: Request):
    """Streams new alerts as they are generated."""
    await _alert_mgr.connect(websocket)

    # Register alert callback when first client connects
    alert_svc = getattr(request.app.state, "alerts", None)
    if alert_svc:
        async def _send(alert: dict):
            await _alert_mgr.broadcast({"type": "alert", "data": alert})
        # Wrap async callback in sync
        def _sync_cb(a):
            asyncio.create_task(_send(a))
        alert_svc.subscribe(_sync_cb)

    try:
        while True:
            await asyncio.sleep(30)   # keep-alive ping
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        _alert_mgr.disconnect(websocket)


async def broadcast_reading(reading: dict):
    """Call from anywhere to push a reading to all WebSocket clients."""
    await _realtime_mgr.broadcast({"type": "reading", "data": reading})


async def broadcast_alert(alert: dict):
    """Push an alert to all subscribed WebSocket clients."""
    await _alert_mgr.broadcast({"type": "alert", "data": alert})