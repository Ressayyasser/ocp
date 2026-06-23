import os
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 10  # seconds


def get(path: str, params: dict = None) -> dict | list:
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def post(path: str, payload: dict = None) -> dict:
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=payload or {}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}
