"""
simulator.py — Main SCADA simulator loop.
Emits one reading per second via emit() callback or asyncio queue.

Usage (standalone)
──────────────────
python -m backend.scada_simulator.simulator

Usage (from FastAPI)
────────────────────
from backend.scada_simulator.simulator import SCADASimulator
sim = SCADASimulator()
sim.start(callback=my_fn)   # calls my_fn(reading_dict) every second
"""

from __future__ import annotations
import threading
import time
from datetime import datetime

from scada_simulator.generators.production    import ProductionGenerator
from scada_simulator.generators.pressure      import PressureGenerator
from scada_simulator.generators.vibration     import VibrationGenerator
from scada_simulator.generators.temperature   import TemperatureGenerator
from scada_simulator.generators.steam         import SteamGenerator
from scada_simulator.generators.anomaly_generator import AnomalyGenerator


class SCADASimulator:

    def __init__(self, tick_rate: float = 1.0):
        self.tick_rate = tick_rate
        self._running  = False
        self._thread: threading.Thread | None = None

        # Sub-generators
        self.prod  = ProductionGenerator()
        self.pres  = PressureGenerator()
        self.vib   = VibrationGenerator()
        self.temp  = TemperatureGenerator()
        self.steam = SteamGenerator()
        self.anomaly_gen = AnomalyGenerator(self.prod, self.pres, self.vib,
                                             self.temp, self.steam)
        self._callbacks: list = []
        self._second = 0

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self, callback=None):
        if callback:
            self._callbacks.append(callback)
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[SCADA] Simulator started")

    def stop(self):
        self._running = False
        print("[SCADA] Simulator stopped")

    def subscribe(self, callback):
        self._callbacks.append(callback)

    def inject_scenario(self, name: str):
        self.anomaly_gen.inject(name)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            t0 = time.time()
            reading = self.emit()
            for cb in self._callbacks:
                try:
                    cb(reading)
                except Exception as exc:
                    print(f"[SCADA] Callback error: {exc}")
            self.anomaly_gen.tick()
            self._second += 1
            elapsed = time.time() - t0
            time.sleep(max(0, self.tick_rate - elapsed))

    def emit(self) -> dict:
        """Generate one complete sensor reading."""
        hour = (self._second // 3600) % 24

        prod_data  = self.prod.step(hour_of_day=hour)
        pres_data  = self.pres.step()
        vib_data   = self.vib.step()
        temp_data  = self.temp.step()
        steam_data = self.steam.step(production=prod_data["production"])

        production  = prod_data["production"]
        consumption = production * 0.14
        bilan_net   = production - consumption
        steam_hp    = steam_data["steam_hp"]
        efficiency  = production / steam_hp if steam_hp > 0 else 0.0

        reading = {
            "timestamp":   datetime.now().isoformat(),
            "gta1":        prod_data["gta1"],
            "gta2":        prod_data["gta2"],
            "gta3":        prod_data["gta3"],
            "gtaa":        prod_data["gtaa"],
            "gtab":        prod_data["gtab"],
            "production":  round(production,  2),
            "consumption": round(consumption, 2),
            "bilan_net":   round(bilan_net,   2),
            "steam_hp":    steam_data["steam_hp"],
            "steam_mp":    steam_data["steam_mp"],
            "steam_bp":    steam_data["steam_bp"],
            "steam_ratio": steam_data["steam_ratio"],
            "pressure":    pres_data["pressure"],
            "vibration":   vib_data["vibration"],
            "temperature": temp_data["temperature"],
            "efficiency":  round(efficiency, 4),
            "hour":        hour,
        }
        return reading


if __name__ == "__main__":
    def _print(r): print(r)
    sim = SCADASimulator()
    sim.subscribe(_print)
    # Demo: inject GTA2 vibration anomaly after 5 seconds
    threading.Timer(5, lambda: sim.inject_scenario("gta2_vibration")).start()
    sim.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sim.stop()