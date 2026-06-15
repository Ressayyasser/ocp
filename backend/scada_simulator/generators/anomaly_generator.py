"""
anomaly_generator.py — Injects scripted fault scenarios into the SCADA simulator.

Scenarios
─────────
gta2_vibration   — GTA2 vibration spike (jury demo scenario)
pressure_drop    — HP steam pressure drop
steam_loss       — HP steam leak
production_drop  — Sudden GTA3 production loss
overtemperature  — GTA1 exhaust overtemperature
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Scenario:
    name:         str
    description:  str
    duration_sec: int
    applied:      bool = False


class AnomalyGenerator:
    """Coordinates fault injection across all sub-generators."""

    SCENARIOS: dict[str, Scenario] = {
        "gta2_vibration": Scenario(
            "gta2_vibration",
            "GTA2 vibration spike to 5.8 mm/s (bearing fault simulation)",
            120,
        ),
        "pressure_drop": Scenario(
            "pressure_drop",
            "HP steam pressure drops from 42 to 28 bar over 60 seconds",
            60,
        ),
        "steam_loss": Scenario(
            "steam_loss",
            "HP steam leak — flow loss of 25 t/h",
            90,
        ),
        "production_drop": Scenario(
            "production_drop",
            "GTA3 partial trip — production reduces by 25%",
            180,
        ),
        "overtemperature": Scenario(
            "overtemperature",
            "GTA1 exhaust overtemperature +35°C",
            60,
        ),
    }

    def __init__(self, production_gen, pressure_gen, vibration_gen,
                 temperature_gen, steam_gen):
        self.prod  = production_gen
        self.pres  = pressure_gen
        self.vib   = vibration_gen
        self.temp  = temperature_gen
        self.steam = steam_gen
        self._active: list[str] = []
        self._timers: dict[str, int] = {}

    def inject(self, scenario_name: str):
        """Activate a fault scenario."""
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        s = self.SCENARIOS[scenario_name]
        if scenario_name in self._active:
            return

        if scenario_name == "gta2_vibration":
            self.vib.inject_vibration("gta2", extra_mm_s=4.4)
        elif scenario_name == "pressure_drop":
            self.pres.inject_pressure_drop(rate=0.23)
        elif scenario_name == "steam_loss":
            self.steam.inject_steam_loss(loss_rate=25.0)
        elif scenario_name == "production_drop":
            self.prod.inject_fault("gta3", degradation=0.25)
        elif scenario_name == "overtemperature":
            self.temp.inject_overtemperature("gta1", delta=35.0)

        self._active.append(scenario_name)
        self._timers[scenario_name] = s.duration_sec
        s.applied = True
        print(f"[SCADA] Scenario injected: {scenario_name} — {s.description}")

    def tick(self):
        """Call every second to auto-clear expired faults."""
        expired = []
        for name in list(self._timers):
            self._timers[name] -= 1
            if self._timers[name] <= 0:
                self._clear(name)
                expired.append(name)
        for name in expired:
            self._active.remove(name)
            del self._timers[name]
            print(f"[SCADA] Scenario cleared: {name}")

    def _clear(self, name: str):
        if name == "gta2_vibration":   self.vib.clear_fault("gta2")
        elif name == "pressure_drop":  self.pres.clear_fault()
        elif name == "steam_loss":     self.steam.clear_fault()
        elif name == "production_drop":self.prod.clear_fault("gta3")
        elif name == "overtemperature":self.temp.clear_fault("gta1")

    @property
    def active_scenarios(self) -> list[str]:
        return list(self._active)