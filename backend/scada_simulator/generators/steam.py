"""
generators/steam.py — Simulates steam flow rates (t/h).
"""

from __future__ import annotations
import numpy as np


class SteamGenerator:
    # Real nominal: sum of Débit adm across 3 GTAs ~530 t/h HP,
    # Débit sout ~420 t/h MP, Débit ext ~145 t/h BP (from 2022–2025 data)
    _NOMINAL = {"steam_hp": 197.0, "steam_mp": 145.0, "steam_bp": 20.0}

    def __init__(self, noise_std: float = 2.0, rng_seed: int | None = None):
        self.rng        = np.random.default_rng(rng_seed)
        self.noise_std  = noise_std
        self._state     = dict(self._NOMINAL)
        self._leak_rate = 0.0

    def step(self, production: float = 1585.0) -> dict:
        # Nominal daily production ~1585 MWh (GTA1+GTA2+GTA3)
        prod_factor = production / 1585.0
        out = {}
        for key, nominal in self._NOMINAL.items():
            noise = self.rng.normal(0, self.noise_std)
            leak  = self._leak_rate if "hp" in key else 0.0
            val   = max(0.0, nominal * prod_factor - leak + noise)
            self._state[key] = val
            out[key] = round(float(val), 2)

        out["steam_ratio"] = (out["steam_mp"] / out["steam_hp"]
                              if out["steam_hp"] > 0 else 0.0)
        return out

    def inject_steam_loss(self, loss_rate: float = 10.0):
        """Simulate HP steam leak at *loss_rate* t/h."""
        self._leak_rate = loss_rate

    def clear_fault(self):
        self._leak_rate = 0.0