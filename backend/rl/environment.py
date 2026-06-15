"""
environment.py — Gymnasium-compatible cogeneration RL environment.

State  (dim=9): production, bilan_net, steam_hp, steam_mp,
                pressure, vibration, anomaly_score, efficiency, month
Actions (n=11): see ACTIONS dict below
"""

from __future__ import annotations
import numpy as np

# ── Action catalogue ──────────────────────────────────────────────────────────
ACTIONS: dict[int, dict] = {
    0:  {"name": "do_nothing",        "label": "Do Nothing",        "cost": 0.00},
    1:  {"name": "increase_gta1",     "label": "Increase GTA1",     "cost": 0.05},
    2:  {"name": "increase_gta2",     "label": "Increase GTA2",     "cost": 0.05},
    3:  {"name": "increase_gta3",     "label": "Increase GTA3",     "cost": 0.05},
    4:  {"name": "increase_gtaa",     "label": "Increase GTAA",     "cost": 0.08},
    5:  {"name": "optimize_mp_steam", "label": "Optimize MP Steam", "cost": 0.03},
    6:  {"name": "reduce_steam_loss", "label": "Reduce Steam Loss", "cost": 0.02},
    7:  {"name": "maintenance_gta1",  "label": "Maintenance GTA1",  "cost": 0.30},
    8:  {"name": "maintenance_gta2",  "label": "Maintenance GTA2",  "cost": 0.30},
    9:  {"name": "maintenance_gta3",  "label": "Maintenance GTA3",  "cost": 0.30},
    10: {"name": "activate_boiler",   "label": "Activate Boiler",   "cost": 0.10},
}

N_ACTIONS  = len(ACTIONS)   # 11
STATE_DIM  = 9
STATE_NAMES = [
    "production", "bilan_net", "steam_hp", "steam_mp",
    "pressure", "vibration", "anomaly_score", "efficiency", "month",
]

# State index constants
_PROD, _BILAN, _SHP, _SMP, _PRES, _VIB, _ANOM, _EFF, _MON = range(9)


class CogenEnv:
    """
    Cogeneration optimization environment.
    Compatible with the Gymnasium step() / reset() API without requiring
    Gymnasium to be installed.
    """

    metadata = {"render_modes": []}

    def __init__(self, data=None, episode_length: int = 720):
        self.episode_length = episode_length
        self.n_actions      = N_ACTIONS
        self.state_dim      = STATE_DIM

        self._means = np.zeros(STATE_DIM, dtype=np.float32)
        self._stds  = np.ones(STATE_DIM,  dtype=np.float32)
        self.data   = None

        if data is not None:
            self._load(data)

        self.current_step = 0
        self.current_idx  = 0
        self.state        = np.zeros(STATE_DIM, dtype=np.float32)
        self.wear         = {"gta1": 0.0, "gta2": 0.0, "gta3": 0.0}

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.wear         = {"gta1": 0.0, "gta2": 0.0, "gta3": 0.0}
        if self.data is not None:
            max_start         = max(0, len(self.data) - self.episode_length)
            self.current_idx  = np.random.randint(0, max_start + 1)
            self.state        = self._get_state(self.current_idx)
        else:
            self.state = (np.random.randn(STATE_DIM) * 0.3).astype(np.float32)
        return self.state.copy()

    def step(self, action: int) -> tuple:
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"
        prev = self.state.copy()
        self._apply_wear(action)
        self.current_step += 1
        self.current_idx  += 1

        if self.data is not None and self.current_idx < len(self.data):
            self.state = self._get_state(self.current_idx)
        else:
            self.state = self._sim_next(prev, action)

        reward   = self._reward(prev, self.state, action)
        done     = self.current_step >= self.episode_length
        truncated = (self.data is not None and self.current_idx >= len(self.data) - 1)
        info      = {"step": self.current_step,
                     "action": ACTIONS[action]["label"],
                     "wear": dict(self.wear)}
        return self.state.copy(), reward, done, truncated, info

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load(self, data):
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            cols  = [c for c in STATE_NAMES if c in data.columns]
            arr   = np.zeros((len(data), STATE_DIM), dtype=np.float32)
            for i, c in enumerate(STATE_NAMES):
                if c in data.columns:
                    arr[:, i] = data[c].values.astype(np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
        arr        = np.nan_to_num(arr, 0.0)
        self._means = arr.mean(axis=0)
        self._stds  = arr.std(axis=0)
        self._stds[self._stds == 0] = 1.0
        self.data   = arr

    def _get_state(self, idx: int) -> np.ndarray:
        raw = self.data[idx]
        return ((raw - self._means) / self._stds).astype(np.float32)

    def _sim_next(self, s: np.ndarray, action: int) -> np.ndarray:
        nxt  = s.copy()
        name = ACTIONS[action]["name"]
        if "increase_gta" in name:
            nxt[_PROD]  += 0.10; nxt[_BILAN] += 0.08; nxt[_VIB] += 0.03
        elif name == "optimize_mp_steam":
            nxt[_SMP]   += 0.05; nxt[_EFF]   += 0.02
        elif name == "reduce_steam_loss":
            nxt[_SHP]   += 0.03; nxt[_EFF]   += 0.03
        elif "maintenance" in name:
            nxt[_VIB]   -= 0.15; nxt[_ANOM]  += 0.10
        elif name == "activate_boiler":
            nxt[_SHP]   += 0.08; nxt[_SMP]   += 0.05
        nxt += (np.random.randn(STATE_DIM) * 0.05).astype(np.float32)
        return nxt

    def _apply_wear(self, action: int):
        name = ACTIONS[action]["name"]
        if   name == "increase_gta1":    self.wear["gta1"] += 0.002
        elif name == "increase_gta2":    self.wear["gta2"] += 0.002
        elif name == "increase_gta3":    self.wear["gta3"] += 0.002
        elif name == "maintenance_gta1": self.wear["gta1"] = max(0, self.wear["gta1"] - 0.10)
        elif name == "maintenance_gta2": self.wear["gta2"] = max(0, self.wear["gta2"] - 0.10)
        elif name == "maintenance_gta3": self.wear["gta3"] = max(0, self.wear["gta3"] - 0.10)

    def _reward(self, prev: np.ndarray, nxt: np.ndarray, action: int) -> float:
        """
        reward = 1.0·Δbilan + 0.5·Δefficiency
               - 0.3·action_cost
               - 0.4·vibration_penalty
               - 1.0·anomaly_penalty
               - 0.2·wear_penalty
        """
        delta_bilan = float(nxt[_BILAN] - prev[_BILAN])
        delta_eff   = float(nxt[_EFF]   - prev[_EFF])
        cost        = ACTIONS[action]["cost"]
        vib_pen     = float(max(0, nxt[_VIB] - 0.5))
        anom_pen    = float(max(0, -nxt[_ANOM]))
        wear_pen    = float(sum(self.wear.values()))

        return (1.0  * delta_bilan
                + 0.5  * delta_eff
                - 0.3  * cost
                - 0.4  * vib_pen
                - 1.0  * anom_pen
                - 0.2  * wear_pen)

    def action_name(self, a: int) -> str:
        return ACTIONS[a]["label"]