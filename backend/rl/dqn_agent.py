"""
dqn_agent.py — Double DQN with Prioritized Experience Replay (PyTorch).

Network: Linear(9→256) → ReLU → Linear(256→128) → ReLU
         → Linear(128→64) → ReLU → Linear(64→11)

Hyperparameters
───────────────
gamma=0.99  lr=0.0003  buffer=100k  batch=256  target_update=1000
"""

from __future__ import annotations
import json
import os
import pickle
from datetime import datetime

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from rl.replay_buffer import PrioritizedReplayBuffer
from rl.environment   import ACTIONS, N_ACTIONS, STATE_DIM

MODEL_DIR = os.environ.get("MODEL_DIR", "models/rl")


# ── Network ───────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128,  64),       nn.ReLU(),
            nn.Linear( 64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DoubleDQNAgent:

    def __init__(
        self,
        state_dim:     int   = STATE_DIM,
        action_dim:    int   = N_ACTIONS,
        gamma:         float = 0.99,
        lr:            float = 0.0003,
        buffer_size:   int   = 100_000,
        batch_size:    int   = 256,
        target_update: int   = 1_000,
        eps_start:     float = 1.0,
        eps_end:       float = 0.01,
        eps_decay:     int   = 50_000,
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.eps_start     = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.steps_done    = 0
        self.losses: list[float] = []

        self.device = torch.device("cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu")

        if _HAS_TORCH:
            self.policy = QNetwork(state_dim, action_dim).to(self.device)
            self.target = QNetwork(state_dim, action_dim).to(self.device)
            self.target.load_state_dict(self.policy.state_dict())
            self.target.eval()
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer(capacity=buffer_size)

    # ── Epsilon ───────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        t = min(1.0, self.steps_done / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1 - t)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if not _HAS_TORCH:
            return np.random.randint(self.action_dim)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy(s).argmax(1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        if not _HAS_TORCH:
            return np.random.randn(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy(s).cpu().numpy()[0]

    # ── Memory ────────────────────────────────────────────────────────────────

    def store(self, s, a, r, ns, done):
        self.memory.push(s, a, r, ns, done)

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self) -> float:
        if not _HAS_TORCH or len(self.memory) < self.batch_size:
            return 0.0

        s, a, r, ns, d, idxs, w = self.memory.sample(self.batch_size)
        S  = torch.FloatTensor(s).to(self.device)
        A  = torch.LongTensor(a).to(self.device)
        R  = torch.FloatTensor(r).to(self.device)
        NS = torch.FloatTensor(ns).to(self.device)
        D  = torch.FloatTensor(d).to(self.device)
        W  = torch.FloatTensor(w).to(self.device)

        # Current Q
        curr_q = self.policy(S).gather(1, A.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_a = self.policy(NS).argmax(1, keepdim=True)
            next_q = self.target(NS).gather(1, next_a).squeeze(1)
            tgt_q  = R + self.gamma * next_q * (1 - D)

        td_err = (curr_q - tgt_q).detach().cpu().numpy()
        self.memory.update_priorities(idxs, td_err)

        loss = (W * (curr_q - tgt_q) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())

        lv = float(loss.item())
        self.losses.append(lv)
        return lv

    # ── Recommendation ────────────────────────────────────────────────────────

    def recommend(self, state: np.ndarray) -> dict:
        qv   = self.q_values(state)
        best = int(np.argmax(qv))
        exp  = np.exp(qv - qv.max())
        conf = float(exp[best] / exp.sum())
        return {
            "action":       ACTIONS[best]["label"],
            "action_index": best,
            "confidence":   round(conf, 3),
            "q_values":     {ACTIONS[i]["label"]: round(float(qv[i]), 4)
                             for i in range(len(qv))},
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        if not _HAS_TORCH:
            return
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or os.path.join(MODEL_DIR, "dqn_agent.pt")
        torch.save({
            "policy":     self.policy.state_dict(),
            "target":     self.target.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)
        with open(path.replace(".pt", "_meta.json"), "w") as f:
            json.dump({
                "steps_done": self.steps_done,
                "epsilon":    self.epsilon,
                "buffer":     len(self.memory),
                "saved_at":   datetime.now().isoformat(),
            }, f, indent=2)

    def load(self, path: str | None = None):
        if not _HAS_TORCH:
            return
        path = path or os.path.join(MODEL_DIR, "dqn_agent.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.target.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.steps_done = ck.get("steps_done", 0)