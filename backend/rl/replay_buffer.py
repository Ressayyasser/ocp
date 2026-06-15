"""
replay_buffer.py — Standard and Prioritized Experience Replay.
"""

from __future__ import annotations
import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Uniform experience replay."""

    def __init__(self, capacity: int = 100_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self): return len(self.buffer)


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (PER)."""

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6,
                 beta: float = 0.4, beta_inc: float = 0.001):
        self.capacity = capacity
        self.alpha    = alpha
        self.beta     = beta
        self.beta_inc = beta_inc
        self.max_prio = 1.0
        self.buffer: list   = []
        self.priorities     = np.zeros(capacity, dtype=np.float32)
        self.pos  = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        t = (np.asarray(state, np.float32), int(action), float(reward),
             np.asarray(next_state, np.float32), bool(done))
        if self.size < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.pos] = t
        self.priorities[self.pos] = self.max_prio
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        idxs  = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        self.beta = min(1.0, self.beta + self.beta_inc)
        w = (self.size * probs[idxs]) ** (-self.beta)
        w /= w.max()
        batch = [self.buffer[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, np.float32),
                np.array(ns), np.array(d, np.float32), idxs, w.astype(np.float32))

    def update_priorities(self, idxs, td_errors):
        for i, err in zip(idxs, td_errors):
            p = abs(float(err)) + 1e-6
            self.priorities[i] = p
            self.max_prio = max(self.max_prio, p)

    def __len__(self): return self.size