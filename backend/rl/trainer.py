"""
trainer.py — Orchestrates Double-DQN training with logging and backtest.
"""

from __future__ import annotations
import numpy as np
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.environment  import CogenEnv
from rl.dqn_agent    import DoubleDQNAgent
from database.database import insert_many


class Trainer:

    def __init__(self, data=None, episodes: int = 500, episode_length: int = 720):
        self.env             = CogenEnv(data=data, episode_length=episode_length)
        self.agent           = DoubleDQNAgent()
        self.episodes        = episodes
        self.episode_length  = episode_length
        self.ep_rewards: list[float] = []
        self.ep_losses:  list[float] = []
        self._active         = False

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, callback=None) -> dict:
        self._active  = True
        best_reward   = -float("inf")
        t0            = time.time()

        for ep in range(self.episodes):
            if not self._active:
                break

            state        = self.env.reset()
            total_reward = 0.0
            losses: list[float] = []

            for step in range(self.episode_length):
                action                         = self.agent.select_action(state, training=True)
                next_state, reward, done, trunc, _ = self.env.step(action)
                self.agent.store(state, action, reward, next_state, done or trunc)
                loss = self.agent.train_step()
                if loss > 0:
                    losses.append(loss)
                total_reward += reward
                state         = next_state
                if done or trunc:
                    break

            avg_loss = float(np.mean(losses)) if losses else 0.0
            self.ep_rewards.append(total_reward)
            self.ep_losses.append(avg_loss)

            if total_reward > best_reward:
                best_reward = total_reward
                self.agent.save()

            if ep % 10 == 0:
                insert_many("rl_episodes", [{"episode": ep, "total_reward": round(total_reward, 4),
                                              "steps": step + 1, "avg_bilan": 0}])
            if callback:
                callback(ep, total_reward, avg_loss)
            if ep % 50 == 0:
                print(f"Ep {ep:>4}/{self.episodes} | reward={total_reward:>9.2f} "
                      f"| ε={self.agent.epsilon:.3f} | loss={avg_loss:.5f} "
                      f"| best={best_reward:.2f} | t={time.time()-t0:.0f}s")

        self.agent.save()
        self._active = False
        return self.summary()

    def stop(self):
        self._active = False

    def summary(self) -> dict:
        n = len(self.ep_rewards)
        return {} if n == 0 else {
            "episodes":           n,
            "best_reward":        float(max(self.ep_rewards)),
            "avg_reward_last50":  float(np.mean(self.ep_rewards[-50:])),
            "avg_loss_last50":    float(np.mean(self.ep_losses[-50:])),
            "epsilon":            self.agent.epsilon,
            "steps_done":         self.agent.steps_done,
        }

    # ── Backtest ──────────────────────────────────────────────────────────────

    def backtest(self, strategies: list[str] | None = None) -> dict:
        strategies = strategies or ["do_nothing", "random", "dqn"]
        results    = {}
        for strat in strategies:
            state        = self.env.reset(seed=42)
            total_reward = 0.0
            for step in range(self.episode_length):
                if   strat == "do_nothing": action = 0
                elif strat == "random":     action = np.random.randint(self.env.n_actions)
                else:                       action = self.agent.select_action(state, training=False)
                state, reward, done, trunc, _ = self.env.step(action)
                total_reward += reward
                if done or trunc:
                    break
            results[strat] = {"total_reward": round(total_reward, 2), "steps": step + 1}
        return results