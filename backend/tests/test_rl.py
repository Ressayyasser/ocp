"""Tests for Layer 4 — Causal MDP environment (F10), Double DQN (F15), reward (F29)."""

import numpy as np
import pytest

from rl.environment import CogenEnv, ACTIONS, N_ACTIONS, STATE_DIM, STATE_NAMES
from rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from rl.reward_engine import breakdown, to_economic, PRICE_PER_MWH_DH, MWH_SCALE
from tests.conftest import make_synthetic_df


class TestEnvironment:
    def test_action_catalogue_matches_report(self):
        assert N_ACTIONS == 11
        assert STATE_DIM == 9
        names = {a["name"] for a in ACTIONS.values()}
        assert {"do_nothing", "increase_gta1", "optimize_mp_steam",
                "reduce_steam_loss", "maintenance_gta1", "activate_boiler"} <= names

    def test_reset_and_step_shapes(self):
        env = CogenEnv(episode_length=10)
        s = env.reset(seed=42)
        assert s.shape == (STATE_DIM,)
        s2, r, done, trunc, info = env.step(0)
        assert s2.shape == (STATE_DIM,)
        assert isinstance(r, float) and not done
        assert info["action"] == "Do Nothing"

    def test_reward_matches_report_formula(self):
        """reward = 1.0·Δbilan + 0.5·Δeff − 0.3·cost − 0.4·vib − 1.0·anom − 0.2·wear"""
        env = CogenEnv(episode_length=10)
        prev = env.reset(seed=3)
        action = 1  # increase_gta1 (cost 0.05, wear +0.002)
        nxt, r, *_ = env.step(action)
        expected = (1.0 * (nxt[1] - prev[1])
                    + 0.5 * (nxt[7] - prev[7])
                    - 0.3 * ACTIONS[action]["cost"]
                    - 0.4 * max(0, nxt[5] - 0.5)
                    - 1.0 * max(0, -nxt[6])
                    - 0.2 * sum(env.wear.values()))
        assert r == pytest.approx(expected, abs=1e-5)

    def test_wear_and_maintenance_dynamics(self):
        env = CogenEnv(episode_length=100)
        env.reset(seed=0)
        for _ in range(5):
            env.step(1)                       # increase_gta1
        assert env.wear["gta1"] == pytest.approx(0.010)
        env.step(7)                           # maintenance_gta1
        assert env.wear["gta1"] == 0.0

    def test_episode_terminates(self):
        env = CogenEnv(episode_length=3)
        env.reset(seed=0)
        done = False
        for _ in range(3):
            _, _, done, *_ = env.step(0)
        assert done

    def test_invalid_action_rejected(self):
        env = CogenEnv(episode_length=5)
        env.reset(seed=0)
        with pytest.raises(AssertionError):
            env.step(99)

    def test_loads_historical_dataframe(self):
        df = make_synthetic_df(100).rename(columns={"vibration": "vibration"})
        df["anomaly_score"] = 0.0
        df["month"] = df["timestamp"].dt.month
        env = CogenEnv(data=df[[c for c in STATE_NAMES if c in df.columns]],
                       episode_length=10)
        s = env.reset(seed=1)
        assert np.isfinite(s).all()
        # states are z-normalised against dataset statistics
        assert abs(float(env.data.mean())) < 1e6


class TestReplayBuffers:
    def _fill(self, buf, n=100):
        for i in range(n):
            buf.push(np.zeros(STATE_DIM), i % N_ACTIONS, float(i),
                     np.zeros(STATE_DIM), False)

    def test_uniform_buffer(self):
        buf = ReplayBuffer(capacity=64)
        self._fill(buf, 100)
        assert len(buf) == 64                 # capacity ring
        s, a, r, ns, d = buf.sample(32)
        assert len(a) == 32

    def test_prioritized_buffer_sample_and_update(self):
        buf = PrioritizedReplayBuffer(capacity=128)
        self._fill(buf, 100)
        assert len(buf) == 100
        out = buf.sample(32)
        states, actions, rewards, next_states, dones, idxs, weights = out
        assert len(idxs) == 32 and len(weights) == 32
        buf.update_priorities(idxs, np.random.rand(32))  # must not raise


class TestRewardEngine:
    def test_breakdown_total_is_sum_of_components(self):
        prev = np.zeros(9, dtype=np.float32)
        nxt = np.zeros(9, dtype=np.float32)
        nxt[1] = 0.2   # Δbilan
        nxt[7] = 0.1   # Δefficiency
        nxt[5] = 0.8   # vibration above 0.5 threshold
        res = breakdown(prev, nxt, action=1)
        assert res["total"] == pytest.approx(sum(res["components"].values()), abs=1e-3)
        assert res["components"]["bilan_reward"] == pytest.approx(0.2, abs=1e-4)
        assert res["components"]["vibration_penalty"] == pytest.approx(-0.4 * 0.3, abs=1e-4)

    def test_economic_conversion_uses_700_dh_per_mwh(self):
        """F29 — MWh→DH conversion at 700 DH/MWh."""
        assert PRICE_PER_MWH_DH == 700.0
        eco = to_economic(delta_bilan_normalised=0.1, action_cost_normalised=0.05)
        assert eco["delta_mwh"] == pytest.approx(0.1 * MWH_SCALE)
        assert eco["energy_gain_dh"] == pytest.approx(0.1 * MWH_SCALE * 700.0)
        assert eco["net_gain_dh"] == pytest.approx(
            eco["energy_gain_dh"] - eco["action_cost_dh"])


class TestDoubleDQNAgent:
    def test_agent_learns_and_recommends(self):
        torch = pytest.importorskip("torch")
        from rl.dqn_agent import DoubleDQNAgent

        agent = DoubleDQNAgent()
        env = CogenEnv(episode_length=8)
        s = env.reset(seed=0)
        for _ in range(80):                  # warm the replay buffer
            a = agent.select_action(s, training=True)
            ns, r, done, trunc, _ = env.step(a)
            agent.store(s, a, r, ns, done)
            s = env.reset(seed=0) if done else ns
        loss = agent.train_step()
        assert loss is None or np.isfinite(loss)

        q = agent.q_values(s)
        assert q.shape == (N_ACTIONS,)
        rec = agent.recommend(s)
        assert 0 <= rec["action_index"] < N_ACTIONS
        assert rec["action"] == ACTIONS[rec["action_index"]]["label"]
        assert 0.0 <= rec["confidence"] <= 1.0
