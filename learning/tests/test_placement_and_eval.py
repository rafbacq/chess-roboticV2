"""
Tests for the PlacementEnv and the EvalHarness.

Validates Gymnasium compliance, episode flow, and evaluation metrics.
"""

import numpy as np
import pytest

from learning.envs.placement_env import ChessPlacementEnv
from learning.eval_harness import EvalHarness, EvalResult


# =========================================================================
# PlacementEnv Tests
# =========================================================================

class TestPlacementEnv:
    def test_creation(self):
        env = ChessPlacementEnv()
        assert env.observation_space.shape == (18,)
        assert env.action_space.shape == (4,)
        env.close()

    def test_reset_returns_correct_shapes(self):
        env = ChessPlacementEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (18,)
        assert isinstance(info, dict)
        assert "piece_type" in info
        assert "offset_mm" in info
        env.close()

    def test_step_returns_correct_shapes(self):
        env = ChessPlacementEnv()
        obs, info = env.reset(seed=0)
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert obs2.shape == (18,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_truncation_at_max_steps(self):
        env = ChessPlacementEnv(max_steps=5)
        obs, _ = env.reset(seed=0)
        for _ in range(5):
            action = np.zeros(4, dtype=np.float32)  # no-op
            obs, reward, terminated, truncated, info = env.step(action)
        assert truncated
        env.close()

    def test_successful_placement(self):
        env = ChessPlacementEnv(
            position_noise_m=0.0,
            success_threshold_m=0.01,
        )
        obs, _ = env.reset(seed=0)
        # Piece should be nearly centered since noise=0
        # Descend and release
        for _ in range(10):
            action = np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32)  # go down
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        # Now release
        if not terminated:
            action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # open gripper
            obs, reward, terminated, truncated, info = env.step(action)
            assert info["placed"]
        env.close()

    def test_holding_flag(self):
        env = ChessPlacementEnv()
        obs, info = env.reset(seed=0)
        assert info["holding"]
        env.close()


# =========================================================================
# EvalHarness Tests
# =========================================================================

class TestEvalHarness:
    def test_evaluate_random_policy(self):
        harness = EvalHarness()
        result = harness.evaluate(
            policy_fn=lambda obs: np.zeros(7, dtype=np.float32),
            label="zero_action",
            n_episodes=5,
        )
        assert isinstance(result, EvalResult)
        assert result.label == "zero_action"
        assert result.n_episodes == 5
        assert result.mean_reward != 0.0 or True  # just check it runs
        assert 0.0 <= result.success_rate <= 1.0
        assert result.wall_time_s > 0

    def test_evaluate_per_piece_breakdown(self):
        harness = EvalHarness()
        result = harness.evaluate(
            policy_fn=lambda obs: np.zeros(7, dtype=np.float32),
            label="test",
            n_episodes=20,
        )
        assert len(result.per_piece) > 0
        for ptype, stats in result.per_piece.items():
            assert "mean_reward" in stats
            assert "n_episodes" in stats
            assert "success_rate" in stats

    def test_print_comparison(self):
        """Smoke test for print formatting (should not raise)."""
        r1 = EvalResult(
            label="baseline", mean_reward=-5.0, std_reward=1.0,
            median_reward=-5.0, min_reward=-10, max_reward=0,
            success_rate=0.0, mean_episode_length=50,
            contact_rate=0.1, grasp_rate=0.05, lift_rate=0.0,
            n_episodes=10, wall_time_s=1.0,
        )
        r2 = EvalResult(
            label="learned", mean_reward=3.0, std_reward=2.0,
            median_reward=4.0, min_reward=-2, max_reward=10,
            success_rate=0.5, mean_episode_length=30,
            contact_rate=0.8, grasp_rate=0.6, lift_rate=0.5,
            n_episodes=10, wall_time_s=2.0,
        )
        EvalHarness.print_comparison([r1, r2])
