"""
Unit tests for the RL grasp environment and reward functions.
"""

import numpy as np
import pytest

from learning.envs.grasp_env import ChessGraspEnv, GraspEnvConfig
from learning.envs.rewards import (
    CollisionPenalty,
    CompositeReward,
    ContactReward,
    DistanceShapingReward,
    StepPenalty,
    SuccessReward,
)


class TestGraspEnv:
    """Tests for the ChessGraspEnv Gymnasium environment."""

    @pytest.fixture
    def env(self):
        env = ChessGraspEnv(GraspEnvConfig(max_steps=50))
        yield env
        env.close()

    def test_observation_shape(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (28,), f"Expected (28,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_action_space(self, env):
        assert env.action_space.shape == (7,)

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert not np.any(np.isnan(obs))
        assert "piece_type" in info
        assert "step" in info

    def test_step_returns_correct_types(self, env):
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)

        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info2, dict)

    def test_truncation_on_max_steps(self, env):
        obs, _ = env.reset(seed=42)
        # Zero action should not cause early termination
        for i in range(50):
            obs, reward, terminated, truncated, info = env.step(np.zeros(7))
            if terminated or truncated:
                break
        assert truncated or terminated  # should truncate at max_steps=50

    def test_reproducible_with_seed(self, env):
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_different_obs(self, env):
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert not np.array_equal(obs1, obs2)

    def test_render_rgb_array(self):
        env = ChessGraspEnv(render_mode="rgb_array")
        env.reset(seed=42)
        img = env.render()
        assert img is not None
        assert img.shape == (200, 200, 3)
        assert img.dtype == np.uint8
        env.close()

    def test_info_keys(self, env):
        obs, info = env.reset(seed=42)
        required_keys = {"step", "piece_type", "contacted", "grasped",
                        "lifted", "lift_stable_steps", "gripper_width",
                        "distance_to_piece"}
        assert required_keys.issubset(info.keys())

    def test_gripper_closes_on_positive_action(self, env):
        obs, _ = env.reset(seed=42)
        # Action with positive gripper command should decrease width
        action = np.zeros(7)
        action[6] = 1.0  # close
        _, _, _, _, info = env.step(action)
        assert info["gripper_width"] < 1.0


class TestRewardComponents:
    """Tests for individual reward components."""

    def test_success_reward(self):
        r = SuccessReward(10.0)
        assert r.compute({"success": True}) == 10.0
        assert r.compute({"success": False}) == 0.0

    def test_step_penalty(self):
        r = StepPenalty(-0.5)
        assert r.compute({}) == -0.5

    def test_contact_reward_once(self):
        r = ContactReward(2.0)
        assert r.compute({"contacted": False}) == 0.0
        assert r.compute({"contacted": True}) == 2.0
        # Second contact should give 0 (one-time reward)
        assert r.compute({"contacted": True}) == 0.0

    def test_contact_reward_reset(self):
        r = ContactReward(2.0)
        r.compute({"contacted": True})
        r.reset()
        assert r.compute({"contacted": True}) == 2.0

    def test_collision_penalty(self):
        r = CollisionPenalty(-5.0)
        assert r.compute({"collision": True}) == -5.0
        assert r.compute({"collision": False}) == 0.0

    def test_distance_shaping(self):
        r = DistanceShapingReward(1.0)
        # First call: no previous distance, return 0
        assert r.compute({"distance_to_piece": 0.1}) == 0.0
        # Second call: distance decreased → positive reward
        reward = r.compute({"distance_to_piece": 0.05})
        assert reward > 0
        # Third call: distance increased → negative reward
        reward = r.compute({"distance_to_piece": 0.08})
        assert reward < 0


class TestCompositeReward:
    def test_composition(self):
        composite = CompositeReward([
            SuccessReward(10.0),
            StepPenalty(-0.5),
        ])

        info = {"success": True}
        total = composite.compute(info)
        assert total == 9.5

    def test_breakdown(self):
        composite = CompositeReward([
            SuccessReward(10.0),
            StepPenalty(-0.5),
        ])

        breakdown = composite.get_breakdown({"success": False})
        assert breakdown["success"] == 0.0
        assert breakdown["step_penalty"] == -0.5

    def test_reset(self):
        contact = ContactReward(2.0)
        composite = CompositeReward([contact, StepPenalty(-0.5)])

        composite.compute({"contacted": True})
        assert contact.compute({"contacted": True}) == 0.0  # already given

        composite.reset()
        assert contact.compute({"contacted": True}) == 2.0  # reset, gives again
