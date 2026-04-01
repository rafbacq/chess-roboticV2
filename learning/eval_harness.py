"""
RL evaluation harness: systematic comparison of learned vs heuristic policies.

Provides standardized evaluation with:
  - Multiple random seeds for statistical significance
  - Per-piece-type breakdown
  - Success rate, reward, episode length metrics
  - Formatted results table
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from learning.envs.grasp_env import ChessGraspEnv, GraspEnvConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Results from evaluating a single policy."""
    label: str
    mean_reward: float
    std_reward: float
    median_reward: float
    min_reward: float
    max_reward: float
    success_rate: float
    mean_episode_length: float
    contact_rate: float
    grasp_rate: float
    lift_rate: float
    n_episodes: int
    wall_time_s: float
    per_piece: dict = field(default_factory=dict)


class EvalHarness:
    """
    Systematic policy evaluation framework.

    Usage:
        harness = EvalHarness(env_config)
        result = harness.evaluate(policy_fn, label="ppo", n_episodes=200)
        harness.print_comparison([result_heuristic, result_ppo])
    """

    def __init__(self, env_config: GraspEnvConfig | None = None) -> None:
        self.env_config = env_config or GraspEnvConfig()

    def evaluate(
        self,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        label: str = "policy",
        n_episodes: int = 200,
        seed_offset: int = 0,
    ) -> EvalResult:
        """
        Evaluate a policy over multiple episodes.

        Args:
            policy_fn: Function mapping observation -> action.
            label: Name for this policy.
            n_episodes: Number of evaluation episodes.
            seed_offset: Starting seed for reproducibility.

        Returns:
            EvalResult with comprehensive metrics.
        """
        env = ChessGraspEnv(self.env_config)

        rewards = []
        ep_lengths = []
        successes = 0
        contacts = 0
        grasps = 0
        lifts = 0
        piece_stats: dict[str, list[float]] = {}

        t0 = time.time()

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed_offset + ep)
            piece_type = info.get("piece_type", "UNKNOWN")
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = policy_fn(obs)
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                steps += 1
                done = term or trunc

            rewards.append(total_reward)
            ep_lengths.append(steps)

            if info.get("contacted", False):
                contacts += 1
            if info.get("grasped", False):
                grasps += 1
            if info.get("lifted", False):
                lifts += 1
                successes += 1

            piece_stats.setdefault(piece_type, []).append(total_reward)

        env.close()
        wall_time = time.time() - t0

        # Per-piece breakdown
        per_piece = {}
        for ptype, rews in piece_stats.items():
            per_piece[ptype] = {
                "mean_reward": float(np.mean(rews)),
                "n_episodes": len(rews),
                "success_rate": sum(1 for r in rews if r > 5.0) / len(rews),
            }

        return EvalResult(
            label=label,
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            median_reward=float(np.median(rewards)),
            min_reward=float(np.min(rewards)),
            max_reward=float(np.max(rewards)),
            success_rate=successes / n_episodes,
            mean_episode_length=float(np.mean(ep_lengths)),
            contact_rate=contacts / n_episodes,
            grasp_rate=grasps / n_episodes,
            lift_rate=lifts / n_episodes,
            n_episodes=n_episodes,
            wall_time_s=wall_time,
            per_piece=per_piece,
        )

    @staticmethod
    def print_comparison(results: list[EvalResult]) -> None:
        """Print a formatted comparison table."""
        print("\n" + "=" * 80)
        print("  POLICY COMPARISON")
        print("=" * 80)

        header = (
            f"{'Policy':<15} {'Reward':>10} {'Std':>8} "
            f"{'Success':>9} {'Contact':>9} {'Grasp':>9} "
            f"{'EpLen':>8} {'Time':>8}"
        )
        print(header)
        print("-" * 80)

        for r in results:
            print(
                f"{r.label:<15} {r.mean_reward:>10.3f} {r.std_reward:>8.3f} "
                f"{r.success_rate:>8.1%} {r.contact_rate:>8.1%} "
                f"{r.grasp_rate:>8.1%} {r.mean_episode_length:>8.1f} "
                f"{r.wall_time_s:>7.1f}s"
            )

        print("-" * 80)

        # Per-piece breakdown for best policy
        if results:
            best = max(results, key=lambda r: r.mean_reward)
            if best.per_piece:
                print(f"\nPer-piece breakdown ({best.label}):")
                for ptype, stats in sorted(best.per_piece.items()):
                    print(
                        f"  {ptype:<10} reward={stats['mean_reward']:>7.3f}  "
                        f"success={stats['success_rate']:>6.1%}  "
                        f"n={stats['n_episodes']}"
                    )

        print("=" * 80 + "\n")

    @staticmethod
    def print_result(result: EvalResult) -> None:
        """Print a single result."""
        EvalHarness.print_comparison([result])
