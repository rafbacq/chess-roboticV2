#!/usr/bin/env python3
"""
Policy evaluation script: systematically compare heuristic vs random baselines
on both RL environments (ChessGraspEnv and ChessPlacementEnv).

Usage:
    python scripts/evaluate_policies.py
    python scripts/evaluate_policies.py --episodes 200 --env grasp
    python scripts/evaluate_policies.py --env placement --episodes 50
"""

import argparse
import logging
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from learning.envs.grasp_env import ChessGraspEnv, GraspEnvConfig
from learning.envs.placement_env import ChessPlacementEnv
from learning.eval_harness import EvalHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# =========================================================================
# Grasp Policies
# =========================================================================

def grasp_heuristic(obs: np.ndarray) -> np.ndarray:
    """Move toward piece, close gripper when near."""
    action = np.zeros(7, dtype=np.float32)
    if len(obs) >= 3:
        action[0:3] = np.clip(-obs[0:3] * 2.0, -1, 1)
    if len(obs) >= 3 and np.linalg.norm(obs[0:3]) < 0.15:
        action[6] = 1.0  # close
    return action


def grasp_random(obs: np.ndarray) -> np.ndarray:
    """Random actions in grasp env."""
    return np.random.uniform(-1, 1, size=7).astype(np.float32)


# =========================================================================
# Placement Policies
# =========================================================================

def placement_heuristic(obs: np.ndarray) -> np.ndarray:
    """Move toward target center, descend, release when low and centered."""
    action = np.zeros(4, dtype=np.float32)
    # offset_xy is obs[0:2]
    offset = obs[0:2]
    action[0:2] = np.clip(-offset * 5.0, -1, 1)

    # height is obs[2]
    if abs(offset[0]) < 0.002 and abs(offset[1]) < 0.002:
        action[2] = -0.5  # descend
        if obs[2] < 0.005:
            action[3] = -1.0  # release
    return action


def placement_random(obs: np.ndarray) -> np.ndarray:
    return np.random.uniform(-1, 1, size=4).astype(np.float32)


# =========================================================================
# Main
# =========================================================================

def evaluate_grasp(n_episodes: int):
    print("\n" + "=" * 70)
    print("  Chess Grasp Environment — Policy Evaluation")
    print("=" * 70)

    harness = EvalHarness(GraspEnvConfig(max_steps=100))

    results = []
    for label, fn in [("heuristic", grasp_heuristic), ("random", grasp_random)]:
        print(f"\n  Evaluating '{label}' policy ({n_episodes} episodes)...")
        t0 = time.time()
        result = harness.evaluate(fn, label=label, n_episodes=n_episodes)
        elapsed = time.time() - t0
        print(f"    reward={result.mean_reward:.3f} ± {result.std_reward:.3f}, "
              f"success={result.success_rate:.1%}, time={elapsed:.1f}s")
        results.append(result)

    harness.print_comparison(results)


def evaluate_placement(n_episodes: int):
    print("\n" + "=" * 70)
    print("  Chess Placement Environment — Policy Evaluation")
    print("=" * 70)

    env = ChessPlacementEnv()
    for label, fn in [("heuristic", placement_heuristic), ("random", placement_random)]:
        print(f"\n  Evaluating '{label}' policy ({n_episodes} episodes)...")
        rewards = []
        successes = 0
        t0 = time.time()

        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            done = False
            total_reward = 0.0
            while not done:
                action = fn(obs)
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                done = term or trunc
            rewards.append(total_reward)
            if info.get("placed", False):
                successes += 1

        elapsed = time.time() - t0
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"    reward={mean_r:.3f} ± {std_r:.3f}, "
              f"success={successes / n_episodes:.1%}, time={elapsed:.1f}s")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies on RL environments")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--env", choices=["grasp", "placement", "both"], default="both")
    args = parser.parse_args()

    if args.env in ("grasp", "both"):
        evaluate_grasp(args.episodes)
    if args.env in ("placement", "both"):
        evaluate_placement(args.episodes)

    print("\nDone.")


if __name__ == "__main__":
    main()
