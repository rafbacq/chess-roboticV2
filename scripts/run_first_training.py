#!/usr/bin/env python3
"""
First RL training run: train a PPO agent on ChessGraspEnv and
compare performance against the heuristic baseline.

Runs a short training (configurable steps, default 50K for quick demo)
and evaluates both the learned policy and the heuristic baseline.

Usage:
    python scripts/run_first_training.py
    python scripts/run_first_training.py --steps 100000 --eval-episodes 50
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from learning.envs.grasp_env import ChessGraspEnv, GraspEnvConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("training")


def evaluate_policy(env, policy_fn, n_episodes: int = 100, label: str = "") -> dict:
    """
    Evaluate a policy on the environment.

    Args:
        env: Gymnasium environment.
        policy_fn: Callable(obs) -> action
        n_episodes: Number of evaluation episodes.
        label: Label for logging.

    Returns:
        Dict with statistics.
    """
    rewards = []
    successes = 0
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 1000)
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
        episode_lengths.append(steps)
        if info.get("is_success", False) or total_reward > 0.5:
            successes += 1

    stats = {
        "label": label,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "success_rate": successes / n_episodes,
        "mean_ep_length": float(np.mean(episode_lengths)),
        "n_episodes": n_episodes,
    }

    return stats


def heuristic_policy(obs: np.ndarray) -> np.ndarray:
    """
    Simple heuristic: move toward target position, keep gripper vertical.

    Observation layout (28-dim):
      [0:3] = EE position relative to target
      [3:6] = EE orientation (euler)
      [6:9] = gripper state
      [9:12] = target position
      [12:15] = piece dimensions
      [15:28] = proprioception noise / extras
    """
    action = np.zeros(7, dtype=np.float32)

    # Move toward piece (reduce position error)
    if len(obs) >= 9:
        pos_error = obs[0:3]  # approx offset
        action[0:3] = np.clip(-pos_error * 2.0, -1, 1)

    # Close gripper when close enough
    if len(obs) >= 3 and np.linalg.norm(obs[0:3]) < 0.3:
        action[6] = -1.0  # close

    return action


def run_training(total_steps: int, eval_episodes: int, output_dir: str):
    """Run the training and evaluation pipeline."""

    print("\n" + "=" * 60)
    print("  Chess-Robotic V2 — First RL Training Run")
    print(f"  Steps: {total_steps:,} | Eval episodes: {eval_episodes}")
    print("=" * 60 + "\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Environment config
    env_config = GraspEnvConfig(
        max_steps=50,
        piece_pose_noise_xy_m=0.005,
        piece_pose_noise_yaw_rad=0.02,
    )

    # =========================================================================
    # Step 1: Evaluate heuristic baseline
    # =========================================================================
    print("Phase 1: Evaluating heuristic baseline...")
    eval_env = ChessGraspEnv(env_config)
    t0 = time.time()
    heuristic_stats = evaluate_policy(
        eval_env, heuristic_policy, n_episodes=eval_episodes, label="heuristic"
    )
    heuristic_time = time.time() - t0
    eval_env.close()
    print(f"  Heuristic: reward={heuristic_stats['mean_reward']:.3f} ± {heuristic_stats['std_reward']:.3f}, "
          f"success={heuristic_stats['success_rate']:.1%}, time={heuristic_time:.1f}s")

    # =========================================================================
    # Step 2: Train PPO
    # =========================================================================
    print(f"\nPhase 2: Training PPO for {total_steps:,} steps...")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        # Create vectorized environment
        vec_env = make_vec_env(
            lambda: ChessGraspEnv(env_config),
            n_envs=4,
        )

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(output_path / "tb_logs"),
        )

        t0 = time.time()
        model.learn(
            total_timesteps=total_steps,
            progress_bar=True,
        )
        training_time = time.time() - t0

        # Save model
        model_path = str(output_path / "ppo_grasp_model")
        model.save(model_path)
        print(f"  Model saved to: {model_path}")
        print(f"  Training time: {training_time:.1f}s ({total_steps / training_time:.0f} steps/s)")

        vec_env.close()

        # =====================================================================
        # Step 3: Evaluate trained model
        # =====================================================================
        print(f"\nPhase 3: Evaluating trained PPO model...")

        eval_env = ChessGraspEnv(env_config)
        t0 = time.time()
        ppo_stats = evaluate_policy(
            eval_env,
            lambda obs: model.predict(obs, deterministic=True)[0],
            n_episodes=eval_episodes,
            label="ppo",
        )
        ppo_time = time.time() - t0
        eval_env.close()
        print(f"  PPO: reward={ppo_stats['mean_reward']:.3f} ± {ppo_stats['std_reward']:.3f}, "
              f"success={ppo_stats['success_rate']:.1%}, time={ppo_time:.1f}s")

    except ImportError:
        print("  ⚠ stable-baselines3 not installed. Skipping PPO training.")
        print("  Install with: pip install stable-baselines3")
        ppo_stats = None
        training_time = 0

    # =========================================================================
    # Step 4: Print comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    header = f"{'Policy':<15} {'Mean Reward':>12} {'Std':>8} {'Success':>10} {'Ep Length':>10}"
    print(header)
    print("-" * len(header))

    print(f"{'Heuristic':<15} {heuristic_stats['mean_reward']:>12.3f} "
          f"{heuristic_stats['std_reward']:>8.3f} "
          f"{heuristic_stats['success_rate']:>9.1%} "
          f"{heuristic_stats['mean_ep_length']:>10.1f}")

    if ppo_stats:
        print(f"{'PPO':<15} {ppo_stats['mean_reward']:>12.3f} "
              f"{ppo_stats['std_reward']:>8.3f} "
              f"{ppo_stats['success_rate']:>9.1%} "
              f"{ppo_stats['mean_ep_length']:>10.1f}")

        # Improvement
        if heuristic_stats['mean_reward'] != 0:
            improvement = ((ppo_stats['mean_reward'] - heuristic_stats['mean_reward'])
                          / abs(heuristic_stats['mean_reward'])) * 100
            print(f"\n  Reward improvement: {improvement:+.1f}%")

    # Save results
    results = {
        "total_steps": total_steps,
        "eval_episodes": eval_episodes,
        "training_time_s": training_time,
        "heuristic": heuristic_stats,
    }
    if ppo_stats:
        results["ppo"] = ppo_stats

    results_path = output_path / "training_results.npz"
    np.savez(str(results_path), **{k: str(v) for k, v in results.items()})

    print(f"\n  Results saved to: {output_path}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="First RL training run")
    parser.add_argument("--steps", type=int, default=50000,
                       help="Total training steps (default: 50K)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="logs/first_training",
                       help="Output directory")
    args = parser.parse_args()

    run_training(args.steps, args.eval_episodes, args.output_dir)


if __name__ == "__main__":
    main()
