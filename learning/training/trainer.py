"""
RL training pipeline for manipulation subtasks.

Provides a clean training loop using stable-baselines3 (PPO baseline)
with logging to TensorBoard and optional W&B. Includes evaluation
against heuristic baselines.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    # Algorithm
    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    # Network architecture
    policy_net: list[int] = field(default_factory=lambda: [256, 256])
    value_net: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "ReLU"

    # Environment
    n_envs: int = 8  # number of parallel environments
    env_name: str = "ChessGraspEnv"

    # Evaluation
    eval_freq_steps: int = 50_000
    eval_episodes: int = 100

    # Logging and saving
    log_dir: str = "logs/training"
    save_dir: str = "models"
    save_freq_steps: int = 100_000
    use_wandb: bool = False
    wandb_project: str = "chess-robotic-rl"
    experiment_name: str = ""

    # Curriculum
    use_curriculum: bool = False
    curriculum_stages: list[dict] = field(default_factory=list)


class Trainer:
    """
    Training manager for RL policies.

    Usage:
        trainer = Trainer(TrainingConfig())
        trainer.train()
        trainer.evaluate()
        trainer.export_policy("models/grasp_policy.pt")
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._model = None
        self._env = None
        self._eval_env = None

    def setup(self) -> None:
        """Initialize environments and model."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
            from stable_baselines3.common.callbacks import (
                EvalCallback,
                CheckpointCallback,
            )
        except ImportError:
            raise ImportError(
                "stable-baselines3 required for training. "
                "Install with: pip install 'chess-robotic[learning]'"
            )

        from learning.envs.grasp_env import ChessGraspEnv

        # Create vectorized training environment
        def make_env():
            def _init():
                return ChessGraspEnv()
            return _init

        if self.config.n_envs > 1:
            self._env = SubprocVecEnv([make_env() for _ in range(self.config.n_envs)])
        else:
            self._env = DummyVecEnv([make_env()])

        # Create eval environment
        self._eval_env = DummyVecEnv([make_env()])

        # Create log directories
        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Initialize W&B if requested
        if self.config.use_wandb:
            try:
                import wandb
                exp_name = self.config.experiment_name or f"train_{int(time.time())}"
                wandb.init(
                    project=self.config.wandb_project,
                    name=exp_name,
                    config=vars(self.config),
                )
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")

        # Create PPO model
        policy_kwargs = dict(
            net_arch=dict(
                pi=self.config.policy_net,
                vf=self.config.value_net,
            ),
        )

        self._model = PPO(
            "MlpPolicy",
            self._env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.log_dir,
            verbose=1,
        )

        logger.info(
            f"Training setup complete: {self.config.algorithm}, "
            f"{self.config.n_envs} envs, {self.config.total_timesteps} steps"
        )

    def train(self) -> dict[str, Any]:
        """
        Run the training loop.

        Returns:
            Dict with training summary (final metrics, best model path, etc.)
        """
        if self._model is None:
            self.setup()

        from stable_baselines3.common.callbacks import (
            EvalCallback,
            CheckpointCallback,
        )

        # Setup callbacks
        eval_callback = EvalCallback(
            self._eval_env,
            best_model_save_path=self.config.save_dir,
            log_path=self.config.log_dir,
            eval_freq=self.config.eval_freq_steps // self.config.n_envs,
            n_eval_episodes=self.config.eval_episodes,
            deterministic=True,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq_steps // self.config.n_envs,
            save_path=self.config.save_dir,
            name_prefix="checkpoint",
        )

        logger.info("Starting training...")
        t0 = time.time()

        self._model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
        )

        training_time = time.time() - t0
        logger.info(f"Training complete in {training_time:.0f}s")

        # Save final model
        final_path = Path(self.config.save_dir) / "final_model"
        self._model.save(str(final_path))
        logger.info(f"Final model saved to {final_path}")

        return {
            "training_time_s": training_time,
            "total_timesteps": self.config.total_timesteps,
            "final_model_path": str(final_path),
        }

    def evaluate(
        self,
        n_episodes: int = 100,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate the trained model.

        Returns:
            Dict with success_rate, mean_reward, mean_episode_length, etc.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call setup() or load() first.")

        from learning.envs.grasp_env import ChessGraspEnv

        env = ChessGraspEnv()
        successes = 0
        total_reward = 0.0
        total_steps = 0

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                action, _ = self._model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1

            if info.get("lifted", False):
                successes += 1
            total_reward += ep_reward
            total_steps += ep_steps

        env.close()

        results = {
            "success_rate": successes / n_episodes,
            "mean_reward": total_reward / n_episodes,
            "mean_episode_length": total_steps / n_episodes,
            "n_episodes": n_episodes,
        }

        logger.info(
            f"Evaluation: success_rate={results['success_rate']:.2%}, "
            f"mean_reward={results['mean_reward']:.2f}, "
            f"mean_ep_len={results['mean_episode_length']:.0f}"
        )
        return results

    def evaluate_against_baseline(
        self,
        n_episodes: int = 100,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate the trained model against the heuristic baseline.

        Returns:
            Dict with "learned" and "heuristic" sub-dicts.
        """
        learned_results = self.evaluate(n_episodes)

        # Evaluate heuristic baseline
        from learning.envs.grasp_env import ChessGraspEnv
        from learning.heuristic_baselines import HeuristicGraspPolicy

        env = ChessGraspEnv()
        heuristic = HeuristicGraspPolicy()
        successes = 0
        total_reward = 0.0
        total_steps = 0

        for ep in range(n_episodes):
            obs_array, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                # Convert obs array to dict for heuristic policy
                obs_dict = {
                    "piece_rel_xy": obs_array[:2],
                    "piece_rel_z": float(obs_array[2]),
                    "gripper_width": float(obs_array[20]),
                    "piece_grasped": info.get("grasped", False),
                }
                action_dict = heuristic.predict(obs_dict)

                # Convert action dict to array for env
                action = np.zeros(7, dtype=np.float32)
                action[:6] = action_dict["delta_ee"]
                action[6] = action_dict["gripper_cmd"]

                obs_array, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1

            if info.get("lifted", False):
                successes += 1
            total_reward += ep_reward
            total_steps += ep_steps

        env.close()

        heuristic_results = {
            "success_rate": successes / n_episodes,
            "mean_reward": total_reward / n_episodes,
            "mean_episode_length": total_steps / n_episodes,
        }

        comparison = {
            "learned": learned_results,
            "heuristic": heuristic_results,
        }

        logger.info(
            f"Comparison: learned={learned_results['success_rate']:.2%} "
            f"vs heuristic={heuristic_results['success_rate']:.2%}"
        )
        return comparison

    def export_torchscript(self, output_path: str) -> None:
        """Export the trained policy as TorchScript for deployment."""
        if self._model is None:
            raise RuntimeError("No model loaded.")

        import torch

        # Extract the policy network
        policy = self._model.policy
        policy.eval()

        # Create dummy input
        dummy = torch.randn(1, 28)

        # Trace the policy
        traced = torch.jit.trace(policy.action_net, dummy)
        traced.save(output_path)
        logger.info(f"TorchScript policy exported to {output_path}")

    def cleanup(self) -> None:
        """Clean up environments."""
        if self._env is not None:
            self._env.close()
        if self._eval_env is not None:
            self._eval_env.close()
