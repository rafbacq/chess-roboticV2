#!/usr/bin/env python3
"""
Script to train a grasp acquisition RL policy.

Usage:
    python -m learning.scripts.train_grasp_policy
    python -m learning.scripts.train_grasp_policy --timesteps 500000 --wandb
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train grasp RL policy")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                       help="Evaluation frequency in steps")
    parser.add_argument("--eval-episodes", type=int, default=100,
                       help="Episodes per evaluation")
    parser.add_argument("--save-dir", type=str, default="models/grasp",
                       help="Model save directory")
    parser.add_argument("--log-dir", type=str, default="logs/grasp_training",
                       help="TensorBoard log directory")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable W&B logging")
    parser.add_argument("--experiment-name", type=str, default="",
                       help="Experiment name for logging")
    parser.add_argument("--compare-baseline", action="store_true",
                       help="After training, compare against heuristic baseline")
    args = parser.parse_args()

    from learning.training.trainer import Trainer, TrainingConfig

    config = TrainingConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        eval_freq_steps=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        use_wandb=args.wandb,
        experiment_name=args.experiment_name,
    )

    trainer = Trainer(config)

    try:
        logger.info("=" * 60)
        logger.info("CHESS GRASP RL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Timesteps: {config.total_timesteps:,}")
        logger.info(f"Envs: {config.n_envs}")
        logger.info(f"LR: {config.learning_rate}")
        logger.info(f"Save dir: {config.save_dir}")
        logger.info("=" * 60)

        results = trainer.train()

        logger.info(f"\nTraining Summary:")
        logger.info(f"  Time: {results['training_time_s']:.0f}s")
        logger.info(f"  Model: {results['final_model_path']}")

        if args.compare_baseline:
            logger.info("\nComparing against heuristic baseline...")
            comparison = trainer.evaluate_against_baseline(n_episodes=args.eval_episodes)
            logger.info(f"  Learned: {comparison['learned']['success_rate']:.2%}")
            logger.info(f"  Heuristic: {comparison['heuristic']['success_rate']:.2%}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
