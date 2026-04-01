"""
Chess-Robotic V2 CLI entry point.

Usage:
    python -m chess_robotic play                   # Start a game (human vs engine)
    python -m chess_robotic play --both-human       # Human vs human
    python -m chess_robotic demo                    # Run the e2e4 demo
    python -m chess_robotic train --steps 100000    # Train RL policy
    python -m chess_robotic eval                    # Evaluate policies
    python -m chess_robotic calibrate               # Run calibration
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_play(args: argparse.Namespace) -> int:
    """Run a chess game."""
    setup_logging(args.log_level)
    from chess_core.game_manager import PlayerType
    from system_factory import SystemFactory

    factory = SystemFactory(args.config)
    orchestrator = factory.build()

    # Override player types
    if args.both_human:
        orchestrator.game.config.white_player = PlayerType.HUMAN
        orchestrator.game.config.black_player = PlayerType.HUMAN
    elif args.engine_white:
        orchestrator.game.config.white_player = PlayerType.ENGINE
        orchestrator.game.config.black_player = PlayerType.HUMAN

    orchestrator.start_game()

    if args.both_human and args.moves:
        # Pre-programmed moves for automated play
        moves = args.moves.split(",")
        summary = orchestrator.run_game_loop(max_moves=args.max_moves, human_moves=moves)
    else:
        print("Interactive mode: enter UCI moves (e.g., 'e2e4')")
        print("Type 'quit' to stop.\n")
        while orchestrator.game.phase.name == "AWAITING_MOVE":
            if orchestrator.game.current_player_type == PlayerType.ENGINE:
                result = orchestrator.execute_turn()
            else:
                uci = input(f"[{orchestrator.game.current_color.name}] Your move: ").strip()
                if uci.lower() in ("quit", "exit", "q"):
                    break
                result = orchestrator.execute_turn(manual_uci=uci)
            print(f"  -> {result.status.name}")

    orchestrator.stop_game()
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the e2e4 demonstration."""
    setup_logging(args.log_level)

    # Import and run the demo directly
    sys.path.insert(0, ".")
    from scripts.demo_e2e4_pybullet import run_demo
    success = run_demo()
    return 0 if success else 1


def cmd_train(args: argparse.Namespace) -> int:
    """Train an RL policy."""
    setup_logging(args.log_level)
    sys.path.insert(0, ".")
    from scripts.run_first_training import run_training
    run_training(args.steps, args.eval_episodes, args.output_dir)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate RL policies."""
    setup_logging(args.log_level)

    import numpy as np
    from learning.eval_harness import EvalHarness
    from learning.envs.grasp_env import GraspEnvConfig

    env_config = GraspEnvConfig(max_steps=50)
    harness = EvalHarness(env_config)

    results = []

    # Heuristic baseline
    def heuristic(obs):
        action = np.zeros(7, dtype=np.float32)
        if len(obs) >= 3:
            action[0:3] = np.clip(-obs[0:3] * 2.0, -1, 1)
        if len(obs) >= 3 and np.linalg.norm(obs[0:3]) < 0.3:
            action[6] = 1.0
        return action

    print("Evaluating heuristic baseline...")
    results.append(harness.evaluate(heuristic, "heuristic", n_episodes=args.episodes))

    # Random policy
    def random_policy(obs):
        return np.random.uniform(-1, 1, 7).astype(np.float32)

    print("Evaluating random baseline...")
    results.append(harness.evaluate(random_policy, "random", n_episodes=args.episodes))

    # Learned policy (if model exists)
    try:
        from stable_baselines3 import PPO
        from pathlib import Path

        model_path = Path(args.model_path)
        if model_path.exists():
            model = PPO.load(str(model_path))
            print(f"Evaluating learned policy from {model_path}...")
            results.append(harness.evaluate(
                lambda obs: model.predict(obs, deterministic=True)[0],
                "ppo",
                n_episodes=args.episodes,
            ))
    except ImportError:
        pass
    except Exception as e:
        print(f"  Could not load model: {e}")

    harness.print_comparison(results)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run camera calibration."""
    setup_logging(args.log_level)

    from calibration.calibrator import (
        CalibrationConfig,
        CameraCalibrator,
        build_calibration_bundle,
    )

    config = CalibrationConfig()
    calibrator = CameraCalibrator(config)

    if args.synthetic:
        print("Running synthetic calibration test...")
        frames = calibrator.generate_synthetic_frames(n_frames=20)
        detected = 0
        for frame in frames:
            if calibrator.add_frame(frame):
                detected += 1
        print(f"  Detected checkerboard in {detected}/{len(frames)} frames")

        result = calibrator.calibrate()
        if result:
            print(f"  RMS reprojection error: {result['reprojection_error']:.4f}px")
            print("  Intrinsic calibration succeeded!")
        else:
            print("  Calibration failed (insufficient detections)")
    else:
        print("Live calibration requires a camera connection.")
        print("Use --synthetic for testing without camera.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="chess-robotic",
        description="Chess-Robotic V2: AI-powered robotic chess system",
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # play
    p_play = sub.add_parser("play", help="Start a chess game")
    p_play.add_argument("--both-human", action="store_true", help="Both players human")
    p_play.add_argument("--engine-white", action="store_true", help="Engine plays white")
    p_play.add_argument("--moves", type=str, default="", help="Comma-separated UCI moves")
    p_play.add_argument("--max-moves", type=int, default=200, help="Max total moves")

    # demo
    sub.add_parser("demo", help="Run the e2e4 demonstration")

    # train
    p_train = sub.add_parser("train", help="Train RL policy")
    p_train.add_argument("--steps", type=int, default=50000, help="Training steps")
    p_train.add_argument("--eval-episodes", type=int, default=100)
    p_train.add_argument("--output-dir", default="logs/training")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate policies")
    p_eval.add_argument("--episodes", type=int, default=200, help="Eval episodes per policy")
    p_eval.add_argument("--model-path", default="logs/first_training/ppo_grasp_model.zip")

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Run calibration")
    p_cal.add_argument("--synthetic", action="store_true", help="Use synthetic test images")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "play": cmd_play,
        "demo": cmd_demo,
        "train": cmd_train,
        "eval": cmd_eval,
        "calibrate": cmd_calibrate,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
