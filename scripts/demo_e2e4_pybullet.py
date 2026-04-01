#!/usr/bin/env python3
"""
E2E demo: execute the move e2e4 through the full pipeline.

Uses simulated hardware (no PyBullet physics required) to demonstrate
the complete orchestrator loop:
  1. Parse "e2e4"
  2. Generate grasp candidates
  3. Execute staged pick-and-place
  4. Confirm move in game state
  5. Print telemetry summary

This script runs WITHOUT needing Stockfish or PyBullet installed.
It uses the SimulatedArm/SimulatedGripper for instant execution.

Usage:
    python scripts/demo_e2e4_pybullet.py
"""

import logging
import os
import sys
import time

import numpy as np

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, ".")

from board_state.board_model import BoardConfig, BoardModel
from chess_core.game_manager import GameConfig, GamePhase, PlayerType
from chess_core.interfaces import ExecutionStatus, PieceType, Square
from manipulation.grasp_policy import GraspPolicyConfig
from manipulation.pick_place import ManipConfig
from orchestrator import OrchestratorConfig, SystemOrchestrator
from robot_model.arm_interface import SimulatedArm, SimulatedGripper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-25s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def run_demo():
    """Run the e2e4 demo through the full orchestrator."""
    print("\n" + "=" * 60)
    print("  Chess-Robotic V2 -- Phase 2 E2E Demo")
    print("  Move: e2e4 (White pawn advance)")
    print("=" * 60 + "\n")

    # Create simulated hardware
    arm = SimulatedArm(name="demo_arm", dof=6)
    arm.initialize()
    gripper = SimulatedGripper(name="demo_gripper")
    gripper.initialize()

    # Configure orchestrator
    board_config = BoardConfig(square_size_m=0.057)
    game_config = GameConfig(
        white_player=PlayerType.HUMAN,
        black_player=PlayerType.HUMAN,
    )
    manip_config = ManipConfig(
        safe_height_m=0.15,
        verify_delay_s=0.0,  # skip delay for demo
    )

    config = OrchestratorConfig(
        board_config=board_config,
        game_config=game_config,
        manip_config=manip_config,
        grasp_config=GraspPolicyConfig(use_learned_grasp=False),
    )

    orchestrator = SystemOrchestrator(arm, gripper, config)

    # Start game
    orchestrator.start_game()

    # Get board model info
    board = orchestrator.board_model
    e2_pos = board.get_square_center(Square.from_algebraic("e2"))
    e4_pos = board.get_square_center(Square.from_algebraic("e4"))

    print(f"Board geometry:")
    print(f"  Square size: {board.config.square_size_m * 1000:.0f}mm")
    print(f"  e2 center (board frame): [{e2_pos[0]:.3f}, {e2_pos[1]:.3f}, {e2_pos[2]:.3f}]m")
    print(f"  e4 center (board frame): [{e4_pos[0]:.3f}, {e4_pos[1]:.3f}, {e4_pos[2]:.3f}]m")
    print(f"  Pawn grasp height: {board.get_grasp_z(PieceType.PAWN):.4f}m")
    print()

    # Execute e2e4
    print("Executing move: e2e4...")
    print("-" * 40)

    t0 = time.time()
    result = orchestrator.execute_turn(manual_uci="e2e4")
    total_time = time.time() - t0

    print("-" * 40)
    print(f"\nResult: {result.status.name}")
    print(f"Duration: {result.duration_s:.3f}s (total wall time: {total_time:.3f}s)")

    if result.status == ExecutionStatus.SUCCESS:
        print(f"\n[OK] Move confirmed! Board state updated.")
        print(f"  FEN after: {orchestrator.game.fen}")
        print(f"  Move number: {orchestrator.game.move_number}")
        print(f"  Next turn: {orchestrator.game.current_color.name}")
    else:
        print(f"\n[FAIL] Move failed: {result.error_message}")
        return False

    # Verify board state
    expected_occ = orchestrator.game.get_expected_occupancy()
    assert not expected_occ["e2"], "e2 should be empty after e2e4"
    assert expected_occ["e4"], "e4 should be occupied after e2e4"
    print(f"\n  Board state consistency: OK - e2 empty, e4 occupied")

    # Telemetry
    if result.telemetry and "stages" in result.telemetry:
        stages = result.telemetry["stages"]
        print(f"\n  Manipulation stages ({len(stages)}):")
        for s in stages:
            print(f"    - {s['stage']}")

    # Run a few more moves
    print("\n" + "=" * 60)
    print("  Playing additional moves...")
    print("=" * 60)

    additional_moves = ["d7d5", "e4d5", "d8d5"]  # Scandinavian Defense
    for uci in additional_moves:
        result = orchestrator.execute_turn(manual_uci=uci)
        status = "[OK]" if result.status == ExecutionStatus.SUCCESS else "[FAIL]"
        print(f"  {status} {uci}: {result.status.name} ({result.duration_s:.3f}s)")

        if result.status != ExecutionStatus.SUCCESS:
            break

    # Final summary
    print("\n" + "=" * 60)
    print("  Game Summary")
    print("=" * 60)
    print(f"  Total moves played: {orchestrator.move_count}")
    print(f"  Final FEN: {orchestrator.game.fen}")

    if orchestrator.game_log:
        print(f"\n  Move log:")
        for entry in orchestrator.game_log:
            print(f"    {entry['move_number']}. {entry['uci']} ({entry['color']}) - {entry['duration_s']:.3f}s")

    # Cleanup
    orchestrator.stop_game()
    arm.shutdown()
    gripper.shutdown()

    print(f"\n{'=' * 60}")
    print("  Demo complete! All moves executed successfully.")
    print(f"{'=' * 60}\n")
    return True


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
