"""
Tests for the system orchestrator.

Validates the complete pipeline using simulated hardware:
  - Single move execution
  - Multi-move game loop
  - Failure recovery
  - Board state consistency
"""

import chess
import numpy as np
import pytest

from board_state.board_model import BoardConfig
from chess_core.game_manager import GameConfig, GamePhase, PlayerType
from chess_core.interfaces import ExecutionStatus
from manipulation.grasp_policy import GraspPolicyConfig
from manipulation.pick_place import ManipConfig
from orchestrator import OrchestratorConfig, SystemOrchestrator
from robot_model.arm_interface import SimulatedArm, SimulatedGripper


@pytest.fixture
def orchestrator():
    """Create an orchestrator with simulated hardware."""
    arm = SimulatedArm(name="test_arm", dof=6)
    arm.initialize()
    gripper = SimulatedGripper(name="test_gripper")
    gripper.initialize()

    config = OrchestratorConfig(
        board_config=BoardConfig(square_size_m=0.057),
        game_config=GameConfig(
            white_player=PlayerType.HUMAN,
            black_player=PlayerType.HUMAN,
        ),
        manip_config=ManipConfig(verify_delay_s=0.0),
        grasp_config=GraspPolicyConfig(use_learned_grasp=False),
    )

    orch = SystemOrchestrator(arm, gripper, config)
    orch.start_game()
    yield orch
    orch.stop_game()


class TestOrchestratorSingleMove:
    """Test single move execution through the orchestrator."""

    def test_e2e4(self, orchestrator):
        result = orchestrator.execute_turn(manual_uci="e2e4")
        assert result.status == ExecutionStatus.SUCCESS
        assert orchestrator.move_count == 1
        assert orchestrator.game.phase == GamePhase.AWAITING_MOVE

        # Board state
        occ = orchestrator.game.get_expected_occupancy()
        assert not occ["e2"]
        assert occ["e4"]

    def test_d7d5(self, orchestrator):
        # First play e2e4
        orchestrator.execute_turn(manual_uci="e2e4")

        # Then d7d5
        result = orchestrator.execute_turn(manual_uci="d7d5")
        assert result.status == ExecutionStatus.SUCCESS
        assert orchestrator.move_count == 2

    def test_invalid_move_rejected(self, orchestrator):
        result = orchestrator.execute_turn(manual_uci="e2e5")  # illegal
        assert result.status == ExecutionStatus.ILLEGAL_MOVE

    def test_no_uci_for_human(self, orchestrator):
        result = orchestrator.execute_turn(manual_uci="")
        assert result.status == ExecutionStatus.ILLEGAL_MOVE


class TestOrchestratorGameLoop:
    """Test multi-move game loop."""

    def test_4_move_game(self, orchestrator):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
        summary = orchestrator.run_game_loop(max_moves=10, human_moves=moves)

        assert summary["total_moves"] == 4
        assert len(summary["move_log"]) == 4
        assert summary["result"] == "in_progress"

    def test_scholars_mate(self, orchestrator):
        """Scholar's Mate in 4 moves."""
        moves = [
            "e2e4", "e7e5",
            "d1h5", "b8c6",
            "f1c4", "g8f6",
            "h5f7",  # checkmate!
        ]
        summary = orchestrator.run_game_loop(max_moves=10, human_moves=moves)

        assert summary["total_moves"] == 7
        assert summary["result"] == "1-0"
        assert orchestrator.game.phase == GamePhase.GAME_OVER

    def test_capture_move(self, orchestrator):
        # Scandinavian Defense: 1.e4 d5 2.exd5
        moves = ["e2e4", "d7d5", "e4d5"]
        summary = orchestrator.run_game_loop(max_moves=10, human_moves=moves)
        assert summary["total_moves"] == 3

        occ = orchestrator.game.get_expected_occupancy()
        assert not occ["e4"], "e4 should be empty after exd5"
        assert occ["d5"], "d5 should be occupied by white pawn"

    def test_castling(self, orchestrator):
        # Play to enable kingside castling
        moves = [
            "e2e4", "e7e5",
            "g1f3", "b8c6",
            "f1e2", "d7d6",
            "e1g1",  # castle kingside
        ]
        summary = orchestrator.run_game_loop(max_moves=10, human_moves=moves)
        assert summary["total_moves"] == 7

        occ = orchestrator.game.get_expected_occupancy()
        assert occ["g1"], "King should be on g1"
        assert occ["f1"], "Rook should be on f1"
        assert not occ["e1"], "e1 should be empty"
        assert not occ["h1"], "h1 should be empty"


class TestOrchestratorState:

    def test_game_log_populated(self, orchestrator):
        orchestrator.execute_turn(manual_uci="e2e4")
        log = orchestrator.game_log
        assert len(log) == 1
        assert log[0]["uci"] == "e2e4"
        assert log[0]["color"] == "WHITE"
        assert log[0]["status"] == "SUCCESS"

    def test_move_count_increments(self, orchestrator):
        assert orchestrator.move_count == 0
        orchestrator.execute_turn(manual_uci="e2e4")
        assert orchestrator.move_count == 1
        orchestrator.execute_turn(manual_uci="e7e5")
        assert orchestrator.move_count == 2
