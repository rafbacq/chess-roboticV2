"""
Integration test: execute move e2e4 through the full pipeline.

Tests the complete flow:
    1. Game manager validates the move
    2. Board model generates 3D poses
    3. Grasp policy generates candidates
    4. Pick-place executes (with simulated hardware)
    5. Board state is updated

This is the MVP demo path — proving end-to-end functionality
without real hardware or simulation physics.
"""

import pytest
import numpy as np
import chess

from board_state.board_model import BoardModel, BoardConfig
from chess_core.game_manager import GameManager, GameConfig, GamePhase, PlayerType
from chess_core.interfaces import (
    ChessMove,
    ExecutionStatus,
    MoveType,
    PieceType,
    PieceColor,
    Square,
)
from chess_core.move_parser import parse_uci_move
from manipulation.grasp_policy import GraspPolicyManager, GraspPolicyConfig
from manipulation.pick_place import PickAndPlace, ManipConfig
from manipulation.failure_classifier import FailureClassifier
from robot_model.arm_interface import SimulatedArm, SimulatedGripper


class TestMoveE2E4Integration:
    """End-to-end integration test for a simple pawn advance."""

    @pytest.fixture
    def setup(self):
        """Set up the complete pipeline."""
        # Board model
        board_model = BoardModel(BoardConfig(square_size_m=0.057))

        # Game manager (no Stockfish needed for this test)
        game_config = GameConfig(
            white_player=PlayerType.HUMAN,
            black_player=PlayerType.HUMAN,
        )

        # Simulated hardware
        arm = SimulatedArm(name="test_arm", dof=6)
        arm.initialize()
        gripper = SimulatedGripper(name="test_gripper")
        gripper.initialize()

        # Grasp policy (heuristic only)
        grasp_policy = GraspPolicyManager(
            board_model,
            GraspPolicyConfig(use_learned_grasp=False),
        )

        # Pick-place executor
        manip_config = ManipConfig()
        pick_place = PickAndPlace(arm, gripper, board_model, manip_config)

        return {
            "board_model": board_model,
            "arm": arm,
            "gripper": gripper,
            "grasp_policy": grasp_policy,
            "pick_place": pick_place,
        }

    def test_e2e4_full_pipeline(self, setup):
        """Test the complete e2e4 execution pipeline."""
        board_model = setup["board_model"]
        grasp_policy = setup["grasp_policy"]
        pick_place = setup["pick_place"]

        # Step 1: Parse and validate move
        board = chess.Board()
        move = parse_uci_move("e2e4", board)

        assert move.source == Square.from_algebraic("e2")
        assert move.target == Square.from_algebraic("e4")
        assert move.piece == PieceType.PAWN
        assert move.color == PieceColor.WHITE
        assert move.move_type == MoveType.NORMAL

        # Step 2: Get 3D poses
        source_pose = board_model.get_square_center(move.source)
        target_pose = board_model.get_square_center(move.target)

        # Source e2: file=4, rank=1 → (0.228, 0.057, 0)
        assert source_pose[0] == pytest.approx(4 * 0.057, abs=1e-6)
        assert source_pose[1] == pytest.approx(1 * 0.057, abs=1e-6)

        # Target e4: file=4, rank=3 → (0.228, 0.171, 0)
        assert target_pose[0] == pytest.approx(4 * 0.057, abs=1e-6)
        assert target_pose[1] == pytest.approx(3 * 0.057, abs=1e-6)

        # Step 3: Generate grasp candidates
        candidates = grasp_policy.get_grasp_candidates(
            square=move.source,
            piece_type=move.piece,
            occupied_squares={"d2", "f2"},  # neighboring pawns
        )

        assert len(candidates) > 0
        best_grasp = candidates[0]
        assert best_grasp.piece_type == PieceType.PAWN
        assert best_grasp.score > 0
        assert best_grasp.pose.shape == (4, 4)

        # Step 4: Execute the move
        result = pick_place.execute_move(move, best_grasp)

        assert result.status == ExecutionStatus.SUCCESS
        assert result.move == move
        assert result.duration_s >= 0

    def test_capture_integration(self, setup):
        """Test a capture move through the pipeline."""
        board_model = setup["board_model"]
        grasp_policy = setup["grasp_policy"]
        pick_place = setup["pick_place"]

        # Position with a capture available
        board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
        )
        move = parse_uci_move("e4d5", board)

        assert move.move_type == MoveType.CAPTURE
        assert move.captured_piece == PieceType.PAWN
        assert move.is_capture

        candidates = grasp_policy.get_grasp_candidates(
            square=move.source,
            piece_type=move.piece,
        )
        result = pick_place.execute_move(move, candidates[0])
        assert result.status == ExecutionStatus.SUCCESS

    def test_castling_integration(self, setup):
        """Test kingside castling through the pipeline."""
        board_model = setup["board_model"]
        grasp_policy = setup["grasp_policy"]
        pick_place = setup["pick_place"]

        board = chess.Board(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        )
        move = parse_uci_move("e1g1", board)

        assert move.move_type == MoveType.CASTLING_KINGSIDE
        assert move.piece == PieceType.KING

        candidates = grasp_policy.get_grasp_candidates(
            square=move.source,
            piece_type=PieceType.KING,
        )
        result = pick_place.execute_move(move, candidates[0])
        assert result.status == ExecutionStatus.SUCCESS

    def test_en_passant_integration(self, setup):
        """Test en passant through the pipeline."""
        board_model = setup["board_model"]
        grasp_policy = setup["grasp_policy"]
        pick_place = setup["pick_place"]

        board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        )
        move = parse_uci_move("e5d6", board)

        assert move.move_type == MoveType.EN_PASSANT
        assert move.captured_piece == PieceType.PAWN

        candidates = grasp_policy.get_grasp_candidates(
            square=move.source,
            piece_type=PieceType.PAWN,
        )
        result = pick_place.execute_move(move, candidates[0])
        assert result.status == ExecutionStatus.SUCCESS

    def test_promotion_integration(self, setup):
        """Test pawn promotion through the pipeline."""
        board_model = setup["board_model"]
        grasp_policy = setup["grasp_policy"]
        pick_place = setup["pick_place"]

        board = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
        move = parse_uci_move("e7e8q", board)

        assert move.move_type == MoveType.PROMOTION
        assert move.promotion_piece == PieceType.QUEEN

        candidates = grasp_policy.get_grasp_candidates(
            square=move.source,
            piece_type=PieceType.PAWN,
        )
        result = pick_place.execute_move(move, candidates[0])
        assert result.status == ExecutionStatus.SUCCESS


class TestFailureRecovery:
    """Test failure classification and recovery logic."""

    def test_pickup_failure_escalation(self):
        """Verify that repeated failures escalate properly."""
        from chess_core.interfaces import FailureEvent

        classifier = FailureClassifier()
        move = ChessMove(
            source=Square.from_algebraic("e2"),
            target=Square.from_algebraic("e4"),
            move_type=MoveType.NORMAL,
            piece=PieceType.PAWN,
            color=PieceColor.WHITE,
            uci_string="e2e4",
        )

        failure = FailureEvent(
            status=ExecutionStatus.PICKUP_FAILED,
            move=move,
        )

        from chess_core.interfaces import RecoveryAction

        # First attempt: retry same grasp
        action = classifier.classify_and_recommend(failure)
        assert action == RecoveryAction.RETRY_SAME_GRASP

        # Second attempt: reobserve and regrasp
        action = classifier.classify_and_recommend(failure)
        assert action == RecoveryAction.REOBSERVE_AND_REGRASP

        # Third attempt: slow approach
        action = classifier.classify_and_recommend(failure)
        assert action == RecoveryAction.SLOW_APPROACH

        # Fourth attempt: human intervention
        action = classifier.classify_and_recommend(failure)
        assert action == RecoveryAction.REQUEST_HUMAN

    def test_reset_retries(self):
        classifier = FailureClassifier()
        classifier._attempt_counts["e2e4"] = 5
        classifier.reset_for_move("e2e4")
        assert "e2e4" not in classifier._attempt_counts


class TestBoardStateConsistency:
    """Test board state consistency checking."""

    def test_consistent_after_e2e4(self):
        gm = GameManager()
        gm._board = chess.Board()
        gm._board.push(chess.Move.from_uci("e2e4"))

        expected = gm.get_expected_occupancy()
        assert expected["e2"] is False  # pawn moved away
        assert expected["e4"] is True   # pawn arrived

    def test_detect_discrepancy(self):
        gm = GameManager()
        gm._board = chess.Board()

        # Simulate: pawn should be on e2 but observed as empty
        observed = gm.get_expected_occupancy()
        observed["e2"] = False  # force a discrepancy

        discrepancies = gm.check_board_state_consistency(observed)
        assert len(discrepancies) == 1
        assert "e2" in discrepancies[0]
