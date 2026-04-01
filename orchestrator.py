"""
System orchestrator: top-level controller wiring all subsystems together.

Connects: GameManager → GraspPolicyManager → MotionPlanner → Executor →
          MoveVerifier → FailureClassifier → back to GameManager.

This is the single entry point for running a chess game against the robot.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from board_state.board_model import BoardConfig, BoardModel
from chess_core.game_manager import GameConfig, GameManager, GamePhase, PlayerType
from chess_core.interfaces import (
    ChessMove,
    ExecutionResult,
    ExecutionStatus,
    FailureEvent,
    RecoveryAction,
    VerificationResult,
)
from manipulation.failure_classifier import FailureClassifier
from manipulation.grasp_policy import GraspPolicyConfig, GraspPolicyManager
from manipulation.pick_place import ManipConfig, PickAndPlace
from robot_model.arm_interface import ArmInterface, GripperInterface

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the system orchestrator."""
    board_config: BoardConfig = field(default_factory=BoardConfig)
    game_config: GameConfig = field(default_factory=GameConfig)
    manip_config: ManipConfig = field(default_factory=ManipConfig)
    grasp_config: GraspPolicyConfig = field(default_factory=GraspPolicyConfig)
    T_robot_board: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    max_move_retries: int = 3
    verify_moves: bool = False  # requires camera
    log_telemetry: bool = True


class SystemOrchestrator:
    """
    Top-level controller for the chess robot system.

    Orchestrates the complete game loop:
        1. Get move (from engine or human)
        2. Generate grasp candidates
        3. Execute manipulation
        4. Optionally verify via camera
        5. Confirm move in game state
        6. Handle failures with retry/escalation

    Usage:
        orch = SystemOrchestrator(arm, gripper, config)
        orch.start_game()

        # Play one turn:
        result = orch.execute_turn()

        # Or run full game loop:
        orch.run_game_loop(max_moves=100)
    """

    def __init__(
        self,
        arm: ArmInterface,
        gripper: GripperInterface,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()

        # Subsystems
        self.board_model = BoardModel(self.config.board_config)
        self.game = GameManager(self.config.game_config)
        self.grasp_policy = GraspPolicyManager(
            self.board_model, self.config.grasp_config
        )

        manip_config = self.config.manip_config
        manip_config.T_robot_board = self.config.T_robot_board
        self.pick_place = PickAndPlace(arm, gripper, self.board_model, manip_config)

        self.failure_classifier = FailureClassifier()

        self._arm = arm
        self._gripper = gripper
        self._move_count = 0
        self._game_log: list[dict] = []

    def start_game(self) -> None:
        """Initialize and start a new chess game."""
        self.game.start_game()
        self.failure_classifier.reset_all()
        self.board_model.reset_tray()
        self._move_count = 0
        self._game_log.clear()
        logger.info("=== GAME STARTED ===")

    def stop_game(self) -> None:
        """Stop the current game."""
        self.game.stop_game()
        logger.info(f"=== GAME STOPPED after {self._move_count} moves ===")

    def execute_turn(self, manual_uci: str = "") -> ExecutionResult:
        """
        Execute a single turn (one player's move).

        For engine turns, queries Stockfish for the best move.
        For human turns, expects `manual_uci` to be provided.

        Args:
            manual_uci: UCI move string for human moves (e.g., "e2e4").

        Returns:
            ExecutionResult with status.
        """
        if self.game.phase != GamePhase.AWAITING_MOVE:
            logger.error(f"Cannot execute turn in phase {self.game.phase.name}")
            return ExecutionResult(
                status=ExecutionStatus.ILLEGAL_MOVE,
                move=None,
                error_message=f"Game not in AWAITING_MOVE phase",
            )

        player_type = self.game.current_player_type
        color = self.game.current_color

        logger.info(
            f"\n{'='*50}\n"
            f"Turn {self._move_count + 1}: {color.name} ({player_type.name})\n"
            f"{'='*50}"
        )

        # Step 1: Get the move
        try:
            if player_type == PlayerType.ENGINE:
                chess_move = self.game.get_engine_move()
            elif manual_uci:
                chess_move = self.game.validate_and_parse_move(manual_uci)
            else:
                logger.error("Human turn but no UCI string provided")
                return ExecutionResult(
                    status=ExecutionStatus.ILLEGAL_MOVE,
                    move=None,
                    error_message="No move provided for human turn",
                )
        except ValueError as e:
            logger.error(f"Invalid move: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ILLEGAL_MOVE,
                move=None,
                error_message=str(e),
            )

        logger.info(f"Move: {chess_move}")

        # Step 2: Execute with retry loop
        result = self._execute_with_retries(chess_move)

        # Step 3: If success, confirm in game state
        if result.status == ExecutionStatus.SUCCESS:
            self.game.confirm_move(chess_move, execution_time_s=result.duration_s)
            self._move_count += 1
            self._game_log.append({
                "move_number": self._move_count,
                "move": str(chess_move),
                "uci": chess_move.uci_string,
                "duration_s": result.duration_s,
                "color": color.name,
                "status": result.status.name,
            })
            logger.info(f"✓ Move {self._move_count} confirmed: {chess_move}")
        else:
            # Reset game phase on failure
            self.game.set_phase(GamePhase.AWAITING_MOVE)
            logger.error(f"✗ Move failed: {result.error_message}")

        return result

    def _execute_with_retries(self, move: ChessMove) -> ExecutionResult:
        """Execute a move with retry logic on failure."""
        for attempt in range(1, self.config.max_move_retries + 1):
            # Generate grasp candidates
            occupied = self._get_occupied_neighbors(move.source)
            candidates = self.grasp_policy.get_grasp_candidates(
                square=move.source,
                piece_type=move.piece,
                occupied_squares=occupied,
            )

            if not candidates:
                return ExecutionResult(
                    status=ExecutionStatus.PLANNER_FAILED,
                    move=move,
                    error_message="No grasp candidates generated",
                )

            # Execute
            self.game.set_phase(GamePhase.EXECUTING)
            t0 = time.time()
            result = self.pick_place.execute_move(move, candidates[0])
            result.duration_s = time.time() - t0

            if result.status == ExecutionStatus.SUCCESS:
                return result

            # Classify failure and decide recovery
            failure = FailureEvent(
                status=result.status,
                move=move,
                timestamp=time.time(),
            )
            recovery = self.failure_classifier.classify_and_recommend(failure)

            if recovery == RecoveryAction.REQUEST_HUMAN:
                logger.warning("Requesting human intervention")
                return result
            elif recovery == RecoveryAction.ABORT_GAME:
                logger.error("Aborting game due to critical failure")
                return result

            logger.info(
                f"Attempt {attempt}/{self.config.max_move_retries} failed, "
                f"recovery: {recovery.name}"
            )

        return result

    def run_game_loop(
        self,
        max_moves: int = 200,
        human_moves: Optional[list[str]] = None,
    ) -> dict:
        """
        Run a complete game loop.

        Args:
            max_moves: Maximum total moves before stopping.
            human_moves: Pre-programmed UCI moves for human player.

        Returns:
            Game summary dict.
        """
        human_move_idx = 0
        t_game_start = time.time()

        while (
            self._move_count < max_moves
            and self.game.phase == GamePhase.AWAITING_MOVE
        ):
            # Provide UCI for human moves
            uci = ""
            if self.game.current_player_type == PlayerType.HUMAN:
                if human_moves and human_move_idx < len(human_moves):
                    uci = human_moves[human_move_idx]
                    human_move_idx += 1
                else:
                    logger.info("No more human moves — stopping game loop")
                    break

            result = self.execute_turn(manual_uci=uci)

            if result.status != ExecutionStatus.SUCCESS:
                logger.warning(
                    f"Move failed ({result.status.name}), stopping game loop"
                )
                break

            # Check game over
            if self.game.phase == GamePhase.GAME_OVER:
                break

        game_time = time.time() - t_game_start

        summary = {
            "total_moves": self._move_count,
            "game_time_s": game_time,
            "result": self.game.board.result() if self.game.board.is_game_over() else "in_progress",
            "final_fen": self.game.fen,
            "move_log": self._game_log,
        }

        logger.info(
            f"\n{'='*50}\n"
            f"GAME SUMMARY\n"
            f"Moves: {self._move_count}\n"
            f"Time: {game_time:.1f}s\n"
            f"Result: {summary['result']}\n"
            f"{'='*50}"
        )

        return summary

    def _get_occupied_neighbors(self, square) -> set[str]:
        """Get algebraic names of occupied neighboring squares."""
        neighbors = self.board_model.get_neighboring_squares(square)
        occupied = set()
        for n in neighbors:
            piece = self.game.get_piece_at(n)
            if piece is not None:
                occupied.add(n.algebraic)
        return occupied

    @property
    def move_count(self) -> int:
        return self._move_count

    @property
    def game_log(self) -> list[dict]:
        return list(self._game_log)
