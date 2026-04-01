"""
Pick-and-place manipulation primitives for chess piece movement.

Implements the staged task model:
    1. pre-grasp — move above source square at safe height
    2. approach — descend vertically to grasp height
    3. close gripper — grasp the piece
    4. lift — ascend vertically to safe transit height
    5. transit — move to above target square via safe corridor
    6. pre-place — position above target square
    7. place — descend to placement height
    8. release — open gripper
    9. retreat — ascend to safe height

Each stage is a separate trajectory segment with independent
speed/safety constraints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from board_state.board_model import BoardModel
from chess_core.interfaces import (
    ChessMove,
    ExecutionResult,
    ExecutionStatus,
    GraspCandidate,
    MoveType,
    PieceType,
    Square,
    VerificationResult,
)
from robot_model.arm_interface import ArmInterface, GripperInterface

logger = logging.getLogger(__name__)


class ManipStage(Enum):
    """Enumeration of manipulation stages for telemetry and debugging."""
    IDLE = auto()
    PRE_GRASP = auto()
    APPROACH = auto()
    CLOSE_GRIPPER = auto()
    VERIFY_PICKUP = auto()
    LIFT = auto()
    TRANSIT = auto()
    PRE_PLACE = auto()
    PLACE = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class ManipConfig:
    """Configuration for manipulation behavior."""
    # Heights above board surface (meters)
    safe_height_m: float = 0.15       # safe transit height
    approach_clearance_m: float = 0.05  # clearance above piece top for pre-grasp
    lift_height_m: float = 0.12       # lift height after pickup
    place_clearance_m: float = 0.003  # small gap above board for placement

    # Speeds
    transit_speed: float = 0.4        # velocity scale for free-space transit
    approach_speed: float = 0.15      # velocity scale near board
    retreat_speed: float = 0.3        # velocity scale for retreat

    # Gripper
    grasp_force_n: float = 10.0       # default grasp force
    finger_width_tolerance_mm: float = 5.0  # extra opening beyond piece width

    # Timeouts
    stage_timeout_s: float = 15.0     # timeout per stage
    verify_delay_s: float = 0.5       # delay before verifying pickup

    # Board-to-robot transform (set by calibration)
    T_robot_board: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )


class PickAndPlace:
    """
    Staged pick-and-place executor for chess piece manipulation.

    Usage:
        pp = PickAndPlace(arm, gripper, board_model, config)
        result = pp.execute_move(chess_move, grasp_candidate)
    """

    def __init__(
        self,
        arm: ArmInterface,
        gripper: GripperInterface,
        board: BoardModel,
        config: ManipConfig | None = None,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.board = board
        self.config = config or ManipConfig()
        self._current_stage = ManipStage.IDLE
        self._telemetry: dict = {}

    @property
    def current_stage(self) -> ManipStage:
        return self._current_stage

    def execute_move(
        self,
        move: ChessMove,
        grasp: GraspCandidate,
    ) -> ExecutionResult:
        """
        Execute a complete chess move as a staged pick-and-place task.

        Handles standard moves, captures, castling, en passant, and promotions
        by decomposing them into appropriate sequences of pick-place primitives.

        Args:
            move: The chess move to execute.
            grasp: Pre-computed grasp candidate for the piece.

        Returns:
            ExecutionResult with status, timing, and telemetry.
        """
        t_start = time.time()
        self._telemetry = {"move": str(move), "stages": []}

        logger.info(f"Executing move: {move}")

        try:
            if move.move_type == MoveType.CAPTURE:
                result = self._execute_capture(move, grasp)
            elif move.is_castling:
                result = self._execute_castling(move, grasp)
            elif move.move_type == MoveType.EN_PASSANT:
                result = self._execute_en_passant(move, grasp)
            elif move.is_promotion:
                result = self._execute_promotion(move, grasp)
            else:
                result = self._execute_simple_move(move, grasp)

        except Exception as e:
            logger.error(f"Manipulation failed: {e}", exc_info=True)
            result = ExecutionResult(
                status=ExecutionStatus.PLANNER_FAILED,
                move=move,
                error_message=str(e),
            )

        result.duration_s = time.time() - t_start
        result.telemetry = self._telemetry
        self._current_stage = ManipStage.IDLE

        return result

    def _execute_simple_move(
        self, move: ChessMove, grasp: GraspCandidate,
    ) -> ExecutionResult:
        """Execute a simple non-capture move: pick from source, place at target."""
        source_center = self.board.get_square_center(move.source)
        target_center = self.board.get_square_center(move.target)

        # Pick from source
        success = self._pick(source_center, move.piece, grasp)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PICKUP_FAILED,
                move=move,
                error_message="Pickup failed at source square",
            )

        # Place at target
        success = self._place(target_center, move.piece)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PLACE_OFFCENTER,
                move=move,
                error_message="Placement failed at target square",
            )

        return ExecutionResult(status=ExecutionStatus.SUCCESS, move=move)

    def _execute_capture(
        self, move: ChessMove, grasp: GraspCandidate,
    ) -> ExecutionResult:
        """
        Execute a capture: pick captured piece → tray, then pick moving piece → target.
        """
        target_center = self.board.get_square_center(move.target)
        source_center = self.board.get_square_center(move.source)
        tray_pos = self.board.get_tray_position()

        # First: remove captured piece to tray
        assert move.captured_piece is not None
        logger.info(f"Removing captured {move.captured_piece.name} from {move.target}")

        cap_grasp = self._make_default_grasp(move.captured_piece, target_center)
        success = self._pick(target_center, move.captured_piece, cap_grasp)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PICKUP_FAILED,
                move=move,
                error_message="Failed to pick captured piece",
            )

        success = self._place(tray_pos, move.captured_piece)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PLACE_OFFCENTER,
                move=move,
                error_message="Failed to place captured piece in tray",
            )

        # Second: move the attacking piece
        success = self._pick(source_center, move.piece, grasp)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PICKUP_FAILED,
                move=move,
                error_message="Failed to pick attacking piece",
            )

        success = self._place(target_center, move.piece)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PLACE_OFFCENTER,
                move=move,
                error_message="Failed to place attacking piece",
            )

        return ExecutionResult(status=ExecutionStatus.SUCCESS, move=move)

    def _execute_castling(
        self, move: ChessMove, grasp: GraspCandidate,
    ) -> ExecutionResult:
        """Execute castling: move king, then move rook."""
        from chess_core.move_parser import get_castling_rook_move

        rook_src, rook_tgt = get_castling_rook_move(move)

        # Move king first
        king_src = self.board.get_square_center(move.source)
        king_tgt = self.board.get_square_center(move.target)

        success = self._pick(king_src, PieceType.KING, grasp)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PICKUP_FAILED,
                move=move,
                error_message="Failed to pick king for castling",
            )

        success = self._place(king_tgt, PieceType.KING)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PLACE_OFFCENTER,
                move=move,
                error_message="Failed to place king for castling",
            )

        # Move rook
        rook_src_pos = self.board.get_square_center(rook_src)
        rook_tgt_pos = self.board.get_square_center(rook_tgt)
        rook_grasp = self._make_default_grasp(PieceType.ROOK, rook_src_pos)

        success = self._pick(rook_src_pos, PieceType.ROOK, rook_grasp)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PICKUP_FAILED,
                move=move,
                error_message="Failed to pick rook for castling",
            )

        success = self._place(rook_tgt_pos, PieceType.ROOK)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PLACE_OFFCENTER,
                move=move,
                error_message="Failed to place rook for castling",
            )

        return ExecutionResult(status=ExecutionStatus.SUCCESS, move=move)

    def _execute_en_passant(
        self, move: ChessMove, grasp: GraspCandidate,
    ) -> ExecutionResult:
        """Execute en passant: move pawn, then remove captured pawn from its actual square."""
        from chess_core.move_parser import get_en_passant_capture_square

        # Move the attacking pawn first
        source_center = self.board.get_square_center(move.source)
        target_center = self.board.get_square_center(move.target)

        success = self._pick(source_center, PieceType.PAWN, grasp)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PICKUP_FAILED,
                move=move,
                error_message="Failed to pick pawn for en passant",
            )

        success = self._place(target_center, PieceType.PAWN)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PLACE_OFFCENTER,
                move=move,
                error_message="Failed to place pawn for en passant",
            )

        # Remove captured pawn
        cap_square = get_en_passant_capture_square(move)
        cap_pos = self.board.get_square_center(cap_square)
        tray_pos = self.board.get_tray_position()
        cap_grasp = self._make_default_grasp(PieceType.PAWN, cap_pos)

        success = self._pick(cap_pos, PieceType.PAWN, cap_grasp)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PICKUP_FAILED,
                move=move,
                error_message="Failed to pick captured pawn in en passant",
            )

        success = self._place(tray_pos, PieceType.PAWN)
        if not success:
            return ExecutionResult(
                status=ExecutionStatus.PLACE_OFFCENTER,
                move=move,
                error_message="Failed to tray captured pawn in en passant",
            )

        return ExecutionResult(status=ExecutionStatus.SUCCESS, move=move)

    def _execute_promotion(
        self, move: ChessMove, grasp: GraspCandidate,
    ) -> ExecutionResult:
        """
        Execute pawn promotion.

        Currently: move the pawn to the promotion square.
        TODO: physically swap the pawn for the promotion piece if using real pieces.
        """
        # Handle capture first if promotion-capture
        if move.move_type == MoveType.PROMOTION_CAPTURE and move.captured_piece:
            target_center = self.board.get_square_center(move.target)
            tray_pos = self.board.get_tray_position()
            cap_grasp = self._make_default_grasp(move.captured_piece, target_center)

            success = self._pick(target_center, move.captured_piece, cap_grasp)
            if not success:
                return ExecutionResult(
                    status=ExecutionStatus.PICKUP_FAILED, move=move,
                    error_message="Failed to remove captured piece for promotion",
                )
            self._place(tray_pos, move.captured_piece)

        # Move pawn to promotion square
        return self._execute_simple_move(move, grasp)

    # =========================================================================
    # Low-level pick/place primitives
    # =========================================================================

    def _pick(
        self,
        position: np.ndarray,
        piece_type: PieceType,
        grasp: GraspCandidate,
    ) -> bool:
        """
        Pick a piece at the given board-frame position.

        Stages: pre-grasp → approach → close → verify → lift
        """
        grasp_z = self.board.get_grasp_z(piece_type)
        safe_z = self.config.safe_height_m

        # Transform to robot frame
        pos_robot = self._board_to_robot(position)

        # Stage 1: Pre-grasp — move above piece at safe height
        self._set_stage(ManipStage.PRE_GRASP)
        pre_grasp = self._make_top_down_pose(pos_robot, safe_z)
        self.gripper.open(
            width_mm=grasp.finger_width_mm + self.config.finger_width_tolerance_mm
        )
        if not self.arm.move_to_pose(pre_grasp, velocity_scale=self.config.transit_speed):
            return False

        # Stage 2: Approach — descend vertically
        self._set_stage(ManipStage.APPROACH)
        approach = self._make_top_down_pose(pos_robot, grasp_z)
        if not self.arm.move_cartesian_linear(approach, velocity_ms=0.03):
            return False

        # Stage 3: Close gripper
        self._set_stage(ManipStage.CLOSE_GRIPPER)
        self.gripper.close(
            force_n=self.config.grasp_force_n,
            width_mm=grasp.finger_width_mm,
        )

        # Stage 4: Verify pickup
        self._set_stage(ManipStage.VERIFY_PICKUP)
        time.sleep(self.config.verify_delay_s)
        if not self.gripper.is_gripping():
            logger.warning("Gripper reports no object grasped")
            self.gripper.open()
            # Retreat even on failure
            self.arm.move_cartesian_linear(pre_grasp, velocity_ms=0.05)
            return False

        # Stage 5: Lift
        self._set_stage(ManipStage.LIFT)
        lift = self._make_top_down_pose(pos_robot, self.config.lift_height_m)
        if not self.arm.move_cartesian_linear(lift, velocity_ms=0.05):
            return False

        logger.info(f"Pick successful: {piece_type.name}")
        return True

    def _place(self, position: np.ndarray, piece_type: PieceType) -> bool:
        """
        Place a held piece at the given board-frame position.

        Stages: transit → pre-place → place → release → retreat
        """
        piece_height = self.board.get_piece_top_z(piece_type)
        place_z = piece_height * 0.05 + self.config.place_clearance_m  # just above surface
        safe_z = self.config.safe_height_m

        pos_robot = self._board_to_robot(position)

        # Stage 6: Transit to above target
        self._set_stage(ManipStage.TRANSIT)
        pre_place = self._make_top_down_pose(pos_robot, safe_z)
        if not self.arm.move_to_pose(pre_place, velocity_scale=self.config.transit_speed):
            return False

        # Stage 7: Pre-place — already there from transit

        # Stage 8: Place — descend
        self._set_stage(ManipStage.PLACE)
        place_pose = self._make_top_down_pose(pos_robot, place_z)
        if not self.arm.move_cartesian_linear(place_pose, velocity_ms=0.03):
            return False

        # Stage 9: Release
        self._set_stage(ManipStage.RELEASE)
        self.gripper.open()
        time.sleep(0.2)

        # Stage 10: Retreat
        self._set_stage(ManipStage.RETREAT)
        if not self.arm.move_cartesian_linear(pre_place, velocity_ms=0.05):
            return False

        logger.info(f"Place successful: {piece_type.name}")
        self._set_stage(ManipStage.DONE)
        return True

    # =========================================================================
    # Helpers
    # =========================================================================

    def _board_to_robot(self, point_board: np.ndarray) -> np.ndarray:
        """Transform a 3D point from board frame to robot frame."""
        p = np.ones(4)
        p[:3] = point_board[:3]
        return (self.config.T_robot_board @ p)[:3]

    def _make_top_down_pose(
        self, position_robot: np.ndarray, height: float,
    ) -> np.ndarray:
        """
        Create a top-down EE pose in robot frame.

        The gripper Z-axis points downward, X along the board X-axis.
        """
        pose = np.eye(4, dtype=np.float64)
        # Rotation: Z down, X forward, Y right
        pose[0, 0] = 1.0
        pose[1, 1] = -1.0
        pose[2, 2] = -1.0
        # Position
        pose[0, 3] = position_robot[0]
        pose[1, 3] = position_robot[1]
        pose[2, 3] = height
        return pose

    def _make_default_grasp(
        self, piece_type: PieceType, position: np.ndarray,
    ) -> GraspCandidate:
        """Create a default top-down grasp candidate for a piece."""
        from chess_core.interfaces import PIECE_GEOMETRY

        geom = PIECE_GEOMETRY[piece_type]
        pose = self.board.get_approach_pose(
            Square(0, 0),  # dummy — position overridden
            piece_type,
        )
        pose[:3, 3] = position

        return GraspCandidate(
            pose=pose,
            piece_type=piece_type,
            finger_width_mm=geom["grip_width_mm"],
            approach_height_mm=geom["height_mm"] * 0.65,
            score=0.8,
            source="heuristic_default",
        )

    def _set_stage(self, stage: ManipStage) -> None:
        self._current_stage = stage
        self._telemetry.setdefault("stages", []).append(
            {"stage": stage.name, "time": time.time()}
        )
        logger.debug(f"Stage: {stage.name}")
