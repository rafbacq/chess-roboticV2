"""
Core data models and interfaces for the chess robotic system.

These types are the canonical representations used across ALL subsystems.
Import from here, not from individual packages, for cross-package types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# =============================================================================
# Chess Domain Types
# =============================================================================


class PieceType(Enum):
    """Chess piece types."""
    PAWN = auto()
    KNIGHT = auto()
    BISHOP = auto()
    ROOK = auto()
    QUEEN = auto()
    KING = auto()


class PieceColor(Enum):
    """Piece color / player side."""
    WHITE = auto()
    BLACK = auto()


class MoveType(Enum):
    """Classification of chess move for physical execution planning."""
    NORMAL = auto()
    CAPTURE = auto()
    CASTLING_KINGSIDE = auto()
    CASTLING_QUEENSIDE = auto()
    EN_PASSANT = auto()
    PROMOTION = auto()
    PROMOTION_CAPTURE = auto()


# =============================================================================
# Board Geometry
# =============================================================================


@dataclass(frozen=True)
class Square:
    """
    A board square identified by file and rank indices.

    Convention:
        file: 0-7 maps to a-h
        rank: 0-7 maps to 1-8

    The board frame origin is at the center of square a1 (file=0, rank=0).
    X-axis points from a-file toward h-file.
    Y-axis points from rank 1 toward rank 8.
    Z-axis points up from the board surface.
    """
    file: int  # 0=a, 7=h
    rank: int  # 0=1, 7=8

    def __post_init__(self) -> None:
        if not (0 <= self.file <= 7):
            raise ValueError(f"File must be 0-7, got {self.file}")
        if not (0 <= self.rank <= 7):
            raise ValueError(f"Rank must be 0-7, got {self.rank}")

    @property
    def algebraic(self) -> str:
        """Convert to algebraic notation, e.g. 'e4'."""
        return f"{chr(ord('a') + self.file)}{self.rank + 1}"

    @classmethod
    def from_algebraic(cls, notation: str) -> Square:
        """
        Parse algebraic notation like 'e4' into a Square.

        Args:
            notation: Two-character string like 'a1', 'h8', 'e4'.

        Returns:
            Square instance.

        Raises:
            ValueError: If notation is invalid.
        """
        if len(notation) != 2:
            raise ValueError(f"Algebraic notation must be 2 chars, got '{notation}'")
        file_char, rank_char = notation[0].lower(), notation[1]
        if file_char < 'a' or file_char > 'h':
            raise ValueError(f"Invalid file character: '{file_char}'")
        if rank_char < '1' or rank_char > '8':
            raise ValueError(f"Invalid rank character: '{rank_char}'")
        return cls(file=ord(file_char) - ord('a'), rank=int(rank_char) - 1)

    @property
    def is_light_square(self) -> bool:
        """True if this is a light-colored square."""
        return (self.file + self.rank) % 2 == 1

    def __str__(self) -> str:
        return self.algebraic

    def __repr__(self) -> str:
        return f"Square({self.algebraic})"


@dataclass
class SquarePose:
    """
    3D pose of a square center in the board frame.

    The position is computed from the board model's square size and origin convention.
    The normal is typically [0, 0, 1] for a flat board surface.
    """
    square: Square
    position: np.ndarray  # [x, y, z] in board frame (meters)
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)


# =============================================================================
# Piece State
# =============================================================================

# Piece geometry priors (height_mm, radius_mm) — FIDE tournament standard Staunton
PIECE_GEOMETRY: dict[PieceType, dict[str, float]] = {
    PieceType.PAWN:   {"height_mm": 46.0, "radius_mm": 13.0, "grip_width_mm": 18.0},
    PieceType.KNIGHT: {"height_mm": 57.0, "radius_mm": 14.0, "grip_width_mm": 22.0},
    PieceType.BISHOP: {"height_mm": 65.0, "radius_mm": 14.0, "grip_width_mm": 20.0},
    PieceType.ROOK:   {"height_mm": 48.0, "radius_mm": 15.0, "grip_width_mm": 24.0},
    PieceType.QUEEN:  {"height_mm": 75.0, "radius_mm": 15.0, "grip_width_mm": 22.0},
    PieceType.KING:   {"height_mm": 95.0, "radius_mm": 15.0, "grip_width_mm": 22.0},
}


@dataclass
class PieceState:
    """State of a single chess piece."""
    piece_type: PieceType
    color: PieceColor
    square: Optional[Square] = None  # None if captured / off-board
    is_captured: bool = False

    @property
    def height_mm(self) -> float:
        return PIECE_GEOMETRY[self.piece_type]["height_mm"]

    @property
    def radius_mm(self) -> float:
        return PIECE_GEOMETRY[self.piece_type]["radius_mm"]

    @property
    def grip_width_mm(self) -> float:
        return PIECE_GEOMETRY[self.piece_type]["grip_width_mm"]


# =============================================================================
# Chess Moves
# =============================================================================


@dataclass
class ChessMove:
    """
    A validated chess move ready for physical execution.

    Contains all information needed to plan the manipulation sequence:
    source and target squares, move type, piece type, and any special
    move details (captured piece, promotion piece).
    """
    source: Square
    target: Square
    move_type: MoveType
    piece: PieceType
    color: PieceColor
    captured_piece: Optional[PieceType] = None
    promotion_piece: Optional[PieceType] = None
    uci_string: str = ""

    @property
    def is_capture(self) -> bool:
        return self.move_type in (
            MoveType.CAPTURE,
            MoveType.EN_PASSANT,
            MoveType.PROMOTION_CAPTURE,
        )

    @property
    def is_castling(self) -> bool:
        return self.move_type in (
            MoveType.CASTLING_KINGSIDE,
            MoveType.CASTLING_QUEENSIDE,
        )

    @property
    def is_promotion(self) -> bool:
        return self.move_type in (
            MoveType.PROMOTION,
            MoveType.PROMOTION_CAPTURE,
        )

    def __str__(self) -> str:
        return f"{self.color.name} {self.piece.name} {self.source}→{self.target} ({self.move_type.name})"


# =============================================================================
# Execution & Verification
# =============================================================================


class ExecutionStatus(Enum):
    """Classification of execution outcomes and failure modes."""
    SUCCESS = auto()
    PICKUP_FAILED = auto()
    PIECE_SLIPPED = auto()
    PLACE_OFFCENTER = auto()
    COLLISION = auto()
    PLANNER_FAILED = auto()
    TIMEOUT = auto()
    BOARD_MISMATCH = auto()
    CAMERA_LOW_CONFIDENCE = auto()
    CALIBRATION_DRIFT = auto()
    ILLEGAL_MOVE = auto()
    GRIPPER_FAULT = auto()
    EMERGENCY_STOP = auto()


@dataclass
class GraspCandidate:
    """
    A candidate grasp pose for picking up a chess piece.

    The pose is a 4x4 SE(3) transform in the board frame, representing
    the desired end-effector pose when grasping the piece.
    """
    pose: np.ndarray                   # 4x4 SE(3) in board frame
    piece_type: PieceType
    finger_width_mm: float
    approach_height_mm: float          # height above board surface for approach
    score: float = 1.0                 # higher is better
    source: str = "heuristic"          # "heuristic" or "learned:<model_name>"

    def __post_init__(self) -> None:
        self.pose = np.asarray(self.pose, dtype=np.float64)
        assert self.pose.shape == (4, 4), f"Grasp pose must be 4x4, got {self.pose.shape}"


@dataclass
class PlanRequest:
    """Request to plan a manipulation sequence for a chess move."""
    move: ChessMove
    grasp_candidate: GraspCandidate
    planning_profile: str = "default"  # "default", "cautious", "fast"
    max_planning_time_s: float = 10.0
    allow_replanning: bool = True


@dataclass
class PlanResult:
    """Result of motion planning for a chess move."""
    success: bool
    trajectory_stages: list = field(default_factory=list)  # ordered list of trajectory segments
    estimated_duration_s: float = 0.0
    planning_time_s: float = 0.0
    error_message: str = ""


@dataclass
class ExecutionResult:
    """Result of executing a chess move."""
    status: ExecutionStatus
    move: ChessMove
    duration_s: float = 0.0
    telemetry: dict = field(default_factory=dict)
    error_message: str = ""


@dataclass
class VerificationResult:
    """Result of post-move visual verification."""
    success: bool
    source_empty: bool = False
    target_occupied: bool = False
    target_piece_centered: bool = False
    confidence: float = 0.0
    diagnostic_image_path: str = ""
    mismatch_details: str = ""


@dataclass
class FailureEvent:
    """A failure event with context for logging and recovery."""
    status: ExecutionStatus
    move: ChessMove
    timestamp: float = 0.0
    context: dict = field(default_factory=dict)
    recovery_suggestion: str = ""


class RecoveryAction(Enum):
    """Available recovery actions after a failure."""
    RETRY_SAME_GRASP = auto()
    REOBSERVE_AND_REGRASP = auto()
    REPLAN_WIDER_CLEARANCE = auto()
    SLOW_APPROACH = auto()
    REQUEST_HUMAN = auto()
    ABORT_GAME = auto()
    SKIP_AND_LOG = auto()


# =============================================================================
# Calibration
# =============================================================================


@dataclass
class CalibrationBundle:
    """
    Complete calibration data for the camera-board-robot system.

    All transforms are 4x4 SE(3) homogeneous transformation matrices.
    Convention: T_A_B means "transform that takes points from frame B to frame A".
    """
    camera_matrix: np.ndarray          # 3x3 intrinsic matrix
    dist_coeffs: np.ndarray            # distortion coefficients (1x5 or 1x8)
    T_camera_board: np.ndarray         # 4x4: board frame → camera frame
    T_robot_board: np.ndarray          # 4x4: board frame → robot base frame
    T_robot_camera: np.ndarray         # 4x4: camera frame → robot base frame
    reprojection_error_px: float = 0.0
    timestamp: float = 0.0
    valid: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64)
        self.T_camera_board = np.asarray(self.T_camera_board, dtype=np.float64)
        self.T_robot_board = np.asarray(self.T_robot_board, dtype=np.float64)
        self.T_robot_camera = np.asarray(self.T_robot_camera, dtype=np.float64)

    def transform_board_to_robot(self, point_board: np.ndarray) -> np.ndarray:
        """Transform a 3D point from board frame to robot base frame."""
        p = np.append(point_board, 1.0)
        return (self.T_robot_board @ p)[:3]

    def transform_board_to_camera(self, point_board: np.ndarray) -> np.ndarray:
        """Transform a 3D point from board frame to camera frame."""
        p = np.append(point_board, 1.0)
        return (self.T_camera_board @ p)[:3]
