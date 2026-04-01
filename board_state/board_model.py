"""
Board model: canonical board geometry, coordinate mapping, and square pose generation.

This module defines the physical board layout in 3D space. All coordinates
are in the BOARD FRAME:
    - Origin: center of square a1
    - X-axis: a1 → h1 (file direction)
    - Y-axis: a1 → a8 (rank direction)
    - Z-axis: up from board surface
    - Units: meters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from chess_core.interfaces import (
    PIECE_GEOMETRY,
    PieceType,
    Square,
    SquarePose,
)

logger = logging.getLogger(__name__)


@dataclass
class BoardConfig:
    """Physical board configuration."""
    square_size_m: float = 0.057       # 57mm FIDE standard
    board_thickness_m: float = 0.015   # 15mm board thickness
    board_height_m: float = 0.0        # height of board surface above table
    num_files: int = 8
    num_ranks: int = 8

    # Captured piece tray configuration
    tray_offset_x_m: float = 0.10      # tray offset from board edge in X
    tray_offset_y_m: float = 0.0       # tray offset in Y
    tray_slot_spacing_m: float = 0.035  # spacing between tray slots
    tray_side: str = "right"           # "right" or "left" of the board

    @property
    def board_width_m(self) -> float:
        """Total board width (X direction)."""
        return self.num_files * self.square_size_m

    @property
    def board_depth_m(self) -> float:
        """Total board depth (Y direction)."""
        return self.num_ranks * self.square_size_m


class BoardModel:
    """
    3D board geometry model.

    Provides coordinate transformations between:
        - Algebraic notation (e.g., "e4")
        - Square indices (file=4, rank=3)
        - 3D positions in board frame (meters)
        - Tray positions for captured pieces

    Usage:
        model = BoardModel(BoardConfig())
        pose = model.get_square_pose(Square.from_algebraic("e4"))
        print(pose.position)  # [0.228, 0.171, 0.0]  (in meters)
    """

    def __init__(self, config: BoardConfig | None = None) -> None:
        self.config = config or BoardConfig()
        self._validate_config()
        self._tray_index = 0  # next available tray slot

        logger.info(
            f"BoardModel initialized: {self.config.num_files}x{self.config.num_ranks}, "
            f"square_size={self.config.square_size_m * 1000:.0f}mm"
        )

    def _validate_config(self) -> None:
        assert self.config.square_size_m > 0, "Square size must be positive"
        assert self.config.num_files == 8, "Standard chess = 8 files"
        assert self.config.num_ranks == 8, "Standard chess = 8 ranks"

    def get_square_center(self, square: Square) -> np.ndarray:
        """
        Get the 3D position of a square's center in the board frame.

        The origin is at the center of a1, so:
            a1 = (0, 0, 0)
            h1 = (7 * sq_size, 0, 0)
            a8 = (0, 7 * sq_size, 0)
            h8 = (7 * sq_size, 7 * sq_size, 0)

        Returns:
            np.ndarray [x, y, z] in meters, board frame.
        """
        x = square.file * self.config.square_size_m
        y = square.rank * self.config.square_size_m
        z = 0.0  # board surface
        return np.array([x, y, z], dtype=np.float64)

    def get_square_pose(self, square: Square) -> SquarePose:
        """Get the full SquarePose for a square."""
        return SquarePose(
            square=square,
            position=self.get_square_center(square),
            normal=np.array([0.0, 0.0, 1.0]),
        )

    def get_all_square_poses(self) -> list[SquarePose]:
        """Get poses for all 64 squares."""
        poses = []
        for rank in range(8):
            for file in range(8):
                sq = Square(file=file, rank=rank)
                poses.append(self.get_square_pose(sq))
        return poses

    def get_piece_top_z(self, piece_type: PieceType) -> float:
        """
        Get the Z coordinate of the top of a piece (for grasp approach height).

        Returns:
            Height in meters above board surface.
        """
        height_mm = PIECE_GEOMETRY[piece_type]["height_mm"]
        return height_mm / 1000.0

    def get_grasp_z(self, piece_type: PieceType, grasp_ratio: float = 0.65) -> float:
        """
        Get the Z coordinate for grasping a piece.

        Default grasps at 65% of piece height (upper body, below the top).

        Args:
            piece_type: Type of piece being grasped.
            grasp_ratio: Fraction of piece height to grasp at (0=base, 1=top).

        Returns:
            Height in meters above board surface.
        """
        height_mm = PIECE_GEOMETRY[piece_type]["height_mm"]
        return (height_mm * grasp_ratio) / 1000.0

    def get_approach_pose(
        self,
        square: Square,
        piece_type: PieceType,
        clearance_m: float = 0.05,
    ) -> np.ndarray:
        """
        Get a pre-grasp approach pose above a piece.

        Args:
            square: Target square.
            piece_type: Type of piece (determines approach height).
            clearance_m: Additional clearance above piece top.

        Returns:
            4x4 SE(3) pose in board frame (top-down orientation).
        """
        center = self.get_square_center(square)
        piece_top = self.get_piece_top_z(piece_type)

        pose = np.eye(4, dtype=np.float64)
        # Top-down approach: gripper pointing down (-Z)
        pose[0, 0] = 1.0   # X_ee = X_board
        pose[1, 1] = -1.0  # Y_ee = -Y_board (gripper Y flipped)
        pose[2, 2] = -1.0  # Z_ee = -Z_board (pointing down)
        pose[0, 3] = center[0]
        pose[1, 3] = center[1]
        pose[2, 3] = piece_top + clearance_m

        return pose

    def get_tray_position(self, slot_index: Optional[int] = None) -> np.ndarray:
        """
        Get the 3D position of a captured-piece tray slot.

        Args:
            slot_index: Specific slot index (0-15). If None, uses next available.

        Returns:
            np.ndarray [x, y, z] in board frame (meters).
        """
        if slot_index is None:
            slot_index = self._tray_index
            self._tray_index += 1

        if slot_index >= 16:
            logger.warning(f"Tray slot {slot_index} exceeds capacity (16)")
            slot_index = slot_index % 16

        # Tray layout: single column alongside the board
        row = slot_index % 8
        col = slot_index // 8

        if self.config.tray_side == "right":
            x = self.config.board_width_m + self.config.tray_offset_x_m + col * self.config.tray_slot_spacing_m
        else:
            x = -self.config.tray_offset_x_m - col * self.config.tray_slot_spacing_m

        y = row * self.config.tray_slot_spacing_m + self.config.tray_offset_y_m
        z = 0.0

        return np.array([x, y, z], dtype=np.float64)

    def reset_tray(self) -> None:
        """Reset the tray slot counter."""
        self._tray_index = 0

    def get_board_corners(self) -> np.ndarray:
        """
        Get the four outer corners of the board in board frame.

        Returns:
            4x3 array of corner positions:
                [0] = a1 outer corner (bottom-left)
                [1] = h1 outer corner (bottom-right)
                [2] = h8 outer corner (top-right)
                [3] = a8 outer corner (top-left)
        """
        hs = self.config.square_size_m / 2  # half square
        w = self.config.board_width_m
        d = self.config.board_depth_m

        return np.array([
            [-hs, -hs, 0.0],       # a1 outer corner
            [w - hs, -hs, 0.0],    # h1 outer corner (Note: w - hs = 7*sq + sq/2)
            [w - hs, d - hs, 0.0], # h8 outer corner
            [-hs, d - hs, 0.0],    # a8 outer corner
        ], dtype=np.float64)

    def get_neighboring_squares(self, square: Square) -> list[Square]:
        """Get all squares adjacent to the given square (up to 8 neighbors)."""
        neighbors = []
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if df == 0 and dr == 0:
                    continue
                f, r = square.file + df, square.rank + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    neighbors.append(Square(file=f, rank=r))
        return neighbors

    def get_safe_waypoint(self, height_m: float = 0.15) -> np.ndarray:
        """
        Get a safe waypoint above the board center for transit moves.

        Args:
            height_m: Height above board surface.

        Returns:
            [x, y, z] position in board frame.
        """
        cx = self.config.board_width_m / 2
        cy = self.config.board_depth_m / 2
        return np.array([cx, cy, height_m], dtype=np.float64)

    def square_distance_m(self, sq1: Square, sq2: Square) -> float:
        """Euclidean distance between two square centers in meters."""
        p1 = self.get_square_center(sq1)
        p2 = self.get_square_center(sq2)
        return float(np.linalg.norm(p1 - p2))
