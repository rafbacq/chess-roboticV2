"""
Collision world manager for the chess manipulation environment.

Maintains the set of collision objects (board, pieces, tray) that
MoveIt 2 needs for collision-aware motion planning. Updates dynamically
as pieces move.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from board_state.board_model import BoardModel
from chess_core.interfaces import PIECE_GEOMETRY, PieceType, Square

logger = logging.getLogger(__name__)


@dataclass
class CollisionObject:
    """A collision object in the planning scene."""
    name: str
    shape: str  # "box", "cylinder", "mesh"
    pose: np.ndarray  # 4x4 SE(3) in robot frame
    dimensions: dict  # shape-specific dimensions
    attached: bool = False  # attached to end-effector?

    @staticmethod
    def box(name: str, pose: np.ndarray, size_xyz: tuple[float, float, float]) -> CollisionObject:
        return CollisionObject(name=name, shape="box", pose=pose,
                             dimensions={"x": size_xyz[0], "y": size_xyz[1], "z": size_xyz[2]})

    @staticmethod
    def cylinder(name: str, pose: np.ndarray, radius: float, height: float) -> CollisionObject:
        return CollisionObject(name=name, shape="cylinder", pose=pose,
                             dimensions={"radius": radius, "height": height})


class CollisionWorldManager:
    """
    Manages collision objects for the chess planning scene.

    Provides methods to:
        - Add the board surface as a collision plane
        - Add/remove piece collision objects as pieces move
        - Attach/detach pieces to the end-effector during grasping
        - Add the captured piece tray
        - Update the entire scene from board state

    Usage:
        cwm = CollisionWorldManager(board_model, T_robot_board)
        cwm.build_static_scene()  # board surface + tray
        cwm.update_pieces(occupancy_map)  # add piece collision cylinders
        cwm.attach_piece("e2")  # attach piece to EE for transport
        cwm.detach_piece("e4")  # detach at target
    """

    def __init__(
        self,
        board: BoardModel,
        T_robot_board: np.ndarray,
    ) -> None:
        self.board = board
        self.T_robot_board = T_robot_board.copy()
        self._objects: dict[str, CollisionObject] = {}
        self._attached_piece: Optional[str] = None

    def build_static_scene(self) -> list[CollisionObject]:
        """
        Build static collision objects: board surface and tray.

        Returns:
            List of collision objects to add to the planning scene.
        """
        objects = []

        # Board surface as a thin box
        board_w = self.board.config.board_width_m
        board_d = self.board.config.board_depth_m
        board_t = self.board.config.board_thickness_m

        board_center_board = np.array([
            board_w / 2 - self.board.config.square_size_m / 2,
            board_d / 2 - self.board.config.square_size_m / 2,
            -board_t / 2,
        ])
        board_center_robot = self._to_robot(board_center_board)

        board_pose = np.eye(4)
        board_pose[:3, 3] = board_center_robot
        board_obj = CollisionObject.box(
            "chess_board",
            board_pose,
            (board_w + 0.02, board_d + 0.02, board_t),  # slight padding
        )
        self._objects["chess_board"] = board_obj
        objects.append(board_obj)

        # Table surface below board
        table_pose = np.eye(4)
        table_pose[:3, 3] = board_center_robot
        table_pose[2, 3] = -board_t - 0.01  # just below board
        table_obj = CollisionObject.box(
            "table_surface",
            table_pose,
            (1.0, 1.0, 0.02),  # large thin box
        )
        self._objects["table_surface"] = table_obj
        objects.append(table_obj)

        logger.info(f"Built static scene: {len(objects)} objects")
        return objects

    def add_piece(self, square: Square, piece_type: PieceType) -> CollisionObject:
        """
        Add a piece collision object at the given square.

        Pieces are modeled as cylinders with type-specific dimensions.
        """
        geom = PIECE_GEOMETRY[piece_type]
        radius = geom["radius_mm"] / 1000.0
        height = geom["height_mm"] / 1000.0

        center_board = self.board.get_square_center(square)
        center_board[2] = height / 2  # center of cylinder
        center_robot = self._to_robot(center_board)

        pose = np.eye(4)
        pose[:3, 3] = center_robot

        name = f"piece_{square.algebraic}"
        obj = CollisionObject.cylinder(name, pose, radius, height)
        self._objects[name] = obj

        return obj

    def remove_piece(self, square: Square) -> Optional[CollisionObject]:
        """Remove a piece collision object from a square."""
        name = f"piece_{square.algebraic}"
        return self._objects.pop(name, None)

    def attach_piece(self, square: Square) -> Optional[CollisionObject]:
        """
        Attach a piece to the end-effector (for transport).
        Removes it from the planning scene collision objects.
        """
        name = f"piece_{square.algebraic}"
        if name in self._objects:
            obj = self._objects.pop(name)
            obj.attached = True
            self._attached_piece = name
            logger.debug(f"Attached piece: {name}")
            return obj
        return None

    def detach_piece(self, target_square: Square, piece_type: PieceType) -> CollisionObject:
        """
        Detach the currently attached piece at the target square.
        Re-adds it as a collision object at the new position.
        """
        self._attached_piece = None
        return self.add_piece(target_square, piece_type)

    def update_from_occupancy(
        self,
        occupancy: dict[str, tuple[PieceType, bool]],
    ) -> None:
        """
        Update all piece collision objects from an occupancy map.

        Args:
            occupancy: Dict mapping algebraic squares to (PieceType, is_occupied).
        """
        # Remove all existing piece objects
        piece_keys = [k for k in self._objects if k.startswith("piece_")]
        for k in piece_keys:
            del self._objects[k]

        # Add pieces at occupied squares
        for sq_name, (piece_type, occupied) in occupancy.items():
            if occupied:
                sq = Square.from_algebraic(sq_name)
                self.add_piece(sq, piece_type)

    @property
    def all_objects(self) -> list[CollisionObject]:
        return list(self._objects.values())

    @property
    def piece_objects(self) -> list[CollisionObject]:
        return [o for o in self._objects.values() if o.name.startswith("piece_")]

    def _to_robot(self, point_board: np.ndarray) -> np.ndarray:
        """Transform a point from board frame to robot frame."""
        p = np.ones(4)
        p[:3] = point_board[:3]
        return (self.T_robot_board @ p)[:3]
