"""
Simulation scene builder.

Constructs the chess manipulation scene for simulation environments:
board, pieces, tray, robot arm, camera, and collision objects.

Supports both PyBullet (for quick prototyping) and Isaac Sim (for
production training and digital twin).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from board_state.board_model import BoardConfig, BoardModel
from chess_core.interfaces import PIECE_GEOMETRY, PieceColor, PieceType, Square

logger = logging.getLogger(__name__)


@dataclass
class SimSceneConfig:
    """Configuration for the simulation scene."""
    backend: str = "pybullet"          # "pybullet" or "isaac_sim"
    board_config: BoardConfig = field(default_factory=BoardConfig)
    robot_urdf: str = ""
    gravity: float = -9.81
    timestep: float = 1.0 / 240
    gui: bool = True
    camera_position: tuple[float, float, float] = (0.2, 0.2, 0.6)  # above board center
    camera_target: tuple[float, float, float] = (0.2, 0.2, 0.0)    # board center


class SimSceneBuilder:
    """
    Builds the chess simulation scene.

    Creates the complete environment including:
        - Board surface with colored squares
        - Chess pieces as collision objects
        - Captured piece tray
        - Camera for rendering
        - Robot arm (loaded from URDF)

    Usage:
        builder = SimSceneBuilder(SimSceneConfig())
        scene = builder.build()
        builder.reset_to_starting_position()
    """

    def __init__(self, config: SimSceneConfig | None = None) -> None:
        self.config = config or SimSceneConfig()
        self._board = BoardModel(self.config.board_config)
        self._piece_ids: dict[str, int] = {}  # square_name → sim object ID
        self._sim_client = None
        self._initialized = False

    def build(self) -> bool:
        """Build the complete simulation scene."""
        if self.config.backend == "pybullet":
            return self._build_pybullet()
        elif self.config.backend == "isaac_sim":
            return self._build_isaac_sim()
        else:
            raise ValueError(f"Unknown sim backend: {self.config.backend}")

    def _build_pybullet(self) -> bool:
        """Build scene using PyBullet."""
        try:
            import pybullet as p
            import pybullet_data
        except ImportError:
            logger.error(
                "PyBullet not installed. Install with: pip install pybullet"
            )
            return False

        mode = p.GUI if self.config.gui else p.DIRECT
        self._sim_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        p.setTimeStep(self.config.timestep)

        # Ground plane
        p.loadURDF("plane.urdf")

        # Build board surface
        self._build_board_pybullet(p)

        # Place starting position pieces
        self.reset_to_starting_position()

        self._initialized = True
        logger.info("PyBullet simulation scene built successfully")
        return True

    def _build_board_pybullet(self, p) -> None:
        """Create the chess board in PyBullet."""
        sq_size = self._board.config.square_size_m
        board_w = self._board.config.board_width_m
        board_d = self._board.config.board_depth_m
        board_h = self._board.config.board_thickness_m

        # Board as a flat box
        board_half = [board_w / 2, board_d / 2, board_h / 2]
        board_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=board_half)
        board_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=board_half,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],  # wood color
        )

        cx = board_w / 2 - sq_size / 2
        cy = board_d / 2 - sq_size / 2
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=board_col,
            baseVisualShapeIndex=board_vis,
            basePosition=[cx, cy, -board_h / 2],
        )

    def reset_to_starting_position(self) -> None:
        """Place pieces in standard chess starting position."""
        # Clear existing pieces
        self.clear_pieces()

        # Standard starting position
        back_rank = [
            PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
            PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK,
        ]

        # White pieces (ranks 1-2)
        for file in range(8):
            self.add_piece(Square(file, 0), back_rank[file], PieceColor.WHITE)
            self.add_piece(Square(file, 1), PieceType.PAWN, PieceColor.WHITE)

        # Black pieces (ranks 7-8)
        for file in range(8):
            self.add_piece(Square(file, 7), back_rank[file], PieceColor.BLACK)
            self.add_piece(Square(file, 6), PieceType.PAWN, PieceColor.BLACK)

        logger.info("Pieces reset to starting position")

    def add_piece(
        self,
        square: Square,
        piece_type: PieceType,
        color: PieceColor,
    ) -> Optional[int]:
        """Add a piece to the simulation at the given square."""
        if self.config.backend == "pybullet":
            return self._add_piece_pybullet(square, piece_type, color)
        return None

    def _add_piece_pybullet(
        self,
        square: Square,
        piece_type: PieceType,
        color: PieceColor,
    ) -> int:
        """Add a piece as a cylinder in PyBullet."""
        import pybullet as p

        geom = PIECE_GEOMETRY[piece_type]
        radius = geom["radius_mm"] / 1000.0
        height = geom["height_mm"] / 1000.0

        center = self._board.get_square_center(square)

        # Color
        if color == PieceColor.WHITE:
            rgba = [0.9, 0.9, 0.85, 1.0]
        else:
            rgba = [0.15, 0.12, 0.1, 1.0]

        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)

        body_id = p.createMultiBody(
            baseMass=0.05,  # ~50g for a chess piece
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[center[0], center[1], height / 2],
        )

        self._piece_ids[square.algebraic] = body_id
        return body_id

    def remove_piece(self, square: Square) -> None:
        """Remove a piece from the simulation."""
        sq_name = square.algebraic
        if sq_name in self._piece_ids:
            if self.config.backend == "pybullet":
                import pybullet as p
                p.removeBody(self._piece_ids[sq_name])
            del self._piece_ids[sq_name]

    def clear_pieces(self) -> None:
        """Remove all pieces from the simulation."""
        if self.config.backend == "pybullet":
            import pybullet as p
            for body_id in self._piece_ids.values():
                try:
                    p.removeBody(body_id)
                except Exception:
                    pass
        self._piece_ids.clear()

    def get_camera_image(
        self,
        width: int = 640,
        height: int = 480,
    ) -> Optional[np.ndarray]:
        """Render a camera image from the simulated overhead camera."""
        if self.config.backend == "pybullet":
            import pybullet as p

            view_matrix = p.computeViewMatrix(
                cameraEyePosition=self.config.camera_position,
                cameraTargetPosition=self.config.camera_target,
                cameraUpVector=[0, 1, 0],
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width / height,
                nearVal=0.01,
                farVal=2.0,
            )

            _, _, pixels, _, _ = p.getCameraImage(
                width, height, view_matrix, proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )

            # PyBullet returns RGBA
            img = np.array(pixels, dtype=np.uint8).reshape(height, width, 4)
            return img[:, :, :3]  # Drop alpha, return RGB

        return None

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        if self.config.backend == "pybullet":
            import pybullet as p
            p.stepSimulation()

    def shutdown(self) -> None:
        """Clean up the simulation."""
        if self.config.backend == "pybullet" and self._sim_client is not None:
            import pybullet as p
            p.disconnect()
            self._sim_client = None
        self._initialized = False
        logger.info("Simulation shut down")

    @property
    def is_initialized(self) -> bool:
        return self._initialized
