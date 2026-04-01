"""
Perception manager: coordinates camera, board detection, piece detection,
and move verification into a unified perception pipeline.

This is the glue layer between raw camera data and the orchestrator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from chess_core.interfaces import ChessMove, VerificationResult
from perception.board_detector import BoardDetector, BoardDetectorConfig
from perception.camera_interface import CameraInterface
from perception.move_verifier import MoveVerifier, VerificationConfig
from perception.piece_detector import PieceDetector, PieceDetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class PerceptionConfig:
    """Configuration for the perception pipeline."""
    board_detector: BoardDetectorConfig = None
    piece_detector: PieceDetectorConfig = None
    verifier: VerificationConfig = None
    warp_size: int = 512  # warped board image size (pixels)
    capture_delay_s: float = 0.5  # delay before capture for settling

    def __post_init__(self):
        if self.board_detector is None:
            self.board_detector = BoardDetectorConfig()
        if self.piece_detector is None:
            self.piece_detector = PieceDetectorConfig()
        if self.verifier is None:
            self.verifier = VerificationConfig()


class PerceptionManager:
    """
    Unified perception pipeline for the chess robot.

    Workflow:
        1. Capture image from camera
        2. Detect board corners
        3. Warp to top-down view
        4. Detect piece occupancy
        5. Verify moves by before/after comparison

    Usage:
        pm = PerceptionManager(camera, config)
        occupancy = pm.get_occupancy()
        pm.capture_before_move()
        # ... robot executes move ...
        result = pm.verify_move(chess_move)
    """

    def __init__(
        self,
        camera: CameraInterface,
        config: PerceptionConfig | None = None,
    ) -> None:
        self.config = config or PerceptionConfig()
        self.camera = camera

        self.board_detector = BoardDetector(self.config.board_detector)
        self.piece_detector = PieceDetector(self.config.piece_detector)
        self.verifier = MoveVerifier(self.config.verifier)

        self._warp_matrix: Optional[np.ndarray] = None
        self._last_warped: Optional[np.ndarray] = None
        self._corners: Optional[np.ndarray] = None

    def capture_and_detect(self) -> Optional[dict[str, bool]]:
        """
        Capture an image and detect piece occupancy.

        Returns:
            Dict mapping square names to occupancy, or None on failure.
        """
        image = self._capture()
        if image is None:
            return None

        # Detect board
        corners = self.board_detector.detect(image)
        if corners is None or len(corners) < 4:
            logger.warning("Board detection failed")
            return None

        self._corners = corners

        # Warp to top-down
        warped = self._warp_board(image, corners)
        if warped is None:
            return None

        self._last_warped = warped

        # Detect pieces
        occupancy = self.piece_detector.detect_occupancy(warped)
        return occupancy

    def get_occupancy(self) -> Optional[dict[str, bool]]:
        """Get current board occupancy from camera."""
        return self.capture_and_detect()

    def capture_before_move(self) -> bool:
        """
        Capture the board state before a move for verification.

        Returns:
            True if capture succeeded.
        """
        time.sleep(self.config.capture_delay_s)

        image = self._capture()
        if image is None:
            return False

        corners = self.board_detector.detect(image)
        if corners is None or len(corners) < 4:
            return False

        warped = self._warp_board(image, corners)
        if warped is None:
            return False

        occupancy = self.piece_detector.detect_occupancy(warped)
        self.verifier.capture_before(warped, occupancy)
        self._last_warped = warped

        logger.debug("Before-move state captured for verification")
        return True

    def verify_move(self, move: ChessMove) -> VerificationResult:
        """
        Verify a move by comparing before/after board images.

        Args:
            move: The move that was executed.

        Returns:
            VerificationResult.
        """
        time.sleep(self.config.capture_delay_s)

        image = self._capture()
        if image is None:
            return VerificationResult(
                success=False,
                mismatch_details="Camera capture failed",
            )

        corners = self.board_detector.detect(image)
        if corners is None or len(corners) < 4:
            return VerificationResult(
                success=False,
                mismatch_details="Board detection failed for verification",
            )

        warped = self._warp_board(image, corners)
        if warped is None:
            return VerificationResult(
                success=False,
                mismatch_details="Warp failed for verification",
            )

        return self.verifier.verify(warped, move)

    def get_diagnostic_image(self) -> Optional[np.ndarray]:
        """Get the last warped board image for display."""
        return self._last_warped

    def _capture(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera."""
        try:
            frame = self.camera.capture()
            if frame is None:
                logger.warning("Camera returned None")
            return frame
        except Exception as e:
            logger.error(f"Camera capture failed: {e}")
            return None

    def _warp_board(
        self,
        image: np.ndarray,
        corners: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Warp the detected board region to a square top-down view."""
        if len(corners) < 4:
            return None

        size = self.config.warp_size
        dst_pts = np.array([
            [0, size],
            [size, size],
            [size, 0],
            [0, 0],
        ], dtype=np.float32)

        src_pts = corners[:4].astype(np.float32).reshape(-1, 2)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (size, size))
        self._warp_matrix = M

        return warped
