"""
Perception manager: coordinates camera, board detection, piece detection,
and move verification into a unified perception pipeline.

This is the glue layer between raw camera data and the orchestrator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from chess_core.interfaces import ChessMove, VerificationResult
from perception.board_detector import BoardDetector
from perception.camera_interface import CameraInterface
from perception.move_verifier import MoveVerifier, VerificationConfig
from perception.piece_detector import PieceDetector

logger = logging.getLogger(__name__)


@dataclass
class PerceptionConfig:
    """Configuration for the perception pipeline."""
    # Board detection
    detection_method: str = "corners"  # corners, apriltag, lines
    # Piece detection
    occupancy_threshold: float = 30.0
    color_threshold: float = 50.0
    # Verification
    verifier: VerificationConfig = field(default_factory=VerificationConfig)
    # Image processing
    warp_size: int = 512  # warped board image size (pixels)
    capture_delay_s: float = 0.5  # delay before capture for settling


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

        self.board_detector = BoardDetector(method=self.config.detection_method)
        self.piece_detector = PieceDetector(
            occupancy_threshold=self.config.occupancy_threshold,
            color_threshold=self.config.color_threshold,
        )
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
        result = self.board_detector.detect(image)
        if not result.found or result.corners is None:
            logger.warning("Board detection failed")
            return None

        self._corners = result.corners

        # Warp to top-down
        warped = self.board_detector.warp_board(image, result, self.config.warp_size)
        if warped is None:
            return None

        self._last_warped = warped

        # Detect pieces
        analysis = self.piece_detector.detect(warped)
        return analysis.get_occupancy_map()

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

        result = self.board_detector.detect(image)
        if not result.found or result.corners is None:
            return False

        warped = self.board_detector.warp_board(image, result, self.config.warp_size)
        if warped is None:
            return False

        analysis = self.piece_detector.detect(warped)
        self.verifier.capture_before(warped, analysis.get_occupancy_map())
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

        det_result = self.board_detector.detect(image)
        if not det_result.found or det_result.corners is None:
            return VerificationResult(
                success=False,
                mismatch_details="Board detection failed for verification",
            )

        warped = self.board_detector.warp_board(
            image, det_result, self.config.warp_size
        )
        if warped is None:
            return VerificationResult(
                success=False,
                mismatch_details="Warp failed for verification",
            )

        return self.verifier.verify(warped, move)

    def get_diagnostic_image(self) -> Optional[np.ndarray]:
        """Get the last warped board image for display."""
        return self._last_warped

    def calibrate_empty_board(self) -> bool:
        """
        Capture images of an empty board to calibrate the piece detector.

        Returns:
            True if calibration succeeded.
        """
        image = self._capture()
        if image is None:
            return False

        result = self.board_detector.detect(image)
        if not result.found:
            return False

        warped = self.board_detector.warp_board(image, result, self.config.warp_size)
        if warped is None:
            return False

        self.piece_detector.calibrate_empty_board(warped)
        logger.info("Perception pipeline calibrated with empty board")
        return True

    def _capture(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera."""
        try:
            frame = self.camera.get_frame()
            if frame is None:
                logger.warning("Camera returned None")
            return frame
        except Exception as e:
            logger.error(f"Camera capture failed: {e}")
            return None
