"""
Board detector: find the chessboard in a camera image.

Uses classical computer vision (OpenCV) to locate the chessboard
via corner detection, line detection, or AprilTag-based localization.
Outputs the four outer corners of the board in image coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoardDetectionResult:
    """Result of board detection in an image."""
    found: bool
    corners: Optional[np.ndarray] = None  # 4x2 image coords (a1, h1, h8, a8 outer corners)
    homography: Optional[np.ndarray] = None  # 3x3 image→board homography
    confidence: float = 0.0
    method: str = ""


class BoardDetector:
    """
    Detects and localizes the chessboard in a camera image.

    Supports multiple detection methods:
        - "corners": OpenCV chessboard corner detection
        - "apriltag": AprilTag-based detection (4 tags at board corners)
        - "lines": Hough line detection + intersection

    Usage:
        detector = BoardDetector(method="corners")
        result = detector.detect(image)
        if result.found:
            warped = detector.warp_board(image, result)
    """

    def __init__(
        self,
        method: str = "corners",
        board_inner_corners: tuple[int, int] = (7, 7),
        apriltag_ids: Optional[list[int]] = None,
    ) -> None:
        self._method = method
        self._inner_corners = board_inner_corners
        self._apriltag_ids = apriltag_ids or [0, 1, 2, 3]  # IDs for a1, h1, h8, a8

    def detect(self, image: np.ndarray) -> BoardDetectionResult:
        """
        Detect the chessboard in the given image.

        Args:
            image: BGR image from camera.

        Returns:
            BoardDetectionResult with corner positions and homography.
        """
        if self._method == "corners":
            return self._detect_corners(image)
        elif self._method == "apriltag":
            return self._detect_apriltags(image)
        elif self._method == "lines":
            return self._detect_lines(image)
        else:
            raise ValueError(f"Unknown detection method: {self._method}")

    def _detect_corners(self, image: np.ndarray) -> BoardDetectionResult:
        """Detect board using OpenCV chessboard corner detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            self._inner_corners,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found or corners is None:
            return BoardDetectionResult(found=False, method="corners")

        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        # Extract the four outer corners from the inner corner grid
        n_cols, n_rows = self._inner_corners
        outer_corners = np.array([
            corners[0, 0],                      # a1 region
            corners[n_cols - 1, 0],             # h1 region
            corners[(n_rows - 1) * n_cols + (n_cols - 1), 0],  # h8 region
            corners[(n_rows - 1) * n_cols, 0],  # a8 region
        ], dtype=np.float32)

        # Compute homography to ideal board coordinates
        ideal_corners = self._get_ideal_corners()
        H, _ = cv2.findHomography(outer_corners, ideal_corners)

        return BoardDetectionResult(
            found=True,
            corners=outer_corners,
            homography=H,
            confidence=0.9,
            method="corners",
        )

    def _detect_apriltags(self, image: np.ndarray) -> BoardDetectionResult:
        """Detect board using AprilTag markers at the four corners."""
        try:
            from pupil_apriltags import Detector
        except ImportError:
            logger.warning(
                "pupil_apriltags not installed. Install with: pip install pupil-apriltags"
            )
            return BoardDetectionResult(found=False, method="apriltag")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detector = Detector(families="tag36h11")
        detections = detector.detect(gray)

        # Find our 4 target tag IDs
        tag_centers = {}
        for det in detections:
            if det.tag_id in self._apriltag_ids:
                tag_centers[det.tag_id] = det.center

        if len(tag_centers) < 4:
            logger.debug(f"Only found {len(tag_centers)}/4 AprilTags")
            return BoardDetectionResult(found=False, method="apriltag")

        # Order: a1, h1, h8, a8
        try:
            corners = np.array([
                tag_centers[self._apriltag_ids[0]],
                tag_centers[self._apriltag_ids[1]],
                tag_centers[self._apriltag_ids[2]],
                tag_centers[self._apriltag_ids[3]],
            ], dtype=np.float32)
        except KeyError:
            return BoardDetectionResult(found=False, method="apriltag")

        ideal_corners = self._get_ideal_corners()
        H, _ = cv2.findHomography(corners, ideal_corners)

        return BoardDetectionResult(
            found=True,
            corners=corners,
            homography=H,
            confidence=0.95,
            method="apriltag",
        )

    def _detect_lines(self, image: np.ndarray) -> BoardDetectionResult:
        """
        Detect board using Hough line detection.
        Finds two groups of parallel lines and computes intersections.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

        if lines is None or len(lines) < 4:
            return BoardDetectionResult(found=False, method="lines")

        # This is a simplified stub — full implementation would cluster lines
        # into two perpendicular groups and find the board grid
        logger.debug(f"Found {len(lines)} lines, board extraction not yet implemented")
        return BoardDetectionResult(found=False, method="lines", confidence=0.0)

    def warp_board(
        self,
        image: np.ndarray,
        result: BoardDetectionResult,
        output_size: int = 800,
    ) -> Optional[np.ndarray]:
        """
        Warp the board region to a top-down square view.

        Args:
            image: Original camera image.
            result: Board detection result with homography.
            output_size: Output image size (square).

        Returns:
            Warped top-down view of the board, or None.
        """
        if not result.found or result.corners is None:
            return None

        ideal = self._get_ideal_corners(output_size)
        H, _ = cv2.findHomography(result.corners, ideal)

        warped = cv2.warpPerspective(image, H, (output_size, output_size))
        return warped

    def draw_detection(
        self,
        image: np.ndarray,
        result: BoardDetectionResult,
    ) -> np.ndarray:
        """Draw detection overlay on the image for diagnostics."""
        vis = image.copy()

        if not result.found or result.corners is None:
            cv2.putText(vis, "Board NOT detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return vis

        # Draw corners
        corner_labels = ["a1", "h1", "h8", "a8"]
        for i, (label, pt) in enumerate(zip(corner_labels, result.corners)):
            pt_int = tuple(pt.astype(int))
            cv2.circle(vis, pt_int, 8, (0, 255, 0), -1)
            cv2.putText(vis, label, (pt_int[0] + 10, pt_int[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw board outline
        pts = result.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

        cv2.putText(
            vis,
            f"Board detected ({result.method}, conf={result.confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        return vis

    @staticmethod
    def _get_ideal_corners(size: float = 800.0) -> np.ndarray:
        """Get ideal corner positions for a top-down board view."""
        return np.array([
            [0, size],        # a1 (bottom-left)
            [size, size],     # h1 (bottom-right)
            [size, 0],        # h8 (top-right)
            [0, 0],           # a8 (top-left)
        ], dtype=np.float32)
