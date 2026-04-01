"""
Camera calibration pipeline.

Supports:
  1. Intrinsic calibration from checkerboard images
  2. Extrinsic calibration (board pose) via PnP or AprilTags
  3. Hand-eye calibration stub for eye-in-hand / eye-to-hand setups

All results are stored as CalibrationBundle objects and persisted to disk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from chess_core.interfaces import CalibrationBundle
from calibration.transform_manager import save_calibration

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for calibration procedures."""
    checkerboard_size: tuple[int, int] = (7, 7)  # inner corners
    checkerboard_square_mm: float = 30.0
    min_frames_intrinsic: int = 10
    apriltag_family: str = "tag36h11"
    apriltag_size_mm: float = 50.0
    save_dir: str = "data/calibration"


class CameraCalibrator:
    """
    Intrinsic camera calibration using checkerboard pattern.

    Usage:
        calibrator = CameraCalibrator(CalibrationConfig())
        for image in calibration_images:
            calibrator.add_frame(image)
        result = calibrator.calibrate()
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._image_points: list[np.ndarray] = []
        self._object_points: list[np.ndarray] = []
        self._image_size: Optional[tuple[int, int]] = None
        self._obj_pattern = self._make_object_pattern()

    def _make_object_pattern(self) -> np.ndarray:
        """Create the 3D object points for the checkerboard pattern."""
        rows, cols = self.config.checkerboard_size
        sq_m = self.config.checkerboard_square_mm / 1000.0
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_m
        return objp

    def add_frame(self, image: np.ndarray) -> bool:
        """
        Add a calibration frame. Returns True if checkerboard was found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self._image_size is None:
            self._image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, self.config.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found or corners is None:
            return False

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        self._object_points.append(self._obj_pattern)
        self._image_points.append(corners)

        logger.debug(f"Added calibration frame {len(self._image_points)}")
        return True

    def calibrate(self) -> Optional[dict]:
        """
        Run intrinsic calibration.

        Returns:
            Dict with camera_matrix, dist_coeffs, reprojection_error,
            or None if insufficient frames.
        """
        if len(self._image_points) < self.config.min_frames_intrinsic:
            logger.error(
                f"Need at least {self.config.min_frames_intrinsic} frames, "
                f"got {len(self._image_points)}"
            )
            return None

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._object_points,
            self._image_points,
            self._image_size,
            None,
            None,
        )

        logger.info(f"Intrinsic calibration complete: RMS error = {ret:.4f}px")

        return {
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "reprojection_error": ret,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "n_frames": len(self._image_points),
        }

    def generate_synthetic_frames(
        self,
        n_frames: int = 15,
        image_size: tuple[int, int] = (640, 480),
    ) -> list[np.ndarray]:
        """
        Generate synthetic checkerboard images for testing calibration.

        Returns:
            List of synthetic BGR images with rendered checkerboards.
        """
        rows, cols = self.config.checkerboard_size
        sq_px = 30  # pixels per square
        board_w = (cols + 1) * sq_px
        board_h = (rows + 1) * sq_px

        images = []
        rng = np.random.default_rng(42)

        for i in range(n_frames):
            # Create checkerboard pattern
            board = np.zeros((board_h, board_w), dtype=np.uint8)
            for r in range(rows + 1):
                for c in range(cols + 1):
                    if (r + c) % 2 == 0:
                        y1 = r * sq_px
                        x1 = c * sq_px
                        board[y1:y1 + sq_px, x1:x1 + sq_px] = 255

            # Create full image with board placed randomly
            img = np.full((image_size[1], image_size[0]), 128, dtype=np.uint8)

            # Random offset
            max_x = image_size[0] - board_w - 10
            max_y = image_size[1] - board_h - 10
            ox = rng.integers(10, max(11, max_x))
            oy = rng.integers(10, max(11, max_y))

            # Clip to fit
            bh = min(board_h, image_size[1] - oy)
            bw = min(board_w, image_size[0] - ox)
            img[oy:oy + bh, ox:ox + bw] = board[:bh, :bw]

            # Convert to BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            images.append(img_bgr)

        return images


class ExtrinsicCalibrator:
    """
    Extrinsic calibration: estimate the board pose relative to the camera.

    Uses either:
      - Known 3D-2D correspondences (PnP from detected board corners)
      - AprilTag detection

    The result is T_camera_board (transform from board frame to camera frame).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        board_square_size_m: float = 0.057,
    ) -> None:
        self._K = camera_matrix.copy()
        self._dist = dist_coeffs.copy()
        self._sq_size = board_square_size_m

    def estimate_from_corners(
        self,
        image_corners: np.ndarray,
        board_corners_3d: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Estimate board pose from 4 detected corner points.

        Args:
            image_corners: 4x2 image coordinates of board corners (a1, h1, h8, a8).
            board_corners_3d: Optional 4x3 3D positions. Defaults to standard board corners.

        Returns:
            4x4 T_camera_board transform, or None on failure.
        """
        if board_corners_3d is None:
            w = 7 * self._sq_size  # distance between outer corners
            board_corners_3d = np.array([
                [0, 0, 0],      # a1
                [w, 0, 0],      # h1
                [w, w, 0],      # h8
                [0, w, 0],      # a8
            ], dtype=np.float64)

        image_pts = image_corners.astype(np.float64).reshape(-1, 1, 2)
        obj_pts = board_corners_3d.astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts, image_pts, self._K, self._dist,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if not success:
            logger.warning("PnP solve failed for board extrinsic")
            return None

        # Convert to 4x4 transform
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()

        logger.info("Extrinsic calibration: T_camera_board estimated via PnP")
        return T

    def estimate_reprojection_error(
        self,
        T_camera_board: np.ndarray,
        image_corners: np.ndarray,
        board_corners_3d: Optional[np.ndarray] = None,
    ) -> float:
        """Compute mean reprojection error for a given transform."""
        if board_corners_3d is None:
            w = 7 * self._sq_size
            board_corners_3d = np.array([
                [0, 0, 0], [w, 0, 0], [w, w, 0], [0, w, 0],
            ], dtype=np.float64)

        rvec, _ = cv2.Rodrigues(T_camera_board[:3, :3])
        tvec = T_camera_board[:3, 3]

        projected, _ = cv2.projectPoints(
            board_corners_3d, rvec, tvec, self._K, self._dist
        )
        projected = projected.reshape(-1, 2)
        corners_2d = image_corners.reshape(-1, 2)

        errors = np.linalg.norm(projected - corners_2d, axis=1)
        return float(np.mean(errors))


def build_calibration_bundle(
    intrinsic_result: dict,
    T_camera_board: np.ndarray,
    T_robot_camera: np.ndarray = np.eye(4),
) -> CalibrationBundle:
    """
    Create a complete CalibrationBundle from calibration results.

    Args:
        intrinsic_result: Dict from CameraCalibrator.calibrate()
        T_camera_board: 4x4 from ExtrinsicCalibrator
        T_robot_camera: 4x4 from hand-eye calibration (defaults to identity)

    Returns:
        Complete CalibrationBundle.
    """
    T_robot_board = T_robot_camera @ T_camera_board

    bundle = CalibrationBundle(
        camera_matrix=intrinsic_result["camera_matrix"],
        dist_coeffs=intrinsic_result["dist_coeffs"],
        T_camera_board=T_camera_board,
        T_robot_board=T_robot_board,
        T_robot_camera=T_robot_camera,
        reprojection_error_px=intrinsic_result["reprojection_error"],
        timestamp=time.time(),
        valid=True,
        notes=f"Calibrated from {intrinsic_result['n_frames']} frames",
    )

    return bundle
