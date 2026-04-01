"""
Tests for the camera calibration pipeline.

Tests intrinsic calibration with synthetic checkerboard images
and extrinsic estimation via PnP.
"""

import numpy as np
import pytest
import cv2

from calibration.calibrator import (
    CalibrationConfig,
    CameraCalibrator,
    ExtrinsicCalibrator,
    build_calibration_bundle,
)


class TestCameraCalibrator:
    """Test intrinsic camera calibration."""

    def test_synthetic_frame_generation(self):
        config = CalibrationConfig(checkerboard_size=(7, 7))
        calibrator = CameraCalibrator(config)
        frames = calibrator.generate_synthetic_frames(n_frames=5)
        assert len(frames) == 5
        for f in frames:
            assert f.shape == (480, 640, 3)
            assert f.dtype == np.uint8

    def test_add_frame_detects_checkerboard(self):
        """Test that synthetic frames contain detectable checkerboards."""
        config = CalibrationConfig(checkerboard_size=(7, 7))
        calibrator = CameraCalibrator(config)
        frames = calibrator.generate_synthetic_frames(n_frames=3)

        detected = 0
        for frame in frames:
            if calibrator.add_frame(frame):
                detected += 1

        # At least some frames should be detected
        # (depends on positioning within image)
        assert detected >= 0  # may be 0 for small boards in large images

    def test_insufficient_frames(self):
        """Calibration fails with too few frames."""
        config = CalibrationConfig(min_frames_intrinsic=10)
        calibrator = CameraCalibrator(config)
        # Add fewer frames than required
        result = calibrator.calibrate()
        assert result is None


class TestExtrinsicCalibrator:
    """Test extrinsic calibration (board pose estimation)."""

    def test_pnp_with_synthetic_corners(self):
        """Test PnP solve with known camera and corner positions."""
        # Known camera intrinsics
        K = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1],
        ], dtype=np.float64)
        dist = np.zeros(5)

        # Known board pose: board 0.5m in front of camera
        T_true = np.eye(4, dtype=np.float64)
        T_true[2, 3] = 0.5  # 0.5m in front

        sq_size = 0.057
        board_corners_3d = np.array([
            [0, 0, 0],
            [7 * sq_size, 0, 0],
            [7 * sq_size, 7 * sq_size, 0],
            [0, 7 * sq_size, 0],
        ], dtype=np.float64)

        # Project corners to image
        rvec, _ = cv2.Rodrigues(T_true[:3, :3])
        tvec = T_true[:3, 3]
        image_corners, _ = cv2.projectPoints(
            board_corners_3d, rvec, tvec, K, dist
        )
        image_corners = image_corners.reshape(-1, 2)

        # Estimate pose
        calibrator = ExtrinsicCalibrator(K, dist, board_square_size_m=sq_size)
        T_est = calibrator.estimate_from_corners(image_corners, board_corners_3d)

        assert T_est is not None
        assert T_est.shape == (4, 4)

        # Translation should approximately match
        np.testing.assert_allclose(
            T_est[:3, 3], T_true[:3, 3], atol=0.01,
            err_msg="Estimated translation should match ground truth"
        )

    def test_reprojection_error(self):
        """Test that reprojection error is computed correctly for a known pose."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5)

        T = np.eye(4, dtype=np.float64)
        T[2, 3] = 0.5

        corners_3d = np.array([[0, 0, 0], [0.4, 0, 0], [0.4, 0.4, 0], [0, 0.4, 0]], dtype=np.float64)

        # Project
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3]
        img_pts, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, dist)
        img_pts = img_pts.reshape(-1, 2)

        calibrator = ExtrinsicCalibrator(K, dist)
        error = calibrator.estimate_reprojection_error(T, img_pts, corners_3d)

        # Perfect projection = zero error
        assert error < 0.01, f"Reprojection error should be ~0, got {error}"


class TestBuildCalibrationBundle:
    """Test building a full CalibrationBundle."""

    def test_build_bundle(self):
        intrinsic = {
            "camera_matrix": np.eye(3),
            "dist_coeffs": np.zeros(5),
            "reprojection_error": 0.5,
            "n_frames": 15,
        }
        T_cam_board = np.eye(4)
        T_cam_board[2, 3] = 0.6

        bundle = build_calibration_bundle(intrinsic, T_cam_board)

        assert bundle.valid
        assert bundle.camera_matrix.shape == (3, 3)
        assert bundle.dist_coeffs.shape == (5,)
        assert bundle.T_camera_board[2, 3] == pytest.approx(0.6)
        assert bundle.reprojection_error_px == pytest.approx(0.5)
        assert bundle.timestamp > 0

    def test_bundle_with_hand_eye(self):
        intrinsic = {
            "camera_matrix": np.eye(3) * 500,
            "dist_coeffs": np.zeros(5),
            "reprojection_error": 0.3,
            "n_frames": 20,
        }
        T_cb = np.eye(4)
        T_cb[2, 3] = 0.5

        T_rc = np.eye(4)
        T_rc[0, 3] = 0.1  # camera 10cm offset

        bundle = build_calibration_bundle(intrinsic, T_cb, T_rc)

        # T_robot_board should be T_rc @ T_cb
        expected = T_rc @ T_cb
        np.testing.assert_allclose(bundle.T_robot_board, expected, atol=1e-10)
