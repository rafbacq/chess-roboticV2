"""
Unit tests for the calibration transform manager.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from calibration.transform_manager import TransformManager, save_calibration, load_calibration
from chess_core.interfaces import CalibrationBundle


class TestTransformManager:
    """Tests for the TransformManager frame graph."""

    @pytest.fixture
    def tm(self):
        return TransformManager()

    def test_identity_same_frame(self, tm):
        T = tm.get_transform("robot_base", "robot_base")
        np.testing.assert_allclose(T, np.eye(4))

    def test_direct_transform(self, tm):
        T = np.eye(4)
        T[0, 3] = 1.0  # 1m translation in X
        tm.set_transform("robot_base", "camera", T)

        result = tm.get_transform("robot_base", "camera")
        np.testing.assert_allclose(result, T)

    def test_inverse_transform(self, tm):
        T = np.eye(4)
        T[0, 3] = 1.0
        tm.set_transform("robot_base", "camera", T)

        T_inv = tm.get_transform("camera", "robot_base")
        np.testing.assert_allclose(T_inv[0, 3], -1.0)

    def test_composed_transform(self, tm):
        """robot_base ← camera ← board should compose correctly."""
        T_robot_camera = np.eye(4)
        T_robot_camera[0, 3] = 1.0

        T_camera_board = np.eye(4)
        T_camera_board[1, 3] = 2.0

        tm.set_transform("robot_base", "camera", T_robot_camera)
        tm.set_transform("camera", "board", T_camera_board)

        T_robot_board = tm.get_transform("robot_base", "board")
        # Should be T_robot_camera @ T_camera_board
        expected = T_robot_camera @ T_camera_board
        np.testing.assert_allclose(T_robot_board, expected, atol=1e-10)

    def test_point_transform(self, tm):
        T = np.eye(4)
        T[0, 3] = 1.0  # translate 1m in X
        tm.set_transform("world", "local", T)

        p_local = np.array([0.5, 0.0, 0.0])
        p_world = tm.transform_point(p_local, "world", "local")
        np.testing.assert_allclose(p_world, [1.5, 0.0, 0.0])

    def test_batch_point_transform(self, tm):
        T = np.eye(4)
        T[2, 3] = 0.5  # translate 0.5m in Z
        tm.set_transform("world", "board", T)

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = tm.transform_points(points, "world", "board")
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[:, 2], [0.5, 0.5, 0.5])

    def test_missing_path_raises(self, tm):
        tm.set_transform("A", "B", np.eye(4))
        tm.set_transform("C", "D", np.eye(4))
        with pytest.raises(KeyError):
            tm.get_transform("A", "D")  # no path

    def test_consistency_check_passes(self, tm):
        T = np.eye(4)
        T[0, 3] = 1.0
        tm.set_transform("A", "B", T)

        warnings = tm.check_consistency()
        assert len(warnings) == 0

    def test_frames_property(self, tm):
        tm.set_transform("A", "B", np.eye(4))
        assert "A" in tm.frames
        assert "B" in tm.frames

    def test_rotation_transform(self, tm):
        # 90-degree rotation about Z
        T = np.eye(4)
        T[0, 0] = 0; T[0, 1] = -1
        T[1, 0] = 1; T[1, 1] = 0
        tm.set_transform("world", "rotated", T)

        # Point at (1, 0, 0) in rotated frame → (0, 1, 0) in world
        p = tm.transform_point(np.array([1, 0, 0]), "world", "rotated")
        np.testing.assert_allclose(p, [0, 1, 0], atol=1e-10)

    def test_save_and_load(self, tm, tmp_path):
        T = np.eye(4)
        T[0, 3] = 1.5
        tm.set_transform("robot", "camera", T)

        filepath = str(tmp_path / "transforms.yaml")
        tm.save(filepath)

        tm2 = TransformManager()
        tm2.load(filepath)

        result = tm2.get_transform("robot", "camera")
        np.testing.assert_allclose(result[0, 3], 1.5)


class TestCalibrationPersistence:
    def test_save_and_load_calibration(self, tmp_path):
        bundle = CalibrationBundle(
            camera_matrix=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]),
            dist_coeffs=np.array([0.1, -0.2, 0, 0, 0.05]),
            T_camera_board=np.eye(4),
            T_robot_board=np.eye(4),
            T_robot_camera=np.eye(4),
            reprojection_error_px=0.45,
            timestamp=1234567890.0,
            valid=True,
            notes="test calibration",
        )

        filepath = str(tmp_path / "calibration.json")
        save_calibration(bundle, filepath)

        loaded = load_calibration(filepath)
        np.testing.assert_allclose(loaded.camera_matrix, bundle.camera_matrix)
        np.testing.assert_allclose(loaded.dist_coeffs, bundle.dist_coeffs)
        assert loaded.reprojection_error_px == pytest.approx(0.45)
        assert loaded.valid is True
        assert loaded.notes == "test calibration"

    def test_calibration_transform_methods(self):
        T_robot_board = np.eye(4)
        T_robot_board[0, 3] = 0.5  # 0.5m offset in X

        bundle = CalibrationBundle(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            T_camera_board=np.eye(4),
            T_robot_board=T_robot_board,
            T_robot_camera=np.eye(4),
        )

        p_robot = bundle.transform_board_to_robot(np.array([0, 0, 0]))
        np.testing.assert_allclose(p_robot, [0.5, 0, 0])
