"""
Unit tests for board model: coordinate system, square mapping, geometry.
"""

import numpy as np
import pytest

from board_state.board_model import BoardConfig, BoardModel
from chess_core.interfaces import PieceType, Square


class TestBoardModel:
    """Tests for the BoardModel coordinate system and geometry."""

    @pytest.fixture
    def board(self):
        return BoardModel(BoardConfig(square_size_m=0.057))

    def test_a1_at_origin(self, board):
        """a1 (file=0, rank=0) should be at origin."""
        center = board.get_square_center(Square(0, 0))
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0])

    def test_h1_position(self, board):
        """h1 (file=7, rank=0) should be at (7*0.057, 0, 0)."""
        center = board.get_square_center(Square(7, 0))
        expected_x = 7 * 0.057
        np.testing.assert_allclose(center, [expected_x, 0.0, 0.0], atol=1e-10)

    def test_a8_position(self, board):
        """a8 (file=0, rank=7) should be at (0, 7*0.057, 0)."""
        center = board.get_square_center(Square(0, 7))
        expected_y = 7 * 0.057
        np.testing.assert_allclose(center, [0.0, expected_y, 0.0], atol=1e-10)

    def test_h8_position(self, board):
        """h8 (file=7, rank=7) = (7*sq, 7*sq, 0)."""
        center = board.get_square_center(Square(7, 7))
        expected = 7 * 0.057
        np.testing.assert_allclose(center, [expected, expected, 0.0], atol=1e-10)

    def test_e4_position(self, board):
        """e4 (file=4, rank=3) = (4*0.057, 3*0.057, 0)."""
        center = board.get_square_center(Square(4, 3))
        np.testing.assert_allclose(
            center,
            [4 * 0.057, 3 * 0.057, 0.0],
            atol=1e-10,
        )

    def test_square_distance(self, board):
        """Distance between adjacent squares should be square_size."""
        dist = board.square_distance_m(
            Square.from_algebraic("e4"),
            Square.from_algebraic("e5"),
        )
        np.testing.assert_allclose(dist, 0.057, atol=1e-10)

    def test_diagonal_distance(self, board):
        """Diagonal distance should be sqrt(2) * square_size."""
        dist = board.square_distance_m(
            Square.from_algebraic("e4"),
            Square.from_algebraic("f5"),
        )
        np.testing.assert_allclose(dist, 0.057 * np.sqrt(2), atol=1e-10)

    def test_all_64_squares_unique(self, board):
        """All 64 square centers should be distinct."""
        poses = board.get_all_square_poses()
        assert len(poses) == 64

        positions = [tuple(p.position.tolist()) for p in poses]
        assert len(set(positions)) == 64

    def test_board_width(self, board):
        assert board.config.board_width_m == 8 * 0.057

    def test_board_depth(self, board):
        assert board.config.board_depth_m == 8 * 0.057

    def test_piece_top_z(self, board):
        """Piece top Z should equal piece height in meters."""
        z = board.get_piece_top_z(PieceType.KING)
        assert z == pytest.approx(0.095, abs=1e-6)  # 95mm king

    def test_grasp_z(self, board):
        """Grasp Z at 65% of king height."""
        z = board.get_grasp_z(PieceType.KING, grasp_ratio=0.65)
        expected = 95.0 * 0.65 / 1000.0
        assert z == pytest.approx(expected, abs=1e-6)

    def test_neighbors_corner(self, board):
        """a1 should have 3 neighbors."""
        neighbors = board.get_neighboring_squares(Square.from_algebraic("a1"))
        assert len(neighbors) == 3

    def test_neighbors_center(self, board):
        """e4 should have 8 neighbors."""
        neighbors = board.get_neighboring_squares(Square.from_algebraic("e4"))
        assert len(neighbors) == 8

    def test_neighbors_edge(self, board):
        """a4 should have 5 neighbors."""
        neighbors = board.get_neighboring_squares(Square.from_algebraic("a4"))
        assert len(neighbors) == 5

    def test_tray_position_increments(self, board):
        """Sequential tray positions should be at different locations."""
        pos1 = board.get_tray_position(0)
        pos2 = board.get_tray_position(1)
        assert not np.allclose(pos1, pos2)

    def test_approach_pose_is_se3(self, board):
        """Approach pose should be a valid SE(3) matrix."""
        pose = board.get_approach_pose(
            Square.from_algebraic("e4"),
            PieceType.PAWN,
        )
        assert pose.shape == (4, 4)
        # Bottom row should be [0, 0, 0, 1]
        np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1])
        # Rotation should be orthogonal
        R = pose[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_board_corners(self, board):
        """Board corners should form a rectangle."""
        corners = board.get_board_corners()
        assert corners.shape == (4, 3)
        # All Z should be 0
        np.testing.assert_allclose(corners[:, 2], 0.0)

    def test_safe_waypoint_above_board(self, board):
        """Safe waypoint should be above the board center."""
        wp = board.get_safe_waypoint(height_m=0.15)
        assert wp[2] == 0.15
        # Should be near board center
        assert wp[0] == pytest.approx(board.config.board_width_m / 2, abs=1e-10)


class TestBoardConfig:
    def test_default_fide_standard(self):
        config = BoardConfig()
        assert config.square_size_m == 0.057
        assert config.num_files == 8
        assert config.num_ranks == 8

    def test_custom_size(self):
        config = BoardConfig(square_size_m=0.050)
        board = BoardModel(config)
        center = board.get_square_center(Square(1, 0))
        assert center[0] == pytest.approx(0.050, abs=1e-10)
