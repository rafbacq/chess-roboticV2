"""
Tests for the perception pipeline: camera interface, piece detection,
move verification, and the perception manager.

Uses synthetic images (SimulatedCamera) so no real hardware needed.
"""

import numpy as np
import pytest
import cv2

from perception.camera_interface import SimulatedCamera, CameraInfo
from perception.piece_detector import PieceDetector, BoardAnalysis
from perception.move_verifier import MoveVerifier, VerificationConfig
from chess_core.interfaces import ChessMove, MoveType, PieceColor, PieceType, Square


# =========================================================================
# SimulatedCamera Tests
# =========================================================================

class TestSimulatedCamera:
    def test_initialize(self):
        cam = SimulatedCamera(width=640, height=480)
        assert cam.initialize()

    def test_get_frame_shape(self):
        cam = SimulatedCamera(width=320, height=240)
        cam.initialize()
        frame = cam.get_frame()
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_get_camera_info(self):
        cam = SimulatedCamera(width=640, height=480)
        cam.initialize()
        info = cam.get_camera_info()
        assert isinstance(info, CameraInfo)
        assert info.width == 640
        assert info.height == 480
        assert info.fx == pytest.approx(500.0)

    def test_camera_matrix_shape(self):
        cam = SimulatedCamera()
        cam.initialize()
        info = cam.get_camera_info()
        K = info.camera_matrix
        assert K.shape == (3, 3)
        assert K[0, 0] == info.fx
        assert K[1, 1] == info.fy

    def test_set_piece_positions(self):
        cam = SimulatedCamera()
        cam.initialize()
        cam.set_piece_positions({"e2": "pawn", "d1": "queen"})
        frame = cam.get_frame()
        assert frame is not None

    def test_frame_count_increments(self):
        cam = SimulatedCamera()
        cam.initialize()
        assert cam._frame_count == 0
        cam.get_frame()
        assert cam._frame_count == 1
        cam.get_frame()
        assert cam._frame_count == 2


# =========================================================================
# PieceDetector Tests
# =========================================================================

def _make_warped_board(occupied_squares: set[str], size: int = 512) -> np.ndarray:
    """Generate a synthetic warped board image with pieces on specific squares."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // 8

    for rank in range(8):
        for file in range(8):
            if (file + rank) % 2 == 1:
                color = (200, 200, 180)
            else:
                color = (80, 120, 80)

            x1 = file * sq
            y1 = (7 - rank) * sq
            cv2.rectangle(img, (x1, y1), (x1 + sq, y1 + sq), color, -1)

            sq_name = f"{chr(ord('a') + file)}{rank + 1}"
            if sq_name in occupied_squares:
                cx = x1 + sq // 2
                cy = y1 + sq // 2
                cv2.circle(img, (cx, cy), sq // 4, (220, 220, 255), -1)

    return img


class TestPieceDetector:
    def test_calibrate_empty_board(self):
        detector = PieceDetector()
        empty = _make_warped_board(set())
        detector.calibrate_empty_board(empty)
        assert len(detector._empty_baselines) == 64

    def test_detect_occupied_squares(self):
        detector = PieceDetector(occupancy_threshold=15.0)
        empty = _make_warped_board(set())
        detector.calibrate_empty_board(empty)

        occupied_set = {"e2", "d7", "a1"}
        board_img = _make_warped_board(occupied_set)
        analysis = detector.detect(board_img)

        assert isinstance(analysis, BoardAnalysis)
        for sq in occupied_set:
            assert analysis.squares[sq].is_occupied, f"{sq} should be occupied"

    def test_detect_empty_squares(self):
        detector = PieceDetector(occupancy_threshold=15.0)
        empty = _make_warped_board(set())
        detector.calibrate_empty_board(empty)

        analysis = detector.detect(empty)
        for sq_name, sq_data in analysis.squares.items():
            assert not sq_data.is_occupied, f"{sq_name} should be empty"

    def test_occupancy_map(self):
        detector = PieceDetector(occupancy_threshold=15.0)
        empty = _make_warped_board(set())
        detector.calibrate_empty_board(empty)

        occupied_set = {"e4", "d5"}
        board_img = _make_warped_board(occupied_set)
        analysis = detector.detect(board_img)

        occ_map = analysis.get_occupancy_map()
        assert isinstance(occ_map, dict)
        assert len(occ_map) == 64

    def test_draw_occupancy(self):
        detector = PieceDetector()
        empty = _make_warped_board(set())
        detector.calibrate_empty_board(empty)

        board_img = _make_warped_board({"e2"})
        analysis = detector.detect(board_img)
        vis = detector.draw_occupancy(board_img, analysis)
        assert vis.shape == board_img.shape


# =========================================================================
# MoveVerifier Tests
# =========================================================================

class TestMoveVerifier:
    def test_verify_no_before_image(self):
        verifier = MoveVerifier(VerificationConfig(save_diagnostics=False))
        move = ChessMove(
            source=Square(4, 1), target=Square(4, 3),
            piece=PieceType.PAWN, color=PieceColor.WHITE, move_type=MoveType.NORMAL,
        )
        after = _make_warped_board({"e4"})
        result = verifier.verify(after, move)
        assert not result.success
        assert "before" in result.mismatch_details.lower()

    def test_verify_successful_move(self):
        verifier = MoveVerifier(VerificationConfig(
            change_threshold=10.0,
            save_diagnostics=False,
            min_confidence=0.1,
        ))
        move = ChessMove(
            source=Square(4, 1), target=Square(4, 3),
            piece=PieceType.PAWN, color=PieceColor.WHITE, move_type=MoveType.NORMAL,
        )
        before = _make_warped_board({"e2"})
        verifier.capture_before(before)

        after = _make_warped_board({"e4"})
        result = verifier.verify(after, move)
        assert result.source_empty
        assert result.target_occupied

    def test_verify_no_change_fails(self):
        verifier = MoveVerifier(VerificationConfig(
            change_threshold=10.0,
            save_diagnostics=False,
        ))
        move = ChessMove(
            source=Square(4, 1), target=Square(4, 3),
            piece=PieceType.PAWN, color=PieceColor.WHITE, move_type=MoveType.NORMAL,
        )
        board = _make_warped_board({"e2"})
        verifier.capture_before(board)
        result = verifier.verify(board, move)
        assert not result.success


# =========================================================================
# Integration: perception_manager imports correctly
# =========================================================================

class TestPerceptionManagerImport:
    def test_import(self):
        from perception.perception_manager import PerceptionManager, PerceptionConfig
        assert PerceptionManager is not None
        assert PerceptionConfig is not None
