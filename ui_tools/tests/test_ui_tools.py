"""
Tests for ui_tools: board display, telemetry viewer, formatting utilities.
"""

import numpy as np
import pytest

from ui_tools.board_display import (
    render_board_ascii,
    format_telemetry_table,
    format_move_history,
    format_joint_state,
    format_ee_pose,
    TelemetrySummary,
)
from ui_tools.telemetry_viewer import (
    analyze_trajectory,
    format_analysis,
    TrajectoryAnalysis,
)


class TestBoardDisplay:
    def test_starting_position(self):
        board = render_board_ascii()
        assert "a" in board
        assert "h" in board
        assert "R" in board  # white rook
        assert "k" in board  # black king

    def test_unicode_symbols(self):
        board = render_board_ascii(use_unicode=True)
        assert "♔" in board or "♚" in board

    def test_empty_board(self):
        board = render_board_ascii("8/8/8/8/8/8/8/8")
        assert "·" in board

    def test_highlight_squares(self):
        board = render_board_ascii(highlight_squares={"e2", "e4"})
        assert "*" in board or ">" in board

    def test_custom_fen(self):
        board = render_board_ascii("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
        assert "P" in board


class TestTelemetryTable:
    def test_format_table(self):
        records = [
            TelemetrySummary(
                move_uci="e2e4", duration_s=2.5, n_waypoints=45,
                max_joint_velocity_rads=1.2, max_ee_velocity_ms=0.15,
                peak_gripper_force_n=5.0,
                stages=["pre_grasp", "approach", "grasp", "lift", "transit", "place"],
            ),
        ]
        table = format_telemetry_table(records)
        assert "e2e4" in table
        assert "pre_grasp" in table

    def test_empty_records(self):
        table = format_telemetry_table([])
        assert "Move" in table  # header still present


class TestMoveHistory:
    def test_format_history(self):
        moves = [
            {"uci": "e2e4", "status": "SUCCESS", "duration_s": 2.1, "verified": True},
            {"uci": "e7e5", "status": "SUCCESS", "duration_s": 2.3, "verified": True},
        ]
        history = format_move_history(moves)
        assert "e2e4" in history
        assert "✓" in history

    def test_truncation(self):
        moves = [{"uci": f"move{i}", "status": "OK", "duration_s": 1.0, "verified": True}
                 for i in range(30)]
        history = format_move_history(moves, max_display=5)
        assert "hidden" in history


class TestJointState:
    def test_format_joint_state(self):
        names = ["j1", "j2", "j3", "j4", "j5", "j6"]
        positions = np.array([0.1, 0.2, -0.3, 0.0, 1.0, -0.5])
        output = format_joint_state(names, positions)
        assert "j1" in output
        assert "0.1000" in output

    def test_with_limits(self):
        names = ["j1", "j2"]
        positions = np.array([0.5, -0.5])
        vels = np.array([0.1, 0.0])
        lo = np.array([-3.14, -3.14])
        hi = np.array([3.14, 3.14])
        output = format_joint_state(names, positions, vels, lo, hi)
        assert "%" in output


class TestEEPose:
    def test_identity(self):
        output = format_ee_pose(np.eye(4))
        assert "0.0000" in output
        assert "Position" in output

    def test_translated(self):
        pose = np.eye(4)
        pose[0, 3] = 0.3
        pose[2, 3] = 0.5
        output = format_ee_pose(pose)
        assert "0.3000" in output
        assert "0.5000" in output


class TestTrajectoryAnalysis:
    def test_analyze_simple_trajectory(self):
        telemetry = {
            "timestamps": np.array([0.0, 0.1, 0.2, 0.3]),
            "joint_positions": np.array([
                [0, 0, 0, 0, 0, 0],
                [0.1, 0, 0, 0, 0, 0],
                [0.2, 0, 0, 0, 0, 0],
                [0.3, 0, 0, 0, 0, 0],
            ]),
            "ee_poses": np.stack([np.eye(4)] * 4),
            "gripper_width_mm": np.array([50, 50, 10, 10]),
        }
        result = analyze_trajectory(telemetry, "e2e4")
        assert isinstance(result, TrajectoryAnalysis)
        assert result.total_duration_s == pytest.approx(0.3)
        assert result.n_samples == 4
        assert result.max_joint_velocity_rads > 0
        assert result.gripper_transitions == 1  # open → closed

    def test_format_output(self):
        analysis = TrajectoryAnalysis(
            move_uci="d2d4", total_duration_s=1.5, n_samples=30,
            max_joint_velocity_rads=2.0, mean_joint_velocity_rads=0.8,
            max_ee_velocity_ms=0.2, path_length_m=0.35,
            joint_range_used_rad=np.ones(6) * 0.5, gripper_transitions=2,
        )
        output = format_analysis(analysis)
        assert "d2d4" in output
        assert "350.0 mm" in output

    def test_single_sample(self):
        telemetry = {
            "timestamps": np.array([0.0]),
            "joint_positions": np.array([[0, 0, 0, 0, 0, 0]]),
            "ee_poses": np.stack([np.eye(4)]),
            "gripper_width_mm": np.array([50]),
        }
        result = analyze_trajectory(telemetry)
        assert result.n_samples == 1
        assert result.total_duration_s == 0
