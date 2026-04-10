"""
Tests for the MoveIt2 motion planner bridge.

Verifies that MoveIt2Planner implements MotionPlannerInterface,
falls back to WaypointPlanner when ROS 2 is unavailable, and
generates valid trajectory stages.
"""

import numpy as np
import pytest

from board_state.board_model import BoardConfig, BoardModel
from chess_core.interfaces import (
    ChessMove,
    GraspCandidate,
    MoveType,
    PieceColor,
    PieceType,
    PlanRequest,
    Square,
)
from motion_planning.moveit2_planner import MoveIt2Planner
from motion_planning.planner_interface import MotionPlannerInterface
from robot_model.arm_interface import SimulatedArm


@pytest.fixture
def board():
    return BoardModel(BoardConfig())


@pytest.fixture
def arm():
    a = SimulatedArm(name="test_arm")
    a.initialize()
    return a


@pytest.fixture
def planner(arm, board):
    return MoveIt2Planner(arm=arm, board=board, safe_height_m=0.15)


class TestMoveIt2PlannerInterface:
    """Verify MoveIt2Planner implements the abstract interface."""

    def test_is_subclass(self):
        assert issubclass(MoveIt2Planner, MotionPlannerInterface)

    def test_has_required_methods(self):
        methods = {"plan_to_pose", "plan_cartesian_path", "plan_pick_place"}
        planner_methods = set(dir(MoveIt2Planner))
        assert methods.issubset(planner_methods)


class TestMoveIt2PlannerFallback:
    """Test that the planner falls back to WaypointPlanner without ROS 2."""

    def test_plan_to_pose(self, planner):
        target = np.eye(4)
        target[0, 3] = 0.3
        target[2, 3] = 0.2
        result = planner.plan_to_pose(target)
        assert result.success
        assert len(result.trajectory_stages) >= 1
        assert result.planning_time_s >= 0

    def test_plan_cartesian_path(self, planner):
        waypoints = [np.eye(4) for _ in range(3)]
        for i, wp in enumerate(waypoints):
            wp[0, 3] = 0.1 * i
        result = planner.plan_cartesian_path(waypoints)
        assert result.success
        assert len(result.trajectory_stages) == 3

    def test_plan_pick_place(self, planner):
        move = ChessMove(
            source=Square(file=4, rank=1),
            target=Square(file=4, rank=3),
            piece=PieceType.PAWN,
            color=PieceColor.WHITE,
            move_type=MoveType.NORMAL,
        )
        grasp = GraspCandidate(
            pose=np.eye(4),
            piece_type=PieceType.PAWN,
            finger_width_mm=30.0,
            approach_height_mm=50.0,
            score=0.9,
        )
        request = PlanRequest(move=move, grasp_candidate=grasp)
        result = planner.plan_pick_place(request)
        assert result.success
        assert len(result.trajectory_stages) == 9  # full staged trajectory

    def test_stages_have_labels(self, planner):
        move = ChessMove(
            source=Square(file=0, rank=0),
            target=Square(file=0, rank=2),
            piece=PieceType.ROOK,
            color=PieceColor.WHITE,
            move_type=MoveType.NORMAL,
        )
        grasp = GraspCandidate(
            pose=np.eye(4),
            piece_type=PieceType.ROOK,
            finger_width_mm=35.0,
            approach_height_mm=50.0,
            score=0.85,
        )
        request = PlanRequest(move=move, grasp_candidate=grasp)
        result = planner.plan_pick_place(request)
        labels = [s.get("label") for s in result.trajectory_stages if "label" in s]
        expected = {"pre_grasp", "approach", "lift", "transit", "place", "retreat"}
        assert expected.issubset(set(labels))
