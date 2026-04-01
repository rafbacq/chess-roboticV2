"""
Motion planning interface and implementations.

Provides an abstract planner interface with two concrete implementations:
  1. PyBulletPlanner — uses PyBullet IK + linear interpolation
  2. WaypointPlanner — simple staged waypoint planner using board model geometry

Both return PlanResult objects compatible with the execution module.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from board_state.board_model import BoardModel
from chess_core.interfaces import (
    ChessMove,
    GraspCandidate,
    PieceType,
    PlanRequest,
    PlanResult,
    Square,
)
from robot_model.arm_interface import ArmInterface

logger = logging.getLogger(__name__)


class MotionPlannerInterface(ABC):
    """
    Abstract motion planner interface.

    Planners take a current robot state and a target pose, and produce
    a trajectory (sequence of joint configurations) that avoids collisions.
    """

    @abstractmethod
    def plan_to_pose(
        self,
        target_pose: np.ndarray,
        max_time_s: float = 5.0,
    ) -> PlanResult:
        """Plan a trajectory from current state to target pose."""
        ...

    @abstractmethod
    def plan_cartesian_path(
        self,
        waypoints: list[np.ndarray],
        max_time_s: float = 5.0,
    ) -> PlanResult:
        """Plan a Cartesian-space path through waypoints."""
        ...

    @abstractmethod
    def plan_pick_place(
        self,
        request: PlanRequest,
    ) -> PlanResult:
        """Plan a complete pick-and-place sequence for a chess move."""
        ...


class WaypointPlanner(MotionPlannerInterface):
    """
    Simple waypoint-based motion planner.

    Generates staged trajectories using the board model geometry:
    pre-grasp → approach → grasp → lift → transit → pre-place → place → retreat

    Each stage is a Cartesian waypoint. No collision checking —
    relies on staged heights to avoid collisions.
    """

    def __init__(
        self,
        arm: ArmInterface,
        board: BoardModel,
        safe_height_m: float = 0.15,
        approach_clearance_m: float = 0.05,
        T_robot_board: np.ndarray | None = None,
    ) -> None:
        self.arm = arm
        self.board = board
        self._safe_height = safe_height_m
        self._approach_clearance = approach_clearance_m
        self._T_rb = T_robot_board if T_robot_board is not None else np.eye(4)

    def plan_to_pose(
        self,
        target_pose: np.ndarray,
        max_time_s: float = 5.0,
    ) -> PlanResult:
        t0 = time.time()
        return PlanResult(
            success=True,
            trajectory_stages=[{"type": "move_to_pose", "pose": target_pose}],
            planning_time_s=time.time() - t0,
        )

    def plan_cartesian_path(
        self,
        waypoints: list[np.ndarray],
        max_time_s: float = 5.0,
    ) -> PlanResult:
        t0 = time.time()
        stages = [{"type": "cartesian", "pose": wp} for wp in waypoints]
        return PlanResult(
            success=True,
            trajectory_stages=stages,
            planning_time_s=time.time() - t0,
        )

    def plan_pick_place(self, request: PlanRequest) -> PlanResult:
        """
        Generate a complete staged trajectory for a chess move.

        Stages:
            1. Open gripper
            2. Move to pre-grasp (above source at safe height)
            3. Descend to grasp height
            4. Close gripper
            5. Lift to safe height
            6. Transit to above target at safe height
            7. Descend to placement height
            8. Open gripper
            9. Retreat to safe height
        """
        t0 = time.time()
        move = request.move
        grasp = request.grasp_candidate

        source_board = self.board.get_square_center(move.source)
        target_board = self.board.get_square_center(move.target)
        grasp_z = self.board.get_grasp_z(move.piece)
        piece_top = self.board.get_piece_top_z(move.piece)

        source_robot = self._to_robot(source_board)
        target_robot = self._to_robot(target_board)

        stages = []

        # 1. Open gripper
        stages.append({
            "type": "gripper",
            "action": "open",
            "width_mm": grasp.finger_width_mm + 5.0,
        })

        # 2. Pre-grasp: above source at safe height
        stages.append({
            "type": "cartesian",
            "pose": self._top_down_pose(source_robot, self._safe_height),
            "speed": "transit",
            "label": "pre_grasp",
        })

        # 3. Approach: descend to grasp height
        stages.append({
            "type": "cartesian_linear",
            "pose": self._top_down_pose(source_robot, grasp_z),
            "speed": "approach",
            "label": "approach",
        })

        # 4. Close gripper
        stages.append({
            "type": "gripper",
            "action": "close",
            "width_mm": grasp.finger_width_mm,
            "force_n": 10.0,
        })

        # 5. Lift
        stages.append({
            "type": "cartesian_linear",
            "pose": self._top_down_pose(source_robot, self._safe_height),
            "speed": "retreat",
            "label": "lift",
        })

        # 6. Transit to target
        stages.append({
            "type": "cartesian",
            "pose": self._top_down_pose(target_robot, self._safe_height),
            "speed": "transit",
            "label": "transit",
        })

        # 7. Place: descend
        place_z = 0.003 + piece_top * 0.05  # slight clearance
        stages.append({
            "type": "cartesian_linear",
            "pose": self._top_down_pose(target_robot, place_z),
            "speed": "approach",
            "label": "place",
        })

        # 8. Open gripper
        stages.append({
            "type": "gripper",
            "action": "open",
        })

        # 9. Retreat
        stages.append({
            "type": "cartesian_linear",
            "pose": self._top_down_pose(target_robot, self._safe_height),
            "speed": "retreat",
            "label": "retreat",
        })

        planning_time = time.time() - t0
        logger.info(
            f"Waypoint plan generated: {len(stages)} stages, "
            f"{planning_time * 1000:.1f}ms"
        )

        return PlanResult(
            success=True,
            trajectory_stages=stages,
            planning_time_s=planning_time,
        )

    def _to_robot(self, point_board: np.ndarray) -> np.ndarray:
        p = np.ones(4)
        p[:3] = point_board[:3]
        return (self._T_rb @ p)[:3]

    @staticmethod
    def _top_down_pose(pos_robot: np.ndarray, height: float) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 0] = 1.0
        pose[1, 1] = -1.0
        pose[2, 2] = -1.0
        pose[0, 3] = pos_robot[0]
        pose[1, 3] = pos_robot[1]
        pose[2, 3] = height
        return pose
