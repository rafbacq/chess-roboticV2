"""
Heuristic baselines for manipulation subtasks.

Every RL-trained policy has a paired heuristic baseline for:
  1. Comparison during evaluation (is RL actually better?)
  2. Fallback when the learned policy is unavailable
  3. Data collection for imitation learning bootstrapping

These baselines are intentionally simple and deterministic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from chess_core.interfaces import PIECE_GEOMETRY, GraspCandidate, PieceType, Square
from learning.interfaces import LearningPolicy

logger = logging.getLogger(__name__)


class HeuristicGraspPolicy(LearningPolicy):
    """
    Simple top-down grasp heuristic.

    Strategy:
        1. Move EE directly above piece center
        2. Descend vertically to grasp_height = piece_height * grasp_ratio
        3. Close gripper to piece_width + tolerance
        4. Lift vertically

    This policy outputs actions compatible with ChessGraspEnv.
    """

    def __init__(
        self,
        grasp_ratio: float = 0.65,
        tolerance_mm: float = 3.0,
        descent_speed: float = 0.3,
    ) -> None:
        self._grasp_ratio = grasp_ratio
        self._tolerance_mm = tolerance_mm
        self._descent_speed = descent_speed
        self._phase = "approach_xy"  # approach_xy → descend → close → lift

    def predict(self, observation: dict) -> dict:
        """
        Given an observation dict, produce an action dict.

        Expected observation keys:
            piece_rel_xy: (2,) relative XY of piece to EE
            piece_rel_z: float, relative Z
            gripper_width: float, normalized gripper width
            piece_grasped: bool

        Returns:
            Action dict with "delta_ee" (6,) and "gripper_cmd" (float).
        """
        piece_rel_xy = observation.get("piece_rel_xy", np.zeros(2))
        piece_rel_z = observation.get("piece_rel_z", 0.0)
        gripper_width = observation.get("gripper_width", 1.0)
        grasped = observation.get("piece_grasped", False)

        dist_xy = np.linalg.norm(piece_rel_xy)

        if grasped:
            # Lift phase
            return {
                "delta_ee": np.array([0, 0, 1.0, 0, 0, 0]),  # up
                "gripper_cmd": 1.0,  # stay closed
            }

        if dist_xy > 0.005:
            # Approach XY — move toward piece center
            direction = piece_rel_xy / (dist_xy + 1e-6)
            speed = min(dist_xy * 5, 1.0)
            return {
                "delta_ee": np.array([
                    direction[0] * speed,
                    direction[1] * speed,
                    0, 0, 0, 0,
                ]),
                "gripper_cmd": -1.0,  # keep open
            }

        if piece_rel_z > 0.005:
            # Descend
            return {
                "delta_ee": np.array([0, 0, -self._descent_speed, 0, 0, 0]),
                "gripper_cmd": -1.0,
            }

        # Close gripper
        return {
            "delta_ee": np.zeros(6),
            "gripper_cmd": 1.0,
        }

    def is_available(self) -> bool:
        return True  # always available

    def get_fallback_policy_name(self) -> str:
        return "heuristic_grasp"  # it IS the fallback

    def get_name(self) -> str:
        return "heuristic_grasp"


class HeuristicPlacementPolicy(LearningPolicy):
    """
    Simple vertical placement heuristic.

    Strategy:
        1. Move above target square center at safe height
        2. Descend vertically to board_height + piece_base_clearance
        3. Open gripper
        4. Retreat vertically
    """

    def __init__(self, placement_clearance_m: float = 0.003) -> None:
        self._clearance = placement_clearance_m

    def predict(self, observation: dict) -> dict:
        target_rel_xy = observation.get("target_rel_xy", np.zeros(2))
        target_rel_z = observation.get("target_rel_z", 0.0)
        piece_grasped = observation.get("piece_grasped", True)

        dist_xy = np.linalg.norm(target_rel_xy)

        if not piece_grasped:
            # Already placed — retreat
            return {
                "delta_ee": np.array([0, 0, 1.0, 0, 0, 0]),
                "gripper_cmd": -1.0,
            }

        if dist_xy > 0.003:
            direction = target_rel_xy / (dist_xy + 1e-6)
            speed = min(dist_xy * 3, 0.5)
            return {
                "delta_ee": np.array([
                    direction[0] * speed,
                    direction[1] * speed,
                    0, 0, 0, 0,
                ]),
                "gripper_cmd": 1.0,
            }

        if target_rel_z > self._clearance:
            return {
                "delta_ee": np.array([0, 0, -0.3, 0, 0, 0]),
                "gripper_cmd": 1.0,
            }

        # Release
        return {
            "delta_ee": np.zeros(6),
            "gripper_cmd": -1.0,
        }

    def is_available(self) -> bool:
        return True

    def get_fallback_policy_name(self) -> str:
        return "heuristic_placement"

    def get_name(self) -> str:
        return "heuristic_placement"


class HeuristicRetryPolicy(LearningPolicy):
    """
    Simple retry heuristic after a failed grasp.

    Strategy:
        1. Retreat to safe height
        2. Re-center above piece (with optional small offset)
        3. Attempt grasp again
        Max retries: 3, then request human intervention
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        self._attempt = 0

    def predict(self, observation: dict) -> dict:
        self._attempt += 1

        if self._attempt > self._max_retries:
            return {"action": "request_human", "reason": "max_retries_exceeded"}

        # Add small random offset to avoid repeating same failed grasp
        offset = np.random.uniform(-0.003, 0.003, size=2)

        return {
            "action": "retry",
            "attempt": self._attempt,
            "xy_offset": offset,
            "use_slower_approach": self._attempt >= 2,
            "increase_force": self._attempt >= 3,
        }

    def is_available(self) -> bool:
        return True

    def get_fallback_policy_name(self) -> str:
        return "heuristic_retry"

    def get_name(self) -> str:
        return "heuristic_retry"

    def reset(self) -> None:
        self._attempt = 0
