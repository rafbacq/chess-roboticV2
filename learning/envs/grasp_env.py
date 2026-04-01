"""
Grasp acquisition RL environment for chess pieces.

This is the primary RL environment for the project. The task is to acquire
a stable grasp on a chess piece under pose uncertainty, varying geometry,
and neighboring piece occlusion.

Compatible with:
  - Gymnasium (for CPU-based training / prototyping)
  - Isaac Lab (for massively parallel GPU training) via adapter

The environment is deliberately self-contained and does NOT import ROS.
It communicates only through numpy arrays and dicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chess_core.interfaces import PIECE_GEOMETRY, PieceType

logger = logging.getLogger(__name__)


@dataclass
class GraspEnvConfig:
    """Configuration for the grasp RL environment."""

    # Episode parameters
    max_steps: int = 200
    dt: float = 0.02  # simulation timestep (s)

    # Observation noise
    piece_pose_noise_xy_m: float = 0.005   # ±5mm
    piece_pose_noise_z_m: float = 0.002    # ±2mm
    piece_pose_noise_yaw_rad: float = 0.05  # ±3°

    # Action scaling
    max_delta_xy_m: float = 0.01    # max XY delta per step
    max_delta_z_m: float = 0.005    # max Z delta per step
    max_delta_rot_rad: float = 0.05  # max rotation delta per step

    # Success criteria
    lift_threshold_m: float = 0.02   # piece must be lifted this high
    stability_hold_steps: int = 25   # hold stable for this many steps

    # Reward weights
    reward_success: float = 10.0
    reward_contact: float = 2.0
    reward_per_step: float = -0.5
    reward_collision: float = -5.0
    reward_knockover: float = -10.0
    reward_distance_shaped: float = 1.0

    # Domain randomization ranges
    dr_piece_friction: tuple[float, float] = (0.3, 1.0)
    dr_gripper_friction: tuple[float, float] = (0.5, 1.2)
    dr_piece_scale: tuple[float, float] = (0.95, 1.05)
    dr_table_height_m: tuple[float, float] = (-0.002, 0.002)

    # Neighbor configuration
    max_neighbors: int = 8
    neighbor_min_dist_m: float = 0.057  # one square away
    neighbor_max_dist_m: float = 0.170  # three squares away


class ChessGraspEnv(gym.Env):
    """
    Gymnasium environment for learning robust chess piece grasping.

    Observation Space (28-dim continuous):
        - piece_pose_relative: (6,) — piece 6-DoF pose relative to EE
        - piece_type_onehot: (6,) — one-hot piece type encoding
        - neighbor_distances: (8,) — distances to 8 nearest neighbor pieces
        - gripper_width: (1,) — current gripper opening (normalized)
        - ee_pose: (7,) — EE position (3) + quaternion (4)

    Action Space (7-dim continuous):
        - delta_ee: (6,) — Cartesian delta (dx, dy, dz, droll, dpitch, dyaw)
        - gripper_command: (1,) — [-1, 1] where -1=open, 1=close

    Reward:
        +10.0   successful grasp (piece lifted >20mm, stable for 25 steps)
         +2.0   first piece contact
         -0.5   per timestep (encourage efficiency)
         -5.0   collision with board or neighbor piece
        -10.0   piece knocked over
         +1.0   shaped: decrease in distance to piece center (per step)

    Episode Termination:
        - Success: piece lifted and stable
        - Failure: piece knocked over, collision, timeout

    This env uses an internal simplified physics model for prototyping.
    For production training, use the Isaac Lab adapter (see isaac_lab_adapter.py).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: GraspEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GraspEnvConfig()
        self.render_mode = render_mode

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Internal state
        self._step_count = 0
        self._piece_type = PieceType.PAWN
        self._piece_pose = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self._ee_pose = np.zeros(7)     # [x, y, z, qw, qx, qy, qz]
        self._gripper_width = 1.0       # normalized [0=closed, 1=fully open]
        self._neighbor_positions: list[np.ndarray] = []
        self._piece_contacted = False
        self._piece_grasped = False
        self._piece_lifted = False
        self._lift_stable_steps = 0
        self._initial_distance = 0.0

        # Domain randomization state
        self._piece_friction = 0.6
        self._piece_scale = 1.0
        self._table_height = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        self._step_count = 0
        self._piece_contacted = False
        self._piece_grasped = False
        self._piece_lifted = False
        self._lift_stable_steps = 0

        # Randomize piece type
        piece_types = list(PieceType)
        self._piece_type = self.np_random.choice(piece_types)

        # Apply domain randomization
        self._piece_friction = self.np_random.uniform(*self.config.dr_piece_friction)
        self._piece_scale = self.np_random.uniform(*self.config.dr_piece_scale)
        self._table_height = self.np_random.uniform(*self.config.dr_table_height_m)

        # Piece pose: centered with randomized offset
        geom = PIECE_GEOMETRY[self._piece_type]
        piece_height_m = geom["height_mm"] / 1000.0 * self._piece_scale

        self._piece_pose = np.array([
            self.np_random.uniform(-self.config.piece_pose_noise_xy_m,
                                   self.config.piece_pose_noise_xy_m),
            self.np_random.uniform(-self.config.piece_pose_noise_xy_m,
                                   self.config.piece_pose_noise_xy_m),
            self._table_height + piece_height_m / 2,  # piece center Z
            0.0, 0.0,
            self.np_random.uniform(-self.config.piece_pose_noise_yaw_rad,
                                   self.config.piece_pose_noise_yaw_rad),
        ], dtype=np.float64)

        # EE starts above and offset from piece
        self._ee_pose = np.array([
            self._piece_pose[0] + self.np_random.uniform(-0.02, 0.02),
            self._piece_pose[1] + self.np_random.uniform(-0.02, 0.02),
            piece_height_m + 0.05,  # 5cm above piece top
            1.0, 0.0, 0.0, 0.0,  # identity quaternion
        ], dtype=np.float64)

        self._gripper_width = 1.0

        # Generate random neighbor positions
        n_neighbors = self.np_random.integers(0, self.config.max_neighbors + 1)
        self._neighbor_positions = []
        for _ in range(n_neighbors):
            angle = self.np_random.uniform(0, 2 * np.pi)
            dist = self.np_random.uniform(
                self.config.neighbor_min_dist_m,
                self.config.neighbor_max_dist_m,
            )
            self._neighbor_positions.append(np.array([
                self._piece_pose[0] + dist * np.cos(angle),
                self._piece_pose[1] + dist * np.sin(angle),
            ]))

        self._initial_distance = np.linalg.norm(
            self._ee_pose[:2] - self._piece_pose[:2]
        )

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take one environment step.

        Args:
            action: 7-dim array [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Apply action: delta EE position
        delta_pos = action[:3] * np.array([
            self.config.max_delta_xy_m,
            self.config.max_delta_xy_m,
            self.config.max_delta_z_m,
        ])
        self._ee_pose[:3] += delta_pos

        # Apply rotation delta (simplified — just rotate yaw for prototype)
        # Full rotation would use quaternion math
        delta_yaw = action[5] * self.config.max_delta_rot_rad
        # Simplified: not updating full quaternion, just tracking for obs

        # Gripper command
        gripper_cmd = action[6]
        if gripper_cmd > 0.3:
            self._gripper_width = max(0.0, self._gripper_width - 0.1)
        elif gripper_cmd < -0.3:
            self._gripper_width = min(1.0, self._gripper_width + 0.1)

        # Simple physics simulation
        reward, terminated, truncated = self._simulate_physics()

        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _simulate_physics(self) -> tuple[float, bool, bool]:
        """
        Simplified physics for prototyping. Returns (reward, terminated, truncated).

        For real training, this should be replaced by Isaac Lab / Isaac Sim physics.
        """
        reward = self.config.reward_per_step
        terminated = False
        truncated = False

        ee_pos = self._ee_pose[:3]
        piece_pos = self._piece_pose[:3]
        dist_xy = np.linalg.norm(ee_pos[:2] - piece_pos[:2])
        dist_z = abs(ee_pos[2] - piece_pos[2])

        geom = PIECE_GEOMETRY[self._piece_type]
        piece_radius_m = geom["radius_mm"] / 1000.0 * self._piece_scale

        # Shaped reward: distance decrease
        current_dist = np.linalg.norm(ee_pos[:2] - piece_pos[:2])
        if self._initial_distance > 0:
            reward += self.config.reward_distance_shaped * (
                self._initial_distance - current_dist
            ) / self._initial_distance

        # Check for contact
        if dist_xy < piece_radius_m and dist_z < 0.02:
            if not self._piece_contacted:
                self._piece_contacted = True
                reward += self.config.reward_contact

            # Check for grasp
            if self._gripper_width < 0.3:
                self._piece_grasped = True

        # Check for collision with neighbors
        for np_pos in self._neighbor_positions:
            if np.linalg.norm(ee_pos[:2] - np_pos) < 0.02:
                reward += self.config.reward_collision
                terminated = True
                return reward, terminated, truncated

        # Check for piece knocked over (EE too low with high lateral force)
        if ee_pos[2] < self._table_height + 0.005 and dist_xy < piece_radius_m * 2:
            if not self._piece_grasped:
                reward += self.config.reward_knockover
                terminated = True
                return reward, terminated, truncated

        # Check for successful lift
        if self._piece_grasped and ee_pos[2] > piece_pos[2] + self.config.lift_threshold_m:
            self._piece_lifted = True
            self._lift_stable_steps += 1

            if self._lift_stable_steps >= self.config.stability_hold_steps:
                reward += self.config.reward_success
                terminated = True
                return reward, terminated, truncated

        # Truncation check
        if self._step_count >= self.config.max_steps:
            truncated = True

        return reward, terminated, truncated

    def _get_observation(self) -> np.ndarray:
        """Construct the 28-dim observation vector."""
        # Piece pose relative to EE (6-dim)
        piece_rel = np.zeros(6, dtype=np.float32)
        piece_rel[:3] = (self._piece_pose[:3] - self._ee_pose[:3]).astype(np.float32)
        piece_rel[3:] = self._piece_pose[3:].astype(np.float32)

        # Piece type one-hot (6-dim)
        piece_onehot = np.zeros(6, dtype=np.float32)
        piece_idx = list(PieceType).index(self._piece_type)
        piece_onehot[piece_idx] = 1.0

        # Neighbor distances (8-dim) — sorted, padded with large value
        neighbor_dists = np.full(8, 1.0, dtype=np.float32)
        for i, np_pos in enumerate(self._neighbor_positions[:8]):
            neighbor_dists[i] = float(np.linalg.norm(self._ee_pose[:2] - np_pos))
        neighbor_dists.sort()

        # Gripper width (1-dim)
        gripper = np.array([self._gripper_width], dtype=np.float32)

        # EE pose (7-dim)
        ee = self._ee_pose.astype(np.float32)

        obs = np.concatenate([piece_rel, piece_onehot, neighbor_dists, gripper, ee])
        assert obs.shape == (28,), f"Obs shape mismatch: {obs.shape}"
        return obs

    def _get_info(self) -> dict:
        """Return episode info dict."""
        return {
            "step": self._step_count,
            "piece_type": self._piece_type.name,
            "contacted": self._piece_contacted,
            "grasped": self._piece_grasped,
            "lifted": self._piece_lifted,
            "lift_stable_steps": self._lift_stable_steps,
            "gripper_width": self._gripper_width,
            "distance_to_piece": float(np.linalg.norm(
                self._ee_pose[:2] - self._piece_pose[:2]
            )),
        }

    def render(self) -> Optional[np.ndarray]:
        """Render the environment (placeholder for visualization)."""
        if self.render_mode == "rgb_array":
            # Return a simple top-down view as RGB array
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            # Draw piece
            cx = int(100 + self._piece_pose[0] * 2000)
            cy = int(100 + self._piece_pose[1] * 2000)
            img[max(0,cy-5):min(200,cy+5), max(0,cx-5):min(200,cx+5)] = [0, 255, 0]
            # Draw EE
            ex = int(100 + self._ee_pose[0] * 2000)
            ey = int(100 + self._ee_pose[1] * 2000)
            img[max(0,ey-3):min(200,ey+3), max(0,ex-3):min(200,ex+3)] = [255, 0, 0]
            return img
        return None

    def close(self) -> None:
        """Clean up."""
        pass
