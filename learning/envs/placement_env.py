"""
Placement refinement RL environment.

Task: precisely center a held chess piece on the target square,
compensating for gripper pose error and piece orientation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chess_core.interfaces import PIECE_GEOMETRY, PieceType

logger = logging.getLogger(__name__)


class ChessPlacementEnv(gym.Env):
    """
    Gymnasium environment for learning precise piece placement.

    The agent holds a piece above the target square and must
    place it accurately despite positional noise and orientation error.

    Observation Space (18-dim):
        - target_offset_xy: (2,) — XY offset from target center
        - ee_height: (1,) — height above board
        - piece_type_onehot: (6,) — piece type
        - gripper_holding: (1,) — whether piece is held
        - ee_pose: (7,) — position (3) + quaternion (4)
        - step_fraction: (1,) — fraction of max episode steps

    Action Space (4-dim):
        - delta_xy: (2,) — XY correction
        - delta_z: (1,) — descent/ascent
        - gripper_cmd: (1,) — open/close
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_steps: int = 100,
        position_noise_m: float = 0.008,
        success_threshold_m: float = 0.003,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.position_noise = position_noise_m
        self.success_threshold = success_threshold_m
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._step_count = 0
        self._target_xy = np.zeros(2)
        self._ee_pos = np.zeros(3)
        self._piece_type = PieceType.PAWN
        self._holding = True
        self._placed = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._holding = True
        self._placed = False

        self._piece_type = self.np_random.choice(list(PieceType))
        self._target_xy = np.zeros(2)  # target is at origin

        piece_height = PIECE_GEOMETRY[self._piece_type]["height_mm"] / 1000.0
        self._ee_pos = np.array([
            self.np_random.uniform(-self.position_noise, self.position_noise),
            self.np_random.uniform(-self.position_noise, self.position_noise),
            piece_height + 0.01,  # slightly above
        ])

        return self._get_obs(), self._get_info()

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Apply XY correction
        self._ee_pos[0] += action[0] * 0.002
        self._ee_pos[1] += action[1] * 0.002
        self._ee_pos[2] += action[2] * 0.002
        self._ee_pos[2] = max(0.001, self._ee_pos[2])

        reward = -0.1  # step penalty
        terminated = False
        truncated = self._step_count >= self.max_steps

        offset = np.linalg.norm(self._ee_pos[:2] - self._target_xy)

        # Shaped reward: distance to center
        reward += -offset * 10

        # Release action
        if action[3] < -0.5 and self._holding and self._ee_pos[2] < 0.01:
            self._holding = False
            if offset < self.success_threshold:
                reward += 10.0  # success!
                self._placed = True
                terminated = True
            else:
                reward += -5.0  # placed off-center
                terminated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self):
        offset = self._ee_pos[:2] - self._target_xy
        piece_onehot = np.zeros(6, dtype=np.float32)
        piece_onehot[list(PieceType).index(self._piece_type)] = 1.0

        obs = np.concatenate([
            offset.astype(np.float32),
            np.array([self._ee_pos[2]], dtype=np.float32),
            piece_onehot,
            np.array([float(self._holding)], dtype=np.float32),
            self._ee_pos.astype(np.float32),
            np.array([1.0, 0, 0, 0], dtype=np.float32),  # quaternion placeholder
            np.array([self._step_count / self.max_steps], dtype=np.float32),
        ])
        return obs

    def _get_info(self):
        return {
            "step": self._step_count,
            "piece_type": self._piece_type.name,
            "offset_mm": float(np.linalg.norm(self._ee_pos[:2] - self._target_xy) * 1000),
            "placed": self._placed,
            "holding": self._holding,
            "height_mm": float(self._ee_pos[2] * 1000),
        }

    def render(self):
        return None

    def close(self):
        pass
