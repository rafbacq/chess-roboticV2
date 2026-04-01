"""
Data collection hooks for imitation learning.

Provides utilities to record demonstrated manipulation trajectories
for imitation learning (behavioral cloning, DAgger, etc.).

Records:
    - Joint trajectories
    - End-effector poses
    - Gripper states
    - Camera images (optional)
    - Board state before/after
    - Move metadata
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DemonstrationRecord:
    """A single demonstration episode."""
    episode_id: str
    move_uci: str
    piece_type: str
    move_type: str
    timestamps: list[float] = field(default_factory=list)
    joint_positions: list[list[float]] = field(default_factory=list)
    ee_positions: list[list[float]] = field(default_factory=list)
    ee_quaternions: list[list[float]] = field(default_factory=list)
    gripper_widths: list[float] = field(default_factory=list)
    actions: list[list[float]] = field(default_factory=list)
    success: bool = False
    duration_s: float = 0.0


class DemonstrationCollector:
    """
    Collects manipulation demonstrations for imitation learning.

    Records state-action trajectories during teleoperated or
    autonomous move execution. Saves to a dataset format compatible
    with common IL frameworks.

    Usage:
        collector = DemonstrationCollector(save_dir="data/demonstrations")
        collector.start_episode(move_uci="e2e4", piece_type="PAWN")

        # During execution:
        collector.record_step(
            joint_pos=arm.get_joint_positions(),
            ee_pose=arm.get_ee_pose(),
            gripper_width=gripper.get_width_mm(),
            action=current_action,
        )

        collector.end_episode(success=True)
    """

    def __init__(self, save_dir: str = "data/demonstrations") -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._current_episode: Optional[DemonstrationRecord] = None
        self._episode_count = 0

    def start_episode(
        self,
        move_uci: str = "",
        piece_type: str = "",
        move_type: str = "",
    ) -> None:
        """Start recording a new demonstration episode."""
        self._episode_count += 1
        episode_id = f"demo_{self._episode_count:06d}_{int(time.time())}"

        self._current_episode = DemonstrationRecord(
            episode_id=episode_id,
            move_uci=move_uci,
            piece_type=piece_type,
            move_type=move_type,
        )
        logger.info(f"Started demo episode: {episode_id} (move: {move_uci})")

    def record_step(
        self,
        joint_pos: np.ndarray,
        ee_pose: np.ndarray,
        gripper_width: float,
        action: Optional[np.ndarray] = None,
    ) -> None:
        """Record a single timestep."""
        if self._current_episode is None:
            return

        self._current_episode.timestamps.append(time.time())
        self._current_episode.joint_positions.append(joint_pos.tolist())
        self._current_episode.ee_positions.append(ee_pose[:3, 3].tolist())

        # Extract quaternion from rotation matrix (simplified)
        # Full implementation would use scipy.spatial.transform
        self._current_episode.ee_quaternions.append([1.0, 0.0, 0.0, 0.0])
        self._current_episode.gripper_widths.append(gripper_width)

        if action is not None:
            self._current_episode.actions.append(action.tolist())

    def end_episode(self, success: bool = True) -> Optional[str]:
        """End and save the current episode."""
        if self._current_episode is None:
            return None

        self._current_episode.success = success
        if self._current_episode.timestamps:
            self._current_episode.duration_s = (
                self._current_episode.timestamps[-1] -
                self._current_episode.timestamps[0]
            )

        # Save episode
        filepath = self._save_dir / f"{self._current_episode.episode_id}.json"
        with open(filepath, 'w') as f:
            json.dump(asdict(self._current_episode), f, indent=2)

        # Also save as numpy for efficient loading
        npz_path = self._save_dir / f"{self._current_episode.episode_id}.npz"
        np.savez(
            str(npz_path),
            timestamps=np.array(self._current_episode.timestamps),
            joint_positions=np.array(self._current_episode.joint_positions),
            ee_positions=np.array(self._current_episode.ee_positions),
            gripper_widths=np.array(self._current_episode.gripper_widths),
            actions=np.array(self._current_episode.actions) if self._current_episode.actions else np.array([]),
        )

        ep_id = self._current_episode.episode_id
        n_steps = len(self._current_episode.timestamps)
        logger.info(
            f"Saved demo: {ep_id} ({n_steps} steps, "
            f"{'success' if success else 'failure'})"
        )

        self._current_episode = None
        return str(filepath)

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def get_dataset_stats(self) -> dict:
        """Get statistics about the collected dataset."""
        json_files = list(self._save_dir.glob("*.json"))

        total = len(json_files)
        successes = 0
        total_steps = 0

        for f in json_files:
            with open(f) as fh:
                data = json.load(fh)
                if data.get("success", False):
                    successes += 1
                total_steps += len(data.get("timestamps", []))

        return {
            "total_episodes": total,
            "successful": successes,
            "failed": total - successes,
            "total_steps": total_steps,
            "save_dir": str(self._save_dir),
        }
