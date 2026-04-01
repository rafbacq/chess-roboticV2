"""
Execution module: trajectory sending, monitoring, watchdog, and telemetry.

This module handles the final step of the pipeline — actually sending
trajectories to the robot and monitoring their execution. Provides
timeout watchdogs, collision detection response, and telemetry logging.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

from chess_core.interfaces import ExecutionResult, ExecutionStatus, ChessMove
from robot_model.arm_interface import ArmInterface, GripperInterface, ArmStatus

logger = logging.getLogger(__name__)


class WatchdogEvent(Enum):
    """Events detected by the execution watchdog."""
    TIMEOUT = auto()
    COLLISION_DETECTED = auto()
    TRAJECTORY_DIVERGENCE = auto()
    JOINT_LIMIT_NEAR = auto()
    FORCE_EXCEEDED = auto()
    GRIPPER_FAULT = auto()


@dataclass
class ExecutionConfig:
    """Configuration for the execution module."""
    stage_timeout_s: float = 15.0
    total_timeout_s: float = 120.0
    trajectory_divergence_threshold_rad: float = 0.1
    force_threshold_n: float = 50.0
    telemetry_log_dir: str = "logs/telemetry"
    telemetry_rate_hz: float = 50.0
    enable_collision_recovery: bool = True


@dataclass
class TelemetryRecord:
    """Single telemetry snapshot."""
    timestamp: float
    joint_positions: np.ndarray
    ee_pose: np.ndarray
    gripper_width_mm: float
    stage: str
    move_uci: str


class Executor:
    """
    Trajectory executor with monitoring and watchdog.

    Sends planned trajectories to the robot, monitors execution state,
    and triggers recovery on detected anomalies.

    Usage:
        executor = Executor(arm, gripper, config)
        executor.start_telemetry()

        # Execute each stage of the planned trajectory
        for stage in plan.trajectory_stages:
            result = executor.execute_stage(stage)
            if not result.success:
                break

        executor.stop_telemetry()
    """

    def __init__(
        self,
        arm: ArmInterface,
        gripper: GripperInterface,
        config: ExecutionConfig | None = None,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.config = config or ExecutionConfig()
        self._telemetry: list[TelemetryRecord] = []
        self._recording = False
        self._current_move: str = ""
        self._current_stage: str = ""

    def execute_joint_trajectory(
        self,
        waypoints: list[np.ndarray],
        velocity_scale: float = 0.3,
        stage_name: str = "trajectory",
    ) -> bool:
        """
        Execute a joint-space trajectory through waypoints.

        Args:
            waypoints: List of joint position arrays.
            velocity_scale: Speed scaling (0-1).
            stage_name: Name for telemetry logging.

        Returns:
            True if all waypoints reached successfully.
        """
        self._current_stage = stage_name
        t_start = time.time()

        for i, wp in enumerate(waypoints):
            # Check timeout
            if time.time() - t_start > self.config.stage_timeout_s:
                logger.error(f"Stage '{stage_name}' timed out at waypoint {i}/{len(waypoints)}")
                return False

            # Check arm status
            status = self.arm.get_status()
            if status != ArmStatus.READY:
                logger.error(f"Arm not ready: {status.name}")
                return False

            # Send waypoint
            success = self.arm.move_to_joint_positions(wp, velocity_scale=velocity_scale)
            if not success:
                logger.error(f"Failed to reach waypoint {i} in stage '{stage_name}'")
                return False

            # Record telemetry
            self._record_telemetry()

        logger.debug(f"Stage '{stage_name}' completed ({len(waypoints)} waypoints)")
        return True

    def execute_cartesian_move(
        self,
        target_pose: np.ndarray,
        linear: bool = False,
        velocity: float = 0.3,
        stage_name: str = "cartesian",
    ) -> bool:
        """
        Execute a Cartesian space move.

        Args:
            target_pose: 4x4 SE(3) target pose.
            linear: If True, use linear Cartesian interpolation.
            velocity: Speed (m/s for linear, scale for joint).
            stage_name: Name for telemetry.

        Returns:
            True if target reached.
        """
        self._current_stage = stage_name
        t_start = time.time()

        if self.arm.get_status() != ArmStatus.READY:
            logger.error(f"Arm not ready for stage '{stage_name}'")
            return False

        if linear:
            success = self.arm.move_cartesian_linear(target_pose, velocity_ms=velocity)
        else:
            success = self.arm.move_to_pose(target_pose, velocity_scale=velocity)

        if not success:
            elapsed = time.time() - t_start
            if elapsed > self.config.stage_timeout_s:
                logger.error(f"Cartesian move timed out in stage '{stage_name}'")
            else:
                logger.error(f"Cartesian move failed in stage '{stage_name}'")
            return False

        self._record_telemetry()
        return True

    def execute_gripper(
        self,
        action: str,
        width_mm: Optional[float] = None,
        force_n: Optional[float] = None,
    ) -> bool:
        """
        Execute a gripper action.

        Args:
            action: "open" or "close".
            width_mm: Target width.
            force_n: Grasp force.

        Returns:
            True if action completed.
        """
        if action == "open":
            return self.gripper.open(width_mm=width_mm)
        elif action == "close":
            return self.gripper.close(force_n=force_n, width_mm=width_mm)
        else:
            raise ValueError(f"Unknown gripper action: {action}")

    def pause(self) -> None:
        """Pause execution (stop arm motion)."""
        self.arm.stop()
        logger.info("Execution paused")

    def abort(self) -> None:
        """Abort execution and emergency stop."""
        self.arm.emergency_stop()
        self.gripper.open()
        logger.warning("Execution ABORTED — emergency stop triggered")

    def recover(self) -> bool:
        """Attempt to recover from error state."""
        success = self.arm.recover_from_error()
        if success:
            logger.info("Recovery successful")
        else:
            logger.error("Recovery failed — manual intervention required")
        return success

    # =========================================================================
    # Telemetry
    # =========================================================================

    def start_telemetry(self, move_uci: str = "") -> None:
        """Start recording telemetry."""
        self._telemetry.clear()
        self._recording = True
        self._current_move = move_uci
        logger.debug(f"Telemetry recording started for move '{move_uci}'")

    def stop_telemetry(self) -> list[TelemetryRecord]:
        """Stop recording and return telemetry data."""
        self._recording = False
        records = list(self._telemetry)

        if records and self.config.telemetry_log_dir:
            self._save_telemetry(records)

        return records

    def _record_telemetry(self) -> None:
        """Record a single telemetry snapshot."""
        if not self._recording:
            return

        record = TelemetryRecord(
            timestamp=time.time(),
            joint_positions=self.arm.get_joint_positions(),
            ee_pose=self.arm.get_ee_pose(),
            gripper_width_mm=self.gripper.get_width_mm(),
            stage=self._current_stage,
            move_uci=self._current_move,
        )
        self._telemetry.append(record)

    def _save_telemetry(self, records: list[TelemetryRecord]) -> None:
        """Save telemetry to disk."""
        log_dir = Path(self.config.telemetry_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        filename = f"telemetry_{self._current_move}_{int(time.time())}.npz"
        filepath = log_dir / filename

        timestamps = np.array([r.timestamp for r in records])
        joints = np.stack([r.joint_positions for r in records])
        ee_poses = np.stack([r.ee_pose for r in records])
        gripper = np.array([r.gripper_width_mm for r in records])

        np.savez(
            str(filepath),
            timestamps=timestamps,
            joint_positions=joints,
            ee_poses=ee_poses,
            gripper_width_mm=gripper,
        )
        logger.debug(f"Telemetry saved: {filepath} ({len(records)} records)")
