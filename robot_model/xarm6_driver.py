"""
xArm6 Hardware Driver.

Implements the ArmInterface and GripperInterface for the UFACTORY xArm6.
Connects via the official xarm-python-sdk. Includes defensive error
handling for cases where the physical robot is offline or the SDK
is missing.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np

from robot_model.arm_interface import (
    ArmCapabilities,
    ArmInterface,
    ArmStatus,
    GripperCapabilities,
    GripperInterface,
    GripperStatus,
)

logger = logging.getLogger(__name__)

# Attempt to load xArm SDK
try:
    from xarm.wrapper import XArmAPI
    HAS_XARM_SDK = True
except ImportError:
    HAS_XARM_SDK = False


class XArm6Arm(ArmInterface):
    """
    Hardware driver for UFACTORY xArm6.

    Uses the xarm-python-sdk. If the SDK is missing or the robot IP
    cannot be reached, it falls back to a mock mode to prevent crashing
    the entire system orchestrator.
    """

    def __init__(self, name: str = "xarm6", ip: str = "192.168.1.197") -> None:
        self._name = name
        self.ip = ip
        self._arm: Optional['XArmAPI'] = None
        self._status = ArmStatus.NOT_INITIALIZED
        self._mock_mode = not HAS_XARM_SDK
        self._mock_ee_pose = np.eye(4)
        self._mock_joints = np.zeros(6)

        self._capabilities = ArmCapabilities(
            name=self._name,
            dof=6,
            max_reach_m=0.691,
            max_payload_kg=5.0,
            max_joint_velocity_rads=3.14,
            max_cartesian_velocity_ms=1.0,
            repeatability_mm=0.1,
        )

    # -- Lifecycle ----------------------------------------------------------

    def initialize(self) -> bool:
        logger.info(f"Initializing {self._name} at IP {self.ip}...")
        if self._mock_mode:
            logger.info("MOCK MODE: Simulating xArm6 connection.")
            self._status = ArmStatus.READY
            return True
        try:
            self._arm = XArmAPI(self.ip, is_radian=False)
            self._arm.connect()
            self._arm.motion_enable(enable=True)
            self._arm.set_mode(0)
            self._arm.set_state(state=0)
            self._status = ArmStatus.READY
            logger.info(f"{self._name} connection successful.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to xArm6 at {self.ip}: {e}")
            logger.warning("Falling back to MOCK mode.")
            self._mock_mode = True
            self._status = ArmStatus.READY
            return True

    def shutdown(self) -> None:
        logger.info(f"Shutting down {self._name}...")
        if not self._mock_mode and self._arm:
            try:
                self._arm.motion_enable(enable=False)
                self._arm.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting xArm6: {e}")
        self._status = ArmStatus.NOT_INITIALIZED

    def is_ready(self) -> bool:
        return self._status == ArmStatus.READY

    def get_status(self) -> ArmStatus:
        if not self._mock_mode and self._arm:
            try:
                if self._arm.has_err_warn:
                    return ArmStatus.ERROR
            except Exception:
                pass
        return self._status

    def get_capabilities(self) -> ArmCapabilities:
        return self._capabilities

    # -- Queries ------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        if self._mock_mode:
            return self._mock_joints.copy()
        try:
            code, joints = self._arm.get_servo_angle(is_radian=True)
            if code == 0:
                return np.array(joints[:6], dtype=np.float64)
        except Exception:
            pass
        return self._mock_joints.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return np.zeros(self._capabilities.dof)

    def get_ee_pose(self) -> np.ndarray:
        if self._mock_mode:
            return self._mock_ee_pose.copy()
        try:
            code, pose = self._arm.get_position(is_radian=True)
            if code == 0:
                x, y, z, r, p, yw = pose
                T = np.eye(4)
                cr, sr = math.cos(r), math.sin(r)
                cp, sp = math.cos(p), math.sin(p)
                cy, sy = math.cos(yw), math.sin(yw)
                T[:3, :3] = np.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [-sp,   cp*sr,            cp*cr],
                ])
                T[0, 3] = x / 1000.0
                T[1, 3] = y / 1000.0
                T[2, 3] = z / 1000.0
                return T
        except Exception:
            pass
        return self._mock_ee_pose.copy()

    # -- Motion commands ----------------------------------------------------

    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        if self._mock_mode:
            self._mock_joints = positions.copy()
            return True
        try:
            speed = 3.14 * velocity_scale
            accel = 10.0 * acceleration_scale
            code = self._arm.set_servo_angle(
                angle=positions.tolist(),
                speed=speed, mvacc=accel,
                wait=True, is_radian=True,
            )
            return code == 0
        except Exception as e:
            logger.error(f"xArm joint move failed: {e}")
            return False

    def move_to_pose(
        self,
        pose: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        if self._mock_mode:
            self._mock_ee_pose = pose.copy()
            return True
        # Delegate to Cartesian linear for real hardware
        return self.move_cartesian_linear(pose, velocity_ms=0.1 * velocity_scale)

    def move_cartesian_linear(
        self,
        pose: np.ndarray,
        velocity_ms: float = 0.05,
    ) -> bool:
        if self._mock_mode:
            self._mock_ee_pose = pose.copy()
            return True

        R = pose[:3, :3]
        x = pose[0, 3] * 1000.0
        y = pose[1, 3] * 1000.0
        z = pose[2, 3] * 1000.0
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy > 1e-6:
            r = math.atan2(R[2, 1], R[2, 2])
            p = math.atan2(-R[2, 0], sy)
            yw = math.atan2(R[1, 0], R[0, 0])
        else:
            r = math.atan2(-R[1, 2], R[1, 1])
            p = math.atan2(-R[2, 0], sy)
            yw = 0.0

        speed = velocity_ms * 1000.0
        try:
            code = self._arm.set_position(
                x, y, z, r, p, yw,
                speed=speed, mvacc=speed * 5,
                wait=True, is_radian=True,
            )
            return code == 0
        except Exception as e:
            logger.error(f"xArm linear move failed: {e}")
            return False

    # -- Safety -------------------------------------------------------------

    def stop(self) -> None:
        if not self._mock_mode and self._arm:
            try:
                self._arm.set_state(3)  # pause
            except Exception:
                pass
        self._status = ArmStatus.READY

    def emergency_stop(self) -> None:
        if not self._mock_mode and self._arm:
            try:
                self._arm.emergency_stop()
            except Exception:
                pass
        self._status = ArmStatus.EMERGENCY_STOP

    def recover_from_error(self) -> bool:
        if self._mock_mode:
            self._status = ArmStatus.READY
            return True
        try:
            self._arm.clean_error()
            self._arm.clean_warn()
            self._arm.motion_enable(enable=True)
            self._arm.set_mode(0)
            self._arm.set_state(state=0)
            self._status = ArmStatus.READY
            return True
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False


class XArmGripper(GripperInterface):
    """
    Hardware driver for UFACTORY xArm Gripper.

    Implements all GripperInterface abstract methods. Falls back to
    mock mode if the xArm SDK is not installed.
    """

    def __init__(self, name: str = "xarm_gripper", ip: str = "192.168.1.197") -> None:
        self._name = name
        self.ip = ip
        self._arm: Optional['XArmAPI'] = None
        self._status = GripperStatus.CLOSED
        self._mock_mode = not HAS_XARM_SDK
        self._width_mm = 0.0
        self._gripping = False
        self._initialized = False

        self._capabilities = GripperCapabilities(
            name=self._name,
            min_width_mm=0.0,
            max_width_mm=85.0,
            max_force_n=30.0,
            has_force_sensing=False,
        )

    def initialize(self) -> bool:
        logger.info(f"Initializing {self._name}...")
        self._initialized = True
        if self._mock_mode:
            self._status = GripperStatus.CLOSED
            return True
        try:
            self._arm = XArmAPI(self.ip)
            self._arm.connect()
            self._arm.set_gripper_enable(True)
            self._arm.set_gripper_mode(0)
            self._arm.set_gripper_speed(3000)
            self._status = GripperStatus.CLOSED
            return True
        except Exception:
            self._mock_mode = True
            self._status = GripperStatus.CLOSED
            return True

    def shutdown(self) -> None:
        if not self._mock_mode and self._arm:
            try:
                self._arm.set_gripper_enable(False)
            except Exception:
                pass
        self._status = GripperStatus.ERROR
        self._initialized = False

    def get_status(self) -> GripperStatus:
        return self._status

    def get_capabilities(self) -> GripperCapabilities:
        return self._capabilities

    def open(self, width_mm: Optional[float] = None, speed: float = 0.5) -> bool:
        target = width_mm if width_mm is not None else self._capabilities.max_width_mm
        if self._mock_mode:
            self._width_mm = target
            self._gripping = False
            self._status = GripperStatus.OPEN
            return True
        try:
            self._arm.set_gripper_position(target, wait=True)
            self._width_mm = target
            self._gripping = False
            self._status = GripperStatus.OPEN
            return True
        except Exception:
            return False

    def close(
        self,
        force_n: Optional[float] = None,
        width_mm: Optional[float] = None,
        speed: float = 0.3,
    ) -> bool:
        target = width_mm if width_mm is not None else 0.0
        if self._mock_mode:
            self._width_mm = target
            self._gripping = target < 30.0
            self._status = GripperStatus.GRIPPING if self._gripping else GripperStatus.CLOSED
            return True
        try:
            self._arm.set_gripper_position(target, wait=True)
            self._width_mm = target
            self._gripping = True
            self._status = GripperStatus.GRIPPING
            return True
        except Exception:
            return False

    def get_width_mm(self) -> float:
        if not self._mock_mode and self._arm:
            try:
                code, pos = self._arm.get_gripper_position()
                if code == 0:
                    self._width_mm = pos
            except Exception:
                pass
        return self._width_mm

    def is_gripping(self) -> bool:
        return self._gripping

    def is_ready(self) -> bool:
        """Convenience: not part of abstract interface but used by factory."""
        return self._initialized and self._status != GripperStatus.ERROR
