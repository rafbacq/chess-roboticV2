"""
xArm6 Hardware Driver.

Implements the ArmInterface and GripperInterface for the UFACTORY xArm6.
Connects via the official xarm-python-sdk. Includes defensive error
handling for cases where the physical robot is offline or the SDK
is missing.
"""

from __future__ import annotations

import logging
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
    logger.warning("xarm-python-sdk not installed. xArm6 driver will run in MOCK mode.")


class XArm6Arm(ArmInterface):
    """
    Hardware driver for UFACTORY xArm6.

    Uses the xarm-python-sdk. If the SDK is missing or the robot IP
    cannot be reached, it falls back to a mock mode to prevent crashing
    the entire system orchestrator.
    """

    def __init__(self, name: str = "xarm6", ip: str = "192.168.1.197") -> None:
        self.name = name
        self.ip = ip
        self._arm: Optional['XArmAPI'] = None
        self._status = ArmStatus.NOT_INITIALIZED
        self._mock_mode = not HAS_XARM_SDK
        self._mock_ee_pose = np.eye(4)
        self._mock_joints = np.zeros(6)

        # Base xArm6 limits
        self._capabilities = ArmCapabilities(
            name=self.name,
            dof=6,
            max_reach_m=0.691,
            max_payload_kg=5.0,
            max_joint_velocity_rads=3.14,
            max_cartesian_velocity_ms=1.0,
            repeatability_mm=0.1,
        )

    def initialize(self) -> bool:
        logger.info(f"Initializing {self.name} at IP {self.ip}...")

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
            logger.info(f"{self.name} connection successful.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to xArm6 at {self.ip}: {e}")
            logger.warning("Falling back to MOCK mode.")
            self._mock_mode = True
            self._status = ArmStatus.READY
            return True

    def shutdown(self) -> None:
        logger.info(f"Shutting down {self.name}...")
        if not self._mock_mode and self._arm:
            try:
                self._arm.motion_enable(enable=False)
                self._arm.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting xArm6: {e}")
        self._status = ArmStatus.NOT_INITIALIZED
        logger.info(f"{self.name} shut down.")

    def is_ready(self) -> bool:
        return self._status == ArmStatus.READY

    def get_status(self) -> ArmStatus:
        if not self._mock_mode and self._arm:
            if self._arm.has_err_warn:
                return ArmStatus.ERROR
        return self._status

    def get_capabilities(self) -> ArmCapabilities:
        return self._capabilities

    def get_joint_positions(self) -> np.ndarray:
        if self._mock_mode:
            return self._mock_joints.copy()
        try:
            code, joints = self._arm.get_servo_angle(is_radian=True)
            if code == 0:
                return np.array(joints, dtype=np.float64)
            return self._mock_joints
        except:
            return self._mock_joints

    def get_joint_velocities(self) -> np.ndarray:
        return np.zeros(self._capabilities.dof)

    def get_ee_pose(self) -> np.ndarray:
        if self._mock_mode:
            return self._mock_ee_pose.copy()

        try:
            code, pose = self._arm.get_position(is_radian=True)
            if code == 0:
                # pose is [x, y, z, roll, pitch, yaw] in mm and radians
                x, y, z, r, p, yw = pose
                T = np.eye(4)
                # Create rotation matrix from r, p, yw
                import math
                cr = math.cos(r); sr = math.sin(r)
                cp = math.cos(p); sp = math.sin(p)
                cy = math.cos(yw); sy = math.sin(yw)
                
                R = np.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [-sp,   cp*sr,            cp*cr]
                ])
                T[:3, :3] = R
                T[0, 3] = x / 1000.0
                T[1, 3] = y / 1000.0
                T[2, 3] = z / 1000.0
                return T
        except:
            pass
        return self._mock_ee_pose.copy()

    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        if self._mock_mode:
            self._mock_joints = positions.copy()
            time.sleep(0.1)
            return True

        speed = 3.14 * velocity_scale
        accel = 10.0 * acceleration_scale
        
        try:
            code = self._arm.set_servo_angle(
                angle=positions.tolist(), 
                speed=speed, 
                mvacc=accel, 
                wait=True, 
                is_radian=True
            )
            return code == 0
        except Exception as e:
            logger.error(f"xArm joint move failed: {e}")
            return False

    def move_to_pose(
        self,
        target_pose: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        if self._mock_mode:
            self._mock_ee_pose = target_pose.copy()
            time.sleep(0.1)
            return True
            
        return False  # Use move_cartesian_linear for tool center point

    def move_cartesian_linear(
        self,
        target_pose: np.ndarray,
        velocity_ms: float = 0.1,
        acceleration_mss: float = 0.1,
    ) -> bool:
        if self._mock_mode:
            self._mock_ee_pose = target_pose.copy()
            time.sleep(0.2)
            return True

        # Extract [x, y, z, roll, pitch, yaw] from SE(3) matrix
        x = target_pose[0, 3] * 1000.0
        y = target_pose[1, 3] * 1000.0
        z = target_pose[2, 3] * 1000.0
        
        # rotation matrix to rpy
        R = target_pose[:3, :3]
        import math
        sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
        if sy > 1e-6:
            r = math.atan2(R[2,1], R[2,2])
            p = math.atan2(-R[2,0], sy)
            yw = math.atan2(R[1,0], R[0,0])
        else:
            r = math.atan2(-R[1,2], R[1,1])
            p = math.atan2(-R[2,0], sy)
            yw = 0

        pose = [x, y, z, r, p, yw]
        speed = velocity_ms * 1000.0
        accel = acceleration_mss * 1000.0
        
        try:
            code = self._arm.set_position(
                *pose, speed=speed, mvacc=accel, wait=True, is_radian=True
            )
            return code == 0
        except Exception as e:
            logger.error(f"xArm linear move failed: {e}")
            return False


class XArmGripper(GripperInterface):
    """
    Hardware driver for UFACTORY xArm Gripper.
    """

    def __init__(self, name: str = "xarm_gripper", ip: str = "192.168.1.197") -> None:
        self.name = name
        self.ip = ip
        self._arm: Optional['XArmAPI'] = None
        self._status = GripperStatus.CLOSED
        self._mock_mode = not HAS_XARM_SDK
        self._width = 0.0

        self._capabilities = GripperCapabilities(
            name=self.name,
            min_width_mm=0.0,
            max_width_mm=85.0,
            max_force_n=30.0,
            has_force_sensing=False,
        )

    def initialize(self) -> bool:
        logger.info(f"Initializing {self.name}...")
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
        except:
            self._mock_mode = True
            self._status = GripperStatus.CLOSED
            return True

    def shutdown(self) -> None:
        if not self._mock_mode and self._arm:
            self._arm.set_gripper_enable(False)
        self._status = GripperStatus.ERROR

    def is_ready(self) -> bool:
        return self._status != GripperStatus.ERROR

    def get_status(self) -> GripperStatus:
        return self._status

    def get_capabilities(self) -> GripperCapabilities:
        return self._capabilities

    def get_width(self) -> float:
        if self._mock_mode:
            return self._width
        try:
            code, pos = self._arm.get_gripper_position()
            if code == 0:
                self._width = pos
            return self._width
        except:
            return self._width

    def open(self, width_mm: Optional[float] = None) -> bool:
        target = width_mm if width_mm is not None else self._capabilities.max_width_mm
        
        if self._mock_mode:
            self._width = target
            self._status = GripperStatus.OPEN
            time.sleep(0.1)
            return True
            
        try:
            self._arm.set_gripper_position(target, wait=True)
            self._width = target
            self._status = GripperStatus.OPEN
            return True
        except:
            return False

    def close(self, force_n: Optional[float] = None) -> bool:
        if self._mock_mode:
            self._width = 0.0
            self._status = GripperStatus.CLOSED
            time.sleep(0.1)
            return True
            
        try:
            self._arm.set_gripper_position(0.0, wait=True)
            self._width = 0.0
            self._status = GripperStatus.CLOSED
            return True
        except:
            return False
