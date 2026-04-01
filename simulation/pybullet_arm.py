"""
PyBullet arm and gripper adapters.

Implements ArmInterface and GripperInterface using PyBullet's
joint control and IK solver, enabling the full manipulation stack
to drive a simulated URDF robot arm with physics.
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


class PyBulletArm(ArmInterface):
    """
    ArmInterface implementation backed by PyBullet physics.

    Loads a URDF robot and drives it via position control.
    Uses PyBullet's built-in IK solver for Cartesian moves.

    Usage:
        arm = PyBulletArm(physics_client_id, urdf_path="kuka_iiwa/model.urdf")
        arm.initialize()
        arm.move_to_pose(target_pose)
    """

    def __init__(
        self,
        physics_client: int,
        urdf_path: str = "",
        base_position: tuple[float, float, float] = (0.0, -0.3, 0.0),
        base_orientation: tuple[float, float, float, float] = (0, 0, 0, 1),
        ee_link_index: int = -1,
        use_fixed_base: bool = True,
        position_gain: float = 0.3,
        sim_steps_per_command: int = 120,
    ) -> None:
        self._client = physics_client
        self._urdf_path = urdf_path
        self._base_pos = base_position
        self._base_orn = base_orientation
        self._ee_link_index = ee_link_index
        self._fixed_base = use_fixed_base
        self._position_gain = position_gain
        self._sim_steps = sim_steps_per_command

        self._robot_id: int = -1
        self._num_joints: int = 0
        self._controllable_joints: list[int] = []
        self._joint_lower: np.ndarray = np.array([])
        self._joint_upper: np.ndarray = np.array([])
        self._status = ArmStatus.NOT_INITIALIZED
        self._initialized = False

    def initialize(self) -> bool:
        import pybullet as p

        try:
            if self._urdf_path:
                self._robot_id = p.loadURDF(
                    self._urdf_path,
                    basePosition=self._base_pos,
                    baseOrientation=self._base_orn,
                    useFixedBase=self._fixed_base,
                    physicsClientId=self._client,
                )
            else:
                # Use bundled Kuka IIWA as default
                import pybullet_data
                p.setAdditionalSearchPath(
                    pybullet_data.getDataPath(),
                    physicsClientId=self._client,
                )
                self._robot_id = p.loadURDF(
                    "kuka_iiwa/model.urdf",
                    basePosition=self._base_pos,
                    baseOrientation=self._base_orn,
                    useFixedBase=True,
                    physicsClientId=self._client,
                )

            self._num_joints = p.getNumJoints(
                self._robot_id, physicsClientId=self._client
            )

            # Find controllable (non-fixed) joints
            self._controllable_joints = []
            lowers, uppers = [], []
            for i in range(self._num_joints):
                info = p.getJointInfo(
                    self._robot_id, i, physicsClientId=self._client
                )
                joint_type = info[2]
                if joint_type != p.JOINT_FIXED:
                    self._controllable_joints.append(i)
                    lowers.append(info[8])   # lower limit
                    uppers.append(info[9])   # upper limit

            self._joint_lower = np.array(lowers)
            self._joint_upper = np.array(uppers)

            # Set EE link to last joint if not specified
            if self._ee_link_index < 0:
                self._ee_link_index = self._controllable_joints[-1]

            # Move to a neutral pose
            neutral = (self._joint_lower + self._joint_upper) / 2
            self._set_joint_positions_instant(neutral)

            self._status = ArmStatus.READY
            self._initialized = True

            logger.info(
                f"PyBullet arm initialized: {self._num_joints} joints, "
                f"{len(self._controllable_joints)} controllable, "
                f"EE link={self._ee_link_index}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PyBullet arm: {e}")
            self._status = ArmStatus.ERROR
            return False

    def shutdown(self) -> None:
        self._status = ArmStatus.NOT_INITIALIZED
        self._initialized = False

    def get_status(self) -> ArmStatus:
        return self._status

    def get_capabilities(self) -> ArmCapabilities:
        return ArmCapabilities(
            name="pybullet_kuka",
            dof=len(self._controllable_joints),
            max_reach_m=0.80,
            max_payload_kg=5.0,
            max_joint_velocity_rads=3.14,
            max_cartesian_velocity_ms=0.5,
            repeatability_mm=0.1,
        )

    def get_joint_positions(self) -> np.ndarray:
        import pybullet as p
        positions = []
        for j in self._controllable_joints:
            state = p.getJointState(
                self._robot_id, j, physicsClientId=self._client
            )
            positions.append(state[0])
        return np.array(positions)

    def get_joint_velocities(self) -> np.ndarray:
        import pybullet as p
        velocities = []
        for j in self._controllable_joints:
            state = p.getJointState(
                self._robot_id, j, physicsClientId=self._client
            )
            velocities.append(state[1])
        return np.array(velocities)

    def get_ee_pose(self) -> np.ndarray:
        import pybullet as p
        state = p.getLinkState(
            self._robot_id, self._ee_link_index,
            physicsClientId=self._client,
        )
        pos = state[4]  # world link frame position
        orn = state[5]  # world link frame orientation (quaternion xyzw)

        # Convert to 4x4 SE(3)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos
        return pose

    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        import pybullet as p

        self._status = ArmStatus.MOVING
        max_velocity = 3.14 * velocity_scale

        for j_idx, j_id in enumerate(self._controllable_joints):
            if j_idx < len(positions):
                p.setJointMotorControl2(
                    self._robot_id, j_id,
                    p.POSITION_CONTROL,
                    targetPosition=positions[j_idx],
                    maxVelocity=max_velocity,
                    positionGain=self._position_gain,
                    physicsClientId=self._client,
                )

        # Step simulation until close enough
        for _ in range(self._sim_steps):
            p.stepSimulation(physicsClientId=self._client)

        self._status = ArmStatus.READY
        return True

    def move_to_pose(
        self,
        pose: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        joint_positions = self._solve_ik(pose)
        if joint_positions is None:
            logger.warning("IK failed for target pose")
            return False
        return self.move_to_joint_positions(
            joint_positions, velocity_scale, acceleration_scale
        )

    def move_cartesian_linear(
        self,
        pose: np.ndarray,
        velocity_ms: float = 0.05,
    ) -> bool:
        """
        Move linearly in Cartesian space by interpolating waypoints.
        """
        current_pose = self.get_ee_pose()
        start_pos = current_pose[:3, 3]
        end_pos = pose[:3, 3]

        distance = np.linalg.norm(end_pos - start_pos)
        if distance < 0.001:
            return True

        # Determine number of steps based on distance and velocity
        dt = 1.0 / 240  # sim timestep
        n_steps = max(int(distance / (velocity_ms * dt * self._sim_steps)), 3)
        n_steps = min(n_steps, 50)  # cap

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            interp_pos = start_pos + alpha * (end_pos - start_pos)

            # Interpolated pose with target rotation
            interp_pose = pose.copy()
            interp_pose[:3, 3] = interp_pos

            joints = self._solve_ik(interp_pose)
            if joints is not None:
                self._set_joint_positions_controlled(joints, steps=max(self._sim_steps // n_steps, 10))

        self._status = ArmStatus.READY
        return True

    def stop(self) -> None:
        import pybullet as p
        # Set zero velocity on all joints
        for j in self._controllable_joints:
            state = p.getJointState(
                self._robot_id, j, physicsClientId=self._client
            )
            p.setJointMotorControl2(
                self._robot_id, j,
                p.POSITION_CONTROL,
                targetPosition=state[0],
                maxVelocity=0,
                physicsClientId=self._client,
            )

    def emergency_stop(self) -> None:
        self.stop()
        self._status = ArmStatus.EMERGENCY_STOP

    def recover_from_error(self) -> bool:
        self._status = ArmStatus.READY
        return True

    def is_ready(self) -> bool:
        return self._initialized and self._status == ArmStatus.READY

    # =========================================================================
    # Internal
    # =========================================================================

    def _solve_ik(self, target_pose: np.ndarray) -> Optional[np.ndarray]:
        """Solve IK for a target SE(3) pose using PyBullet's IK solver."""
        import pybullet as p

        pos = target_pose[:3, 3].tolist()
        rot = target_pose[:3, :3]

        # Convert rotation matrix to quaternion
        orn = self._rotation_matrix_to_quaternion(rot)

        joint_positions = p.calculateInverseKinematics(
            self._robot_id,
            self._ee_link_index,
            pos,
            orn,
            lowerLimits=self._joint_lower.tolist(),
            upperLimits=self._joint_upper.tolist(),
            jointRanges=(self._joint_upper - self._joint_lower).tolist(),
            restPoses=((self._joint_lower + self._joint_upper) / 2).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self._client,
        )

        if joint_positions is None:
            return None

        return np.array(joint_positions[:len(self._controllable_joints)])

    def _set_joint_positions_instant(self, positions: np.ndarray) -> None:
        """Teleport joints to positions (no physics)."""
        import pybullet as p
        for j_idx, j_id in enumerate(self._controllable_joints):
            if j_idx < len(positions):
                p.resetJointState(
                    self._robot_id, j_id, positions[j_idx],
                    physicsClientId=self._client,
                )

    def _set_joint_positions_controlled(
        self, positions: np.ndarray, steps: int = 30
    ) -> None:
        """Move to positions via motor control over N sim steps."""
        import pybullet as p

        for j_idx, j_id in enumerate(self._controllable_joints):
            if j_idx < len(positions):
                p.setJointMotorControl2(
                    self._robot_id, j_id,
                    p.POSITION_CONTROL,
                    targetPosition=positions[j_idx],
                    maxVelocity=2.0,
                    positionGain=self._position_gain,
                    physicsClientId=self._client,
                )

        for _ in range(steps):
            p.stepSimulation(physicsClientId=self._client)

    @staticmethod
    def _rotation_matrix_to_quaternion(R: np.ndarray) -> list[float]:
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return [x, y, z, w]


class PyBulletGripper(GripperInterface):
    """
    Gripper implementation using PyBullet constraint-based grasping.

    Simulates a parallel-jaw gripper by creating/removing fixed constraints
    between the EE and nearby objects when closing/opening.
    """

    def __init__(
        self,
        physics_client: int,
        robot_id: int,
        ee_link_index: int,
        max_width_mm: float = 80.0,
        grasp_distance_m: float = 0.03,
    ) -> None:
        self._client = physics_client
        self._robot_id = robot_id
        self._ee_link = ee_link_index
        self._max_width_mm = max_width_mm
        self._grasp_distance = grasp_distance_m

        self._width_mm = max_width_mm
        self._gripping = False
        self._grasp_constraint: Optional[int] = None
        self._grasped_body: Optional[int] = None
        self._initialized = False

    def initialize(self) -> bool:
        self._initialized = True
        logger.info("PyBullet gripper initialized")
        return True

    def shutdown(self) -> None:
        if self._grasp_constraint is not None:
            self.open()
        self._initialized = False

    def get_status(self) -> GripperStatus:
        if self._gripping:
            return GripperStatus.GRIPPING
        if self._width_mm >= self._max_width_mm - 1:
            return GripperStatus.OPEN
        return GripperStatus.CLOSED

    def get_capabilities(self) -> GripperCapabilities:
        return GripperCapabilities(
            name="pybullet_gripper",
            min_width_mm=0.0,
            max_width_mm=self._max_width_mm,
            max_force_n=40.0,
        )

    def open(self, width_mm: Optional[float] = None, speed: float = 0.5) -> bool:
        import pybullet as p

        self._width_mm = width_mm or self._max_width_mm

        # Remove grasp constraint if present
        if self._grasp_constraint is not None:
            try:
                p.removeConstraint(
                    self._grasp_constraint, physicsClientId=self._client
                )
            except Exception:
                pass
            self._grasp_constraint = None
            self._grasped_body = None

        self._gripping = False
        return True

    def close(
        self,
        force_n: Optional[float] = None,
        width_mm: Optional[float] = None,
        speed: float = 0.3,
    ) -> bool:
        import pybullet as p

        self._width_mm = width_mm or 0.0

        # Find nearest body to EE that isn't the robot or ground
        ee_state = p.getLinkState(
            self._robot_id, self._ee_link, physicsClientId=self._client
        )
        ee_pos = ee_state[4]

        # Check for bodies near the EE
        contacts = p.getContactPoints(
            bodyA=self._robot_id,
            physicsClientId=self._client,
        )

        # Also check overlapping objects via AABB
        aabb_min = [
            ee_pos[0] - self._grasp_distance,
            ee_pos[1] - self._grasp_distance,
            ee_pos[2] - self._grasp_distance,
        ]
        aabb_max = [
            ee_pos[0] + self._grasp_distance,
            ee_pos[1] + self._grasp_distance,
            ee_pos[2] + self._grasp_distance,
        ]
        overlaps = p.getOverlappingObjects(
            aabb_min, aabb_max, physicsClientId=self._client
        )

        if overlaps:
            for body_id, link_idx in overlaps:
                # Skip robot itself and ground plane (body 0 often)
                if body_id == self._robot_id:
                    continue
                if body_id == 0:  # ground plane
                    continue

                # Check distance
                body_pos, _ = p.getBasePositionAndOrientation(
                    body_id, physicsClientId=self._client
                )
                dist = np.linalg.norm(np.array(ee_pos) - np.array(body_pos))

                if dist < self._grasp_distance:
                    # Create grasp constraint
                    self._grasp_constraint = p.createConstraint(
                        parentBodyUniqueId=self._robot_id,
                        parentLinkIndex=self._ee_link,
                        childBodyUniqueId=body_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0],
                        physicsClientId=self._client,
                    )
                    p.changeConstraint(
                        self._grasp_constraint,
                        maxForce=force_n or 40.0,
                        physicsClientId=self._client,
                    )
                    self._grasped_body = body_id
                    self._gripping = True
                    logger.debug(f"Grasped body {body_id}")
                    return True

        # No object found to grasp
        self._gripping = False
        return True  # gripper closed successfully, just nothing to grab

    def get_width_mm(self) -> float:
        return self._width_mm

    def is_gripping(self) -> bool:
        return self._gripping

    @property
    def grasped_body_id(self) -> Optional[int]:
        """Get the PyBullet body ID of the grasped object."""
        return self._grasped_body
