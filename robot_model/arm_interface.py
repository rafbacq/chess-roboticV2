"""
Hardware abstraction interfaces for robotic arms and grippers.

These abstract base classes define the contract that every hardware
driver must implement. The system codes to these interfaces, not to
specific hardware — enabling support for multiple arms and grippers
via configuration only.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ArmStatus(Enum):
    """Robotic arm operational status."""
    READY = auto()
    MOVING = auto()
    ERROR = auto()
    EMERGENCY_STOP = auto()
    NOT_INITIALIZED = auto()


class GripperStatus(Enum):
    """Gripper operational status."""
    OPEN = auto()
    CLOSED = auto()
    GRIPPING = auto()  # closed with object detected
    MOVING = auto()
    ERROR = auto()


@dataclass
class ArmCapabilities:
    """Static capabilities of a robotic arm."""
    name: str
    dof: int
    max_reach_m: float
    max_payload_kg: float
    max_joint_velocity_rads: float
    max_cartesian_velocity_ms: float
    repeatability_mm: float
    has_force_torque: bool = False
    has_collision_detection: bool = False


@dataclass
class GripperCapabilities:
    """Static capabilities of a gripper."""
    name: str
    min_width_mm: float
    max_width_mm: float
    max_force_n: float
    has_force_sensing: bool = False
    has_tactile: bool = False


class ArmInterface(ABC):
    """
    Abstract interface for robotic arm hardware.

    Every arm driver (Franka, xArm, UR, SO-100, simulated, etc.)
    must implement this interface. The rest of the system uses only
    these methods.
    """

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the arm hardware and controllers. Returns True on success."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Safely shut down the arm."""
        ...

    @abstractmethod
    def get_status(self) -> ArmStatus:
        """Get the current operational status."""
        ...

    @abstractmethod
    def get_capabilities(self) -> ArmCapabilities:
        """Get static capabilities of this arm."""
        ...

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions in radians."""
        ...

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities in rad/s."""
        ...

    @abstractmethod
    def get_ee_pose(self) -> np.ndarray:
        """Get current end-effector pose as 4x4 SE(3) in robot base frame."""
        ...

    @abstractmethod
    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        """
        Move to target joint positions.

        Args:
            positions: Target joint angles in radians.
            velocity_scale: Fraction of max velocity (0-1).
            acceleration_scale: Fraction of max acceleration (0-1).

        Returns:
            True if motion completed successfully.
        """
        ...

    @abstractmethod
    def move_to_pose(
        self,
        pose: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        """
        Move end-effector to target Cartesian pose.

        Args:
            pose: 4x4 SE(3) target pose in robot base frame.
            velocity_scale: Fraction of max velocity (0-1).
            acceleration_scale: Fraction of max acceleration (0-1).

        Returns:
            True if motion completed successfully.
        """
        ...

    @abstractmethod
    def move_cartesian_linear(
        self,
        pose: np.ndarray,
        velocity_ms: float = 0.05,
    ) -> bool:
        """
        Move end-effector in a straight line in Cartesian space.
        Critical for approach/retreat motions near the board.

        Args:
            pose: 4x4 SE(3) target pose.
            velocity_ms: Linear velocity in m/s.

        Returns:
            True if motion completed successfully.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Immediately stop all motion."""
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """Emergency stop — halt motors immediately."""
        ...

    @abstractmethod
    def recover_from_error(self) -> bool:
        """Attempt to recover from an error state. Returns True if recovery succeeded."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if arm is ready to receive commands."""
        ...


class GripperInterface(ABC):
    """
    Abstract interface for gripper hardware.

    Supports parallel-jaw grippers with optional force sensing and
    tactile feedback. All grippers must implement at minimum:
    open, close, and width/status queries.
    """

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the gripper hardware. Returns True on success."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Safely shut down the gripper."""
        ...

    @abstractmethod
    def get_status(self) -> GripperStatus:
        """Get current gripper status."""
        ...

    @abstractmethod
    def get_capabilities(self) -> GripperCapabilities:
        """Get static capabilities."""
        ...

    @abstractmethod
    def open(self, width_mm: Optional[float] = None, speed: float = 0.5) -> bool:
        """
        Open the gripper.

        Args:
            width_mm: Target width in mm. None = fully open.
            speed: Speed as fraction of max (0-1).

        Returns:
            True if motion completed.
        """
        ...

    @abstractmethod
    def close(
        self,
        force_n: Optional[float] = None,
        width_mm: Optional[float] = None,
        speed: float = 0.3,
    ) -> bool:
        """
        Close the gripper.

        Args:
            force_n: Grasp force in Newtons. None = default.
            width_mm: Minimum width to close to. None = fully close.
            speed: Closing speed as fraction of max (0-1).

        Returns:
            True if object grasped or gripper fully closed.
        """
        ...

    @abstractmethod
    def get_width_mm(self) -> float:
        """Get current gripper width in mm."""
        ...

    @abstractmethod
    def is_gripping(self) -> bool:
        """Check if the gripper has an object grasped."""
        ...

    def get_grasp_force_n(self) -> Optional[float]:
        """Get current grasp force in Newtons (if force sensing available)."""
        return None

    def get_tactile_data(self) -> Optional[dict]:
        """Get tactile sensor data (if tactile sensing available)."""
        return None


# =============================================================================
# Simulated Implementations (for testing without hardware)
# =============================================================================


class SimulatedArm(ArmInterface):
    """
    Simulated arm for testing motion planning and execution pipelines
    without real hardware.
    """

    def __init__(self, name: str = "sim_arm", dof: int = 6) -> None:
        self._name = name
        self._dof = dof
        self._joint_positions = np.zeros(dof)
        self._ee_pose = np.eye(4)
        self._status = ArmStatus.NOT_INITIALIZED
        self._initialized = False

    def initialize(self) -> bool:
        self._status = ArmStatus.READY
        self._initialized = True
        logger.info(f"Simulated arm '{self._name}' initialized ({self._dof} DOF)")
        return True

    def shutdown(self) -> None:
        self._status = ArmStatus.NOT_INITIALIZED
        self._initialized = False

    def get_status(self) -> ArmStatus:
        return self._status

    def get_capabilities(self) -> ArmCapabilities:
        return ArmCapabilities(
            name=self._name,
            dof=self._dof,
            max_reach_m=0.70,
            max_payload_kg=3.0,
            max_joint_velocity_rads=3.14,
            max_cartesian_velocity_ms=0.5,
            repeatability_mm=0.1,
        )

    def get_joint_positions(self) -> np.ndarray:
        return self._joint_positions.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return np.zeros(self._dof)

    def get_ee_pose(self) -> np.ndarray:
        return self._ee_pose.copy()

    def move_to_joint_positions(
        self, positions: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        self._joint_positions = positions.copy()
        logger.debug(f"SimArm: moved to joints {positions}")
        return True

    def move_to_pose(
        self, pose: np.ndarray,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> bool:
        self._ee_pose = pose.copy()
        logger.debug(f"SimArm: moved to pose, position={pose[:3, 3]}")
        return True

    def move_cartesian_linear(self, pose: np.ndarray, velocity_ms: float = 0.05) -> bool:
        self._ee_pose = pose.copy()
        return True

    def stop(self) -> None:
        logger.debug("SimArm: stop")

    def emergency_stop(self) -> None:
        self._status = ArmStatus.EMERGENCY_STOP
        logger.warning("SimArm: EMERGENCY STOP")

    def recover_from_error(self) -> bool:
        self._status = ArmStatus.READY
        return True

    def is_ready(self) -> bool:
        return self._initialized and self._status == ArmStatus.READY


class SimulatedGripper(GripperInterface):
    """Simulated gripper for testing."""

    def __init__(self, name: str = "sim_gripper") -> None:
        self._name = name
        self._width_mm = 80.0
        self._max_width_mm = 80.0
        self._gripping = False
        self._initialized = False

    def initialize(self) -> bool:
        self._initialized = True
        logger.info(f"Simulated gripper '{self._name}' initialized")
        return True

    def shutdown(self) -> None:
        self._initialized = False

    def get_status(self) -> GripperStatus:
        if self._gripping:
            return GripperStatus.GRIPPING
        if self._width_mm >= self._max_width_mm - 1:
            return GripperStatus.OPEN
        return GripperStatus.CLOSED

    def get_capabilities(self) -> GripperCapabilities:
        return GripperCapabilities(
            name=self._name,
            min_width_mm=0.0,
            max_width_mm=self._max_width_mm,
            max_force_n=40.0,
        )

    def open(self, width_mm: Optional[float] = None, speed: float = 0.5) -> bool:
        self._width_mm = width_mm or self._max_width_mm
        self._gripping = False
        return True

    def close(
        self, force_n: Optional[float] = None,
        width_mm: Optional[float] = None,
        speed: float = 0.3,
    ) -> bool:
        target = width_mm or 0.0
        self._width_mm = target
        # Simulate: if closing past some threshold, assume gripping
        if target < 30.0:
            self._gripping = True
        return True

    def get_width_mm(self) -> float:
        return self._width_mm

    def is_gripping(self) -> bool:
        return self._gripping


# =============================================================================
# Hardware Registry
# =============================================================================


class HardwareRegistry:
    """
    Registry for available arm and gripper implementations.

    Usage:
        registry = HardwareRegistry()
        registry.register_arm("sim", SimulatedArm)
        arm = registry.create_arm("sim", name="test_arm")
    """

    def __init__(self) -> None:
        self._arm_types: dict[str, type[ArmInterface]] = {}
        self._gripper_types: dict[str, type[GripperInterface]] = {}

        # Register built-in simulated implementations
        self.register_arm("simulated", SimulatedArm)
        self.register_gripper("simulated", SimulatedGripper)

    def register_arm(self, name: str, arm_class: type[ArmInterface]) -> None:
        self._arm_types[name] = arm_class
        logger.debug(f"Registered arm type: {name}")

    def register_gripper(self, name: str, gripper_class: type[GripperInterface]) -> None:
        self._gripper_types[name] = gripper_class
        logger.debug(f"Registered gripper type: {name}")

    def create_arm(self, type_name: str, **kwargs) -> ArmInterface:
        if type_name not in self._arm_types:
            available = ", ".join(self._arm_types.keys())
            raise ValueError(f"Unknown arm type '{type_name}'. Available: {available}")
        return self._arm_types[type_name](**kwargs)

    def create_gripper(self, type_name: str, **kwargs) -> GripperInterface:
        if type_name not in self._gripper_types:
            available = ", ".join(self._gripper_types.keys())
            raise ValueError(
                f"Unknown gripper type '{type_name}'. Available: {available}"
            )
        return self._gripper_types[type_name](**kwargs)

    @property
    def available_arms(self) -> list[str]:
        return list(self._arm_types.keys())

    @property
    def available_grippers(self) -> list[str]:
        return list(self._gripper_types.keys())
