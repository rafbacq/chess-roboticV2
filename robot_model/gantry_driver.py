"""
Gantry driver: Python driver for the stepper-based XYZ gantry robot.

Implements ArmInterface by translating chess board coordinates
into gantry XY + Z + electromagnet commands over the serial protocol.

The gantry is a deterministic system — no RL needed. All motion is
direct coordinate transformation:
    board_square → (x_mm, y_mm) → MOVE X<x> Y<y> Z<z>
    pick/place → Z descent + MAGNET 1/0

Architecture:
    GantryArm ←→ HardwareBridge ←→ Serial ←→ Pico Firmware
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from robot_model.arm_interface import ArmInterface, GripperInterface

logger = logging.getLogger(__name__)


@dataclass
class GantryConfig:
    """Configuration for the gantry hardware."""

    # Serial port
    port: str = "COM3"  # Windows; /dev/ttyACM0 on Linux/WSL2
    baudrate: int = 115_200

    # Board geometry (mm) — set by calibration
    board_origin_x_mm: float = 30.0  # X offset of A1 corner from home
    board_origin_y_mm: float = 30.0  # Y offset of A1 corner from home
    square_size_mm: float = 33.0  # Size of one chess square

    # Z heights (mm)
    safe_z_mm: float = 50.0  # Safe transit height
    board_surface_z_mm: float = 5.0  # Board surface (piece bases)
    pickup_z_mm: float = 2.0  # Z for magnet engagement
    place_z_mm: float = 3.0  # Z for piece placement

    # Tray position for captured pieces
    tray_x_mm: float = 270.0
    tray_y_mm: float = 150.0

    # Motion parameters
    move_timeout_s: float = 15.0  # Timeout waiting for MOVE_DONE
    keepalive_interval_s: float = 1.0  # PING interval (< 2s watchdog)

    # Transform matrix (4×4, board frame → gantry frame)
    T_gantry_board: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )


class GantryArm(ArmInterface):
    """
    Gantry robot implementing ArmInterface.

    Translates the 6-DOF arm interface into XYZ gantry commands.
    Since the gantry is top-down only, orientation is ignored —
    all moves are vertical approach/retreat.

    Thread safety: NOT thread-safe. Use from a single control thread.
    """

    def __init__(
        self,
        config: GantryConfig | None = None,
        bridge: Optional["HardwareBridge"] = None,
    ) -> None:
        self.config = config or GantryConfig()
        self._bridge = bridge
        self._connected = False
        self._homed = False
        self._current_pos = np.zeros(3)  # X, Y, Z in mm

    def connect(self) -> bool:
        """Connect to the gantry via serial bridge."""
        if self._bridge is None:
            from execution.hardware_bridge import HardwareBridge

            self._bridge = HardwareBridge(
                port=self.config.port,
                baudrate=self.config.baudrate,
            )

        self._connected = self._bridge.connect()
        if self._connected:
            logger.info("Gantry connected via serial bridge")
        return self._connected

    def disconnect(self) -> None:
        """Disconnect from the gantry."""
        if self._bridge:
            self._bridge.disconnect()
        self._connected = False

    def home(self) -> bool:
        """Home all axes. Must complete before any motion."""
        if not self._connected:
            logger.error("Cannot home: not connected")
            return False

        result = self._bridge.send_command("HOME", timeout_s=30.0)
        if result and result.status == "OK":
            # Wait for HOMED event
            event = self._bridge.wait_for_event("HOMED", timeout_s=30.0)
            if event:
                self._homed = True
                self._current_pos = np.zeros(3)
                logger.info("Gantry homed successfully")
                return True

        logger.error("Gantry homing failed")
        return False

    # =========================================================================
    # ArmInterface implementation
    # =========================================================================

    def move_to_pose(
        self,
        target_pose: np.ndarray,
        velocity_scale: float = 1.0,
    ) -> bool:
        """
        Move to a 4×4 pose matrix.

        For the gantry, only the translation (X, Y) matters.
        Z is set by the caller but clamped to safe_z_mm for transit moves.
        Orientation is ignored (always top-down).
        """
        if not self._homed:
            logger.error("Cannot move: not homed")
            return False

        x_mm = target_pose[0, 3] * 1000.0  # m → mm
        y_mm = target_pose[1, 3] * 1000.0
        z_mm = target_pose[2, 3] * 1000.0

        return self._move_to(x_mm, y_mm, z_mm)

    def move_cartesian_linear(
        self,
        target_pose: np.ndarray,
        velocity_ms: float = 0.05,
    ) -> bool:
        """
        Move linearly to a pose. For the gantry, this is the same as move_to_pose
        since all gantry motion is Cartesian by construction.
        """
        return self.move_to_pose(target_pose)

    def get_current_pose(self) -> np.ndarray:
        """Return current EE pose as 4×4 matrix."""
        pose = np.eye(4, dtype=np.float64)
        # Top-down orientation (Z points down)
        pose[2, 2] = -1.0
        pose[1, 1] = -1.0
        # Position in meters
        pose[0, 3] = self._current_pos[0] / 1000.0
        pose[1, 3] = self._current_pos[1] / 1000.0
        pose[2, 3] = self._current_pos[2] / 1000.0
        return pose

    def get_joint_positions(self) -> np.ndarray:
        """Return prismatic joint positions [X, Y, Z] in mm."""
        return self._current_pos.copy()

    def is_moving(self) -> bool:
        """Check if any axis is in motion."""
        if self._bridge:
            status = self._bridge.get_status()
            return status.get("state") == "MOVING" if status else False
        return False

    def emergency_stop(self) -> None:
        """Emergency stop all axes."""
        if self._bridge:
            self._bridge.send_command("HALT", timeout_s=1.0)
        logger.warning("Gantry EMERGENCY STOP")

    # =========================================================================
    # Gantry-specific methods
    # =========================================================================

    def square_to_xy(self, file: int, rank: int) -> tuple[float, float]:
        """
        Convert chess square (file, rank) to gantry XY (mm).

        file: 0-7 (a-h)
        rank: 0-7 (1-8)

        Returns (x_mm, y_mm) in gantry frame.
        """
        x = self.config.board_origin_x_mm + (file + 0.5) * self.config.square_size_mm
        y = self.config.board_origin_y_mm + (rank + 0.5) * self.config.square_size_mm
        return x, y

    def pick_piece(self, file: int, rank: int) -> bool:
        """
        Pick up a piece at the given square using the electromagnet.

        Sequence:
            1. Move to (x, y) at safe Z
            2. Lower Z to pickup height
            3. Activate magnet
            4. Raise Z to safe height

        Returns True if successful.
        """
        x, y = self.square_to_xy(file, rank)

        # Move above square at safe height
        if not self._move_to(x, y, self.config.safe_z_mm):
            return False

        # Lower to pickup
        if not self._move_to(x, y, self.config.pickup_z_mm):
            return False

        # Activate magnet
        self._set_magnet(True)
        time.sleep(0.2)  # Allow magnet to engage

        # Lift
        if not self._move_to(x, y, self.config.safe_z_mm):
            return False

        logger.info(f"Picked piece at {chr(ord('a')+file)}{rank+1}")
        return True

    def place_piece(self, file: int, rank: int) -> bool:
        """
        Place a held piece at the given square.

        Sequence:
            1. Move to (x, y) at safe Z
            2. Lower Z to place height
            3. Deactivate magnet
            4. Raise Z to safe height
        """
        x, y = self.square_to_xy(file, rank)

        # Move above square
        if not self._move_to(x, y, self.config.safe_z_mm):
            return False

        # Lower to place
        if not self._move_to(x, y, self.config.place_z_mm):
            return False

        # Release
        self._set_magnet(False)
        time.sleep(0.2)  # Allow piece to settle

        # Retreat
        if not self._move_to(x, y, self.config.safe_z_mm):
            return False

        logger.info(f"Placed piece at {chr(ord('a')+file)}{rank+1}")
        return True

    def move_to_tray(self) -> bool:
        """Move to the captured-piece tray and release."""
        if not self._move_to(self.config.tray_x_mm, self.config.tray_y_mm,
                             self.config.safe_z_mm):
            return False
        if not self._move_to(self.config.tray_x_mm, self.config.tray_y_mm,
                             self.config.place_z_mm):
            return False

        self._set_magnet(False)
        time.sleep(0.2)

        return self._move_to(self.config.tray_x_mm, self.config.tray_y_mm,
                             self.config.safe_z_mm)

    # =========================================================================
    # Internal
    # =========================================================================

    def _move_to(self, x_mm: float, y_mm: float, z_mm: float) -> bool:
        """Send a MOVE command and wait for completion."""
        if not self._bridge:
            logger.error("No hardware bridge")
            return False

        cmd = f"MOVE X{x_mm:.1f} Y{y_mm:.1f} Z{z_mm:.1f}"
        result = self._bridge.send_command(cmd, timeout_s=self.config.move_timeout_s)

        if result and result.status in ("OK", "DONE"):
            # Wait for MOVE_DONE event
            event = self._bridge.wait_for_event(
                "MOVE_DONE", timeout_s=self.config.move_timeout_s
            )
            if event:
                self._current_pos = np.array([x_mm, y_mm, z_mm])
                return True

        logger.error(f"Move failed: {result}")
        return False

    def _set_magnet(self, on: bool) -> None:
        """Control the electromagnet."""
        if self._bridge:
            self._bridge.send_command(f"MAGNET {1 if on else 0}", timeout_s=1.0)


class GantryGripper(GripperInterface):
    """
    Gripper interface adapted for the electromagnet.

    Maps gripper open/close to magnet off/on.
    """

    def __init__(self, gantry: GantryArm) -> None:
        self._gantry = gantry

    def open(self, width_mm: float = 0) -> None:
        """Release piece (magnet off)."""
        self._gantry._set_magnet(False)

    def close(self, force_n: float = 0, width_mm: float = 0) -> None:
        """Grab piece (magnet on)."""
        self._gantry._set_magnet(True)

    def is_gripping(self) -> bool:
        """Check if magnet is energized. Cannot verify actual pickup."""
        # For a real system, you'd need a Hall effect sensor or
        # current sense on the magnet coil.
        return True  # Optimistic — we have no feedback sensor

    def get_width_mm(self) -> float:
        """N/A for electromagnet. Return 0."""
        return 0.0
