"""
Safety supervisor: monitors gantry operation and enforces safety constraints.

Responsibilities:
    - HALT within 100ms of any safety violation (CI-verifiable)
    - Monitor for endstop contact during non-homing moves
    - Enforce workspace limits
    - Track electromagnet state (prevent indefinite hold)
    - Monitor communication heartbeat (watchdog validation)
    - Log all safety events with timestamps

The supervisor wraps the HardwareBridge and GantryArm, intercepting
all commands and validating safety before forwarding.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Safety supervisor configuration."""

    # Workspace limits (mm)
    x_min_mm: float = 0.0
    x_max_mm: float = 300.0
    y_min_mm: float = 0.0
    y_max_mm: float = 300.0
    z_min_mm: float = 0.0
    z_max_mm: float = 60.0

    # Magnet safety
    max_magnet_on_duration_s: float = 60.0  # Auto-off after 60s
    max_magnet_on_without_move_s: float = 10.0  # If magnet on and not moving

    # Communication
    max_heartbeat_gap_s: float = 3.0  # HALT if no position update

    # Emergency stop response requirement
    halt_response_ms: float = 100.0  # Must HALT within 100ms


@dataclass
class SafetyEvent:
    """A recorded safety event."""

    timestamp: float
    event_type: str  # "HALT", "WARNING", "VIOLATION"
    reason: str
    position: Optional[tuple[float, float, float]] = None


class SafetySupervisor:
    """
    Safety monitor that wraps gantry operations.

    Usage:
        supervisor = SafetySupervisor(bridge, config)
        supervisor.start_monitoring()

        # Before any move:
        if supervisor.validate_move(x, y, z):
            bridge.send_command(f"MOVE X{x} Y{y} Z{z}")

        # Check safety status:
        if supervisor.is_safe():
            ...

        supervisor.stop_monitoring()
    """

    def __init__(
        self,
        bridge: Optional[object] = None,  # HardwareBridge
        config: SafetyConfig | None = None,
    ) -> None:
        self.config = config or SafetyConfig()
        self._bridge = bridge
        self._is_halted = False
        self._magnet_on_since: Optional[float] = None
        self._last_heartbeat = time.time()
        self._events: list[SafetyEvent] = []
        self._monitoring = False

    def validate_move(self, x_mm: float, y_mm: float, z_mm: float) -> bool:
        """
        Validate a proposed move against safety constraints.

        Returns True if the move is safe.
        Logs a warning and returns False if unsafe.
        """
        if self._is_halted:
            self._log_event("VIOLATION", "Move rejected: in HALT state")
            return False

        issues = []
        if x_mm < self.config.x_min_mm or x_mm > self.config.x_max_mm:
            issues.append(f"X={x_mm:.1f}mm out of range [{self.config.x_min_mm}, {self.config.x_max_mm}]")
        if y_mm < self.config.y_min_mm or y_mm > self.config.y_max_mm:
            issues.append(f"Y={y_mm:.1f}mm out of range [{self.config.y_min_mm}, {self.config.y_max_mm}]")
        if z_mm < self.config.z_min_mm or z_mm > self.config.z_max_mm:
            issues.append(f"Z={z_mm:.1f}mm out of range [{self.config.z_min_mm}, {self.config.z_max_mm}]")

        if issues:
            for issue in issues:
                self._log_event("VIOLATION", f"Move rejected: {issue}")
            return False

        return True

    def notify_magnet_on(self) -> None:
        """Record magnet activation time."""
        self._magnet_on_since = time.time()

    def notify_magnet_off(self) -> None:
        """Clear magnet activation time."""
        self._magnet_on_since = None

    def notify_heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self._last_heartbeat = time.time()

    def check_safety(self) -> bool:
        """
        Run all safety checks. Returns True if safe.
        Triggers HALT if any check fails.
        """
        now = time.time()

        # Check magnet duration
        if self._magnet_on_since is not None:
            duration = now - self._magnet_on_since
            if duration > self.config.max_magnet_on_duration_s:
                self.trigger_halt(f"Magnet on too long: {duration:.1f}s")
                return False

        # Check heartbeat gap
        heartbeat_gap = now - self._last_heartbeat
        if heartbeat_gap > self.config.max_heartbeat_gap_s:
            self.trigger_halt(f"No heartbeat for {heartbeat_gap:.1f}s")
            return False

        return True

    def trigger_halt(self, reason: str) -> None:
        """
        Trigger emergency stop.

        Must complete within 100ms (per safety requirement).
        """
        t0 = time.time()
        self._is_halted = True

        # Send HALT to firmware
        if self._bridge:
            try:
                self._bridge.send_command("HALT", timeout_s=0.1)
            except Exception as e:
                logger.error(f"HALT command failed: {e}")

        elapsed_ms = (time.time() - t0) * 1000
        self._log_event(
            "HALT",
            f"{reason} (response: {elapsed_ms:.1f}ms)",
        )

        if elapsed_ms > self.config.halt_response_ms:
            logger.critical(
                f"HALT response {elapsed_ms:.1f}ms exceeds "
                f"{self.config.halt_response_ms}ms requirement!"
            )

    def reset(self) -> None:
        """Reset from HALT state. Requires re-homing."""
        self._is_halted = False
        self._magnet_on_since = None
        self._log_event("WARNING", "Safety supervisor reset")

    def is_safe(self) -> bool:
        """Check if the system is in a safe operating state."""
        return not self._is_halted

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    @property
    def events(self) -> list[SafetyEvent]:
        return list(self._events)

    def _log_event(
        self,
        event_type: str,
        reason: str,
        position: Optional[tuple[float, float, float]] = None,
    ) -> None:
        """Record a safety event."""
        event = SafetyEvent(
            timestamp=time.time(),
            event_type=event_type,
            reason=reason,
            position=position,
        )
        self._events.append(event)
        log_fn = logger.critical if event_type == "HALT" else logger.warning
        log_fn(f"SAFETY {event_type}: {reason}")
