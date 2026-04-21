"""
Gantry simulation adapter: virtual serial port for CI/testing.

Simulates the Pico firmware's serial protocol without hardware.
Used by:
    - CI pipeline (test_protocol_smoke.py)
    - Offline development / debugging
    - Integration tests without physical gantry

Emulates the full state machine: BOOT → HOME → IDLE → MOVE → etc.
"""

from __future__ import annotations

import logging
import threading
import time
from io import BytesIO
from queue import Queue
from typing import Optional

logger = logging.getLogger(__name__)


class GantrySimAdapter:
    """
    In-memory serial port simulator that implements the gantry protocol.

    Drop-in replacement for pyserial's Serial object.

    Usage:
        adapter = GantrySimAdapter()
        # Use adapter in place of serial.Serial(...)
        adapter.write(b"1 PING\\n")
        response = adapter.readline()  # → b"ACK 1 OK PONG\\n"
    """

    def __init__(
        self,
        move_delay_s: float = 0.05,  # Simulated move duration
        home_delay_s: float = 0.2,   # Simulated homing duration
    ) -> None:
        self._move_delay = move_delay_s
        self._home_delay = home_delay_s

        # State
        self._homed = False
        self._halted = False
        self._moving = False
        self._magnet = False
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = 0.0

        # Output buffer
        self._output_queue: Queue[str] = Queue()
        self._is_open = True

        # Send initial BOOT event
        self._send_event("BOOT", "v2.0 sim gantry ready")

        # Heartbeat thread
        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="sim-heartbeat"
        )
        self._heartbeat_thread.start()

    @property
    def is_open(self) -> bool:
        return self._is_open

    def close(self) -> None:
        self._heartbeat_running = False
        self._is_open = False

    def write(self, data: bytes) -> int:
        """Process incoming commands."""
        line = data.decode("ascii", errors="replace").strip()
        if line:
            self._process_command(line)
        return len(data)

    def flush(self) -> None:
        pass

    def readline(self) -> bytes:
        """Read a response line. Blocks up to timeout."""
        try:
            line = self._output_queue.get(timeout=0.5)
            return (line + "\n").encode("ascii")
        except Exception:
            return b""

    def _process_command(self, line: str) -> None:
        """Parse and execute a command."""
        parts = line.split(None, 2)
        if len(parts) < 2:
            self._send_ack(0, "ERR", "PARSE_FAIL")
            return

        try:
            seq = int(parts[0])
        except ValueError:
            self._send_ack(0, "ERR", "PARSE_FAIL")
            return

        cmd = parts[1].upper()
        args = parts[2] if len(parts) > 2 else ""

        if cmd == "PING":
            self._send_ack(seq, "OK", "PONG")
            return

        if cmd == "STATUS":
            state = 4 if self._homed and not self._halted else (6 if self._halted else 1)
            detail = (
                f"STATE={state} HOMED={1 if self._homed else 0} "
                f"X={self._pos_x:.2f} Y={self._pos_y:.2f} Z={self._pos_z:.2f} "
                f"MAG={1 if self._magnet else 0}"
            )
            self._send_ack(seq, "OK", detail)
            return

        if cmd == "HALT":
            self._halted = True
            self._moving = False
            self._magnet = False
            self._send_ack(seq, "OK", "HALTED")
            self._send_event("HALT", "CMD_HALT")
            return

        if cmd == "RESET":
            self._halted = False
            self._homed = False
            self._magnet = False
            self._send_ack(seq, "OK", "RESET_NEED_HOME")
            return

        if cmd == "HOME":
            if self._moving:
                self._send_ack(seq, "ERR", "BUSY_MOVING")
                return
            self._send_ack(seq, "OK", "HOMING_STARTED")
            # Simulate homing in background
            threading.Thread(
                target=self._simulate_homing,
                daemon=True,
            ).start()
            return

        if not self._homed:
            self._send_ack(seq, "ERR", "NOT_HOMED")
            return

        if self._halted:
            self._send_ack(seq, "ERR", "IN_HALT_OR_ERROR")
            return

        if cmd == "MOVE":
            if self._moving:
                self._send_ack(seq, "ERR", "BUSY_MOVING")
                return
            # Parse X, Y, Z
            target = self._parse_xyz(args)
            if target is None:
                self._send_ack(seq, "ERR", "MOVE_PARSE_FAIL")
                return

            x, y, z = target
            self._send_ack(seq, "OK", f"TARGET X{x:.1f} Y{y:.1f} Z{z:.1f}")
            # Simulate move in background
            threading.Thread(
                target=self._simulate_move,
                args=(seq, x, y, z),
                daemon=True,
            ).start()
            return

        if cmd == "MAGNET":
            val = int(args.strip()) if args.strip() else 0
            self._magnet = val != 0
            self._send_ack(seq, "OK", "MAGNET_ON" if self._magnet else "MAGNET_OFF")
            return

        self._send_ack(seq, "ERR", "UNKNOWN_CMD")

    def _simulate_homing(self) -> None:
        """Simulate homing sequence with delays."""
        for axis, event in [("Z", "Z_START"), ("Z", "Z_HIT"),
                            ("X", "X_START"), ("X", "X_HIT"),
                            ("Y", "Y_START"), ("Y", "Y_HIT")]:
            time.sleep(self._home_delay / 6)
            self._send_event("HOMING", event)

        time.sleep(self._home_delay / 6)
        self._send_event("HOMING", "BACKOFF_START")
        time.sleep(self._home_delay / 6)

        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = 0.0
        self._homed = True
        self._send_event("HOMED", "ALL_AXES")

    def _simulate_move(self, seq: int, x: float, y: float, z: float) -> None:
        """Simulate move with delay."""
        self._moving = True
        time.sleep(self._move_delay)
        self._pos_x = x
        self._pos_y = y
        self._pos_z = z
        self._moving = False
        detail = f"AT X{x:.2f} Y{y:.2f} Z{z:.2f}"
        self._send_ack(seq, "DONE", detail)
        self._send_event("MOVE_DONE", detail)

    def _parse_xyz(self, args: str) -> Optional[tuple[float, float, float]]:
        """Parse X<val> Y<val> Z<val> from command arguments."""
        x = y = z = None
        for token in args.split():
            token_upper = token.upper()
            if token_upper.startswith("X"):
                try:
                    x = float(token[1:])
                except ValueError:
                    return None
            elif token_upper.startswith("Y"):
                try:
                    y = float(token[1:])
                except ValueError:
                    return None
            elif token_upper.startswith("Z"):
                try:
                    z = float(token[1:])
                except ValueError:
                    return None
        if x is None or y is None or z is None:
            return None
        # Clamp to limits
        x = max(0.0, min(300.0, x))
        y = max(0.0, min(300.0, y))
        z = max(0.0, min(60.0, z))
        return x, y, z

    def _send_ack(self, seq: int, status: str, detail: str = "") -> None:
        line = f"ACK {seq} {status}"
        if detail:
            line += f" {detail}"
        self._output_queue.put(line)

    def _send_event(self, name: str, detail: str = "") -> None:
        line = f"EVT {name}"
        if detail:
            line += f" {detail}"
        self._output_queue.put(line)

    def _heartbeat_loop(self) -> None:
        """Send position heartbeats (like the real firmware)."""
        while self._heartbeat_running:
            time.sleep(0.5)
            if self._homed and not self._halted:
                self._send_event(
                    "POS",
                    f"X{self._pos_x:.2f} Y{self._pos_y:.2f} Z{self._pos_z:.2f} "
                    f"M{1 if self._magnet else 0} S{4 if not self._moving else 5}",
                )
