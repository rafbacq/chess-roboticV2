"""
Hardware bridge: reliable serial communication with the gantry firmware.

Handles:
    - Serial port connection/reconnection
    - Command sequencing and ACK matching
    - Event stream parsing
    - Keepalive (PING) to prevent watchdog timeout
    - Thread-safe command queue (optional, for async use)

The bridge speaks the protocol defined in firmware/PROTOCOL.md.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result from a serial command."""

    seq: int
    status: str  # "OK", "ERR", "DONE"
    detail: str = ""
    round_trip_ms: float = 0.0


@dataclass
class Event:
    """An asynchronous event from the firmware."""

    name: str  # e.g., "HOMED", "MOVE_DONE", "HALT"
    detail: str = ""
    timestamp: float = 0.0


class HardwareBridge:
    """
    Manages serial communication with the gantry Pico firmware.

    Usage:
        bridge = HardwareBridge(port="COM3")
        bridge.connect()
        result = bridge.send_command("HOME", timeout_s=30.0)
        event = bridge.wait_for_event("HOMED", timeout_s=30.0)
        bridge.disconnect()

    For WSL2 users:
        Use usbipd-win to attach the Pico USB to the WSL2 instance:
            usbipd list                    # Find the Pico device
            usbipd bind --busid <BUS-ID>   # Bind it
            usbipd attach --wsl --busid <BUS-ID>  # Attach to WSL2
        Then use port="/dev/ttyACM0" instead of "COM3".
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115_200,
        keepalive_s: float = 1.0,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._keepalive_s = keepalive_s
        self._serial = None
        self._connected = False
        self._seq = 0

        # ACK matching
        self._pending_ack: dict[int, CommandResult] = {}
        self._ack_events: dict[int, threading.Event] = {}

        # Event stream
        self._event_queue: Queue[Event] = Queue(maxsize=100)

        # Background reader thread
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_running = False

        # Keepalive thread
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_running = False

        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Open serial port and start background reader."""
        try:
            import serial

            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=0.1,  # Non-blocking read with 100ms timeout
            )
            self._connected = True

            # Start background reader
            self._reader_running = True
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True, name="serial-reader"
            )
            self._reader_thread.start()

            # Start keepalive
            self._keepalive_running = True
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, daemon=True, name="serial-keepalive"
            )
            self._keepalive_thread.start()

            logger.info(f"Connected to {self._port} at {self._baudrate}")
            return True

        except ImportError:
            logger.error("pyserial not installed: pip install pyserial")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {self._port}: {e}")
            return False

    def disconnect(self) -> None:
        """Close serial port and stop background threads."""
        self._reader_running = False
        self._keepalive_running = False

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_thread.join(timeout=2.0)

        if self._serial and self._serial.is_open:
            self._serial.close()

        self._connected = False
        logger.info("Disconnected from serial port")

    def send_command(
        self, cmd: str, timeout_s: float = 5.0
    ) -> Optional[CommandResult]:
        """
        Send a command and wait for its ACK.

        Args:
            cmd: Command string (without sequence number).
            timeout_s: Max wait time for ACK.

        Returns:
            CommandResult or None on timeout.
        """
        if not self._connected or not self._serial:
            logger.error("Cannot send: not connected")
            return None

        with self._lock:
            self._seq += 1
            seq = self._seq

        # Set up ACK waiter
        ack_event = threading.Event()
        self._ack_events[seq] = ack_event

        # Send
        line = f"{seq} {cmd}\n"
        t0 = time.time()

        try:
            self._serial.write(line.encode("ascii"))
            self._serial.flush()
            logger.debug(f"TX: {line.strip()}")
        except Exception as e:
            logger.error(f"Serial write failed: {e}")
            self._ack_events.pop(seq, None)
            return None

        # Wait for ACK
        if ack_event.wait(timeout=timeout_s):
            result = self._pending_ack.pop(seq, None)
            self._ack_events.pop(seq, None)
            if result:
                result.round_trip_ms = (time.time() - t0) * 1000
                return result

        self._ack_events.pop(seq, None)
        logger.warning(f"Timeout waiting for ACK {seq} (cmd: {cmd})")
        return None

    def wait_for_event(
        self, event_name: str, timeout_s: float = 10.0
    ) -> Optional[Event]:
        """
        Wait for a specific event from the firmware.

        Drains the event queue until a matching event or timeout.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                event = self._event_queue.get(timeout=min(remaining, 0.5))
                if event.name == event_name:
                    return event
                # Non-matching event — log and discard
                logger.debug(f"Skipped event: {event.name} {event.detail}")
            except Exception:
                continue
        return None

    def get_status(self) -> Optional[dict]:
        """Query current firmware status."""
        result = self.send_command("STATUS", timeout_s=2.0)
        if result and result.status == "OK":
            # Parse "STATE=4 HOMED=1 X=100.00 Y=150.00 Z=10.00 MAG=0"
            status = {}
            for part in result.detail.split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        status[k.lower()] = float(v) if "." in v else int(v)
                    except ValueError:
                        status[k.lower()] = v
            return status
        return None

    # =========================================================================
    # Background threads
    # =========================================================================

    def _reader_loop(self) -> None:
        """Background thread: read and parse serial lines."""
        while self._reader_running:
            try:
                if not self._serial or not self._serial.is_open:
                    time.sleep(0.1)
                    continue

                line = self._serial.readline().decode("ascii", errors="replace").strip()
                if not line:
                    continue

                logger.debug(f"RX: {line}")
                self._parse_line(line)

            except Exception as e:
                if self._reader_running:
                    logger.error(f"Serial reader error: {e}")
                    time.sleep(0.5)

    def _parse_line(self, line: str) -> None:
        """Parse a line from the firmware."""
        if line.startswith("ACK "):
            # ACK <seq> <status> [detail]
            parts = line.split(None, 3)
            if len(parts) >= 3:
                try:
                    seq = int(parts[1])
                    status = parts[2]
                    detail = parts[3] if len(parts) > 3 else ""

                    result = CommandResult(seq=seq, status=status, detail=detail)
                    self._pending_ack[seq] = result

                    # Signal the waiter
                    if seq in self._ack_events:
                        self._ack_events[seq].set()
                except (ValueError, IndexError):
                    logger.warning(f"Malformed ACK: {line}")

        elif line.startswith("EVT "):
            # EVT <event> [detail]
            parts = line.split(None, 2)
            if len(parts) >= 2:
                event = Event(
                    name=parts[1],
                    detail=parts[2] if len(parts) > 2 else "",
                    timestamp=time.time(),
                )
                try:
                    self._event_queue.put_nowait(event)
                except Exception:
                    pass  # Queue full — drop oldest if needed

    def _keepalive_loop(self) -> None:
        """Background thread: send PING to prevent watchdog timeout."""
        while self._keepalive_running:
            time.sleep(self._keepalive_s)
            if self._connected and self._serial and self._serial.is_open:
                try:
                    with self._lock:
                        self._seq += 1
                        seq = self._seq
                    self._serial.write(f"{seq} PING\n".encode("ascii"))
                    self._serial.flush()
                except Exception:
                    pass  # Silent failure on keepalive
