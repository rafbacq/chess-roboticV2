"""
Protocol smoke test: verifies the gantry serial protocol
using the simulated adapter (no hardware required).

Tests run in CI to verify:
    - Command/ACK round-trip
    - State machine transitions
    - Homing sequence
    - Move execution
    - Magnet control
    - HALT behavior
    - Safety supervisor integration
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from simulation.gantry_sim_adapter import GantrySimAdapter


class TestGantryProtocol:
    """Test the serial protocol via the simulation adapter."""

    @pytest.fixture
    def adapter(self):
        a = GantrySimAdapter(move_delay_s=0.02, home_delay_s=0.1)
        yield a
        a.close()

    def _send_recv(self, adapter: GantrySimAdapter, cmd: str, seq: int = 1) -> str:
        """Send a command and return the first ACK line."""
        adapter.write(f"{seq} {cmd}\n".encode())
        return adapter.readline().decode().strip()

    def test_ping(self, adapter):
        resp = self._send_recv(adapter, "PING", seq=1)
        assert "ACK 1 OK PONG" in resp

    def test_status_before_home(self, adapter):
        resp = self._send_recv(adapter, "STATUS", seq=2)
        assert "HOMED=0" in resp

    def test_move_before_home_rejected(self, adapter):
        resp = self._send_recv(adapter, "MOVE X100 Y100 Z10", seq=3)
        assert "ERR" in resp
        assert "NOT_HOMED" in resp

    def test_home_sequence(self, adapter):
        resp = self._send_recv(adapter, "HOME", seq=4)
        assert "ACK 4 OK HOMING_STARTED" in resp

        # Wait for homing events
        events = []
        deadline = time.time() + 2.0
        while time.time() < deadline:
            line = adapter.readline().decode().strip()
            if line:
                events.append(line)
                if "HOMED" in line and "ALL_AXES" in line:
                    break

        event_text = " ".join(events)
        assert "Z_START" in event_text
        assert "Z_HIT" in event_text
        assert "X_START" in event_text
        assert "Y_HIT" in event_text
        assert "ALL_AXES" in event_text

    def test_full_move_sequence(self, adapter):
        # Home first
        adapter.write(b"1 HOME\n")
        deadline = time.time() + 2.0
        while time.time() < deadline:
            line = adapter.readline().decode().strip()
            if "ALL_AXES" in line:
                break

        # Move
        adapter.write(b"2 MOVE X100.0 Y200.0 Z10.0\n")
        responses = []
        deadline = time.time() + 2.0
        while time.time() < deadline:
            line = adapter.readline().decode().strip()
            if line:
                responses.append(line)
                if "MOVE_DONE" in line:
                    break

        resp_text = " ".join(responses)
        assert "TARGET" in resp_text
        assert "MOVE_DONE" in resp_text
        assert "100.00" in resp_text
        assert "200.00" in resp_text

    def test_magnet_control(self, adapter):
        # Home first
        adapter.write(b"1 HOME\n")
        deadline = time.time() + 2.0
        while time.time() < deadline:
            line = adapter.readline().decode().strip()
            if "ALL_AXES" in line:
                break

        on_resp = self._send_recv(adapter, "MAGNET 1", seq=2)
        assert "MAGNET_ON" in on_resp

        off_resp = self._send_recv(adapter, "MAGNET 0", seq=3)
        assert "MAGNET_OFF" in off_resp

    def test_halt(self, adapter):
        resp = self._send_recv(adapter, "HALT", seq=10)
        assert "HALTED" in resp

    def test_reset_after_halt(self, adapter):
        adapter.write(b"1 HALT\n")
        adapter.readline()  # ACK
        adapter.readline()  # EVT HALT

        resp = self._send_recv(adapter, "RESET", seq=2)
        assert "RESET_NEED_HOME" in resp

    def test_move_while_moving_rejected(self, adapter):
        # Home
        adapter.write(b"1 HOME\n")
        deadline = time.time() + 2.0
        while time.time() < deadline:
            line = adapter.readline().decode().strip()
            if "ALL_AXES" in line:
                break

        # Use a longer delay to ensure we can send while moving
        adapter._move_delay = 0.5
        adapter.write(b"2 MOVE X100 Y100 Z10\n")
        # Read the OK ack
        adapter.readline()
        time.sleep(0.05)

        # Try second move while first is in progress
        resp = self._send_recv(adapter, "MOVE X200 Y200 Z10", seq=3)
        assert "BUSY_MOVING" in resp


class TestSafetySupervisor:
    """Test the safety supervisor."""

    def test_validate_move_in_bounds(self):
        from execution.safety_supervisor import SafetySupervisor, SafetyConfig

        supervisor = SafetySupervisor(config=SafetyConfig())
        assert supervisor.validate_move(100.0, 100.0, 30.0)
        assert supervisor.validate_move(0.0, 0.0, 0.0)
        assert supervisor.validate_move(300.0, 300.0, 60.0)

    def test_validate_move_out_of_bounds(self):
        from execution.safety_supervisor import SafetySupervisor, SafetyConfig

        supervisor = SafetySupervisor(config=SafetyConfig())
        assert not supervisor.validate_move(-1.0, 100.0, 10.0)
        assert not supervisor.validate_move(100.0, 301.0, 10.0)
        assert not supervisor.validate_move(100.0, 100.0, 61.0)

    def test_halt_state(self):
        from execution.safety_supervisor import SafetySupervisor, SafetyConfig

        supervisor = SafetySupervisor(config=SafetyConfig())
        assert supervisor.is_safe()

        supervisor.trigger_halt("test halt")
        assert not supervisor.is_safe()
        assert supervisor.is_halted
        assert not supervisor.validate_move(100.0, 100.0, 10.0)

        supervisor.reset()
        assert supervisor.is_safe()
        assert supervisor.validate_move(100.0, 100.0, 10.0)

    def test_events_logged(self):
        from execution.safety_supervisor import SafetySupervisor, SafetyConfig

        supervisor = SafetySupervisor(config=SafetyConfig())
        supervisor.trigger_halt("test")
        assert len(supervisor.events) == 1
        assert supervisor.events[0].event_type == "HALT"

    def test_halt_response_time(self):
        """Verify HALT completes within 100ms (no bridge = instant)."""
        from execution.safety_supervisor import SafetySupervisor, SafetyConfig

        supervisor = SafetySupervisor(config=SafetyConfig())
        t0 = time.time()
        supervisor.trigger_halt("response time test")
        elapsed_ms = (time.time() - t0) * 1000
        assert elapsed_ms < 100.0, f"HALT took {elapsed_ms:.1f}ms, exceeds 100ms"
