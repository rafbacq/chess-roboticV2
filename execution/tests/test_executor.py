"""
Tests for the execution module: Executor, watchdog, telemetry.

Uses SimulatedArm/SimulatedGripper so no hardware needed.
"""

import numpy as np
import pytest

from execution.executor import Executor, ExecutionConfig
from robot_model.arm_interface import SimulatedArm, SimulatedGripper, ArmStatus


@pytest.fixture
def arm():
    a = SimulatedArm(name="test_arm")
    a.initialize()
    return a


@pytest.fixture
def gripper():
    g = SimulatedGripper(name="test_grip")
    g.initialize()
    return g


@pytest.fixture
def executor(arm, gripper):
    config = ExecutionConfig(
        stage_timeout_s=5.0,
        total_timeout_s=30.0,
        telemetry_log_dir="",  # disable disk saving
    )
    return Executor(arm, gripper, config)


class TestExecutorJointTrajectory:
    def test_single_waypoint(self, executor):
        wp = [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
        assert executor.execute_joint_trajectory(wp)

    def test_multi_waypoint(self, executor):
        wps = [np.zeros(6) + i * 0.1 for i in range(5)]
        assert executor.execute_joint_trajectory(wps, stage_name="test_multi")

    def test_empty_trajectory(self, executor):
        assert executor.execute_joint_trajectory([])


class TestExecutorCartesian:
    def test_cartesian_move(self, executor):
        target = np.eye(4)
        target[0, 3] = 0.3
        target[2, 3] = 0.2
        assert executor.execute_cartesian_move(target)

    def test_linear_cartesian_move(self, executor):
        target = np.eye(4)
        target[0, 3] = 0.25
        assert executor.execute_cartesian_move(target, linear=True)


class TestExecutorGripper:
    def test_open(self, executor):
        assert executor.execute_gripper("open", width_mm=50.0)

    def test_close(self, executor):
        assert executor.execute_gripper("close")

    def test_unknown_action_raises(self, executor):
        with pytest.raises(ValueError, match="Unknown gripper action"):
            executor.execute_gripper("squeeze")


class TestExecutorSafety:
    def test_pause(self, executor):
        executor.pause()
        # Should not raise

    def test_abort(self, executor):
        executor.abort()
        # Should trigger emergency stop on arm

    def test_recover(self, executor):
        assert executor.recover()


class TestExecutorTelemetry:
    def test_telemetry_recording(self, executor):
        executor.start_telemetry(move_uci="e2e4")
        wp = [np.zeros(6), np.ones(6) * 0.1]
        executor.execute_joint_trajectory(wp, stage_name="test_telem")
        records = executor.stop_telemetry()
        assert len(records) == 2
        assert records[0].move_uci == "e2e4"
        assert records[0].stage == "test_telem"

    def test_no_recording_when_stopped(self, executor):
        wp = [np.zeros(6)]
        executor.execute_joint_trajectory(wp)
        # No telemetry should be recorded
        assert len(executor._telemetry) == 0

    def test_telemetry_clears_on_restart(self, executor):
        executor.start_telemetry("move1")
        executor.execute_joint_trajectory([np.zeros(6)])
        assert len(executor._telemetry) == 1
        executor.start_telemetry("move2")
        assert len(executor._telemetry) == 0
