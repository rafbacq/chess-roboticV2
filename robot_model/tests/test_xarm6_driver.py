"""
Tests for the xArm6 hardware driver.

Tests run in mock mode (no real robot needed). Verifies interface
compliance, mock initialization, and defensive fallback behavior.
"""

import numpy as np
import pytest

from robot_model.xarm6_driver import XArm6Arm, XArmGripper


class TestXArm6ArmInterface:
    """Verify XArm6Arm implements ArmInterface correctly."""

    def test_is_subclass(self):
        from robot_model.arm_interface import ArmInterface
        assert issubclass(XArm6Arm, ArmInterface)

    def test_mock_initialization(self):
        arm = XArm6Arm(name="test_arm", ip="192.168.1.197")
        assert arm.initialize()
        assert arm.is_ready()

    def test_mock_joint_positions(self):
        arm = XArm6Arm()
        arm.initialize()
        joints = arm.get_joint_positions()
        assert isinstance(joints, np.ndarray)
        assert len(joints) == 6
        np.testing.assert_allclose(joints, np.zeros(6))

    def test_mock_ee_pose(self):
        arm = XArm6Arm()
        arm.initialize()
        pose = arm.get_ee_pose()
        assert pose.shape == (4, 4)
        np.testing.assert_allclose(pose, np.eye(4))

    def test_mock_move_to_joint_positions(self):
        arm = XArm6Arm()
        arm.initialize()
        target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert arm.move_to_joint_positions(target)
        np.testing.assert_allclose(arm.get_joint_positions(), target)

    def test_mock_move_to_pose(self):
        arm = XArm6Arm()
        arm.initialize()
        target = np.eye(4)
        target[0, 3] = 0.3
        target[1, 3] = 0.1
        target[2, 3] = 0.5
        assert arm.move_to_pose(target)
        np.testing.assert_allclose(arm.get_ee_pose(), target)

    def test_mock_cartesian_linear(self):
        arm = XArm6Arm()
        arm.initialize()
        target = np.eye(4)
        target[0, 3] = 0.2
        assert arm.move_cartesian_linear(target, velocity_ms=0.1)
        np.testing.assert_allclose(arm.get_ee_pose(), target)

    def test_capabilities(self):
        arm = XArm6Arm()
        arm.initialize()
        caps = arm.get_capabilities()
        assert caps.name == "xarm6"
        assert caps.dof == 6
        assert caps.max_reach_m == pytest.approx(0.691)

    def test_shutdown(self):
        arm = XArm6Arm()
        arm.initialize()
        assert arm.is_ready()
        arm.shutdown()
        assert not arm.is_ready()

    def test_get_status_ready(self):
        from robot_model.arm_interface import ArmStatus
        arm = XArm6Arm()
        arm.initialize()
        assert arm.get_status() == ArmStatus.READY

    def test_get_status_not_initialized(self):
        from robot_model.arm_interface import ArmStatus
        arm = XArm6Arm()
        assert arm.get_status() == ArmStatus.NOT_INITIALIZED

    def test_stop(self):
        arm = XArm6Arm()
        arm.initialize()
        arm.stop()
        assert arm.is_ready()

    def test_emergency_stop(self):
        from robot_model.arm_interface import ArmStatus
        arm = XArm6Arm()
        arm.initialize()
        arm.emergency_stop()
        assert arm.get_status() == ArmStatus.EMERGENCY_STOP

    def test_recover_from_error(self):
        arm = XArm6Arm()
        arm.initialize()
        arm.emergency_stop()
        assert not arm.is_ready()
        assert arm.recover_from_error()
        assert arm.is_ready()

    def test_joint_velocities(self):
        arm = XArm6Arm()
        arm.initialize()
        vels = arm.get_joint_velocities()
        assert len(vels) == 6
        np.testing.assert_allclose(vels, np.zeros(6))


class TestXArmGripperInterface:
    """Verify XArmGripper implements GripperInterface correctly."""

    def test_is_subclass(self):
        from robot_model.arm_interface import GripperInterface
        assert issubclass(XArmGripper, GripperInterface)

    def test_mock_initialization(self):
        gripper = XArmGripper()
        assert gripper.initialize()
        assert gripper.is_ready()

    def test_mock_open_close_cycle(self):
        gripper = XArmGripper()
        gripper.initialize()
        assert gripper.open(width_mm=50.0)
        assert gripper.get_width_mm() == pytest.approx(50.0)
        assert not gripper.is_gripping()

        assert gripper.close()
        assert gripper.get_width_mm() == pytest.approx(0.0)
        assert gripper.is_gripping()

    def test_mock_full_open(self):
        gripper = XArmGripper()
        gripper.initialize()
        assert gripper.open()
        assert gripper.get_width_mm() == pytest.approx(85.0)

    def test_capabilities(self):
        gripper = XArmGripper()
        gripper.initialize()
        caps = gripper.get_capabilities()
        assert caps.max_width_mm == pytest.approx(85.0)
        assert caps.max_force_n == pytest.approx(30.0)

    def test_shutdown(self):
        gripper = XArmGripper()
        gripper.initialize()
        assert gripper.is_ready()
        gripper.shutdown()
        assert not gripper.is_ready()

    def test_close_with_width(self):
        gripper = XArmGripper()
        gripper.initialize()
        gripper.open()
        assert gripper.close(width_mm=40.0)
        assert gripper.get_width_mm() == pytest.approx(40.0)
        # width >= 30, so not gripping
        assert not gripper.is_gripping()

    def test_status_transitions(self):
        from robot_model.arm_interface import GripperStatus
        gripper = XArmGripper()
        gripper.initialize()
        assert gripper.get_status() == GripperStatus.CLOSED
        gripper.open()
        assert gripper.get_status() == GripperStatus.OPEN
        gripper.close()
        assert gripper.get_status() == GripperStatus.GRIPPING
