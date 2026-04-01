"""
Tests for the PyBullet arm and gripper adapters.

These tests validate the interface contracts without requiring
a PyBullet physics session. They use import-level checks and
test the helper methods (rotation conversion, etc.) directly.

Full integration tests with PyBullet physics are only run if
pybullet is installed (marked with pytest.importorskip).
"""

import math

import numpy as np
import pytest


class TestRotationConversion:
    """Test the static rotation matrix -> quaternion conversion."""

    def _get_converter(self):
        from simulation.pybullet_arm import PyBulletArm
        return PyBulletArm._rotation_matrix_to_quaternion

    def test_identity_rotation(self):
        convert = self._get_converter()
        R = np.eye(3)
        quat = convert(R)
        # Identity rotation = [0, 0, 0, 1]
        assert len(quat) == 4
        np.testing.assert_allclose(
            quat, [0, 0, 0, 1], atol=1e-10,
            err_msg="Identity rotation should give unit quaternion"
        )

    def test_90_deg_z_rotation(self):
        convert = self._get_converter()
        # 90 degrees around Z axis
        angle = math.pi / 2
        R = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
            [0, 0, 1],
        ])
        quat = convert(R)
        # Expected: [0, 0, sin(45), cos(45)] = [0, 0, 0.707, 0.707]
        expected_w = math.cos(angle / 2)
        expected_z = math.sin(angle / 2)
        np.testing.assert_allclose(
            [abs(quat[0]), abs(quat[1]), abs(quat[2]), abs(quat[3])],
            [0, 0, expected_z, expected_w],
            atol=1e-6,
        )

    def test_180_deg_x_rotation(self):
        convert = self._get_converter()
        # 180 degrees around X axis (top-down gripper)
        R = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
        ], dtype=np.float64)
        quat = convert(R)
        # Should be a valid unit quaternion
        norm = math.sqrt(sum(q**2 for q in quat))
        assert abs(norm - 1.0) < 1e-6, f"Quaternion should be unit, got norm={norm}"

    def test_roundtrip_consistency(self):
        """Test that our R->quat conversion is consistent across angles."""
        convert = self._get_converter()
        for angle_deg in [0, 30, 45, 90, 135, 180]:
            angle = math.radians(angle_deg)
            R = np.array([
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle),  math.cos(angle), 0],
                [0, 0, 1],
            ])
            quat = convert(R)
            norm = math.sqrt(sum(q**2 for q in quat))
            assert abs(norm - 1.0) < 1e-6, (
                f"Quaternion should be unit at {angle_deg}deg, got norm={norm}"
            )


class TestPyBulletArmInterface:
    """Test that PyBulletArm correctly implements ArmInterface."""

    def test_is_subclass(self):
        from simulation.pybullet_arm import PyBulletArm
        from robot_model.arm_interface import ArmInterface
        assert issubclass(PyBulletArm, ArmInterface)

    def test_is_abstract_methods_implemented(self):
        """Check all abstract methods are defined."""
        from simulation.pybullet_arm import PyBulletArm
        from robot_model.arm_interface import ArmInterface

        abstract_methods = set(ArmInterface.__abstractmethods__)
        arm_methods = set(dir(PyBulletArm))
        missing = abstract_methods - arm_methods
        assert not missing, f"Missing abstract methods: {missing}"


class TestPyBulletGripperInterface:
    """Test that PyBulletGripper correctly implements GripperInterface."""

    def test_is_subclass(self):
        from simulation.pybullet_arm import PyBulletGripper
        from robot_model.arm_interface import GripperInterface
        assert issubclass(PyBulletGripper, GripperInterface)

    def test_is_abstract_methods_implemented(self):
        from simulation.pybullet_arm import PyBulletGripper
        from robot_model.arm_interface import GripperInterface

        abstract_methods = set(GripperInterface.__abstractmethods__)
        gripper_methods = set(dir(PyBulletGripper))
        missing = abstract_methods - gripper_methods
        assert not missing, f"Missing abstract methods: {missing}"


try:
    import pybullet
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False


@pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not installed")
class TestPyBulletArmPhysics:
    """Integration tests requiring PyBullet (skipped if not installed)."""

    @pytest.fixture
    def physics_env(self):
        """Create a PyBullet DIRECT mode environment."""
        import pybullet as p
        client = p.connect(p.DIRECT)
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        yield client
        p.disconnect(client)

    def test_arm_initialization(self, physics_env):
        from simulation.pybullet_arm import PyBulletArm
        arm = PyBulletArm(physics_env)
        assert arm.initialize()
        assert arm.is_ready()

    def test_arm_joint_query(self, physics_env):
        from simulation.pybullet_arm import PyBulletArm
        arm = PyBulletArm(physics_env)
        arm.initialize()
        joints = arm.get_joint_positions()
        assert isinstance(joints, np.ndarray)
        assert len(joints) > 0

    def test_arm_ee_pose(self, physics_env):
        from simulation.pybullet_arm import PyBulletArm
        arm = PyBulletArm(physics_env)
        arm.initialize()
        pose = arm.get_ee_pose()
        assert pose.shape == (4, 4)

    def test_arm_move_to_pose(self, physics_env):
        from simulation.pybullet_arm import PyBulletArm
        arm = PyBulletArm(physics_env)
        arm.initialize()

        # Top-down pose above the board
        target = np.eye(4)
        target[0, 0] = 1.0
        target[1, 1] = -1.0
        target[2, 2] = -1.0
        target[0, 3] = 0.3
        target[1, 3] = 0.0
        target[2, 3] = 0.4

        result = arm.move_to_pose(target)
        assert result  # IK should succeed for reachable target
