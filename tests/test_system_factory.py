"""
Tests for the system factory: config-driven construction of all subsystems.

Verifies that SystemFactory can build a complete orchestrator from
YAML config, and handles missing configs gracefully.
"""

import numpy as np
import pytest

from system_factory import SystemFactory, register_arm_driver, register_gripper_driver


class TestSystemFactory:

    def test_load_default_config(self):
        factory = SystemFactory("config/default.yaml")
        cfg = factory.load_config()
        assert isinstance(cfg, dict)
        assert "system" in cfg
        assert "hardware" in cfg
        assert "board" in cfg

    def test_load_missing_config(self):
        factory = SystemFactory("config/nonexistent.yaml")
        cfg = factory.load_config()
        assert cfg == {}

    def test_build_with_defaults(self):
        """Build an orchestrator using defaults (simulated hardware)."""
        factory = SystemFactory("config/default.yaml")
        orch = factory.build()
        assert orch is not None
        assert orch.game is not None
        assert orch.board_model is not None
        assert orch.pick_place is not None

    def test_build_without_config_file(self):
        """Build from scratch (no config file) — uses all defaults."""
        factory = SystemFactory("__nonexistent__.yaml")
        orch = factory.build()
        assert orch is not None
        assert orch.game is not None

    def test_build_with_arm_override(self):
        from robot_model.arm_interface import SimulatedArm
        arm = SimulatedArm(name="override_arm")
        arm.initialize()

        factory = SystemFactory("config/default.yaml")
        orch = factory.build(arm_override=arm)
        assert orch._arm is arm

    def test_build_with_gripper_override(self):
        from robot_model.arm_interface import SimulatedGripper
        grip = SimulatedGripper(name="override_grip")
        grip.initialize()

        factory = SystemFactory("config/default.yaml")
        orch = factory.build(gripper_override=grip)
        assert orch._gripper is grip

    def test_board_config_from_yaml(self):
        factory = SystemFactory("config/default.yaml")
        orch = factory.build()
        # Default YAML has square_size_mm: 57.0
        assert orch.board_model.config.square_size_m == pytest.approx(0.057)

    def test_start_and_stop_game(self):
        factory = SystemFactory("config/default.yaml")
        orch = factory.build()
        orch.start_game()
        assert orch.move_count == 0
        orch.stop_game()

    def test_execute_single_move(self):
        factory = SystemFactory("config/default.yaml")
        orch = factory.build()
        orch.start_game()
        result = orch.execute_turn(manual_uci="e2e4")
        from chess_core.interfaces import ExecutionStatus
        assert result.status == ExecutionStatus.SUCCESS
        assert orch.move_count == 1
        orch.stop_game()


class TestDriverRegistration:

    def test_register_custom_arm(self):
        from robot_model.arm_interface import SimulatedArm

        class CustomArm(SimulatedArm):
            pass

        register_arm_driver("custom_test", CustomArm)
        factory = SystemFactory("config/default.yaml")
        cfg = factory.load_config()
        # Manually tweak config to use our custom driver
        cfg.setdefault("hardware", {})["arm_type"] = "custom_test"
        orch = factory.build()
        assert isinstance(orch._arm, CustomArm)
