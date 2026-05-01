"""
System factory: config-driven construction of all subsystems.

Reads the master YAML configuration and instantiates the complete
chess robot system, including hardware, perception, calibration,
manipulation, and game management subsystems.

Usage:
    factory = SystemFactory("config/default.yaml")
    orchestrator = factory.build()
    orchestrator.start_game()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from board_state.board_model import BoardConfig
from chess_core.engine import EngineConfig
from chess_core.game_manager import GameConfig, PlayerType
from manipulation.grasp_policy import GraspPolicyConfig
from manipulation.pick_place import ManipConfig
from orchestrator import OrchestratorConfig, SystemOrchestrator
from robot_model.arm_interface import (
    ArmInterface,
    GripperInterface,
    SimulatedArm,
    SimulatedGripper,
)
from robot_model.xarm6_driver import XArm6Arm, XArmGripper
from robot_model.gantry_driver import GantryArm, GantryGripper
from motion_planning.moveit2_planner import MoveIt2Planner

logger = logging.getLogger(__name__)

# Registry of known hardware drivers (type_name -> class)
_ARM_DRIVERS: dict[str, type] = {
    "simulated": SimulatedArm,
    "xarm6": XArm6Arm,
    "gantry": GantryArm,
}
_GRIPPER_DRIVERS: dict[str, type] = {
    "simulated": SimulatedGripper,
    "xarm_gripper": XArmGripper,
    "electromagnet": GantryGripper,
}


def register_arm_driver(name: str, cls: type) -> None:
    """Register a hardware arm driver by name."""
    _ARM_DRIVERS[name] = cls


def register_gripper_driver(name: str, cls: type) -> None:
    """Register a hardware gripper driver by name."""
    _GRIPPER_DRIVERS[name] = cls


class SystemFactory:
    """
    Factory that builds the full system from YAML configuration.

    Reads config/default.yaml and constructs:
      - Hardware (arm + gripper)
      - Board model
      - Game manager
      - Manipulation pipeline
      - Orchestrator

    Usage:
        factory = SystemFactory("config/default.yaml")
        orch = factory.build()
        orch.start_game()
    """

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        self._config_path = Path(config_path)
        self._raw: dict = {}

    def load_config(self) -> dict:
        """Load and return the raw YAML configuration."""
        if not self._config_path.exists():
            logger.warning(
                f"Config not found at {self._config_path}, using defaults"
            )
            self._raw = {}
            return self._raw

        with open(self._config_path, "r") as f:
            self._raw = yaml.safe_load(f) or {}

        logger.info(f"Configuration loaded from {self._config_path}")
        return self._raw

    def build(
        self,
        arm_override: Optional[ArmInterface] = None,
        gripper_override: Optional[GripperInterface] = None,
    ) -> SystemOrchestrator:
        """
        Build the complete system orchestrator.

        Args:
            arm_override: Use this arm instead of config-specified one.
            gripper_override: Use this gripper instead of config-specified one.

        Returns:
            Fully configured SystemOrchestrator.
        """
        if not self._raw:
            self.load_config()

        cfg = self._raw

        # Build subsystem configs
        board_config = self._build_board_config(cfg.get("board", {}))
        game_config = self._build_game_config(cfg.get("engine", {}))
        manip_config = self._build_manip_config(cfg.get("manipulation", {}))
        grasp_config = self._build_grasp_config(cfg.get("learning", {}))

        # Hardware
        arm = arm_override or self._build_arm(cfg.get("hardware", {}))
        gripper = gripper_override or self._build_gripper(cfg.get("hardware", {}))

        # Initialize hardware
        if not arm.is_ready():
            arm.initialize()
        # GripperInterface has no is_ready — use get_status check
        from robot_model.arm_interface import GripperStatus
        if gripper.get_status() == GripperStatus.ERROR or not hasattr(gripper, '_initialized') or not getattr(gripper, '_initialized', False):
            gripper.initialize()


        # Transform
        T_robot_board = self._load_calibration(cfg.get("calibration", {}))

        # Orchestrator config
        orch_config = OrchestratorConfig(
            board_config=board_config,
            game_config=game_config,
            manip_config=manip_config,
            grasp_config=grasp_config,
            T_robot_board=T_robot_board,
        )

        # Build system
        orchestrator = SystemOrchestrator(arm, gripper, orch_config)

        # Configure logging
        sys_cfg = cfg.get("system", {})
        log_level = sys_cfg.get("log_level", "INFO")
        logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

        logger.info(
            f"System built: arm={cfg.get('hardware', {}).get('arm_type', 'simulated')}, "
            f"board={board_config.square_size_m * 1000:.0f}mm squares"
        )

        return orchestrator

    def _build_board_config(self, board_cfg: dict) -> BoardConfig:
        return BoardConfig(
            square_size_m=board_cfg.get("square_size_mm", 57.0) / 1000.0,
            board_thickness_m=board_cfg.get("board_thickness_mm", 15.0) / 1000.0,
            board_height_m=board_cfg.get("board_height_mm", 0.0) / 1000.0,
            tray_side=board_cfg.get("tray_side", "right"),
            tray_offset_x_m=board_cfg.get("tray_offset_mm", 100.0) / 1000.0,
        )

    def _build_game_config(self, engine_cfg: dict) -> GameConfig:
        return GameConfig(
            white_player=PlayerType.HUMAN,
            black_player=PlayerType.HUMAN,
            engine_config=EngineConfig(
                stockfish_path=engine_cfg.get("stockfish_path", "stockfish"),
                depth=engine_cfg.get("depth", 15),
                time_limit_ms=engine_cfg.get("time_limit_ms", 2000),
                threads=engine_cfg.get("threads", 2),
                hash_mb=engine_cfg.get("hash_mb", 256),
                skill_level=engine_cfg.get("skill_level", 20),
            ),
        )

    def _build_manip_config(self, manip_cfg: dict) -> ManipConfig:
        return ManipConfig(
            safe_height_m=manip_cfg.get("safe_height_m", 0.15),
            lift_height_m=manip_cfg.get("lift_height_m", 0.12),
            approach_clearance_m=manip_cfg.get("approach_clearance_m", 0.05),
            place_clearance_m=manip_cfg.get("place_clearance_m", 0.003),
            grasp_force_n=manip_cfg.get("grasp_force_n", 10.0),
        )

    def _build_grasp_config(self, learning_cfg: dict) -> GraspPolicyConfig:
        return GraspPolicyConfig(
            use_learned_grasp=learning_cfg.get("use_learned_grasp", False),
        )

    def _build_arm(self, hw_cfg: dict) -> ArmInterface:
        arm_type = hw_cfg.get("arm_type", "simulated")
        cls = _ARM_DRIVERS.get(arm_type)
        if cls is not None:
            return cls(name=f"{arm_type}_arm")
        logger.warning(f"Arm type '{arm_type}' not registered, using simulated")
        return SimulatedArm(name="sim_arm")

    def _build_gripper(self, hw_cfg: dict) -> GripperInterface:
        gripper_type = hw_cfg.get("gripper_type", "simulated")
        cls = _GRIPPER_DRIVERS.get(gripper_type)
        if cls is not None:
            return cls(name=f"{gripper_type}_gripper")
        logger.warning(f"Gripper type '{gripper_type}' not registered, using simulated")
        return SimulatedGripper(name="sim_gripper")

    def _load_calibration(self, cal_cfg: dict) -> np.ndarray:
        """Load calibration transform or return identity."""
        if cal_cfg.get("auto_load", False):
            cal_dir = cal_cfg.get("calibration_data_dir", "data/calibration")
            cal_file = Path(cal_dir) / "calibration.npz"
            if cal_file.exists():
                try:
                    data = np.load(str(cal_file))
                    if "T_robot_board" in data:
                        logger.info(f"Loaded calibration from {cal_file}")
                        return data["T_robot_board"]
                except Exception as e:
                    logger.warning(f"Failed to load calibration: {e}")

        return np.eye(4, dtype=np.float64)
