# Chess-Robotic V2

A modular, production-quality chess-playing robotic arm system with computer vision, classical manipulation, and an optional reinforcement learning research layer.

## Architecture

The system follows a **hybrid classical + optional learning** design:

- **Track A (Classical)**: Deterministic chess engine (Stockfish) + calibrated perception + motion planning (MoveIt 2) + staged pick-and-place execution
- **Track B (Learning)**: Isolated RL/IL modules for manipulation subtasks (grasp scoring, placement refinement, slip recovery)

Track B can be disabled entirely without affecting system functionality.

## Subsystems

| Package | Responsibility |
|---------|---------------|
| `chess_core` | Stockfish integration, game logic, move validation |
| `board_state` | Canonical board representation, coordinate mapping, DGT adapter |
| `perception` | Camera pipeline, board detection, piece detection, move verification |
| `calibration` | Hand-eye calibration, frame management, persistence |
| `robot_model` | Hardware abstraction layer for arms and grippers |
| `motion_planning` | MoveIt 2 integration, waypoint generation, constraints |
| `manipulation` | Pick-place primitives, grasp policy, special moves, retry logic |
| `execution` | Trajectory execution, monitoring, recovery |
| `simulation` | Isaac Sim digital twin, domain randomization |
| `learning` | RL environments, training, evaluation, heuristic baselines |
| `ui_tools` | Visualization, debugging, replay |
| `chess_msgs` | ROS 2 message/service/action definitions |

## Hardware Support

| Tier | Arms |
|------|------|
| A (Research) | Franka Research 3 |
| B (Premium) | xArm 6/7, UR3e, DOBOT Nova |
| C (Educational) | SO-100/101, myCobot, Ned2 |

## Quick Start

```bash
# Clone and setup workspace
git clone <repo-url> chess-roboticV2
cd chess-roboticV2

# Install dependencies
pip install -e ".[dev]"

# Install Stockfish (platform-dependent)
# See docs/hardware_setup.md

# Run unit tests
pytest tests/ chess_core/tests/ board_state/tests/

# Launch simulation demo
ros2 launch launch/demo_e2e4.launch.py
```

## Board State Modes

- **DGT Mode**: Electronic board as source of truth, camera for verification
- **Vision-Only Mode**: Board state inferred from overhead camera

## Documentation

See `docs/` for:
- [Architecture Overview](docs/architecture.md)
- [Frame Conventions](docs/frame_conventions.md)
- [Calibration Guide](docs/calibration_guide.md)
- [Simulation Setup](docs/simulation_setup.md)
- [Hardware Setup](docs/hardware_setup.md)
- [Adding a New Arm](docs/adding_new_arm.md)
- [Adding a New Gripper](docs/adding_new_gripper.md)
- [Adding a Learned Module](docs/adding_learned_module.md)
- [Debugging Guide](docs/debugging_guide.md)
- [Safety Notes](docs/safety_notes.md)

## License

MIT
