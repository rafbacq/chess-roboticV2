# Chess-Robotic V2

A modular, production-quality chess-playing robotic arm system with computer vision, classical manipulation, and a reinforcement learning research layer.

## Architecture

The system follows a **hybrid classical + optional learning** design:

- **Track A (Classical)**: Deterministic chess engine (Stockfish) + calibrated perception + motion planning + staged pick-and-place execution
- **Track B (Learning)**: Isolated RL/IL modules for manipulation subtasks (grasp scoring, placement refinement, slip recovery)

Track B can be disabled entirely without affecting system functionality.

## Subsystems

| Package | Responsibility |
|---------|---------------|
| `chess_core` | Stockfish integration, game logic, move validation |
| `board_state` | Canonical board representation, coordinate mapping, DGT adapter |
| `perception` | Camera pipeline, board detection, piece detection, move verification |
| `calibration` | Camera intrinsic/extrinsic calibration, hand-eye calibration |
| `robot_model` | Hardware abstraction layer for arms and grippers |
| `motion_planning` | Waypoint planner, MoveIt 2 bridge, trajectory generation |
| `manipulation` | Pick-place primitives, grasp policy, special moves, retry logic |
| `execution` | Trajectory execution, monitoring, watchdog, telemetry |
| `simulation` | PyBullet scene builder, simulated arm/gripper adapters |
| `learning` | RL environments (ChessGraspEnv), PPO training, heuristic baselines, evaluation harness |
| `orchestrator` | Top-level game loop controller connecting all subsystems |
| `system_factory` | Config-driven system construction from YAML |

## Hardware Support

| Tier | Arms | Status |
|------|------|--------|
| A (Simulated) | SimulatedArm (built-in) | Fully tested |
| B (Production) | xArm6 (via xarm-python-sdk) | Driver complete, mock mode available |
| C (Advanced) | MoveIt 2 planning (ROS 2 Humble) | Bridge + fallback implemented |

## Quick Start

```bash
# Clone and setup workspace
git clone https://github.com/rafbacq/chess-roboticV2.git
cd chess-roboticV2

# Install core dependencies
pip install -e ".[dev]"

# Run the full test suite (170 tests)
python -m pytest -v

# Launch the CLI (play, train, evaluate, calibrate)
python -m chess_robotic --help

# Play a game with simulated hardware
python -m chess_robotic play --white human --black human

# Run a demo e2e4 move in PyBullet
python scripts/demo_e2e4_pybullet.py

# Start RL training (requires stable-baselines3)
python scripts/run_first_training.py
```

## Configuration

All system parameters are centralized in `config/default.yaml`:

```yaml
system:
  log_level: INFO
hardware:
  arm_type: simulated      # or "xarm6"
  gripper_type: simulated  # or "xarm_gripper"
board:
  square_size_mm: 57.0
engine:
  stockfish_path: stockfish
  depth: 15
```

## Board State Modes

- **DGT Mode**: Electronic board as source of truth, camera for verification
- **Vision-Only Mode**: Board state inferred from overhead camera
- **Manual UCI Mode**: Human enters moves via CLI (for testing)

## Project Structure

```
chess-roboticV2/
├── board_state/         # Board model, coordinate transforms, DGT adapter
├── calibration/         # Camera calibration pipeline (intrinsic + extrinsic)
├── chess_core/          # Stockfish engine, game manager, interfaces
├── config/              # YAML configuration files
├── docs/                # Architecture docs, frame conventions
├── execution/           # Trajectory executor, watchdog, telemetry
├── learning/            # RL environments, training, evaluation, baselines
├── manipulation/        # Pick-place, grasp policy, failure classification
├── motion_planning/     # Waypoint planner, MoveIt 2 bridge
├── perception/          # Camera, board detection, piece detection, verification
├── robot_model/         # Arm/gripper interfaces, xArm6 driver, simulated hardware
├── scripts/             # Demo scripts, training launchers
├── simulation/          # PyBullet scene builder, simulated arm/gripper
├── tests/               # Integration tests, system factory tests
├── orchestrator.py      # System orchestrator (game loop)
├── system_factory.py    # Config-driven system builder
└── chess_robotic.py     # CLI entry point
```

## Tests

The project has **170 passing tests** covering:
- Chess core (engine, game manager, move parsing)
- Board state model and coordinate transforms
- Perception (camera, piece detection, move verification)
- Manipulation (pick-place, grasp policy, failure classification)
- Motion planning (waypoint planner, MoveIt 2 fallback)
- Hardware drivers (xArm6 mock mode, simulated arm/gripper)
- Calibration pipeline (intrinsic, extrinsic)
- RL environments (ChessGraspEnv)
- System factory and orchestrator integration
- Simulation (PyBullet arm adapter)

## Documentation

See `docs/` for:
- [Architecture Overview](docs/architecture.md)
- [Frame Conventions](docs/frame_conventions.md)

## License

MIT
