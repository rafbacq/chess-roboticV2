# Architecture Overview

## System Design

The chess-robotic system is a **hybrid classical + optional-learning** stack
for a chess-playing robotic arm with overhead camera.

### Dual-Track Architecture

| Track | Purpose | Stack | Status |
|-------|---------|-------|--------|
| **A: Classical Baseline** | Reliable deterministic manipulation | ROS 2, MoveIt 2, OpenCV, Stockfish | Primary path |
| **B: Learning Research** | RL/IL for manipulation subtasks | Isaac Lab, PyTorch, Gymnasium | Optional extension |

The system is fully operational with Track B disabled.

### Design Principles

1. **Chess logic is deterministic** — Stockfish handles strategy, never RL
2. **Classical manipulation first** — calibration → IK → planning → execution → verification
3. **RL is isolated** — limited to subtasks (grasp scoring, slip recovery, placement refinement)
4. **Hardware-agnostic** — abstractions support multiple arms via config
5. **Simulation-first** — develop and test in sim before real hardware
6. **Explicit transforms** — all coordinate frames documented, no hidden assumptions

## Package Dependency Graph

```
                         ┌────────────┐
                         │ chess_msgs │ (ROS 2 messages)
                         └─────┬──────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐
    │ chess_core  │    │ board_state │    │  perception │
    │ (Stockfish, │    │ (geometry,  │    │ (camera,    │
    │  game logic)│    │  mapping)   │    │  detection) │
    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
           │                  │                   │
           │           ┌──────┴──────┐    ┌──────┴──────┐
           │           │ calibration │    │ robot_model │
           │           │ (transforms,│    │ (HAL, URDF, │
           │           │  persistence│    │  collision)  │
           │           └──────┬──────┘    └──────┬──────┘
           │                  │                   │
           │           ┌──────┴───────────────────┘
           │           │
    ┌──────┴───────────┴──────┐
    │     motion_planning     │
    │  (MoveIt 2, waypoints,  │
    │   Task Constructor)     │
    └──────────┬──────────────┘
               │
    ┌──────────┴──────────────┐
    │      manipulation       │───── (optional) ─────┐
    │  (pick/place, grasp,    │                       │
    │   special moves, retry) │              ┌────────┴───────┐
    └──────────┬──────────────┘              │    learning    │
               │                             │  (RL envs,    │
    ┌──────────┴──────────────┐              │   training,   │
    │       execution         │              │   baselines)  │
    │  (trajectory, watchdog, │              └────────────────┘
    │   telemetry, recovery)  │
    └──────────┬──────────────┘
               │
    ┌──────────┴──────────────┐
    │      simulation         │
    │  (Isaac Sim, PyBullet,  │
    │   domain randomization) │
    └─────────────────────────┘
```

## Move Execution Pipeline

When the system receives a move like `e2e4`:

```
1. VALIDATE     ─── chess_core validates against current board state
2. CLASSIFY     ─── move_parser identifies move type (normal/capture/castle/etc.)
3. CONVERT      ─── board_state converts squares to 3D board-frame poses
4. GRASP SELECT ─── grasp_policy generates ranked grasp candidates
5. PLAN         ─── motion_planning creates staged trajectory:
                     pre-grasp → approach → close → lift → transit →
                     pre-place → place → release → retreat
6. EXECUTE      ─── execution sends trajectory to robot, monitors
7. VERIFY       ─── perception checks source empty, target occupied
8. CONFIRM      ─── game_manager updates internal board state
9. RECOVER      ─── if verification fails → classify failure → retry/escalate
```

## Board State Modes

| Mode | Source of Truth | Camera Role |
|------|----------------|-------------|
| DGT Mode | DGT e-board | Verification, diagnostics, calibration |
| Vision-Only | Overhead camera | Primary board state inference |

## Hardware Tiers

| Tier | Arms | Class |
|------|------|-------|
| A (Research) | Franka FR3 | 7-DOF, force/torque, <0.1mm repeatability |
| B (Premium) | xArm 6/7, UR3e | 6/7-DOF, good ROS support |
| C (Educational) | SO-100/101, myCobot | Low-cost, limited payload |

All arms implement `ArmInterface`. A `HardwareRegistry` enables runtime selection.
