# Gantry vs Arm: ML Component Transfer Analysis

## Hardware Comparison

| Property | xArm6 (6-DOF Arm) | Cartesian Gantry (XY+Z) |
|----------|-------------------|--------------------------|
| DOF | 6 revolute joints | 3 prismatic joints |
| Action space | 7-dim (Cartesian + gripper) | 4-dim (X, Y, Z, magnet) |
| Pickup mechanism | Parallel jaw gripper | Electromagnet (N52 neodymium) |
| Motion planning | RRT/PRM via MoveIt2 | Direct Cartesian interpolation |
| Orientation control | Full SO(3) | Fixed top-down only |
| Collision avoidance | Complex (self-collision + env) | Simple (Z clearance only) |

## Why RL is Unnecessary for the Gantry

The gantry's action space is so constrained that a **deterministic controller is provably optimal**:

1. **No orientation ambiguity**: The gantry can only approach from directly above. There is exactly one approach direction.
2. **No grasp policy needed**: The electromagnet either activates or not — binary, not continuous.
3. **Perfect repeatability**: Stepper motors with homing provide ±0.5mm repeatability. There is no kinematic uncertainty to learn from.
4. **Trivial path planning**: Move Z up → move XY → move Z down. No self-collision, no joint limits to navigate around.
5. **No force control**: The electromagnet's holding force is either sufficient or not. There's no gripper width/force to optimize.

### Formal Argument

The gantry's control problem reduces to:
```
π*(s) = [move_z(z_safe), move_xy(target_xy), move_z(z_pickup), mag_on, move_z(z_safe), ...]
```

This deterministic policy achieves 100% success rate given:
- Correct calibration (squares mapped to step counts)
- Functional electromagnet
- Pieces with steel washers

No learned policy can exceed 100%. Therefore, the deterministic policy is optimal. QED.

## ML Component Transfer Matrix

| Component | Transfers to Gantry? | Notes |
|-----------|:--------------------:|-------|
| **Perception classifier** | ✅ Yes | Hardware-agnostic. Camera sees the same board regardless of manipulator. Highest ROI ML investment. |
| **Grasp policy (RL)** | ❌ No | Designed for 7-dim continuous arm actions. Gantry has discrete magnet on/off. Deterministic controller is optimal. |
| **Placement policy (RL)** | ⚠️ Partial | Placement accuracy matters, but the gantry's stepper repeatability makes learned refinement unnecessary. The perception classifier can verify placement. |
| **Failure recovery classifier** | ✅ Yes | Failure classification (retry/abort/recalibrate) is hardware-agnostic. The *recovery actions* differ but the *classification* transfers. |
| **Imitation learning data** | ✅ Yes (format) | Demonstration format is shared, but arm IL data cannot train gantry policies. Gantry demos are trivial (deterministic trajectories). |
| **Evaluation harness** | ✅ Yes | Metrics (success rate, placement accuracy) are hardware-agnostic. |
| **Curriculum config** | ❌ No | Curriculum stages (large→small pieces) are for arm grasp difficulty. Gantry electromagnet doesn't care about piece size. |

## What the Gantry DOES Need

1. **Perception classifier** (M5): Identify which piece is on which square. This is the primary ML deliverable and is completely hardware-agnostic.

2. **Failure recovery classifier** (M8): When a pick-place fails (piece didn't attach to magnet, piece fell during transit), classify the failure and decide recovery strategy. The inputs (post-attempt image + state) and outputs (retry/abort/recalibrate) are hardware-agnostic.

3. **Deterministic controller** (`robot_model/gantry_driver.py`): A simple state machine that:
   - Homes all axes on startup
   - Converts algebraic squares to step coordinates via calibration
   - Executes pick-place as: lift Z → move XY → lower Z → magnet on → lift Z → move XY → lower Z → magnet off

4. **Calibration** (`calibration/gantry_calibration.py`): Map board corners to step coordinates. This is a simple affine transform, not ML.

## What to Skip for Gantry

- **All RL grasp/placement training** (M9): Skip entirely. Document as arm-backend-only.
- **Curriculum learning**: The `config/learning/grasp_rl.yaml:56-67` curriculum stays `enabled: false` for gantry.
- **MoveIt2/ROS integration**: Mark as deferred. The gantry doesn't need motion planning.
- **Isaac Lab adapter**: Not applicable for prismatic joints.

## Arm Backend (Future)

If the project later adds a 6-DOF arm backend:
- The RL environments (`learning/envs/grasp_env.py`, `placement_env.py`) are ready
- The PPO training pipeline (`learning/training/trainer.py`) works
- The grasp policy manager (`manipulation/grasp_policy.py:66-85`) has load paths for learned models
- Curriculum config just needs `enabled: true`

The system factory already supports `simulated`, `xarm6`, and other arm backends. Adding an arm is a config change + driver registration.
