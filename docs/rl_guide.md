# Reinforcement Learning Guide

This guide covers the RL research subsystem: environments, training, evaluation, and extending.

## Architecture

```
learning/
├── envs/
│   ├── grasp_env.py        # ChessGraspEnv — grasp acquisition
│   ├── placement_env.py    # ChessPlacementEnv — placement refinement
│   └── rewards.py          # Shared reward functions
├── training/
│   ├── trainer.py          # Trainer class (PPO via stable-baselines3)
│   └── __init__.py
├── heuristic_baselines.py  # Hand-coded policies for comparison
├── eval_harness.py         # Systematic evaluation framework
├── interfaces.py           # Policy interface contracts
├── datasets/
│   └── collector.py        # Demonstration collection
├── tests/
│   ├── test_grasp_env.py
│   └── test_placement_and_eval.py
└── scripts/                # (in repo root: scripts/)
    ├── run_first_training.py
    └── evaluate_policies.py
```

## Environments

### ChessGraspEnv

**Task**: Acquire a stable grasp on a chess piece under pose uncertainty.

| | |
|-|-|
| **Observation** | 28-dim: relative piece pose (6), piece type one-hot (6), neighbor distances (8), gripper width (1), EE pose (7) |
| **Action** | 7-dim: Cartesian delta (6) + gripper command (1) |
| **Success** | Piece lifted >20mm and held stable for 25 steps |
| **Max steps** | 200 |

**Reward structure**:
- `+10.0` successful grasp/lift
- `+2.0` first contact
- `-0.5` per step (efficiency pressure)
- `-5.0` collision
- `-10.0` piece knocked over
- `+1.0` shaped (distance reduction)

### ChessPlacementEnv

**Task**: Precisely center a held piece on the target square.

| | |
|-|-|
| **Observation** | 18-dim: target offset (2), height (1), piece type (6), holding flag (1), EE pose (7), step fraction (1) |
| **Action** | 4-dim: XY correction (2) + Z delta (1) + gripper command (1) |
| **Success** | Piece placed within 3mm of square center |
| **Max steps** | 100 |

## Quick Start

### 1. Evaluate Baselines

```bash
# Compare heuristic vs random on both environments
python scripts/evaluate_policies.py --episodes 200

# Grasp env only
python scripts/evaluate_policies.py --env grasp --episodes 100
```

### 2. Train a PPO Policy

```bash
# Quick demo (50K steps, ~2 min)
python scripts/run_first_training.py --steps 50000

# Production training (1M steps)
python scripts/run_first_training.py --steps 1000000 --eval-episodes 200
```

### 3. Evaluate Trained Model

```bash
# After training, use the eval harness
python -c "
from learning.eval_harness import EvalHarness
from stable_baselines3 import PPO

model = PPO.load('logs/first_training/ppo_grasp_model')
harness = EvalHarness()
result = harness.evaluate(
    lambda obs: model.predict(obs, deterministic=True)[0],
    label='ppo', n_episodes=200
)
harness.print_result(result)
"
```

## Extending

### Adding a New Environment

1. Create `learning/envs/your_env.py` implementing `gymnasium.Env`
2. Define observation/action spaces matching the task
3. Add tests in `learning/tests/`
4. Register with `gymnasium.register()` if needed

### Adding a New Policy

1. Implement the policy function: `def policy(obs: np.ndarray) -> np.ndarray`
2. Add it to `scripts/evaluate_policies.py`
3. Compare against baselines using the `EvalHarness`

### Using Isaac Lab (GPU Training)

For massively parallel training on GPU:

1. Create `learning/envs/isaac_lab_adapter.py` wrapping `ChessGraspEnv`
2. Configure IsaacSim environment with the chess board URDF
3. Use RSL-RL or rl_games for multi-GPU PPO training
4. Export trained policy to ONNX for deployment

## Design Decisions

1. **Self-contained physics**: The gym environments use simplified internal physics (no PyBullet dependency) for fast prototyping. Isaac Lab should be used for production-quality sim.

2. **Gymnasium compliance**: Both environments pass `gymnasium.utils.env_checker.check_env()`.

3. **Deterministic evaluation**: The eval harness uses explicit seeds for reproducibility.

4. **Policy interface**: All policies are simple `obs → action` callables, making it trivial to swap between learned, heuristic, and scripted approaches.
