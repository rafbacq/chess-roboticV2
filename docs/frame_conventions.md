# Coordinate Frame Conventions

## Overview

This document defines ALL coordinate frames used in the chess-robotic system.
**Read this before writing any code that manipulates transforms.**

## Frame Summary

| Frame Name | Origin | X | Y | Z |
|-----------|--------|---|---|---|
| `robot_base` / `base_link` | Robot mounting point | Robot forward | Robot left | Up |
| `board_link` | Center of square a1 | a→h (file direction) | 1→8 (rank direction) | Up |
| `camera_optical_frame` | Camera lens center | Right | Down | Forward |
| `camera_link` | Camera body center | Forward | Left | Up |
| `ee_link` | Tool flange center | (per URDF) | (per URDF) | (per URDF) |
| `grasp_link` | Gripper TCP | (per config) | (per config) | Along fingers |
| `tray_link` | First tray slot | Along tray length | Across tray | Up |

## Board Frame (`board_link`)

```
       Y-axis (rank direction, toward rank 8 / Black)
       ↑
       │
  a8   │  ···  h8
  ···  │  ···  ···
  a2   │  ···  h2
  a1───┼──···──h1 ──→ X-axis (file direction, a→h)
       │
       Origin = center of a1
```

### Rules
1. **Origin** = center of square a1 (file=0, rank=0)
2. **X-axis** = from a-file toward h-file
3. **Y-axis** = from rank 1 toward rank 8
4. **Z-axis** = up from board surface
5. **Units** = meters
6. **Square center** (file `f`, rank `r`) = `(f * square_size, r * square_size, 0)`
   - `f` ∈ {0, 1, ..., 7} mapping to {a, b, ..., h}
   - `r` ∈ {0, 1, ..., 7} mapping to {1, 2, ..., 8}

### Example Coordinates (57mm squares)
| Square | File | Rank | X (m) | Y (m) | Z (m) |
|--------|------|------|-------|-------|-------|
| a1 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| e1 | 4 | 0 | 0.228 | 0.000 | 0.000 |
| a8 | 0 | 7 | 0.000 | 0.399 | 0.000 |
| e4 | 4 | 3 | 0.228 | 0.171 | 0.000 |
| h8 | 7 | 7 | 0.399 | 0.399 | 0.000 |

## Camera Frame

The camera follows ROS convention:
- `camera_link`: body frame (X forward, Y left, Z up)
- `camera_optical_frame`: optical frame (X right, Y down, Z forward)

The transform between them is a fixed rotation.

## Transform Naming Convention

All transforms follow the convention:

```
T_target_source
```

Meaning: **transforms points FROM source frame TO target frame**.

Example:
- `T_robot_board` = transform from board frame to robot base frame
- `p_robot = T_robot_board @ p_board`

## Frame Graph

```
 robot_base (base_link)
    │
    ├── T_robot_ee ──── ee_link
    │                      │
    │                      └── T_ee_grasp ──── grasp_link
    │
    ├── T_robot_camera ──── camera_link
    │                          │
    │                          └── T_camera_optical ──── camera_optical_frame
    │
    └── T_robot_board ──── board_link
                              │
                              ├── square frames (computed)
                              │
                              └── T_board_tray ──── tray_link
```

## How Transforms Are Determined

| Transform | Method |
|-----------|--------|
| `T_robot_ee` | Forward kinematics from joint state |
| `T_ee_grasp` | Fixed offset from URDF or config |
| `T_robot_camera` | Hand-eye calibration (if camera on robot) or fixed-mount calibration |
| `T_camera_optical` | Fixed rotation per ROS convention |
| `T_robot_board` | Camera calibration → T_camera_board, then composed with T_robot_camera |
| `T_board_tray` | Fixed offset from config |

## Common Errors to Avoid

1. **Board mirroring**: Ensure a1 is at the BOTTOM-LEFT from White's perspective
2. **Off-by-one**: File/rank indexing starts at 0 internally, 1 in algebraic
3. **Units**: Everything internally in meters, config may use mm
4. **Camera convention**: Don't mix up camera_link and camera_optical_frame
5. **Transform direction**: `T_A_B` transforms FROM B TO A, not the other way
6. **Board orientation**: Verify board orientation matches player perspective before game start
