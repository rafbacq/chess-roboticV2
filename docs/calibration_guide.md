# Calibration Guide

This guide covers the three calibration procedures needed to get the chess robot operational with real hardware.

## 1. Camera Intrinsic Calibration

**Purpose**: Determine focal length, principal point, and distortion coefficients for the overhead camera.

### Requirements
- Printed checkerboard pattern (9×6 inner corners, 25mm squares recommended)
- Camera mounted in its final operating position

### Procedure

```bash
# Capture calibration images (at least 15, from different angles)
python -m calibration.intrinsic --device 0 --output data/calibration/intrinsic/

# Run calibration
python -m calibration.intrinsic --compute \
    --images data/calibration/intrinsic/ \
    --pattern 9x6 \
    --square-size 25 \
    --output data/calibration/camera_intrinsics.npz
```

The output `camera_intrinsics.npz` contains:
- `camera_matrix`: 3×3 intrinsic matrix K
- `dist_coeffs`: Distortion coefficients (k1, k2, p1, p2, k3)
- `reprojection_error`: Mean reprojection error (should be < 0.5 px)

### Validation
```bash
python -m calibration.intrinsic --verify data/calibration/camera_intrinsics.npz
```

## 2. Hand-Eye (Extrinsic) Calibration

**Purpose**: Determine the fixed transform between the robot base frame and the camera frame (T_base_camera).

### Method: Eye-to-Hand

The camera is static (not mounted on the robot). We solve:

```
T_camera_board = T_camera_base × T_base_ee × T_ee_board
```

### Requirements
- Camera intrinsics calibrated (step 1)
- AprilTag or checkerboard mounted on the robot's end-effector
- Robot at 15+ recorded poses

### Procedure

```bash
# Record calibration poses (robot moves, camera sees the tag)
python -m calibration.extrinsic --record \
    --arm-type xarm6 \
    --arm-ip 192.168.1.197 \
    --n-poses 20 \
    --output data/calibration/handeye_poses/

# Compute the transform
python -m calibration.extrinsic --compute \
    --poses data/calibration/handeye_poses/ \
    --intrinsics data/calibration/camera_intrinsics.npz \
    --output data/calibration/T_base_camera.npz
```

## 3. Board-to-Robot Calibration

**Purpose**: Determine the transform from robot base to the chess board origin (a1 corner).

### Using AprilTags (Recommended)

Place 4 AprilTags at known board corners. The system uses the tag detections plus the camera extrinsics to compute:

```
T_robot_board = T_base_camera⁻¹ × T_camera_board
```

```bash
python -m calibration.board --detect \
    --intrinsics data/calibration/camera_intrinsics.npz \
    --extrinsics data/calibration/T_base_camera.npz \
    --output data/calibration/calibration.npz
```

### Manual Teaching (Fallback)

If AprilTags aren't available, manually teach the 4 corner positions:

```bash
python -m calibration.board --teach \
    --arm-type xarm6 \
    --arm-ip 192.168.1.197 \
    --output data/calibration/calibration.npz
```

The robot will prompt you to jog the arm to each of the 4 board corners (a1, h1, h8, a8).

## Output Files

All calibration outputs go to `data/calibration/`:

| File | Contents |
|------|----------|
| `camera_intrinsics.npz` | K, dist_coeffs |
| `T_base_camera.npz` | 4×4 hand-eye transform |
| `calibration.npz` | T_robot_board, T_board_robot, board_origin_robot |

## Tips

1. **Lighting**: Keep lighting consistent between calibration and operation
2. **Temperature**: Re-calibrate if the ambient temperature changes significantly (thermal expansion)
3. **Validation**: After calibration, command the robot to touch each board corner — it should land within 2mm
4. **Re-calibration**: Required if the camera or board is moved, even slightly
