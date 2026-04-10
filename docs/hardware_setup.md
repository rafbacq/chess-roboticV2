# Hardware Setup Guide

This guide covers physical setup and configuration for the chess robot system.

## Supported Hardware

### Arms
| Arm | Driver | Config |
|-----|--------|--------|
| UFACTORY xArm6 | `robot_model.xarm6_driver.XArm6Arm` | `config/hardware/xarm6.yaml` |
| Simulated (testing) | `robot_model.arm_interface.SimulatedArm` | Built-in |

### Grippers
| Gripper | Driver | Config |
|---------|--------|--------|
| xArm Gripper | `robot_model.xarm6_driver.XArmGripper` | `config/hardware/xarm6.yaml` |
| Simulated (testing) | `robot_model.arm_interface.SimulatedGripper` | Built-in |

### Cameras
| Camera | Driver | Config |
|--------|--------|--------|
| USB Webcam (1080p) | `perception.camera_interface.OpenCVCamera` | `config/perception/camera_rgb.yaml` |
| Intel RealSense | `perception.camera_interface.RealSenseCamera` (future) | — |
| Simulated | `perception.camera_interface.SimulatedCamera` | Built-in |

## Physical Layout

```
                Camera (overhead, ~60cm above board)
                    |
                    v
    +---+---+---+---+---+---+---+---+
    | a8| b8| c8| d8| e8| f8| g8| h8|  ← Black side
    +---+---+---+---+---+---+---+---+
    |   |   |   |   |   |   |   |   |
    ...                              ...
    +---+---+---+---+---+---+---+---+
    | a1| b1| c1| d1| e1| f1| g1| h1|  ← White side
    +---+---+   +---+---+---+---+---+
              ^
              |
        Robot base (xArm6)
```

### Mounting Requirements

1. **Robot**: Mount the xArm6 base so that the board is within the 691mm reach envelope
2. **Camera**: Overhead mount, centered above the board, looking straight down
3. **Board**: Must be rigidly fixed — any movement invalidates calibration
4. **Lighting**: Uniform diffuse lighting; avoid harsh shadows and specular reflections

## Network Configuration

The xArm6 communicates over Ethernet:

| Parameter | Value |
|-----------|-------|
| Robot IP | `192.168.1.197` (default) |
| Host IP | `192.168.1.x` (same subnet) |
| Protocol | TCP |
| SDK | `xarm-python-sdk` |

```bash
# Install the xArm SDK
pip install xarm-python-sdk

# Test connection
python -c "from xarm.wrapper import XArmAPI; arm = XArmAPI('192.168.1.197'); arm.connect(); print('Connected!')"
```

## Configuration

Edit `config/default.yaml` to select hardware:

```yaml
hardware:
  arm_type: xarm6              # 'simulated' for testing
  arm_ip: "192.168.1.197"
  gripper_type: xarm_gripper   # 'simulated' for testing

perception:
  camera_type: opencv
  camera_device_id: 0
  resolution: [1280, 720]

board:
  square_size_mm: 57.0         # Measure your board!
  origin_corner: a1
```

## First Run Checklist

1. [ ] Mount robot, camera, and board
2. [ ] Power on xArm6, verify network connectivity
3. [ ] Run camera intrinsic calibration (see [Calibration Guide](calibration_guide.md))
4. [ ] Run hand-eye calibration
5. [ ] Run board calibration
6. [ ] Test with `python -m chess_robotic play --white human --black human`
7. [ ] Verify first move (e2e4) — robot should pick and place cleanly

## Safety

- **Emergency stop**: The xArm6 has a hardware E-stop button. Keep it accessible.
- **Speed limits**: Production moves run at 30% velocity by default (`velocity_scale=0.3`)
- **Force limits**: The driver monitors force thresholds (default: 50N)
- **Mock mode**: If the SDK can't connect, the driver enters mock mode automatically. No physical motion occurs.

## Adding a New Arm

To add support for a different arm (e.g., Franka FR3):

1. Create `robot_model/franka_driver.py` implementing `ArmInterface`
2. Create `config/hardware/franka.yaml` with arm-specific parameters
3. Register in `system_factory.py`:
   ```python
   register_arm_driver("franka", FrankaArm)
   ```
4. Set `arm_type: franka` in your config YAML

The system will automatically use your driver via the factory.
