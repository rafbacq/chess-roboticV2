# WSL2 + CUDA + PyTorch Setup Guide

## Overview

The chess-roboticV2 project runs all Python code inside **WSL2 Ubuntu 22.04** on a Windows host with an **RTX 4070 Ti (12 GB VRAM)**, CUDA 13.0 driver. This document covers the complete setup.

## Prerequisites

- Windows 11 (or Windows 10 21H2+) with WSL2 support
- NVIDIA RTX 4070 Ti with latest Game Ready or Studio driver (≥560-series recommended)
- CUDA 13.0 driver (comes with driver; no separate CUDA toolkit install needed in WSL2)

## 1. Install WSL2

```powershell
# From elevated PowerShell:
wsl --install -d Ubuntu-22.04
# Restart if prompted
```

After reboot, launch Ubuntu-22.04 from the Start menu and create a user account.

## 2. Verify NVIDIA GPU in WSL2

```bash
nvidia-smi
# Should show RTX 4070 Ti, Driver Version 560.xx+, CUDA Version 13.0
```

> **Note**: If `nvidia-smi` fails, ensure you have the NVIDIA driver for WSL (not a standalone CUDA toolkit). The Windows GPU driver automatically provides CUDA support inside WSL2.

## 3. Install Python 3.11

```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

## 4. Create Virtual Environment

```bash
cd /mnt/c/Users/tdc65/chess-roboticV2
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

## 5. Install PyTorch (CU12.4 wheels)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Verify GPU

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

Expected output:
```
PyTorch: 2.6.x
CUDA available: True
CUDA version: 12.4
GPU: NVIDIA GeForce RTX 4070 Ti
GPU memory: 12.0 GB
```

> **Baseline fact**: GPU device = `NVIDIA GeForce RTX 4070 Ti`, 12 GB VRAM, CUDA 12.4 via PyTorch wheels on CUDA 13.0 driver.

## 6. Install Project Dependencies

```bash
pip install -e ".[all]"
```

### Bleeding-Edge Driver Caveat

If `stable-baselines3`, `d3rlpy`, or `PyBullet` break on the CUDA 13.0 driver, install NVIDIA Studio driver 560-series instead of Game Ready:
- Download from: https://www.nvidia.com/drivers/
- Select "Studio Driver" instead of "Game Ready"
- This provides a more stable CUDA runtime

## 7. USB Device Passthrough (usbipd-win)

To pass the Raspberry Pi Pico's USB serial to WSL2:

### On Windows (elevated PowerShell):

```powershell
# Install usbipd-win
winget install --interactive --exact dorssel.usbipd-win

# List USB devices
usbipd list

# Bind and attach the Pico (replace BUSID with actual):
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

### In WSL2:

```bash
# Verify device appears:
ls /dev/ttyACM*
# Should show /dev/ttyACM0

# Add user to dialout group for serial access:
sudo usermod -a -G dialout $USER
# Log out and back in for group change to take effect
```

## 8. Run Tests

```bash
source .venv/bin/activate
python -m pytest -v
# Expected: all 198+ tests pass
```

## 9. Quick GPU Benchmark

```bash
python -c "
import torch, time
x = torch.randn(10000, 10000, device='cuda')
t0 = time.time()
for _ in range(100):
    y = x @ x
torch.cuda.synchronize()
print(f'100x matmul (10k×10k): {time.time()-t0:.2f}s')
"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvidia-smi` not found in WSL2 | Update Windows GPU driver to ≥560-series |
| `torch.cuda.is_available()` returns False | Ensure PyTorch CU12.4 wheels, not CPU-only |
| `/dev/ttyACM0` not appearing | Re-attach via `usbipd attach --wsl --busid <BUSID>` |
| OOM during training | Reduce batch size; 12GB VRAM supports batch_size=128 for perception classifier |
| `pybullet` segfaults | Known issue with CUDA 13.0 driver; use Studio 560 driver |
