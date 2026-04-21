# Environment Baseline

Pinned at Milestone 1 commit. All subsequent training and benchmarking
must reproduce this environment for result validity.

## System
- **OS**: Windows 10 (10.0.26200)
- **GPU**: NVIDIA GeForce RTX 4070 Ti (12.9 GB VRAM)
- **Driver**: NVIDIA 581.95
- **CUDA**: 13.0 (driver), 12.4 (PyTorch wheels — backward compatible)
- **cuDNN**: 9.1.0

## Python
- **Version**: 3.11.9
- **Virtualenv**: `.venv/` (created with virtualenv 21.2.4)

## Key Packages
| Package | Version |
|---------|---------|
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| stable-baselines3 | 2.8.0 |
| gymnasium | 1.2.3 |
| numpy | 2.4.4 |
| opencv-python | 4.13.0.92 |
| python-chess | 1.999 |
| scipy | 1.17.1 |

## PyTorch Installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> **Note on CUDA 13.0 driver**: PyTorch doesn't publish CU13 wheels yet.
> CU12.4 wheels are forward-compatible with newer drivers.
> If exotic libraries break (SB3 JIT, d3rlpy, PyBullet GPU rendering),
> downgrade to Studio 560-series driver rather than chasing bleeding edge.

## GPU Verification
```
torch.cuda.is_available() → True
torch.cuda.get_device_name(0) → NVIDIA GeForce RTX 4070 Ti
torch.version.cuda → 12.4
torch.backends.cudnn.version() → 90100
GPU matmul smoke test → OK
```

## Test Suite
```
215 passed, 4 skipped (13.08s)
```
