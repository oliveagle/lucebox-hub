# Contributing to Luce DFlash

Thank you for your interest in contributing. This document covers the system setup required to build and develop the project.

## System Requirements

### Hardware

- NVIDIA GPU with compute capability sm_86 or higher (RTX 3090, A10, A40, 4090)
- 24 GB VRAM minimum to run inference (build/tests only need the GPU at compile time)

### Required system packages

On Ubuntu 22.04 or 24.04, a setup script handles everything — apt build tools, the NVIDIA CUDA repo keyring, and `nvcc` PATH configuration:

```bash
sudo scripts/setup_system.sh
```

The script is idempotent: it skips steps that are already done. For other distros, see the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and install the packages below manually.

| Tool | Minimum version | Notes |
|------|----------------|-------|
| GCC / G++ | 11 | Ships as `build-essential` on Ubuntu 22.04+ |
| CMake | 3.18 | `cmake` package; Ubuntu 22.04 ships 3.22, Ubuntu 24.04 ships 3.28 |
| Git | 2.x | Needed for submodule checkout |
| git-lfs | any | Required to pull the LFS-tracked model blobs in `deps/` |
| CUDA Toolkit | 12.0+ | Provides `nvcc`; the driver alone (`nvidia-smi`) is not sufficient |

> **Note:** The `nvidia-cuda-toolkit` package from the Ubuntu universe repository is
> typically older (12.0 on noble). The setup script installs from NVIDIA's own repo
> so the toolkit version matches your driver.

## Getting the source

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub/dflash
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

git-lfs objects are fetched automatically by `git clone` and `git pull` once
`git lfs install` has been run (it installs per-user on first use).

## Building

```bash
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_dflash -j
```

To build all targets including the smoke tests:

```bash
cmake --build build -j
```

## Running the tests

Smoke tests require the model files described in the [Quick start](README.md#quick-start) section. Numeric correctness is validated against a pre-generated oracle:

```bash
build/test_vs_oracle
```

## Python scripts

The `scripts/` and `examples/` directories require Python 3.10+ with a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn transformers jinja2
```

## Submitting changes

Open a pull request against `Luce-Org/lucebox-hub`. See the [Contributing section in the README](README.md#contributing) for a list of good first issues.
