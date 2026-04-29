#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# GaussianAvatars — deterministic installer for Python 3.11 + PyTorch 2.9.1 +
# CUDA 12.8 (RTX 5090 / Blackwell sm_120 ready).
# -----------------------------------------------------------------------------
# This mirrors the `install_128.sh` / `setup.sh` pattern used on the companion
# branches:
#   https://github.com/MTamon/DECA/tree/release/cuda128
#   https://github.com/MTamon/HRAvatar/tree/release/cuda128
#   https://github.com/MTamon/FlashAvatar/tree/release/cuda128-fixed
#
# Usage:
#   bash setup.sh              # create conda env gaussian-avatars, then install everything
#   bash setup.sh --pip-only   # skip conda, install into the active python
#
# Preconditions:
# - System CUDA Toolkit 12.8 is installed (nvcc must be on PATH, or CUDA_HOME set).
# - gcc-11 / g++-11 are installed (Ubuntu 22.04: `sudo apt install gcc-11 g++-11`).
# - Git submodules are initialized:
#     git submodule update --init --recursive
# -----------------------------------------------------------------------------

# This script mirrors MTamon/HRAvatar/setup.sh as closely as possible to keep
# library versions aligned. The `--no-deps` flag is used everywhere to prevent
# pip from mutating the pin set via transitive resolution.

# After pinned deps, it additionally:
#   1. installs chumpy from GitHub (numpy 2.x compatible main branch) — the
#      FLAME pickle in flame_model/flame.py loads chumpy objects via
#      `pickle.load(..., encoding="latin1")` so chumpy must be importable,
#   2. source-builds nvdiffrast from NVlabs HEAD,
#   3. builds the two local CUDA extensions under submodules/.

set -eo pipefail


PIP_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --pip-only)   PIP_ONLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"


# ----------------------------------------------------------------------------
# Toolchain setup for CUDA extensions.
# ----------------------------------------------------------------------------
# PyTorch 2.9 + CUDA 12.8 needs gcc <= 13; gcc-11 is the DECA128-tested choice.
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"

# Point CUDA_HOME at the system CUDA 12.8 install (Ubuntu standard path).
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
  else
    echo "[setup.sh] WARNING: CUDA_HOME is not set and /usr/local/cuda-12.8 was not found."
    echo "[setup.sh] Set CUDA_HOME manually to your CUDA 12.8 install path before rerunning."
    exit 1
  fi
fi
export PATH="${CUDA_HOME}/bin:${PATH}"

# Force nvcc to emit code for common modern arches; narrow to your own GPU
# to speed up the build (e.g. TORCH_CUDA_ARCH_LIST="12.0" for RTX 5090).
# Turing 7.5, Ampere 8.0/8.6, Ada 8.9, Hopper 9.0, Blackwell 12.0.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;12.0}"

# Force submodule setups to build with CUDA support.
export FORCE_CUDA=1

echo "[setup.sh] CC=${CC} CXX=${CXX}"
echo "[setup.sh] CUDA_HOME=${CUDA_HOME}"
echo "[setup.sh] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
nvcc --version || { echo "[setup.sh] nvcc not found on PATH"; exit 1; }


if [[ ${PIP_ONLY} -eq 0 ]]; then
  echo "[1/5] Creating conda env gaussian-avatars (Python 3.11, CUDA 12.8 toolkit)"

  if conda env list | awk '{print $1}' | grep -qx "gaussian-avatars"; then
    echo " -> conda env 'gaussian-avatars' already exists, skipping creation."
  else
    conda env create --file environment.yml
  fi

  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate gaussian-avatars
else
  echo "[1/5] Using currently-active python (pip-only mode)"
fi


# ----------------------------------------------------------------------------
# 2. Upgrade pip and install pinned dependencies (HRAvatar128-aligned).
# ----------------------------------------------------------------------------
echo "[2/5] Upgrading pip to 25.2 (matches MTamon install_128.sh)"
python -m pip install --upgrade pip==25.2

# Pillow first (Image is imported by render.py / metrics.py / local_viewer.py
# / scene/dataset_readers.py / scene/__init__.py).
python -m pip install --no-deps pillow==12.0.0

# chumpy (mattloper master, pinned to a specific SHA — numpy 2.x friendly;
# HRAvatar128/DECA128 use the exact same source). The PyPI wheel (0.70) breaks
# under numpy 2.x; the master branch is patched but unreleased. Required
# because flame_model/flame.py loads the FLAME pkl with encoding="latin1",
# which deserialises chumpy.Ch arrays.
#
# SHA 580566ea is the Aug-2025 maintenance commit
# ("ci: drop Python 2 checks; migrate CI to CircleCI 2.1 + Python 3.12").
# At this SHA chumpy/version.py reports '0.71', matching the chumpy-fork
# PyPI fork that the companion VHAP branch uses, so version checks across
# both stacks line up.
CHUMPY_SHA="580566eafc9ac68b2614b64d6f7aaa84eebb70da"
python -m pip install "git+https://github.com/mattloper/chumpy.git@${CHUMPY_SHA}"
python - <<'PY'
import chumpy
assert chumpy.__version__ == "0.71", \
    f"chumpy version drift: got {chumpy.__version__}, expected 0.71"
assert hasattr(chumpy, "Ch"), \
    "chumpy.Ch missing — FLAME pickle deserialisation will fail"
PY

# Pinned deps. Installed with --no-deps to avoid pip rewriting the pin set.
python -m pip install --no-deps filelock==3.20.0
python -m pip install --no-deps fsspec==2025.10.0
python -m pip install --no-deps iopath==0.1.10
python -m pip install --no-deps ninja==1.13.0
python -m pip install --no-deps numpy==2.2.6
python -m pip install --no-deps nvidia-cublas-cu12==12.8.4.1
python -m pip install --no-deps nvidia-cuda-cupti-cu12==12.8.90
python -m pip install --no-deps nvidia-cuda-nvrtc-cu12==12.8.93
python -m pip install --no-deps nvidia-cuda-runtime-cu12==12.8.90
python -m pip install --no-deps nvidia-cudnn-cu12==9.10.2.21
python -m pip install --no-deps nvidia-cufft-cu12==11.3.3.83
python -m pip install --no-deps nvidia-cufile-cu12==1.13.1.3
python -m pip install --no-deps nvidia-curand-cu12==10.3.9.90
python -m pip install --no-deps nvidia-cusolver-cu12==11.7.3.90
python -m pip install --no-deps nvidia-cusparse-cu12==12.5.8.93
python -m pip install --no-deps nvidia-cusparselt-cu12==0.7.1
python -m pip install --no-deps nvidia-nccl-cu12==2.27.5
python -m pip install --no-deps nvidia-nvjitlink-cu12==12.8.93
python -m pip install --no-deps nvidia-nvshmem-cu12==3.3.20
python -m pip install --no-deps nvidia-nvtx-cu12==12.8.90
python -m pip install --no-deps scipy==1.16.3
python -m pip install --no-deps torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps tqdm==4.67.1
python -m pip install --no-deps triton==3.5.1
python -m pip install --no-deps typing_extensions==4.15.0

# tensorboard (train.py imports torch.utils.tensorboard.SummaryWriter).
# 2.20.0 is the first release that explicitly supports numpy 2.x.
python -m pip install --no-deps absl-py==2.3.1
python -m pip install --no-deps grpcio==1.80.0
python -m pip install --no-deps markdown==3.10.2
python -m pip install --no-deps protobuf==4.25.5
python -m pip install --no-deps tensorboard==2.20.0
python -m pip install --no-deps tensorboard-data-server==0.7.2
python -m pip install --no-deps werkzeug==3.1.8

# matplotlib + transitive deps (matplotlib is imported by scene/__init__.py,
# render.py, etc.).
python -m pip install --no-deps contourpy==1.3.3
python -m pip install --no-deps cycler==0.12.1
python -m pip install --no-deps fonttools==4.60.1
python -m pip install --no-deps kiwisolver==1.4.9
python -m pip install --no-deps matplotlib==3.10.7
python -m pip install --no-deps packaging==25.0
python -m pip install --no-deps pyparsing==3.2.5
python -m pip install --no-deps python-dateutil==2.9.0.post0
python -m pip install --no-deps six==1.17.0

# Other torch / generic transitive deps.
python -m pip install --no-deps charset-normalizer==3.4.0
python -m pip install --no-deps certifi==2024.8.30
python -m pip install --no-deps idna==3.10
python -m pip install --no-deps jinja2==3.1.6
python -m pip install --no-deps markupsafe==3.0.3
python -m pip install --no-deps mpmath==1.3.0
python -m pip install --no-deps networkx==3.5
python -m pip install --no-deps requests==2.32.3
python -m pip install --no-deps sympy==1.14.0
python -m pip install --no-deps urllib3==2.2.3

# GaussianAvatars-specific runtime libs.
# plyfile -- ply save/load in scene/gaussian_model.py.
python -m pip install --no-deps plyfile==1.1.2
# roma -- quaternion math in scene/gaussian_model.py / flame_gaussian_model.py.
python -m pip install --no-deps roma==1.5.1
# tyro -- CLI parsing in local_viewer.py / remote_viewer.py / fps_benchmark_*.
python -m pip install --no-deps docstring-parser==0.16
python -m pip install --no-deps shtab==1.7.1
python -m pip install --no-deps colorama==0.4.6
python -m pip install --no-deps rich==13.9.4
python -m pip install --no-deps pygments==2.19.1
python -m pip install --no-deps mdurl==0.1.2
python -m pip install --no-deps markdown-it-py==3.0.0
python -m pip install --no-deps typeguard==4.4.2
python -m pip install --no-deps tyro==0.9.13
# dearpygui -- viewer UI in local_viewer.py / remote_viewer.py.
python -m pip install --no-deps dearpygui==2.1.4


# ----------------------------------------------------------------------------
# 3. nvdiffrast (source build from MTamon/nvdiffrast cuda128-backface-culling).
# ----------------------------------------------------------------------------
# Used by mesh_renderer/__init__.py for mesh rasterisation in the viewer and
# during training when --bind_to_mesh is on.
#
# Source: https://github.com/MTamon/nvdiffrast/tree/cuda128-backface-culling
# This fork carries CUDA 12.8 / sm_120 compatibility patches plus an optional
# back-face culling path that the GaussianAvatars mesh renderer benefits from.
# Must be built with --no-build-isolation so it can find the already-installed torch.
echo "[3/5] Building nvdiffrast from MTamon/nvdiffrast@cuda128-backface-culling"
NVDIFFRAST_TMP="$(mktemp -d)"
git -C "${NVDIFFRAST_TMP}" init -q
git -C "${NVDIFFRAST_TMP}" remote add origin https://github.com/MTamon/nvdiffrast.git
git -C "${NVDIFFRAST_TMP}" fetch --depth 1 origin cuda128-backface-culling
git -C "${NVDIFFRAST_TMP}" checkout FETCH_HEAD
python -m pip install --no-build-isolation --no-deps "${NVDIFFRAST_TMP}"
rm -rf "${NVDIFFRAST_TMP}"


# ----------------------------------------------------------------------------
# 4. Local CUDA extensions (diff-gaussian-rasterization and simple-knn).
# ----------------------------------------------------------------------------
# The submodules MUST be populated before running this script:
#   git submodule update --init --recursive
#
# Pinned commits (verified to build under PyTorch 2.9.1 + CUDA 12.8 + sm_120
# in the FlashAvatar release/cuda128-fixed branch):
#   - diff-gaussian-rasterization: 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
#   - simple-knn:                  60f461f4a56b7967e5d8045bf92f8c33f36976d0
echo "[4/5] Building local CUDA extensions (diff-gaussian-rasterization, simple-knn)"

if [ ! -f submodules/diff-gaussian-rasterization/setup.py ] || [ ! -f submodules/simple-knn/setup.py ]; then
  echo "[setup.sh] submodules not initialized."
  echo "[setup.sh] Run: git submodule update --init --recursive"
  exit 1
fi

python -m pip install --no-build-isolation --no-deps ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation --no-deps ./submodules/simple-knn


# ----------------------------------------------------------------------------
# 5. Sanity check.
# ----------------------------------------------------------------------------
echo "[5/5] Sanity check"
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "tv", torchvision.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
import nvdiffrast.torch  # noqa: F401
import diff_gaussian_rasterization  # noqa: F401
import simple_knn  # noqa: F401
import chumpy  # noqa: F401
import roma, plyfile, tyro, dearpygui  # noqa: F401
import iopath  # noqa: F401
print("OK")
PY

# Optional: print pip's view of the dependency graph. With --no-deps installs
# we expect a small number of "X requires Y, which is not installed" lines for
# pure-optional extras; any real version conflict should be surfaced and
# addressed here.
echo "[5/5] pip check (informational)"
python -m pip check || true


echo "Done."
