## Requirements

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+ (Blackwell sm_120 / RTX 5090 supported on the cuda128 branch)
- 11 GB VRAM (we used RTX 2080Ti)

### Software Requirements

- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio for Windows, GCC for Linux)
- CUDA SDK for PyTorch extensions, install *after* Visual Studio or GCC
- C++ Compiler and CUDA SDK must be compatible
- FFMPEG to create result videos

### Additional python packages

- RoMa (for rotation representations)
- DearPyGUI (for viewer interface)
- NVDiffRast (for mesh rendering in viewer)

### Tested Platforms

| PyTorch Version | CUDA version | Linux | Windows (VS2022) | Windows (VS2019) |
|-| - | - | - | - |
| 2.0.1 | 11.7.1 | Pass | Fail to compile | Pass |
| 2.2.0 | 12.1.1 | Pass | Pass | Pass |
| 2.9.1 | 12.8 (sm_120) | Pass (Linux) | Not tested | Not tested |

## Installation

### Recommended (Linux, CUDA 12.8, RTX 5090 / Blackwell)

This branch ships a deterministic installer that mirrors the
`install_128.sh` / `setup.sh` pattern from
[MTamon/DECA](https://github.com/MTamon/DECA/tree/release/cuda128),
[MTamon/HRAvatar](https://github.com/MTamon/HRAvatar/tree/release/cuda128) and
[MTamon/FlashAvatar](https://github.com/MTamon/FlashAvatar/tree/release/cuda128-fixed).
Every dependency is pip-installed with `--no-deps` against a pinned version
list, so the resulting environment is reproducible across machines.

#### Preconditions

- System CUDA Toolkit **12.8** is installed (or the conda env created by
  `environment.yml` is used).
- gcc-11 / g++-11 are available (Ubuntu 22.04: `sudo apt install gcc-11 g++-11`).
- Submodules are initialised:

```shell
git clone https://github.com/MTamon/GaussianAvatars.git --recursive
cd GaussianAvatars
# (or) git submodule update --init --recursive
```

#### One-shot install

```shell
# Creates conda env `gaussian-avatars` (Python 3.11 + CUDA 12.8 toolkit)
# and installs every Python dep, nvdiffrast, and the two CUDA submodules.
bash setup.sh
```

To install into an already-active python (e.g. an existing conda or venv) and
skip the conda-env step:

```shell
bash setup.sh --pip-only
```

The script narrows `TORCH_CUDA_ARCH_LIST` to `7.5;8.0;8.6;8.9;9.0;12.0` by
default; export your own value beforehand to override (e.g.
`export TORCH_CUDA_ARCH_LIST="12.0"` for an RTX 5090-only build, which
significantly cuts compile time for the local CUDA extensions).

If pip reuses an older cached build, reinstall the local extensions with:

```shell
python -m pip install --no-build-isolation --no-deps --force-reinstall \
  --no-cache-dir ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation --no-deps --force-reinstall \
  --no-cache-dir ./submodules/simple-knn
```

#### Submodule pins

This branch uses the same submodule commits as
MTamon/FlashAvatar `release/cuda128-fixed`, both of which build cleanly
against PyTorch 2.9.1 + CUDA 12.8 with `TORCH_CUDA_ARCH_LIST` including
`12.0`:

- `diff-gaussian-rasterization`: `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`
  (graphdeco-inria upstream)
- `simple-knn`: `60f461f4a56b7967e5d8045bf92f8c33f36976d0`
  (camenduru fork; the original INRIA GitLab URL has been replaced)

### Legacy installation (CUDA 11.7 / 12.1)

The original conda + `requirements.txt` flow is still available for the older
PyTorch / CUDA combinations listed in the table above:

```shell
conda create --name gaussian-avatars -y python=3.10
conda activate gaussian-avatars
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt   # NOTE: only valid on the legacy branches
```

### 2. Setup paths

#### For Linux

```shell
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
conda env config vars set CUDA_HOME=$CONDA_PREFIX  # for compilation
```

#### For Windows with PowerShell

```shell
conda env config vars set CUDA_PATH="$env:CONDA_PREFIX"  

## Visual Studio 2022 (modify the version number `14.39.33519` accordingly)
conda env config vars set PATH="$env:CONDA_PREFIX\Script;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64;$env:PATH"
## or Visual Studio 2019 (modify the version number `14.29.30133` accordingly)
conda env config vars set PATH="$env:CONDA_PREFIX\Script;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;$env:PATH" 

# re-activate the environment to make the above eonvironment variables effective
conda deactivate
conda activate gaussian-avatars
```

#### For Windows with Command Prompt

```shell
conda env config vars set CUDA_PATH=%CONDA_PREFIX%

## Visual Studio 2022 (modify the version number `14.39.33519` accordingly)
conda env config vars set PATH="%CONDA_PREFIX%\Script;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64;%PATH%"
## or Visual Studio 2019 (modify the version number `14.29.30133` accordingly)
conda env config vars set PATH="%CONDA_PREFIX%\Script;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;%PATH%"

# re-activate the environment to make the above eonvironment variables effective
conda deactivate
conda activate gaussian-avatars
```
