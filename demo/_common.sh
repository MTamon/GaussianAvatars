#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Shared helpers for the demo scripts under demo/.
#
# - Resolves the repository root and the VHAP submodule path.
# - Activates a single unified conda env (default: gaussian-avatars) that
#   carries both the GaussianAvatars and the VHAP dependency stack. Use
#   `bash demo/setup_env.sh` to populate it.
# - Forces unbuffered Python output so tqdm progress bars render live during
#   long-running preprocess / train / render stages.
# -----------------------------------------------------------------------------

set -eo pipefail

# Repo root = parent of demo/
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DEMO_DIR}/.." && pwd)"
VHAP_DIR="${REPO_ROOT}/submodules/VHAP"

# Single unified conda env. Demo entrypoints also expose --env for explicit
# per-run selection; GA_ENV remains as a low-level fallback for compatibility.
GA_ENV="${GA_ENV:-gaussian-avatars}"

# tqdm writes to stderr by default. Disabling Python stdout/stderr buffering
# ensures progress bars update in real time when scripts are executed under
# nohup/tee/CI runners. PYTHON is the canonical interpreter shim used below.
export PYTHONUNBUFFERED=1
PYTHON="python -u"

activate_env() {
  local env_name="${1:-${GA_ENV}}"

  if ! command -v conda >/dev/null 2>&1; then
    echo "[_common.sh] conda is not on PATH. Install/initialise conda first." >&2
    exit 1
  fi

  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[_common.sh] conda env '${env_name}' not found." >&2
    echo "  -> create the unified env with: bash ${DEMO_DIR}/setup_env.sh" >&2
    exit 1
  fi

  conda activate "${env_name}"
}

require_vhap_submodule() {
  if [ ! -f "${VHAP_DIR}/vhap/preprocess_video.py" ]; then
    echo "[_common.sh] VHAP submodule is not initialised at ${VHAP_DIR}." >&2
    echo "  -> run: git submodule update --init --recursive" >&2
    exit 1
  fi
}

log() {
  # Write to stderr so log lines remain visible even when a caller captures
  # a function's stdout via $(...) (e.g. run_track_one_clip in 01c, which uses
  # the captured stdout to return the export folder path).
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2
}

die_usage() {
  local script_name="$1"
  local message="$2"
  echo "[${script_name}] ${message}" >&2
  echo "Run 'bash demo/${script_name} --help' for usage." >&2
  exit 2
}

require_option_value() {
  local script_name="$1"
  local option_name="$2"
  local remaining_count="$3"
  if [[ "${remaining_count}" -lt 2 ]]; then
    die_usage "${script_name}" "${option_name} requires a value"
  fi
}
