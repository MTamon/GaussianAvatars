#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Shared helpers for the demo scripts under demo/.
#
# - Resolves the repository root and the VHAP submodule path.
# - Provides a small wrapper around `conda activate` that errors clearly if the
#   target environment does not exist.
# - Defines default conda env names used across the demos.
# -----------------------------------------------------------------------------

set -eo pipefail

# Repo root = parent of demo/
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DEMO_DIR}/.." && pwd)"
VHAP_DIR="${REPO_ROOT}/submodules/VHAP"

# Conda env names. Override by exporting before invoking the demo scripts.
GA_ENV="${GA_ENV:-gaussian-avatars}"
VHAP_ENV="${VHAP_ENV:-VHAP}"

activate_env() {
  local env_name="$1"
  if [ -z "${env_name}" ]; then
    echo "[_common.sh] activate_env: missing env name" >&2
    exit 2
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "[_common.sh] conda is not on PATH. Install/initialise conda first." >&2
    exit 1
  fi

  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[_common.sh] conda env '${env_name}' not found." >&2
    if [ "${env_name}" = "${GA_ENV}" ]; then
      echo "  -> create it with: bash ${REPO_ROOT}/setup.sh" >&2
    elif [ "${env_name}" = "${VHAP_ENV}" ]; then
      echo "  -> create it with: bash ${VHAP_DIR}/setup.sh" >&2
    fi
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
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}
