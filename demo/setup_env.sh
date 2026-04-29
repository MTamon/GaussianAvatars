#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Unified environment installer: place GaussianAvatars + VHAP into a single
# conda env (default: `gaussian-avatars`).
#
# Both projects pin Python 3.11 + PyTorch 2.9.1 + CUDA 12.8 and use
# `--no-deps` everywhere, so a sequential install is safe. Where versions
# overlap, the second installer (VHAP) wins; this is intentional because
# VHAP's chumpy-fork alias mechanism needs to be the active one for both
# pipelines.
#
# Usage:
#   bash demo/setup_env.sh                # full install
#   bash demo/setup_env.sh --skip-ga      # skip GA setup (env already built)
#   bash demo/setup_env.sh --skip-vhap    # skip VHAP setup
#   GA_ENV=my-env bash demo/setup_env.sh  # override conda env name
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SKIP_GA=0
SKIP_VHAP=0
for arg in "$@"; do
  case "$arg" in
    --skip-ga)   SKIP_GA=1 ;;
    --skip-vhap) SKIP_VHAP=1 ;;
    -h|--help)
      sed -n '1,/^# ---/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "[setup_env.sh] unknown flag: $arg" >&2; exit 2 ;;
  esac
done

require_vhap_submodule

if [[ ${SKIP_GA} -eq 0 ]]; then
  log "[1/2] Installing GaussianAvatars stack into env '${GA_ENV}'"
  cd "${REPO_ROOT}"
  bash setup.sh
fi

# Activate the unified env before the VHAP --pip-only install.
if ! command -v conda >/dev/null 2>&1; then
  echo "[setup_env.sh] conda is not on PATH." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${GA_ENV}"

if [[ ${SKIP_VHAP} -eq 0 ]]; then
  log "[2/2] Adding VHAP stack into the same env (--pip-only)"
  cd "${VHAP_DIR}"
  ENV_NAME="${GA_ENV}" bash setup.sh --pip-only --no-assets
fi

log "Done. Both stacks share env '${GA_ENV}'."
log "Next steps:"
log "  1. Place FLAME assets (see demo/README.md §0.3)"
log "  2. bash demo/01_vhap_preprocess_monocular.sh   (or *_nersemble.sh)"
log "  3. bash demo/02_train.sh"
log "  4. bash demo/03_render.sh"
