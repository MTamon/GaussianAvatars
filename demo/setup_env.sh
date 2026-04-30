#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Unified environment installer: place GaussianAvatars + VHAP into a single
# conda env (default: `gaussian-avatars`).
#
# Both projects pin Python 3.11 + PyTorch 2.9.1 + CUDA 12.8 and use
# `--no-deps` everywhere, so a sequential install is safe.
#
# Install order: GA first, then VHAP --pip-only into the same env.
# When the two pin sets disagree the LAST writer (VHAP) wins, which is the
# right outcome for the demo flows under demo/:
#
#   - tyro     : VHAP 0.8.14 wins over GA 0.9.13.
#                The demo *uses* VHAP CLI scripts (track*.py /
#                preprocess_video.py / export_as_nerf_dataset.py), all of
#                which are tested with 0.8.14. GA's train.py and render.py
#                use argparse, NOT tyro, so the GA demo flow is unaffected.
#                Only GA's optional dev viewers (local_viewer.py /
#                remote_viewer.py) might be impacted; those are not part of
#                the demo. If you need them on 0.9.13:
#                  pip install --no-deps tyro==0.9.13
#
#   - dearpygui: VHAP 1.11.1 wins over GA 2.1.4. DPG 2.0 is a major-bump
#                with breaking ImPlot/ImNodes changes. Demos do NOT use any
#                viewer, so this only matters if you launch a viewer:
#                  - VHAP flame_viewer.py / flame_editor.py: need 1.x.
#                  - GA  local_viewer.py  / remote_viewer.py: basic widgets;
#                    works with 1.11.1 in practice.
#                Override with:
#                  pip install --no-deps dearpygui==2.1.4   (or 1.11.1)
#
#   - chumpy   : Both stacks install chumpy 0.71. GA pins to mattloper
#                @580566ea (numpy 2 friendly). The companion VHAP cuda128
#                branch may still install chumpy-fork==0.71 with an alias
#                hack on chumpy/__init__.py — that alias overwrites GA's
#                chumpy install, but `import chumpy; chumpy.Ch` still works
#                because chumpy_fork ships the same Ch class. Once VHAP is
#                migrated to mattloper@580566ea (handover doc circulated
#                separately), both installs are bit-identical.
#
# Usage:
#   bash demo/setup_env.sh                  # full install (no FLAME download)
#   bash demo/setup_env.sh --skip-ga        # skip GA setup (env already built)
#   bash demo/setup_env.sh --skip-vhap      # skip VHAP setup
#   DOWNLOAD_ASSETS=1 bash demo/setup_env.sh
#                                           # also run VHAP's download_assets.sh
#                                           # (interactive: prompts for FLAME
#                                           # username/password; populates only
#                                           # submodules/VHAP/asset/flame/, not
#                                           # the GA-side flame_model/assets/)
#   GA_ENV=my-env bash demo/setup_env.sh    # override conda env name
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SKIP_GA=0
SKIP_VHAP=0
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}"

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

if [[ ${SKIP_GA} -eq 0 ]]; then
  log "[1/2] Installing GaussianAvatars stack into env '${GA_ENV}'"
  cd "${REPO_ROOT}"
  # GA's setup.sh now auto-initialises submodules (including VHAP) when
  # they're missing, so a separate `git submodule update --init` is no
  # longer required here.
  bash setup.sh
fi

# require_vhap_submodule remains useful when --skip-ga is passed: it gives a
# clear error before the VHAP --pip-only step runs.
require_vhap_submodule

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
  vhap_args=( --pip-only )
  if [[ "${DOWNLOAD_ASSETS}" != "1" ]]; then
    # Default: skip VHAP's download_assets.sh because it prompts for FLAME
    # credentials interactively and only populates the VHAP-side asset
    # directory anyway. README §0.3 documents manual placement for both
    # projects (a single download + symlink covers both).
    vhap_args+=( --no-assets )
  fi
  ENV_NAME="${GA_ENV}" bash setup.sh "${vhap_args[@]}"
fi

log "Done. Both stacks share env '${GA_ENV}'."
log "Next steps:"
if [[ "${DOWNLOAD_ASSETS}" != "1" ]]; then
  log "  1. Place FLAME assets (see demo/README.md §0.3)"
fi
log "  2. bash demo/01_vhap_preprocess_monocular.sh   (or *_nersemble.sh)"
log "  3. bash demo/02_train.sh"
log "  4. bash demo/03_render.sh"
