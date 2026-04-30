#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Unified environment installer: place GaussianAvatars + VHAP into a single
# conda env (default: `gaussian-avatars`) AND fetch the credential-gated
# FLAME assets needed by both projects.
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
# FLAME assets: prompted ONCE for FLAME username/password, then handed to
# GA's download_assets.sh (which writes flame_model/assets/flame/*.pkl).
# The VHAP-side paths under submodules/VHAP/asset/flame/ are populated by
# symlinking the GA copies — no second round-trip to the FLAME server.
# Set SKIP_DOWNLOAD_ASSETS=1 to opt out (you'll have to place the .pkl
# files yourself, see demo/README.md §0.3).
#
# Usage:
#   bash demo/setup_env.sh                          # full install + assets
#   bash demo/setup_env.sh --skip-ga                # skip GA setup
#   bash demo/setup_env.sh --skip-vhap              # skip VHAP setup
#   SKIP_DOWNLOAD_ASSETS=1 bash demo/setup_env.sh   # do not fetch FLAME
#   FLAME_USER=u FLAME_PASS=p bash demo/setup_env.sh   # non-interactive
#   GA_ENV=my-env bash demo/setup_env.sh            # override conda env
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SKIP_GA=0
SKIP_VHAP=0
SKIP_DOWNLOAD_ASSETS="${SKIP_DOWNLOAD_ASSETS:-0}"
FLAME_USER="${FLAME_USER:-}"
FLAME_PASS="${FLAME_PASS:-}"

for arg in "$@"; do
  case "$arg" in
    --skip-ga)              SKIP_GA=1 ;;
    --skip-vhap)            SKIP_VHAP=1 ;;
    --skip-download-assets) SKIP_DOWNLOAD_ASSETS=1 ;;
    -h|--help)
      sed -n '1,/^# ---/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "[setup_env.sh] unknown flag: $arg" >&2; exit 2 ;;
  esac
done

prompt_flame_credentials() {
  # Resolve credentials before any heavy work so we can fail fast and avoid
  # leaving the user staring at a blocked prompt mid-install.
  if [[ "${SKIP_DOWNLOAD_ASSETS}" == "1" ]]; then
    return 0
  fi

  local ga_pkl="${REPO_ROOT}/flame_model/assets/flame/flame2023.pkl"
  local ga_msk="${REPO_ROOT}/flame_model/assets/flame/FLAME_masks.pkl"
  local vh_pkl="${VHAP_DIR}/asset/flame/flame2023.pkl"
  local vh_msk="${VHAP_DIR}/asset/flame/FLAME_masks.pkl"

  if [[ -f "${ga_pkl}" && -f "${ga_msk}" && -e "${vh_pkl}" && -e "${vh_msk}" ]]; then
    log "FLAME assets already in place; skipping credential prompt."
    SKIP_DOWNLOAD_ASSETS=1
    return 0
  fi

  if [[ -z "${FLAME_USER}" ]]; then
    if [[ ! -t 0 ]]; then
      echo "[setup_env.sh] FLAME credentials needed but stdin is not a TTY." >&2
      echo "  Re-run with FLAME_USER=... FLAME_PASS=... or SKIP_DOWNLOAD_ASSETS=1." >&2
      exit 1
    fi
    read -r  -p "Username (FLAME): " FLAME_USER
  fi
  if [[ -z "${FLAME_PASS}" ]]; then
    if [[ ! -t 0 ]]; then
      echo "[setup_env.sh] FLAME password not provided and stdin is not a TTY." >&2
      exit 1
    fi
    read -r -s -p "Password (FLAME): " FLAME_PASS
    echo
  fi
}

# Prompt up-front so the long pip install stage can run unattended after
# credentials have been collected.
prompt_flame_credentials

if [[ ${SKIP_GA} -eq 0 ]]; then
  log "[1/3] Installing GaussianAvatars stack into env '${GA_ENV}'"
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
  log "[2/3] Adding VHAP stack into the same env (--pip-only)"
  cd "${VHAP_DIR}"
  # Always pass --no-assets here; we fetch FLAME assets centrally in step
  # [3/3] and symlink into the VHAP tree, avoiding a duplicate FLAME-server
  # round-trip.
  ENV_NAME="${GA_ENV}" bash setup.sh --pip-only --no-assets
fi

if [[ "${SKIP_DOWNLOAD_ASSETS}" != "1" ]]; then
  log "[3/3] Fetching FLAME assets (single credential set covers both projects)"
  cd "${REPO_ROOT}"
  bash download_assets.sh \
    --flame_user "${FLAME_USER}" \
    --flame_pass "${FLAME_PASS}"

  # Mirror the GA-side .pkl files into the VHAP asset/flame directory.
  # We use symlinks rather than copies so disk usage stays single-counted
  # and any future re-download via download_assets.sh stays in sync.
  log "    Linking VHAP asset/flame/* to the GA copies"
  mkdir -p "${VHAP_DIR}/asset/flame"
  for f in flame2023.pkl FLAME_masks.pkl; do
    src="${REPO_ROOT}/flame_model/assets/flame/${f}"
    dst="${VHAP_DIR}/asset/flame/${f}"
    if [[ -f "${src}" ]]; then
      ln -sf "${src}" "${dst}"
      echo "    ${dst} -> ${src}"
    else
      echo "    [WARN] ${src} missing; cannot link ${dst}" >&2
    fi
  done
else
  log "[3/3] SKIP_DOWNLOAD_ASSETS=1 — skipping FLAME asset fetch."
fi

log "Done. Both stacks share env '${GA_ENV}'."
log "Next steps:"
if [[ "${SKIP_DOWNLOAD_ASSETS}" == "1" ]]; then
  log "  0. Place FLAME assets manually (see demo/README.md §0.3)"
fi
log "  1. bash demo/01_vhap_preprocess_monocular.sh   (or *_nersemble.sh)"
log "  2. bash demo/02_train.sh"
log "  3. bash demo/03_render.sh"
