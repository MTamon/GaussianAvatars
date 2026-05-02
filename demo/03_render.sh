#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo 3: Render images and MP4s from a trained GaussianAvatars model.
#
# Streams the tqdm "Rendering progress" bar from render.py:70 in real time
# (PYTHONUNBUFFERED=1 set in _common.sh).
#
# Inputs:
#   --model-path        Absolute path to the trained model directory
#                       (the --model-path from demo/02_train.sh).
#   --iteration         Iteration to load (default: latest if -1).
#   --select-camera-id  Restrict rendering to a single camera id (e.g. 8 = front
#                       view in NeRSemble). Omit to render all cameras.
#   --target-path       (Optional) cross-identity reenactment: another VHAP-
#                       exported sequence whose FLAME motion will drive the
#                       trained avatar.
#
# Outputs:
#   ${MODEL_PATH}/{train,val,test}/ours_<iter>/{renders,gt}/*.png
#   ${MODEL_PATH}/{train,val,test}/ours_<iter>/{renders,gt}.mp4
#
# Examples:
#   bash demo/03_render.sh --model-path "$PWD/output/obama_whiteBg_staticOffset_maskBelowLine"
#
#   bash demo/03_render.sh \
#     --model-path "$PWD/output/074_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine" \
#     --select-camera-id 8
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_PATH=""
ITERATION="-1"
SELECT_CAMERA_ID=""
TARGET_PATH=""

usage() {
  cat <<'USAGE'
Usage:
  bash demo/03_render.sh --model-path PATH [options]

Options:
  --model-path PATH        Trained model directory containing cfg_args (required)
  --iteration N            Iteration to load; -1 means latest (default: -1)
  --select-camera-id ID    Render only one camera id
  --target-path PATH       VHAP export used as reenactment motion
  --env NAME               Conda env to activate (default: gaussian-avatars)
  -h, --help               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) require_option_value "$(basename "$0")" "$1" "$#"; MODEL_PATH="$2"; shift 2 ;;
    --iteration) require_option_value "$(basename "$0")" "$1" "$#"; ITERATION="$2"; shift 2 ;;
    --select-camera-id) require_option_value "$(basename "$0")" "$1" "$#"; SELECT_CAMERA_ID="$2"; shift 2 ;;
    --target-path) require_option_value "$(basename "$0")" "$1" "$#"; TARGET_PATH="$2"; shift 2 ;;
    --env) require_option_value "$(basename "$0")" "$1" "$#"; GA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die_usage "$(basename "$0")" "unknown option: $1" ;;
  esac
done

if [ -z "${MODEL_PATH}" ]; then
  die_usage "$(basename "$0")" "--model-path is required (path to a trained model directory)."
fi
if [ ! -f "${MODEL_PATH}/cfg_args" ]; then
  echo "[demo] ${MODEL_PATH} does not look like a trained model (no cfg_args)." >&2
  exit 1
fi

[[ -n "${ITERATION}" ]] || die_usage "$(basename "$0")" "--iteration requires a value"
[[ -n "${GA_ENV}" ]] || die_usage "$(basename "$0")" "--env requires a value"

activate_env
cd "${REPO_ROOT}"

RENDER_ARGS=( -m "${MODEL_PATH}" --iteration "${ITERATION}" )
if [ -n "${SELECT_CAMERA_ID:-}" ]; then
  RENDER_ARGS+=( --select_camera_id "${SELECT_CAMERA_ID}" )
fi
if [ -n "${TARGET_PATH:-}" ]; then
  if [ ! -f "${TARGET_PATH}/transforms_train.json" ]; then
    echo "[demo] --target-path '${TARGET_PATH}' is not a VHAP export." >&2
    exit 1
  fi
  RENDER_ARGS+=( -t "${TARGET_PATH}" )
fi

log "Rendering trained avatar"
log "  model  : ${MODEL_PATH}"
log "  iter   : ${ITERATION}"
[ -n "${SELECT_CAMERA_ID:-}" ] && log "  camera : ${SELECT_CAMERA_ID}"
[ -n "${TARGET_PATH:-}" ]      && log "  target : ${TARGET_PATH}"

${PYTHON} render.py "${RENDER_ARGS[@]}"

log "Rendering finished. PNG sequences and MP4s are under:"
echo "    ${MODEL_PATH}/{train,val,test}/ours_*/"
