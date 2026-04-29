#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo 3: Render images and MP4s from a trained GaussianAvatars model.
#
# Inputs:
#   MODEL_PATH        Absolute path to the trained model directory
#                     (the ${MODEL_PATH} from demo/02_train.sh).
#   ITERATION         Iteration to load (default: latest if -1).
#   SELECT_CAMERA_ID  Restrict rendering to a single camera id (e.g. 8 = front
#                     view in NeRSemble). Empty = render all cameras.
#   TARGET_PATH       (Optional) cross-identity reenactment: another VHAP-
#                     exported sequence whose FLAME motion will drive the
#                     trained avatar.
#
# Outputs:
#   ${MODEL_PATH}/{train,val,test}/ours_<iter>/{renders,gt}/*.png
#   ${MODEL_PATH}/{train,val,test}/ours_<iter>/{renders,gt}.mp4
#
# Examples:
#   MODEL_PATH=$PWD/output/obama_whiteBg_staticOffset_maskBelowLine bash demo/03_render.sh
#
#   MODEL_PATH=$PWD/output/074_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine \
#     SELECT_CAMERA_ID=8 \
#     bash demo/03_render.sh
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ -z "${MODEL_PATH:-}" ]; then
  echo "[demo] MODEL_PATH is required (path to a trained model directory)." >&2
  exit 2
fi
if [ ! -f "${MODEL_PATH}/cfg_args" ]; then
  echo "[demo] ${MODEL_PATH} does not look like a trained model (no cfg_args)." >&2
  exit 1
fi

ITERATION="${ITERATION:--1}"

activate_env "${GA_ENV}"
cd "${REPO_ROOT}"

RENDER_ARGS=( -m "${MODEL_PATH}" --iteration "${ITERATION}" )
if [ -n "${SELECT_CAMERA_ID:-}" ]; then
  RENDER_ARGS+=( --select_camera_id "${SELECT_CAMERA_ID}" )
fi
if [ -n "${TARGET_PATH:-}" ]; then
  if [ ! -f "${TARGET_PATH}/transforms_train.json" ]; then
    echo "[demo] TARGET_PATH=${TARGET_PATH} is not a VHAP export." >&2
    exit 1
  fi
  RENDER_ARGS+=( -t "${TARGET_PATH}" )
fi

log "Rendering trained avatar"
log "  model  : ${MODEL_PATH}"
log "  iter   : ${ITERATION}"
[ -n "${SELECT_CAMERA_ID:-}" ] && log "  camera : ${SELECT_CAMERA_ID}"
[ -n "${TARGET_PATH:-}" ]      && log "  target : ${TARGET_PATH}"

python render.py "${RENDER_ARGS[@]}"

log "Rendering finished. PNG sequences and MP4s are under:"
echo "    ${MODEL_PATH}/{train,val,test}/ours_*/"
