#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo 2: Train a GaussianAvatars head avatar.
#
# Inputs:
#   SOURCE_PATH    Absolute path to a VHAP-exported NeRF-style dataset folder
#                  (the ${EXPORT_OUTPUT_FOLDER} printed by demo 01_*).
#                  Must contain transforms_train.json (+ val/test for --eval).
#   MODEL_PATH     Where to write the trained model (defaults under output/).
#   ITERATIONS     Total training iterations (default: 600000 to match the
#                  paper's 600k schedule; lower for a quick demo).
#   PORT           GUI server port for the optional remote viewer.
#
# Outputs:
#   ${MODEL_PATH}/point_cloud/iteration_*/point_cloud.ply
#   ${MODEL_PATH}/cameras.json, cfg_args, ...
#
# Examples:
#   SOURCE_PATH=$PWD/submodules/VHAP/export/monocular/obama_whiteBg_staticOffset_maskBelowLine \
#     bash demo/02_train.sh
#
#   SOURCE_PATH=$PWD/submodules/VHAP/export/nersemble/074_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine \
#     ITERATIONS=30000 \
#     bash demo/02_train.sh
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ -z "${SOURCE_PATH:-}" ]; then
  echo "[demo] SOURCE_PATH is required (path to a VHAP NeRF export)." >&2
  echo "       e.g. SOURCE_PATH=\$PWD/submodules/VHAP/export/monocular/obama_whiteBg_staticOffset_maskBelowLine bash demo/02_train.sh" >&2
  exit 2
fi

if [ ! -f "${SOURCE_PATH}/transforms_train.json" ]; then
  echo "[demo] ${SOURCE_PATH} does not look like a VHAP export (no transforms_train.json)." >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-$(basename "${SOURCE_PATH}")}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/output/${RUN_NAME}}"
ITERATIONS="${ITERATIONS:-600000}"
PORT="${PORT:-60000}"

activate_env "${GA_ENV}"
cd "${REPO_ROOT}"

mkdir -p "$(dirname "${MODEL_PATH}")"

log "Training GaussianAvatars"
log "  source : ${SOURCE_PATH}"
log "  model  : ${MODEL_PATH}"
log "  iters  : ${ITERATIONS}"

python train.py \
  -s "${SOURCE_PATH}" \
  -m "${MODEL_PATH}" \
  --eval \
  --bind_to_mesh \
  --white_background \
  --iterations "${ITERATIONS}" \
  --port "${PORT}"

log "Training finished. Pass MODEL_PATH=${MODEL_PATH} to demo/03_render.sh."
