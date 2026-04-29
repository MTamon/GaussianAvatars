#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo 1a: VHAP preprocessing for a MONOCULAR video.
#
# Pipeline:
#   1. Frame extraction + foreground matting (Robust Video Matting).
#   2. FLAME face tracking (sequential init + 30 epochs of global optimisation).
#   3. Export as a NeRF-style dataset (transforms_*.json + images + FLAME params)
#      that GaussianAvatars/train.py can consume directly.
#
# All three stages stream tqdm progress bars to stderr (PYTHONUNBUFFERED=1
# is set in _common.sh so the bars render live).
#
# Inputs (place before running):
#   submodules/VHAP/data/monocular/${SEQUENCE_FILE}     # e.g. obama.mp4
#   submodules/VHAP/asset/flame/flame2023.pkl
#   submodules/VHAP/asset/flame/FLAME_masks.pkl
#
# Outputs:
#   submodules/VHAP/output/monocular/<run>/...          # raw tracking results
#   submodules/VHAP/export/monocular/<run>/             # NeRF-style export
#       └── transforms_train.json / transforms_val.json / transforms_test.json
#
# Override anything via env vars, e.g.:
#   SEQUENCE_FILE=alice.mp4 SEQUENCE=alice bash demo/01_vhap_preprocess_monocular.sh
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SEQUENCE_FILE="${SEQUENCE_FILE:-obama.mp4}"
SEQUENCE="${SEQUENCE:-${SEQUENCE_FILE%.*}}"
SUFFIX="${SUFFIX:-whiteBg_staticOffset}"
EXPORT_SUFFIX="${EXPORT_SUFFIX:-whiteBg_staticOffset_maskBelowLine}"

TRACK_OUTPUT_FOLDER="output/monocular/${SEQUENCE}_${SUFFIX}"
EXPORT_OUTPUT_FOLDER="export/monocular/${SEQUENCE}_${EXPORT_SUFFIX}"

require_vhap_submodule
activate_env

cd "${VHAP_DIR}"

INPUT_VIDEO="data/monocular/${SEQUENCE_FILE}"
if [ ! -e "${INPUT_VIDEO}" ]; then
  echo "[demo] input video not found: ${VHAP_DIR}/${INPUT_VIDEO}" >&2
  echo "       Place a monocular video there and re-run." >&2
  exit 1
fi

log "[1/3] Preprocess (frame extraction + matting): ${INPUT_VIDEO}"
${PYTHON} vhap/preprocess_video.py \
  --input "${INPUT_VIDEO}" \
  --matting_method robust_video_matting

log "[2/3] FLAME tracking -> ${TRACK_OUTPUT_FOLDER}"
${PYTHON} vhap/track.py \
  --data.root_folder "data/monocular" \
  --exp.output_folder "${TRACK_OUTPUT_FOLDER}" \
  --data.sequence "${SEQUENCE}"

log "[3/3] Export as NeRF-style dataset -> ${EXPORT_OUTPUT_FOLDER}"
${PYTHON} vhap/export_as_nerf_dataset.py \
  --src_folder "${TRACK_OUTPUT_FOLDER}" \
  --tgt_folder "${EXPORT_OUTPUT_FOLDER}" \
  --background-color white

log "Done. Pass this path to demo/02_train.sh:"
echo "    SOURCE_PATH=${VHAP_DIR}/${EXPORT_OUTPUT_FOLDER}"
