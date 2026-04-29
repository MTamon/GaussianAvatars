#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo 1b: VHAP preprocessing for the multi-view NeRSemble dataset.
#
# Pipeline:
#   1. Frame extraction + foreground matting (Background Matting V2, multi-view).
#   2. FLAME face tracking across all 16 views.
#   3. Export as a NeRF-style dataset that GaussianAvatars/train.py can consume.
#
# Inputs (place before running):
#   submodules/VHAP/data/nersemble/${SUBJECT}/${SEQUENCE}*  # NeRSemble raw layout
#   submodules/VHAP/asset/flame/flame2023.pkl
#   submodules/VHAP/asset/flame/FLAME_masks.pkl
#
# Outputs:
#   submodules/VHAP/output/nersemble/<run>/             # raw tracking results
#   submodules/VHAP/export/nersemble/<run>/             # NeRF-style export
#
# Override anything via env vars, e.g.:
#   SUBJECT=306 SEQUENCE=EMO-1 bash demo/01_vhap_preprocess_nersemble.sh
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SUBJECT="${SUBJECT:-074}"
SEQUENCE="${SEQUENCE:-EMO-1}"
DOWNSAMPLE="${DOWNSAMPLE:-4}"
SUFFIX="${SUFFIX:-v16_DS${DOWNSAMPLE}_wBg_staticOffset}"
EXPORT_SUFFIX="${EXPORT_SUFFIX:-v16_DS${DOWNSAMPLE}_whiteBg_staticOffset_maskBelowLine}"

TRACK_OUTPUT_FOLDER="output/nersemble/${SUBJECT}_${SEQUENCE}_${SUFFIX}"
EXPORT_OUTPUT_FOLDER="export/nersemble/${SUBJECT}_${SEQUENCE}_${EXPORT_SUFFIX}"

require_vhap_submodule
activate_env "${VHAP_ENV}"

cd "${VHAP_DIR}"

INPUT_GLOB="data/nersemble/${SUBJECT}/${SEQUENCE}*"
if ! compgen -G "${INPUT_GLOB}" > /dev/null; then
  echo "[demo] no NeRSemble inputs match: ${VHAP_DIR}/${INPUT_GLOB}" >&2
  echo "       Place the raw NeRSemble subject directory there and re-run." >&2
  exit 1
fi

log "[1/3] Preprocess (frame extraction + matting): ${INPUT_GLOB}"
# shellcheck disable=SC2086  # we want glob expansion of INPUT_GLOB
python vhap/preprocess_video.py \
  --input ${INPUT_GLOB} \
  --downsample_scales 2 ${DOWNSAMPLE} \
  --matting_method background_matting_v2

log "[2/3] FLAME tracking (16 views) -> ${TRACK_OUTPUT_FOLDER}"
python vhap/track_nersemble.py \
  --data.root_folder "data/nersemble" \
  --exp.output_folder "${TRACK_OUTPUT_FOLDER}" \
  --data.subject "${SUBJECT}" \
  --data.sequence "${SEQUENCE}" \
  --data.n_downsample_rgb "${DOWNSAMPLE}"

log "[3/3] Export as NeRF-style dataset -> ${EXPORT_OUTPUT_FOLDER}"
python vhap/export_as_nerf_dataset.py \
  --src_folder "${TRACK_OUTPUT_FOLDER}" \
  --tgt_folder "${EXPORT_OUTPUT_FOLDER}" \
  --background-color white

log "Done. Pass this path to demo/02_train.sh:"
echo "    SOURCE_PATH=${VHAP_DIR}/${EXPORT_OUTPUT_FOLDER}"
