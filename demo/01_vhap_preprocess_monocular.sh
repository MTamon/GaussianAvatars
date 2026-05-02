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
# Usage:
#   bash demo/01_vhap_preprocess_monocular.sh --sequence-file alice.mp4 --sequence alice
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SEQUENCE_FILE="obama.mp4"
SEQUENCE=""
SUFFIX="whiteBg_staticOffset"
EXPORT_SUFFIX="whiteBg_staticOffset_maskBelowLine"

usage() {
  cat <<'USAGE'
Usage:
  bash demo/01_vhap_preprocess_monocular.sh [options]

Options:
  --sequence-file PATH   Input video under submodules/VHAP/data/monocular (default: obama.mp4)
  --sequence NAME        Sequence name used by VHAP/output folders (default: file stem)
  --suffix NAME          Tracking output suffix (default: whiteBg_staticOffset)
  --export-suffix NAME   Export output suffix (default: whiteBg_staticOffset_maskBelowLine)
  --env NAME             Conda env to activate (default: gaussian-avatars)
  -h, --help             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequence-file) require_option_value "$(basename "$0")" "$1" "$#"; SEQUENCE_FILE="$2"; shift 2 ;;
    --sequence) require_option_value "$(basename "$0")" "$1" "$#"; SEQUENCE="$2"; shift 2 ;;
    --suffix) require_option_value "$(basename "$0")" "$1" "$#"; SUFFIX="$2"; shift 2 ;;
    --export-suffix) require_option_value "$(basename "$0")" "$1" "$#"; EXPORT_SUFFIX="$2"; shift 2 ;;
    --env) require_option_value "$(basename "$0")" "$1" "$#"; GA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die_usage "$(basename "$0")" "unknown option: $1" ;;
  esac
done

[[ -n "${SEQUENCE_FILE}" ]] || die_usage "$(basename "$0")" "--sequence-file requires a value"
[[ -n "${SUFFIX}" ]] || die_usage "$(basename "$0")" "--suffix requires a value"
[[ -n "${EXPORT_SUFFIX}" ]] || die_usage "$(basename "$0")" "--export-suffix requires a value"
[[ -n "${GA_ENV}" ]] || die_usage "$(basename "$0")" "--env requires a value"
SEQUENCE="${SEQUENCE:-${SEQUENCE_FILE%.*}}"

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
echo "    bash demo/02_train.sh --source-path \"${VHAP_DIR}/${EXPORT_OUTPUT_FOLDER}\""
