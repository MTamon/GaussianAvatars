#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo 1b: VHAP preprocessing for the multi-view NeRSemble dataset.
#
# Pipeline:
#   1. Frame extraction + foreground matting (Background Matting V2, multi-view).
#   2. FLAME face tracking across all 16 views.
#   3. Export as a NeRF-style dataset that GaussianAvatars/train.py can consume.
#
# All three stages stream tqdm progress bars to stderr.
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
# Usage:
#   bash demo/01_vhap_preprocess_nersemble.sh --subject 306 --sequence EMO-1
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SUBJECT="074"
SEQUENCE="EMO-1"
DOWNSAMPLE="4"
SUFFIX=""
EXPORT_SUFFIX=""

usage() {
  cat <<'USAGE'
Usage:
  bash demo/01_vhap_preprocess_nersemble.sh [options]

Options:
  --subject ID           NeRSemble subject id (default: 074)
  --sequence NAME        NeRSemble sequence name (default: EMO-1)
  --downsample N         Downsample factor passed to VHAP (default: 4)
  --suffix NAME          Tracking output suffix (default: v16_DS<N>_wBg_staticOffset)
  --export-suffix NAME   Export output suffix (default: v16_DS<N>_whiteBg_staticOffset_maskBelowLine)
  --env NAME             Conda env to activate (default: gaussian-avatars)
  -h, --help             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) require_option_value "$(basename "$0")" "$1" "$#"; SUBJECT="$2"; shift 2 ;;
    --sequence) require_option_value "$(basename "$0")" "$1" "$#"; SEQUENCE="$2"; shift 2 ;;
    --downsample) require_option_value "$(basename "$0")" "$1" "$#"; DOWNSAMPLE="$2"; shift 2 ;;
    --suffix) require_option_value "$(basename "$0")" "$1" "$#"; SUFFIX="$2"; shift 2 ;;
    --export-suffix) require_option_value "$(basename "$0")" "$1" "$#"; EXPORT_SUFFIX="$2"; shift 2 ;;
    --env) require_option_value "$(basename "$0")" "$1" "$#"; GA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die_usage "$(basename "$0")" "unknown option: $1" ;;
  esac
done

[[ -n "${SUBJECT}" ]] || die_usage "$(basename "$0")" "--subject requires a value"
[[ -n "${SEQUENCE}" ]] || die_usage "$(basename "$0")" "--sequence requires a value"
[[ -n "${DOWNSAMPLE}" ]] || die_usage "$(basename "$0")" "--downsample requires a value"
[[ -n "${GA_ENV}" ]] || die_usage "$(basename "$0")" "--env requires a value"
SUFFIX="${SUFFIX:-v16_DS${DOWNSAMPLE}_wBg_staticOffset}"
EXPORT_SUFFIX="${EXPORT_SUFFIX:-v16_DS${DOWNSAMPLE}_whiteBg_staticOffset_maskBelowLine}"

TRACK_OUTPUT_FOLDER="output/nersemble/${SUBJECT}_${SEQUENCE}_${SUFFIX}"
EXPORT_OUTPUT_FOLDER="export/nersemble/${SUBJECT}_${SEQUENCE}_${EXPORT_SUFFIX}"

require_vhap_submodule
activate_env

cd "${VHAP_DIR}"

INPUT_GLOB="data/nersemble/${SUBJECT}/${SEQUENCE}*"
if ! compgen -G "${INPUT_GLOB}" > /dev/null; then
  echo "[demo] no NeRSemble inputs match: ${VHAP_DIR}/${INPUT_GLOB}" >&2
  echo "       Place the raw NeRSemble subject directory there and re-run." >&2
  exit 1
fi

log "[1/3] Preprocess (frame extraction + matting): ${INPUT_GLOB}"
# shellcheck disable=SC2086  # we want glob expansion of INPUT_GLOB
${PYTHON} vhap/preprocess_video.py \
  --input ${INPUT_GLOB} \
  --downsample_scales 2 ${DOWNSAMPLE} \
  --matting_method background_matting_v2

log "[2/3] FLAME tracking (16 views) -> ${TRACK_OUTPUT_FOLDER}"
${PYTHON} vhap/track_nersemble.py \
  --data.root_folder "data/nersemble" \
  --exp.output_folder "${TRACK_OUTPUT_FOLDER}" \
  --data.subject "${SUBJECT}" \
  --data.sequence "${SEQUENCE}" \
  --data.n_downsample_rgb "${DOWNSAMPLE}"

log "[3/3] Export as NeRF-style dataset -> ${EXPORT_OUTPUT_FOLDER}"
${PYTHON} vhap/export_as_nerf_dataset.py \
  --src_folder "${TRACK_OUTPUT_FOLDER}" \
  --tgt_folder "${EXPORT_OUTPUT_FOLDER}" \
  --background-color white

log "Done. Pass this path to demo/02_train.sh:"
echo "    bash demo/02_train.sh --source-path \"${VHAP_DIR}/${EXPORT_OUTPUT_FOLDER}\""
