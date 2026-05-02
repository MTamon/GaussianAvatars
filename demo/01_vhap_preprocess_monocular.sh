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
BATCH_SIZE="16"
LANDMARK_NJOBS="1"
SKIP_EXISTING=1

usage() {
  cat <<'USAGE'
Usage:
  bash demo/01_vhap_preprocess_monocular.sh [options]

Options:
  --sequence-file PATH   Input video under submodules/VHAP/data/monocular (default: obama.mp4)
  --sequence NAME        Sequence name used by VHAP/output folders (default: file stem)
  --suffix NAME          Tracking output suffix (default: whiteBg_staticOffset)
  --export-suffix NAME   Export output suffix (default: whiteBg_staticOffset_maskBelowLine)
  --batch-size N         VHAP tracking frame batch size (default: 16).
                         Use 1 for the original conservative tracking behavior.
  --landmark-njobs N     Forwarded as --data.landmark-detector-njobs to
                         vhap/track.py (default: 1). VHAP's upstream default
                         is 8, but joblib forks worker processes after the
                         parent has already initialised a CUDA context, which
                         on PyTorch+CUDA tends to silently hang or contend
                         over GPU0. Keep at 1 unless you know your environment
                         tolerates fork after CUDA init.
  --no-skip-existing     Re-run all stages even if completed outputs already exist.
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
    --batch-size) require_option_value "$(basename "$0")" "$1" "$#"; BATCH_SIZE="$2"; shift 2 ;;
    --landmark-njobs) require_option_value "$(basename "$0")" "$1" "$#"; LANDMARK_NJOBS="$2"; shift 2 ;;
    --no-skip-existing) SKIP_EXISTING=0; shift ;;
    --env) require_option_value "$(basename "$0")" "$1" "$#"; GA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die_usage "$(basename "$0")" "unknown option: $1" ;;
  esac
done

[[ -n "${SEQUENCE_FILE}" ]] || die_usage "$(basename "$0")" "--sequence-file requires a value"
[[ -n "${SUFFIX}" ]] || die_usage "$(basename "$0")" "--suffix requires a value"
[[ -n "${EXPORT_SUFFIX}" ]] || die_usage "$(basename "$0")" "--export-suffix requires a value"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die_usage "$(basename "$0")" "--batch-size must be a positive integer"
[[ "${LANDMARK_NJOBS}" =~ ^[1-9][0-9]*$ ]] || die_usage "$(basename "$0")" "--landmark-njobs must be a positive integer"
[[ -n "${GA_ENV}" ]] || die_usage "$(basename "$0")" "--env requires a value"
SEQUENCE="${SEQUENCE:-${SEQUENCE_FILE%.*}}"

TRACK_REUSE_ARGS=()
if (( SKIP_EXISTING == 0 )); then
  TRACK_REUSE_ARGS+=(--exp.no-reuse-landmarks)
fi

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

has_matching_file() {
  local dir="$1"
  local pattern="$2"
  find "${dir}" -maxdepth 1 -type f -name "${pattern}" -print -quit 2>/dev/null | grep -q .
}

preprocess_complete() {
  local sequence_dir="${INPUT_VIDEO%.*}"
  has_matching_file "${sequence_dir}/images" "*.jpg" &&
    has_matching_file "${sequence_dir}/alpha_maps" "*.jpg"
}

latest_track_npz() {
  [ -d "${TRACK_OUTPUT_FOLDER}" ] || return 0
  find "${TRACK_OUTPUT_FOLDER}" -mindepth 2 -maxdepth 2 -type f -name "tracked_flame_params*.npz" 2>/dev/null | sort -V | tail -n 1
}

track_complete() {
  [ -n "$(latest_track_npz)" ]
}

export_complete() {
  [ -f "${EXPORT_OUTPUT_FOLDER}/transforms_train.json" ] &&
    [ -f "${EXPORT_OUTPUT_FOLDER}/transforms_val.json" ] &&
    [ -f "${EXPORT_OUTPUT_FOLDER}/transforms_test.json" ] &&
    [ -f "${EXPORT_OUTPUT_FOLDER}/canonical_flame_param.npz" ]
}

if (( SKIP_EXISTING == 1 )) && preprocess_complete; then
  log "[1/3] Skip preprocess; frames and alpha maps already exist for ${INPUT_VIDEO}"
else
  log "[1/3] Preprocess (frame extraction + matting): ${INPUT_VIDEO}"
  ${PYTHON} vhap/preprocess_video.py \
    --input "${INPUT_VIDEO}" \
    --matting_method robust_video_matting
fi

if (( SKIP_EXISTING == 1 )) && track_complete; then
  log "[2/3] Skip FLAME tracking; found $(latest_track_npz)"
else
  log "[2/3] FLAME tracking -> ${TRACK_OUTPUT_FOLDER}"
  ${PYTHON} vhap/track.py \
    --data.root_folder "data/monocular" \
    --exp.output_folder "${TRACK_OUTPUT_FOLDER}" \
    --data.sequence "${SEQUENCE}" \
    --data.landmark-detector-njobs "${LANDMARK_NJOBS}" \
    --batch-size "${BATCH_SIZE}" \
    "${TRACK_REUSE_ARGS[@]}"
fi

if (( SKIP_EXISTING == 1 )) && export_complete; then
  log "[3/3] Skip export; NeRF-style dataset already exists at ${EXPORT_OUTPUT_FOLDER}"
else
  log "[3/3] Export as NeRF-style dataset -> ${EXPORT_OUTPUT_FOLDER}"
  ${PYTHON} vhap/export_as_nerf_dataset.py \
    --src_folder "${TRACK_OUTPUT_FOLDER}" \
    --tgt_folder "${EXPORT_OUTPUT_FOLDER}" \
    --background-color white
fi

log "Done. Pass this path to demo/02_train.sh:"
echo "    bash demo/02_train.sh --source-path \"${VHAP_DIR}/${EXPORT_OUTPUT_FOLDER}\""
