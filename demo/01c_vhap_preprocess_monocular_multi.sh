#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo 1c: VHAP preprocessing for MULTIPLE monocular videos of the same subject
#          captured under matching conditions (same hairstyle, same lighting,
#          same camera, same background) at different times.
#
# Pipeline:
#   Phase A (shape source clip):
#     1. Frame extraction + matting
#     2. FLAME tracking with all parameters optimised
#        -> produces tracked_flame_params_<epoch>.npz with the canonical
#           subject identity (shape, lights, focal_length, tex_*, static_offset)
#     3. Export as NeRF-style dataset
#
#   Phase B (every other clip):
#     1. Frame extraction + matting
#     2. FLAME tracking with --load-globals-only --freeze-globals-from-init
#        and --model.flame-params-path pointing at Phase A's npz, so per-clip
#        rotation / translation / pose / expression are optimised against the
#        same global identity / texture / lighting / focal length.
#     3. Export as NeRF-style dataset
#
#   Phase C (combine):
#     vhap/combine_nerf_datasets.py merges all per-clip exports into a single
#     dataset under submodules/VHAP/export/monocular/<subject>_UNION<N>_... .
#     The combined transforms_{train,val,test}.json is what GaussianAvatars
#     train.py consumes directly.
#
# Inputs (place before running):
#   <sequence_file>                                     # one video path per --sequence-file
#   submodules/VHAP/asset/flame/flame2023.pkl
#   submodules/VHAP/asset/flame/FLAME_masks.pkl
#
# Usage example:
#   bash demo/01c_vhap_preprocess_monocular_multi.sh \
#     --subject subj01 \
#     --shape-source-clip take1 \
#     --sequence-file take1.mp4 \
#     --sequence-file take2.mp4 \
#     --sequence-file take3.mp4 \
#     --sequence-file take4.mp4 \
#     --sequence-file take5.mp4 \
#     --sequence-file take6.mp4
#
# The frozen globals across Phase B clips assume:
#   - same person (shape)
#   - same hairstyle / collar (static_offset)
#   - same lighting environment (lights / SH coefficients)
#   - same recording device, no zoom change between takes (focal_length, tex_*)
# If any of these differ across takes, fall back to running each clip with
# 01_vhap_preprocess_monocular.sh and combining manually with
# vhap/combine_nerf_datasets.py.
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SUBJECT=""
SHAPE_SOURCE_CLIP=""
SEQUENCE_FILES=()
SUFFIX="whiteBg_staticOffset"
EXPORT_SUFFIX="whiteBg_staticOffset_maskBelowLine"
DIVISION_MODE="last"
BATCH_SIZE="16"
SKIP_EXISTING=1

usage() {
  cat <<'USAGE'
Usage:
  bash demo/01c_vhap_preprocess_monocular_multi.sh [options]

Required:
  --subject NAME             Subject identifier used as the common prefix for
                             every per-clip output folder (e.g. subj01).
                             combine_nerf_datasets.py asserts that all source
                             folders share the same first '_'-separated token,
                             so this token must be unique per subject.
  --shape-source-clip STEM   Stem (with or without extension) of the clip whose
                             tracking run produces the canonical FLAME globals
                             (shape, lights, tex_*, static_offset, focal_length).
                             Must match one of the --sequence-file entries.
                             Pick the longest / most-frontal / best-lit take.
  --sequence-file PATH       Input video path.
                             Relative paths are resolved from the directory
                             where this script is launched. A bare filename
                             still falls back to submodules/VHAP/data/monocular
                             for backward compatibility.
                             Repeat the flag once per take.

Options:
  --suffix NAME              Tracking output suffix (default: whiteBg_staticOffset)
  --export-suffix NAME       Per-clip export suffix
                             (default: whiteBg_staticOffset_maskBelowLine)
  --division-mode MODE       Passed to combine_nerf_datasets.py.
                             One of: random_single, random_group, last
                             (default: last)
  --batch-size N             VHAP tracking frame batch size (default: 16).
                             Use 1 for the original conservative tracking behavior.
  --no-skip-existing         Re-run all stages even if completed outputs already exist.
  --env NAME                 Conda env to activate (default: gaussian-avatars)
  -h, --help                 Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) require_option_value "$(basename "$0")" "$1" "$#"; SUBJECT="$2"; shift 2 ;;
    --shape-source-clip) require_option_value "$(basename "$0")" "$1" "$#"; SHAPE_SOURCE_CLIP="$2"; shift 2 ;;
    --sequence-file) require_option_value "$(basename "$0")" "$1" "$#"; SEQUENCE_FILES+=("$2"); shift 2 ;;
    --suffix) require_option_value "$(basename "$0")" "$1" "$#"; SUFFIX="$2"; shift 2 ;;
    --export-suffix) require_option_value "$(basename "$0")" "$1" "$#"; EXPORT_SUFFIX="$2"; shift 2 ;;
    --division-mode) require_option_value "$(basename "$0")" "$1" "$#"; DIVISION_MODE="$2"; shift 2 ;;
    --batch-size) require_option_value "$(basename "$0")" "$1" "$#"; BATCH_SIZE="$2"; shift 2 ;;
    --no-skip-existing) SKIP_EXISTING=0; shift ;;
    --env) require_option_value "$(basename "$0")" "$1" "$#"; GA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die_usage "$(basename "$0")" "unknown option: $1" ;;
  esac
done

[[ -n "${SUBJECT}" ]] || die_usage "$(basename "$0")" "--subject is required"
[[ -n "${SHAPE_SOURCE_CLIP}" ]] || die_usage "$(basename "$0")" "--shape-source-clip is required"
(( ${#SEQUENCE_FILES[@]} >= 1 )) || die_usage "$(basename "$0")" "at least one --sequence-file is required"
[[ -n "${SUFFIX}" ]] || die_usage "$(basename "$0")" "--suffix requires a value"
[[ -n "${EXPORT_SUFFIX}" ]] || die_usage "$(basename "$0")" "--export-suffix requires a value"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die_usage "$(basename "$0")" "--batch-size must be a positive integer"
[[ -n "${GA_ENV}" ]] || die_usage "$(basename "$0")" "--env requires a value"

case "${DIVISION_MODE}" in
  random_single|random_group|last) ;;
  *) die_usage "$(basename "$0")" "--division-mode must be one of: random_single, random_group, last" ;;
esac

# Subject names with underscores would collide with combine_nerf_datasets.py's
# subject prefix assertion (it splits on '_' and takes [0]).
if [[ "${SUBJECT}" == *"_"* ]]; then
  die_usage "$(basename "$0")" "--subject must not contain '_' (combine_nerf_datasets.py uses '_' as the subject delimiter)"
fi

# Strip extension from --shape-source-clip and validate it matches one of the
# --sequence-file stems.
SHAPE_SOURCE_STEM="$(basename "${SHAPE_SOURCE_CLIP}")"
SHAPE_SOURCE_STEM="${SHAPE_SOURCE_STEM%.*}"

declare -a CLIP_STEMS=()
SHAPE_SOURCE_FOUND=0
for sf in "${SEQUENCE_FILES[@]}"; do
  stem="$(basename "${sf}")"
  stem="${stem%.*}"
  CLIP_STEMS+=("${stem}")
  if [[ "${stem}" == "${SHAPE_SOURCE_STEM}" ]]; then
    SHAPE_SOURCE_FOUND=1
  fi
done

if (( SHAPE_SOURCE_FOUND == 0 )); then
  die_usage "$(basename "$0")" "--shape-source-clip '${SHAPE_SOURCE_CLIP}' does not match any --sequence-file stem (got: ${CLIP_STEMS[*]})"
fi

require_vhap_submodule
activate_env

LAUNCH_DIR="$(pwd -P)"

canonical_existing_path() {
  local path="$1"
  local dir
  dir="$(cd "$(dirname "${path}")" && pwd -P)"
  echo "${dir}/$(basename "${path}")"
}

resolve_sequence_file() {
  local arg="$1"
  local launch_candidate
  local legacy_candidate

  if [[ "${arg}" == /* ]]; then
    launch_candidate="${arg}"
  else
    launch_candidate="${LAUNCH_DIR}/${arg}"
  fi

  if [ -e "${launch_candidate}" ]; then
    canonical_existing_path "${launch_candidate}"
    return
  fi

  legacy_candidate="${VHAP_DIR}/data/monocular/${arg}"
  if [ -e "${legacy_candidate}" ]; then
    canonical_existing_path "${legacy_candidate}"
    return
  fi

  echo "[demo] input video not found: ${launch_candidate}" >&2
  if [[ "${arg}" != /* ]]; then
    echo "       Also checked legacy VHAP path: ${legacy_candidate}" >&2
  fi
  echo "       Pass an existing video path, for example: --sequence-file data/src/example.mp4" >&2
  exit 1
}

# Validate every input file exists before we start any expensive work.
declare -a RESOLVED_SEQUENCE_FILES=()
for sf in "${SEQUENCE_FILES[@]}"; do
  RESOLVED_SEQUENCE_FILES+=("$(resolve_sequence_file "${sf}")")
done

cd "${VHAP_DIR}"

# Per-clip naming convention: <subject>_<stem>_<suffix>
# The combined dataset name is:    <subject>_UNION<N>_<export_suffix>
# combine_nerf_datasets.py requires all source folders to live in the same
# parent directory as the target folder, so we keep everything under
# submodules/VHAP/export/monocular/.
NUM_CLIPS=${#SEQUENCE_FILES[@]}
COMBINED_NAME="${SUBJECT}_UNION${NUM_CLIPS}_${EXPORT_SUFFIX}"
COMBINED_FOLDER="export/monocular/${COMBINED_NAME}"

has_matching_file() {
  local dir="$1"
  local pattern="$2"
  find "${dir}" -maxdepth 1 -type f -name "${pattern}" -print -quit 2>/dev/null | grep -q .
}

preprocess_complete() {
  local input_video="$1"
  local sequence_dir="${input_video%.*}"
  has_matching_file "${sequence_dir}/images" "*.jpg" &&
    has_matching_file "${sequence_dir}/alpha_maps" "*.jpg"
}

latest_track_npz() {
  local track_out="$1"
  [ -d "${track_out}" ] || return 0
  find "${track_out}" -mindepth 2 -maxdepth 2 -type f -name "tracked_flame_params*.npz" 2>/dev/null | sort -V | tail -n 1
}

track_complete() {
  local track_out="$1"
  [ -n "$(latest_track_npz "${track_out}")" ]
}

export_complete() {
  local export_out="$1"
  [ -f "${export_out}/transforms.json" ] &&
    [ -f "${export_out}/transforms_train.json" ] &&
    [ -f "${export_out}/transforms_val.json" ] &&
    [ -f "${export_out}/transforms_test.json" ] &&
    [ -f "${export_out}/canonical_flame_param.npz" ]
}

combined_complete() {
  [ -f "${COMBINED_FOLDER}/transforms_train.json" ] &&
    [ -f "${COMBINED_FOLDER}/transforms_val.json" ] &&
    [ -f "${COMBINED_FOLDER}/transforms_test.json" ] &&
    [ -f "${COMBINED_FOLDER}/canonical_flame_param.npz" ] &&
    [ -f "${COMBINED_FOLDER}/sequences_trainval.txt" ] &&
    [ -f "${COMBINED_FOLDER}/sequences_test.txt" ]
}

run_track_one_clip() {
  local stem="$1"
  local input_video="$2"
  local data_root="$3"
  local extra_args=("${@:4}")

  local sequence="${SUBJECT}_${stem}"
  local track_out="output/monocular/${sequence}_${SUFFIX}"
  local export_out="export/monocular/${sequence}_${EXPORT_SUFFIX}"

  if (( SKIP_EXISTING == 1 )) && preprocess_complete "${input_video}"; then
    log "  skip preprocess ${input_video}"
  else
    log "  preprocess ${input_video}"
    ${PYTHON} vhap/preprocess_video.py \
      --input "${input_video}" \
      --matting_method robust_video_matting
  fi

  if (( SKIP_EXISTING == 1 )) && track_complete "${track_out}"; then
    log "  skip FLAME tracking; found $(latest_track_npz "${track_out}")"
  else
    log "  FLAME tracking -> ${track_out}"
    ${PYTHON} vhap/track.py \
      --data.root_folder "${data_root}" \
      --exp.output_folder "${track_out}" \
      --data.sequence "${stem}" \
      --batch-size "${BATCH_SIZE}" \
      "${extra_args[@]}"
  fi

  if (( SKIP_EXISTING == 1 )) && export_complete "${export_out}"; then
    log "  skip export; NeRF-style dataset already exists at ${export_out}"
  else
    log "  export -> ${export_out}"
    ${PYTHON} vhap/export_as_nerf_dataset.py \
      --src_folder "${track_out}" \
      --tgt_folder "${export_out}" \
      --background-color white
  fi

  # Echo the per-clip export folder name so the caller can collect it.
  echo "${export_out}"
}

# Resolve the latest tracked_flame_params*.npz produced by Phase A. track.py
# writes results into <track_out>/<TIMESTAMP>/tracked_flame_params_<epoch>.npz,
# and export_as_nerf_dataset.py's load_config picks the latest timestamp via
# `sorted(src_folder.iterdir())[-1]` — we mirror that here so the value we pass
# into Phase B as --model.flame-params-path is unambiguous.
resolve_pass_a_npz() {
  local stem="$1"
  local sequence="${SUBJECT}_${stem}"
  local track_out="output/monocular/${sequence}_${SUFFIX}"

  local latest_run
  latest_run=$(find "${track_out}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
  if [ -z "${latest_run}" ]; then
    echo "[demo] no run dir found under ${VHAP_DIR}/${track_out}" >&2
    exit 1
  fi

  local latest_npz
  latest_npz=$(find "${latest_run}" -maxdepth 1 -name "tracked_flame_params*.npz" | sort -V | tail -n 1)
  if [ -z "${latest_npz}" ]; then
    echo "[demo] no tracked_flame_params*.npz under ${VHAP_DIR}/${latest_run}" >&2
    exit 1
  fi

  echo "${latest_npz}"
}

if (( SKIP_EXISTING == 1 )) && combined_complete; then
  log "Skip all phases; combined dataset already exists at ${COMBINED_FOLDER}"
  log "Done. Pass this path to demo/02_train.sh:"
  echo "    bash demo/02_train.sh --source-path \"${VHAP_DIR}/${COMBINED_FOLDER}\""
  exit 0
fi

# ----------------------------------------------------------------------------
# Phase A: shape source clip — full-parameter tracking.
# ----------------------------------------------------------------------------
declare -a EXPORT_FOLDERS=()

log "[A] Phase A — shape source clip: ${SHAPE_SOURCE_STEM}"
SHAPE_SOURCE_FILE=""
for sf in "${RESOLVED_SEQUENCE_FILES[@]}"; do
  stem_a="$(basename "${sf}")"
  stem_a="${stem_a%.*}"
  if [[ "${stem_a}" == "${SHAPE_SOURCE_STEM}" ]]; then
    SHAPE_SOURCE_FILE="${sf}"
    break
  fi
done

shape_export=$(run_track_one_clip "${SHAPE_SOURCE_STEM}" "${SHAPE_SOURCE_FILE}" "$(dirname "${SHAPE_SOURCE_FILE}")" | tail -n 1)
EXPORT_FOLDERS+=("${shape_export}")

PASS_A_NPZ=$(resolve_pass_a_npz "${SHAPE_SOURCE_STEM}")
log "[A] canonical FLAME globals -> ${PASS_A_NPZ}"

# ----------------------------------------------------------------------------
# Phase B: every other clip — frozen globals.
# ----------------------------------------------------------------------------
log "[B] Phase B — re-track remaining clips with frozen globals"
for sf in "${RESOLVED_SEQUENCE_FILES[@]}"; do
  stem_b="$(basename "${sf}")"
  stem_b="${stem_b%.*}"
  if [[ "${stem_b}" == "${SHAPE_SOURCE_STEM}" ]]; then
    continue
  fi

  log "  -> clip ${stem_b}"
  clip_export=$(run_track_one_clip "${stem_b}" "${sf}" "$(dirname "${sf}")" \
    --model.flame-params-path "${PASS_A_NPZ}" \
    --load-globals-only \
    --freeze-globals-from-init | tail -n 1)
  EXPORT_FOLDERS+=("${clip_export}")
done

# ----------------------------------------------------------------------------
# Phase C: combine all per-clip exports into one NeRF-style dataset.
# ----------------------------------------------------------------------------
log "[C] Phase C — combine ${#EXPORT_FOLDERS[@]} clip exports -> ${COMBINED_FOLDER}"

# combine_nerf_datasets.py requires every src_folder.parent == tgt_folder.parent
# (i.e. siblings). Since we wrote every export under export/monocular/, this
# holds by construction.
${PYTHON} vhap/combine_nerf_datasets.py \
  --src-folders ${EXPORT_FOLDERS[@]} \
  --tgt-folder "${COMBINED_FOLDER}" \
  --division-mode "${DIVISION_MODE}"

log "Done. Pass this path to demo/02_train.sh:"
echo "    bash demo/02_train.sh --source-path \"${VHAP_DIR}/${COMBINED_FOLDER}\""
