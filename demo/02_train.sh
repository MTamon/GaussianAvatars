#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo 2: Train a GaussianAvatars head avatar.
#
# Streams the tqdm "Training progress" bar from train.py:60 in real time
# (PYTHONUNBUFFERED=1 set in _common.sh).
#
# Inputs:
#   --source-path  Absolute path to a VHAP-exported NeRF-style dataset folder
#                  (the path printed by demo 01_*).
#                  Must contain transforms_train.json (+ val/test for --eval).
#   --model-path   Where to write the trained model (defaults under output/).
#   --iterations   Total training iterations (default: 600000 to match the
#                  paper's 600k schedule; lower for a quick demo).
#   --port         GUI server port for the optional remote viewer.
#
# Outputs:
#   ${MODEL_PATH}/point_cloud/iteration_*/point_cloud.ply
#   ${MODEL_PATH}/cameras.json, cfg_args, ...
#
# Examples:
#   bash demo/02_train.sh \
#     --source-path "$PWD/submodules/VHAP/export/monocular/obama_whiteBg_staticOffset_maskBelowLine"
#
#   bash demo/02_train.sh \
#     --source-path "$PWD/submodules/VHAP/export/nersemble/074_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine" \
#     --iterations 30000
# -----------------------------------------------------------------------------

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SOURCE_PATH=""
MODEL_PATH=""
RUN_NAME=""
ITERATIONS="600000"
PORT="60000"

usage() {
  cat <<'USAGE'
Usage:
  bash demo/02_train.sh --source-path PATH [options]

Options:
  --source-path PATH   VHAP export directory containing transforms_train.json (required)
  --model-path PATH    Output model directory (default: output/<source basename>)
  --run-name NAME      Name used when deriving --model-path (default: source basename)
  --iterations N       Training iterations (default: 600000)
  --port PORT          Remote viewer port (default: 60000)
  --env NAME           Conda env to activate (default: gaussian-avatars)
  -h, --help           Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-path) require_option_value "$(basename "$0")" "$1" "$#"; SOURCE_PATH="$2"; shift 2 ;;
    --model-path) require_option_value "$(basename "$0")" "$1" "$#"; MODEL_PATH="$2"; shift 2 ;;
    --run-name) require_option_value "$(basename "$0")" "$1" "$#"; RUN_NAME="$2"; shift 2 ;;
    --iterations) require_option_value "$(basename "$0")" "$1" "$#"; ITERATIONS="$2"; shift 2 ;;
    --port) require_option_value "$(basename "$0")" "$1" "$#"; PORT="$2"; shift 2 ;;
    --env) require_option_value "$(basename "$0")" "$1" "$#"; GA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die_usage "$(basename "$0")" "unknown option: $1" ;;
  esac
done

if [ -z "${SOURCE_PATH}" ]; then
  die_usage "$(basename "$0")" "--source-path is required (path to a VHAP NeRF export)."
fi

if [ ! -f "${SOURCE_PATH}/transforms_train.json" ]; then
  echo "[demo] ${SOURCE_PATH} does not look like a VHAP export (no transforms_train.json)." >&2
  exit 1
fi

[[ -n "${ITERATIONS}" ]] || die_usage "$(basename "$0")" "--iterations requires a value"
[[ -n "${PORT}" ]] || die_usage "$(basename "$0")" "--port requires a value"
[[ -n "${GA_ENV}" ]] || die_usage "$(basename "$0")" "--env requires a value"
RUN_NAME="${RUN_NAME:-$(basename "${SOURCE_PATH}")}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/output/${RUN_NAME}}"

activate_env
cd "${REPO_ROOT}"

mkdir -p "$(dirname "${MODEL_PATH}")"

log "Training GaussianAvatars"
log "  source : ${SOURCE_PATH}"
log "  model  : ${MODEL_PATH}"
log "  iters  : ${ITERATIONS}"

${PYTHON} train.py \
  -s "${SOURCE_PATH}" \
  -m "${MODEL_PATH}" \
  --eval \
  --bind_to_mesh \
  --white_background \
  --iterations "${ITERATIONS}" \
  --port "${PORT}"

log "Training finished. Pass this path to demo/03_render.sh:"
echo "    bash demo/03_render.sh --model-path \"${MODEL_PATH}\""
