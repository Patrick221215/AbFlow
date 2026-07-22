#!/usr/bin/env bash
set -euo pipefail

# Manual/formal AbFlow structure evaluation.
#
# This script writes generated structures and runs cal_metrics.py.  It is not
# used by the train-time in-memory validation rollout.
#
# Important fix:
#   valid.json -> valid.pkl + valid_surf.pkl
#   test.json  -> test.pkl  + test_surf.pkl
#
# The previous script always used test.pkl/test_surf.pkl, even when TEST_SET was
# valid.json, which caused missing-key errors such as KeyError: '6x4t'.

CODE_DIR=$(realpath "$(dirname "$0")/../..")
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-20}"
N_STEPS="${N_STEPS:-10}"
SHOW_SAMPLE_PROGRESS="${SHOW_SAMPLE_PROGRESS:-1}"
GPU="${GPU:-0}"

CKPT="${1:-}"
TEST_SET="${2:-}"
SAVE_DIR="${3:-}"
TASK="${4:-rabd}"
SURF_FILE_ARG="${5:-}"

if [[ -z "$CKPT" || -z "$TEST_SET" ]]; then
    echo "Usage: bash $0 <checkpoint> <dataset.json> [save_dir] [task] [surf_file]"
    echo "  task: rabd, igfold, or custom"
    exit 2
fi

[[ -f "$CKPT" ]] || {
    echo "[ERROR] Checkpoint not found: $CKPT" >&2
    exit 2
}
[[ -f "$TEST_SET" ]] || {
    echo "[ERROR] Dataset JSON not found: $TEST_SET" >&2
    exit 2
}

CKPT=$(realpath "$CKPT")
TEST_SET=$(realpath "$TEST_SET")
TEST_DIR=$(dirname "$TEST_SET")
SPLIT_NAME=$(basename "$TEST_SET")
SPLIT_NAME="${SPLIT_NAME%.json}"

if [[ -z "$SAVE_DIR" ]]; then
    SAVE_DIR="$(dirname "$CKPT")/results_${SPLIT_NAME}"
fi
SAVE_DIR=$(realpath -m "$SAVE_DIR")

SCRIPT="generate.py"
PEP_FILE=""
SURF_FILE=""

case "$TASK" in
    rabd)
        SCRIPT="generate.py"
        PEP_FILE="${ABFLOW_PEP_FILE:-${TEST_DIR}/${SPLIT_NAME}.pkl}"
        SURF_FILE="${ABFLOW_SURF_FILE:-${TEST_DIR}/${SPLIT_NAME}_surf.pkl}"
        ;;
    igfold)
        SCRIPT="struct_generate.py"
        PEP_FILE="${ABFLOW_PEP_FILE:-${TEST_DIR}/${SPLIT_NAME}.pkl}"
        SURF_FILE="${ABFLOW_SURF_FILE:-${TEST_DIR}/${SPLIT_NAME}_surf.pkl}"
        ;;
    *)
        SCRIPT="generate.py"
        PEP_FILE="${ABFLOW_PEP_FILE:-}"
        SURF_FILE="${ABFLOW_SURF_FILE:-${SURF_FILE_ARG}}"
        ;;
esac

if [[ -n "$PEP_FILE" && ! -f "$PEP_FILE" ]]; then
    echo "[ERROR] Proposal sidecar not found: $PEP_FILE" >&2
    echo "For ${SPLIT_NAME}.json, expected ${SPLIT_NAME}.pkl." >&2
    echo "Override explicitly with ABFLOW_PEP_FILE=/path/to/file.pkl." >&2
    exit 2
fi
if [[ -n "$SURF_FILE" && ! -f "$SURF_FILE" ]]; then
    echo "[ERROR] Surface sidecar not found: $SURF_FILE" >&2
    echo "For ${SPLIT_NAME}.json, expected ${SPLIT_NAME}_surf.pkl." >&2
    echo "Override explicitly with ABFLOW_SURF_FILE=/path/to/file.pkl." >&2
    exit 2
fi

echo "Locate the project folder at ${CODE_DIR}"
echo "Using GPU: ${GPU}"
echo "Evaluating: ${CKPT}"
echo "Dataset: ${TEST_SET}"
echo "Dataset split: ${SPLIT_NAME}"
echo "Batch size: ${BATCH_SIZE}"
echo "Sampling steps: ${N_STEPS}"
echo "Show sample progress: ${SHOW_SAMPLE_PROGRESS}"
echo "Proposal sidecar: ${PEP_FILE:-<none>}"
echo "Surface sidecar: ${SURF_FILE:-<none>}"
echo "Results: ${SAVE_DIR}"
echo "Task: ${TASK}"
echo "Script: ${SCRIPT}"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$CODE_DIR"
mkdir -p "$SAVE_DIR"

args=(
    "$SCRIPT"
    --ckpt "$CKPT"
    --test_set "$TEST_SET"
    --save_dir "$SAVE_DIR"
    --batch_size "$BATCH_SIZE"
    --gpu 0
    --n_steps "$N_STEPS"
)

if [[ -n "$PEP_FILE" ]]; then
    args+=(--pep_file "$PEP_FILE")
fi
if [[ -n "$SURF_FILE" ]]; then
    args+=(--surf_file "$SURF_FILE")
fi
if [[ "$SHOW_SAMPLE_PROGRESS" == "1" ]]; then
    args+=(--show_sample_progress)
fi

python "${args[@]}"

echo "Done generation"

SUMMARY_FILE="${SAVE_DIR}/summary.json"
if [[ ! -f "$SUMMARY_FILE" ]]; then
    echo "[ERROR] Generation failed: ${SUMMARY_FILE} was not created." >&2
    exit 1
fi

OPENMM_CPU_THREADS=1 python cal_metrics.py \
    --test_set "$SUMMARY_FILE" \
    --num_workers "$NUM_WORKERS"

echo "Done evaluation"
