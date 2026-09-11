#!/bin/bash
set -euo pipefail

# AbFlow V207 formal Test entry (shared by in-Trainer and standalone testing).
#
# RAbD path:
#   - one checkpoint / one model snapshot
#   - one or many GPUs cooperate on the SAME test set
#   - uses the same utils/epoch_test.py core as Trainer epoch Test
#   - same logical batches, same batch-keyed RNG, same model.sample(), same cal_metrics.py
#
# Other historical tasks keep the legacy single-process path below.

CODE_DIR=$(realpath "$(dirname "$0")/../..")
NUM_WORKERS="${ABFLOW_EPOCH_TEST_METRIC_WORKERS:-${NUM_WORKERS:-4}}"
BATCH_SIZE="${ABFLOW_EPOCH_TEST_BATCH_SIZE:-${BATCH_SIZE:-20}}"
N_STEPS="${ABFLOW_EPOCH_TEST_N_STEPS:-${N_STEPS:-10}}"
BASE_SEED="${ABFLOW_EPOCH_TEST_BASE_SEED:-2023}"
EVAL_CDR="${ABFLOW_EPOCH_TEST_CDR:-}"
SHOW_SAMPLE_PROGRESS="${ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS:-${SHOW_SAMPLE_PROGRESS:-1}}"
KEEP_STRUCTURES="${ABFLOW_EPOCH_TEST_KEEP_STRUCTURES:-on}"
METRICS_ONLY="${ABFLOW_EPOCH_TEST_METRICS_ONLY:-off}"
GPU="${GPU:-0}"
PORT="${PORT:-29500}"

MODE="generate"
if [[ "${1:-}" == "metrics-only" ]]; then
  MODE="metrics-only"
  CKPT=""
  TEST_SET="${2:-}"
  SAVE_DIR="${3:-}"
  TASK="${4:-rabd}"
  SURF_FILE="${5:-}"
  METRICS_ONLY="on"
else
  CKPT="${1:-}"
  TEST_SET="${2:-}"
  SAVE_DIR="${3:-}"
  TASK="${4:-rabd}"
  SURF_FILE="${5:-}"
fi

if [[ -z "$TEST_SET" || ( "$MODE" == "generate" && -z "$CKPT" ) ]]; then
  echo "Usage: GPU=2,3 bash $0 <checkpoint> <test set> [save_dir] [task] [surf_file]"
  echo "       GPU=0 bash $0 metrics-only <test set> <existing_save_dir> [task] [surf_file]"
  echo "  task: rabd (formal DDP parity path), igfold, or custom path"
  exit 2
fi
if [[ "$MODE" == "metrics-only" && "$TASK" != "rabd" ]]; then
  echo "[ERROR] metrics-only is supported only for the formal rabd path." >&2
  exit 2
fi

if [[ "$MODE" == "generate" ]]; then
  [[ -f "$CKPT" ]] || { echo "[ERROR] checkpoint not found: $CKPT" >&2; exit 2; }
fi
[[ -f "$TEST_SET" ]] || { echo "[ERROR] test set not found: $TEST_SET" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] BATCH_SIZE must be positive: $BATCH_SIZE" >&2; exit 2; }
[[ "$N_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] N_STEPS must be positive: $N_STEPS" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] NUM_WORKERS must be positive: $NUM_WORKERS" >&2; exit 2; }

if [[ "$MODE" == "generate" ]]; then
  CKPT=$(realpath "$CKPT")
fi
TEST_SET=$(realpath "$TEST_SET")
TEST_DIR=$(dirname "$TEST_SET")

if [[ -z "$SAVE_DIR" ]]; then
  if [[ "$MODE" == "metrics-only" ]]; then
    echo "[ERROR] metrics-only requires an existing save_dir." >&2
    exit 2
  fi
  SAVE_DIR="$(dirname "$CKPT")/results"
fi
mkdir -p "$SAVE_DIR"
SAVE_DIR=$(realpath "$SAVE_DIR")

# ---------------------------------------------------------------------------
# Formal RAbD path: SAME engine as in-Trainer epoch Test.
# ---------------------------------------------------------------------------
if [[ "$TASK" == "rabd" ]]; then
  PEP_FILE="${ABFLOW_EPOCH_TEST_PEP:-${TEST_DIR}/test.pkl}"
  if [[ -n "$SURF_FILE" ]]; then
    SURF_FILE=$(realpath "$SURF_FILE")
  else
    SURF_FILE="${ABFLOW_EPOCH_TEST_SURF:-${TEST_DIR}/test_surf.pkl}"
  fi

  [[ -f "$PEP_FILE" ]] || { echo "[ERROR] pep file not found: $PEP_FILE"; exit 2; }
  [[ -f "$SURF_FILE" ]] || { echo "[ERROR] surf file not found: $SURF_FILE"; exit 2; }

  IFS=',' read -r -a GPU_ARRAY <<< "$GPU"
  NPROC=${#GPU_ARRAY[@]}
  if (( NPROC < 1 )); then
    echo "[ERROR] invalid GPU list: $GPU"
    exit 2
  fi

  export CUDA_VISIBLE_DEVICES="$GPU"
  export ABFLOW_PROJECT_ROOT="$CODE_DIR"

  ARGS=(
    --test_set "$TEST_SET"
    --save_dir "$SAVE_DIR"
    --pep_file "$PEP_FILE"
    --surf_file "$SURF_FILE"
    --batch_size "$BATCH_SIZE"
    --n_steps "$N_STEPS"
    --base_seed "$BASE_SEED"
    --metric_workers "$NUM_WORKERS"
  )

  if [[ "$MODE" == "generate" ]]; then
    ARGS+=(--ckpt "$CKPT")
  fi

  if [[ -n "$EVAL_CDR" ]]; then
    ARGS+=(--eval_cdr "$EVAL_CDR")
  fi

  case "${SHOW_SAMPLE_PROGRESS,,}" in
    1|true|yes|y|on) ARGS+=(--show_sample_progress) ;;
  esac
  case "${KEEP_STRUCTURES,,}" in
    0|false|no|n|off) ARGS+=(--delete_structures_after_metrics) ;;
  esac
  case "${METRICS_ONLY,,}" in
    1|true|yes|y|on) ARGS+=(--metrics_only) ;;
  esac

  echo "[FormalTest] task=rabd"
  echo "[FormalTest] mode=$MODE checkpoint=${CKPT:-not_used}"
  echo "[FormalTest] GPUs=$GPU world_size=$NPROC"
  echo "[FormalTest] test_set=$TEST_SET"
  echo "[FormalTest] logical_batch_size=$BATCH_SIZE n_steps=$N_STEPS base_seed=$BASE_SEED"
  echo "[FormalTest] eval_cdr=${EVAL_CDR:-checkpoint_contract} metric_workers=$NUM_WORKERS"
  echo "[FormalTest] metrics_only=$METRICS_ONLY keep_structures=$KEEP_STRUCTURES"
  echo "[FormalTest] save_dir=$SAVE_DIR"

  cd "$CODE_DIR"
  export PYTHONUNBUFFERED=1
  if (( NPROC > 1 )); then
    torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
      scripts/test/generate_epoch_test_ddp.py "${ARGS[@]}"
  else
    python scripts/test/generate_epoch_test_ddp.py "${ARGS[@]}"
  fi
  echo "[FormalTest] Done."
  exit 0
fi

# ---------------------------------------------------------------------------
# Historical fallback for non-RAbD tasks only.
# This keeps the prior CLI behavior but is NOT used by the formal R01/R02/R03
# epoch-test protocol.
# ---------------------------------------------------------------------------
LEGACY_NUM_WORKERS="${NUM_WORKERS:-8}"
LEGACY_BATCH_SIZE="${BATCH_SIZE:-20}"
LEGACY_N_STEPS="${N_STEPS:-10}"

if [[ "$GPU" == *,* ]]; then
  echo "[ERROR] historical non-rabd fallback is single-GPU only; got GPU=$GPU" >&2
  exit 2
fi

if [[ "$TASK" == "igfold" ]]; then
  PEP_ARG="--pep_file ${TEST_DIR}/test.pkl"
  SURF_ARG="--surf_file ${TEST_DIR}/test_surf.pkl"
  SCRIPT="struct_generate.py"
else
  PEP_ARG=""
  SURF_ARG="${SURF_FILE:+--surf_file ${SURF_FILE}}"
  SCRIPT="generate.py"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$CODE_DIR"
mkdir -p "$SAVE_DIR"

GEN_EXTRA_ARGS="--n_steps ${LEGACY_N_STEPS}"
case "${SHOW_SAMPLE_PROGRESS,,}" in
  1|true|yes|y|on) GEN_EXTRA_ARGS="${GEN_EXTRA_ARGS} --show_sample_progress" ;;
esac

python "$SCRIPT" \
  --ckpt "$CKPT" \
  --test_set "$TEST_SET" \
  --save_dir "$SAVE_DIR" \
  --batch_size "$LEGACY_BATCH_SIZE" \
  --gpu 0 \
  $PEP_ARG \
  $SURF_ARG \
  $GEN_EXTRA_ARGS

SUMMARY_FILE="${SAVE_DIR}/summary.json"
[[ -f "$SUMMARY_FILE" ]] || { echo "[ERROR] Generation failed: $SUMMARY_FILE was not created."; exit 1; }

OPENMM_CPU_THREADS=1 python cal_metrics.py \
  --test_set "$SUMMARY_FILE" \
  --num_workers "$LEGACY_NUM_WORKERS"
