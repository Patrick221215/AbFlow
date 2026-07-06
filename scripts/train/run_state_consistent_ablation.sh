#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
EXP_ID=${2:-}
GPU_ID=${3:-0}

# Common controlled settings.
PER_SAMPLE_T=on
TIME_EMBED=on
T_SAMPLING=uniform
LOSS_MODE=off
SAMPLER_MODE=bridge
COORD_PRIOR=0.0
COORD_PRIOR_MODE=blend
COORD_PRIOR_SIGMA=1.0
SEQ_PRIOR=0.0
SEQ_INPUT_MODE=state
SEQ_CE_WEIGHT=${ABFLOW_SEQ_CE_WEIGHT:-1.0}

case "$EXP_ID" in
  # Time-module ablations without peptide prior.
  T0) PER_SAMPLE_T=off; TIME_EMBED=off ;;
  T1) PER_SAMPLE_T=on;  TIME_EMBED=off ;;
  T2) PER_SAMPLE_T=off; TIME_EMBED=on ;;
  T3) PER_SAMPLE_T=on;  TIME_EMBED=on ;;

  # Peptide-prior ablations under the default time-on setting.
  P0)  COORD_PRIOR=0.0; SEQ_PRIOR=0.0 ;;
  PX)  COORD_PRIOR=0.5; SEQ_PRIOR=0.0 ;;
  PS)  COORD_PRIOR=0.0; SEQ_PRIOR=0.5 ;;
  PB)  COORD_PRIOR=1.0; SEQ_PRIOR=1.0 ;;
  PX1) COORD_PRIOR=1.0; SEQ_PRIOR=0.0 ;;
  PS1) COORD_PRIOR=0.0; SEQ_PRIOR=1.0 ;;

  # Explicit original-prior baselines. These avoid the ambiguity of PB.
  B0) PER_SAMPLE_T=off; TIME_EMBED=off; COORD_PRIOR=1.0; SEQ_PRIOR=1.0 ;;
  B1) PER_SAMPLE_T=on;  TIME_EMBED=off; COORD_PRIOR=1.0; SEQ_PRIOR=1.0 ;;
  B2) PER_SAMPLE_T=off; TIME_EMBED=on;  COORD_PRIOR=1.0; SEQ_PRIOR=1.0 ;;
  B3) PER_SAMPLE_T=on;  TIME_EMBED=on;  COORD_PRIOR=1.0; SEQ_PRIOR=1.0 ;;

  # Sequence-conditioning diagnostics.
  # These require AbFlow_model.py to support ABFLOW_SEQ_INPUT_MODE=pep_condition.
  S0) PER_SAMPLE_T=off; TIME_EMBED=off; COORD_PRIOR=1.0; SEQ_PRIOR=1.0; SEQ_INPUT_MODE=pep_condition ;;
  S1) PER_SAMPLE_T=on;  TIME_EMBED=off; COORD_PRIOR=1.0; SEQ_PRIOR=1.0; SEQ_INPUT_MODE=pep_condition ;;
  S2) PER_SAMPLE_T=off; TIME_EMBED=on;  COORD_PRIOR=1.0; SEQ_PRIOR=1.0; SEQ_INPUT_MODE=pep_condition ;;
  S3) PER_SAMPLE_T=on;  TIME_EMBED=on;  COORD_PRIOR=1.0; SEQ_PRIOR=1.0; SEQ_INPUT_MODE=pep_condition ;;
  S3CG05) PER_SAMPLE_T=on; TIME_EMBED=on; COORD_PRIOR=1.0; COORD_PRIOR_MODE=conditional_gaussian; COORD_PRIOR_SIGMA=0.5; SEQ_PRIOR=1.0; SEQ_INPUT_MODE=pep_condition ;;
  S3CG10) PER_SAMPLE_T=on; TIME_EMBED=on; COORD_PRIOR=1.0; COORD_PRIOR_MODE=conditional_gaussian; COORD_PRIOR_SIGMA=1.0; SEQ_PRIOR=1.0; SEQ_INPUT_MODE=pep_condition ;;
  S3CE2) PER_SAMPLE_T=on; TIME_EMBED=on; COORD_PRIOR=1.0; SEQ_PRIOR=1.0; SEQ_INPUT_MODE=pep_condition; SEQ_CE_WEIGHT=2.0 ;;
  S3CE5) PER_SAMPLE_T=on; TIME_EMBED=on; COORD_PRIOR=1.0; SEQ_PRIOR=1.0; SEQ_INPUT_MODE=pep_condition; SEQ_CE_WEIGHT=5.0 ;;

  Q0) T_SAMPLING=uniform ;;
  Q1) T_SAMPLING=low_t ;;
  Q2) T_SAMPLING=stratified ;;

  L0) LOSS_MODE=off;      COORD_PRIOR=0.5; SEQ_PRIOR=0.5 ;;
  L1) LOSS_MODE=x1;       COORD_PRIOR=1; SEQ_PRIOR=1 ;;
  L2) LOSS_MODE=x1_vel;   COORD_PRIOR=0.5; SEQ_PRIOR=0.5 ;;
  L3) LOSS_MODE=dtm_core; COORD_PRIOR=0.5; SEQ_PRIOR=0.5 ;;
  L4) LOSS_MODE=full;     COORD_PRIOR=0.5; SEQ_PRIOR=0.5 ;;

  *)
    echo "Unknown EXP_ID: $EXP_ID"
    echo "Supported:"
    echo "  Time:   T0 T1 T2 T3"
    echo "  Prior:  P0 PX PS PB PX1 PS1"
    echo "  Base:   B0 B1 B2 B3"
    echo "  Seq:    S0 S1 S2 S3"
    echo "  T-dist: Q0 Q1 Q2"
    echo "  Loss:   L0 L1 L2 L3 L4"
    exit 2
    ;;
esac

run_with_env() {
  ABFLOW_SCOREFM_PER_SAMPLE_T="$PER_SAMPLE_T" \
  ABFLOW_SCOREFM_TIME_EMBED="$TIME_EMBED" \
  ABFLOW_SCOREFM_T_SAMPLING="$T_SAMPLING" \
  ABFLOW_SCOREFM_LOSS_MODE="$LOSS_MODE" \
  ABFLOW_SCOREFM_SAMPLER_MODE="$SAMPLER_MODE" \
  ABFLOW_COORD_PEP_PRIOR_WEIGHT="$COORD_PRIOR" \
  ABFLOW_COORD_PEP_PRIOR_MODE="$COORD_PRIOR_MODE" \
  ABFLOW_COORD_PEP_PRIOR_SIGMA="$COORD_PRIOR_SIGMA" \
  ABFLOW_SEQ_PEP_PRIOR_WEIGHT="$SEQ_PRIOR" \
  ABFLOW_SEQ_INPUT_MODE="$SEQ_INPUT_MODE" \
  ABFLOW_SEQ_CE_WEIGHT="$SEQ_CE_WEIGHT" \
  GPU="$GPU_ID" \
  "$@"
}

if [[ "$MODE" == "test" ]]; then
  CKPT=${4:-}
  RESULT_DIR=${5:-}
  TEST_JSON=${6:-datasets/RAbD/test.json}

  if [[ -z "$CKPT" || -z "$RESULT_DIR" ]]; then
    echo "Usage: bash $0 test <EXP_ID> <GPU_ID> <CKPT> <RESULT_DIR> [TEST_JSON]"
    exit 2
  fi
  if [[ ! -f "$CKPT" ]]; then
    echo "Checkpoint not found: $CKPT"
    exit 2
  fi
  if [[ ! -f "$TEST_JSON" ]]; then
    echo "Test JSON not found: $TEST_JSON"
    exit 2
  fi

  echo "Testing experiment: $EXP_ID"
  echo "Checkpoint: $CKPT"
  echo "Result dir: $RESULT_DIR"
  echo "Test JSON: $TEST_JSON"
  run_with_env bash scripts/test/test.sh \
    "$CKPT" "$TEST_JSON" "$RESULT_DIR" rabd
  exit 0
fi

if [[ "$MODE" != "train" ]]; then
  echo "Train: bash $0 train <EXP_ID> <GPU_ID> <BASE_CONFIG>"
  echo "Test:  bash $0 test <EXP_ID> <GPU_ID> <CKPT> <RESULT_DIR> [TEST_JSON]"
  exit 2
fi

BASE_CONFIG=${4:-scripts/train/configs/single_cdr_design_state_consistent.json}
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Base config not found: $BASE_CONFIG"
  exit 2
fi

RUN_ROOT="/home/data3/cjm/project/AbFlow/results_dtm"
RUN_DIR="${RUN_ROOT}/${EXP_ID}"
CONFIG_DIR="${RUN_ROOT}/generated_configs"
RUN_CONFIG="${CONFIG_DIR}/${EXP_ID}.json"

mkdir -p "$RUN_DIR" "$CONFIG_DIR"

python - "$BASE_CONFIG" "$RUN_CONFIG" "$RUN_DIR" <<'PY'
import json
import sys

src, dst, save_dir = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg["save_dir"] = save_dir
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

echo "Experiment: $EXP_ID"
echo "GPU: $GPU_ID"
echo "Config: $RUN_CONFIG"
echo "Save dir: $RUN_DIR"
echo "PER_SAMPLE_T=$PER_SAMPLE_T"
echo "TIME_EMBED=$TIME_EMBED"
echo "T_SAMPLING=$T_SAMPLING"
echo "LOSS_MODE=$LOSS_MODE"
echo "SAMPLER_MODE=$SAMPLER_MODE"
echo "COORD_PRIOR=$COORD_PRIOR"
echo "COORD_PRIOR_MODE=$COORD_PRIOR_MODE"
echo "COORD_PRIOR_SIGMA=$COORD_PRIOR_SIGMA"
echo "SEQ_PRIOR=$SEQ_PRIOR"
echo "SEQ_INPUT_MODE=$SEQ_INPUT_MODE"
echo "SEQ_CE_WEIGHT=$SEQ_CE_WEIGHT"

if [[ "${ABFLOW_DRY_RUN:-0}" == "1" ]]; then
  echo "ABFLOW_DRY_RUN=1: configuration generated; training was not started."
  exit 0
fi

run_with_env bash scripts/train/train.sh "$RUN_CONFIG"
