#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
EXP_ID=${2:-}
GPU_ID=${3:-0}

# ============================================================
# Clean Score-FM ablation script
# ============================================================
# First-principle design:
#   1. X_0/S_0 are generated states from the reference/base distribution.
#   2. Peptide-derived information never overwrites X_0/S_0.
#   3. No peptide-prior weights are used anywhere.
#   4. Peptide coordinate/sequence information can only be used as condition.
#   5. Loss ablations keep the same source/condition semantics.

STATE_PATH=on
PER_SAMPLE_T=on
TIME_EMBED=on
T_SAMPLING=uniform
LOSS_MODE=off
SAMPLER_MODE=bridge
COORD_PEP_AS_CONDITION=off
SEQ_INPUT_MODE=state
SEQ_CE_WEIGHT=${ABFLOW_SEQ_CE_WEIGHT:-1.0}

# Score-FM auxiliary loss weights. These are loss weights, not peptide-prior weights.
SCOREFM_LOSS_WEIGHT=${ABFLOW_SCOREFM_LOSS_WEIGHT:-0.05}
SCOREFM_X1_WEIGHT=${ABFLOW_SCOREFM_X1_WEIGHT:-0.1}
SCOREFM_DSM_WEIGHT=${ABFLOW_SCOREFM_DSM_WEIGHT:-1.0}
SCOREFM_VELOCITY_WEIGHT=${ABFLOW_SCOREFM_VELOCITY_WEIGHT:-1.0}
SCOREFM_LOCAL_DIST_WEIGHT=${ABFLOW_SCOREFM_LOCAL_DIST_WEIGHT:-0.05}
SCOREFM_CONTACT_WEIGHT=${ABFLOW_SCOREFM_CONTACT_WEIGHT:-0.05}
SCOREFM_INTER_CLASH_WEIGHT=${ABFLOW_SCOREFM_INTER_CLASH_WEIGHT:-0.0}
SCOREFM_INTRA_CLASH_WEIGHT=${ABFLOW_SCOREFM_INTRA_CLASH_WEIGHT:-0.0}

case "$EXP_ID" in
  # Clean reference flow: no peptide condition.
  # Purpose: establish the real generation baseline from reference state.
  REF)
    LOSS_MODE=off
    COORD_PEP_AS_CONDITION=off
    SEQ_INPUT_MODE=state
    ;;

  # Sequence-conditioned reference flow: S_pep is condition, not S_0 overwrite.
  # Purpose: isolate the value of sequence condition.
  REF_SEQ)
    LOSS_MODE=off
    COORD_PEP_AS_CONDITION=off
    SEQ_INPUT_MODE=pep_condition
    ;;

  # Coordinate-and-sequence-conditioned reference flow.
  # Purpose: main clean prior-as-condition setting.
  REF_COND)
    LOSS_MODE=off
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;

  # Loss ablations under the fixed REF_COND source/condition semantics.
  L1_X1)
    LOSS_MODE=x1
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;
  L2_X1_VEL)
    LOSS_MODE=x1_vel
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;
  L3_CORE)
    LOSS_MODE=dtm_core
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;
  L4_CONTACT)
    LOSS_MODE=no_clash
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;

  # Time-sampling diagnostics. Run only after L3_CORE is stable.
  Q_UNIFORM)
    LOSS_MODE=dtm_core
    T_SAMPLING=uniform
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;
  Q_LOW_T)
    LOSS_MODE=dtm_core
    T_SAMPLING=low_t
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;
  Q_STRATIFIED)
    LOSS_MODE=dtm_core
    T_SAMPLING=stratified
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;

  *)
    echo "Unknown EXP_ID: $EXP_ID"
    echo "Supported experiments:"
    echo "  Main source/condition chain: REF REF_SEQ REF_COND"
    echo "  Loss ablations:              L1_X1 L2_X1_VEL L3_CORE L4_CONTACT"
    echo "  Time diagnostics:            Q_UNIFORM Q_LOW_T Q_STRATIFIED"
    exit 2
    ;;
esac

run_with_env() {
  ABFLOW_SCOREFM_STATE_PATH="$STATE_PATH" \
  ABFLOW_SCOREFM_PER_SAMPLE_T="$PER_SAMPLE_T" \
  ABFLOW_SCOREFM_TIME_EMBED="$TIME_EMBED" \
  ABFLOW_SCOREFM_T_SAMPLING="$T_SAMPLING" \
  ABFLOW_SCOREFM_LOSS_MODE="$LOSS_MODE" \
  ABFLOW_SCOREFM_SAMPLER_MODE="$SAMPLER_MODE" \
  ABFLOW_COORD_PEP_AS_CONDITION="$COORD_PEP_AS_CONDITION" \
  ABFLOW_SEQ_INPUT_MODE="$SEQ_INPUT_MODE" \
  ABFLOW_SEQ_CE_WEIGHT="$SEQ_CE_WEIGHT" \
  ABFLOW_SCOREFM_LOSS_WEIGHT="$SCOREFM_LOSS_WEIGHT" \
  ABFLOW_SCOREFM_X1_WEIGHT="$SCOREFM_X1_WEIGHT" \
  ABFLOW_SCOREFM_DSM_WEIGHT="$SCOREFM_DSM_WEIGHT" \
  ABFLOW_SCOREFM_VELOCITY_WEIGHT="$SCOREFM_VELOCITY_WEIGHT" \
  ABFLOW_SCOREFM_LOCAL_DIST_WEIGHT="$SCOREFM_LOCAL_DIST_WEIGHT" \
  ABFLOW_SCOREFM_CONTACT_WEIGHT="$SCOREFM_CONTACT_WEIGHT" \
  ABFLOW_SCOREFM_INTER_CLASH_WEIGHT="$SCOREFM_INTER_CLASH_WEIGHT" \
  ABFLOW_SCOREFM_INTRA_CLASH_WEIGHT="$SCOREFM_INTRA_CLASH_WEIGHT" \
  GPU="$GPU_ID" \
  "$@"
}

print_settings() {
  echo "Experiment: $EXP_ID"
  echo "GPU: $GPU_ID"
  echo "STATE_PATH=$STATE_PATH"
  echo "PER_SAMPLE_T=$PER_SAMPLE_T"
  echo "TIME_EMBED=$TIME_EMBED"
  echo "T_SAMPLING=$T_SAMPLING"
  echo "LOSS_MODE=$LOSS_MODE"
  echo "SAMPLER_MODE=$SAMPLER_MODE"
  echo "COORD_PEP_AS_CONDITION=$COORD_PEP_AS_CONDITION"
  echo "SEQ_INPUT_MODE=$SEQ_INPUT_MODE"
  echo "SEQ_CE_WEIGHT=$SEQ_CE_WEIGHT"
  echo "SCOREFM_LOSS_WEIGHT=$SCOREFM_LOSS_WEIGHT"
  echo "SCOREFM_X1_WEIGHT=$SCOREFM_X1_WEIGHT"
  echo "SCOREFM_DSM_WEIGHT=$SCOREFM_DSM_WEIGHT"
  echo "SCOREFM_VELOCITY_WEIGHT=$SCOREFM_VELOCITY_WEIGHT"
  echo "SCOREFM_LOCAL_DIST_WEIGHT=$SCOREFM_LOCAL_DIST_WEIGHT"
  echo "SCOREFM_CONTACT_WEIGHT=$SCOREFM_CONTACT_WEIGHT"
  echo "SCOREFM_INTER_CLASH_WEIGHT=$SCOREFM_INTER_CLASH_WEIGHT"
  echo "SCOREFM_INTRA_CLASH_WEIGHT=$SCOREFM_INTRA_CLASH_WEIGHT"
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

  print_settings
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

RUN_ROOT=${ABFLOW_RUN_ROOT:-/home/data3/cjm/project/AbFlow/results_scorefm_clean}
RUN_DIR="${RUN_ROOT}/${EXP_ID}"
CONFIG_DIR="${RUN_ROOT}/generated_configs"
RUN_CONFIG="${CONFIG_DIR}/${EXP_ID}.json"

mkdir -p "$RUN_DIR" "$CONFIG_DIR"

run_with_env python - "$BASE_CONFIG" "$RUN_CONFIG" "$RUN_DIR" "$EXP_ID" <<'PY'
import json
import os
import sys

src, dst, save_dir, exp_id = sys.argv[1:5]

with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["save_dir"] = save_dir

resume_from_env = os.environ.get("ABFLOW_RESUME_CKPT", "").strip()
resume_from_cfg = str(cfg.get("resume_checkpoint", "")).strip()
resume_ckpt = resume_from_env or resume_from_cfg
cfg["resume_checkpoint"] = resume_ckpt if resume_ckpt else ""
if resume_ckpt and not os.path.isfile(resume_ckpt):
    raise FileNotFoundError(f"resume_checkpoint does not exist: {resume_ckpt}")

# Remove obsolete peptide-prior configuration from older base configs.
# The clean implementation only allows peptide-derived information as condition.
for key in list(cfg.keys()):
    low = key.lower()
    if (
        "pep_prior" in low
        or "peptide_prior" in low
        or low in {"coord_prior", "seq_prior"}
        or ("prior_weight" in low and "pep" in low)
    ):
        cfg.pop(key, None)

cfg["abflow_runtime"] = {
    "experiment_id": exp_id,
    "state_path": os.environ.get("ABFLOW_SCOREFM_STATE_PATH", ""),
    "per_sample_t": os.environ.get("ABFLOW_SCOREFM_PER_SAMPLE_T", ""),
    "time_embed": os.environ.get("ABFLOW_SCOREFM_TIME_EMBED", ""),
    "t_sampling": os.environ.get("ABFLOW_SCOREFM_T_SAMPLING", ""),
    "loss_mode": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", ""),
    "sampler_mode": os.environ.get("ABFLOW_SCOREFM_SAMPLER_MODE", ""),
    "coord_pep_as_condition": os.environ.get("ABFLOW_COORD_PEP_AS_CONDITION", ""),
    "seq_input_mode": os.environ.get("ABFLOW_SEQ_INPUT_MODE", ""),
    "seq_ce_weight": os.environ.get("ABFLOW_SEQ_CE_WEIGHT", ""),
    "scorefm_loss_weight": os.environ.get("ABFLOW_SCOREFM_LOSS_WEIGHT", ""),
    "scorefm_x1_weight": os.environ.get("ABFLOW_SCOREFM_X1_WEIGHT", ""),
    "scorefm_dsm_weight": os.environ.get("ABFLOW_SCOREFM_DSM_WEIGHT", ""),
    "scorefm_velocity_weight": os.environ.get("ABFLOW_SCOREFM_VELOCITY_WEIGHT", ""),
    "scorefm_local_dist_weight": os.environ.get("ABFLOW_SCOREFM_LOCAL_DIST_WEIGHT", ""),
    "scorefm_contact_weight": os.environ.get("ABFLOW_SCOREFM_CONTACT_WEIGHT", ""),
    "scorefm_inter_clash_weight": os.environ.get("ABFLOW_SCOREFM_INTER_CLASH_WEIGHT", ""),
    "scorefm_intra_clash_weight": os.environ.get("ABFLOW_SCOREFM_INTRA_CLASH_WEIGHT", ""),
    "clean_reference_state": "true",
    "peptide_prior_state_injection": "false",
    "peptide_prior_weighting": "false",
}

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

print_settings
echo "Config: $RUN_CONFIG"
echo "Save dir: $RUN_DIR"

EFFECTIVE_RESUME_CKPT=$(python - "$RUN_CONFIG" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("resume_checkpoint", ""))
PY
)
echo "resume_checkpoint=$EFFECTIVE_RESUME_CKPT"

if [[ "${ABFLOW_DRY_RUN:-0}" == "1" ]]; then
  echo "ABFLOW_DRY_RUN=1: configuration generated; training was not started."
  exit 0
fi

run_with_env bash scripts/train/train.sh "$RUN_CONFIG"
