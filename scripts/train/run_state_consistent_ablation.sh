#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
EXP_ID=${2:-}
GPU_ID=${3:-0}

# ============================================================
# AbFlow v26 condition-final training ablations
# ============================================================
# 1. X_0/S_0 always come from the reference distribution.
# 2. X_pep/S_pep are conditions only; they never overwrite X_t/S_t.
# 3. No peptide-prior weights.
# 4. No independent score head and no pair-time conditioning.
# 5. One coordinate objective per complex:
#      endpoint      : unique per-complex endpoint loss
#      analytic_core : endpoint outside [t_min,t_max], CA analytic DSM inside
# 6. No x1/velocity/DSM stacking and no geometry/contact auxiliary in this stage.

STATE_PATH=on
PER_SAMPLE_T=on
TIME_EMBED=on
T_SAMPLING=uniform
LOSS_MODE=endpoint
SAMPLER_MODE=bridge
MIN_SIGMA=${ABFLOW_SCOREFM_MIN_SIGMA:-0.01}
DSM_T_MIN=${ABFLOW_SCOREFM_DSM_T_MIN:-0.2}
DSM_T_MAX=${ABFLOW_SCOREFM_DSM_T_MAX:-0.8}

COORD_PEP_AS_CONDITION=off
SEQ_INPUT_MODE=state
SEQ_CE_WEIGHT=${ABFLOW_SEQ_CE_WEIGHT:-1.0}

case "$EXP_ID" in
  # Pure reference-state endpoint FM.
  REF)
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=off
    SEQ_INPUT_MODE=state
    ;;

  # Isolate proposal-sequence conditioning.
  REF_SEQ)
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=off
    SEQ_INPUT_MODE=pep_condition
    ;;

  # Isolate coordinate conditioning without proposal-sequence conditioning.
  REF_COORD)
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=state
    ;;

  # Main clean conditional endpoint baseline.
  REF_COND)
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;

  # Main analytic-score experiment under identical source/condition semantics.
  CORE)
    LOSS_MODE=analytic_core
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;

  *)
    echo "Unknown EXP_ID: $EXP_ID"
    echo "Supported: REF REF_SEQ REF_COORD REF_COND CORE"
    exit 2
    ;;
esac

run_with_env() {
  ABFLOW_SCOREFM_STATE_PATH="$STATE_PATH" \
  ABFLOW_SCOREFM_PER_SAMPLE_T="$PER_SAMPLE_T" \
  ABFLOW_SCOREFM_TIME_EMBED="$TIME_EMBED" \
  ABFLOW_SCOREFM_T_SAMPLING="$T_SAMPLING" \
  ABFLOW_SCOREFM_LOSS_MODE="$LOSS_MODE" \
  ABFLOW_SCOREFM_MIN_SIGMA="$MIN_SIGMA" \
  ABFLOW_SCOREFM_DSM_T_MIN="$DSM_T_MIN" \
  ABFLOW_SCOREFM_DSM_T_MAX="$DSM_T_MAX" \
  ABFLOW_SCOREFM_SAMPLER_MODE="$SAMPLER_MODE" \
  ABFLOW_COORD_PEP_AS_CONDITION="$COORD_PEP_AS_CONDITION" \
  ABFLOW_SEQ_INPUT_MODE="$SEQ_INPUT_MODE" \
  ABFLOW_SEQ_CE_WEIGHT="$SEQ_CE_WEIGHT" \
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
  echo "MIN_SIGMA=$MIN_SIGMA"
  echo "DSM_T_MIN=$DSM_T_MIN"
  echo "DSM_T_MAX=$DSM_T_MAX"
  echo "SAMPLER_MODE=$SAMPLER_MODE"
  echo "COORD_PEP_AS_CONDITION=$COORD_PEP_AS_CONDITION"
  echo "SEQ_INPUT_MODE=$SEQ_INPUT_MODE"
  echo "SEQ_CE_WEIGHT=$SEQ_CE_WEIGHT"
}

if [[ "$MODE" != "train" ]]; then
  echo "Usage: bash $0 train <EXP_ID> <GPU_ID> <BASE_CONFIG>"
  echo "Supported EXP_ID: REF REF_SEQ REF_COORD REF_COND CORE"
  exit 2
fi

BASE_CONFIG=${4:-scripts/train/configs/single_cdr_design.json}
[[ -f "$BASE_CONFIG" ]] || { echo "Base config not found: $BASE_CONFIG"; exit 2; }

RUN_ROOT=${ABFLOW_RUN_ROOT:-/home/data3/cjm/project/AbFlow/results_dtm}
RUN_DIR="${RUN_ROOT}/${EXP_ID}"
CONFIG_DIR="${RUN_ROOT}/generated_configs"
RUN_CONFIG="${CONFIG_DIR}/${EXP_ID}.json"
RUNTIME_META="${RUN_DIR}/abflow_runtime.json"

mkdir -p "$RUN_DIR" "$CONFIG_DIR"

run_with_env python - "$BASE_CONFIG" "$RUN_CONFIG" "$RUN_DIR" "$EXP_ID" "$RUNTIME_META" <<'PY'
import json
import os
import sys

src, dst, save_dir, exp_id, runtime_meta = sys.argv[1:6]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["save_dir"] = save_dir
# Fresh start is the default.  A legacy checkpoint in the base JSON must never
# silently leak into a new condition ablation.  Resume is allowed only through
# the explicit ABFLOW_RESUME_CKPT environment variable.
resume_ckpt = os.environ.get("ABFLOW_RESUME_CKPT", "").strip()
cfg["resume_checkpoint"] = resume_ckpt
if resume_ckpt and not os.path.isfile(resume_ckpt):
    raise FileNotFoundError(
        f"ABFLOW_RESUME_CKPT does not exist: {resume_ckpt}"
    )

# Remove obsolete weighted-prior and redundant objective controls.
obsolete_exact = {
    "coord_prior",
    "seq_prior",
    "coord_pep_prior_weight",
    "seq_pep_prior_weight",
    "scorefm_loss_weight",
    "scorefm_x1_weight",
    "scorefm_velocity_weight",
    "scorefm_dsm_weight",
    "scorefm_transport_weight",
    "scorefm_local_dist_weight",
    "scorefm_contact_weight",
    "scorefm_inter_clash_weight",
    "scorefm_intra_clash_weight",
    "scorefm_t_threshold",
    "scorefm_hybrid_gate_k",
    "scorefm_hybrid_mix_mode",
    "scorefm_max_effective_snr",
    "coord_pep_max_fraction",
}
for key in list(cfg.keys()):
    low = key.lower()
    if (
        low in obsolete_exact
        or "pep_prior" in low
        or "peptide_prior" in low
        or ("prior_weight" in low and "pep" in low)
    ):
        cfg.pop(key, None)

runtime = {
    "experiment_id": exp_id,
    "state_path": os.environ.get("ABFLOW_SCOREFM_STATE_PATH", ""),
    "per_sample_t": os.environ.get("ABFLOW_SCOREFM_PER_SAMPLE_T", ""),
    "time_embed": os.environ.get("ABFLOW_SCOREFM_TIME_EMBED", ""),
    "t_sampling": os.environ.get("ABFLOW_SCOREFM_T_SAMPLING", ""),
    "loss_mode": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", ""),
    "min_sigma": os.environ.get("ABFLOW_SCOREFM_MIN_SIGMA", ""),
    "dsm_t_min": os.environ.get("ABFLOW_SCOREFM_DSM_T_MIN", ""),
    "dsm_t_max": os.environ.get("ABFLOW_SCOREFM_DSM_T_MAX", ""),
    "sampler_mode": os.environ.get("ABFLOW_SCOREFM_SAMPLER_MODE", ""),
    "coord_pep_as_condition": os.environ.get(
        "ABFLOW_COORD_PEP_AS_CONDITION", ""
    ),
    "seq_input_mode": os.environ.get("ABFLOW_SEQ_INPUT_MODE", ""),
    "seq_ce_weight": os.environ.get("ABFLOW_SEQ_CE_WEIGHT", ""),
    "clean_reference_state": "true",
    "peptide_state_injection": "false",
    "peptide_prior_weighting": "false",
    "independent_score_head": "false",
    "pair_time_conditioning": "false",
    "coordinate_objective_stacking": "false",
    "analytic_score_scope": "CA_translation_only",
    "true_path_endpoint": "1.0",
    "condition_representation": (
        "proposal_local_frame_radial_log1p_state_time_residual"
    ),
    "direct_coordinate_condition_update": "false",
    "train_only_script": "true",
}

# Store non-training metadata in a sidecar file.  Keeping it out of the model
# config prevents train.sh/argparse from receiving an unknown --abflow_runtime
# argument.
os.makedirs(os.path.dirname(runtime_meta), exist_ok=True)
with open(runtime_meta, "w", encoding="utf-8") as f:
    json.dump(runtime, f, indent=2, ensure_ascii=False)
    f.write("\n")

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

print_settings
echo "Config: $RUN_CONFIG"
echo "Save dir: $RUN_DIR"
echo "Runtime metadata: $RUNTIME_META"

EFFECTIVE_RESUME_CKPT=$(python - "$RUN_CONFIG" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("resume_checkpoint", ""))
PY
)
echo "resume_checkpoint=$EFFECTIVE_RESUME_CKPT"

run_with_env bash scripts/train/train.sh "$RUN_CONFIG"