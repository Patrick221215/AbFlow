#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
EXP_ID=${2:-}
GPU_ID=${3:-0}

# Resolve the current launcher itself so AutoTopK evaluates with the same
# experiment definitions used for training.  Do not hard-code the legacy
# run_state_consistent_ablation.sh, which does not know v52 EXP_IDs.
SELF_LAUNCHER=$(python - "${BASH_SOURCE[0]}" <<'PYSELF'
import os, sys
print(os.path.realpath(sys.argv[1]))
PYSELF
)

# ============================================================
# AbFlow v56: Structured-Primary causal base experiment
# ============================================================
# S00: batch-56 matched PCS-RC endpoint baseline.
# S01: structured H3 graph-translation state is the PRIMARY training path;
#      clean native endpoint remains the coordinate target.
# S02: same stochastic path as S01; only the coordinate target changes to the
#      path-consistent endpoint parameterization induced by conditional FM.
#
# Historical BASE / SATC_CORE / GT_SATC are not rerun here.
# Pair-time, SATC auxiliary and GT second-query auxiliary are all disabled.
# AbFlow paper numbers are external locators only. Formal causal comparisons:
#      S01-S00 = structured-state exposure
#      S02-S01 = path/field consistency
#      S02-S00 = complete structured-primary base
# ============================================================
STATE_PATH=on
PER_SAMPLE_T=on
TIME_EMBED=on
PAIR_TIME_SCOPE=${ABFLOW_PAIR_TIME_SCOPE:-off}
T_SAMPLING=uniform
LOSS_MODE=endpoint
SAMPLER_MODE=bridge
MIN_SIGMA=${ABFLOW_SCOREFM_MIN_SIGMA:-0.01}
DSM_T_MIN=${ABFLOW_SCOREFM_DSM_T_MIN:-0.2}
DSM_T_MAX=${ABFLOW_SCOREFM_DSM_T_MAX:-0.8}

SOURCE_MODE=reference
RECURRENT_PROPOSAL_CONTEXT=off
COORD_PEP_SOURCE_WEIGHT=${ABFLOW_COORD_PEP_SOURCE_WEIGHT:-1.0}
SEQ_PEP_SOURCE_WEIGHT=${ABFLOW_SEQ_PEP_SOURCE_WEIGHT:-1.0}
COORD_PEP_AS_CONDITION=off
SEQ_INPUT_MODE=state
SHADOW_SEQ_STATE=${ABFLOW_SHADOW_SEQ_STATE:-off}
DUAL_SEQUENCE_STATE=${ABFLOW_DUAL_SEQUENCE_STATE:-off}
DUAL_SEQUENCE_ATOM_MODE=${ABFLOW_DUAL_SEQUENCE_ATOM_MODE:-hidden_only}
SEQUENCE_CONTEXT_MODE=${ABFLOW_SEQUENCE_CONTEXT_MODE:-legacy}
FINAL_READOUT_MODE=${ABFLOW_FINAL_READOUT_MODE:-integrated_endpoint}
SEQUENCE_DECODE_MODE=${ABFLOW_SEQUENCE_DECODE_MODE:-argmax}
DETERMINISTIC_VALIDATION=${ABFLOW_DETERMINISTIC_VALIDATION:-on}
SEQ_CE_WEIGHT=${ABFLOW_SEQ_CE_WEIGHT:-1.0}
PROPOSAL_ADAPTER_START_ROUND=${ABFLOW_PROPOSAL_ADAPTER_START_ROUND:-0}
SI_GAMMA_SCALE=${ABFLOW_SI_GAMMA_SCALE:-0.25}
SI_SCORE_WEIGHT=${ABFLOW_SI_SCORE_WEIGHT:-0.002}
SI_VELOCITY_WEIGHT=${ABFLOW_SI_VELOCITY_WEIGHT:-0.01}
STRUCTURED_GAMMA_SCALE=${ABFLOW_STRUCTURED_GAMMA_SCALE:-0.05}
STRUCTURED_TRANSPORT_MAX=${ABFLOW_STRUCTURED_TRANSPORT_MAX:-20.0}
STRUCTURED_GAMMA_ABS_MAX=${ABFLOW_STRUCTURED_GAMMA_ABS_MAX:-1.0}
STRUCTURED_LOCAL_GAMMA_SCALE=${ABFLOW_STRUCTURED_LOCAL_GAMMA_SCALE:-0.05}

TRAJ_CONSISTENCY_WEIGHT=${ABFLOW_TRAJ_CONSISTENCY_WEIGHT:-0.05}
TRAJ_VELOCITY_WEIGHT=${ABFLOW_TRAJ_VELOCITY_WEIGHT:-0.0}
TRAJ_DELTA_T=${ABFLOW_TRAJ_DELTA_T:-0.15}
TRAJ_T_MIN=${ABFLOW_TRAJ_T_MIN:-0.05}
TRAJ_T_MAX=${ABFLOW_TRAJ_T_MAX:-0.80}

SATC_APPLY_PROB=${ABFLOW_SATC_APPLY_PROB:-0.50}
SATC_GAMMA_SCALE=${ABFLOW_SATC_GAMMA_SCALE:-0.08}
SATC_TUBE_MODE=${ABFLOW_SATC_TUBE_MODE:-legacy_absolute}
SATC_TRANSPORT_RMS_MIN=${ABFLOW_SATC_TRANSPORT_RMS_MIN:-0.25}
SATC_TRANSPORT_RMS_MAX=${ABFLOW_SATC_TRANSPORT_RMS_MAX:-20.0}
SATC_GAMMA_ABS_MAX=${ABFLOW_SATC_GAMMA_ABS_MAX:-0.50}
SATC_PROJECTION_BOUND_MODE=${ABFLOW_SATC_PROJECTION_BOUND_MODE:-legacy_tanh}
SATC_MAGNITUDE_LOSS_MODE=${ABFLOW_SATC_MAGNITUDE_LOSS_MODE:-legacy_tanh}
SATC_SCORE_WEIGHT=${ABFLOW_SATC_SCORE_WEIGHT:-0.02}
SATC_VELOCITY_WEIGHT=${ABFLOW_SATC_VELOCITY_WEIGHT:-0.003}
SATC_NT_MIN_PULL=${ABFLOW_SATC_NT_MIN_PULL:-0.15}
SATC_NT_PULL_CLIP=${ABFLOW_SATC_NT_PULL_CLIP:-2.0}
SATC_T_MIN=${ABFLOW_SATC_T_MIN:-0.10}
SATC_T_MAX=${ABFLOW_SATC_T_MAX:-0.80}
SATC_INTERFACE_WEIGHT_ALPHA=${ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA:-0.0}
SATC_INTERFACE_CUTOFF=${ABFLOW_SATC_INTERFACE_CUTOFF:-8.0}
SATC_INTERFACE_TEMPERATURE=${ABFLOW_SATC_INTERFACE_TEMPERATURE:-1.0}
SATC_INTERFACE_NORMALIZE=${ABFLOW_SATC_INTERFACE_NORMALIZE:-on}
SATC_SCHEDULE=${ABFLOW_SATC_SCHEDULE:-constant}
SATC_STEPS_PER_EPOCH=${ABFLOW_SATC_STEPS_PER_EPOCH:-52}
SATC_DECAY_START_EPOCH=${ABFLOW_SATC_DECAY_START_EPOCH:-100}
SATC_DECAY_END_EPOCH=${ABFLOW_SATC_DECAY_END_EPOCH:-130}
SATC_PERTURB_FINAL_SCALE=${ABFLOW_SATC_PERTURB_FINAL_SCALE:-1.0}
SATC_SCORE_FINAL_SCALE=${ABFLOW_SATC_SCORE_FINAL_SCALE:-1.0}
SATC_VELOCITY_FINAL_SCALE=${ABFLOW_SATC_VELOCITY_FINAL_SCALE:-1.0}
SATC_GT_INTERVAL=${ABFLOW_SATC_GT_INTERVAL:-4}
SATC_GT_START_EPOCH=${ABFLOW_SATC_GT_START_EPOCH:-5}
SAMPLE_N_STEPS=${ABFLOW_SAMPLE_N_STEPS:-}

AMP=${ABFLOW_AMP:-on}
AMP_DTYPE=${ABFLOW_AMP_DTYPE:-bf16}
ALLOW_TF32=${ABFLOW_ALLOW_TF32:-on}

NUM_WORKERS=${ABFLOW_NUM_WORKERS:-8}
PREFETCH_FACTOR=${ABFLOW_PREFETCH_FACTOR:-4}
VALID_NUM_WORKERS=${ABFLOW_VALID_NUM_WORKERS:-2}
VALID_PREFETCH_FACTOR=${ABFLOW_VALID_PREFETCH_FACTOR:-2}
VALID_PERSISTENT_WORKERS=${ABFLOW_VALID_PERSISTENT_WORKERS:-off}

LOG_INTERVAL=${ABFLOW_LOG_INTERVAL:-1}
TQDM_MININTERVAL=${ABFLOW_TQDM_MININTERVAL:-5.0}
SAVE_INTERVAL=${ABFLOW_SAVE_INTERVAL:-1}
CONDITION_DIAGNOSTICS=${ABFLOW_CONDITION_DIAGNOSTICS:-on}
DIAGNOSTIC_FILE=${ABFLOW_DIAGNOSTIC_FILE:-on}
DIAGNOSTIC_FILE_INTERVAL=${ABFLOW_DIAGNOSTIC_FILE_INTERVAL:-0}
DIAGNOSTIC_VALID_INTERVAL=${ABFLOW_DIAGNOSTIC_VALID_INTERVAL:-1}
GRAD_CONFLICT_DIAGNOSTICS=${ABFLOW_GRAD_CONFLICT_DIAGNOSTICS:-on}
GRAD_DIAGNOSTIC_INTERVAL=${ABFLOW_GRAD_DIAGNOSTIC_INTERVAL:-0}
MAX_EPOCH=${ABFLOW_MAX_EPOCH:-}
FORCE_SCRATCH=${ABFLOW_FORCE_SCRATCH:-off}

ABLATION_PARENT=PCS_RC_LC_R1
EXPERIMENT_FACTOR=unassigned
SINGLE_FACTOR_ABLATION=true
MODULE_ID=unassigned
MODULE_PARENT=unassigned

case "$EXP_ID" in
  PCS_RC_LC_R1_STRUCT_GLOBAL_ENDPOINT|S01_STRUCT_GLOBAL_ENDPOINT)
    ABLATION_PARENT=PCS_RC_LC_R1
    EXPERIMENT_FACTOR=primary_structured_global_state
    MODULE_ID=S01_STRUCT_GLOBAL_ENDPOINT
    MODULE_PARENT=HISTORICAL_PCS_RC_LC_R1
    SINGLE_FACTOR_ABLATION=true

    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=structured_global_endpoint
    T_SAMPLING=uniform
    SAMPLER_MODE=bridge
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    SHADOW_SEQ_STATE=off
    DUAL_SEQUENCE_STATE=off
    DUAL_SEQUENCE_ATOM_MODE=hidden_only
    SEQUENCE_CONTEXT_MODE=legacy
    FINAL_READOUT_MODE=integrated_endpoint
    SEQUENCE_DECODE_MODE=argmax
    DETERMINISTIC_VALIDATION=on
    PROPOSAL_ADAPTER_START_ROUND=${ABFLOW_PROPOSAL_ADAPTER_START_ROUND:-1}
    PAIR_TIME_SCOPE=off

    STRUCTURED_GAMMA_SCALE=${ABFLOW_STRUCTURED_GAMMA_SCALE:-0.05}
    STRUCTURED_TRANSPORT_MAX=${ABFLOW_STRUCTURED_TRANSPORT_MAX:-20.0}
    STRUCTURED_GAMMA_ABS_MAX=${ABFLOW_STRUCTURED_GAMMA_ABS_MAX:-1.0}
    STRUCTURED_LOCAL_GAMMA_SCALE=${ABFLOW_STRUCTURED_LOCAL_GAMMA_SCALE:-0.05}

    SATC_APPLY_PROB=0.0
    SATC_SCORE_WEIGHT=0.0
    SATC_VELOCITY_WEIGHT=0.0
    SATC_INTERFACE_WEIGHT_ALPHA=0.0
    ;;

  PCS_RC_LC_R1_STRUCT_GLOBAL_CFM|S02_STRUCT_GLOBAL_CFM)
    ABLATION_PARENT=PCS_RC_LC_R1_STRUCT_GLOBAL_ENDPOINT
    EXPERIMENT_FACTOR=path_consistent_primary_cfm_target
    MODULE_ID=S02_STRUCT_GLOBAL_CFM
    MODULE_PARENT=S01_STRUCT_GLOBAL_ENDPOINT
    SINGLE_FACTOR_ABLATION=true

    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=structured_global_cfm
    T_SAMPLING=uniform
    SAMPLER_MODE=bridge
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    SHADOW_SEQ_STATE=off
    DUAL_SEQUENCE_STATE=off
    DUAL_SEQUENCE_ATOM_MODE=hidden_only
    SEQUENCE_CONTEXT_MODE=legacy
    FINAL_READOUT_MODE=integrated_endpoint
    SEQUENCE_DECODE_MODE=argmax
    DETERMINISTIC_VALIDATION=on
    PROPOSAL_ADAPTER_START_ROUND=${ABFLOW_PROPOSAL_ADAPTER_START_ROUND:-1}
    PAIR_TIME_SCOPE=off

    STRUCTURED_GAMMA_SCALE=${ABFLOW_STRUCTURED_GAMMA_SCALE:-0.05}
    STRUCTURED_TRANSPORT_MAX=${ABFLOW_STRUCTURED_TRANSPORT_MAX:-20.0}
    STRUCTURED_GAMMA_ABS_MAX=${ABFLOW_STRUCTURED_GAMMA_ABS_MAX:-1.0}
    STRUCTURED_LOCAL_GAMMA_SCALE=${ABFLOW_STRUCTURED_LOCAL_GAMMA_SCALE:-0.05}

    SATC_APPLY_PROB=0.0
    SATC_SCORE_WEIGHT=0.0
    SATC_VELOCITY_WEIGHT=0.0
    SATC_INTERFACE_WEIGHT_ALPHA=0.0
    ;;

  PCS_RC_LC_R1_STRUCT_MULTISCALE_CFM|S03_STRUCT_MULTISCALE_CFM)
    # S02 + one orthogonal local deformation subspace.
    # Local deformation is target-aligned, residue-rigid and zero-centroid.
    ABLATION_PARENT=PCS_RC_LC_R1_STRUCT_GLOBAL_CFM
    EXPERIMENT_FACTOR=orthogonal_local_structured_path
    MODULE_ID=S03_STRUCT_MULTISCALE_CFM
    MODULE_PARENT=S02_STRUCT_GLOBAL_CFM
    SINGLE_FACTOR_ABLATION=true

    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=structured_multiscale_cfm
    T_SAMPLING=uniform
    SAMPLER_MODE=bridge
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    SHADOW_SEQ_STATE=off
    DUAL_SEQUENCE_STATE=off
    DUAL_SEQUENCE_ATOM_MODE=hidden_only
    SEQUENCE_CONTEXT_MODE=legacy
    FINAL_READOUT_MODE=integrated_endpoint
    SEQUENCE_DECODE_MODE=argmax
    DETERMINISTIC_VALIDATION=on
    PROPOSAL_ADAPTER_START_ROUND=${ABFLOW_PROPOSAL_ADAPTER_START_ROUND:-1}
    PAIR_TIME_SCOPE=off

    STRUCTURED_GAMMA_SCALE=${ABFLOW_STRUCTURED_GAMMA_SCALE:-0.05}
    STRUCTURED_TRANSPORT_MAX=${ABFLOW_STRUCTURED_TRANSPORT_MAX:-20.0}
    STRUCTURED_GAMMA_ABS_MAX=${ABFLOW_STRUCTURED_GAMMA_ABS_MAX:-1.0}
    STRUCTURED_LOCAL_GAMMA_SCALE=${ABFLOW_STRUCTURED_LOCAL_GAMMA_SCALE:-0.05}

    SATC_APPLY_PROB=0.0
    SATC_SCORE_WEIGHT=0.0
    SATC_VELOCITY_WEIGHT=0.0
    SATC_INTERFACE_WEIGHT_ALPHA=0.0
    ;;

  *)
    echo "Unknown EXP_ID: $EXP_ID"
    echo "Supported:"
    echo "  PCS_RC_LC_R1_STRUCT_GLOBAL_ENDPOINT"
    echo "  PCS_RC_LC_R1_STRUCT_GLOBAL_CFM"
    echo "  PCS_RC_LC_R1_STRUCT_MULTISCALE_CFM"
    exit 2
    ;;
esac

run_with_env() {
  ABFLOW_ABLATION_PARENT="$ABLATION_PARENT" \
  ABFLOW_EXPERIMENT_FACTOR="$EXPERIMENT_FACTOR" \
  ABFLOW_SINGLE_FACTOR_ABLATION="$SINGLE_FACTOR_ABLATION" \
  ABFLOW_MODULE_ID="$MODULE_ID" \
  ABFLOW_MODULE_PARENT="$MODULE_PARENT" \
  ABFLOW_SOURCE_MODE="$SOURCE_MODE" \
  ABFLOW_RECURRENT_PROPOSAL_CONTEXT="$RECURRENT_PROPOSAL_CONTEXT" \
  ABFLOW_COORD_PEP_SOURCE_WEIGHT="$COORD_PEP_SOURCE_WEIGHT" \
  ABFLOW_SEQ_PEP_SOURCE_WEIGHT="$SEQ_PEP_SOURCE_WEIGHT" \
  ABFLOW_SCOREFM_STATE_PATH="$STATE_PATH" \
  ABFLOW_SCOREFM_PER_SAMPLE_T="$PER_SAMPLE_T" \
  ABFLOW_SCOREFM_TIME_EMBED="$TIME_EMBED" \
  ABFLOW_PAIR_TIME_SCOPE="$PAIR_TIME_SCOPE" \
  ABFLOW_SCOREFM_T_SAMPLING="$T_SAMPLING" \
  ABFLOW_SCOREFM_LOSS_MODE="$LOSS_MODE" \
  ABFLOW_SCOREFM_MIN_SIGMA="$MIN_SIGMA" \
  ABFLOW_SCOREFM_DSM_T_MIN="$DSM_T_MIN" \
  ABFLOW_SCOREFM_DSM_T_MAX="$DSM_T_MAX" \
  ABFLOW_SCOREFM_SAMPLER_MODE="$SAMPLER_MODE" \
  ABFLOW_COORD_PEP_AS_CONDITION="$COORD_PEP_AS_CONDITION" \
  ABFLOW_SEQ_INPUT_MODE="$SEQ_INPUT_MODE" \
  ABFLOW_SHADOW_SEQ_STATE="$SHADOW_SEQ_STATE" \
  ABFLOW_DUAL_SEQUENCE_STATE="$DUAL_SEQUENCE_STATE" \
  ABFLOW_DUAL_SEQUENCE_ATOM_MODE="$DUAL_SEQUENCE_ATOM_MODE" \
  ABFLOW_SEQUENCE_CONTEXT_MODE="$SEQUENCE_CONTEXT_MODE" \
  ABFLOW_FINAL_READOUT_MODE="$FINAL_READOUT_MODE" \
  ABFLOW_SEQUENCE_DECODE_MODE="$SEQUENCE_DECODE_MODE" \
  ABFLOW_DETERMINISTIC_VALIDATION="$DETERMINISTIC_VALIDATION" \
  ABFLOW_SEQ_CE_WEIGHT="$SEQ_CE_WEIGHT" \
  ABFLOW_PROPOSAL_ADAPTER_START_ROUND="$PROPOSAL_ADAPTER_START_ROUND" \
  ABFLOW_SI_GAMMA_SCALE="$SI_GAMMA_SCALE" \
  ABFLOW_SI_SCORE_WEIGHT="$SI_SCORE_WEIGHT" \
  ABFLOW_SI_VELOCITY_WEIGHT="$SI_VELOCITY_WEIGHT" \
  ABFLOW_STRUCTURED_GAMMA_SCALE="$STRUCTURED_GAMMA_SCALE" \
  ABFLOW_STRUCTURED_TRANSPORT_MAX="$STRUCTURED_TRANSPORT_MAX" \
  ABFLOW_STRUCTURED_GAMMA_ABS_MAX="$STRUCTURED_GAMMA_ABS_MAX" \
  ABFLOW_STRUCTURED_LOCAL_GAMMA_SCALE="$STRUCTURED_LOCAL_GAMMA_SCALE" \
  ABFLOW_TRAJ_CONSISTENCY_WEIGHT="$TRAJ_CONSISTENCY_WEIGHT" \
  ABFLOW_TRAJ_VELOCITY_WEIGHT="$TRAJ_VELOCITY_WEIGHT" \
  ABFLOW_TRAJ_DELTA_T="$TRAJ_DELTA_T" \
  ABFLOW_TRAJ_T_MIN="$TRAJ_T_MIN" \
  ABFLOW_TRAJ_T_MAX="$TRAJ_T_MAX" \
  ABFLOW_SATC_APPLY_PROB="$SATC_APPLY_PROB" \
  ABFLOW_SATC_GAMMA_SCALE="$SATC_GAMMA_SCALE" \
  ABFLOW_SATC_TUBE_MODE="$SATC_TUBE_MODE" \
  ABFLOW_SATC_TRANSPORT_RMS_MIN="$SATC_TRANSPORT_RMS_MIN" \
  ABFLOW_SATC_TRANSPORT_RMS_MAX="$SATC_TRANSPORT_RMS_MAX" \
  ABFLOW_SATC_GAMMA_ABS_MAX="$SATC_GAMMA_ABS_MAX" \
  ABFLOW_SATC_PROJECTION_BOUND_MODE="$SATC_PROJECTION_BOUND_MODE" \
  ABFLOW_SATC_MAGNITUDE_LOSS_MODE="$SATC_MAGNITUDE_LOSS_MODE" \
  ABFLOW_SATC_SCORE_WEIGHT="$SATC_SCORE_WEIGHT" \
  ABFLOW_SATC_VELOCITY_WEIGHT="$SATC_VELOCITY_WEIGHT" \
  ABFLOW_SATC_NT_MIN_PULL="$SATC_NT_MIN_PULL" \
  ABFLOW_SATC_NT_PULL_CLIP="$SATC_NT_PULL_CLIP" \
  ABFLOW_SATC_T_MIN="$SATC_T_MIN" \
  ABFLOW_SATC_T_MAX="$SATC_T_MAX" \
  ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA="$SATC_INTERFACE_WEIGHT_ALPHA" \
  ABFLOW_SATC_INTERFACE_CUTOFF="$SATC_INTERFACE_CUTOFF" \
  ABFLOW_SATC_INTERFACE_TEMPERATURE="$SATC_INTERFACE_TEMPERATURE" \
  ABFLOW_SATC_INTERFACE_NORMALIZE="$SATC_INTERFACE_NORMALIZE" \
  ABFLOW_SATC_SCHEDULE="$SATC_SCHEDULE" \
  ABFLOW_SATC_STEPS_PER_EPOCH="$SATC_STEPS_PER_EPOCH" \
  ABFLOW_SATC_DECAY_START_EPOCH="$SATC_DECAY_START_EPOCH" \
  ABFLOW_SATC_DECAY_END_EPOCH="$SATC_DECAY_END_EPOCH" \
  ABFLOW_SATC_PERTURB_FINAL_SCALE="$SATC_PERTURB_FINAL_SCALE" \
  ABFLOW_SATC_SCORE_FINAL_SCALE="$SATC_SCORE_FINAL_SCALE" \
  ABFLOW_SATC_VELOCITY_FINAL_SCALE="$SATC_VELOCITY_FINAL_SCALE" \
  ABFLOW_SATC_GT_INTERVAL="$SATC_GT_INTERVAL" \
  ABFLOW_SATC_GT_START_EPOCH="$SATC_GT_START_EPOCH" \
  ABFLOW_SAMPLE_N_STEPS="$SAMPLE_N_STEPS" \
  ABFLOW_CONDITION_DIAGNOSTICS="$CONDITION_DIAGNOSTICS" \
  ABFLOW_DIAGNOSTIC_FILE="$DIAGNOSTIC_FILE" \
  ABFLOW_DIAGNOSTIC_FILE_INTERVAL="$DIAGNOSTIC_FILE_INTERVAL" \
  ABFLOW_DIAGNOSTIC_VALID_INTERVAL="$DIAGNOSTIC_VALID_INTERVAL" \
  ABFLOW_GRAD_CONFLICT_DIAGNOSTICS="$GRAD_CONFLICT_DIAGNOSTICS" \
  ABFLOW_GRAD_DIAGNOSTIC_INTERVAL="$GRAD_DIAGNOSTIC_INTERVAL" \
  ABFLOW_MAX_EPOCH="$MAX_EPOCH" \
  ABFLOW_FORCE_SCRATCH="$FORCE_SCRATCH" \
  ABFLOW_AMP="$AMP" \
  ABFLOW_AMP_DTYPE="$AMP_DTYPE" \
  ABFLOW_ALLOW_TF32="$ALLOW_TF32" \
  ABFLOW_NUM_WORKERS="$NUM_WORKERS" \
  ABFLOW_PREFETCH_FACTOR="$PREFETCH_FACTOR" \
  ABFLOW_VALID_NUM_WORKERS="$VALID_NUM_WORKERS" \
  ABFLOW_VALID_PREFETCH_FACTOR="$VALID_PREFETCH_FACTOR" \
  ABFLOW_VALID_PERSISTENT_WORKERS="$VALID_PERSISTENT_WORKERS" \
  ABFLOW_LOG_INTERVAL="$LOG_INTERVAL" \
  ABFLOW_TQDM_MININTERVAL="$TQDM_MININTERVAL" \
  ABFLOW_SAVE_INTERVAL="$SAVE_INTERVAL" \
  GPU="$GPU_ID" \
  "$@"
}

print_settings() {
  echo "Experiment: $EXP_ID"
  echo "ABLATION_PARENT=$ABLATION_PARENT"
  echo "EXPERIMENT_FACTOR=$EXPERIMENT_FACTOR"
  echo "SINGLE_FACTOR_ABLATION=$SINGLE_FACTOR_ABLATION"
  echo "MODULE_ID=$MODULE_ID"
  echo "MODULE_PARENT=$MODULE_PARENT"
  echo "GPU: $GPU_ID"
  echo "SOURCE_MODE=$SOURCE_MODE"
  echo "RECURRENT_PROPOSAL_CONTEXT=$RECURRENT_PROPOSAL_CONTEXT"
  echo "COORD_PEP_SOURCE_WEIGHT=$COORD_PEP_SOURCE_WEIGHT"
  echo "SEQ_PEP_SOURCE_WEIGHT=$SEQ_PEP_SOURCE_WEIGHT"
  echo "STATE_PATH=$STATE_PATH"
  echo "PER_SAMPLE_T=$PER_SAMPLE_T"
  echo "TIME_EMBED=$TIME_EMBED"
  echo "PAIR_TIME_SCOPE=$PAIR_TIME_SCOPE"
  echo "T_SAMPLING=$T_SAMPLING"
  echo "LOSS_MODE=$LOSS_MODE"
  echo "MIN_SIGMA=$MIN_SIGMA"
  echo "DSM_T_MIN=$DSM_T_MIN"
  echo "DSM_T_MAX=$DSM_T_MAX"
  echo "SAMPLER_MODE=$SAMPLER_MODE"
  echo "COORD_PEP_AS_CONDITION=$COORD_PEP_AS_CONDITION"
  echo "SEQ_INPUT_MODE=$SEQ_INPUT_MODE"
  echo "SHADOW_SEQ_STATE=$SHADOW_SEQ_STATE"
  echo "DUAL_SEQUENCE_STATE=$DUAL_SEQUENCE_STATE"
  echo "DUAL_SEQUENCE_ATOM_MODE=$DUAL_SEQUENCE_ATOM_MODE"
  echo "SEQUENCE_CONTEXT_MODE=$SEQUENCE_CONTEXT_MODE"
  echo "FINAL_READOUT_MODE=$FINAL_READOUT_MODE"
  echo "SEQUENCE_DECODE_MODE=$SEQUENCE_DECODE_MODE"
  echo "DETERMINISTIC_VALIDATION=$DETERMINISTIC_VALIDATION"
  echo "SEQ_CE_WEIGHT=$SEQ_CE_WEIGHT"
  echo "PROPOSAL_ADAPTER_START_ROUND=$PROPOSAL_ADAPTER_START_ROUND"
  echo "SI_GAMMA_SCALE=$SI_GAMMA_SCALE"
  echo "SI_SCORE_WEIGHT=$SI_SCORE_WEIGHT"
  echo "SI_VELOCITY_WEIGHT=$SI_VELOCITY_WEIGHT"
  echo "STRUCTURED_GAMMA_SCALE=$STRUCTURED_GAMMA_SCALE"
  echo "STRUCTURED_TRANSPORT_MAX=$STRUCTURED_TRANSPORT_MAX"
  echo "STRUCTURED_GAMMA_ABS_MAX=$STRUCTURED_GAMMA_ABS_MAX"
  echo "STRUCTURED_LOCAL_GAMMA_SCALE=$STRUCTURED_LOCAL_GAMMA_SCALE"
  echo "TRAJ_CONSISTENCY_WEIGHT=$TRAJ_CONSISTENCY_WEIGHT"
  echo "TRAJ_VELOCITY_WEIGHT=$TRAJ_VELOCITY_WEIGHT"
  echo "TRAJ_DELTA_T=$TRAJ_DELTA_T"
  echo "TRAJ_T_MIN=$TRAJ_T_MIN"
  echo "TRAJ_T_MAX=$TRAJ_T_MAX"
  echo "SATC_APPLY_PROB=$SATC_APPLY_PROB"
  echo "SATC_GAMMA_SCALE=$SATC_GAMMA_SCALE"
  echo "SATC_TUBE_MODE=$SATC_TUBE_MODE"
  echo "SATC_TRANSPORT_RMS_MIN=$SATC_TRANSPORT_RMS_MIN"
  echo "SATC_TRANSPORT_RMS_MAX=$SATC_TRANSPORT_RMS_MAX"
  echo "SATC_GAMMA_ABS_MAX=$SATC_GAMMA_ABS_MAX"
  echo "SATC_PROJECTION_BOUND_MODE=$SATC_PROJECTION_BOUND_MODE"
  echo "SATC_MAGNITUDE_LOSS_MODE=$SATC_MAGNITUDE_LOSS_MODE"
  echo "SATC_SCORE_WEIGHT=$SATC_SCORE_WEIGHT"
  echo "SATC_VELOCITY_WEIGHT=$SATC_VELOCITY_WEIGHT"
  echo "SATC_NT_MIN_PULL=$SATC_NT_MIN_PULL"
  echo "SATC_NT_PULL_CLIP=$SATC_NT_PULL_CLIP"
  echo "SATC_T_MIN=$SATC_T_MIN"
  echo "SATC_T_MAX=$SATC_T_MAX"
  echo "SATC_INTERFACE_WEIGHT_ALPHA=$SATC_INTERFACE_WEIGHT_ALPHA"
  echo "SATC_INTERFACE_CUTOFF=$SATC_INTERFACE_CUTOFF"
  echo "SATC_INTERFACE_TEMPERATURE=$SATC_INTERFACE_TEMPERATURE"
  echo "SATC_INTERFACE_NORMALIZE=$SATC_INTERFACE_NORMALIZE"
  echo "SATC_SCHEDULE=$SATC_SCHEDULE"
  echo "SATC_STEPS_PER_EPOCH=$SATC_STEPS_PER_EPOCH"
  echo "SATC_DECAY_START_EPOCH=$SATC_DECAY_START_EPOCH"
  echo "SATC_DECAY_END_EPOCH=$SATC_DECAY_END_EPOCH"
  echo "SATC_PERTURB_FINAL_SCALE=$SATC_PERTURB_FINAL_SCALE"
  echo "SATC_SCORE_FINAL_SCALE=$SATC_SCORE_FINAL_SCALE"
  echo "SATC_VELOCITY_FINAL_SCALE=$SATC_VELOCITY_FINAL_SCALE"
  echo "SATC_GT_INTERVAL=$SATC_GT_INTERVAL"
  echo "SATC_GT_START_EPOCH=$SATC_GT_START_EPOCH"
  echo "SAMPLE_N_STEPS=${SAMPLE_N_STEPS:-<model/default>}"
  echo "AMP=$AMP"
  echo "AMP_DTYPE=$AMP_DTYPE"
  echo "ALLOW_TF32=$ALLOW_TF32"
  echo "NUM_WORKERS=$NUM_WORKERS"
  echo "PREFETCH_FACTOR=$PREFETCH_FACTOR"
  echo "VALID_NUM_WORKERS=$VALID_NUM_WORKERS"
  echo "VALID_PREFETCH_FACTOR=$VALID_PREFETCH_FACTOR"
  echo "VALID_PERSISTENT_WORKERS=$VALID_PERSISTENT_WORKERS"
  echo "LOG_INTERVAL=$LOG_INTERVAL"
  echo "TQDM_MININTERVAL=$TQDM_MININTERVAL"
  echo "SAVE_INTERVAL=$SAVE_INTERVAL"
  echo "MAX_EPOCH=${MAX_EPOCH:-<base_config>}"
  echo "FORCE_SCRATCH=$FORCE_SCRATCH"
  echo "CONDITION_DIAGNOSTICS=$CONDITION_DIAGNOSTICS"
  echo "DIAGNOSTIC_FILE=$DIAGNOSTIC_FILE"
  echo "DIAGNOSTIC_FILE_INTERVAL=$DIAGNOSTIC_FILE_INTERVAL"
  echo "DIAGNOSTIC_VALID_INTERVAL=$DIAGNOSTIC_VALID_INTERVAL"
  echo "GRAD_CONFLICT_DIAGNOSTICS=$GRAD_CONFLICT_DIAGNOSTICS"
  echo "GRAD_DIAGNOSTIC_INTERVAL=$GRAD_DIAGNOSTIC_INTERVAL"
}


# ============================================================
# Automatic topk-map test evaluation helpers
# ============================================================
# Principle:
#   The original training command remains unchanged:
#     bash scripts/train/run_gt_satc_matched_v55.sh train <EXP_ID> <GPU_ID> <BASE_CONFIG>
#   When ABFLOW_AUTO_TOPK_EVAL=on (explicit opt-in), the launcher automatically starts
#   a background watcher if spare GPUs are available.  It reads topk_map.txt,
#   evaluates new checkpoints with the original test pipeline and writes CSV
#   next to topk_map.txt.  If no spare GPU is available, the launcher performs
#   a final catch-up evaluation after training finishes.

PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-/home/data3/cjm/project/AbFlow}
AUTO_TOPK_EVAL=${ABFLOW_AUTO_TOPK_EVAL:-off}
AUTO_TOPK_POLL_INTERVAL=${ABFLOW_TOPK_POLL_INTERVAL:-300}
AUTO_TOPK_MAX_NEW=${ABFLOW_TOPK_MAX_NEW:-1}
AUTO_TOPK_LATEST_ONLY=${ABFLOW_TOPK_LATEST_ONLY:-off}
AUTO_TOPK_MAX_EVAL_GPUS=${ABFLOW_AUTO_TOPK_MAX_EVAL_GPUS:-1}
AUTO_TOPK_TEST_JSON=${ABFLOW_TOPK_TEST_JSON:-${PROJECT_ROOT}/datasets/RAbD/test.json}
AUTO_TOPK_EVAL_SCRIPT=${ABFLOW_TOPK_EVAL_SCRIPT:-scripts/test/evaluate_topk_map.py}
AUTO_TOPK_FORCE=${ABFLOW_TOPK_FORCE:-off}

_is_on() {
  local v="${1:-off}"
  case "${v,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

_csv_contains() {
  local csv=",$1,"
  local item="$2"
  [[ "$csv" == *",${item},"* ]]
}

_infer_spare_eval_gpus() {
  if [[ -n "${ABFLOW_EVAL_GPUS:-}" ]]; then
    echo "$ABFLOW_EVAL_GPUS"
    return 0
  fi
  if [[ -n "${ABFLOW_EVAL_GPU:-}" ]]; then
    echo "$ABFLOW_EVAL_GPU"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  local all_ids train_ids selected id count
  all_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//') || all_ids=""
  train_ids="$GPU_ID"
  selected=""
  count=0
  IFS=',' read -ra ids <<< "$all_ids"
  for id in "${ids[@]}"; do
    id=$(echo "$id" | xargs)
    [[ -z "$id" ]] && continue
    if _csv_contains "$train_ids" "$id"; then
      continue
    fi
    if [[ -z "$selected" ]]; then
      selected="$id"
    else
      selected="${selected},${id}"
    fi
    count=$((count + 1))
    if [[ "$count" -ge "$AUTO_TOPK_MAX_EVAL_GPUS" ]]; then
      break
    fi
  done
  # normalize accidental leading pattern if any
  selected=$(echo "$selected" | sed 's/^,//;s/,,*/,/g')
  echo "$selected"
}

_start_auto_topk_watcher() {
  local eval_gpus="$1"
  local run_dir="$2"
  local log_file="$run_dir/auto_topk_eval.log"
  local pid_file="$run_dir/auto_topk_eval.pid"

  if ! _is_on "$AUTO_TOPK_EVAL"; then
    echo "[AutoTopK] disabled by ABFLOW_AUTO_TOPK_EVAL=$AUTO_TOPK_EVAL"
    return 0
  fi
  if [[ -z "$eval_gpus" ]]; then
    echo "[AutoTopK] no spare GPU inferred; watcher not started during training. Final catch-up evaluation will run after training."
    return 0
  fi
  if [[ ! -f "$AUTO_TOPK_TEST_JSON" ]]; then
    echo "[AutoTopK] test json not found: $AUTO_TOPK_TEST_JSON; auto evaluation disabled."
    return 0
  fi
  if [[ ! -f "$AUTO_TOPK_EVAL_SCRIPT" ]]; then
    echo "[AutoTopK] evaluator script not found: $AUTO_TOPK_EVAL_SCRIPT; auto evaluation disabled."
    return 0
  fi
  mkdir -p "$run_dir"
  echo "[AutoTopK] starting watcher: exp=$EXP_ID eval_gpus=$eval_gpus run_dir=$run_dir" | tee -a "$log_file"
  (
    python "$AUTO_TOPK_EVAL_SCRIPT" \
      --exp-id "$EXP_ID" \
      --run-dir "$run_dir" \
      --test-json "$AUTO_TOPK_TEST_JSON" \
      --gpu-ids "$eval_gpus" \
      --project-root "$PROJECT_ROOT" \
      --launcher "$SELF_LAUNCHER" \
      --watch \
      --poll-interval "$AUTO_TOPK_POLL_INTERVAL" \
      --max-new "$AUTO_TOPK_MAX_NEW" \
      $( _is_on "$AUTO_TOPK_LATEST_ONLY" && echo --latest-only ) \
      $( _is_on "$AUTO_TOPK_FORCE" && echo --force )
  ) >> "$log_file" 2>&1 &
  echo $! > "$pid_file"
  echo "[AutoTopK] watcher pid=$(cat "$pid_file") log=$log_file"
}

_stop_auto_topk_watcher() {
  local run_dir="$1"
  local pid_file="$run_dir/auto_topk_eval.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(cat "$pid_file" || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "[AutoTopK] stopping watcher pid=$pid"
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

_run_auto_topk_once() {
  local eval_gpus="$1"
  local run_dir="$2"
  local log_file="$run_dir/auto_topk_eval_final.log"

  if ! _is_on "$AUTO_TOPK_EVAL"; then
    return 0
  fi
  if [[ -z "$eval_gpus" ]]; then
    eval_gpus="$GPU_ID"
  fi
  if [[ ! -f "$AUTO_TOPK_TEST_JSON" || ! -f "$AUTO_TOPK_EVAL_SCRIPT" ]]; then
    return 0
  fi
  mkdir -p "$run_dir"
  echo "[AutoTopK] final catch-up evaluation: exp=$EXP_ID eval_gpus=$eval_gpus" | tee -a "$log_file"
  python "$AUTO_TOPK_EVAL_SCRIPT" \
    --exp-id "$EXP_ID" \
    --run-dir "$run_dir" \
    --test-json "$AUTO_TOPK_TEST_JSON" \
    --gpu-ids "$eval_gpus" \
    --project-root "$PROJECT_ROOT" \
    --launcher "$SELF_LAUNCHER" \
    $( _is_on "$AUTO_TOPK_LATEST_ONLY" && echo --latest-only ) \
    $( _is_on "$AUTO_TOPK_FORCE" && echo --force ) \
    >> "$log_file" 2>&1 || true
}

if [[ "$MODE" == "attach_eval" ]]; then
  RUN_ROOT=${ABFLOW_RUN_ROOT:-/home/data3/cjm/project/AbFlow/results_dtm}
  RUN_DIR="${RUN_ROOT}/${EXP_ID}"
  EVAL_GPUS_ARG=${4:-}
  if [[ -z "$EVAL_GPUS_ARG" || "$EVAL_GPUS_ARG" == "auto" ]]; then
    EVAL_GPUS_ARG=$(_infer_spare_eval_gpus)
    [[ -z "$EVAL_GPUS_ARG" ]] && EVAL_GPUS_ARG="$GPU_ID"
  fi
  echo "[AutoTopK] attach mode: exp=$EXP_ID run_dir=$RUN_DIR eval_gpus=$EVAL_GPUS_ARG"
  python "$AUTO_TOPK_EVAL_SCRIPT" \
    --exp-id "$EXP_ID" \
    --run-dir "$RUN_DIR" \
    --test-json "$AUTO_TOPK_TEST_JSON" \
    --gpu-ids "$EVAL_GPUS_ARG" \
    --project-root "$PROJECT_ROOT" \
    --launcher "$SELF_LAUNCHER" \
    --watch \
    --poll-interval "$AUTO_TOPK_POLL_INTERVAL" \
    --max-new "$AUTO_TOPK_MAX_NEW" \
    $( _is_on "$AUTO_TOPK_LATEST_ONLY" && echo --latest-only ) \
    $( _is_on "$AUTO_TOPK_FORCE" && echo --force )
  exit 0
fi

if [[ "$MODE" == "test" ]]; then
  CKPT=${4:-}
  RESULT_DIR=${5:-}
  TEST_JSON=${6:-datasets/RAbD/test.json}

  if [[ -z "$CKPT" || -z "$RESULT_DIR" ]]; then
    echo "Usage: bash $0 test <EXP_ID> <GPU_ID> <CKPT> <RESULT_DIR> [TEST_JSON]"
    exit 2
  fi

  [[ -f "$CKPT" ]] || { echo "Checkpoint not found: $CKPT"; exit 2; }
  [[ -f "$TEST_JSON" ]] || { echo "Test JSON not found: $TEST_JSON"; exit 2; }

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
  echo "Test:  bash $0 test  <EXP_ID> <GPU_ID> <CKPT> <RESULT_DIR> [TEST_JSON]"
  echo "Attach current training auto-eval: bash $0 attach_eval <EXP_ID> <GPU_ID|auto> [EVAL_GPU_ID|auto]"
  echo "Structured-primary EXP_ID: PCS_RC_LC_R1_STRUCT_GLOBAL_ENDPOINT, PCS_RC_LC_R1_STRUCT_GLOBAL_CFM, PCS_RC_LC_R1_STRUCT_MULTISCALE_CFM"
  exit 2
fi

BASE_CONFIG=${4:-scripts/train/configs/single_cdr_design.json}
[[ -f "$BASE_CONFIG" ]] || { echo "Base config not found: $BASE_CONFIG"; exit 2; }

RUN_ROOT=${ABFLOW_RUN_ROOT:-/home/data3/cjm/project/AbFlow/results_dtm}
RUN_DIR="${RUN_ROOT}/${EXP_ID}"
CONFIG_DIR="${RUN_ROOT}/generated_configs"

# Resume/scratch mode is determined by one source of truth:
#   1) ABFLOW_FORCE_SCRATCH=on explicitly requests a fresh run and clears
#      resume_checkpoint.
#   2) Otherwise, a non-empty resume_checkpoint in BASE_CONFIG means resume.
#   3) Otherwise, this is a fresh run.
#
# A resume is allowed to reuse the existing experiment directory.  A fresh run
# is still protected from silently mixing with old checkpoints/logs.
BASE_RESUME_CHECKPOINT=$(python - "$BASE_CONFIG" <<'PYRESUME'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

print(str(cfg.get("resume_checkpoint", "") or "").strip())
PYRESUME
)

if _is_on "$FORCE_SCRATCH"; then
  RUN_MODE="scratch"
  EFFECTIVE_RESUME_CHECKPOINT=""
elif [[ -n "$BASE_RESUME_CHECKPOINT" ]]; then
  RUN_MODE="resume"
  EFFECTIVE_RESUME_CHECKPOINT="$BASE_RESUME_CHECKPOINT"
else
  RUN_MODE="scratch"
  EFFECTIVE_RESUME_CHECKPOINT=""
fi

if [[ "$RUN_MODE" == "resume" ]]; then
  if [[ ! -f "$EFFECTIVE_RESUME_CHECKPOINT" ]]; then
    echo "ERROR: resume checkpoint not found: $EFFECTIVE_RESUME_CHECKPOINT" >&2
    exit 2
  fi

  EXPECTED_PREFIX="${RUN_DIR}/version_"
  case "$(realpath "$EFFECTIVE_RESUME_CHECKPOINT")" in
    "${EXPECTED_PREFIX}"*/checkpoint/last_step*.pt)
      ;;
    *)
      echo "ERROR: resume checkpoint must be a last_step*.pt under:" >&2
      echo "       ${RUN_DIR}/version_N/checkpoint/" >&2
      echo "Got:   $EFFECTIVE_RESUME_CHECKPOINT" >&2
      exit 2
      ;;
  esac
else
  if [[ -d "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    if ! _is_on "${ABFLOW_ALLOW_NONEMPTY_RUN_DIR:-off}"; then
      echo "ERROR: fresh-run directory is not empty: $RUN_DIR" >&2
      echo "Set resume_checkpoint in BASE_CONFIG to resume, or use a new ABFLOW_RUN_ROOT." >&2
      echo "ABFLOW_ALLOW_NONEMPTY_RUN_DIR=on is only for an intentional fresh overwrite." >&2
      exit 2
    fi
  fi
fi

echo "Run mode: $RUN_MODE"
echo "Effective resume checkpoint: ${EFFECTIVE_RESUME_CHECKPOINT:-<none>}"

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

# v38 self-documenting configs may contain a private metadata block.
# It is validated here and removed before train.sh converts JSON keys to CLI flags.
meta = cfg.pop("_experiment", None)
if isinstance(meta, dict):
    expected = str(meta.get("exp_id", "") or "").strip()
    if expected and expected != exp_id:
        raise ValueError(
            f"Config/EXP_ID mismatch: config expects {expected}, launcher got {exp_id}"
        )

# train.sh converts every top-level JSON key into --<key>.
# Therefore uppercase keys such as VALID_NUM_WORKERS become
# --VALID_NUM_WORKERS, which train.py does not accept.
for key in list(cfg.keys()):
    if key.isupper():
        cfg.pop(key, None)

for key in [
    "VALID_NUM_WORKERS",
    "VALID_PREFETCH_FACTOR",
    "VALID_PERSISTENT_WORKERS",
    "NUM_WORKERS",
    "PREFETCH_FACTOR",
    "LOG_INTERVAL",
    "SAVE_INTERVAL",
    "CONDITION_DIAGNOSTICS",
    "SOURCE_MODE",
    "RECURRENT_PROPOSAL_CONTEXT",
    "COORD_PEP_SOURCE_WEIGHT",
    "SEQ_PEP_SOURCE_WEIGHT",
    "PROPOSAL_ADAPTER_START_ROUND",
]:
    cfg.pop(key, None)

cfg["save_dir"] = save_dir

def _env_on(name, default="off"):
    return os.environ.get(name, default).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }

def _env_int(name, default):
    return int(os.environ.get(name, str(default)))

def _env_float(name, default):
    return float(os.environ.get(name, str(default)))

# Resume logic:
#   - BASE_CONFIG resume_checkpoint is preserved by default;
#   - ABFLOW_FORCE_SCRATCH=on is the only explicit override that clears it.
# Experiment profiles must never silently force scratch mode.
force_scratch = _env_on("ABFLOW_FORCE_SCRATCH", "off")
if force_scratch:
    resume_ckpt = ""
else:
    resume_ckpt = str(cfg.get("resume_checkpoint", "") or "").strip()
cfg["resume_checkpoint"] = resume_ckpt
if resume_ckpt and not os.path.isfile(resume_ckpt):
    raise FileNotFoundError(
        f"resume_checkpoint in base config does not exist: {resume_ckpt}"
    )

cfg["num_workers"] = _env_int(
    "ABFLOW_NUM_WORKERS",
    cfg.get("num_workers", 8),
)
cfg["prefetch_factor"] = _env_int(
    "ABFLOW_PREFETCH_FACTOR",
    cfg.get("prefetch_factor", 4),
)
cfg["valid_num_workers"] = _env_int(
    "ABFLOW_VALID_NUM_WORKERS",
    cfg.get("valid_num_workers", 2),
)
cfg["valid_prefetch_factor"] = _env_int(
    "ABFLOW_VALID_PREFETCH_FACTOR",
    cfg.get("valid_prefetch_factor", 2),
)

if _env_on("ABFLOW_VALID_PERSISTENT_WORKERS", "off"):
    cfg["valid_persistent_workers"] = True
else:
    cfg.pop("valid_persistent_workers", None)

cfg["log_interval"] = _env_int(
    "ABFLOW_LOG_INTERVAL",
    cfg.get("log_interval", 1),
)
cfg["tqdm_mininterval"] = _env_float(
    "ABFLOW_TQDM_MININTERVAL",
    cfg.get("tqdm_mininterval", 5.0),
)
cfg["save_interval"] = _env_int(
    "ABFLOW_SAVE_INTERVAL",
    cfg.get("save_interval", 1),
)

max_epoch_env = os.environ.get("ABFLOW_MAX_EPOCH", "").strip()
if max_epoch_env:
    cfg["max_epoch"] = int(max_epoch_env)

amp_dtype = os.environ.get(
    "ABFLOW_AMP_DTYPE",
    str(cfg.get("amp_dtype", "bf16")),
).strip().lower()

if amp_dtype not in {"bf16", "fp16"}:
    raise ValueError("ABFLOW_AMP_DTYPE must be bf16 or fp16.")

cfg["amp_dtype"] = amp_dtype

if _env_on("ABFLOW_AMP", "on"):
    cfg["amp"] = True
else:
    cfg.pop("amp", None)

if _env_on("ABFLOW_ALLOW_TF32", "on"):
    cfg["allow_tf32"] = True
else:
    cfg.pop("allow_tf32", None)

# Remove obsolete peptide-prior and redundant objective controls.
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
    "abflow_runtime",
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
    "ablation_parent": os.environ.get("ABFLOW_ABLATION_PARENT", "PCS_RC_LC_R1"),
    "experiment_factor": os.environ.get("ABFLOW_EXPERIMENT_FACTOR", ""),
    "single_factor_ablation": os.environ.get("ABFLOW_SINGLE_FACTOR_ABLATION", "true"),
    "module_id": os.environ.get("ABFLOW_MODULE_ID", ""),
    "module_parent": os.environ.get("ABFLOW_MODULE_PARENT", ""),
    "reference_policy": "module effects vs matched internal BASE; paper gaps are external only",
    "source_mode": os.environ.get("ABFLOW_SOURCE_MODE", ""),
    "recurrent_proposal_context": os.environ.get("ABFLOW_RECURRENT_PROPOSAL_CONTEXT", ""),
    "coord_pep_source_weight": os.environ.get("ABFLOW_COORD_PEP_SOURCE_WEIGHT", ""),
    "seq_pep_source_weight": os.environ.get("ABFLOW_SEQ_PEP_SOURCE_WEIGHT", ""),
    "state_path": os.environ.get("ABFLOW_SCOREFM_STATE_PATH", ""),
    "per_sample_t": os.environ.get("ABFLOW_SCOREFM_PER_SAMPLE_T", ""),
    "time_embed": os.environ.get("ABFLOW_SCOREFM_TIME_EMBED", ""),
    "pair_time_scope": os.environ.get("ABFLOW_PAIR_TIME_SCOPE", "off"),
    "pair_time_conditioning": os.environ.get("ABFLOW_PAIR_TIME_SCOPE", "off") != "off",
    "t_sampling": os.environ.get("ABFLOW_SCOREFM_T_SAMPLING", ""),
    "loss_mode": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", ""),
    "min_sigma": os.environ.get("ABFLOW_SCOREFM_MIN_SIGMA", ""),
    "dsm_t_min": os.environ.get("ABFLOW_SCOREFM_DSM_T_MIN", ""),
    "dsm_t_max": os.environ.get("ABFLOW_SCOREFM_DSM_T_MAX", ""),
    "sampler_mode": os.environ.get("ABFLOW_SCOREFM_SAMPLER_MODE", ""),
    "coord_pep_as_condition": os.environ.get("ABFLOW_COORD_PEP_AS_CONDITION", ""),
    "seq_input_mode": os.environ.get("ABFLOW_SEQ_INPUT_MODE", ""),
    "shadow_seq_state": os.environ.get("ABFLOW_SHADOW_SEQ_STATE", ""),
    "dual_sequence_state": os.environ.get("ABFLOW_DUAL_SEQUENCE_STATE", ""),
    "dual_sequence_atom_mode": os.environ.get("ABFLOW_DUAL_SEQUENCE_ATOM_MODE", ""),
    "sequence_context_mode": os.environ.get("ABFLOW_SEQUENCE_CONTEXT_MODE", ""),
    "final_readout_mode": os.environ.get("ABFLOW_FINAL_READOUT_MODE", ""),
    "sequence_decode_mode": os.environ.get("ABFLOW_SEQUENCE_DECODE_MODE", ""),
    "deterministic_validation": os.environ.get("ABFLOW_DETERMINISTIC_VALIDATION", ""),
    "seq_ce_weight": os.environ.get("ABFLOW_SEQ_CE_WEIGHT", ""),
    "proposal_adapter_start_round": os.environ.get("ABFLOW_PROPOSAL_ADAPTER_START_ROUND", ""),
    "si_gamma_scale": os.environ.get("ABFLOW_SI_GAMMA_SCALE", ""),
    "si_score_weight": os.environ.get("ABFLOW_SI_SCORE_WEIGHT", ""),
    "si_velocity_weight": os.environ.get("ABFLOW_SI_VELOCITY_WEIGHT", ""),
    "structured_gamma_scale": os.environ.get("ABFLOW_STRUCTURED_GAMMA_SCALE", ""),
    "structured_transport_max": os.environ.get("ABFLOW_STRUCTURED_TRANSPORT_MAX", ""),
    "structured_gamma_abs_max": os.environ.get("ABFLOW_STRUCTURED_GAMMA_ABS_MAX", ""),
    "structured_local_gamma_scale": os.environ.get("ABFLOW_STRUCTURED_LOCAL_GAMMA_SCALE", ""),
    "structured_multiscale_cfm": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", "") == "structured_multiscale_cfm",
    "traj_consistency_weight": os.environ.get("ABFLOW_TRAJ_CONSISTENCY_WEIGHT", ""),
    "traj_velocity_weight": os.environ.get("ABFLOW_TRAJ_VELOCITY_WEIGHT", ""),
    "traj_delta_t": os.environ.get("ABFLOW_TRAJ_DELTA_T", ""),
    "traj_t_min": os.environ.get("ABFLOW_TRAJ_T_MIN", ""),
    "traj_t_max": os.environ.get("ABFLOW_TRAJ_T_MAX", ""),
    "satc_apply_prob": os.environ.get("ABFLOW_SATC_APPLY_PROB", ""),
    "satc_gamma_scale": os.environ.get("ABFLOW_SATC_GAMMA_SCALE", ""),
    "satc_tube_mode": os.environ.get("ABFLOW_SATC_TUBE_MODE", ""),
    "satc_transport_rms_min": os.environ.get("ABFLOW_SATC_TRANSPORT_RMS_MIN", ""),
    "satc_transport_rms_max": os.environ.get("ABFLOW_SATC_TRANSPORT_RMS_MAX", ""),
    "satc_gamma_abs_max": os.environ.get("ABFLOW_SATC_GAMMA_ABS_MAX", ""),
    "satc_projection_bound_mode": os.environ.get("ABFLOW_SATC_PROJECTION_BOUND_MODE", ""),
    "satc_magnitude_loss_mode": os.environ.get("ABFLOW_SATC_MAGNITUDE_LOSS_MODE", ""),
    "satc_score_weight": os.environ.get("ABFLOW_SATC_SCORE_WEIGHT", ""),
    "satc_velocity_weight": os.environ.get("ABFLOW_SATC_VELOCITY_WEIGHT", ""),
    "satc_nt_min_pull": os.environ.get("ABFLOW_SATC_NT_MIN_PULL", ""),
    "satc_nt_pull_clip": os.environ.get("ABFLOW_SATC_NT_PULL_CLIP", ""),
    "satc_t_min": os.environ.get("ABFLOW_SATC_T_MIN", ""),
    "satc_t_max": os.environ.get("ABFLOW_SATC_T_MAX", ""),
    "satc_interface_weight_alpha": os.environ.get("ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA", ""),
    "satc_interface_cutoff": os.environ.get("ABFLOW_SATC_INTERFACE_CUTOFF", ""),
    "satc_interface_temperature": os.environ.get("ABFLOW_SATC_INTERFACE_TEMPERATURE", ""),
    "satc_interface_normalize": os.environ.get("ABFLOW_SATC_INTERFACE_NORMALIZE", ""),
    "satc_schedule": os.environ.get("ABFLOW_SATC_SCHEDULE", ""),
    "satc_steps_per_epoch": os.environ.get("ABFLOW_SATC_STEPS_PER_EPOCH", ""),
    "satc_decay_start_epoch": os.environ.get("ABFLOW_SATC_DECAY_START_EPOCH", ""),
    "satc_decay_end_epoch": os.environ.get("ABFLOW_SATC_DECAY_END_EPOCH", ""),
    "satc_perturb_final_scale": os.environ.get("ABFLOW_SATC_PERTURB_FINAL_SCALE", ""),
    "satc_score_final_scale": os.environ.get("ABFLOW_SATC_SCORE_FINAL_SCALE", ""),
    "satc_velocity_final_scale": os.environ.get("ABFLOW_SATC_VELOCITY_FINAL_SCALE", ""),
    "satc_gt_interval": os.environ.get("ABFLOW_SATC_GT_INTERVAL", ""),
    "satc_gt_start_epoch": os.environ.get("ABFLOW_SATC_GT_START_EPOCH", ""),
    "sample_n_steps": os.environ.get("ABFLOW_SAMPLE_N_STEPS", ""),
    "amp": os.environ.get("ABFLOW_AMP", "on"),
    "amp_dtype": os.environ.get("ABFLOW_AMP_DTYPE", "bf16"),
    "allow_tf32": os.environ.get("ABFLOW_ALLOW_TF32", "on"),
    "num_workers": os.environ.get("ABFLOW_NUM_WORKERS", str(cfg.get("num_workers", 8))),
    "prefetch_factor": os.environ.get("ABFLOW_PREFETCH_FACTOR", str(cfg.get("prefetch_factor", 4))),
    "valid_num_workers": os.environ.get("ABFLOW_VALID_NUM_WORKERS", str(cfg.get("valid_num_workers", 2))),
    "valid_prefetch_factor": os.environ.get("ABFLOW_VALID_PREFETCH_FACTOR", str(cfg.get("valid_prefetch_factor", 2))),
    "valid_persistent_workers": os.environ.get("ABFLOW_VALID_PERSISTENT_WORKERS", "off"),
    "log_interval": os.environ.get("ABFLOW_LOG_INTERVAL", str(cfg.get("log_interval", 1))),
    "save_interval": os.environ.get("ABFLOW_SAVE_INTERVAL", str(cfg.get("save_interval", 1))),
    "condition_diagnostics": os.environ.get("ABFLOW_CONDITION_DIAGNOSTICS", "on"),
    "diagnostic_file": os.environ.get("ABFLOW_DIAGNOSTIC_FILE", "on"),
    "diagnostic_file_interval": os.environ.get("ABFLOW_DIAGNOSTIC_FILE_INTERVAL", "0"),
    "diagnostic_valid_interval": os.environ.get("ABFLOW_DIAGNOSTIC_VALID_INTERVAL", "1"),
    "grad_conflict_diagnostics": os.environ.get("ABFLOW_GRAD_CONFLICT_DIAGNOSTICS", "on"),
    "grad_diagnostic_interval": os.environ.get("ABFLOW_GRAD_DIAGNOSTIC_INTERVAL", "0"),
    "max_epoch": os.environ.get("ABFLOW_MAX_EPOCH", str(cfg.get("max_epoch", ""))),
    "resume_checkpoint": resume_ckpt,
    "force_scratch": force_scratch,
    "run_mode": "scratch" if force_scratch or not resume_ckpt else "resume",
    "clean_reference_source": os.environ.get("ABFLOW_SOURCE_MODE", "") == "reference",
    "proposal_conditioned_source": os.environ.get("ABFLOW_SOURCE_MODE", "") in {"pcs", "pcs_rc"},
    "proposal_recurrent_context": os.environ.get("ABFLOW_RECURRENT_PROPOSAL_CONTEXT", "") == "on",
    "peptide_state_injection": "false",
    "peptide_prior_weighting": "false",
    "independent_score_head": "false",
    "independent_velocity_head": "false",
    "stochastic_interpolant_training": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", "") in {"si_score", "si_score_fm"},
    "trajectory_consistency_training": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", "") in {"traj_consistency", "traj_consistency_fm"},
    "score_aware_trajectory_lite_training": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", "").startswith("score_aware_traj_"),
    "interface_weighted_satc": "_if_" in os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", ""),
    "coordinate_objective_stacking": "false",
    "true_path_endpoint": "1.0",
}

os.makedirs(os.path.dirname(runtime_meta), exist_ok=True)

with open(runtime_meta, "w", encoding="utf-8") as f:
    json.dump(runtime, f, indent=2, ensure_ascii=False)
    f.write("\n")

uppercase_keys = [k for k in cfg if k.isupper()]
if uppercase_keys:
    raise ValueError(
        "Generated config still contains uppercase keys that would break "
        f"argparse: {uppercase_keys}"
    )

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

print_settings

echo "Config: $RUN_CONFIG"
echo "Save dir: $RUN_DIR"
echo "Runtime metadata: $RUNTIME_META"

python - "$RUN_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

print("Generated loader/log/resume config:")
for key in [
    "num_workers",
    "prefetch_factor",
    "valid_num_workers",
    "valid_prefetch_factor",
    "valid_persistent_workers",
    "log_interval",
    "save_interval",
    "amp",
    "amp_dtype",
    "allow_tf32",
    "max_epoch",
    "resume_checkpoint",
]:
    print(f"  {key}={cfg.get(key, '')}")

bad = [k for k in cfg if k.isupper()]
if bad:
    raise SystemExit(f"ERROR: uppercase keys remain in generated config: {bad}")
PY

if [[ "${ABFLOW_DRY_RUN:-0}" == "1" ]]; then
  echo "ABFLOW_DRY_RUN=1: configuration generated; training was not started."
  exit 0
fi

AUTO_EVAL_GPUS=$(_infer_spare_eval_gpus)
_start_auto_topk_watcher "$AUTO_EVAL_GPUS" "$RUN_DIR"

set +e
run_with_env bash scripts/train/train.sh "$RUN_CONFIG"
TRAIN_STATUS=$?
set -e

# Evaluate only after a successful training process.  A failed run has no valid
# new checkpoint and must not launch a catch-up evaluation job that obscures the
# original exception or consumes another GPU.
if [[ "$TRAIN_STATUS" -eq 0 ]]; then
  _run_auto_topk_once "${AUTO_EVAL_GPUS:-$GPU_ID}" "$RUN_DIR"
else
  echo "[AutoTopK] training failed with status=$TRAIN_STATUS; skipping final evaluation."
fi
_stop_auto_topk_watcher "$RUN_DIR"
exit "$TRAIN_STATUS"
