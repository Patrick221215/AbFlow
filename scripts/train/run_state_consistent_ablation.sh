#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
EXP_ID=${2:-}
GPU_ID=${3:-0}

# ============================================================
# AbFlow formal state-consistent train/test ablation launcher
# ============================================================
# Core experimental hierarchy:
#
#   REF:
#       reference source, no peptide hidden condition.
#
#   REF_COND:
#       reference source + peptide hidden condition.
#
#   PCS:
#       proposal-conditioned source only. This tests whether using X_pep/S_pep
#       as the declared source is enough.
#
#   PCS_RC:
#       proposal-conditioned source + recurrent proposal context. This is the
#       recommended strong base: X_pep/S_pep define the source and also build
#       the recurrent global proposal context at every _forward call, while the
#       explicit generated state Xt/St is never overwritten.
#
#   PCS_RC_COND:
#       PCS_RC plus unrestricted proposal-relative adapters from round 0.
#       This is the already-tested upper local-correction endpoint.
#
#   PCS_RC_LC_R1:
#       PCS_RC plus delayed local correction from round 1.  Round 0 is reserved
#       for H3 placement; later rounds use proposal-relative adapters for local
#       structure/sequence correction.
#
#   PCS_RC_LC_R2:
#       PCS_RC plus late local correction from round 2.  With iter_round=3 this
#       means only the final refinement round uses proposal-relative adapters.
#
#   PCS_RC_LC_R1_SI_SCORE:
#       PCS_RC_LC_R1 plus stochastic-interpolant analytic score regularization.
#       The score is induced from the endpoint head and known injected noise; no
#       independent score head is added.
#
#   PCS_RC_LC_R1_SI_SCORE_FM:
#       PCS_RC_LC_R1 plus stochastic-interpolant analytic score and velocity
#       consistency. Retained as a historical diagnostic.
#
#   PCS_RC_LC_R1_TRAJ:
#       PCS_RC_LC_R1 plus model-induced trajectory endpoint consistency.
#       The same endpoint model is queried at Xt and at Xt+dt; no extra head is
#       added. This tests dynamic self-consistency of the learned flow.
#
#   PCS_RC_LC_R1_TRAJ_FM:
#       PCS_RC_LC_R1_TRAJ plus induced velocity-field consistency. This is a
#       full two-query diagnostic and can be expensive.
#
#   PCS_RC_LC_R1_SATC_LITE:
#       Lightweight score-aware trajectory/tangent consistency.  The model sees
#       score-defined off-path states and learns a one-forward correction
#       direction without an extra _forward call.
#
#   PCS_RC_LC_R1_SATC_FM_LITE:
#       SATC_LITE plus a small projected velocity-correction consistency.
#   PCS_RC_LC_R1_SATC_IF_MAIN:
#       Interface-weighted SATC main candidate for DockQ/CAAR/LDDT.
#   PCS_RC_LC_R1_SATC_IF_FM_SOFT:
#       Interface-weighted SATC plus very soft velocity-magnitude consistency.
#       This is the recommended next experiment for score+velocity.
#
#   CORE:
#       reference source + analytic_core objective.
#       Analytic score is currently valid only for reference source.
#
# Formal defaults:
#   - full observability: LOG_INTERVAL=1, SAVE_INTERVAL=1
#   - condition diagnostics on
#   - train/valid DataLoader resources separated
#   - resume behavior follows resume_checkpoint in BASE_CONFIG:
#       ""       -> train from scratch
#       nonempty -> resume from that checkpoint

STATE_PATH=on
PER_SAMPLE_T=on
TIME_EMBED=on
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
SEQ_CE_WEIGHT=${ABFLOW_SEQ_CE_WEIGHT:-1.0}
PROPOSAL_ADAPTER_START_ROUND=${ABFLOW_PROPOSAL_ADAPTER_START_ROUND:-0}
SI_GAMMA_SCALE=${ABFLOW_SI_GAMMA_SCALE:-0.25}
SI_SCORE_WEIGHT=${ABFLOW_SI_SCORE_WEIGHT:-0.002}
SI_VELOCITY_WEIGHT=${ABFLOW_SI_VELOCITY_WEIGHT:-0.01}

TRAJ_CONSISTENCY_WEIGHT=${ABFLOW_TRAJ_CONSISTENCY_WEIGHT:-0.05}
TRAJ_VELOCITY_WEIGHT=${ABFLOW_TRAJ_VELOCITY_WEIGHT:-0.0}
TRAJ_DELTA_T=${ABFLOW_TRAJ_DELTA_T:-0.15}
TRAJ_T_MIN=${ABFLOW_TRAJ_T_MIN:-0.05}
TRAJ_T_MAX=${ABFLOW_TRAJ_T_MAX:-0.80}

SATC_APPLY_PROB=${ABFLOW_SATC_APPLY_PROB:-0.50}
SATC_GAMMA_SCALE=${ABFLOW_SATC_GAMMA_SCALE:-0.08}
SATC_SCORE_WEIGHT=${ABFLOW_SATC_SCORE_WEIGHT:-0.02}
SATC_VELOCITY_WEIGHT=${ABFLOW_SATC_VELOCITY_WEIGHT:-0.003}
SATC_T_MIN=${ABFLOW_SATC_T_MIN:-0.10}
SATC_T_MAX=${ABFLOW_SATC_T_MAX:-0.80}
SATC_INTERFACE_WEIGHT_ALPHA=${ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA:-0.0}
SATC_INTERFACE_CUTOFF=${ABFLOW_SATC_INTERFACE_CUTOFF:-8.0}
SATC_INTERFACE_TEMPERATURE=${ABFLOW_SATC_INTERFACE_TEMPERATURE:-1.0}
SATC_INTERFACE_NORMALIZE=${ABFLOW_SATC_INTERFACE_NORMALIZE:-on}

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

case "$EXP_ID" in
  REF)
    SOURCE_MODE=reference
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=off
    SEQ_INPUT_MODE=state
    ;;

  REF_SEQ)
    SOURCE_MODE=reference
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=off
    SEQ_INPUT_MODE=pep_condition
    ;;

  REF_COORD)
    SOURCE_MODE=reference
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=state
    ;;

  REF_COND)
    SOURCE_MODE=reference
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;

  PCS|CS)
    SOURCE_MODE=pcs
    RECURRENT_PROPOSAL_CONTEXT=off
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=off
    SEQ_INPUT_MODE=state
    ;;

  PCS_RC)
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=off
    SEQ_INPUT_MODE=state
    ;;

  PCS_RC_COND|CS_COND)
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=0
    ;;

  PCS_RC_LC|PCS_RC_LC_R1)
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    ;;

  PCS_RC_LC_R2)
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=endpoint
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=2
    ;;

  PCS_RC_LC_R1_SI_SCORE)
    # R1 strong baseline + stochastic-interpolant analytic score.
    # This is the safer diagnostic: endpoint remains the primary objective;
    # the analytic score term is a small regularizer on noisy mid-trajectory
    # states induced by the endpoint head.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=si_score
    T_SAMPLING=mid_t
    DSM_T_MIN=${ABFLOW_SCOREFM_DSM_T_MIN:-0.2}
    DSM_T_MAX=${ABFLOW_SCOREFM_DSM_T_MAX:-0.8}
    SI_GAMMA_SCALE=${ABFLOW_SI_GAMMA_SCALE:-0.25}
    SI_SCORE_WEIGHT=${ABFLOW_SI_SCORE_WEIGHT:-0.002}
    SI_VELOCITY_WEIGHT=${ABFLOW_SI_VELOCITY_WEIGHT:-0.0}
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    ;;

  PCS_RC_LC_R1_SI_SCORE_FM)
    # R1 strong baseline + stochastic-interpolant analytic score and velocity
    # consistency. Retained as a historical diagnostic.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=si_score_fm
    T_SAMPLING=mid_t
    DSM_T_MIN=${ABFLOW_SCOREFM_DSM_T_MIN:-0.2}
    DSM_T_MAX=${ABFLOW_SCOREFM_DSM_T_MAX:-0.8}
    SI_GAMMA_SCALE=${ABFLOW_SI_GAMMA_SCALE:-0.25}
    SI_SCORE_WEIGHT=${ABFLOW_SI_SCORE_WEIGHT:-0.001}
    SI_VELOCITY_WEIGHT=${ABFLOW_SI_VELOCITY_WEIGHT:-0.01}
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    ;;

  PCS_RC_LC_R1_TRAJ)
    # R1 strong baseline + local trajectory endpoint consistency.
    # Endpoint loss remains primary. The same endpoint model is queried at
    # Xt and at a short model-induced neighboring state Xt+dt.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=traj_consistency
    T_SAMPLING=uniform
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    TRAJ_CONSISTENCY_WEIGHT=${ABFLOW_TRAJ_CONSISTENCY_WEIGHT:-0.05}
    TRAJ_VELOCITY_WEIGHT=${ABFLOW_TRAJ_VELOCITY_WEIGHT:-0.0}
    TRAJ_DELTA_T=${ABFLOW_TRAJ_DELTA_T:-0.15}
    TRAJ_T_MIN=${ABFLOW_TRAJ_T_MIN:-0.05}
    TRAJ_T_MAX=${ABFLOW_TRAJ_T_MAX:-0.80}
    ;;

  PCS_RC_LC_R1_TRAJ_FM)
    # R1 strong baseline + trajectory endpoint consistency + induced velocity
    # field consistency. This directly tests flow-field self-consistency without
    # adding a score or velocity head.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=traj_consistency_fm
    T_SAMPLING=uniform
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    TRAJ_CONSISTENCY_WEIGHT=${ABFLOW_TRAJ_CONSISTENCY_WEIGHT:-0.03}
    TRAJ_VELOCITY_WEIGHT=${ABFLOW_TRAJ_VELOCITY_WEIGHT:-0.01}
    TRAJ_DELTA_T=${ABFLOW_TRAJ_DELTA_T:-0.15}
    TRAJ_T_MIN=${ABFLOW_TRAJ_T_MIN:-0.05}
    TRAJ_T_MAX=${ABFLOW_TRAJ_T_MAX:-0.80}
    ;;

  PCS_RC_LC_R1_SATC_LITE|PCS_RC_LC_R1_SCORE_AWARE_TRAJ_LITE)
    # Recommended lightweight score-aware trajectory/tangent consistency.
    # One forward per batch: the state is perturbed off the bridge by a known
    # score direction, and the induced correction velocity is aligned with it.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=score_aware_traj_lite
    T_SAMPLING=uniform
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    SATC_APPLY_PROB=${ABFLOW_SATC_APPLY_PROB:-0.50}
    SATC_GAMMA_SCALE=${ABFLOW_SATC_GAMMA_SCALE:-0.08}
    SATC_SCORE_WEIGHT=${ABFLOW_SATC_SCORE_WEIGHT:-0.02}
    SATC_VELOCITY_WEIGHT=${ABFLOW_SATC_VELOCITY_WEIGHT:-0.0}
    SATC_T_MIN=${ABFLOW_SATC_T_MIN:-0.10}
    SATC_T_MAX=${ABFLOW_SATC_T_MAX:-0.80}
    ;;

  PCS_RC_LC_R1_SATC_FM_LITE|PCS_RC_LC_R1_SCORE_AWARE_TRAJ_FM_LITE)
    # Recommended score+velocity candidate.  It keeps one-forward efficiency
    # and adds a small projected correction-magnitude term to make the
    # endpoint-induced velocity field explicitly score-aware.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=score_aware_traj_fm_lite
    T_SAMPLING=uniform
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    SATC_APPLY_PROB=${ABFLOW_SATC_APPLY_PROB:-0.50}
    SATC_GAMMA_SCALE=${ABFLOW_SATC_GAMMA_SCALE:-0.08}
    SATC_SCORE_WEIGHT=${ABFLOW_SATC_SCORE_WEIGHT:-0.015}
    SATC_VELOCITY_WEIGHT=${ABFLOW_SATC_VELOCITY_WEIGHT:-0.003}
    SATC_T_MIN=${ABFLOW_SATC_T_MIN:-0.10}
    SATC_T_MAX=${ABFLOW_SATC_T_MAX:-0.80}
    ;;

  PCS_RC_LC_R1_SATC_MAIN|PCS_RC_LC_R1_SCORE_AWARE_TRAJ_MAIN)
    # Final recommended main method after the current SATC results.
    # It is the successful direction-only score-aware tangent consistency: score
    # appears as the analytic off-path direction and velocity appears as the
    # endpoint-induced correction velocity aligned to that score direction.
    # No extra forward, no independent score/velocity head, and no explicit
    # velocity-magnitude forcing.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=score_aware_traj_lite
    T_SAMPLING=uniform
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    SATC_APPLY_PROB=${ABFLOW_SATC_APPLY_PROB:-0.50}
    SATC_GAMMA_SCALE=${ABFLOW_SATC_GAMMA_SCALE:-0.08}
    SATC_SCORE_WEIGHT=${ABFLOW_SATC_SCORE_WEIGHT:-0.02}
    SATC_VELOCITY_WEIGHT=${ABFLOW_SATC_VELOCITY_WEIGHT:-0.0}
    SATC_T_MIN=${ABFLOW_SATC_T_MIN:-0.10}
    SATC_T_MAX=${ABFLOW_SATC_T_MAX:-0.80}
    ;;

  PCS_RC_LC_R1_SATC_FM_SOFT|PCS_RC_LC_R1_SCORE_AWARE_TRAJ_FM_SOFT)
    # Soft velocity-magnitude diagnostic.  The v36 FM_LITE result improved
    # AAR/CAAR but hurt H3 raw RMSD and DockQ; therefore v37 keeps the same
    # score-aware tangent term and reduces the projected magnitude term.  This
    # is the next controlled test of whether explicit velocity magnitude can be
    # added without damaging placement.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=score_aware_traj_fm_lite
    T_SAMPLING=uniform
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    SATC_APPLY_PROB=${ABFLOW_SATC_APPLY_PROB:-0.50}
    SATC_GAMMA_SCALE=${ABFLOW_SATC_GAMMA_SCALE:-0.08}
    SATC_SCORE_WEIGHT=${ABFLOW_SATC_SCORE_WEIGHT:-0.018}
    SATC_VELOCITY_WEIGHT=${ABFLOW_SATC_VELOCITY_WEIGHT:-0.001}
    SATC_T_MIN=${ABFLOW_SATC_T_MIN:-0.10}
    SATC_T_MAX=${ABFLOW_SATC_T_MAX:-0.80}
    ;;

  PCS_RC_LC_R1_SATC_IF_MAIN|PCS_RC_LC_R1_SCORE_AWARE_TRAJ_IF_MAIN)
    # Interface-weighted SATC main candidate.  It keeps the validated
    # SATC_MAIN direction-only mechanism and reweights only the SATC regularizer
    # toward native interface/contact residues.  This targets DockQ/CAAR/LDDT
    # without adding a second forward or a new score/velocity head.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=score_aware_traj_if_lite
    T_SAMPLING=uniform
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    SATC_APPLY_PROB=${ABFLOW_SATC_APPLY_PROB:-0.50}
    SATC_GAMMA_SCALE=${ABFLOW_SATC_GAMMA_SCALE:-0.08}
    SATC_SCORE_WEIGHT=${ABFLOW_SATC_SCORE_WEIGHT:-0.02}
    SATC_VELOCITY_WEIGHT=${ABFLOW_SATC_VELOCITY_WEIGHT:-0.0}
    SATC_T_MIN=${ABFLOW_SATC_T_MIN:-0.10}
    SATC_T_MAX=${ABFLOW_SATC_T_MAX:-0.80}
    SATC_INTERFACE_WEIGHT_ALPHA=${ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA:-1.0}
    SATC_INTERFACE_CUTOFF=${ABFLOW_SATC_INTERFACE_CUTOFF:-8.0}
    SATC_INTERFACE_TEMPERATURE=${ABFLOW_SATC_INTERFACE_TEMPERATURE:-1.0}
    SATC_INTERFACE_NORMALIZE=${ABFLOW_SATC_INTERFACE_NORMALIZE:-on}
    ;;

  PCS_RC_LC_R1_SATC_IF_FM_SOFT|PCS_RC_LC_R1_SCORE_AWARE_TRAJ_IF_FM_SOFT)
    # Interface-weighted SATC with very soft projected velocity magnitude.
    # This is a controlled ablation for whether explicit velocity magnitude can
    # improve AAR/CAAR while the interface weighting protects H3 placement/DockQ.
    SOURCE_MODE=pcs_rc
    RECURRENT_PROPOSAL_CONTEXT=on
    LOSS_MODE=score_aware_traj_if_fm_lite
    T_SAMPLING=uniform
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    PROPOSAL_ADAPTER_START_ROUND=1
    SATC_APPLY_PROB=${ABFLOW_SATC_APPLY_PROB:-0.50}
    SATC_GAMMA_SCALE=${ABFLOW_SATC_GAMMA_SCALE:-0.08}
    SATC_SCORE_WEIGHT=${ABFLOW_SATC_SCORE_WEIGHT:-0.018}
    SATC_VELOCITY_WEIGHT=${ABFLOW_SATC_VELOCITY_WEIGHT:-0.0005}
    SATC_T_MIN=${ABFLOW_SATC_T_MIN:-0.10}
    SATC_T_MAX=${ABFLOW_SATC_T_MAX:-0.80}
    SATC_INTERFACE_WEIGHT_ALPHA=${ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA:-1.0}
    SATC_INTERFACE_CUTOFF=${ABFLOW_SATC_INTERFACE_CUTOFF:-8.0}
    SATC_INTERFACE_TEMPERATURE=${ABFLOW_SATC_INTERFACE_TEMPERATURE:-1.0}
    SATC_INTERFACE_NORMALIZE=${ABFLOW_SATC_INTERFACE_NORMALIZE:-on}
    ;;

  CORE)
    SOURCE_MODE=reference
    LOSS_MODE=analytic_core
    COORD_PEP_AS_CONDITION=on
    SEQ_INPUT_MODE=pep_condition
    ;;

  *)
    echo "Unknown EXP_ID: $EXP_ID"
    echo "Supported EXP_ID: REF REF_SEQ REF_COORD REF_COND PCS PCS_RC PCS_RC_COND PCS_RC_LC_R1 PCS_RC_LC_R2 PCS_RC_LC_R1_SI_SCORE PCS_RC_LC_R1_SI_SCORE_FM PCS_RC_LC_R1_TRAJ PCS_RC_LC_R1_TRAJ_FM PCS_RC_LC_R1_SATC_LITE PCS_RC_LC_R1_SATC_FM_LITE PCS_RC_LC_R1_SATC_MAIN PCS_RC_LC_R1_SATC_FM_SOFT PCS_RC_LC_R1_SATC_IF_MAIN PCS_RC_LC_R1_SATC_IF_FM_SOFT CORE"
    exit 2
    ;;
esac

run_with_env() {
  ABFLOW_SOURCE_MODE="$SOURCE_MODE" \
  ABFLOW_RECURRENT_PROPOSAL_CONTEXT="$RECURRENT_PROPOSAL_CONTEXT" \
  ABFLOW_COORD_PEP_SOURCE_WEIGHT="$COORD_PEP_SOURCE_WEIGHT" \
  ABFLOW_SEQ_PEP_SOURCE_WEIGHT="$SEQ_PEP_SOURCE_WEIGHT" \
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
  ABFLOW_PROPOSAL_ADAPTER_START_ROUND="$PROPOSAL_ADAPTER_START_ROUND" \
  ABFLOW_SI_GAMMA_SCALE="$SI_GAMMA_SCALE" \
  ABFLOW_SI_SCORE_WEIGHT="$SI_SCORE_WEIGHT" \
  ABFLOW_SI_VELOCITY_WEIGHT="$SI_VELOCITY_WEIGHT" \
  ABFLOW_TRAJ_CONSISTENCY_WEIGHT="$TRAJ_CONSISTENCY_WEIGHT" \
  ABFLOW_TRAJ_VELOCITY_WEIGHT="$TRAJ_VELOCITY_WEIGHT" \
  ABFLOW_TRAJ_DELTA_T="$TRAJ_DELTA_T" \
  ABFLOW_TRAJ_T_MIN="$TRAJ_T_MIN" \
  ABFLOW_TRAJ_T_MAX="$TRAJ_T_MAX" \
  ABFLOW_SATC_APPLY_PROB="$SATC_APPLY_PROB" \
  ABFLOW_SATC_GAMMA_SCALE="$SATC_GAMMA_SCALE" \
  ABFLOW_SATC_SCORE_WEIGHT="$SATC_SCORE_WEIGHT" \
  ABFLOW_SATC_VELOCITY_WEIGHT="$SATC_VELOCITY_WEIGHT" \
  ABFLOW_SATC_T_MIN="$SATC_T_MIN" \
  ABFLOW_SATC_T_MAX="$SATC_T_MAX" \
  ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA="$SATC_INTERFACE_WEIGHT_ALPHA" \
  ABFLOW_SATC_INTERFACE_CUTOFF="$SATC_INTERFACE_CUTOFF" \
  ABFLOW_SATC_INTERFACE_TEMPERATURE="$SATC_INTERFACE_TEMPERATURE" \
  ABFLOW_SATC_INTERFACE_NORMALIZE="$SATC_INTERFACE_NORMALIZE" \
  ABFLOW_CONDITION_DIAGNOSTICS="$CONDITION_DIAGNOSTICS" \
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
  echo "GPU: $GPU_ID"
  echo "SOURCE_MODE=$SOURCE_MODE"
  echo "RECURRENT_PROPOSAL_CONTEXT=$RECURRENT_PROPOSAL_CONTEXT"
  echo "COORD_PEP_SOURCE_WEIGHT=$COORD_PEP_SOURCE_WEIGHT"
  echo "SEQ_PEP_SOURCE_WEIGHT=$SEQ_PEP_SOURCE_WEIGHT"
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
  echo "PROPOSAL_ADAPTER_START_ROUND=$PROPOSAL_ADAPTER_START_ROUND"
  echo "SI_GAMMA_SCALE=$SI_GAMMA_SCALE"
  echo "SI_SCORE_WEIGHT=$SI_SCORE_WEIGHT"
  echo "SI_VELOCITY_WEIGHT=$SI_VELOCITY_WEIGHT"
  echo "TRAJ_CONSISTENCY_WEIGHT=$TRAJ_CONSISTENCY_WEIGHT"
  echo "TRAJ_VELOCITY_WEIGHT=$TRAJ_VELOCITY_WEIGHT"
  echo "TRAJ_DELTA_T=$TRAJ_DELTA_T"
  echo "TRAJ_T_MIN=$TRAJ_T_MIN"
  echo "TRAJ_T_MAX=$TRAJ_T_MAX"
  echo "SATC_APPLY_PROB=$SATC_APPLY_PROB"
  echo "SATC_GAMMA_SCALE=$SATC_GAMMA_SCALE"
  echo "SATC_SCORE_WEIGHT=$SATC_SCORE_WEIGHT"
  echo "SATC_VELOCITY_WEIGHT=$SATC_VELOCITY_WEIGHT"
  echo "SATC_T_MIN=$SATC_T_MIN"
  echo "SATC_T_MAX=$SATC_T_MAX"
  echo "SATC_INTERFACE_WEIGHT_ALPHA=$SATC_INTERFACE_WEIGHT_ALPHA"
  echo "SATC_INTERFACE_CUTOFF=$SATC_INTERFACE_CUTOFF"
  echo "SATC_INTERFACE_TEMPERATURE=$SATC_INTERFACE_TEMPERATURE"
  echo "SATC_INTERFACE_NORMALIZE=$SATC_INTERFACE_NORMALIZE"
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
  echo "CONDITION_DIAGNOSTICS=$CONDITION_DIAGNOSTICS"
}

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
  echo "Supported EXP_ID: REF REF_SEQ REF_COORD REF_COND PCS PCS_RC PCS_RC_COND PCS_RC_LC_R1 PCS_RC_LC_R2 PCS_RC_LC_R1_SI_SCORE PCS_RC_LC_R1_SI_SCORE_FM PCS_RC_LC_R1_TRAJ PCS_RC_LC_R1_TRAJ_FM PCS_RC_LC_R1_SATC_LITE PCS_RC_LC_R1_SATC_FM_LITE PCS_RC_LC_R1_SATC_MAIN PCS_RC_LC_R1_SATC_FM_SOFT PCS_RC_LC_R1_SATC_IF_MAIN PCS_RC_LC_R1_SATC_IF_FM_SOFT CORE"
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
#   resume_checkpoint == ""       -> train from scratch
#   resume_checkpoint == nonempty -> resume from that checkpoint
# The launcher must not silently override this field.
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
    "source_mode": os.environ.get("ABFLOW_SOURCE_MODE", ""),
    "recurrent_proposal_context": os.environ.get("ABFLOW_RECURRENT_PROPOSAL_CONTEXT", ""),
    "coord_pep_source_weight": os.environ.get("ABFLOW_COORD_PEP_SOURCE_WEIGHT", ""),
    "seq_pep_source_weight": os.environ.get("ABFLOW_SEQ_PEP_SOURCE_WEIGHT", ""),
    "state_path": os.environ.get("ABFLOW_SCOREFM_STATE_PATH", ""),
    "per_sample_t": os.environ.get("ABFLOW_SCOREFM_PER_SAMPLE_T", ""),
    "time_embed": os.environ.get("ABFLOW_SCOREFM_TIME_EMBED", ""),
    "t_sampling": os.environ.get("ABFLOW_SCOREFM_T_SAMPLING", ""),
    "loss_mode": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", ""),
    "min_sigma": os.environ.get("ABFLOW_SCOREFM_MIN_SIGMA", ""),
    "dsm_t_min": os.environ.get("ABFLOW_SCOREFM_DSM_T_MIN", ""),
    "dsm_t_max": os.environ.get("ABFLOW_SCOREFM_DSM_T_MAX", ""),
    "sampler_mode": os.environ.get("ABFLOW_SCOREFM_SAMPLER_MODE", ""),
    "coord_pep_as_condition": os.environ.get("ABFLOW_COORD_PEP_AS_CONDITION", ""),
    "seq_input_mode": os.environ.get("ABFLOW_SEQ_INPUT_MODE", ""),
    "seq_ce_weight": os.environ.get("ABFLOW_SEQ_CE_WEIGHT", ""),
    "proposal_adapter_start_round": os.environ.get("ABFLOW_PROPOSAL_ADAPTER_START_ROUND", ""),
    "si_gamma_scale": os.environ.get("ABFLOW_SI_GAMMA_SCALE", ""),
    "si_score_weight": os.environ.get("ABFLOW_SI_SCORE_WEIGHT", ""),
    "si_velocity_weight": os.environ.get("ABFLOW_SI_VELOCITY_WEIGHT", ""),
    "traj_consistency_weight": os.environ.get("ABFLOW_TRAJ_CONSISTENCY_WEIGHT", ""),
    "traj_velocity_weight": os.environ.get("ABFLOW_TRAJ_VELOCITY_WEIGHT", ""),
    "traj_delta_t": os.environ.get("ABFLOW_TRAJ_DELTA_T", ""),
    "traj_t_min": os.environ.get("ABFLOW_TRAJ_T_MIN", ""),
    "traj_t_max": os.environ.get("ABFLOW_TRAJ_T_MAX", ""),
    "satc_apply_prob": os.environ.get("ABFLOW_SATC_APPLY_PROB", ""),
    "satc_gamma_scale": os.environ.get("ABFLOW_SATC_GAMMA_SCALE", ""),
    "satc_score_weight": os.environ.get("ABFLOW_SATC_SCORE_WEIGHT", ""),
    "satc_velocity_weight": os.environ.get("ABFLOW_SATC_VELOCITY_WEIGHT", ""),
    "satc_t_min": os.environ.get("ABFLOW_SATC_T_MIN", ""),
    "satc_t_max": os.environ.get("ABFLOW_SATC_T_MAX", ""),
    "satc_interface_weight_alpha": os.environ.get("ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA", ""),
    "satc_interface_cutoff": os.environ.get("ABFLOW_SATC_INTERFACE_CUTOFF", ""),
    "satc_interface_temperature": os.environ.get("ABFLOW_SATC_INTERFACE_TEMPERATURE", ""),
    "satc_interface_normalize": os.environ.get("ABFLOW_SATC_INTERFACE_NORMALIZE", ""),
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
    "resume_checkpoint": resume_ckpt,
    "clean_reference_source": os.environ.get("ABFLOW_SOURCE_MODE", "") == "reference",
    "proposal_conditioned_source": os.environ.get("ABFLOW_SOURCE_MODE", "") in {"pcs", "pcs_rc"},
    "proposal_recurrent_context": os.environ.get("ABFLOW_RECURRENT_PROPOSAL_CONTEXT", "") == "on",
    "peptide_state_injection": "false",
    "peptide_prior_weighting": "false",
    "independent_score_head": "false",
    "independent_velocity_head": "false",
    "stochastic_interpolant_training": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", "") in {"si_score", "si_score_fm"},
    "trajectory_consistency_training": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", "") in {"traj_consistency", "traj_consistency_fm"},
    "score_aware_trajectory_lite_training": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", "") in {"score_aware_traj_lite", "score_aware_traj_fm_lite", "score_aware_traj_if_lite", "score_aware_traj_if_fm_lite"},
    "interface_weighted_satc": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", "") in {"score_aware_traj_if_lite", "score_aware_traj_if_fm_lite"},
    "pair_time_conditioning": "false",
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

run_with_env bash scripts/train/train.sh "$RUN_CONFIG"
