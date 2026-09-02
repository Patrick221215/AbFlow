#!/usr/bin/env bash
set -euo pipefail

# Formal v111 integrated experiment launcher.
# One purpose only: freeze the accepted scientific/runtime settings and launch
# the existing AbFlow train.sh. No preflight, validator, smoke or AutoTopK
# watcher is inserted into the formal execution chain.

MODE="${1:-train}"

# Train/flowtest:
#   bash run_MFSC_v111_full.sh flowtest 2,3
#   bash run_MFSC_v111_full.sh train    2,3
#
# Standalone/Test bridge (kept compatible with evaluate_topk_map.py):
#   bash run_MFSC_v111_full.sh test <EXP_ID> 2,3 <CKPT> <RESULT_DIR> [TEST_JSON]
if [[ "$MODE" == "test" ]]; then
  EXP_ID="${2:-MFSC_V111_PCS_RC_ABX_U02_FULL}"
  GPU_ID="${3:-2,3}"
else
  GPU_ID="${2:-2,3}"
  if [[ "$MODE" == "flowtest" ]]; then
    EXP_ID="MFSC_V123_SPEED_PROFILE_BS8"
  else
    EXP_ID="MFSC_V119_PCS_RC_ABX_U02_FULL_EXACT_RUNTIME"
  fi
fi

if [[ "$MODE" != "train" && "$MODE" != "flowtest" && "$MODE" != "test" ]]; then
  echo "Usage:"
  echo "  bash $0 flowtest 2,3   # Train=Val, Valid=Val, global bs=10, Pairformer checkpoint+chunking"
  echo "  bash $0 train 2,3"
  echo "  bash $0 test MFSC_V119_PCS_RC_ABX_U02_FULL_EXACT_RUNTIME 2,3 <ckpt> <result_dir> [test.json]"
  exit 2
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

BASE_CONFIG="${PROJECT_ROOT}/scripts/train/configs/MFSC_v111_full/${EXP_ID}.json"

[[ -f "$BASE_CONFIG" ]] || {
  echo "ERROR: config not found: $BASE_CONFIG" >&2
  exit 2
}

# =====================================================================
# Accepted PCS-RC + U02/F01 Score--Flow parent
# =====================================================================
export ABFLOW_SOURCE_MODE="pcs_rc"
export ABFLOW_RECURRENT_PROPOSAL_CONTEXT="on"
export ABFLOW_COORD_PEP_SOURCE_WEIGHT="1.0"
export ABFLOW_SEQ_PEP_SOURCE_WEIGHT="1.0"
export ABFLOW_COORD_PEP_AS_CONDITION="off"
export ABFLOW_PROPOSAL_ADAPTER_START_ROUND="0"

export ABFLOW_ABX_COMMON_CENTER="on"

export ABFLOW_SCOREFM_STATE_PATH="on"
export ABFLOW_SCOREFM_PER_SAMPLE_T="on"
export ABFLOW_SCOREFM_TIME_EMBED="on"
export ABFLOW_SCOREFM_T_SAMPLING="uniform"
export ABFLOW_SCOREFM_MIN_SIGMA="0.01"
export ABFLOW_SCOREFM_DSM_T_MIN="0.20"
export ABFLOW_SCOREFM_DSM_T_MAX="0.80"

# Explicitly freeze the defaults already used by the accepted v111 parent.
export ABFLOW_R3_G_MODE="adaptive_transport"
export ABFLOW_R3_NOISE_SCOPE="global"
export ABFLOW_R3_FLOW_COORDINATE_SCALING="0.1"
export ABFLOW_R3_TRANSPORT_FRACTION="0.05"
export ABFLOW_R3_TRANSPORT_MAX="20.0"
export ABFLOW_R3_PATH_MIN_SIGMA="0.0"
export ABFLOW_R3_SCORE_MIN_SIGMA="0.01"

export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
export ABFLOW_F01_HYBRID_T_MIN="0.20"
export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
export ABFLOW_FLOW_T_MIN="0.0"
export ABFLOW_FLOW_T_MAX="1.0"
export ABFLOW_FINAL_READOUT_MODE="integrated_endpoint"

# =====================================================================
# Accepted v111 MFDesign semantic closure
# =====================================================================
export ABFLOW_MFDESIGN_RECYCLING_STEPS="3"
export ABFLOW_MFDESIGN_RANDOM_RECYCLING="on"
export ABFLOW_CURRENT_STATE_PAIR_GEOMETRY="on"

export ABFLOW_MODERN_OBJECTIVE_MODE="transport_sequence_distogram"
export ABFLOW_COORDINATE_AUTHORITY="carrier_single"
export ABFLOW_PAIR_DISTOGRAM="on"
export ABFLOW_PAIR_DISTOGRAM_SCOPE="design"

export ABFLOW_CLEAN_SELF_CONDITION="on"
export ABFLOW_CLEAN_SELF_CONDITION_PROB="0.50"
export ABFLOW_MODEL_COORD_SCALE_ANGSTROM="10.0"

export ABFLOW_SEQUENCE_GENERATIVE_MODE="masked_absorbing"
export ABFLOW_SEQUENCE_CONTEXT_MODE="off"
export ABFLOW_SEQUENCE_LOSS_SCOPE="final"
export ABFLOW_SEQ_INPUT_MODE="state"
export ABFLOW_DUAL_SEQUENCE_STATE="off"
export ABFLOW_RECURRENT_PROPOSAL_SEQUENCE_CONTEXT="off"
export ABFLOW_MASKED_SEQUENCE_SELF_CONDITION_MODE="off"
export ABFLOW_SEQUENCE_RECYCLE_MODE="off"
export ABFLOW_STRUCTURE_SEQ_READOUT="off"

export ABFLOW_JOINT_SAMPLER_MODE="synchronized"
export ABFLOW_JOINT_SEQUENCE_TERMINAL="integrated_state"

export ABFLOW_CONFIDENCE="on"
export ABFLOW_CONFIDENCE_PAE="on"

# =====================================================================
# Rejected/competing branches remain disabled
# =====================================================================
export ABFLOW_PAIR_TIME_SCOPE="off"
export ABFLOW_PAIR_TIME_CONDITIONING="off"
export ABFLOW_SCOREFLOW_PAIR_MODE="off"
export ABFLOW_SCOREFLOW_PAIR_STOP_GRAD="on"
export ABFLOW_R3_SCORE_DSM_WEIGHT="0.0"
export ABFLOW_R3_PATHFLOW_WEIGHT="0.0"
export ABFLOW_SF2M_SCORE_WEIGHT="0.0"
export ABFLOW_SATC_APPLY_PROB="0.0"
export ABFLOW_SATC_SCORE_WEIGHT="0.0"
export ABFLOW_SATC_VELOCITY_WEIGHT="0.0"
export ABFLOW_SUPPORT_FACTORIZED_COORD="off"
export ABFLOW_TRANSLATION_ROUND_CREDIT="off"
export ABFLOW_ROUND_CONSISTENT_COORD_SUPERVISION="off"

# =====================================================================
# Accepted optimization/runtime
# =====================================================================
export ABFLOW_OPTIMIZER="adamw"
export ABFLOW_WEIGHT_DECAY="0.01"
export ABFLOW_WARMUP_EPOCHS="5"
# v117 formal runtime optimization is semantics-preserving only.
# DDP-safe non-reentrant checkpointing is wrapped with the exact BF16 autocast
# state from the original forward.  No scientific branch is skipped.
export ABFLOW_PAIRFORMER_ACTIVATION_CHECKPOINT="${ABFLOW_PAIRFORMER_ACTIVATION_CHECKPOINT:-on}"
export ABFLOW_PAIRFORMER_CHECKPOINT_MODE="triangle"

# v118 exact low-memory TriangleAttention.
# Outer anchor chunking is off by default; AFAttention chooses full or exact
# blockwise log-sum-exp from the actual FP32 logits memory footprint.
export ABFLOW_TRIANGLE_ATTENTION_CHUNK_SIZE="${ABFLOW_TRIANGLE_ATTENTION_CHUNK_SIZE:-0}"
export ABFLOW_TRIANGLE_ATTN_BACKEND="${ABFLOW_TRIANGLE_ATTN_BACKEND:-auto}"
export ABFLOW_TRIANGLE_FULL_LOGITS_LIMIT_MB="${ABFLOW_TRIANGLE_FULL_LOGITS_LIMIT_MB:-384}"
export ABFLOW_TRIANGLE_LMA_Q_CHUNK_SIZE="${ABFLOW_TRIANGLE_LMA_Q_CHUNK_SIZE:-64}"
export ABFLOW_TRIANGLE_LMA_KV_CHUNK_SIZE="${ABFLOW_TRIANGLE_LMA_KV_CHUNK_SIZE:-128}"
# v119: three-layer exact memory architecture.
# Avoid nested checkpoint recomputation. Per-complex transient attention peak is
# already bounded by exact LMA; outer complex checkpoint owns activation lifetime.

# v123: OOM root cause is resolved. Remove the expensive per-atom/per-complex
# trace from ordinary flowtest and profile only the quantities relevant to the
# current question: where training time goes and whether full/LMA attention is
# actually selected. Formal train mode has zero profiling synchronization.
if [[ "$MODE" == "flowtest" ]]; then
  export ABFLOW_MEMORY_DIAGNOSTICS="${ABFLOW_MEMORY_DIAGNOSTICS:-off}"
  export ABFLOW_MEMORY_OWNER_DIAGNOSTICS="${ABFLOW_MEMORY_OWNER_DIAGNOSTICS:-off}"
  export ABFLOW_RUNTIME_TRACE="${ABFLOW_RUNTIME_TRACE:-off}"
  export ABFLOW_DEEP_CHECKPOINT_TRACE="${ABFLOW_DEEP_CHECKPOINT_TRACE:-off}"
  export ABFLOW_PERF_DIAGNOSTICS="${ABFLOW_PERF_DIAGNOSTICS:-on}"
  export ABFLOW_PERF_DIAGNOSTIC_STEPS="${ABFLOW_PERF_DIAGNOSTIC_STEPS:-8}"
else
  export ABFLOW_MEMORY_DIAGNOSTICS="${ABFLOW_MEMORY_DIAGNOSTICS:-off}"
  export ABFLOW_MEMORY_OWNER_DIAGNOSTICS="${ABFLOW_MEMORY_OWNER_DIAGNOSTICS:-off}"
  export ABFLOW_RUNTIME_TRACE="${ABFLOW_RUNTIME_TRACE:-off}"
  export ABFLOW_DEEP_CHECKPOINT_TRACE="${ABFLOW_DEEP_CHECKPOINT_TRACE:-off}"
  export ABFLOW_PERF_DIAGNOSTICS="${ABFLOW_PERF_DIAGNOSTICS:-off}"
fi

export ABFLOW_AMP="${ABFLOW_AMP:-on}"
export ABFLOW_AMP_DTYPE="${ABFLOW_AMP_DTYPE:-bf16}"
export ABFLOW_ALLOW_TF32="${ABFLOW_ALLOW_TF32:-on}"
export ABFLOW_NUM_WORKERS="${ABFLOW_NUM_WORKERS:-8}"
export ABFLOW_PREFETCH_FACTOR="${ABFLOW_PREFETCH_FACTOR:-4}"
export ABFLOW_VALID_NUM_WORKERS="${ABFLOW_VALID_NUM_WORKERS:-2}"
export ABFLOW_VALID_PREFETCH_FACTOR="${ABFLOW_VALID_PREFETCH_FACTOR:-2}"
export ABFLOW_VALID_PERSISTENT_WORKERS="${ABFLOW_VALID_PERSISTENT_WORKERS:-off}"
export ABFLOW_LOG_INTERVAL="${ABFLOW_LOG_INTERVAL:-10}"
export ABFLOW_TQDM_MININTERVAL="${ABFLOW_TQDM_MININTERVAL:-10.0}"
export ABFLOW_SAVE_INTERVAL="${ABFLOW_SAVE_INTERVAL:-1}"

# =====================================================================
# Formal every-epoch Train -> Validation -> Test
# =====================================================================
export ABFLOW_THREE_PHASE_PROTOCOL="on"
export ABFLOW_DDP_VALIDATION="on"
export ABFLOW_EPOCH_TEST="on"
export ABFLOW_EPOCH_TEST_INTERVAL="1"
export ABFLOW_EPOCH_TEST_JSON="${ABFLOW_EPOCH_TEST_JSON:-${PROJECT_ROOT}/datasets/RAbD/test.json}"
export ABFLOW_EPOCH_TEST_PEP="${ABFLOW_EPOCH_TEST_PEP:-${PROJECT_ROOT}/datasets/RAbD/test.pkl}"
export ABFLOW_EPOCH_TEST_SURF="${ABFLOW_EPOCH_TEST_SURF:-${PROJECT_ROOT}/datasets/RAbD/test_surf.pkl}"
if [[ "$MODE" == "flowtest" ]]; then
  # Train/Validation global batch is 8 from the v123 speed-profile JSON.
  # Test uses a separate world-size-independent logical batch protocol.
  # Keep it small by default for stable bring-up; override explicitly if needed.
  export ABFLOW_EPOCH_TEST_BATCH_SIZE="${ABFLOW_FLOWTEST_TEST_BATCH_SIZE:-2}"
  export ABFLOW_EPOCH_TEST_N_STEPS="${ABFLOW_FLOWTEST_TEST_N_STEPS:-2}"
  export ABFLOW_EPOCH_TEST_METRIC_WORKERS="${ABFLOW_FLOWTEST_TEST_METRIC_WORKERS:-2}"
else
  export ABFLOW_EPOCH_TEST_BATCH_SIZE="${ABFLOW_EPOCH_TEST_BATCH_SIZE:-20}"
  export ABFLOW_EPOCH_TEST_N_STEPS="${ABFLOW_EPOCH_TEST_N_STEPS:-10}"
  export ABFLOW_EPOCH_TEST_METRIC_WORKERS="${ABFLOW_EPOCH_TEST_METRIC_WORKERS:-8}"
fi
export ABFLOW_EPOCH_TEST_BASE_SEED="${ABFLOW_EPOCH_TEST_BASE_SEED:-2023}"
export ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS="${ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS:-off}"
export ABFLOW_EPOCH_TEST_KEEP_STRUCTURES="${ABFLOW_EPOCH_TEST_KEEP_STRUCTURES:-off}"
export ABFLOW_EPOCH_TEST_FAIL_FAST="on"

# Historical AutoTopK watcher is not part of the formal experiment.
export ABFLOW_AUTO_TOPK_EVAL="off"

# =====================================================================
# Unified standalone Test bridge
# =====================================================================
# A single checkpoint is evaluated cooperatively by the complete GPU list.
# Example: GPU_ID=2,3 -> test.sh -> torchrun world_size=2.
if [[ "$MODE" == "test" ]]; then
  CKPT="${4:-}"
  RESULT_DIR="${5:-}"
  TEST_JSON="${6:-${ABFLOW_EPOCH_TEST_JSON}}"

  if [[ -z "$CKPT" || -z "$RESULT_DIR" ]]; then
    echo "Usage: bash $0 test <EXP_ID> <GPU_LIST> <CKPT> <RESULT_DIR> [TEST_JSON]"
    exit 2
  fi
  [[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 2; }
  [[ -f "$TEST_JSON" ]] || { echo "ERROR: test json not found: $TEST_JSON" >&2; exit 2; }

  echo "============================================================"
  echo "Experiment : $EXP_ID"
  echo "Mode       : standalone cooperative DDP Test"
  echo "GPUs       : $GPU_ID"
  echo "Checkpoint : $CKPT"
  echo "Test JSON  : $TEST_JSON"
  echo "Result dir : $RESULT_DIR"
  echo "============================================================"

  GPU="$GPU_ID" bash "$PROJECT_ROOT/scripts/test/test.sh"     "$CKPT" "$TEST_JSON" "$RESULT_DIR" rabd "$ABFLOW_EPOCH_TEST_SURF"
  exit 0
fi

# Observational diagnostics already built into the accepted trainer.
export ABFLOW_DIAGNOSTIC_FILE="on"
export ABFLOW_CONDITION_DIAGNOSTICS="on"
export ABFLOW_MECHANISM_DIAGNOSTICS="on"
export ABFLOW_GRAD_CONFLICT_DIAGNOSTICS="${ABFLOW_GRAD_CONFLICT_DIAGNOSTICS:-off}"

# =====================================================================
# Resolve the requested result root into one ordinary train.py JSON.
# This is the only generated file needed because train.sh accepts save_dir
# through JSON rather than ABFLOW_RUN_ROOT directly.
# =====================================================================
RUN_ROOT="${ABFLOW_RUN_ROOT:-${PROJECT_ROOT}/results_module}"
RUN_DIR="${RUN_ROOT}/${EXP_ID}"
CONFIG_DIR="${RUN_ROOT}/generated_configs"
RUN_CONFIG="${CONFIG_DIR}/${EXP_ID}.json"

mkdir -p "$CONFIG_DIR"

FORCE_SCRATCH="${ABFLOW_FORCE_SCRATCH:-off}"
RESUME_FROM_ENV="${ABFLOW_RESUME_CHECKPOINT:-}"

python - "$BASE_CONFIG" "$RUN_CONFIG" "$RUN_DIR" "$FORCE_SCRATCH" "$RESUME_FROM_ENV" <<'PY'
import json, os, sys

src, dst, save_dir, force_scratch, resume_env = sys.argv[1:6]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Runtime-only flowtest epoch override.  The three-phase protocol remains
# Train -> Validation -> Test for EVERY epoch because interval=1 is enforced
# by AbFlowTrainer when ABFLOW_THREE_PHASE_PROTOCOL=on.
flowtest_max_epoch = os.environ.get("ABFLOW_FLOWTEST_MAX_EPOCH", "").strip()
if "FLOWTEST" in str(save_dir) and flowtest_max_epoch:
    cfg["max_epoch"] = int(flowtest_max_epoch)

is_on = force_scratch.strip().lower() in {"1","true","yes","y","on"}

cfg["save_dir"] = save_dir
if is_on:
    cfg["resume_checkpoint"] = ""
else:
    cfg["resume_checkpoint"] = resume_env.strip() or str(
        cfg.get("resume_checkpoint", "") or ""
    ).strip()

resume = cfg["resume_checkpoint"]
if resume and not os.path.isfile(resume):
    raise FileNotFoundError(f"resume checkpoint not found: {resume}")

os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

echo "============================================================"
echo "Experiment : $EXP_ID"
echo "Mode       : $MODE"
echo "GPUs       : $GPU_ID"
echo "Run dir    : $RUN_DIR"
echo "Config     : $RUN_CONFIG"
echo "Protocol   : Train -> Validation -> Test EVERY epoch"
python - "$RUN_CONFIG" "$ABFLOW_EPOCH_TEST_JSON" <<'PYRUN'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    c = json.load(f)
print(f"Train set  : {c.get('train_set')}")
print(f"Valid set  : {c.get('valid_set')}")
print(f"Test set   : {sys.argv[2]}")
print(f"Global BS  : {c.get('batch_size')}")
print(f"Max epoch  : {c.get('max_epoch')}")
print(f"Save topk  : {c.get('save_topk')}")
PYRUN
echo "Test BS    : $ABFLOW_EPOCH_TEST_BATCH_SIZE (logical generation batch)"
echo "Test steps : $ABFLOW_EPOCH_TEST_N_STEPS"
echo "Checkpoint : non-reentrant DDP-safe + exact BF16 autocast"
echo "Tri outer  : $ABFLOW_TRIANGLE_ATTENTION_CHUNK_SIZE"
echo "Tri backend: $ABFLOW_TRIANGLE_ATTN_BACKEND"
echo "Full logits: ${ABFLOW_TRIANGLE_FULL_LOGITS_LIMIT_MB} MB"
echo "LMA chunks : Q=${ABFLOW_TRIANGLE_LMA_Q_CHUNK_SIZE} KV=${ABFLOW_TRIANGLE_LMA_KV_CHUNK_SIZE}"
echo "Pair ckpt   : $ABFLOW_PAIRFORMER_CHECKPOINT_MODE (v118 backward-proven path)"
if [[ "$MODE" == "flowtest" ]]; then
  echo "Deep trace  : OFF"
  echo "Perf profile: $ABFLOW_PERF_DIAGNOSTICS first=${ABFLOW_PERF_DIAGNOSTIC_STEPS} steps"
fi
echo "AutoTopK   : OFF"
echo "============================================================"

cd "$PROJECT_ROOT"
GPU="$GPU_ID" bash scripts/train/train.sh "$RUN_CONFIG"
