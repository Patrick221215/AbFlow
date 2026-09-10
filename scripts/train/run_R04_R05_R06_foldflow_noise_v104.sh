#!/usr/bin/env bash
# V202_JSON_BATCH_MEMORY_CLOSURE
# Three cumulative experiments only.  Each experiment owns exactly two GPUs.
# R28 -> 2,3 ; R29 -> 4,5 ; R30 -> 6,7
set -euo pipefail

MODE=${1:-}
EXP_ID=${2:-}
GPU_ID=${3:-}
BASE_CONFIG=${4:-}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CFG_ROOT=${CFG_ROOT:-scripts/train/configs/R05_ABX_NATIVE_V200_PERSISTENT_PAIR_LOCALIZED}
RUN_ROOT=${ABFLOW_RUN_ROOT:-$PROJECT_ROOT/results_module}
export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT"

case "$EXP_ID" in
  R28_R05_ABX_NATIVE_PAIR_EGNN_U02)
    RECOMMENDED_GPU="2,3"; DEFAULT_PORT=29628 ;;
  R29_R05_ABX_NATIVE_PAIR_EGNN_DISTOGRAM_U02)
    RECOMMENDED_GPU="4,5"; DEFAULT_PORT=29629 ;;
  R30_R05_ABX_NATIVE_PAIR_EGNN_DISTOGRAM_LDDT_U02)
    RECOMMENDED_GPU="6,7"; DEFAULT_PORT=29630 ;;
  *) echo "Unknown EXP_ID: $EXP_ID" >&2; exit 2 ;;
esac
[[ "$MODE" == "train" ]] || {
  echo "Usage: bash $0 train <EXP_ID> [GPU_PAIR] [config.json]" >&2
  exit 2
}

# Canonical tmux-visible stdout/stderr log + launcher-owned DDP version.
# Reserve version_N atomically before torchrun, so both ranks use the same
# trainer directory and the same log. Historical failed versions are retained.
RUN_DIR="$RUN_ROOT/$EXP_ID"
mkdir -p "$RUN_DIR"
RUN_VERSION=0
while ! mkdir "$RUN_DIR/version_${RUN_VERSION}" 2>/dev/null; do
  RUN_VERSION=$((RUN_VERSION + 1))
done
export ABFLOW_FIXED_VERSION="$RUN_VERSION"
RUN_VERSION_DIR="$RUN_DIR/version_${RUN_VERSION}"
RUN_TIME_LOG="$RUN_VERSION_DIR/run_time.log"
LATEST_RUN_LOG="$RUN_DIR/run_time.log"
export ABFLOW_RUN_TIME_LOG="$RUN_TIME_LOG"
: > "$RUN_TIME_LOG"
# Keep a stable path for tmux/tail workflows while preserving every run under
# version_N.  If an older launcher left a regular run_time.log, archive it once
# instead of deleting historical evidence.
if [[ -e "$LATEST_RUN_LOG" && ! -L "$LATEST_RUN_LOG" ]]; then
  LEGACY_LOG="$RUN_DIR/run_time.pre_v193.$(date +%Y%m%d_%H%M%S).log"
  mv "$LATEST_RUN_LOG" "$LEGACY_LOG"
fi
ln -sfn "version_${RUN_VERSION}/run_time.log" "$LATEST_RUN_LOG"
export PYTHONUNBUFFERED=1
exec > >(stdbuf -oL -eL tee -a "$RUN_TIME_LOG") 2>&1
echo "[RunLog] canonical=$RUN_TIME_LOG latest=$LATEST_RUN_LOG"
echo "[RunVersion] EXP=$EXP_ID fixed_version=$ABFLOW_FIXED_VERSION dir=$RUN_VERSION_DIR"

GPU_ID=${GPU_ID:-$RECOMMENDED_GPU}
if [[ ! "$GPU_ID" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "V186 protocol: every configuration requires exactly two GPUs as A,B; got '$GPU_ID'" >&2
  exit 2
fi
IFS=',' read -r GPU_A GPU_B <<< "$GPU_ID"
[[ "$GPU_A" != "$GPU_B" ]] || { echo "V186 protocol: two distinct GPUs are required" >&2; exit 2; }
if [[ "$GPU_ID" != "$RECOMMENDED_GPU" ]]; then
  echo "[V186GPUAllocationWARN] $EXP_ID formal parallel allocation is $RECOMMENDED_GPU; using requested debug pair $GPU_ID" >&2
fi
PORT=${PORT:-$DEFAULT_PORT}
export PORT MASTER_PORT="$PORT"

if [[ -z "$BASE_CONFIG" ]]; then BASE_CONFIG="$CFG_ROOT/$EXP_ID.json"; fi
[[ -f "$BASE_CONFIG" ]] || { echo "Config not found: $BASE_CONFIG" >&2; exit 2; }

# JSON remains the architecture/loss/runtime source of truth.
eval "$(python - "$BASE_CONFIG" "$EXP_ID" <<'PYENV'
import json,shlex,sys
p,e=sys.argv[1:3]
c=json.load(open(p,encoding='utf-8'))
m=c.get('_experiment',{})
if m.get('exp_id')!=e:
    raise SystemExit(f'EXP_ID mismatch: {m.get("exp_id")} != {e}')
for k,v in m.get('runtime_env',{}).items():
    print('export '+k+'='+shlex.quote(str(v)))
PYENV
)"

# Frozen scientific contract.
[[ "$ABFLOW_SOURCE_MODE" == "pcs_rc" ]] || { echo "PCS-RC must stay on" >&2; exit 2; }
[[ "$ABFLOW_R3_NOISE_SCOPE" == "residue" ]] || { echo "R3 noise scope must be residue" >&2; exit 2; }
[[ "$ABFLOW_R3_G_MODE" == "foldflow_fixed_scaled" ]] || { echo "R05 g-mode must be foldflow_fixed_scaled" >&2; exit 2; }
[[ "$ABFLOW_R3_FIXED_G_SCALED" == "0.1" ]] || { echo "R05 fixed g_scaled must be 0.1" >&2; exit 2; }
[[ "$ABFLOW_R3_FLOW_COORDINATE_SCALING" == "0.1" ]] || { echo "R05 coordinate scaling must be 0.1" >&2; exit 2; }
[[ "$ABFLOW_R3_PATH_MIN_SIGMA" == "0.0" ]] || { echo "U02 path_min_sigma must be 0.0" >&2; exit 2; }
[[ "$ABFLOW_F01_HYBRID_T_MIN" == "0.20" ]] || { echo "U02 hybrid t_min must be 0.20" >&2; exit 2; }
[[ "$ABFLOW_SCOREFM_LOSS_MODE" == "f01_r3_endpoint_canonical_hybrid" ]] || exit 2
[[ "$ABFLOW_SCOREFM_SAMPLER_MODE" == "f01_canonical_carrier" ]] || exit 2
[[ "$ABFLOW_ABX_RECYCLING" == "off" ]] || { echo "AbX recycling is forbidden in V186" >&2; exit 2; }
[[ "$ABFLOW_SCOREFM_TIME_EMBED" == "on" ]] || { echo "R05 flow-time must stay on" >&2; exit 2; }
[[ "$ABFLOW_ABX_TIME_EMBED" == "on" ]] || { echo "AbX timestep embedding must stay on" >&2; exit 2; }
[[ "$ABFLOW_EPOCH_TEST" == "on" ]] || { echo "V203 formal protocol requires ABFLOW_EPOCH_TEST=on" >&2; exit 2; }
[[ "${ABFLOW_EPOCH_TEST_INTERVAL:-}" == "1" ]] || { echo "V203 formal protocol requires ABFLOW_EPOCH_TEST_INTERVAL=1" >&2; exit 2; }
[[ "${ABFLOW_EPOCH_TEST_N_STEPS:-}" == "10" ]] || { echo "V203 formal protocol requires ABFLOW_EPOCH_TEST_N_STEPS=10" >&2; exit 2; }
[[ "${ABFLOW_EPOCH_TEST_FAIL_FAST:-}" == "on" ]] || { echo "V203 formal protocol requires ABFLOW_EPOCH_TEST_FAIL_FAST=on" >&2; exit 2; }
[[ -n "${ABFLOW_EPOCH_TEST_JSON:-}" ]] || { echo "V203 formal protocol requires ABFLOW_EPOCH_TEST_JSON" >&2; exit 2; }
[[ "${ABFLOW_EPOCH_TEST_CDR:-H3}" == "H3" ]] || { echo "V207 formal Test requires ABFLOW_EPOCH_TEST_CDR=H3" >&2; exit 2; }

# V204: H3 task identity is a hard scientific/runtime contract.
# Never allow JSON -> argparse conversion to silently turn cdr into None or
# paratope into a scalar string; that changes Train/Val/Test semantics.
python - "$BASE_CONFIG" <<'PYTASK'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p, encoding='utf-8'))
expected = ['H3']
for key in ('cdr', 'paratope'):
    value = cfg.get(key, None)
    if value != expected:
        raise SystemExit(
            f"V204 task contract failed in source JSON: {key}={value!r}; "
            f"expected {expected!r}. Do not use null or scalar 'H3'."
        )
print(f"[V204TaskContract] source={p} cdr={cfg['cdr']} paratope={cfg['paratope']}")
PYTASK
[[ "$ABFLOW_CONDITION_DIAGNOSTICS" == "on" ]] || { echo "condition diagnostics must stay on" >&2; exit 2; }
[[ "$ABFLOW_GRAD_CONFLICT_DIAGNOSTICS" == "on" ]] || { echo "gradient diagnostics must stay on" >&2; exit 2; }
[[ "${ABFLOW_KABSCH_ALIGNMENT_MODE:-}" == "stopgrad_fp32" ]] || { echo "V200 formal requires ABFLOW_KABSCH_ALIGNMENT_MODE=stopgrad_fp32" >&2; exit 2; }
[[ "${ABFLOW_AUTOGRAD_ANOMALY_STEPS:-0}" == "0" ]] || { echo "V199 formal run requires ABFLOW_AUTOGRAD_ANOMALY_STEPS=0" >&2; exit 2; }
[[ "${ABFLOW_ABX_MEMORY_DIAGNOSTICS:-off}" == "off" ]] || { echo "V199 formal run requires ABFLOW_ABX_MEMORY_DIAGNOSTICS=off" >&2; exit 2; }
[[ "${ABFLOW_TORSION_DIAGNOSTICS:-off}" == "off" ]] || { echo "V199 formal run requires ABFLOW_TORSION_DIAGNOSTICS=off" >&2; exit 2; }

# V203 fixed Train -> Val -> Test protocol. Resolve relative test assets against project root.
resolve_project_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then printf '%s\n' "$p"; else printf '%s\n' "$PROJECT_ROOT/${p#./}"; fi
}
TEST_JSON_ABS=$(resolve_project_path "$ABFLOW_EPOCH_TEST_JSON")
TEST_PEP_ABS=$(resolve_project_path "${ABFLOW_EPOCH_TEST_PEP:-./datasets/RAbD/test.pkl}")
TEST_SURF_ABS=$(resolve_project_path "${ABFLOW_EPOCH_TEST_SURF:-./datasets/RAbD/test_surf.pkl}")
[[ -f "$TEST_JSON_ABS" ]] || { echo "V203 formal Test JSON not found: $TEST_JSON_ABS" >&2; exit 2; }
[[ -f "$TEST_PEP_ABS" ]] || { echo "V203 formal Test pep not found: $TEST_PEP_ABS" >&2; exit 2; }
[[ -f "$TEST_SURF_ABS" ]] || { echo "V203 formal Test surf not found: $TEST_SURF_ABS" >&2; exit 2; }
echo "[V203TestPreflight] interval=1 json=$ABFLOW_EPOCH_TEST_JSON pep=${ABFLOW_EPOCH_TEST_PEP:-./datasets/RAbD/test.pkl} surf=${ABFLOW_EPOCH_TEST_SURF:-./datasets/RAbD/test_surf.pkl} batch=${ABFLOW_EPOCH_TEST_BATCH_SIZE:-20} n_steps=$ABFLOW_EPOCH_TEST_N_STEPS seed=${ABFLOW_EPOCH_TEST_BASE_SEED:-2023} cdr=${ABFLOW_EPOCH_TEST_CDR:-H3} fail_fast=on"
python - <<'PYFORMAL'
import os
eps=float(os.environ.get('ABFLOW_TORSION_NORM_EPS','0'))
if not (eps > 0.0):
    raise SystemExit(f'V200 formal requires ABFLOW_TORSION_NORM_EPS>0, got {eps}')
audit=int(os.environ.get('ABFLOW_ABX_ATOM_MASK_AUDIT_CALLS','0'))
if audit != 1:
    raise SystemExit(f'V200 formal requires exactly one atom-mask contract audit, got {audit}')
profile=os.environ.get('ABFLOW_ABX_WIDTH_PROFILE','')
chunk=int(os.environ.get('ABFLOW_ABX_TRIANGLE_CHUNK_SIZE','0'))
if profile != 'localized':
    raise SystemExit(f'V202 formal requires ABFLOW_ABX_WIDTH_PROFILE=localized, got {profile!r}')
if chunk <= 0:
    raise SystemExit(
        f'V202 requires ABFLOW_ABX_TRIANGLE_CHUNK_SIZE>0 from JSON runtime_env, got {chunk}'
    )
print(f'[V202FormalConfig] torsion_eps={eps:.1e} atom_mask_audit={audit} memory_diag=off torsion_diag=off anomaly=0 profile={profile} triangle_chunk={chunk} chunk_source=json_runtime_env')
PYFORMAL

# Static implementation contract.
# IMPORTANT:
# Do not key preflight to a version comment such as V186/V187.  The model may
# receive a bug-fix-only revision while preserving the same scientific
# contract.  Check the actual required implementation symbols instead, and
# always print a useful error before exiting.
MODEL_FILE="$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"
TRAINER_FILE="$PROJECT_ROOT/trainer/AbFlow_trainer.py"
ABS_TRAINER_FILE="$PROJECT_ROOT/trainer/abs_trainer.py"
AMENC_FILE="$PROJECT_ROOT/models/modules/am_enc.py"
AMEGNN_FILE="$PROJECT_ROOT/models/modules/am_egnn.py"
MATCHER_FILE="$PROJECT_ROOT/models/AbFlow/abflow_r3_matcher.py"
NNUTILS_FILE="$PROJECT_ROOT/utils/nn_utils.py"

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || {
    echo "[V200PreflightFAIL] missing $label: $path" >&2
    exit 2
  }
}

require_symbol() {
  local path="$1"
  local symbol="$2"
  local label="$3"
  grep -Fq "$symbol" "$path" || {
    echo "[V200PreflightFAIL] $label missing required contract: $symbol" >&2
    exit 2
  }
}

require_file "$MODEL_FILE" "AbFlow model"
require_file "$MATCHER_FILE" "R3 matcher"
require_file "$TRAINER_FILE" "AbFlow trainer"
require_file "$ABS_TRAINER_FILE" "base trainer"
require_file "$AMENC_FILE" "AMEncoder"
require_file "$AMEGNN_FILE" "AM-EGNN"
require_file "$NNUTILS_FILE" "nn_utils"

# R05/U02 model contract.
for sym in   'f01_r3_endpoint_canonical_hybrid'   'f01_canonical_carrier'   'foldflow_fixed_scaled'   'noise_scope=self.r3_noise_scope'   'exact_carrier_scoreflow_step_gfree'   'import functools as fn'; do
  require_symbol "$MODEL_FILE" "$sym" "AbFlow_model.py"
done

# Matched R05/U02 matcher contract.
for sym in   'canonical_carrier_target_gfree'   'exact_carrier_scoreflow_step_gfree'   'foldflow_scaled_g_to_raw'; do
  require_symbol "$MATCHER_FILE" "$sym" "abflow_r3_matcher.py"
done

# Stable trainer and native pair->EGNN wiring.
require_symbol "$TRAINER_FILE" 'V185_STABLE_TRAINER_MINIMAL_ABX_DIAGNOSTICS' "AbFlow_trainer.py"
require_symbol "$TRAINER_FILE" 'V203_FORMAL_TRAIN_VAL_TEST_EVERY_EPOCH' "AbFlow_trainer.py"
require_symbol "$TRAINER_FILE" '_run_epoch_test' "AbFlow_trainer.py"
require_symbol "$TRAINER_FILE" 'generate_distributed' "AbFlow_trainer.py"
require_symbol "$TRAINER_FILE" 'run_cal_metrics_rank0' "AbFlow_trainer.py"
require_symbol "$TRAINER_FILE" 'FormalEpochTestPASS' "AbFlow_trainer.py"
require_symbol "$MODEL_FILE" 'ABFLOW_ABX_ACTIVATION_CHECKPOINT' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'ABFLOW_ABX_ANTIGEN_CONTEXT_MODE' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'V199_ABX_OBSERVED_ATOM_MASK_FORMAL_CLOSURE' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'V200_PERSISTENT_ABX_OUTSIDE_R05' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'V200_EXACT_PAIR_DISTANCE_KERNEL' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'V200_EXACT_TRIANGLE_CHUNKING' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'V200_VECTOR_TORSION_TABLES' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'V201_LOCALIZED_WIDTH_CONTRACT_CLOSURE' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'V202_JSON_RUNTIME_TRIANGLE_CHUNK' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'ABFLOW_TORSION_NORM_EPS' "AbFlow_model.py"
require_symbol "$MODEL_FILE" 'Xp_donor = torch.where' "AbFlow_model.py"
require_symbol "$MODEL_FILE" "self.batch_constants['xloss_mask'] = xloss_mask.bool()" "AbFlow_model.py"
require_symbol "$MODEL_FILE" "atom_observed_mask=self.batch_constants.get('xloss_mask')" "AbFlow_model.py"
require_symbol "$MODEL_FILE" '[AbXAtomMaskContract]' "AbFlow_model.py"
require_symbol "$MODEL_FILE" '[NonFiniteTensor]' "AbFlow_model.py"
require_symbol "$ABS_TRAINER_FILE" '[NonFiniteGrad]' "abs_trainer.py"
require_symbol "$AMENC_FILE" 'inter_edge_attr=None, surf_edge_attr=None' "am_enc.py"
require_symbol "$AMEGNN_FILE" 'Empty aligned surface edges are a valid no-message case' "am_egnn.py"
require_symbol "$NNUTILS_FILE" 'V194_KABSCH_STOPGRAD_FP32' "utils/nn_utils.py"
require_symbol "$NNUTILS_FILE" 'V195 AMP contract' "utils/nn_utils.py"
require_symbol "$NNUTILS_FILE" 'requires_grad=False' "utils/nn_utils.py"
require_symbol "$ABS_TRAINER_FILE" 'ABFLOW_AUTOGRAD_ANOMALY_STEPS' "abs_trainer.py"

if grep -Fq '_PairResidualAMEGCL' "$AMENC_FILE"; then
  echo "[V200PreflightFAIL] obsolete _PairResidualAMEGCL found in am_enc.py" >&2
  exit 2
fi
if grep -Fq '_PairResidualMSGCL' "$AMENC_FILE"; then
  echo "[V200PreflightFAIL] obsolete _PairResidualMSGCL found in am_enc.py" >&2
  exit 2
fi

echo "[V200PreflightPASS] formal model/matcher/trainer/native-pair contracts verified"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export ABFLOW_SELECTED_GPUS="$GPU_ID"
export ABFLOW_SELECTED_GPU_COUNT=2

# V202_JSON_BATCH_SOURCE_OF_TRUTH
# batch_size is owned exclusively by the selected JSON config.
# The launcher reads it, validates it against the DDP world size, and derives
# local_batch for logging/runtime diagnostics only.  It never rewrites or locks
# the global batch to a particular value (e.g. 56).
WORLD_SIZE="$ABFLOW_SELECTED_GPU_COUNT"
BATCH_SIZE=$(python - "$BASE_CONFIG" "$WORLD_SIZE" <<'PYBATCH'
import json, sys
p, world = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(p, encoding='utf-8'))
if 'batch_size' not in cfg:
    raise SystemExit('batch_size is missing from JSON config')
b = int(cfg['batch_size'])
if b <= 0:
    raise SystemExit(f'batch_size must be positive, got {b}')
if world <= 0:
    raise SystemExit(f'world_size must be positive, got {world}')
if b % world != 0:
    raise SystemExit(
        f'batch_size={b} from JSON is not divisible by world_size={world}; '
        'choose a JSON batch_size divisible by the selected GPU count'
    )
print(b)
PYBATCH
)
LOCAL_BATCH=$((BATCH_SIZE / WORLD_SIZE))
export ABFLOW_EFFECTIVE_GLOBAL_BATCH="$BATCH_SIZE"
export ABFLOW_LOCAL_BATCH_PER_GPU="$LOCAL_BATCH"
echo "[V202BatchContract] source=json global_batch=$BATCH_SIZE world_size=$WORLD_SIZE local_batch=$LOCAL_BATCH config=$BASE_CONFIG"

# PyTorch 1.11 memory-fragmentation guard.  This does not change model math.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# V200 executes the AbX trunk once per outer forward, outside R05 recurrence.
# Checkpointing therefore applies to only one Seqformer graph; keep it enabled
# once per outer forward. Memory safety is controlled by JSON batch_size plus
# JSON runtime_env triangle chunk; there is no launcher-owned batch constant.
[[ "$ABFLOW_DDP_FIND_UNUSED_PARAMETERS" == "off" ]] || {
  echo "V200 formal requires ABFLOW_DDP_FIND_UNUSED_PARAMETERS=off" >&2; exit 2;
}
[[ "$ABFLOW_DDP_STATIC_GRAPH" == "on" ]] || {
  echo "V199 formal AbX activation checkpoint requires ABFLOW_DDP_STATIC_GRAPH=on" >&2; exit 2;
}
export ABFLOW_FORCE_SCRATCH=${ABFLOW_FORCE_SCRATCH:-1}
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"

python - "$PROJECT_ROOT" <<'PYHASH'
import hashlib, os, sys
root=sys.argv[1]
for rel in [
    'models/AbFlow/AbFlow_model.py',
    'models/modules/am_enc.py',
    'models/modules/am_egnn.py',
    'trainer/AbFlow_trainer.py',
    'trainer/abs_trainer.py',
    'utils/nn_utils.py',
    'utils/epoch_test.py',
    'scripts/test/generate_epoch_test_ddp.py',
    'scripts/train/run_R04_R05_R06_foldflow_noise_v104.sh',
]:
    path=os.path.join(root, rel)
    digest=hashlib.sha256(open(path,'rb').read()).hexdigest()[:16]
    print(f'[SourceSHA256] {rel} {digest}')
PYHASH

GEN_DIR="$RUN_ROOT/generated_configs"
RUN_CONFIG="$GEN_DIR/$EXP_ID.json"
mkdir -p "$GEN_DIR" "$RUN_DIR"

python - "$BASE_CONFIG" "$RUN_CONFIG" "$RUN_DIR" <<'PYCFG'
import json, sys
src, dst, run_dir = sys.argv[1:4]
cfg = json.load(open(src, encoding='utf-8'))
cfg.pop('_experiment', None)

# V202: JSON owns train/runtime values.  The launcher changes only the output
# directory for this run and clears resume for an explicit scratch launch.
cfg['save_dir'] = run_dir
cfg.pop('resume_checkpoint', None)

required = {
    'batch_size': lambda v: int(v) > 0,
    'max_epoch': lambda v: int(v) == 200,
    'use_ema': lambda v: bool(v) is True,
    'ema_decay': lambda v: abs(float(v) - 0.999) < 1e-12,
    'amp': lambda v: bool(v) is True,
    'amp_dtype': lambda v: str(v).lower() in {'bf16', 'bfloat16'},
    'allow_tf32': lambda v: bool(v) is True,
}
for key, check in required.items():
    if key not in cfg:
        raise SystemExit(f'{key} is missing from JSON config')
    if not check(cfg[key]):
        raise SystemExit(f'V202 formal JSON contract failed: {key}={cfg[key]!r}')

# V204: preserve exact H3 task identity into the generated runtime JSON.
expected_task = ['H3']
for key in ('cdr', 'paratope'):
    value = cfg.get(key, None)
    if value != expected_task:
        raise SystemExit(
            f'V204 generated-config task contract failed before write: '
            f'{key}={value!r}; expected {expected_task!r}'
        )

json.dump(cfg, open(dst, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
# Read back the exact generated artifact; fail before launching expensive training.
check_cfg = json.load(open(dst, encoding='utf-8'))
for key in ('cdr', 'paratope'):
    if check_cfg.get(key) != expected_task:
        raise SystemExit(
            f'V204 generated-config task contract failed after write: '
            f'{key}={check_cfg.get(key)!r}; expected {expected_task!r}'
        )
print(f"[V204GeneratedTaskPASS] cdr={check_cfg['cdr']} paratope={check_cfg['paratope']}")
open(dst, 'a').write('\n')
print(
    f"[V202ConfigPASS] {dst} "
    f"batch={cfg['batch_size']} batch_source=json_preserved "
    f"epochs={cfg['max_epoch']} EMA={cfg['ema_decay']} "
    f"AMP={cfg['amp_dtype']} TF32={int(bool(cfg['allow_tf32']))}"
)
PYCFG

printf '%s\n' \
  "[V202RunContract] EXP=$EXP_ID GPUs=$GPU_ID world=$WORLD_SIZE global_batch=$BATCH_SIZE local_batch=$LOCAL_BATCH batch_source=json triangle_chunk=$ABFLOW_ABX_TRIANGLE_CHUNK_SIZE chunk_source=json_runtime_env port=$PORT allocator=$PYTORCH_CUDA_ALLOC_CONF" \
  "[V186Physics] PCS-RC + residue-R3 + U02 canonical carrier + physical recurrence=3" \
  "[V186R05Contract] loss=$ABFLOW_SCOREFM_LOSS_MODE sampler=$ABFLOW_SCOREFM_SAMPLER_MODE g_mode=$ABFLOW_R3_G_MODE g_scaled=$ABFLOW_R3_FIXED_G_SCALED coord_scale=$ABFLOW_R3_FLOW_COORDINATE_SCALING noise_scope=$ABFLOW_R3_NOISE_SCOPE path_min_sigma=$ABFLOW_R3_PATH_MIN_SIGMA u02_t_min=$ABFLOW_F01_HYBRID_T_MIN" \
  "[V186Representation] biological-single=R05-dynamic+AbX-single pair=AbX-z->native-ctx/inter/surf-edge_attr" \
  "[V202Execution] abx_trunk_per_outer=1 persistent_single=1 persistent_dense_pair=1 R05_recurrence=3 sparse_pair_gather=per_round" \
  "[V202AbXProfile] profile=$ABFLOW_ABX_WIDTH_PROFILE seq=256 pair=64 index=16 single_total=272 pair_total=96 seq_heads=8 triangle_heads=4 triangle_chunk=$ABFLOW_ABX_TRIANGLE_CHUNK_SIZE" \
  "[V186Time] same flow_t -> R05-time=on + AbX-time(single+pair)=on; AbX-recycling=off" \
  "[V186Aux] distogram=$ABFLOW_ABX_DISTOGRAM w=$ABFLOW_LOSS_DISTOGRAM_WEIGHT smooth_lddt=$ABFLOW_ABX_SMOOTH_LDDT w=$ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT" \
  "[V202FormalDiagnostics] condition=on grad_conflict=on grad_interval=$ABFLOW_GRAD_DIAGNOSTIC_INTERVAL science_first=$ABFLOW_SCI_LOG_FIRST_STEPS science_interval=$ABFLOW_SCI_LOG_INTERVAL outlier_threshold=$ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD memory_diag=$ABFLOW_ABX_MEMORY_DIAGNOSTICS torsion_diag=$ABFLOW_TORSION_DIAGNOSTICS anomaly_steps=$ABFLOW_AUTOGRAD_ANOMALY_STEPS" \
  "[V202Runtime] abx_checkpoint=$ABFLOW_ABX_ACTIVATION_CHECKPOINT ddp_static_graph=$ABFLOW_DDP_STATIC_GRAPH find_unused=$ABFLOW_DDP_FIND_UNUSED_PARAMETERS antigen_mode=$ABFLOW_ABX_ANTIGEN_CONTEXT_MODE antigen_cap=$ABFLOW_ABX_MAX_ANTIGEN finite_failfast=$ABFLOW_NONFINITE_FAILFAST finite_guard_calls=$ABFLOW_NONFINITE_GUARD_CALLS grad_guard_steps=$ABFLOW_GRAD_FINITE_GUARD_STEPS run_log=$RUN_TIME_LOG" \
  "[V202GeometryGrad] kabsch_alignment=$ABFLOW_KABSCH_ALIGNMENT_MODE geometry_loss_dtype=fp32 svd_backward=off" \
  "[V202TorsionContract] epsilon=$ABFLOW_TORSION_NORM_EPS formula=sqrt(sum_sq+epsilon) fp32=1 fixed_mask_only=1 donor_l2_normalize=1" \
  "[V202AtomMaskContract] trainval=xloss_mask donor_unresolved_atom=zero legacy_sample=ca_fill_fallback_once atom_mask_audit_calls=$ABFLOW_ABX_ATOM_MASK_AUDIT_CALLS ca_fill_tol2=$ABFLOW_ABX_CA_FILL_TOL2 R05_geometry=unchanged" \
  "[V203Eval] flow=Train->Val->Test exact-DDP-validation=$ABFLOW_DDP_VALIDATION deterministic-validation=$ABFLOW_DETERMINISTIC_VALIDATION epoch-test=$ABFLOW_EPOCH_TEST interval=$ABFLOW_EPOCH_TEST_INTERVAL test_json=$ABFLOW_EPOCH_TEST_JSON test_batch=$ABFLOW_EPOCH_TEST_BATCH_SIZE n_steps=$ABFLOW_EPOCH_TEST_N_STEPS seed=$ABFLOW_EPOCH_TEST_BASE_SEED cdr=${ABFLOW_EPOCH_TEST_CDR:-H3} fail_fast=$ABFLOW_EPOCH_TEST_FAIL_FAST"

if [[ "${ABFLOW_DRY_RUN:-0}" == "1" ]]; then
  echo '[DRY RUN] no training launched.'
  exit 0
fi
GPU="$GPU_ID" bash "$PROJECT_ROOT/scripts/train/train.sh" "$RUN_CONFIG"
