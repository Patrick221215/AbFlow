#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CONFIG_PATH=${1:-}
shift || true

GPU_CSV=""
MASTER_PORT=""
MASTER_ADDR=""
RESUME_CHECKPOINT=""

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/train/run_R53_R55_v221.sh <config.json> \
    --gpus 3 --port 29747 [--resume /path/version_N/checkpoint/last_stepXXXX.pt]
  # or: --gpus 2,3 / --gpus 0,2,5,7

GPU ids are execution resources and MUST be supplied on the command line.
They are intentionally not stored in the scientific JSON.
USAGE
}

[[ -n "$CONFIG_PATH" ]] || { usage; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_CSV=${2:-}; shift 2 ;;
    --port) MASTER_PORT=${2:-}; shift 2 ;;
    --master-addr) MASTER_ADDR=${2:-}; shift 2 ;;
    --resume) RESUME_CHECKPOINT=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$GPU_CSV" ]] || { echo "--gpus is required" >&2; usage; exit 2; }
[[ "$GPU_CSV" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "--gpus must be one or more comma-separated physical GPU ids, e.g. 3 or 2,3; got '$GPU_CSV'" >&2
  exit 2
}
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
NPROC=${#GPU_IDS[@]}
(( NPROC >= 1 )) || { echo "No GPU ids parsed" >&2; exit 2; }
# Duplicate physical GPU ids are always a launcher error.
if [[ $(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l) -ne $NPROC ]]; then
  echo "Duplicate GPU id in --gpus '$GPU_CSV'" >&2
  exit 2
fi

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$PROJECT_ROOT/${CONFIG_PATH#./}"
fi
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 2; }

# JSON owns scientific/training protocol.  Physical GPU placement does not.
eval "$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import json, os, shlex, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8'))
root = sys.argv[2]
runtime = cfg.get('runtime', {})
training = cfg['training']
if 'gpus' in runtime:
    raise SystemExit(
        "runtime.gpus is forbidden in V221 configs; pass physical GPUs with --gpus"
    )
output = training['output_dir']
if not os.path.isabs(output):
    output = os.path.abspath(os.path.join(root, output))
# Backward-compatible fallback only. Formal V221 configs leave this empty.
resume = training.get('schedule', {}).get('resume_checkpoint', '') or ''
if resume and not os.path.isabs(resume):
    resume = os.path.abspath(os.path.join(root, resume))
values = {
    'CFG_MASTER_ADDR': runtime.get('master_addr', '127.0.0.1'),
    'CFG_MASTER_PORT': runtime.get('master_port', ''),
    'NNODES': runtime.get('nnodes', 1),
    'OMP_THREADS': runtime.get('omp_num_threads', 2),
    'CUDA_ALLOC': runtime.get('cuda_allocator', 'max_split_size_mb:128'),
    'OUTPUT_ROOT': output,
    'GLOBAL_BATCH_SIZE': training.get('loader', {}).get('batch_size', ''),
    'CFG_RESUME_CHECKPOINT': resume,
}
for key, value in values.items():
    print(f"{key}=" + shlex.quote(str(value)))
PY
)"

MASTER_ADDR=${MASTER_ADDR:-$CFG_MASTER_ADDR}
MASTER_PORT=${MASTER_PORT:-$CFG_MASTER_PORT}
[[ -n "$MASTER_PORT" ]] || {
  echo "--port is required when runtime.master_port is not present" >&2
  exit 2
}
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || { echo "Invalid --port '$MASTER_PORT'" >&2; exit 2; }
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-$CFG_RESUME_CHECKPOINT}

# Batch/world-size contract. Physical GPU ids may be arbitrary, but the formal
# global optimizer batch must split exactly across the selected workers.
[[ "$GLOBAL_BATCH_SIZE" =~ ^[0-9]+$ ]] || {
  echo "Invalid training.loader.batch_size='$GLOBAL_BATCH_SIZE' in $CONFIG_PATH" >&2
  exit 2
}
(( GLOBAL_BATCH_SIZE > 0 )) || { echo "global batch must be > 0" >&2; exit 2; }
if (( GLOBAL_BATCH_SIZE % NPROC != 0 )); then
  echo "global batch $GLOBAL_BATCH_SIZE is not divisible by selected GPU count $NPROC" >&2
  echo "Choose a GPU count dividing $GLOBAL_BATCH_SIZE or change the scientific batch config explicitly." >&2
  exit 2
fi
LOCAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NPROC))

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# Resume reuses the checkpoint's version directory. Scratch reserves version_N.
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || {
    echo "resume checkpoint not found: $RESUME_CHECKPOINT" >&2
    exit 2
  }
  RUN_DIR=$(dirname "$(dirname "$RESUME_CHECKPOINT")")
  VERSION_BASE=$(basename "$RUN_DIR")
  [[ "$VERSION_BASE" =~ ^version_([0-9]+)$ ]] || {
    echo "resume checkpoint must live under version_N/checkpoint: $RESUME_CHECKPOINT" >&2
    exit 2
  }
  VERSION=${BASH_REMATCH[1]}
  RESUME_ROOT=$(dirname "$RUN_DIR")
  if [[ "$(realpath "$RESUME_ROOT")" != "$(realpath "$OUTPUT_ROOT")" ]]; then
    echo "resume checkpoint experiment root does not match config output_dir:" >&2
    echo "  checkpoint root: $RESUME_ROOT" >&2
    echo "  config output:   $OUTPUT_ROOT" >&2
    echo "Cross-experiment resume is forbidden; use a dedicated recovery config/run." >&2
    exit 2
  fi
  unset ABFLOW_FIXED_VERSION || true
else
  VERSION=0
  while ! mkdir "$OUTPUT_ROOT/version_$VERSION" 2>/dev/null; do
    VERSION=$((VERSION + 1))
  done
  RUN_DIR="$OUTPUT_ROOT/version_$VERSION"
  export ABFLOW_FIXED_VERSION="$VERSION"
fi

RUN_LOG="$RUN_DIR/run_time.log"
LATEST_LOG="$OUTPUT_ROOT/run_time.log"
mkdir -p "$RUN_DIR"
ln -sfn "version_$VERSION/run_time.log" "$LATEST_LOG"

export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_CSV"
export ABFLOW_NPROC_PER_NODE="$NPROC"
export ABFLOW_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
export OMP_NUM_THREADS="$OMP_THREADS"
export PYTHONUNBUFFERED=1
[[ -n "$CUDA_ALLOC" ]] && export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC"

# V221 compact diagnostics.  These never alter losses, optimizer, sampler,
# checkpoint selection, or Train->Val->Test ordering.
export ABFLOW_GEOMETRY_FORENSICS="${ABFLOW_GEOMETRY_FORENSICS:-on}"
export ABFLOW_SAMPLE_FORENSICS="${ABFLOW_SAMPLE_FORENSICS:-on}"
export ABFLOW_SAMPLE_FORENSICS_THRESHOLD_A="${ABFLOW_SAMPLE_FORENSICS_THRESHOLD_A:-500}"
export ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD="${ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD:-10000}"
export ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH="${ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH:-3}"
export ABFLOW_GRAD_OVERFLOW_LOG_LIMIT="${ABFLOW_GRAD_OVERFLOW_LOG_LIMIT:-3}"

# Compact controller audit: first two calls + every 40 calls.
export ABFLOW_COORD_AUDIT_INTERVAL="${ABFLOW_COORD_AUDIT_INTERVAL:-40}"
export ABFLOW_COORD_AUDIT_FIRST_STEPS="${ABFLOW_COORD_AUDIT_FIRST_STEPS:-2}"

# Disable duplicate periodic GeometryAuthority; retain failure-only alerts.
export ABFLOW_GEOMETRY_AUTHORITY_INTERVAL=0
export ABFLOW_GEOMETRY_AUTHORITY_BASE_ALERT="${ABFLOW_GEOMETRY_AUTHORITY_BASE_ALERT:-64}"
export ABFLOW_GEOMETRY_AUTHORITY_UPDATE_ALERT="${ABFLOW_GEOMETRY_AUTHORITY_UPDATE_ALERT:-500}"
export ABFLOW_GEOMETRY_AUTHORITY_ALERT_MAX_PER_EPOCH="${ABFLOW_GEOMETRY_AUTHORITY_ALERT_MAX_PER_EPOCH:-3}"


# Log hygiene.
export ABFLOW_TQDM="${ABFLOW_TQDM:-off}"
export ABFLOW_SCI_LOG_FIRST_STEPS="${ABFLOW_SCI_LOG_FIRST_STEPS:-2}"
export ABFLOW_SCI_LOG_INTERVAL="${ABFLOW_SCI_LOG_INTERVAL:-40}"
export ABFLOW_RUNTIME_GUARD_STEPS="${ABFLOW_RUNTIME_GUARD_STEPS:-1}"

# V221 formal Test failure contract. Infrastructure/protocol failures must abort;
# finite model-output invalidity remains observation-only. train.py asserts the same
# contract again so direct invocation cannot silently diverge.
export ABFLOW_EPOCH_TEST_FAIL_FAST="on"
export ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY="record_and_continue"

# Single logging authority: everything visible in tmux from this point onward
# (Python logger, raw print, tqdm, warnings, forensic lines and tracebacks) is
# also appended to the canonical version_N/run_time.log exactly once.
exec > >(stdbuf -oL -eL tee -a "$RUN_LOG") 2>&1

echo "[RunLog] canonical=$RUN_LOG latest=$LATEST_LOG"
echo "[RunVersion] fixed_version=$VERSION dir=$RUN_DIR"
echo "[RunConfig] config=$CONFIG_PATH"
echo "[RunResources] physical_gpus=$GPU_CSV nproc=$NPROC global_batch=$GLOBAL_BATCH_SIZE local_batch=$LOCAL_BATCH_SIZE master_addr=$MASTER_ADDR port=$MASTER_PORT nnodes=$NNODES omp=$OMP_THREADS"
echo "[RunResume] checkpoint=${RESUME_CHECKPOINT:-scratch}"
echo "[Diagnostics] controller_interval=$ABFLOW_COORD_AUDIT_INTERVAL first_steps=$ABFLOW_COORD_AUDIT_FIRST_STEPS grad_authority=disabled geometry_periodic=off geometry_alert_A=$ABFLOW_GEOMETRY_AUTHORITY_UPDATE_ALERT tqdm=$ABFLOW_TQDM"
echo "[TrainValTestContract] order=train->validation->test checkpoint_selection=validation test_metrics=observation_only test_steps=10 test_seed=2023 infra_fail_fast=$ABFLOW_EPOCH_TEST_FAIL_FAST model_invalid=$ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY"

# Fail before GPU training on protocol/provenance regressions.
python - "$CONFIG_PATH" "$OUTPUT_ROOT" <<'PYIDENTITY'
import json, os, sys
config_path, output_root = sys.argv[1:3]
cfg = json.load(open(config_path, encoding='utf-8'))
exp = cfg.get('experiment', {})
exp_id = str(exp.get('id', '') or '').strip()
config_stem = os.path.splitext(os.path.basename(config_path))[0]
output_id = os.path.basename(os.path.normpath(output_root))
if not exp_id:
    raise SystemExit('V221 identity preflight failed: experiment.id is missing')
if not (exp_id == config_stem == output_id):
    raise SystemExit(
        'V221 identity preflight failed:\n'
        f'  experiment.id={exp_id}\n'
        f'  config_stem={config_stem}\n'
        f'  output_dir_id={output_id}\n'
        'All three must match exactly.'
    )
generation_dir = str(cfg.get('generation', {}).get('save_dir', '') or '').strip()
if generation_dir:
    generation_id = os.path.basename(os.path.normpath(generation_dir))
    if generation_id != exp_id:
        raise SystemExit(
            'V221 generation provenance preflight failed:\n'
            f'  experiment.id={exp_id}\n'
            f'  generation.save_dir basename={generation_id}\n'
            'Generation artifacts must not inherit an older experiment identity.'
        )
protocol = str(exp.get('protocol', '') or '').strip()
if protocol != 'formal_train_val_test':
    raise SystemExit(
        f"V221 requires experiment.protocol='formal_train_val_test'; got {protocol!r}"
    )
test_set = str(cfg.get('data', {}).get('test', {}).get('set', '') or '').strip()
if os.path.basename(test_set) != 'test.json':
    raise SystemExit(
        f"V221 requires data.test.set to remain held-out test.json; got {test_set!r}"
    )
print(f'[ExperimentIdentity] PASS id={exp_id} protocol={protocol} test={test_set}')
PYIDENTITY

# Fail before GPU training on syntax/infrastructure regressions.
python -m py_compile \
  "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" \
  "$PROJECT_ROOT/models/modules/am_enc.py" \
  "$PROJECT_ROOT/models/modules/am_egnn.py" \
  "$PROJECT_ROOT/trainer/AbFlow_trainer.py" \
  "$PROJECT_ROOT/train.py"
if ! grep -q "GradientNormOverflowRecovered" "$PROJECT_ROOT/trainer/abs_trainer.py"; then
  echo "V221 requires the already-installed V10 stable finite-gradient norm fallback in trainer/abs_trainer.py" >&2
  exit 2
fi
echo "[Preflight] py_compile=PASS stable_grad_norm=PASS config_gpu_authority=CLI"
python - "$PROJECT_ROOT" <<'PYHASH'
import hashlib, os, sys
root=sys.argv[1]
for rel in [
    'models/AbFlow/AbFlow_model.py',
    'models/modules/am_enc.py',
    'models/modules/am_egnn.py',
    'trainer/AbFlow_trainer.py',
    'train.py',
    'scripts/train/run_R53_R55_v221.sh',
]:
    p=os.path.join(root,rel)
    if os.path.isfile(p):
        h=hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
        print(f'[SourceSHA256] {rel} {h}')
PYHASH

python - "$CONFIG_PATH" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1], encoding='utf-8'))
sp=cfg['model']['representation']['single_pair']
pc=sp.get('pair_coordinate', {})
cc=sp.get('coordinate_controller', {})
sl=cfg.get('loss', {}).get('smooth_lddt', {})

pc_mode=str(pc.get('mode','') or '').strip().lower()
if pc_mode != 'direct_shared':
    raise SystemExit(
        f"V221 requires pair_coordinate.mode='direct_shared'; got {pc_mode!r}"
    )
if 'delta_bound' in pc:
    raise SystemExit(
        'V221 direct_shared must not carry stale pair_coordinate.delta_bound; '
        'there is no bounded Pair residual in the formal controller.'
    )

cc_mode=str(cc.get('mode','') or '').strip().lower()
if cc_mode != 'egnn_prenorm_raw':
    raise SystemExit(
        'V221 requires coordinate_controller.mode=egnn_prenorm_raw; '
        f'got {cc_mode!r}'
    )

sl_weight=float(sl.get('weight', 0.0))
sl_source=str(sl.get('prediction_source', sl.get('target', 'pred_design_endpoint')) or 'pred_design_endpoint')
allowed={'pred_design_endpoint','carrier_implied_endpoint'}
if sl_source not in allowed:
    raise SystemExit(f'V221 invalid smooth-lDDT prediction_source={sl_source!r}; allowed={sorted(allowed)}')

print(
    '[PairCoordinateContract] '
    f'mode={pc_mode} state_pair=full coordinate_pair=direct_shared '
    'pair_adapter_init=zero same_edge_message_for_state_and_geometry=1'
)
print(
    '[CoordinateControllerContract] '
    f'mode={cc_mode} scalar_activation=identity scalar_bound=none action_input_norm=layer_norm '
    'relative_vector=raw_r05_cartesian raw_distance_features=preserved '
    'coordinate_aggregation=mean no_tanh=1 no_coordinate_clipping=1 '
    'no_trust_radius=1 no_hand_tuned_step_scale=1'
)
print(
    '[SmoothLDDTContract] '
    f'weight={sl_weight:g} prediction_source={sl_source} '
    'fixed_context=restored sampler_alignment=' +
    ('exact_carrier_endpoint' if sl_source == 'carrier_implied_endpoint' else 'legacy_full_pred_X')
)
PY

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$PROJECT_ROOT/train.py" \
  --config "$CONFIG_PATH"
