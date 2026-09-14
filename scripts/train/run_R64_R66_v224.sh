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
  bash scripts/train/run_R64_R66_v224.sh <config.json> \
    --gpus 2,3 --port 29764 [--resume /path/version_N/checkpoint/last_stepXXXX.pt]
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
        "runtime.gpus is forbidden in V224 configs; pass physical GPUs with --gpus"
    )
output = training['output_dir']
if not os.path.isabs(output):
    output = os.path.abspath(os.path.join(root, output))
# Backward-compatible fallback only. Formal V224 configs leave this empty.
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

# V224 task-localized relational diagnostics.  These never alter losses, optimizer, sampler,
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
export ABFLOW_GEOMETRY_AUTHORITY_UPDATE_ALERT="${ABFLOW_GEOMETRY_AUTHORITY_UPDATE_ALERT:-100}"
# R53 e31-e34 failed by distribution-wide lever-arm drift long before 500A.
# These are logging thresholds only; they never clip, stop or rescale training.
export ABFLOW_GEOMETRY_DRIFT_DX_RMS_ALERT="${ABFLOW_GEOMETRY_DRIFT_DX_RMS_ALERT:-5}"
export ABFLOW_GEOMETRY_DRIFT_D_MAX_ALERT="${ABFLOW_GEOMETRY_DRIFT_D_MAX_ALERT:-120}"
export ABFLOW_GEOMETRY_AUTHORITY_ALERT_MAX_PER_EPOCH="${ABFLOW_GEOMETRY_AUTHORITY_ALERT_MAX_PER_EPOCH:-3}"


# Log hygiene.
export ABFLOW_TQDM="${ABFLOW_TQDM:-on}"
export ABFLOW_SCI_LOG_FIRST_STEPS="${ABFLOW_SCI_LOG_FIRST_STEPS:-1}"
export ABFLOW_SCI_LOG_INTERVAL="${ABFLOW_SCI_LOG_INTERVAL:-40}"
export ABFLOW_RUNTIME_GUARD_STEPS="${ABFLOW_RUNTIME_GUARD_STEPS:-1}"

# V224 formal Test failure contract. Infrastructure/protocol failures must abort;
# finite model-output invalidity remains observation-only. train.py asserts the same
# contract again so direct invocation cannot silently diverge.
export ABFLOW_EPOCH_TEST_FAIL_FAST="on"
export ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY="record_and_continue"
export ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS="${ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS:-on}"

# Single logging authority: everything visible in tmux from this point onward
# (Python logger, raw print, tqdm, warnings, forensic lines and tracebacks) is
# also appended to the canonical version_N/run_time.log exactly once.
exec > >(stdbuf -oL -eL tee -a "$RUN_LOG") 2>&1

echo "[RunLog] canonical=$RUN_LOG latest=$LATEST_LOG"
echo "[RunVersion] fixed_version=$VERSION dir=$RUN_DIR"
echo "[RunConfig] config=$CONFIG_PATH"
echo "[RunResources] physical_gpus=$GPU_CSV nproc=$NPROC global_batch=$GLOBAL_BATCH_SIZE local_batch=$LOCAL_BATCH_SIZE master_addr=$MASTER_ADDR port=$MASTER_PORT nnodes=$NNODES omp=$OMP_THREADS"
echo "[RunResume] checkpoint=${RESUME_CHECKPOINT:-scratch}"
echo "[Logging] train_progress=$ABFLOW_TQDM val_progress=$ABFLOW_TQDM test_progress=$ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS controller_interval=$ABFLOW_COORD_AUDIT_INTERVAL diagnostics=compact"
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
    raise SystemExit('V224 identity preflight failed: experiment.id is missing')
if not (exp_id == config_stem == output_id):
    raise SystemExit(
        'V224 identity preflight failed:\n'
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
            'V224 generation provenance preflight failed:\n'
            f'  experiment.id={exp_id}\n'
            f'  generation.save_dir basename={generation_id}\n'
            'Generation artifacts must not inherit an older experiment identity.'
        )
protocol = str(exp.get('protocol', '') or '').strip()
if protocol != 'formal_train_val_test':
    raise SystemExit(
        f"V224 requires experiment.protocol='formal_train_val_test'; got {protocol!r}"
    )
test_set = str(cfg.get('data', {}).get('test', {}).get('set', '') or '').strip()
if os.path.basename(test_set) != 'test.json':
    raise SystemExit(
        f"V224 requires data.test.set to remain observational test.json; got {test_set!r}"
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
  echo "V224 requires the already-installed V10 stable finite-gradient norm fallback in trainer/abs_trainer.py" >&2
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
    'scripts/train/run_R64_R66_v224.sh',
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
state=sp.get('coordinate_state', {})
pc=sp.get('pair_coordinate', {})
cc=sp.get('coordinate_controller', {})
loss=cfg.get('loss', {})
disto=loss.get('distogram', {})
anchor=loss.get('coarse_anchor_distance', {})
sl=loss.get('smooth_lddt', {})
r3=cfg['model']['r05']['r3']

state_mode=str(state.get('mode','legacy_dual') or 'legacy_dual').strip().lower()
if state_mode != 'scoreflow_single_endpoint':
    raise SystemExit(
        "V224 requires coordinate_state.mode='scoreflow_single_endpoint'; "
        f"got {state_mode!r}"
    )
if str(state.get('clean_endpoint_authority','')) != 'pred_X_only':
    raise SystemExit('V224 requires clean_endpoint_authority=pred_X_only')

pc_mode=str(pc.get('mode','') or '').strip().lower()
if pc_mode != 'direct_shared':
    raise SystemExit(f"V224 requires pair_coordinate.mode='direct_shared'; got {pc_mode!r}")
if 'delta_bound' in pc:
    raise SystemExit('V224 direct_shared must not carry pair_coordinate.delta_bound')
cc_mode=str(cc.get('mode','') or '').strip().lower()
if cc_mode != 'egnn_prenorm_raw':
    raise SystemExit(f'V224 requires coordinate_controller.mode=egnn_prenorm_raw; got {cc_mode!r}')

disto_weight=float(disto.get('weight',0.0))
disto_scope=str(disto.get('pair_scope','') or '').strip().lower()
disto_reduction=str(disto.get('reduction','pair_mean') or 'pair_mean').strip().lower()
if disto_weight > 0.0 and (disto_scope != 'generation_anchored' or disto_reduction != 'relation_balanced'):
    raise SystemExit(
        'V224 trainable Distogram requires generation_anchored + relation_balanced; '
        f'got weight={disto_weight:g}, scope={disto_scope!r}, reduction={disto_reduction!r}'
    )

anchor_weight=float(anchor.get('weight',0.0))
anchor_cutoff=float(anchor.get('cutoff_A',21.6875))
anchor_balance=str(anchor.get('relation_balance',''))
coord_scale=float(r3.get('coordinate_scaling',0.0))
if anchor_weight > 0.0:
    if anchor_balance != 'DF_DA_equal':
        raise SystemExit('V224 coarse anchor requires relation_balance=DF_DA_equal')
    if abs(anchor_cutoff - float(disto.get('max_bin',21.6875))) > 1e-6:
        raise SystemExit('V224 coarse anchor cutoff must match Distogram max_bin support')
    if abs(coord_scale - 0.1) > 1e-8:
        raise SystemExit('V224 formal coarse anchor expects existing R3 coordinate_scaling=0.1')

sl_weight=float(sl.get('weight',0.0))
if sl_weight != 0.0:
    raise SystemExit(
        'V224 R64-R66 formal suite disables smooth-lDDT: current DF/DA diagnostics are saturated at 1.0'
    )

print(
    '[PairCoordinateContract] '
    f'mode={pc_mode} state_pair=full coordinate_pair=direct_shared '
    'pair_adapter_init=zero same_edge_message_for_state_and_geometry=1'
)
print(
    '[CoordinateControllerContract] '
    f'mode={cc_mode} scalar_activation=identity scalar_bound=none '
    'action_input_norm=layer_norm relative_vector=raw_r05_cartesian '
    'coordinate_aggregation=mean no_tanh=1 no_coordinate_clipping=1 '
    'no_trust_radius=1 no_hand_tuned_step_scale=1'
)
print(
    '[SingleEndpointContract] '
    f'mode={state_mode} endpoint_authority=pred_X_only '
    'shadow_geometry=intra_round_latent_only carrier=analytic_from_endpoint '
    'terminal_readout=integrated_endpoint_generated_region_only fixed_context_transform=0'
)
print(
    '[DistogramContract] '
    f'weight={disto_weight:g} scope={disto_scope} reduction={disto_reduction}'
)
print(
    '[CoarseAnchorContract] '
    f'weight={anchor_weight:g} cutoff_A={anchor_cutoff:g} '
    f'relation_balance={anchor_balance or "off"} scale={coord_scale:g}'
)
PY

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$PROJECT_ROOT/train.py" \
  --config "$CONFIG_PATH"
