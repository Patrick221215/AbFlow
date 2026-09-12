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
  bash scripts/train/run_R31_R34_v213.sh <config.json> \
    --gpus 2,3 --port 29728 [--resume /path/version_N/checkpoint/last_stepXXXX.pt]

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
  echo "--gpus must be a comma-separated list such as 2,3; got '$GPU_CSV'" >&2
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
        "runtime.gpus is forbidden in V213 configs; pass physical GPUs with --gpus"
    )
output = training['output_dir']
if not os.path.isabs(output):
    output = os.path.abspath(os.path.join(root, output))
# Backward-compatible fallback only. Formal V213 configs leave this empty.
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

# Formal stability/forensic settings.  These are runtime diagnostics, not
# scientific hyperparameters and therefore stay outside the JSON.
export ABFLOW_GEOMETRY_FORENSICS="${ABFLOW_GEOMETRY_FORENSICS:-on}"
export ABFLOW_SAMPLE_FORENSICS="${ABFLOW_SAMPLE_FORENSICS:-on}"
export ABFLOW_SAMPLE_FORENSICS_THRESHOLD_A="${ABFLOW_SAMPLE_FORENSICS_THRESHOLD_A:-500}"
export ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD="${ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD:-10000}"
export ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH="${ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH:-3}"
export ABFLOW_GRAD_OVERFLOW_LOG_LIMIT="${ABFLOW_GRAD_OVERFLOW_LOG_LIMIT:-3}"
export ABFLOW_GEOMETRY_AUTHORITY_INTERVAL="${ABFLOW_GEOMETRY_AUTHORITY_INTERVAL:-20}"
export ABFLOW_GEOMETRY_AUTHORITY_BASE_ALERT="${ABFLOW_GEOMETRY_AUTHORITY_BASE_ALERT:-64}"
export ABFLOW_GEOMETRY_AUTHORITY_UPDATE_ALERT="${ABFLOW_GEOMETRY_AUTHORITY_UPDATE_ALERT:-10000}"
export ABFLOW_GEOMETRY_AUTHORITY_ALERT_MAX_PER_EPOCH="${ABFLOW_GEOMETRY_AUTHORITY_ALERT_MAX_PER_EPOCH:-3}"

# V203 formal Test failure contract. Infrastructure/protocol failures must abort;
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
echo "[RunResources] physical_gpus=$GPU_CSV nproc=$NPROC master_addr=$MASTER_ADDR port=$MASTER_PORT nnodes=$NNODES omp=$OMP_THREADS"
echo "[RunResume] checkpoint=${RESUME_CHECKPOINT:-scratch}"
echo "[ForensicsConfig] geometry=$ABFLOW_GEOMETRY_FORENSICS sample=$ABFLOW_SAMPLE_FORENSICS sample_threshold_A=$ABFLOW_SAMPLE_FORENSICS_THRESHOLD_A train_loss_threshold=$ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD train_outliers_per_epoch=$ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH grad_overflow_logs=$ABFLOW_GRAD_OVERFLOW_LOG_LIMIT geometry_authority_interval=$ABFLOW_GEOMETRY_AUTHORITY_INTERVAL base_coeff_alert=$ABFLOW_GEOMETRY_AUTHORITY_BASE_ALERT coord_update_alert=$ABFLOW_GEOMETRY_AUTHORITY_UPDATE_ALERT"
echo "[EpochTestFailureContract] infra_fail_fast=$ABFLOW_EPOCH_TEST_FAIL_FAST model_invalid=$ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY"

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
    raise SystemExit('V213 identity preflight failed: experiment.id is missing')
if not (exp_id == config_stem == output_id):
    raise SystemExit(
        'V213 identity preflight failed:\n'
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
            'V213 generation provenance preflight failed:\n'
            f'  experiment.id={exp_id}\n'
            f'  generation.save_dir basename={generation_id}\n'
            'Generation artifacts must not inherit an older experiment identity.'
        )
print(f'[ExperimentIdentity] PASS id={exp_id}')
PYIDENTITY

# Fail before GPU training on syntax/infrastructure regressions.
python -m py_compile \
  "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" \
  "$PROJECT_ROOT/models/modules/am_enc.py" \
  "$PROJECT_ROOT/models/modules/am_egnn.py" \
  "$PROJECT_ROOT/trainer/AbFlow_trainer.py" \
  "$PROJECT_ROOT/train.py"
if ! grep -q "GradientNormOverflowRecovered" "$PROJECT_ROOT/trainer/abs_trainer.py"; then
  echo "V213 requires the already-installed V10 stable finite-gradient norm fallback in trainer/abs_trainer.py" >&2
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
    'scripts/train/run_R31_R34_v213.sh',
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
sl=cfg.get('loss', {}).get('smooth_lddt', {})
sl_weight=float(sl.get('weight', 0.0))
sl_source=str(sl.get('prediction_source', sl.get('target', 'pred_design_endpoint')) or 'pred_design_endpoint')
allowed={'pred_design_endpoint','carrier_implied_endpoint'}
if sl_source not in allowed:
    raise SystemExit(f'V213 invalid smooth-lDDT prediction_source={sl_source!r}; allowed={sorted(allowed)}')
print(
    '[PairCoordinateContract] '
    f"mode={pc.get('mode','bounded_residual')} "
    f"delta_bound={pc.get('delta_bound',1.0)} "
    'state_pair=full coordinate_pair=bounded_residual'
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
