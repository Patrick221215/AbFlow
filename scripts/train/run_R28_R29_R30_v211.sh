#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CONFIG_PATH=${1:-}

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: bash $0 <experiment.json>" >&2
  exit 2
fi
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$PROJECT_ROOT/${CONFIG_PATH#./}"
fi

eval "$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import json, os, shlex, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8'))
root = sys.argv[2]
runtime = cfg['runtime']
training = cfg['training']
gpus = ','.join(str(x) for x in runtime['gpus'])
output = training['output_dir']
if not os.path.isabs(output):
    output = os.path.abspath(os.path.join(root, output))
resume = training['schedule'].get('resume_checkpoint', '') or ''
if resume and not os.path.isabs(resume):
    resume = os.path.abspath(os.path.join(root, resume))
for key, value in {
    'GPU_CSV': gpus,
    'NPROC': len(runtime['gpus']),
    'MASTER_ADDR': runtime['master_addr'],
    'MASTER_PORT': runtime['master_port'],
    'NNODES': runtime['nnodes'],
    'OMP_THREADS': runtime['omp_num_threads'],
    'CUDA_ALLOC': runtime.get('cuda_allocator', ''),
    'OUTPUT_ROOT': output,
    'RESUME_CHECKPOINT': resume,
}.items():
    print(f"{key}=" + shlex.quote(str(value)))
PY
)"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# Allocate one version directory atomically before torchrun so both DDP ranks
# and the tee logger share exactly the same run authority.
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RUN_DIR=$(dirname "$(dirname "$RESUME_CHECKPOINT")")
  VERSION_BASE=$(basename "$RUN_DIR")
  if [[ ! "$VERSION_BASE" =~ ^version_([0-9]+)$ ]]; then
    echo "resume checkpoint must live under version_N/checkpoint: $RESUME_CHECKPOINT" >&2
    exit 2
  fi
  VERSION="${BASH_REMATCH[1]}"
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
export OMP_NUM_THREADS="$OMP_THREADS"
[[ -n "$CUDA_ALLOC" ]] && export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC"

{
  echo "[RunLog] canonical=$RUN_LOG latest=$LATEST_LOG"
  echo "[RunVersion] fixed_version=$VERSION dir=$RUN_DIR"
  echo "[RunConfig] config=$CONFIG_PATH gpus=$GPU_CSV nproc=$NPROC port=$MASTER_PORT"
} | tee -a "$RUN_LOG"

set +e
torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$PROJECT_ROOT/train.py" \
  --config "$CONFIG_PATH" \
  2>&1 | tee -a "$RUN_LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
