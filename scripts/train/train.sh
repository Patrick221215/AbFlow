#!/bin/bash
set -euo pipefail

CODE_DIR=$(realpath "$(dirname "$0")/../..")
if [ -z "${1:-}" ]; then
    echo "Config missing. Usage: GPU=2,3 bash $0 <config.json>"
    exit 1
fi

CONFIG_PATH=$(realpath "$1")
export ABFLOW_CONFIG_PATH="$CONFIG_PATH"

# One JSON is the complete experiment authority.  Nested dictionaries are
# consumed by AbFlow_model/am_enc/trainer or exported here as runtime_env;
# only scalar top-level fields are converted to train.py CLI arguments.
eval "$(python - "$CONFIG_PATH" <<'PY'
import json, shlex, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8'))
env = cfg.get('runtime_env', {})
if not isinstance(env, dict):
    raise TypeError('runtime_env must be a JSON object')
for k, v in env.items():
    if not str(k).startswith('ABFLOW_'):
        raise ValueError(f'runtime_env key must start with ABFLOW_: {k}')
    print(f'export {k}={shlex.quote(str(v))}')
PY
)"

CONFIG=$(python - "$CONFIG_PATH" <<'PY'
import json, shlex, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8'))
args = []
for key, value in cfg.items():
    if key.startswith('_') or key == 'runtime_env':
        continue
    if isinstance(value, (dict, list)) or value is None or value is False:
        continue
    if value is True:
        args.append(f'--{key}')
    else:
        args += [f'--{key}', str(value)]
print(' '.join(shlex.quote(x) for x in args))
PY
)

GPU="${GPU:--1}"
MASTER_ADDR="${ADDR:-localhost}"
MASTER_PORT="${PORT:-9901}"
export CUDA_VISIBLE_DEVICES="$GPU"
IFS=',' read -ra GPU_ARR <<< "$GPU"

if [ "${#GPU_ARR[@]}" -gt 1 ]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    PREFIX=(torchrun --nproc_per_node="${#GPU_ARR[@]}" --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" --nnodes=1)
else
    PREFIX=(python)
fi

echo "============================================================"
echo "[V134Direct] project = $CODE_DIR"
echo "[V134Direct] config  = $CONFIG_PATH"
echo "[V134Direct] GPUs    = $GPU"
echo "[V134Direct] JSON is architecture/loss/optimizer/runtime source of truth"
echo "[V134Direct] first 8 train steps print forward/memory/loss/runtime diagnostics"
echo "============================================================"

cd "$CODE_DIR"
# Deliberately do NOT enable CUDA_LAUNCH_BLOCKING for formal training.
exec "${PREFIX[@]}" train.py --gpu "${!GPU_ARR[@]}" $CONFIG
