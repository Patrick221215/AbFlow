#!/usr/bin/env bash
set -euo pipefail
# V210_JSON_DRIVEN_TASK_CONTRACT

CODE_DIR=$(realpath "$(dirname "$0")/../..")
CONFIG_PATH=$(realpath "${1:?Usage: GPU=2,3 bash scripts/train/train.sh <config.json>}")
export ABFLOW_CONFIG_PATH="$CONFIG_PATH"

# Runtime environment is owned by the JSON. Accept the formal nested location.
eval "$(python - "$CONFIG_PATH" <<'PY'
import json, shlex, sys
cfg=json.load(open(sys.argv[1],encoding='utf-8'))
env={}
meta=cfg.get('_experiment',{})
if isinstance(meta,dict):
    env.update(meta.get('runtime_env',{}) or {})
env.update(cfg.get('runtime_env',{}) or {})
for k,v in env.items():
    if not str(k).startswith('ABFLOW_'):
        raise SystemExit(f"invalid runtime key: {k}")
    print(f"export {k}={shlex.quote(str(v))}")
PY
)"

# Preserve JSON lists as argparse nargs+ lists. Never drop cdr/paratope.
mapfile -d '' -t CONFIG_ARGS < <(python - "$CONFIG_PATH" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1],encoding='utf-8'))
for k,v in cfg.items():
    if k.startswith('_') or k=='runtime_env' or isinstance(v,dict) or v is None or v is False:
        continue
    vals=[]
    if v is True:
        vals=[f'--{k}']
    elif isinstance(v,list):
        if v:
            vals=[f'--{k}', *map(str,v)]
    else:
        vals=[f'--{k}',str(v)]
    for x in vals:
        sys.stdout.write(x+'\0')
PY
)

GPU="${GPU:--1}"
ADDR="${ADDR:-localhost}"
PORT="${PORT:-9901}"
export CUDA_VISIBLE_DEVICES="$GPU"
IFS=',' read -ra GPU_ARR <<< "$GPU"

if ((${#GPU_ARR[@]} > 1)); then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  PREFIX=(torchrun --nproc_per_node="${#GPU_ARR[@]}" --rdzv_backend=c10d --rdzv_endpoint="${ADDR}:${PORT}" --nnodes=1)
else
  PREFIX=(python)
fi

echo "[V210CLIContract] config=$CONFIG_PATH GPU=$GPU cdr/paratope JSON lists preserved"
cd "$CODE_DIR"
exec "${PREFIX[@]}" train.py --gpus "${!GPU_ARR[@]}" "${CONFIG_ARGS[@]}"
