#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=$(realpath "$(dirname "$0")/../..")
CONFIG_INPUT=${1:-}
if [[ -z "$CONFIG_INPUT" ]]; then
    echo "Config missing. Usage: bash $0 <config.json>" >&2
    exit 2
fi
CONFIG_PATH=$(realpath "$CONFIG_INPUT")
export ABFLOW_CONFIG_PATH="$CONFIG_PATH"

# Parse every run-level choice from JSON. shlex.quote makes the generated
# exports safe, while the Python side validates types before the shell sees
# them. There are intentionally no per-experiment defaults in this script.
eval "$(python - "$CONFIG_PATH" "$CODE_DIR" <<'PY'
import json
import os
import shlex
import sys

path, project_root = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    cfg = json.load(handle)

meta = cfg.get("_experiment")
if not isinstance(meta, dict):
    raise TypeError("_experiment must be a JSON object")
runtime = meta.get("runtime_env")
formal = meta.get("formal_runtime")
if not isinstance(runtime, dict):
    raise TypeError("_experiment.runtime_env must be a JSON object")
if not isinstance(formal, dict):
    raise TypeError("_experiment.formal_runtime must be a JSON object")

for key, value in runtime.items():
    if not str(key).startswith("ABFLOW_"):
        raise ValueError(f"runtime_env key must start with ABFLOW_: {key}")
    print(f"export {key}={shlex.quote(str(value))}")

required = {
    "mode", "physical_gpus", "world_size", "master_addr", "master_port",
    "rdzv_backend", "nnodes", "omp_num_threads", "version_policy",
    "force_scratch", "log_filename",
}
missing = sorted(required - set(formal))
if missing:
    raise ValueError(f"formal_runtime missing keys: {missing}")
if formal["mode"] != "train":
    raise ValueError(f"only mode='train' is supported, got {formal['mode']!r}")
gpus = formal["physical_gpus"]
if not isinstance(gpus, list) or not gpus or any(
    isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in gpus
):
    raise TypeError("physical_gpus must be a non-empty list of GPU integers")
if len(set(gpus)) != len(gpus):
    raise ValueError("physical_gpus contains duplicates")
if formal["world_size"] != len(gpus):
    raise ValueError("world_size must equal len(physical_gpus)")
if formal["version_policy"] != "next_integer":
    raise ValueError("version_policy must be 'next_integer'")
if not isinstance(formal["force_scratch"], bool):
    raise TypeError("force_scratch must be boolean")
if isinstance(formal["master_port"], bool) or not isinstance(formal["master_port"], int):
    raise TypeError("master_port must be an integer")
if not 1024 <= formal["master_port"] <= 65535:
    raise ValueError("master_port must be in [1024, 65535]")
for key in ("world_size", "nnodes", "omp_num_threads"):
    if isinstance(formal[key], bool) or not isinstance(formal[key], int) or formal[key] <= 0:
        raise TypeError(f"{key} must be a positive integer")
log_name = formal["log_filename"]
if not isinstance(log_name, str) or os.path.basename(log_name) != log_name or not log_name:
    raise ValueError("log_filename must be a non-empty basename")

exp_id = meta.get("exp_id")
save_dir = cfg.get("save_dir")
resume = cfg.get("resume_checkpoint", "")
if not isinstance(exp_id, str) or not exp_id:
    raise ValueError("_experiment.exp_id must be a non-empty string")
if not isinstance(save_dir, str) or not save_dir:
    raise ValueError("top-level save_dir must be a non-empty string")
if not isinstance(resume, str):
    raise TypeError("resume_checkpoint must be a string")
if formal["force_scratch"] and resume.strip():
    raise ValueError("force_scratch=true requires resume_checkpoint='' ")
if not os.path.isabs(save_dir):
    save_dir = os.path.join(project_root, save_dir.removeprefix("./"))
save_dir = os.path.realpath(save_dir)

exports = {
    "ABFLOW_EXP_ID": exp_id,
    "ABFLOW_JSON_GPU_LIST": ",".join(str(v) for v in gpus),
    "ABFLOW_JSON_WORLD_SIZE": formal["world_size"],
    "ABFLOW_JSON_MASTER_ADDR": formal["master_addr"],
    "ABFLOW_JSON_MASTER_PORT": formal["master_port"],
    "ABFLOW_JSON_RDZV_BACKEND": formal["rdzv_backend"],
    "ABFLOW_JSON_NNODES": formal["nnodes"],
    "ABFLOW_JSON_OMP_NUM_THREADS": formal["omp_num_threads"],
    "ABFLOW_JSON_FORCE_SCRATCH": "1" if formal["force_scratch"] else "0",
    "ABFLOW_JSON_LOG_FILENAME": log_name,
    "ABFLOW_JSON_SAVE_DIR": save_dir,
}
for key, value in exports.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"

export ABFLOW_FORCE_SCRATCH="$ABFLOW_JSON_FORCE_SCRATCH"
export CUDA_VISIBLE_DEVICES="$ABFLOW_JSON_GPU_LIST"
export OMP_NUM_THREADS="$ABFLOW_JSON_OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

# Reserve one common version directory before torchrun so all ranks share the
# same checkpoint/TensorBoard/log identity. Existing runs are never removed.
mkdir -p "$ABFLOW_JSON_SAVE_DIR"
RUN_VERSION=0
while ! mkdir "$ABFLOW_JSON_SAVE_DIR/version_$RUN_VERSION" 2>/dev/null; do
    RUN_VERSION=$((RUN_VERSION + 1))
done
export ABFLOW_FIXED_VERSION="$RUN_VERSION"
VERSION_DIR="$ABFLOW_JSON_SAVE_DIR/version_$RUN_VERSION"
RUN_LOG="$VERSION_DIR/$ABFLOW_JSON_LOG_FILENAME"
export ABFLOW_RUN_TIME_LOG="$RUN_LOG"
ln -sfn "version_$RUN_VERSION/$ABFLOW_JSON_LOG_FILENAME" \
    "$ABFLOW_JSON_SAVE_DIR/$ABFLOW_JSON_LOG_FILENAME"
exec > >(stdbuf -oL -eL tee -a "$RUN_LOG") 2>&1

# Convert only train.py's top-level CLI fields. JSON lists are emitted as one
# option followed by all list values; nested audit/runtime dictionaries remain
# metadata and are never flattened into accidental CLI flags.
mapfile -d '' CONFIG_ARGS < <(python - "$CONFIG_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = json.load(handle)
args = []
for key, value in cfg.items():
    if key.startswith("_") or key == "runtime_env":
        continue
    if isinstance(value, dict) or value is None or value is False:
        continue
    if value is True:
        args.append(f"--{key}")
    elif isinstance(value, list):
        if value:
            args.append(f"--{key}")
            args.extend(str(item) for item in value)
    else:
        args.extend((f"--{key}", str(value)))
for item in args:
    sys.stdout.write(item)
    sys.stdout.write("\0")
PY
)

IFS=',' read -r -a GPU_ARRAY <<< "$ABFLOW_JSON_GPU_LIST"
LOCAL_GPU_IDS=()
for ((idx=0; idx<${#GPU_ARRAY[@]}; idx++)); do
    LOCAL_GPU_IDS+=("$idx")
done

if (( ABFLOW_JSON_WORLD_SIZE > 1 )); then
    PREFIX=(
        torchrun
        "--nproc_per_node=$ABFLOW_JSON_WORLD_SIZE"
        "--rdzv_backend=$ABFLOW_JSON_RDZV_BACKEND"
        "--rdzv_endpoint=$ABFLOW_JSON_MASTER_ADDR:$ABFLOW_JSON_MASTER_PORT"
        "--nnodes=$ABFLOW_JSON_NNODES"
    )
else
    PREFIX=(python)
fi

echo "============================================================"
echo "[V211JSONAuthority] exp=$ABFLOW_EXP_ID"
echo "[V211JSONAuthority] config=$CONFIG_PATH"
echo "[V211JSONAuthority] GPUs=$ABFLOW_JSON_GPU_LIST world_size=$ABFLOW_JSON_WORLD_SIZE port=$ABFLOW_JSON_MASTER_PORT"
echo "[V211JSONAuthority] save_dir=$ABFLOW_JSON_SAVE_DIR version=$RUN_VERSION force_scratch=$ABFLOW_JSON_FORCE_SCRATCH"
echo "[V211JSONAuthority] pair=$ABFLOW_ABX_NATIVE_REPR bridge=$ABFLOW_ABX_BRIDGE_MODE distogram=$ABFLOW_ABX_DISTOGRAM disto_scope=$ABFLOW_DISTOGRAM_PAIR_SCOPE smooth_lddt=$ABFLOW_MF_SMOOTH_LDDT lddt_context=$ABFLOW_SMOOTH_LDDT_CONTEXT_MODE"
echo "[V211JSONAuthority] RNG=abx_init:$ABFLOW_ABX_INIT_SEED,abx_forward_dropout:$ABFLOW_ABX_FORWARD_SEED"
echo "[V211JSONAuthority] weights=seq:$ABFLOW_LOSS_SEQUENCE_WEIGHT,struct:$ABFLOW_LOSS_STRUCTURE_WEIGHT,interface:$ABFLOW_LOSS_INTERFACE_WEIGHT,edge:$ABFLOW_LOSS_EDGE_WEIGHT,disto:$ABFLOW_LOSS_DISTOGRAM_WEIGHT,lddt:$ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT"
echo "[V211JSONAuthority] JSON controls run, architecture, objective and loss coefficients"
echo "============================================================"

cd "$CODE_DIR"
exec "${PREFIX[@]}" train.py --gpu "${LOCAL_GPU_IDS[@]}" "${CONFIG_ARGS[@]}"
