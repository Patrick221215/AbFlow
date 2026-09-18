#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=$(realpath "$(dirname "$0")/../..")
CONFIG_INPUT=${1:-}
GPU_INPUT=${2:-${ABFLOW_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
MASTER_PORT=${3:-${ABFLOW_MASTER_PORT:-29790}}

if [[ -z "$CONFIG_INPUT" ]]; then
    echo "Usage: bash $0 <config.json> [gpu_list] [master_port]" >&2
    exit 2
fi

CONFIG_PATH=$(realpath "$CONFIG_INPUT")
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 2; }

GPU_INPUT=${GPU_INPUT//，/,}
GPU_INPUT=${GPU_INPUT// /}

if [[ -z "$GPU_INPUT" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INPUT=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
fi
[[ -n "$GPU_INPUT" ]] || {
    echo "No GPU list resolved. Pass e.g. 2,3 as the second argument." >&2
    exit 2
}

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_INPUT"
NPROC=${#GPU_ARRAY[@]}
(( NPROC > 0 )) || { echo "Empty GPU list." >&2; exit 2; }

python - "$CONFIG_PATH" "$NPROC" <<'PY'
import json, sys
path, nproc = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(path, encoding="utf-8"))
required = ("experiment","data","training","model","loss","runtime","generation","evaluation")
for key in required:
    if key not in cfg:
        raise SystemExit(f"missing top-level config key: {key}")
exp = cfg["experiment"]
if exp.get("id") != "R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_FULLATOM_U02_SCOREFLOW":
    raise SystemExit("wrong experiment id")
if exp.get("initialization") != "scratch":
    raise SystemExit("scratch initialization required")
if str(cfg["training"]["schedule"].get("resume_checkpoint","") or "").strip():
    raise SystemExit("resume_checkpoint must be empty")
per_gpu = int(cfg["training"]["loader"]["per_gpu_batch_size"])
if per_gpu <= 0:
    raise SystemExit("per_gpu_batch_size must be positive")
r3 = cfg["model"]["r05"]["r3"]
if r3.get("noise_scope") != "full_atom":
    raise SystemExit("noise_scope must be full_atom")
if r3.get("g_mode") != "foldflow_fixed_scaled":
    raise SystemExit("g_mode must be foldflow_fixed_scaled")
if float(r3.get("path_min_sigma")) != 0.0:
    raise SystemExit("path_min_sigma must be 0")
if int(cfg["generation"]["n_steps"]) != 10 or int(cfg["generation"]["seed"]) != 2023:
    raise SystemExit("generation must remain 10 steps / seed 2023")
print("[R05FullAtomPreflight] PASS")
print(f"[BatchContract] per_gpu={per_gpu} world_size={nproc} effective_global={per_gpu*nproc}")
print(f"[ScientificDelta] noise_scope={r3['noise_scope']} g_mode={r3['g_mode']} path_min_sigma={r3['path_min_sigma']}")
PY

export CUDA_VISIBLE_DEVICES="$GPU_INPUT"
export ABFLOW_NPROC_PER_NODE="$NPROC"
export ABFLOW_PROJECT_ROOT="$CODE_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "============================================================"
echo "[R05FullAtomRun] config=$CONFIG_PATH"
echo "[R05FullAtomRun] physical_gpus=$GPU_INPUT world_size=$NPROC master_port=$MASTER_PORT"
echo "[R05FullAtomRun] entry=train.py --config"
echo "============================================================"

if [[ "${ABFLOW_DRY_RUN:-0}" == "1" ]]; then
    echo "[DryRun] PASS - train.py not launched"
    exit 0
fi

cd "$CODE_DIR"

if (( NPROC > 1 )); then
    exec torchrun \
        "--nproc_per_node=$NPROC" \
        --rdzv_backend=c10d \
        "--rdzv_endpoint=127.0.0.1:$MASTER_PORT" \
        --nnodes=1 \
        train.py --config "$CONFIG_PATH"
else
    exec python train.py --config "$CONFIG_PATH"
fi
