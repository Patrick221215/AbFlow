#!/bin/bash
###
 # @Author: Patrick221215 1427584833@qq.com
 # @Date: 2026-06-13 19:33:31
 # @LastEditors: Patrick221215 1427584833@qq.com
 # @LastEditTime: 2026-07-12 10:55:10
 # @FilePath: /cjm/project/AbFlow/scripts/train/train.sh
 # @Description: AbFlow training launcher with automatic master-port retry.
###
set -uo pipefail

########## Instruction ##########
# Optional environment variables:
#   GPU / ADDR / PORT / PORT_MAX_TRIES
#
# Examples:
#   GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh <config>
#   GPU="2,3,4" PORT=9901 PORT_MAX_TRIES=100 bash train.sh <config>
#
# Behavior:
#   For multi-GPU torchrun, if the master port is occupied, this script tries
#   PORT+1, PORT+2, ... until it finds a usable port or reaches PORT_MAX_TRIES.
######### end of instruction ##########

########## setup project directory ##########
CODE_DIR=$(realpath "$(dirname "$0")/../..")
echo "Locate the project folder at ${CODE_DIR}"

########## parsing JSON configs ##########
if [ $# -lt 1 ] || [ -z "${1:-}" ]; then
    echo "Config missing. Usage example: GPU=0,1 bash $0 <config>"
    exit 1
fi

CONFIG_FILE="$1"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Keep the original project behavior: top-level JSON keys are converted to CLI
# flags. Dict-valued metadata fields are intentionally skipped because train.py
# does not accept arbitrary nested command-line arguments.
CONFIG=$(cat "$CONFIG_FILE" | python -c '
import sys, json, shlex
config = json.load(sys.stdin)
args = []
for key, value in config.items():
    if not value:
        continue
    if isinstance(value, bool):
        args.append(f"--{key}")
    elif isinstance(value, dict):
        # Nested metadata should not be passed to argparse.
        continue
    elif isinstance(value, list):
        if len(value) > 0:
            args.append(f"--{key} " + " ".join(shlex.quote(str(v)) for v in value))
    else:
        args.append(f"--{key} {shlex.quote(str(value))}")
print(" ".join(args))
')

########## setup distributed training ##########
GPU="${GPU:--1}" # default using CPU
MASTER_ADDR="${ADDR:-localhost}"
MASTER_PORT="${PORT:-9901}"
PORT_MAX_TRIES="${PORT_MAX_TRIES:-100}"

export CUDA_VISIBLE_DEVICES="$GPU"
IFS=',' read -ra GPU_ARR <<< "$GPU"

if [ "$GPU" = "-1" ]; then
    TRAIN_GPUS=(-1)
else
    TRAIN_GPUS=("${!GPU_ARR[@]}")
fi

echo "Using GPUs: $GPU"
echo "Master address: ${MASTER_ADDR}, initial master port: ${MASTER_PORT}"

########## helper functions ##########
port_is_free() {
    local port="$1"
    python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])

def can_bind(family, host):
    s = socket.socket(family, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        try:
            s.close()
        except Exception:
            pass
        return False

ok4 = can_bind(socket.AF_INET, "0.0.0.0")
ok6 = True
try:
    ok6 = can_bind(socket.AF_INET6, "::")
except Exception:
    ok6 = True

sys.exit(0 if (ok4 and ok6) else 1)
PY
}

is_port_error_log() {
    local log_file="$1"
    grep -Eqi "Address already in use|failed to bind|failed to listen|errno: 98|EADDRINUSE" "$log_file"
}

run_single_process() {
    cd "$CODE_DIR" || exit 1
    if [ "${ABFLOW_DEBUG_CUDA_SYNC:-0}" = "1" ]; then
        CUDA_LAUNCH_BLOCKING=1 python train.py --gpus "${TRAIN_GPUS[@]}" ${CONFIG}
    else
        python train.py --gpus "${TRAIN_GPUS[@]}" ${CONFIG}
    fi
}

run_distributed_with_auto_port() {
    cd "$CODE_DIR" || exit 1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

    local base_port="$MASTER_PORT"
    local max_tries="$PORT_MAX_TRIES"
    local attempt=0

    while [ "$attempt" -lt "$max_tries" ]; do
        local current_port=$((base_port + attempt))

        if ! port_is_free "$current_port"; then
            echo "Master port ${current_port} is occupied before launch; trying $((current_port + 1))."
            attempt=$((attempt + 1))
            continue
        fi

        echo "Master address: ${MASTER_ADDR}, Master port: ${current_port}"
        local tmp_log
        tmp_log=$(mktemp "/tmp/abflow_torchrun_${current_port}_XXXX.log")

        set +e
        if [ "${ABFLOW_DEBUG_CUDA_SYNC:-0}" = "1" ]; then
            export CUDA_LAUNCH_BLOCKING=1
        else
            unset CUDA_LAUNCH_BLOCKING
        fi

        torchrun \
            --nproc_per_node="${#GPU_ARR[@]}" \
            --rdzv_backend=c10d \
            --rdzv_endpoint="${MASTER_ADDR}:${current_port}" \
            --nnodes=1 \
            train.py --gpus "${TRAIN_GPUS[@]}" ${CONFIG} 2>&1 | tee "$tmp_log"
        status=${PIPESTATUS[0]}
        set -e

        if [ "$status" -eq 0 ]; then
            rm -f "$tmp_log"
            return 0
        fi

        if is_port_error_log "$tmp_log"; then
            echo "torchrun failed because master port ${current_port} is unavailable; retrying with $((current_port + 1))."
            rm -f "$tmp_log"
            attempt=$((attempt + 1))
            continue
        fi

        echo "torchrun failed with non-port error. Log kept at: $tmp_log"
        return "$status"
    done

    echo "Failed to find an available master port after ${max_tries} attempts starting from ${base_port}."
    return 1
}

########## start training ##########
if [ "${#GPU_ARR[@]}" -gt 1 ] && [ "$GPU" != "-1" ]; then
    run_distributed_with_auto_port
else
    echo "Single-process training. Master port is not used."
    run_single_process
fi

# python train.py --gpus 0 --train_set ./all_data/RAbD/train.json --valid_set ./all_data/RAbD/valid.json --save_dir ./all_data/RAbD/models_single_cdr_design --cdr H3 --max_epoch 200 --save_topk 10 --batch_size 16 --shuffle --model_type dyMEAN --embed_dim 64 --hidden_size 128 --k_neighbors 9 --n_layers 3 --iter_round 3 --bind_dist_cutoff 6.6
