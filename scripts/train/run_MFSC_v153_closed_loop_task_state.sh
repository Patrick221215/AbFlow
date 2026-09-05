#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-train}
EXP_ID=${2:-MFSC_V153_CLOSED_LOOP_TASK_STATE_R05_U02}
GPU_IDS=${3:-2,3,4,5}
BASE_CONFIG=${4:-/home/data3/cjm/project/AbFlow/scripts/train/configs/MFSC_v153/MFSC_V153_CLOSED_LOOP_TASK_STATE_R05_U02.json}
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-/home/data3/cjm/project/AbFlow}
PORT=${PORT:-29673}
ADDR=${ADDR:-localhost}

cd "$PROJECT_ROOT"

if [[ "$MODE" != "train" ]]; then
  echo "ERROR: V153 formal launcher supports MODE=train only." >&2
  exit 2
fi
if [[ "$EXP_ID" != "MFSC_V153_CLOSED_LOOP_TASK_STATE_R05_U02" ]]; then
  echo "ERROR: unexpected EXP_ID=$EXP_ID" >&2
  exit 2
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: formal config not found: $BASE_CONFIG" >&2
  exit 2
fi

MODEL_FILE="$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"
AM_FILE="$PROJECT_ROOT/models/modules/am_enc.py"
BASE_TRAINER="$PROJECT_ROOT/trainer/abs_trainer.py"
for f in "$PROJECT_ROOT/train.py" "$MODEL_FILE" "$AM_FILE" "$BASE_TRAINER"; do
  [[ -f "$f" ]] || { echo "ERROR: missing formal file: $f" >&2; exit 2; }
done

grep -qF "_v153_one_step_sampler_exposure" "$MODEL_FILE" || {
  echo "ERROR: V153 sampler-exposure closure missing." >&2; exit 2;
}
grep -qF "framework_endpoint_loss" "$MODEL_FILE" || {
  echo "ERROR: V153 framework endpoint authority missing." >&2; exit 2;
}
grep -qF "mf_sequence_state_proj" "$AM_FILE" || {
  echo "ERROR: V153 MF masked-sequence input missing." >&2; exit 2;
}
grep -qF "find_unused_parameters=(" "$BASE_TRAINER" || {
  echo "ERROR: V153 dynamic-DDP base trainer fix missing." >&2; exit 2;
}

BASE_CONFIG=$(realpath "$BASE_CONFIG")
export ABFLOW_CONFIG_PATH="$BASE_CONFIG"
export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT"

python - "$BASE_CONFIG" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1],encoding="utf-8"))
env=cfg.get("runtime_env",{})
arch=cfg.get("architecture",{})
errors=[]
def req(name,got,expected):
    if str(got)!=str(expected):
        errors.append(f"{name}={got} expected={expected}")
req("iter_round",cfg.get("iter_round"),1)
req("batch_size",cfg.get("batch_size"),2)
req("recycling_steps",arch.get("recycling_steps"),0)
req("ABFLOW_MF_STATEFUL_RECYCLING",env.get("ABFLOW_MF_STATEFUL_RECYCLING"),"on")
req("ABFLOW_SEQUENCE_GENERATIVE_MODE",env.get("ABFLOW_SEQUENCE_GENERATIVE_MODE"),"masked_absorbing")
req("ABFLOW_MF_MASKED_SEQUENCE_INPUT",env.get("ABFLOW_MF_MASKED_SEQUENCE_INPUT"),"on")
req("ABFLOW_SEQUENCE_LOSS_MASK_MODE",env.get("ABFLOW_SEQUENCE_LOSS_MASK_MODE"),"mf_masked_only")
req("ABFLOW_SAMPLER_EXPOSURE_MODE",env.get("ABFLOW_SAMPLER_EXPOSURE_MODE"),"one_step_u02")
req("ABFLOW_DDP_FIND_UNUSED_PARAMETERS",env.get("ABFLOW_DDP_FIND_UNUSED_PARAMETERS"),"on")
req("ABFLOW_R3_G_MODE",env.get("ABFLOW_R3_G_MODE"),"foldflow_fixed_scaled")
req("ABFLOW_R3_NOISE_SCOPE",env.get("ABFLOW_R3_NOISE_SCOPE"),"residue")
req("ABFLOW_SCOREFM_LOSS_MODE",env.get("ABFLOW_SCOREFM_LOSS_MODE"),"f01_r3_endpoint_canonical_hybrid")
req("ABFLOW_SCOREFM_SAMPLER_MODE",env.get("ABFLOW_SCOREFM_SAMPLER_MODE"),"f01_canonical_carrier")
req("ABFLOW_EPOCH_TEST_N_STEPS",env.get("ABFLOW_EPOCH_TEST_N_STEPS"),"10")
if errors:
    raise SystemExit("V153 preflight failed: "+"; ".join(errors))
print("[V153Preflight] PASS outer=1 fixedXtRecycle=on "
      "sequence=MF_masked_absorb maskedCE=on "
      "exposure=postwarmup_teacher_onpolicy "
      "DDP=dynamic_find_unused "
      "R05=fixedg_residue_U02 batch=2/GPU global=8")
PY

eval "$({ python - "$BASE_CONFIG" <<'PY'
import json, shlex, sys
cfg=json.load(open(sys.argv[1],encoding='utf-8'))
for k,v in cfg.get('runtime_env',{}).items():
    if not str(k).startswith('ABFLOW_'):
        raise ValueError(k)
    print(f'export {k}={shlex.quote(str(v))}')
PY
} )"

mapfile -t TRAIN_ARGS < <(python - "$BASE_CONFIG" "$EXP_ID" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1],encoding='utf-8'))
exp=sys.argv[2]
nested={'architecture','self_conditioning','confidence','loss_weights',
        'optimizer_profile','objective','diagnostics','runtime_env',
        'model_selection','observability'}
for k,v in cfg.items():
    if k in nested or k.startswith('_'):
        continue
    if k=='resume_checkpoint' or v is None or v=='' or v is False:
        continue
    if k=='save_dir':
        v=f'./results_module/{exp}'
    if v is True:
        print(f'--{k}')
    elif isinstance(v,list):
        print(f'--{k}')
        for x in v: print(str(x))
    elif isinstance(v,(dict,tuple)):
        raise TypeError(k)
    else:
        print(f'--{k}')
        print(str(v))
PY
)

IFS=',' read -r -a PHYSICAL_GPUS <<< "$GPU_IDS"
NPROC=${#PHYSICAL_GPUS[@]}
LOGICAL_GPUS=()
for ((i=0;i<NPROC;i++)); do LOGICAL_GPUS+=("$i"); done

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export ABFLOW_FORCE_SCRATCH=on

echo "============================================================"
echo "[V153Launcher] exp          = $EXP_ID"
echo "[V153Launcher] physicalGPU  = $GPU_IDS"
echo "[V153Launcher] config       = $BASE_CONFIG"
echo "[V153Launcher] science      = V153 closed-loop task-state; unchanged"
echo "[V153Launcher] DDP          = find_unused_parameters=True"
echo "[V153Launcher] reason       = masked-only CE / sampled K / conditional branches are a genuine dynamic graph"
echo "[V153Launcher] optimizer    = unused local branch keeps grad=None; no fake zero-grad momentum update"
echo "[V153Launcher] ScoreFlow    = R05 fixed-g residue-R3 U02"
echo "[V153Launcher] exposure     = exact on R05 teacher support; canonical affine U02 extension on self-generated states"
echo "[V153Launcher] batch        = 2/GPU x 4 = global 8"
echo "[V153Launcher] scratch      = on"
echo "============================================================"

if [[ "$NPROC" -gt 1 ]]; then
  exec torchrun \
    --nproc_per_node="$NPROC" \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${ADDR}:${PORT}" \
    train.py \
    --gpus "${LOGICAL_GPUS[@]}" \
    "${TRAIN_ARGS[@]}"
else
  exec python train.py --gpus "${LOGICAL_GPUS[@]}" "${TRAIN_ARGS[@]}"
fi
