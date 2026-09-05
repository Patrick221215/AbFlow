#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-train}
EXP_ID=${2:-MFSC_V154_MF_NATIVE_MASKED_STATE_R05_U02}
GPU_IDS=${3:-2,3,4,5}
BASE_CONFIG=${4:-/home/data3/cjm/project/AbFlow/scripts/train/configs/MFSC_v154/MFSC_V154_MF_NATIVE_MASKED_STATE_R05_U02.json}
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-/home/data3/cjm/project/AbFlow}
PORT=${PORT:-29674}
ADDR=${ADDR:-localhost}

cd "$PROJECT_ROOT"
[[ "$MODE" == "train" ]] || { echo "ERROR: V154 formal launcher supports train only" >&2; exit 2; }
[[ "$EXP_ID" == "MFSC_V154_MF_NATIVE_MASKED_STATE_R05_U02" ]] || { echo "ERROR: unexpected EXP_ID=$EXP_ID" >&2; exit 2; }
[[ -f "$BASE_CONFIG" ]] || { echo "ERROR: config missing: $BASE_CONFIG" >&2; exit 2; }
[[ -f "$PROJECT_ROOT/train.py" ]] || { echo "ERROR: train.py missing" >&2; exit 2; }
MODEL_FILE="$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"
AM_FILE="$PROJECT_ROOT/models/modules/am_enc.py"
TRAINER_FILE="$PROJECT_ROOT/trainer/AbFlow_trainer.py"
ABS_TRAINER_FILE="$PROJECT_ROOT/trainer/abs_trainer.py"
for f in "$MODEL_FILE" "$AM_FILE" "$TRAINER_FILE" "$ABS_TRAINER_FILE"; do [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 2; }; done

grep -q "mf_masked_sequence_input" "$MODEL_FILE" || { echo "ERROR: model lacks MF masked-sequence contract" >&2; exit 2; }
grep -q "sequence_loss_mask_mode" "$MODEL_FILE" || { echo "ERROR: model lacks masked-only CE contract" >&2; exit 2; }
grep -q "mf_sequence_state_proj" "$AM_FILE" || { echo "ERROR: AMEncoder lacks direct MF sequence-state projection" >&2; exit 2; }
grep -q "mf_clean_sequence_proj" "$AM_FILE" || { echo "ERROR: AMEncoder lacks clean posterior MF recycle projection" >&2; exit 2; }
grep -q "find_unused_parameters" "$ABS_TRAINER_FILE" || { echo "ERROR: abs_trainer lacks dynamic-DDP support" >&2; exit 2; }

BASE_CONFIG=$(realpath "$BASE_CONFIG")
export ABFLOW_CONFIG_PATH="$BASE_CONFIG"
export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT"

python - "$BASE_CONFIG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1],encoding='utf-8'))
e=c.get('runtime_env',{}); a=c.get('architecture',{}); w=c.get('loss_weights',{})
err=[]
def req(k,g,v):
    if str(g)!=str(v): err.append(f'{k}={g} expected={v}')
req('iter_round',c.get('iter_round'),1)
req('batch_size',c.get('batch_size'),2)
req('architecture.recycling_steps',a.get('recycling_steps'),0)
for k,v in {
 'ABFLOW_SOURCE_MODE':'pcs_rc',
 'ABFLOW_R3_G_MODE':'foldflow_fixed_scaled',
 'ABFLOW_R3_FIXED_G_SCALED':'0.1',
 'ABFLOW_R3_NOISE_SCOPE':'residue',
 'ABFLOW_SCOREFM_LOSS_MODE':'f01_r3_endpoint_canonical_hybrid',
 'ABFLOW_SCOREFM_SAMPLER_MODE':'f01_canonical_carrier',
 'ABFLOW_OUTER_RECYCLE_GRAD_MODE':'off',
 'ABFLOW_MF_STATEFUL_RECYCLING':'on',
 'ABFLOW_MF_STATEFUL_MAX_DEPTH':'3',
 'ABFLOW_MF_STATEFUL_INFERENCE_DEPTH':'3',
 'ABFLOW_MF_STATEFUL_RANDOM_DEPTH':'on',
 'ABFLOW_SEQUENCE_GENERATIVE_MODE':'masked_absorbing',
 'ABFLOW_MF_MASKED_SEQUENCE_INPUT':'on',
 'ABFLOW_MF_SEQUENCE_STATE_VOCAB':'21',
 'ABFLOW_SEQUENCE_LOSS_MASK_MODE':'mf_masked_only',
 'ABFLOW_SEQUENCE_DENOISER_STATE_CONDITIONING':'off',
 'ABFLOW_SAMPLER_EXPOSURE_MODE':'off',
 'ABFLOW_DDP_FIND_UNUSED_PARAMETERS':'on',
 'ABFLOW_FORCE_SCRATCH':'on',
}.items(): req(k,e.get(k),v)
for k,v in {'transport':1.0,'sequence':0.4,'aligned_mse':0.5,'smooth_lddt':0.1,'distogram':0.5,'confidence':0.025,'framework_endpoint':0.0}.items(): req('loss_weights.'+k,w.get(k),v)
if err: raise SystemExit('V154 formal preflight failed: '+'; '.join(err))
print('[V154Preflight] PASS outer=1 fixedXt=1 mf_stateful=train_random1_3/infer3 R05=residue_fixedg U02=matched masked_seq=MF_single masked_CE=final_masked_only framework_primary=OFF exposure=OFF loss_weights=UNCHANGED batch=2/GPU global=8 DDP_find_unused=ON scratch=1')
PY

# Export only ABFLOW_* runtime switches from the formal JSON.
eval "$(python - "$BASE_CONFIG" <<'PY'
import json,shlex,sys
for k,v in json.load(open(sys.argv[1],encoding='utf-8')).get('runtime_env',{}).items():
    if not k.startswith('ABFLOW_'): raise SystemExit(f'bad env key {k}')
    print(f'export {k}={shlex.quote(str(v))}')
PY
)"

# Convert scalar top-level JSON fields to train.py arguments.
mapfile -t TRAIN_ARGS < <(python - "$BASE_CONFIG" "$EXP_ID" <<'PY'
import json,sys
c=json.load(open(sys.argv[1],encoding='utf-8')); exp=sys.argv[2]
nested={'architecture','self_conditioning','confidence','loss_weights','optimizer_profile','objective','diagnostics','runtime_env','model_selection','observability'}
for k,v in c.items():
    if k in nested or k.startswith('_') or k=='resume_checkpoint' or v is None or v=='' or v is False: continue
    if k=='save_dir': v=f'./results_module/{exp}'
    if v is True: print('--'+k)
    elif isinstance(v,list):
        print('--'+k)
        for x in v: print(str(x))
    elif isinstance(v,(dict,tuple)): raise TypeError(f'unexpected nested arg {k}')
    else: print('--'+k); print(str(v))
PY
)

IFS=',' read -r -a PHYSICAL_GPUS <<< "$GPU_IDS"
NPROC=${#PHYSICAL_GPUS[@]}
[[ $NPROC -eq 4 ]] || { echo "ERROR: formal V154 expects four GPUs, got $GPU_IDS" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export ABFLOW_FORCE_SCRATCH=on
[[ -z "${ABFLOW_RESUME_CHECKPOINT:-}" ]] || { echo "ERROR: scratch required" >&2; exit 2; }

echo "[V154Launcher] physicalGPU=$GPU_IDS port=$PORT config=$BASE_CONFIG"
echo "[V154Launcher] science=V152_R05_U02_fixedXt + MFDesign-native masked sequence state; NO framework primary loss; NO rollout exposure"
echo "[V154Launcher] expected dynamic DDP: rare all-revealed local sequence batches may have legitimate unused sequence-head params"

LOGICAL_GPUS=()
for ((i=0;i<NPROC;i++)); do LOGICAL_GPUS+=("$i"); done

exec torchrun \
  --nproc_per_node="$NPROC" \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${ADDR}:${PORT}" \
  train.py \
  --gpus "${LOGICAL_GPUS[@]}" \
  "${TRAIN_ARGS[@]}"
