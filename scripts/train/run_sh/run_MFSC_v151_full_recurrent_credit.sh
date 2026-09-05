#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-train}
EXP_ID=${2:-MFSC_V151_FULL_RECURRENT_CREDIT_U02}
GPU_IDS=${3:-2,3,4,5}
BASE_CONFIG=${4:-/home/data3/cjm/project/AbFlow/scripts/train/configs/MFSC_v151/MFSC_V151_FULL_RECURRENT_CREDIT_U02.json}
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-/home/data3/cjm/project/AbFlow}
PORT=${PORT:-29662}
ADDR=${ADDR:-localhost}

cd "$PROJECT_ROOT"

if [[ "$MODE" != "train" ]]; then
  echo "ERROR: V151 formal launcher supports MODE=train only." >&2
  exit 2
fi
if [[ "$EXP_ID" != "MFSC_V151_FULL_RECURRENT_CREDIT_U02" ]]; then
  echo "ERROR: unexpected EXP_ID=$EXP_ID" >&2
  echo "Expected: MFSC_V151_FULL_RECURRENT_CREDIT_U02" >&2
  exit 2
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: formal config not found: $BASE_CONFIG" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_ROOT/train.py" ]]; then
  echo "ERROR: stable train.py not found: $PROJECT_ROOT/train.py" >&2
  exit 2
fi

MODEL_FILE="$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "ERROR: AbFlow model not found: $MODEL_FILE" >&2
  exit 2
fi

# V151 deployment preflight.  This specifically catches the stale V142 guard
# that caused the repeated constructor crash before any training forward.
if grep -qF "V142 requires iter_round=1" "$MODEL_FILE"; then
  echo "ERROR: stale V142 iter_round=1 guard is still present in $MODEL_FILE" >&2
  echo "ERROR: V151 model file was not actually deployed. Re-unzip the V151 bring-up fix." >&2
  exit 2
fi
if ! grep -qF "Formal state-evolving recycle contract requires" "$MODEL_FILE"; then
  echo "ERROR: expected state-evolving iter_round=3 formal guard is missing from $MODEL_FILE" >&2
  echo "ERROR: wrong/old AbFlow_model.py is active." >&2
  exit 2
fi
if ! grep -qF "_caller_grad_enabled = torch.is_grad_enabled()" "$MODEL_FILE"; then
  echo "ERROR: V151 inference autograd fix is missing from $MODEL_FILE" >&2
  echo "ERROR: wrong/older AbFlow_model.py is active." >&2
  exit 2
fi
if ! grep -qF "self.full_recurrent_credit = True" "$MODEL_FILE"; then
  echo "ERROR: V151 full recurrent credit contract is missing from $MODEL_FILE" >&2
  echo "ERROR: wrong/older AbFlow_model.py is active." >&2
  exit 2
fi

BASE_CONFIG=$(realpath "$BASE_CONFIG")
export ABFLOW_CONFIG_PATH="$BASE_CONFIG"
export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT"

# Fail before torchrun if the JSON does not match the formal V151 state-recycle
# contract.  This is a launcher check only; it changes no scientific setting.
python - "$BASE_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p, encoding="utf-8"))
errors = []
if int(cfg.get("iter_round", -1)) != 3:
    errors.append(f"iter_round={cfg.get('iter_round')} (expected 3)")
arch = cfg.get("architecture", {})
if int(arch.get("recycling_steps", -1)) != 0:
    errors.append(
        f"architecture.recycling_steps={arch.get('recycling_steps')} (expected 0)"
    )
env = cfg.get("runtime_env", {})
if str(env.get("ABFLOW_MFDESIGN_RECYCLING_STEPS")) != "0":
    errors.append(
        "ABFLOW_MFDESIGN_RECYCLING_STEPS="
        f"{env.get('ABFLOW_MFDESIGN_RECYCLING_STEPS')} (expected 0)"
    )
if str(env.get("ABFLOW_OUTER_RECYCLE_GRAD_MODE")) != "all_grad":
    errors.append(
        "ABFLOW_OUTER_RECYCLE_GRAD_MODE="
        f"{env.get('ABFLOW_OUTER_RECYCLE_GRAD_MODE')} (expected all_grad)"
    )
if int(cfg.get("batch_size", -1)) != 1:
    errors.append(
        f"batch_size={cfg.get('batch_size')} "
        "(expected per-GPU microbatch 1 for V151 full-BPTT memory safety)"
    )
if str(env.get("ABFLOW_GRAD_ACCUM_STEPS")) != "2":
    errors.append(
        "ABFLOW_GRAD_ACCUM_STEPS="
        f"{env.get('ABFLOW_GRAD_ACCUM_STEPS')} "
        "(expected 2; effective global optimizer batch remains 8)"
    )
mem = cfg.get("objective", {}).get("memory_execution", {})
if str(mem.get("remainder_policy")) != "flush_final_partial_group_no_drop":
    errors.append(
        "objective.memory_execution.remainder_policy="
        f"{mem.get('remainder_policy')} "
        "(expected flush_final_partial_group_no_drop)"
    )
if str(env.get("ABFLOW_PROPOSAL_ADAPTER_START_ROUND")) != "1":
    errors.append(
        "ABFLOW_PROPOSAL_ADAPTER_START_ROUND="
        f"{env.get('ABFLOW_PROPOSAL_ADAPTER_START_ROUND')} (expected 1)"
    )
if str(env.get("ABFLOW_EPOCH_TEST_BATCH_SIZE")) != "20":
    errors.append(
        "ABFLOW_EPOCH_TEST_BATCH_SIZE="
        f"{env.get('ABFLOW_EPOCH_TEST_BATCH_SIZE')} "
        "(expected 20 after V151 eval-autograd fix)"
    )
if errors:
    raise SystemExit("V151 config preflight failed: " + "; ".join(errors))
print("[V151Preflight] config_contract=PASS outer=3 mf_recycle=0 "
      "outer_grad=all_grad proposal_round=1 train_microbatch=1 accum=2 updates=ceil(microsteps/2) effective_global=8_full/4_tail test_batch=20")
PY

MODEL_SHA256=$(sha256sum "$MODEL_FILE" | awk '{print $1}')
echo "[V151Preflight] model_file=$MODEL_FILE"
echo "[V151Preflight] model_sha256=$MODEL_SHA256"
echo "[V151Preflight] stale_v142_guard=ABSENT"
echo "[V151Preflight] state_evolving_iter3_guard=PRESENT"
echo "[V151Preflight] inference_autograd_fix=PRESENT"
echo "[V151Preflight] full_recurrent_credit=PRESENT"

# Export only the legacy ABFLOW_* switches from the same formal JSON.
# Architecture/loss/optimizer/SC/confidence are read directly from BASE_CONFIG
# inside the Python modules through ABFLOW_CONFIG_PATH.
eval "$({ python - "$BASE_CONFIG" <<'PY'
import json, shlex, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8'))
runtime = cfg.get('runtime_env', {})
if not isinstance(runtime, dict):
    raise TypeError('runtime_env must be a JSON object')
for key, value in runtime.items():
    if not str(key).startswith('ABFLOW_'):
        raise ValueError(f'bad runtime_env key: {key}')
    print(f'export {key}={shlex.quote(str(value))}')
PY
} )"

# Convert only scalar top-level train.py arguments to a real Bash argv array.
# No shell re-parsing of quoted empty strings; nested scientific blocks are
# deliberately skipped because Python modules consume them directly.
mapfile -t TRAIN_ARGS < <(python - "$BASE_CONFIG" "$EXP_ID" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8'))
exp_id = sys.argv[2]
nested = {
    'architecture', 'self_conditioning', 'confidence', 'loss_weights',
    'optimizer_profile', 'objective', 'diagnostics', 'runtime_env',
    'model_selection', 'observability'
}
for key, value in cfg.items():
    if key in nested or key.startswith('_'):
        continue
    if key == 'resume_checkpoint' or value is None or value == '' or value is False:
        continue
    if key == 'save_dir':
        value = f'./results_module/{exp_id}'
    if value is True:
        print(f'--{key}')
    elif isinstance(value, list):
        print(f'--{key}')
        for item in value:
            print(str(item))
    elif isinstance(value, (dict, tuple)):
        raise TypeError(f'unexpected nested top-level train arg: {key}')
    else:
        print(f'--{key}')
        print(str(value))
PY
)

IFS=',' read -r -a PHYSICAL_GPUS <<< "$GPU_IDS"
NPROC=${#PHYSICAL_GPUS[@]}
if [[ "$NPROC" -lt 1 ]]; then
  echo "ERROR: empty GPU list" >&2
  exit 2
fi

LOGICAL_GPUS=()
for ((i=0; i<NPROC; i++)); do
  LOGICAL_GPUS+=("$i")
done

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

# V151 changes train-time state-recycling semantics and must be trained from scratch.
if [[ -n "${ABFLOW_RESUME_CHECKPOINT:-}" ]]; then
  echo "ERROR: V151 changes recurrent credit assignment; resume is forbidden." >&2
  exit 2
fi
export ABFLOW_FORCE_SCRATCH=on
echo "[V151Launcher] resume      = <forbidden; scratch-only>"


# Fail before torchrun if the formal Test contract is incomplete.
if [[ "${ABFLOW_THREE_PHASE_PROTOCOL:-off}" == "on" ]]; then
  : "${ABFLOW_EPOCH_TEST_JSON:?ABFLOW_EPOCH_TEST_JSON is required}"
  if [[ ! -f "$ABFLOW_EPOCH_TEST_JSON" ]]; then
    echo "ERROR: epoch-test JSON not found: $ABFLOW_EPOCH_TEST_JSON" >&2
    exit 2
  fi
fi

# Contract summary: these values must match the first Python-side V151Config logs.
echo "============================================================"
echo "[V151Launcher] mode        = $MODE"
echo "[V151Launcher] exp         = $EXP_ID"
echo "[V151Launcher] physicalGPU = $GPU_IDS"
echo "[V151Launcher] logicalGPU  = ${LOGICAL_GPUS[*]}"
echo "[V151Launcher] config      = $ABFLOW_CONFIG_PATH"
echo "[V151Launcher] train entry = $PROJECT_ROOT/train.py"
echo "[V151Launcher] test JSON   = ${ABFLOW_EPOCH_TEST_JSON:-<off>}"
echo "[V151Launcher] scratch     = ${ABFLOW_FORCE_SCRATCH:-off}"
echo "[V151Launcher] batch       = per-GPU (JSON batch_size is local microbatch)"
echo "[V151Launcher] checkpoint  = MF-wide nonreentrant layer checkpoint (Pairformer + score model; PyTorch1.11 DDP-safe)"
echo "[V151Launcher] recurrence  = outer=3 state-evolving, MF recycling=0, full recurrent credit"
echo "[V151Launcher] frame       = PCS-RC H3 Kabsch -> full template -> proposal common center"
echo "[V151Launcher] frame leak  = native antibody coordinates forbidden"
echo "[V151Launcher] sequence    = raw MF structure latent 512 -> D3PM512"
echo "[V151Launcher] seq legacy  = H256 has zero numerical authority over logits"
echo "[V151Launcher] seq process = reversible Uniform(20) categorical bridge"
echo "[V151Launcher] seq reverse = exact x0-marginalized finite-step kernel"
echo "[V151Launcher] seq path    = internal reversible Uniform(20) bridge retained"
echo "[V151Launcher] seq terminal= MAP clean-endpoint posterior (argmax)"
echo "[V151Launcher] decision    = Bayes-optimal for per-residue AAR/CAAR 0-1 loss"
echo "[V151Launcher] seq state   = S_t sampler-only; hard neural identity/topology OFF"
echo "[V151Launcher] denoiser    = p_theta(S1 | X_t, t, PCS-RC context)"
echo "[V151Launcher] seq proposal= explicit PCS-RC S_pep zero-start residual condition"
echo "[V151Launcher] seq input   = pep_condition; native sequence context OFF"
echo "[V151Launcher] coord prop  = explicit PCS-RC X_pep SE3-invariant local residual"
echo "[V151Launcher] PCS local   = COMPLETE: coordinate + sequence proposal conditions"
echo "[V151Launcher] state recycle= outer_rounds=3; X/S/memory evolve between rounds"
echo "[V151Launcher] MF recycle   = 0 (nested representation recycle disabled)"
echo "[V151Launcher] outer grad   = FULL recurrent BPTT; r0/r1/r2 all grad"
echo "[V151Launcher] train memory = microbatch=1/GPU x accum=2; effective global batch=8"
echo "[V151Launcher] opt cadence  = 404 optimizer updates/epoch; scheduler+EMA update per optimizer step"
echo "[V151Launcher] accum tail   = 807 microsteps -> 403x2 + final 1 -> 404 updates; no sample drop"
echo "[V151Launcher] eval grad    = STRICT no-grad; training all-grad never leaks into eval"
echo "[V151Launcher] epoch Test   = EVERY epoch; logical batch=20; n_steps=10"
echo "[V151Launcher] Test memory  = eval autograd leak fixed; historical batch=20 restored"
echo "[V151Launcher] adapter rnd  = 1 (round0 placement; rounds1-2 local correction)"
echo "[V151Launcher] IMPORTANT: train.sh is NOT used"
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
