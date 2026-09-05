#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-train}
EXP_ID=${2:-MFSC_V150_STATE_EVOLVING_CARTESIAN_RECYCLE_U02}
GPU_IDS=${3:-2,3,4,5}
BASE_CONFIG=${4:-/home/data3/cjm/project/AbFlow/scripts/train/configs/MFSC_v150/MFSC_V150_STATE_EVOLVING_CARTESIAN_RECYCLE_U02.json}
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-/home/data3/cjm/project/AbFlow}
PORT=${PORT:-29660}
ADDR=${ADDR:-localhost}

cd "$PROJECT_ROOT"

if [[ "$MODE" != "train" ]]; then
  echo "ERROR: V150 formal launcher supports MODE=train only." >&2
  exit 2
fi
if [[ "$EXP_ID" != "MFSC_V150_STATE_EVOLVING_CARTESIAN_RECYCLE_U02" ]]; then
  echo "ERROR: unexpected EXP_ID=$EXP_ID" >&2
  echo "Expected: MFSC_V150_STATE_EVOLVING_CARTESIAN_RECYCLE_U02" >&2
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

# V150 deployment preflight.  This specifically catches the stale V142 guard
# that caused the repeated constructor crash before any training forward.
if grep -qF "V142 requires iter_round=1" "$MODEL_FILE"; then
  echo "ERROR: stale V142 iter_round=1 guard is still present in $MODEL_FILE" >&2
  echo "ERROR: V150 model file was not actually deployed. Re-unzip the V150 bring-up fix." >&2
  exit 2
fi
if ! grep -qF "V150 formal contract requires iter_round=3" "$MODEL_FILE"; then
  echo "ERROR: expected V150 iter_round=3 formal guard is missing from $MODEL_FILE" >&2
  echo "ERROR: wrong/old AbFlow_model.py is active." >&2
  exit 2
fi
if ! grep -qF "_caller_grad_enabled = torch.is_grad_enabled()" "$MODEL_FILE"; then
  echo "ERROR: V150 inference autograd fix is missing from $MODEL_FILE" >&2
  echo "ERROR: wrong/older AbFlow_model.py is active." >&2
  exit 2
fi

BASE_CONFIG=$(realpath "$BASE_CONFIG")
export ABFLOW_CONFIG_PATH="$BASE_CONFIG"
export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT"

# Fail before torchrun if the JSON does not match the formal V150 state-recycle
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
if str(env.get("ABFLOW_OUTER_RECYCLE_GRAD_MODE")) != "final_only":
    errors.append(
        "ABFLOW_OUTER_RECYCLE_GRAD_MODE="
        f"{env.get('ABFLOW_OUTER_RECYCLE_GRAD_MODE')} (expected final_only)"
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
        "(expected 20 after V150 eval-autograd fix)"
    )
if errors:
    raise SystemExit("V150 config preflight failed: " + "; ".join(errors))
print("[V150Preflight] config_contract=PASS outer=3 mf_recycle=0 "
      "outer_grad=final_only proposal_round=1 test_batch=20")
PY

MODEL_SHA256=$(sha256sum "$MODEL_FILE" | awk '{print $1}')
echo "[V150Preflight] model_file=$MODEL_FILE"
echo "[V150Preflight] model_sha256=$MODEL_SHA256"
echo "[V150Preflight] stale_v142_guard=ABSENT"
echo "[V150Preflight] v150_iter3_guard=PRESENT"
echo "[V150Preflight] inference_autograd_fix=PRESENT"

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

# V150 changes train-time state-recycling semantics and must be trained from scratch.
if [[ -n "${ABFLOW_RESUME_CHECKPOINT:-}" ]]; then
  echo "ERROR: V150 changes train-time state-recycling semantics; resume is forbidden." >&2
  exit 2
fi
export ABFLOW_FORCE_SCRATCH=on
echo "[V150Launcher] resume      = <forbidden; scratch-only>"


# Fail before torchrun if the formal Test contract is incomplete.
if [[ "${ABFLOW_THREE_PHASE_PROTOCOL:-off}" == "on" ]]; then
  : "${ABFLOW_EPOCH_TEST_JSON:?ABFLOW_EPOCH_TEST_JSON is required}"
  if [[ ! -f "$ABFLOW_EPOCH_TEST_JSON" ]]; then
    echo "ERROR: epoch-test JSON not found: $ABFLOW_EPOCH_TEST_JSON" >&2
    exit 2
  fi
fi

# Contract summary: these values must match the first Python-side V150Config logs.
echo "============================================================"
echo "[V150Launcher] mode        = $MODE"
echo "[V150Launcher] exp         = $EXP_ID"
echo "[V150Launcher] physicalGPU = $GPU_IDS"
echo "[V150Launcher] logicalGPU  = ${LOGICAL_GPUS[*]}"
echo "[V150Launcher] config      = $ABFLOW_CONFIG_PATH"
echo "[V150Launcher] train entry = $PROJECT_ROOT/train.py"
echo "[V150Launcher] test JSON   = ${ABFLOW_EPOCH_TEST_JSON:-<off>}"
echo "[V150Launcher] scratch     = ${ABFLOW_FORCE_SCRATCH:-off}"
echo "[V150Launcher] batch       = per-GPU (JSON batch_size is local microbatch)"
echo "[V150Launcher] checkpoint  = MF-wide nonreentrant layer checkpoint (Pairformer + score model; PyTorch1.11 DDP-safe)"
echo "[V150Launcher] recurrence  = outer=3 state-evolving, MF recycling=0, final outer grad"
echo "[V150Launcher] frame       = PCS-RC H3 Kabsch -> full template -> proposal common center"
echo "[V150Launcher] frame leak  = native antibody coordinates forbidden"
echo "[V150Launcher] sequence    = raw MF structure latent 512 -> D3PM512"
echo "[V150Launcher] seq legacy  = H256 has zero numerical authority over logits"
echo "[V150Launcher] seq process = reversible Uniform(20) categorical bridge"
echo "[V150Launcher] seq reverse = exact x0-marginalized finite-step kernel"
echo "[V150Launcher] seq path    = internal reversible Uniform(20) bridge retained"
echo "[V150Launcher] seq terminal= MAP clean-endpoint posterior (argmax)"
echo "[V150Launcher] decision    = Bayes-optimal for per-residue AAR/CAAR 0-1 loss"
echo "[V150Launcher] seq state   = S_t sampler-only; hard neural identity/topology OFF"
echo "[V150Launcher] denoiser    = p_theta(S1 | X_t, t, PCS-RC context)"
echo "[V150Launcher] seq proposal= explicit PCS-RC S_pep zero-start residual condition"
echo "[V150Launcher] seq input   = pep_condition; native sequence context OFF"
echo "[V150Launcher] coord prop  = explicit PCS-RC X_pep SE3-invariant local residual"
echo "[V150Launcher] PCS local   = COMPLETE: coordinate + sequence proposal conditions"
echo "[V150Launcher] state recycle= outer_rounds=3; X/S/memory evolve between rounds"
echo "[V150Launcher] MF recycle   = 0 (nested representation recycle disabled)"
echo "[V150Launcher] outer grad   = intermediate no-grad; final round grad"
echo "[V150Launcher] eval grad    = STRICT no-grad; inner loop preserves caller context"
echo "[V150Launcher] epoch Test   = EVERY epoch; logical batch=20; n_steps=10"
echo "[V150Launcher] Test memory  = eval autograd leak fixed; historical batch=20 restored"
echo "[V150Launcher] adapter rnd  = 1 (round0 placement; rounds1-2 local correction)"
echo "[V150Launcher] IMPORTANT: train.sh is NOT used"
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
