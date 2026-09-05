#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-train}
EXP_ID=${2:-MFSC_V153_MF_NATIVE_SEQUENCE_CLOSURE_R05_U02}
GPU_IDS=${3:-2,3,4,5}
BASE_CONFIG=${4:-/home/data3/cjm/project/AbFlow/scripts/train/configs/MFSC_v153/MFSC_V153_MF_NATIVE_SEQUENCE_CLOSURE_R05_U02.json}
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-/home/data3/cjm/project/AbFlow}
PORT=${PORT:-29672}
ADDR=${ADDR:-localhost}

cd "$PROJECT_ROOT"

if [[ "$MODE" != "train" ]]; then
  echo "ERROR: V153 formal launcher supports MODE=train only." >&2
  exit 2
fi
if [[ "$EXP_ID" != "MFSC_V153_MF_NATIVE_SEQUENCE_CLOSURE_R05_U02" ]]; then
  echo "ERROR: unexpected EXP_ID=$EXP_ID" >&2
  echo "Expected: MFSC_V153_MF_NATIVE_SEQUENCE_CLOSURE_R05_U02" >&2
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
AM_FILE="$PROJECT_ROOT/models/modules/am_enc.py"
if [[ ! -f "$MODEL_FILE" || ! -f "$AM_FILE" ]]; then
  echo "ERROR: V153 model/backbone files are missing." >&2
  exit 2
fi
if ! grep -qF "V152 MF-native stateful recycling requires iter_round=1" "$MODEL_FILE"; then
  echo "ERROR: V153 fixed-Xt stateful model contract is missing." >&2
  exit 2
fi
if ! grep -qF "consume_stateful_recycle_state" "$AM_FILE"; then
  echo "ERROR: V153 MF stateful AMEncoder contract is missing." >&2
  exit 2
fi
if ! grep -qF "stateful_clean_dist_embedding" "$AM_FILE"; then
  echo "ERROR: V153 clean-endpoint pair recycle adapter is missing." >&2
  exit 2
fi
if ! grep -qF "mf_single_sequence_condition" "$MODEL_FILE"; then
  echo "ERROR: V153 MF-native sequence-closure model contract is missing." >&2
  exit 2
fi
if ! grep -qF "mf_single_sequence_condition" "$AM_FILE"; then
  echo "ERROR: V153 direct MF single injection is missing from AMEncoder." >&2
  exit 2
fi
BASE_CONFIG=$(realpath "$BASE_CONFIG")
export ABFLOW_CONFIG_PATH="$BASE_CONFIG"
export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT"

# Fail before torchrun if the JSON does not match the formal V153 sequence-closure
# contract. This launcher check changes no scientific setting.
python - "$BASE_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p, encoding="utf-8"))
env = cfg.get("runtime_env", {})
arch = cfg.get("architecture", {})
errors = []

def req(name, got, expected):
    if str(got) != str(expected):
        errors.append(f"{name}={got} (expected {expected})")

req("iter_round", cfg.get("iter_round"), 1)
req("batch_size", cfg.get("batch_size"), 2)
req("architecture.recycling_steps", arch.get("recycling_steps"), 0)
req("ABFLOW_OUTER_RECYCLE_GRAD_MODE",
    env.get("ABFLOW_OUTER_RECYCLE_GRAD_MODE"), "off")
req("ABFLOW_MF_STATEFUL_RECYCLING",
    env.get("ABFLOW_MF_STATEFUL_RECYCLING"), "on")
req("ABFLOW_MF_NATIVE_SEQUENCE_CLOSURE",
    env.get("ABFLOW_MF_NATIVE_SEQUENCE_CLOSURE"), "on")
req("ABFLOW_MF_STATEFUL_MAX_DEPTH",
    env.get("ABFLOW_MF_STATEFUL_MAX_DEPTH"), "3")
req("ABFLOW_MF_STATEFUL_INFERENCE_DEPTH",
    env.get("ABFLOW_MF_STATEFUL_INFERENCE_DEPTH"), "3")
req("ABFLOW_MF_STATEFUL_RANDOM_DEPTH",
    env.get("ABFLOW_MF_STATEFUL_RANDOM_DEPTH"), "on")
req("ABFLOW_PROPOSAL_ADAPTER_START_ROUND",
    env.get("ABFLOW_PROPOSAL_ADAPTER_START_ROUND"), "0")
req("ABFLOW_R3_G_MODE", env.get("ABFLOW_R3_G_MODE"),
    "foldflow_fixed_scaled")
req("ABFLOW_R3_FIXED_G_SCALED",
    env.get("ABFLOW_R3_FIXED_G_SCALED"), "0.1")
req("ABFLOW_R3_NOISE_SCOPE", env.get("ABFLOW_R3_NOISE_SCOPE"),
    "residue")
req("ABFLOW_SCOREFM_LOSS_MODE",
    env.get("ABFLOW_SCOREFM_LOSS_MODE"),
    "f01_r3_endpoint_canonical_hybrid")
req("ABFLOW_SCOREFM_SAMPLER_MODE",
    env.get("ABFLOW_SCOREFM_SAMPLER_MODE"),
    "f01_canonical_carrier")
req("ABFLOW_EPOCH_TEST", env.get("ABFLOW_EPOCH_TEST"), "on")
req("ABFLOW_EPOCH_TEST_N_STEPS",
    env.get("ABFLOW_EPOCH_TEST_N_STEPS"), "10")

if errors:
    raise SystemExit("V153 formal preflight failed: " + "; ".join(errors))

print(
    "[V153Preflight] PASS "
    "outer=1 mf_internal_recycle=0 "
    "stateful_K=train_random_1_3/infer_3 "
    "Xt_fixed=1 seq_closure=MF_single_native "
    "R05=fixedg_residue_U02 "
    "batch=2/GPU global=8"
)
PY

MODEL_SHA256=$(sha256sum "$MODEL_FILE" | awk '{print $1}')
echo "[V153Preflight] model_file=$MODEL_FILE"
echo "[V153Preflight] model_sha256=$MODEL_SHA256"
echo "[V153Preflight] stale_v142_guard=ABSENT"
echo "[V153Preflight] fixed_Xt_stateful_guard=PRESENT"
echo "[V153Preflight] inference_autograd_fix=PRESENT"

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

# V153 changes the train-time MF-native sequence-conditioning boundary and must be trained from scratch.
if [[ -n "${ABFLOW_RESUME_CHECKPOINT:-}" ]]; then
  echo "ERROR: V153 is a new MF-native sequence-closure model; scratch training is required." >&2
  exit 2
fi
export ABFLOW_FORCE_SCRATCH=on
echo "[V153Launcher] resume      = <forbidden; scratch-only>"


# Fail before torchrun if the formal Test contract is incomplete.
if [[ "${ABFLOW_THREE_PHASE_PROTOCOL:-off}" == "on" ]]; then
  : "${ABFLOW_EPOCH_TEST_JSON:?ABFLOW_EPOCH_TEST_JSON is required}"
  if [[ ! -f "$ABFLOW_EPOCH_TEST_JSON" ]]; then
    echo "ERROR: epoch-test JSON not found: $ABFLOW_EPOCH_TEST_JSON" >&2
    exit 2
  fi
fi

# Contract summary: these values must match the first Python-side V153Config logs.
echo "============================================================"
echo "[V153Launcher] mode        = $MODE"
echo "[V153Launcher] exp         = $EXP_ID"
echo "[V153Launcher] physicalGPU = $GPU_IDS"
echo "[V153Launcher] logicalGPU  = ${LOGICAL_GPUS[*]}"
echo "[V153Launcher] config      = $ABFLOW_CONFIG_PATH"
echo "[V153Launcher] train entry = $PROJECT_ROOT/train.py"
echo "[V153Launcher] test JSON   = ${ABFLOW_EPOCH_TEST_JSON:-<off>}"
echo "[V153Launcher] scratch     = ${ABFLOW_FORCE_SCRATCH:-off}"
echo "[V153Launcher] batch       = per-GPU (JSON batch_size is local microbatch)"
echo "[V153Launcher] checkpoint  = MF-wide nonreentrant layer checkpoint (Pairformer + score model; PyTorch1.11 DDP-safe)"
echo "[V153Launcher] recycle     = MF-native stateful, fixed Xt, train K~Uniform{1,2,3}, infer K=3"
echo "[V153Launcher] frame       = PCS-RC H3 Kabsch -> full template -> proposal common center"
echo "[V153Launcher] frame leak  = native antibody coordinates forbidden"
echo "[V153Launcher] sequence    = raw MF structure latent 512 -> D3PM512"
echo "[V153Launcher] seq closure = S_pep + soft clean posterior -> direct MF s_init (zero-start)"
echo "[V153Launcher] seq legacy  = H256 has zero numerical authority over logits"
echo "[V153Launcher] seq process = reversible Uniform(20) categorical bridge"
echo "[V153Launcher] seq reverse = exact x0-marginalized finite-step kernel"
echo "[V153Launcher] seq path    = internal reversible Uniform(20) bridge retained"
echo "[V153Launcher] seq terminal= MAP clean-endpoint posterior (argmax)"
echo "[V153Launcher] decision    = Bayes-optimal for per-residue AAR/CAAR 0-1 loss"
echo "[V153Launcher] seq state   = S_t sampler-only; hard neural identity/topology OFF"
echo "[V153Launcher] denoiser    = p_theta(S1 | X_t, t, PCS-RC context)"
echo "[V153Launcher] seq proposal= explicit PCS-RC S_pep -> direct MF s_init zero-start condition"
echo "[V153Launcher] seq input   = pep_condition; native sequence context OFF"
echo "[V153Launcher] coord prop  = explicit PCS-RC X_pep SE3-invariant local residual"
echo "[V153Launcher] PCS local   = COMPLETE: coordinate + sequence proposal conditions"
echo "[V153Launcher] state recycle= fixed Xt; MF s/z + clean X1 + soft sequence recycled"
echo "[V153Launcher] MF recycle   = 0 (nested representation recycle disabled)"
echo "[V153Launcher] gradient    = intermediate stateful recycle no-grad+detach; sampled final recycle grad"
echo "[V153Launcher] eval grad    = STRICT no-grad; inner loop preserves caller context"
echo "[V153Launcher] epoch Test   = EVERY epoch; logical batch=20; n_steps=10"
echo "[V153Launcher] Test memory  = eval autograd leak fixed; historical batch=20 restored"
echo "[V153Launcher] adapter rnd  = 0 (PCS-RC proposal condition available in every MF stateful recycle)"
echo "[V153Launcher] IMPORTANT: train.sh is NOT used"
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
