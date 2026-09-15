#!/usr/bin/env bash
set -euo pipefail

# V242 unified launcher for the active single-field R72 parent and the
# single-state/two-sequential-operator R75 repair. GPU count is runtime-only.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CONFIG_PATH=${1:-}
shift || true
GPU_CSV=""; MASTER_PORT=""; MASTER_ADDR=""; RESUME_CHECKPOINT=""

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/train/run_R70_R71_v234.sh <R72-or-R75-config.json> \
    --gpus 2,3,4,5,6,7 --port 29775 \
    [--resume /same-experiment/version_N/checkpoint/last_stepXXXX.pt]

V242:
  R72 = current carrier-primary parent (latent native workspace; historical control).
  R75 = one physical state + two sequential geometric operators:
        R05 local endpoint operator -> analytic endpoint->carrier sync ->
        interface/Score-Flow transport operator -> analytic carrier->endpoint sync.

The launcher never hard-codes GPU count and never overrides max_epoch/context_ratio.
USAGE
}

[[ -n "$CONFIG_PATH" ]] || { usage; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_CSV=${2:-}; shift 2 ;;
    --port) MASTER_PORT=${2:-}; shift 2 ;;
    --master-addr) MASTER_ADDR=${2:-}; shift 2 ;;
    --resume) RESUME_CHECKPOINT=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

GPU_CSV=${GPU_CSV//，/,}
GPU_CSV=${GPU_CSV//、/,}
GPU_CSV=${GPU_CSV// /}
[[ -n "$GPU_CSV" ]] || { echo "--gpus is required" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
NPROC_PER_NODE=${#GPU_IDS[@]}
(( NPROC_PER_NODE >= 1 )) || { echo "--gpus must contain at least one GPU id" >&2; exit 2; }

if [[ "$CONFIG_PATH" != /* ]]; then CONFIG_PATH="$PROJECT_ROOT/${CONFIG_PATH#./}"; fi
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 2; }

eval "$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import json, os, shlex, sys
cfg=json.load(open(sys.argv[1],encoding='utf-8')); root=sys.argv[2]
tr=cfg['training']; rt=cfg.get('runtime',{}); gen=cfg['generation']; test=cfg['data']['test']; ev=cfg.get('evaluation',{})
if 'gpus' in rt: raise SystemExit('runtime.gpus is forbidden; pass GPUs on CLI')
def abspath(v): return v if os.path.isabs(v) else os.path.abspath(os.path.join(root,v))
out=abspath(tr['output_dir'])
resume=tr.get('schedule',{}).get('resume_checkpoint','') or ''
if resume: resume=abspath(resume)
lg=tr.get('logging',{})
vals={
 'OUTPUT_ROOT':out,
 'PER_GPU_BATCH_SIZE':tr['loader'].get('per_gpu_batch_size', tr['loader'].get('batch_size')),
 'MAX_EPOCH':tr['schedule']['max_epoch'],
 'CFG_RESUME_CHECKPOINT':resume,
 'CFG_MASTER_ADDR':rt.get('master_addr','127.0.0.1'),
 'OMP_THREADS':rt.get('omp_num_threads',2),
 'CUDA_ALLOC':rt.get('cuda_allocator','max_split_size_mb:128'),
 'TEST_JSON':abspath(test['set']), 'TEST_PEP':abspath(test['pep']), 'TEST_SURF':abspath(test['surface']),
 'TEST_BATCH':gen.get('batch_size',20), 'TEST_STEPS':gen.get('n_steps',10), 'TEST_SEED':gen.get('seed',2023),
 'METRIC_WORKERS':ev.get('metric_workers',8),
 'SCI_FIRST':lg.get('science_first_steps',0), 'SCI_INTERVAL':lg.get('science_interval',0),
 'OUTLIER_THRESHOLD':lg.get('train_loss_outlier_threshold',1000.0),
 'EXP_ID':cfg.get('experiment',{}).get('id',''), 'EXP_ROLE':cfg.get('experiment',{}).get('diagnostic_role',''),
}
for k,v in vals.items(): print(f'{k}='+shlex.quote(str(v)))
PY
)"

MASTER_ADDR=${MASTER_ADDR:-$CFG_MASTER_ADDR}
[[ -n "$MASTER_PORT" && "$MASTER_PORT" =~ ^[0-9]+$ ]] || { echo "--port is required" >&2; exit 2; }
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-$CFG_RESUME_CHECKPOINT}
[[ -n "$PER_GPU_BATCH_SIZE" && "$PER_GPU_BATCH_SIZE" =~ ^[0-9]+$ && "$PER_GPU_BATCH_SIZE" -gt 0 ]] || {
  echo "training.loader.per_gpu_batch_size must be a positive integer" >&2; exit 2;
}
EFFECTIVE_GLOBAL_TRAIN_BATCH=$((PER_GPU_BATCH_SIZE * NPROC_PER_NODE))
[[ "$TEST_STEPS" == "10" ]] || { echo "formal observational test must use n_steps=10" >&2; exit 2; }

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "resume checkpoint not found: $RESUME_CHECKPOINT" >&2; exit 2; }
  RUN_DIR=$(dirname "$(dirname "$RESUME_CHECKPOINT")")
  [[ "$(realpath "$(dirname "$RUN_DIR")")" == "$(realpath "$OUTPUT_ROOT")" ]] || {
    echo "cross-experiment resume forbidden: checkpoint must belong to config output_dir" >&2; exit 2;
  }
  VERSION_BASE=$(basename "$RUN_DIR")
  [[ "$VERSION_BASE" =~ ^version_([0-9]+)$ ]] || { echo "resume must live under version_N/checkpoint" >&2; exit 2; }
  VERSION=${BASH_REMATCH[1]}
  unset ABFLOW_FIXED_VERSION || true
else
  VERSION=0
  while ! mkdir "$OUTPUT_ROOT/version_$VERSION" 2>/dev/null; do VERSION=$((VERSION+1)); done
  RUN_DIR="$OUTPUT_ROOT/version_$VERSION"
  export ABFLOW_FIXED_VERSION="$VERSION"
fi
RUN_LOG="$RUN_DIR/run_time.log"; LATEST_LOG="$OUTPUT_ROOT/run_time.log"
mkdir -p "$RUN_DIR"; ln -sfn "version_$VERSION/run_time.log" "$LATEST_LOG"

export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_CSV"
export ABFLOW_NPROC_PER_NODE="$NPROC_PER_NODE"
export ABFLOW_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
export OMP_NUM_THREADS="$OMP_THREADS"
export PYTHONUNBUFFERED=1
[[ -n "$CUDA_ALLOC" ]] && export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC"

# Preserve current development protocol exactly.
export ABFLOW_EPOCH_TEST=on
export ABFLOW_EPOCH_TEST_INTERVAL=1
export ABFLOW_EPOCH_TEST_JSON="$TEST_JSON"
export ABFLOW_EPOCH_TEST_PEP="$TEST_PEP"
export ABFLOW_EPOCH_TEST_SURF="$TEST_SURF"
export ABFLOW_EPOCH_TEST_BATCH_SIZE="$TEST_BATCH"
export ABFLOW_EPOCH_TEST_N_STEPS="$TEST_STEPS"
export ABFLOW_EPOCH_TEST_BASE_SEED="$TEST_SEED"
export ABFLOW_EPOCH_TEST_METRIC_WORKERS="$METRIC_WORKERS"
export ABFLOW_EPOCH_TEST_FAIL_FAST=on
export ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY=record_and_continue
export ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS="${ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS:-on}"

# Progress bars ON; routine forensic spam OFF. Outlier forensics remain available.
export ABFLOW_TQDM="${ABFLOW_TQDM:-on}"
export ABFLOW_GEOMETRY_FORENSICS="${ABFLOW_GEOMETRY_FORENSICS:-off}"
export ABFLOW_SAMPLE_FORENSICS="${ABFLOW_SAMPLE_FORENSICS:-off}"
export ABFLOW_COORD_AUDIT_INTERVAL="${ABFLOW_COORD_AUDIT_INTERVAL:-1000000000}"
export ABFLOW_COORD_AUDIT_FIRST_STEPS="${ABFLOW_COORD_AUDIT_FIRST_STEPS:-0}"
export ABFLOW_GEOMETRY_AUTHORITY_INTERVAL=0
export ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD="${ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD:-$OUTLIER_THRESHOLD}"
export ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH="${ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH:-3}"
export ABFLOW_SCI_LOG_FIRST_STEPS="${ABFLOW_SCI_LOG_FIRST_STEPS:-$SCI_FIRST}"
export ABFLOW_SCI_LOG_INTERVAL="${ABFLOW_SCI_LOG_INTERVAL:-$SCI_INTERVAL}"
export ABFLOW_RUNTIME_GUARD_STEPS="${ABFLOW_RUNTIME_GUARD_STEPS:-1}"

exec > >(stdbuf -oL -eL tee -a "$RUN_LOG") 2>&1

echo "[RunLog] canonical=$RUN_LOG latest=$LATEST_LOG"
echo "[RunVersion] fixed_version=$VERSION dir=$RUN_DIR"
echo "[RunConfig] config=$CONFIG_PATH"
echo "[RunResources] physical_gpus=$GPU_CSV nproc=$NPROC_PER_NODE per_gpu_train_val_batch=$PER_GPU_BATCH_SIZE effective_global_train_batch=$EFFECTIVE_GLOBAL_TRAIN_BATCH test_batch=$TEST_BATCH master_addr=$MASTER_ADDR port=$MASTER_PORT"
echo "[RunResume] checkpoint=${RESUME_CHECKPOINT:-scratch}"
echo "[TrainingHorizon] source=json max_epoch=$MAX_EPOCH launcher_epoch_override=none"
echo "[TrainValTestContract] order=train->validation->test checkpoint_selection=validation test_metrics=observation_only test_steps=$TEST_STEPS test_seed=$TEST_SEED"

python - "$CONFIG_PATH" "$OUTPUT_ROOT" <<'PY'
import json, os, sys
cp,out=sys.argv[1:3]; cfg=json.load(open(cp,encoding='utf-8')); exp=cfg['experiment']
eid=exp['id']; stem=os.path.splitext(os.path.basename(cp))[0]; role=str(exp.get('diagnostic_role','')).lower()
if not (eid==stem==os.path.basename(os.path.normpath(out))): raise SystemExit(f'identity mismatch: {eid} / {stem} / {out}')
if exp.get('protocol')!='formal_train_val_test': raise SystemExit('formal_train_val_test required')
loader=cfg['training']['loader']
if int(loader.get('per_gpu_batch_size',0)) <= 0: raise SystemExit('per_gpu_batch_size must be positive')
if int(cfg['training']['schedule']['max_epoch']) != 200: raise SystemExit('V242 development protocol requires max_epoch=200')
if int(cfg['model']['architecture']['iter_round']) != 3: raise SystemExit('V242 requires three physical rounds')
sp=cfg['model']['representation']['single_pair']; pc=sp['pair_coordinate']; cc=sp['coordinate_controller']; pa=sp['physical_authority']
if pc.get('mode')!='direct_shared': raise SystemExit('Pair direct_shared must remain unchanged')
if cc.get('mode')!='egnn_prenorm_raw': raise SystemExit('R05 raw Cartesian + coordinate PreNorm must remain unchanged')
if not bool(sp.get('time_embed',True)): raise SystemExit('validated explicit time routing must remain enabled')
if float(cfg['loss']['distogram'].get('weight',0)) != 0: raise SystemExit('Distogram must remain off')
if float(cfg['loss']['smooth_lddt'].get('weight',0)) != 0: raise SystemExit('smooth-lDDT must remain off')
if pa.get('mode')!='carrier_primary_analytic': raise SystemExit('carrier_primary_analytic is the only formal physical authority')
if int(pa.get('physical_dof',0)) != 1: raise SystemExit('exactly one physical H3 state is required')
if int(pa.get('rounds',0)) != 3: raise SystemExit('authority must close in all three rounds')
expected={
 'r72_singlefield_carrier_primary_analytic3r':'latent_native_workspace',
 'r75_single_state_sequential_local_transport_analytic3r':'sequential_local_then_transport',
}
if role not in expected: raise SystemExit(f'unsupported V242 role: {role!r}')
op=str(pa.get('geometric_operator_mode','latent_native_workspace')).lower()
if op != expected[role]: raise SystemExit(f'{role} requires geometric_operator_mode={expected[role]}, got {op}')
if bool(pa.get('strict_single_cartesian',False)): raise SystemExit('retired strict_single_cartesian branch must not be enabled')
gen=cfg['generation']
if gen.get('terminal_coordinate_authority')!='integrated_carrier_direct_h3': raise SystemExit('integrated carrier terminal required')
if bool(gen.get('terminal_kabsch_fusion',True)): raise SystemExit('terminal Kabsch fusion forbidden')
if gen.get('fixed_context')!='unchanged_exactly': raise SystemExit('fixed context must remain exact')
print(f'[ExperimentIdentity] PASS id={eid} role={role} protocol={exp["protocol"]}')
print(f'[PhysicalStateContract] physical_dof=1 authority=carrier endpoint=analytic operator_mode={op} rounds=3')
if op=='sequential_local_then_transport':
    print('[SequentialOperatorContract] local=R05_ctx_out_endpoint physical_writeback=analytic_endpoint_to_carrier transport=inter_surface_carrier resync=analytic_carrier_to_endpoint')
print('[CoordinateControllerContract] pair=direct_shared controller=egnn_prenorm_raw raw_R05_vector=1 no_tanh=1 no_clipping=1 no_trust_radius=1')
print('[LossContract] weights=sequence1+structure1+interface1+edge1 distogram0 smooth_lddt0 unchanged=1')
print('[DiagnosticsContract] GeometryValidation=on TrainValidationGap=on SequentialOperatorValidation=on routine_stage_trace=off progress=train+validation+test:on')
PY

python -m py_compile \
  "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" \
  "$PROJECT_ROOT/models/modules/am_enc.py" \
  "$PROJECT_ROOT/models/modules/am_egnn.py" \
  "$PROJECT_ROOT/trainer/AbFlow_trainer.py" \
  "$PROJECT_ROOT/train.py"

grep -Fq "sequential_local_then_transport" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo "V242 sequential operator model path missing" >&2; exit 2; }
grep -Fq "endpoint_to_carrier_sync_fn" "$PROJECT_ROOT/models/modules/am_enc.py" || { echo "local->carrier writeback missing" >&2; exit 2; }
grep -Fq "carrier_to_endpoint_sync_fn" "$PROJECT_ROOT/models/modules/am_enc.py" || { echo "carrier->endpoint resync missing" >&2; exit 2; }
grep -Fq "[SequentialOperatorValidation]" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo "operator validation log missing" >&2; exit 2; }
if grep -Fq "elif self.physical_authority_mode == 'endpoint_primary_analytic'" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"; then
  echo "retired R73 endpoint-primary branch is present" >&2; exit 2
fi
grep -Fq "gen_X[paratope_mask] = interface_X_final" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo "direct integrated carrier terminal missing" >&2; exit 2; }
if grep -Fq "gen_X[ab] = torch.matmul(gen_X[ab], R.T) + trans" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"; then
  echo "legacy terminal Kabsch mutation is present" >&2; exit 2
fi
grep -Fq "return 0.5 * (cos(step / self.max_step * pi) + 1) * 0.9" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo "context_ratio changed unexpectedly" >&2; exit 2; }
echo "[Preflight] py_compile=PASS single_physical_state=PASS sequential_two_operators=PASS endpoint_primary_removed=PASS terminal_carrier=PASS no_terminal_kabsch=PASS context_ratio_unchanged=PASS"

python - "$PROJECT_ROOT" <<'PY'
import hashlib,os,sys
root=sys.argv[1]
for rel in ['models/AbFlow/AbFlow_model.py','models/modules/am_enc.py','models/modules/am_egnn.py','trainer/AbFlow_trainer.py','train.py','scripts/train/run_R70_R71_v234.sh']:
 p=os.path.join(root,rel)
 if os.path.isfile(p): print(f'[SourceSHA256] {rel} {hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]}')
PY

set +e
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "$PROJECT_ROOT/train.py" --config "$CONFIG_PATH"
STATUS=$?
set -e
echo "[RunComplete] experiment=$EXP_ID version=$VERSION exit_code=$STATUS configured_max_epoch=$MAX_EPOCH"
exit "$STATUS"
