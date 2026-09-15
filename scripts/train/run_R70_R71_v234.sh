#!/usr/bin/env bash
set -euo pipefail

# V236 keeps the existing unified launcher filename deliberately: no launcher
# proliferation.  It now serves the two matched single-physical-field authority arms.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CONFIG_PATH=${1:-}
shift || true
GPU_CSV=""; MASTER_PORT=""; MASTER_ADDR=""; RESUME_CHECKPOINT=""

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/train/run_R70_R71_v234.sh <R72-or-R73-config.json> \
    --gpus 2,3,4 --port 29772 [--resume /path/to/last_stepXXXX.pt]

V236: GPU count is a runtime resource choice.  Pass any non-empty physical GPU
list on CLI; train/validation batch size is per GPU in JSON.  The two formal arms
differ ONLY in the primary analytic chart of one physical H3 field:
  R72 = carrier-primary -> endpoint derived analytically each refinement round
  R73 = endpoint-primary -> carrier derived analytically each refinement round
Training horizon comes ONLY from JSON; this launcher never overrides max_epoch.
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

# V236: GPU count is deliberately NOT hard-coded.  Normalize the common
# full-width Chinese comma as well, then derive torchrun nproc from the CLI list.
GPU_CSV=${GPU_CSV//，/,}
GPU_CSV=${GPU_CSV//、/,}
GPU_CSV=${GPU_CSV// /}
[[ -n "$GPU_CSV" ]] || { echo "--gpus is required" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
NPROC_PER_NODE=${#GPU_IDS[@]}

if [[ "$CONFIG_PATH" != /* ]]; then CONFIG_PATH="$PROJECT_ROOT/${CONFIG_PATH#./}"; fi
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 2; }

eval "$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import json, os, shlex, sys
cfg=json.load(open(sys.argv[1],encoding='utf-8')); root=sys.argv[2]
tr=cfg['training']; rt=cfg.get('runtime',{}); gen=cfg['generation']; test=cfg['data']['test']; ev=cfg.get('evaluation',{})
if 'gpus' in rt: raise SystemExit('runtime.gpus is forbidden; pass GPUs on CLI')
def abspath(v):
    return v if os.path.isabs(v) else os.path.abspath(os.path.join(root,v))
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
 'TEST_JSON':abspath(test['set']),
 'TEST_PEP':abspath(test['pep']),
 'TEST_SURF':abspath(test['surface']),
 'TEST_BATCH':gen.get('batch_size',20),
 'TEST_STEPS':gen.get('n_steps',10),
 'TEST_SEED':gen.get('seed',2023),
 'METRIC_WORKERS':ev.get('metric_workers',8),
 'SCI_FIRST':lg.get('science_first_steps',3),
 'SCI_INTERVAL':lg.get('science_interval',20),
 'OUTLIER_THRESHOLD':lg.get('train_loss_outlier_threshold',1000.0),
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
[[ "$TEST_STEPS" == "10" ]] || { echo "formal test must use n_steps=10" >&2; exit 2; }

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "resume checkpoint not found: $RESUME_CHECKPOINT" >&2; exit 2; }
  RUN_DIR=$(dirname "$(dirname "$RESUME_CHECKPOINT")")
  [[ "$(realpath "$(dirname "$RUN_DIR")")" == "$(realpath "$OUTPUT_ROOT")" ]] || { echo "cross-experiment resume forbidden" >&2; exit 2; }
  VERSION_BASE=$(basename "$RUN_DIR"); [[ "$VERSION_BASE" =~ ^version_([0-9]+)$ ]] || { echo "resume must live under version_N/checkpoint" >&2; exit 2; }
  VERSION=${BASH_REMATCH[1]}; unset ABFLOW_FIXED_VERSION || true
else
  VERSION=0
  while ! mkdir "$OUTPUT_ROOT/version_$VERSION" 2>/dev/null; do VERSION=$((VERSION+1)); done
  RUN_DIR="$OUTPUT_ROOT/version_$VERSION"; export ABFLOW_FIXED_VERSION="$VERSION"
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

# Preserve formal Train -> Validation -> observational Test exactly.
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

# Exact diagnostics from the ordinary forward only. PairGradientAudit was removed
# rather than patched because it never produced valid information under AMP.
export ABFLOW_GEOMETRY_FORENSICS=on
export ABFLOW_SAMPLE_FORENSICS=on
export ABFLOW_SAMPLE_STEP_RMS_ALERT_A="${ABFLOW_SAMPLE_STEP_RMS_ALERT_A:-10}"
export ABFLOW_COORD_AUDIT_INTERVAL="${ABFLOW_COORD_AUDIT_INTERVAL:-$SCI_INTERVAL}"
export ABFLOW_COORD_AUDIT_FIRST_STEPS="${ABFLOW_COORD_AUDIT_FIRST_STEPS:-$SCI_FIRST}"
export ABFLOW_GEOMETRY_AUTHORITY_INTERVAL=0
export ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD="${ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD:-$OUTLIER_THRESHOLD}"
export ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH="${ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH:-3}"
export ABFLOW_TQDM="${ABFLOW_TQDM:-on}"
export ABFLOW_SCI_LOG_FIRST_STEPS="${ABFLOW_SCI_LOG_FIRST_STEPS:-$SCI_FIRST}"
export ABFLOW_SCI_LOG_INTERVAL="${ABFLOW_SCI_LOG_INTERVAL:-$SCI_INTERVAL}"
export ABFLOW_RUNTIME_GUARD_STEPS="${ABFLOW_RUNTIME_GUARD_STEPS:-1}"

exec > >(stdbuf -oL -eL tee -a "$RUN_LOG") 2>&1

echo "[RunLog] canonical=$RUN_LOG latest=$LATEST_LOG"
echo "[RunVersion] fixed_version=$VERSION dir=$RUN_DIR"
echo "[RunConfig] config=$CONFIG_PATH"
echo "[RunResources] physical_gpus=$GPU_CSV nproc=$NPROC_PER_NODE per_gpu_train_val_batch=$PER_GPU_BATCH_SIZE effective_global_train_batch=$EFFECTIVE_GLOBAL_TRAIN_BATCH test_batch=$TEST_BATCH master_addr=$MASTER_ADDR port=$MASTER_PORT"
echo "[RunResume] checkpoint=${RESUME_CHECKPOINT:-scratch}"
echo "[TrainingHorizon] source=json max_epoch=$MAX_EPOCH patience=config launcher_epoch_override=none"
echo "[TrainValTestContract] order=train->validation->test checkpoint_selection=validation test_metrics=observation_only test_steps=$TEST_STEPS test_seed=$TEST_SEED"

python - "$CONFIG_PATH" "$OUTPUT_ROOT" <<'PY'
import json, os, sys
cp,out=sys.argv[1:3]; cfg=json.load(open(cp,encoding='utf-8')); exp=cfg['experiment']
eid=exp['id']; stem=os.path.splitext(os.path.basename(cp))[0]; role=str(exp.get('diagnostic_role','')).lower()
if not (eid==stem==os.path.basename(os.path.normpath(out))): raise SystemExit(f'identity mismatch: {eid} / {stem} / {out}')
if exp.get('protocol')!='formal_train_val_test': raise SystemExit('formal_train_val_test required')
loader=cfg['training']['loader']
if 'per_gpu_batch_size' not in loader: raise SystemExit('V236 requires training.loader.per_gpu_batch_size')
if int(loader['per_gpu_batch_size']) <= 0: raise SystemExit('per_gpu_batch_size must be positive')
if os.path.basename(cfg['data']['test']['set'])!='test.json': raise SystemExit('observational test.json required')
if int(cfg['training']['schedule']['max_epoch'])!=200: raise SystemExit('V236 formal configs use max_epoch=200 in JSON')
if int(cfg['model']['architecture']['iter_round'])!=3: raise SystemExit('V236 requires 3 physical refinement rounds')
sp=cfg['model']['representation']['single_pair']; pc=sp['pair_coordinate']; cc=sp['coordinate_controller']; pa=sp.get('physical_authority',{})
if pc.get('mode')!='direct_shared': raise SystemExit('V236 preserves pair_coordinate.mode=direct_shared')
if cc.get('mode')!='egnn_prenorm_raw': raise SystemExit('V236 preserves coordinate_controller.mode=egnn_prenorm_raw')
if not bool(sp.get('time_embed',True)): raise SystemExit('V236 both arms retain matched R70 R05+AbX time routing')
if float(cfg['loss']['distogram'].get('weight',0))!=0: raise SystemExit('V236 keeps Distogram off')
if float(cfg['loss']['smooth_lddt'].get('weight',0))!=0: raise SystemExit('V236 keeps smooth-lDDT off')
if cfg['generation'].get('terminal_coordinate_authority')!='integrated_carrier_direct_h3': raise SystemExit('V236 requires integrated carrier terminal authority')
if bool(cfg['generation'].get('terminal_kabsch_fusion',True)): raise SystemExit('V236 forbids terminal Kabsch fusion')
if cfg['generation'].get('fixed_context')!='unchanged_exactly': raise SystemExit('V236 requires fixed context unchanged exactly')
expected={
 'r72_singlefield_carrier_primary_analytic3r':'carrier_primary_analytic',
 'r73_singlefield_endpoint_primary_analytic3r':'endpoint_primary_analytic',
}
if role not in expected: raise SystemExit(f'V236 diagnostic_role must be one of {sorted(expected)}, got {role!r}')
mode=str(pa.get('mode','')).lower()
if mode!=expected[role]: raise SystemExit(f'{role} requires physical_authority.mode={expected[role]!r}, got {mode!r}')
if int(pa.get('physical_dof',0))!=1: raise SystemExit('V236 requires exactly one physical coordinate degree of freedom')
if int(pa.get('rounds',0))!=3: raise SystemExit('V236 authority closure must occur in all 3 physical rounds')
primary='carrier' if mode=='carrier_primary_analytic' else 'endpoint'
derived='endpoint' if primary=='carrier' else 'carrier'
print(f'[ExperimentIdentity] PASS id={eid} role={role} protocol={exp["protocol"]}')
print('[TimeAuthorityContract] R05_explicit=1 AbX_explicit=1 explicit_routes=2 matched_to_R70=1')
print(f'[SingleFieldContract] physical_dof=1 primary_chart={primary} derived_chart={derived} analytic=F01_gfree_hybrid frame_aware=AG_to_raw_to_AB rounds=3')
print('[AuthorityContract] structure=analytic_endpoint_chart transport=analytic_carrier_chart recurrence=both_charts_same_field sample_terminal=integrated_carrier fixed_context=exact')
print('[CoordinateControllerContract] pair=direct_shared controller=egnn_prenorm_raw raw_R05_vector=1 no_tanh=1 no_clipping=1 no_trust_radius=1')
print('[DiagnosticsContract] stage_actuator=ctx+inter+surf+out chart_roundtrip=on round_refinement=on discarded_proposal_gap=on outlier_stage_trace=on PairGradientAudit=removed')
print('[EvaluationContract] train_val_test=unchanged valgen=0 test=observation_only')
PY

python -m py_compile \
  "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" \
  "$PROJECT_ROOT/models/modules/am_enc.py" \
  "$PROJECT_ROOT/models/modules/am_egnn.py" \
  "$PROJECT_ROOT/trainer/AbFlow_trainer.py" \
  "$PROJECT_ROOT/train.py"

grep -Fq "carrier_primary_analytic" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo "single-field carrier authority code missing" >&2; exit 2; }
grep -Fq "endpoint_primary_analytic" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo "single-field endpoint authority code missing" >&2; exit 2; }
grep -Fq "[StageActuator]" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo "exact stage actuator audit missing" >&2; exit 2; }
grep -Fq "[SingleFieldAuthorityAudit]" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo "single-field authority audit missing" >&2; exit 2; }
if grep -Fq "_pair_gradient_audit" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || grep -Fq "_pair_gradient_audit" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"; then
  echo "failed pair-gradient audit implementation is still present" >&2; exit 2
fi
grep -Fq "gen_X[paratope_mask] = interface_X_final" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo "direct integrated carrier terminal code missing" >&2; exit 2; }
if grep -Fq "gen_X[ab] = torch.matmul(gen_X[ab], R.T) + trans" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"; then
  echo "legacy terminal Kabsch mutation is still present" >&2; exit 2
fi
grep -Fq "return 0.5 * (cos(step / self.max_step * pi) + 1) * 0.9" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || {
  echo "context_ratio schedule changed unexpectedly" >&2; exit 2;
}
echo "[Preflight] py_compile=PASS analytic_singlefield=PASS stage_diagnostics=PASS pair_gradient_audit_removed=PASS terminal_carrier=PASS no_terminal_kabsch=PASS context_ratio_unchanged=PASS"

python - "$PROJECT_ROOT" <<'PY'
import hashlib,os,sys
root=sys.argv[1]
for rel in ['models/AbFlow/AbFlow_model.py','trainer/AbFlow_trainer.py','models/modules/am_enc.py','models/modules/am_egnn.py','train.py','scripts/train/run_R70_R71_v234.sh']:
 p=os.path.join(root,rel)
 if os.path.isfile(p): print(f'[SourceSHA256] {rel} {hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]}')
PY

torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "$PROJECT_ROOT/train.py" --config "$CONFIG_PATH"
