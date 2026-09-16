#!/usr/bin/env bash
set -euo pipefail

# R79 single-factor launcher: R77 + round-consistent current-state Single/Pair.
# No prev_seq/prev_pair/prev_pos recycle state is permitted in this experiment.
# Carrier remains the sole learned H3 Cartesian authority; F01/loss/sampler stay matched to R77.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CONFIG_PATH=${1:-}
shift || true
GPU_CSV=""; MASTER_PORT=""; MASTER_ADDR=""

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/train/run_R79_round_state_singlepair_v247.sh <R79-config.json> \
    --gpus 2,3,4,5,6,7 --port 29779

R79 contract:
  - parent = R77 and starts from scratch;
  - the ONLY scientific factor is current-state Single/Pair recomputation once per macro round;
  - round0 relational H3 state = outer Xt; rounds1/2 = previous analytic endpoint;
  - no prev_seq/prev_pair/prev_pos and no second recurrent memory branch;
  - Pair stays static inside each EGNN round; memory_H/pred_S_dist stay inherited;
  - carrier-primary single field, F01 path/sampler and final-round-only supervision unchanged.
USAGE
}

[[ -n "$CONFIG_PATH" ]] || { usage; exit 2; }
echo "[LauncherStart] config=$CONFIG_PATH args=$*"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_CSV=${2:-}; shift 2 ;;
    --port) MASTER_PORT=${2:-}; shift 2 ;;
    --master-addr) MASTER_ADDR=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

GPU_CSV=${GPU_CSV//，/,}; GPU_CSV=${GPU_CSV//、/,}; GPU_CSV=${GPU_CSV// /}
[[ -n "$GPU_CSV" ]] || { echo "--gpus is required" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
NPROC_PER_NODE=${#GPU_IDS[@]}
(( NPROC_PER_NODE >= 1 )) || { echo "--gpus must contain at least one GPU id" >&2; exit 2; }
if [[ "$CONFIG_PATH" != /* ]]; then CONFIG_PATH="$PROJECT_ROOT/${CONFIG_PATH#./}"; fi
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 2; }

eval "$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY2'
import json, os, shlex, sys
cfg=json.load(open(sys.argv[1],encoding='utf-8')); root=sys.argv[2]
tr=cfg['training']; rt=cfg.get('runtime',{}); gen=cfg['generation']; test=cfg['data']['test']; ev=cfg.get('evaluation',{})
if 'gpus' in rt: raise SystemExit('runtime.gpus is forbidden; pass GPUs on CLI')
def abspath(v): return v if os.path.isabs(v) else os.path.abspath(os.path.join(root,v))
lg=tr.get('logging',{})
vals={
 'OUTPUT_ROOT':abspath(tr['output_dir']),
 'PER_GPU_BATCH_SIZE':tr['loader'].get('per_gpu_batch_size', tr['loader'].get('batch_size')),
 'MAX_EPOCH':tr['schedule']['max_epoch'], 'CFG_MASTER_ADDR':rt.get('master_addr','127.0.0.1'),
 'OMP_THREADS':rt.get('omp_num_threads',2), 'CUDA_ALLOC':rt.get('cuda_allocator','max_split_size_mb:128'),
 'TEST_JSON':abspath(test['set']), 'TEST_PEP':abspath(test['pep']), 'TEST_SURF':abspath(test['surface']),
 'TEST_BATCH':gen.get('batch_size',20), 'TEST_STEPS':gen.get('n_steps',10), 'TEST_SEED':gen.get('seed',2023),
 'METRIC_WORKERS':ev.get('metric_workers',8), 'SCI_FIRST':lg.get('science_first_steps',0),
 'SCI_INTERVAL':lg.get('science_interval',0), 'OUTLIER_THRESHOLD':lg.get('train_loss_outlier_threshold',1000.0),
 'EXP_ID':cfg['experiment']['id'], 'EXP_ROLE':cfg['experiment'].get('diagnostic_role',''),
}
for k,v in vals.items(): print(f'{k}='+shlex.quote(str(v)))
PY2
)"

MASTER_ADDR=${MASTER_ADDR:-$CFG_MASTER_ADDR}
[[ -n "$MASTER_PORT" && "$MASTER_PORT" =~ ^[0-9]+$ ]] || { echo "--port is required" >&2; exit 2; }
[[ "$TEST_STEPS" == "10" ]] || { echo "formal observational Test requires n_steps=10" >&2; exit 2; }
EFFECTIVE_GLOBAL_TRAIN_BATCH=$((PER_GPU_BATCH_SIZE * NPROC_PER_NODE))

mkdir -p "$OUTPUT_ROOT" || { echo "Failed to create OUTPUT_ROOT: $OUTPUT_ROOT" >&2; exit 2; }
VERSION=0
while :; do
  RUN_DIR="$OUTPUT_ROOT/version_$VERSION"
  if mkdir "$RUN_DIR" 2>/dev/null; then break; fi
  if [[ ! -e "$RUN_DIR" ]]; then echo "Failed to create run directory: $RUN_DIR" >&2; exit 2; fi
  VERSION=$((VERSION+1))
done
export ABFLOW_FIXED_VERSION="$VERSION"
export ABFLOW_EXPECTED_RUN_DIR="$RUN_DIR"
export ABFLOW_REQUIRE_SCRATCH=1
export ABFLOW_RESUME_CHECKPOINT=""
RUN_LOG="$RUN_DIR/run_time.log"; LATEST_LOG="$OUTPUT_ROOT/run_time.log"
ln -sfn "version_$VERSION/run_time.log" "$LATEST_LOG"

export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES="$GPU_CSV" ABFLOW_NPROC_PER_NODE="$NPROC_PER_NODE"
export OMP_NUM_THREADS="$OMP_THREADS" PYTHONUNBUFFERED=1
[[ -n "$CUDA_ALLOC" ]] && export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC"
export ABFLOW_EPOCH_TEST=on ABFLOW_EPOCH_TEST_INTERVAL=1
export ABFLOW_EPOCH_TEST_JSON="$TEST_JSON" ABFLOW_EPOCH_TEST_PEP="$TEST_PEP" ABFLOW_EPOCH_TEST_SURF="$TEST_SURF"
export ABFLOW_EPOCH_TEST_BATCH_SIZE="$TEST_BATCH" ABFLOW_EPOCH_TEST_N_STEPS="$TEST_STEPS" ABFLOW_EPOCH_TEST_BASE_SEED="$TEST_SEED"
export ABFLOW_EPOCH_TEST_METRIC_WORKERS="$METRIC_WORKERS" ABFLOW_EPOCH_TEST_FAIL_FAST=on
export ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY=record_and_continue ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS="${ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS:-on}"
export ABFLOW_TQDM="${ABFLOW_TQDM:-on}" ABFLOW_GEOMETRY_FORENSICS="${ABFLOW_GEOMETRY_FORENSICS:-off}"
export ABFLOW_SAMPLE_FORENSICS="${ABFLOW_SAMPLE_FORENSICS:-off}" ABFLOW_SAMPLE_AUTHORITY_DIAGNOSTICS="${ABFLOW_SAMPLE_AUTHORITY_DIAGNOSTICS:-on}"
export ABFLOW_STATE_EXPOSURE_AUDIT=off
export ABFLOW_COORD_AUDIT_INTERVAL="${ABFLOW_COORD_AUDIT_INTERVAL:-1000000000}" ABFLOW_COORD_AUDIT_FIRST_STEPS="${ABFLOW_COORD_AUDIT_FIRST_STEPS:-0}"
export ABFLOW_GEOMETRY_AUTHORITY_INTERVAL=0
export ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD="${ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD:-$OUTLIER_THRESHOLD}"
export ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH="${ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH:-3}"
export ABFLOW_SCI_LOG_FIRST_STEPS="${ABFLOW_SCI_LOG_FIRST_STEPS:-$SCI_FIRST}" ABFLOW_SCI_LOG_INTERVAL="${ABFLOW_SCI_LOG_INTERVAL:-$SCI_INTERVAL}"
export ABFLOW_RUNTIME_GUARD_STEPS="${ABFLOW_RUNTIME_GUARD_STEPS:-1}"

exec > >(stdbuf -oL -eL tee -a "$RUN_LOG") 2>&1

echo "[RunLog] canonical=$RUN_LOG latest=$LATEST_LOG"
echo "[RunVersion] fixed_version=$VERSION dir=$RUN_DIR"
echo "[RunConfig] config=$CONFIG_PATH"
echo "[RunResources] physical_gpus=$GPU_CSV nproc=$NPROC_PER_NODE per_gpu_train_val_batch=$PER_GPU_BATCH_SIZE effective_global_train_batch=$EFFECTIVE_GLOBAL_TRAIN_BATCH test_batch=$TEST_BATCH master_addr=$MASTER_ADDR port=$MASTER_PORT"
echo "[RunInit] mode=scratch parent=R77 optimizer=reset ema=reset epoch=0 best_val=reset"
echo "[TrainingHorizon] source=json max_epoch=$MAX_EPOCH launcher_epoch_override=none"
echo "[TrainValTestContract] order=train->validation->test checkpoint_selection=validation test_metrics=observation_only test_steps=$TEST_STEPS test_seed=$TEST_SEED"

python - "$CONFIG_PATH" "$OUTPUT_ROOT" <<'PY2'
import json, os, sys
cp,out=sys.argv[1:3]; cfg=json.load(open(cp,encoding='utf-8')); exp=cfg['experiment']
eid=exp['id']; stem=os.path.splitext(os.path.basename(cp))[0]; role=str(exp.get('diagnostic_role','')).lower()
if not (eid==stem==os.path.basename(os.path.normpath(out))): raise SystemExit(f'identity mismatch: {eid} / {stem} / {out}')
if exp.get('protocol')!='formal_train_val_test': raise SystemExit('formal_train_val_test required')
if role!='r79_round_state_singlepair_only': raise SystemExit(f'R79 role required, got {role!r}')
if exp.get('parent')!='R77_R05_ABX_R72_AUTHORITY_CLOSED_FINALROUND_ANALYTIC3R_TVT_U02': raise SystemExit('R79 must use R77 as parent identity')
if str(exp.get('initialization','')).lower()!='scratch': raise SystemExit('R79 must start from scratch')
if str(cfg['training']['schedule'].get('resume_checkpoint','') or '').strip(): raise SystemExit('R79 forbids resume_checkpoint')
if int(cfg['model']['architecture']['iter_round']) != 3: raise SystemExit('R79 requires exactly 3 refinement rounds')
sp=cfg['model']['representation']['single_pair']; pc=sp['pair_coordinate']; cc=sp['coordinate_controller']; pa=sp['physical_authority']; rs=sp.get('round_state_conditioning',{})
if pc.get('mode')!='direct_shared' or cc.get('mode')!='egnn_prenorm_raw': raise SystemExit('R77 Pair/controller must remain unchanged')
if not bool(sp.get('time_embed',True)): raise SystemExit('explicit Single/Pair time must remain on')
if pa.get('mode')!='carrier_primary_analytic' or int(pa.get('physical_dof',0))!=1: raise SystemExit('carrier-primary single field required')
if pa.get('geometric_operator_mode')!='latent_native_workspace': raise SystemExit('R77 latent native workspace must be retained')
if pa.get('native_latent_workspace')!='retained_from_R72': raise SystemExit('native latent workspace retention must remain explicit')
if pa.get('refinement_supervision')!='final_round_only_inherited_from_AbFlow': raise SystemExit('final-round-only supervision must remain unchanged')
if pa.get('structure_supervision_mask')!='paratope_only' or pa.get('fixed_context_writeback')!='paratope_only': raise SystemExit('H3 authority closure required')
if not bool(rs.get('enabled',False)) or rs.get('geometry_source')!='current_authoritative_state': raise SystemExit('round-state Single/Pair factor missing')
if any(bool(rs.get(k,False)) for k in ('prev_seq','prev_pair','prev_pos')): raise SystemExit('prev_* recycle is forbidden in formal R79')
if not bool(rs.get('pair_static_within_round',False)): raise SystemExit('Pair must stay static inside each EGNN round')
if bool(rs.get('detach_between_rounds',True)): raise SystemExit('R79 keeps inherited differentiable physical recurrence; no recycle-style detach')
if rs.get('task_atom_observation_source')!='current_state_ca_fill': raise SystemExit('R79 requires current-state H3 atom-observation barrier; native xloss_mask is forbidden on the generated task domain')
ex=sp.get('execution',{})
if int(ex.get('triangle_chunk_size',64)) != 64: raise SystemExit('R79 v247 keeps training triangle chunk=64')
if int(ex.get('pair_distance_chunk_size',16)) != 16: raise SystemExit('R79 v247 requires exact training pair-distance chunk=16')
if int(ex.get('triangle_chunk_size_eval',96)) != 96: raise SystemExit('R79 v247 requires exact eval triangle chunk=96')
if int(ex.get('pair_distance_chunk_size_eval',32)) != 32: raise SystemExit('R79 v247 requires exact eval pair-distance chunk=32')
if float(cfg['loss']['interface']) != 1.0: raise SystemExit('interface weight must remain 1.0')
if float(cfg['loss']['distogram'].get('weight',0)) != 0 or float(cfg['loss']['smooth_lddt'].get('weight',0)) != 0: raise SystemExit('no auxiliary loss may be introduced')
gen=cfg['generation']
if gen.get('terminal_coordinate_authority')!='integrated_carrier_direct_h3' or bool(gen.get('terminal_kabsch_fusion',True)): raise SystemExit('R77 terminal carrier semantics required')
print(f'[ExperimentIdentity] PASS id={eid} parent={exp["parent"]} role={role}')
print('[R77BaseContract] native_latent_workspace=retained Pair=direct_shared controller=egnn_prenorm_raw explicit_time=1 rounds=3')
print('[SingleFieldContract] authority=carrier_primary_analytic physical_dof=1 structure=paratope_only writeback=paratope_only terminal=integrated_carrier')
print('[RoundStateSinglePairContract] refresh=once_per_macro_round round0=outer_Xt later=previous_analytic_endpoint design_geometry=current_model_state geometry_scope=paratope_only other_cmask_geometry=blocked pair_static_within_round=1 prev_seq=0 prev_pair=0 prev_pos=0 detach=0')
print('[LeakageBarrier] task=H3 native_task_xloss_mask=BLOCKED task_atom_observation=current_state_ca_fill fixed_context_observation=ALLOWED')
print('[ExecutionContract] train_pairdist_chunk=16 train_triangle_chunk=64 eval_pairdist_chunk=32 eval_triangle_chunk=96 equations=UNCHANGED train_val_test=UNCHANGED epoch_test=ON')
print('[ScientificDeltaContract] versus_R77=round_state_singlepair_only new_loss=0 new_sampler=0 new_controller=0 new_cartesian_field=0 prev_recycle=0')
print('[DiagnosticsContract] RoundStateValidation=on TestFieldTrajectory=compact TimeFieldValidation=off StateExposureAudit=off routine_stage_trace=off')
PY2

python "$PROJECT_ROOT/tests/validate_R79_static.py" "$PROJECT_ROOT"
grep -Fq 'current_state_observed = _abflow_ca_fill_observed_mask(' "$PROJECT_ROOT/models/AbFlow/abflow_components.py" || { echo 'R79 H3 observation barrier missing' >&2; exit 2; }
grep -Fq 'AuxTask[..., None], current_state_observed, Obsp' "$PROJECT_ROOT/models/AbFlow/abflow_components.py" || { echo 'R79 H3/native observation routing contract missing' >&2; exit 2; }
python -m py_compile "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" "$PROJECT_ROOT/models/AbFlow/abflow_components.py" "$PROJECT_ROOT/utils/nn_utils.py" "$PROJECT_ROOT/models/modules/am_enc.py" "$PROJECT_ROOT/models/modules/am_egnn.py" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" "$PROJECT_ROOT/train.py"
grep -Fq 'structure_supervision_mask = (' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'structure authority closure missing' >&2; exit 2; }
grep -Fq 'X[paratope_mask] = authority_endpoint_native' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'paratope-only writeback missing' >&2; exit 2; }
grep -Fq 'x = native_candidate' "$PROJECT_ROOT/models/modules/am_enc.py" || { echo 'R77 native latent workspace unexpectedly removed' >&2; exit 2; }
grep -Fq 'gen_X[paratope_mask] = interface_X_final' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'integrated carrier terminal missing' >&2; exit 2; }
grep -Fq 'return 0.5 * (cos(step / self.max_step * pi) + 1) * 0.9' "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo 'context_ratio changed unexpectedly' >&2; exit 2; }
echo '[Preflight] py_compile=PASS r77_base=PASS round_state_singlepair=PASS h3_native_observation_blocked=PASS train_pair_chunk16=PASS eval_chunks_exact=PASS train_val_test=UNCHANGED epoch_test=ON prev_recycle=OFF single_field=PASS final_round_only=PASS h3_authority_closed=PASS sampler_unchanged=PASS logs_compact=PASS'

python - "$PROJECT_ROOT" <<'PY2'
import hashlib,os,sys
root=sys.argv[1]
for rel in ['models/AbFlow/AbFlow_model.py','models/AbFlow/abflow_components.py','utils/nn_utils.py','models/modules/am_enc.py','models/modules/am_egnn.py','trainer/AbFlow_trainer.py','train.py','scripts/train/run_R79_round_state_singlepair_v247.sh']:
 p=os.path.join(root,rel)
 if os.path.isfile(p): print(f'[SourceSHA256] {rel} {hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]}')
PY2

set +e
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$PROJECT_ROOT/train.py" --config "$CONFIG_PATH"
STATUS=$?
set -e
echo "[RunComplete] experiment=$EXP_ID version=$VERSION exit_code=$STATUS configured_max_epoch=$MAX_EPOCH"
exit "$STATUS"
