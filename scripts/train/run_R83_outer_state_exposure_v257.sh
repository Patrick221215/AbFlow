#!/usr/bin/env bash
set -euo pipefail

# R83 outer-state exposure launcher: exact R82 parent plus one detached training-state delta.
# Scientific delta versus exact R82 is detached one-step OUTER coordinate-state exposure only.
# R82 common-frame/Angstrom Pair geometry, anti-leak barrier, torque tangent,
# single physical Cartesian field, losses, sequence rollout and formal F01 Test
# sampler remain unchanged.  The main carrier target is recomputed from the
# actually exposed state to preserve state/target consistency.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CONFIG_PATH=${1:-}
shift || true
GPU_CSV=""; MASTER_PORT=""; MASTER_ADDR=""

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/train/run_R83_outer_state_exposure_v257.sh <R83-config.json> \
    --gpus 2,3,4,5,6,7 --port 29784

R83 v257 contract:
  - first initialization = scratch from exact R82 architecture; same-experiment R83 resume is supported after interruption;
  - R82 torque/common-frame/anti-leak/3-round recurrence are retained exactly;
  - from epoch 0, one fixed sparse detached prepass exposes the main forward to a one-step outer coordinate state at the same t;
  - target is recomputed from that exposed state; sequence state remains analytic at the main t;
  - no new loss, sampler, head, recycle path or Cartesian authority;
  - Train -> Validation -> Test remains unchanged (10 steps, seed 2023).
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
 'RESUME_CKPT':tr['schedule'].get('resume_checkpoint','') or '',
}
for k,v in vals.items(): print(f'{k}='+shlex.quote(str(v)))
PY2
)"

MASTER_ADDR=${MASTER_ADDR:-$CFG_MASTER_ADDR}
[[ -n "$MASTER_PORT" && "$MASTER_PORT" =~ ^[0-9]+$ ]] || { echo "--port is required" >&2; exit 2; }
[[ "$TEST_STEPS" == "10" ]] || { echo "formal observational Test requires n_steps=10" >&2; exit 2; }
EFFECTIVE_GLOBAL_TRAIN_BATCH=$((PER_GPU_BATCH_SIZE * NPROC_PER_NODE))

mkdir -p "$OUTPUT_ROOT" || { echo "Failed to create OUTPUT_ROOT: $OUTPUT_ROOT" >&2; exit 2; }
RUN_MODE="scratch"
if [[ -n "${RESUME_CKPT:-}" ]]; then
  if [[ "$RESUME_CKPT" != /* ]]; then RESUME_CKPT="$PROJECT_ROOT/${RESUME_CKPT#./}"; fi
  [[ -f "$RESUME_CKPT" ]] || { echo "Resume checkpoint not found: $RESUME_CKPT" >&2; exit 2; }
  CKPT_DIR=$(dirname "$RESUME_CKPT")
  RUN_DIR=$(dirname "$CKPT_DIR")
  VERSION_NAME=$(basename "$RUN_DIR")
  SOURCE_ROOT=$(dirname "$RUN_DIR")
  [[ "$VERSION_NAME" =~ ^version_([0-9]+)$ ]] || { echo "Resume checkpoint must live under version_N/checkpoint/: $RESUME_CKPT" >&2; exit 2; }
  VERSION=${BASH_REMATCH[1]}
  [[ "$(readlink -f "$SOURCE_ROOT")" == "$(readlink -f "$OUTPUT_ROOT")" ]] || {
    echo "R83 formal resume must continue the same R83 experiment; cross-experiment loading is warm-start, not resume." >&2
    exit 2
  }
  RUN_MODE="resume"
  export ABFLOW_FIXED_VERSION="$VERSION"
  export ABFLOW_EXPECTED_RUN_DIR="$RUN_DIR"
  export ABFLOW_REQUIRE_SCRATCH=0
  export ABFLOW_RESUME_CHECKPOINT="$RESUME_CKPT"
else
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
fi
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
export ABFLOW_STATE_EXPOSURE_EPOCHS="${ABFLOW_STATE_EXPOSURE_EPOCHS:-0,25,50,100,150,199}"
export ABFLOW_STATE_EXPOSURE_STEPS="${ABFLOW_STATE_EXPOSURE_STEPS:-0,5,9}"
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
if [[ "$RUN_MODE" == "resume" ]]; then
  echo "[RunInit] mode=resume experiment=R83 checkpoint=$RESUME_CKPT version=$VERSION optimizer=resume ema=resume epoch=resume best_val=resume"
  echo "[ResumeContract] mode=resume checkpoint=$RESUME_CKPT"
else
  echo "[RunInit] mode=scratch parent=R82_exact optimizer=reset ema=reset epoch=0 best_val=reset"
fi
echo "[TrainingHorizon] source=json max_epoch=$MAX_EPOCH launcher_epoch_override=none"
echo "[TrainValTestContract] order=train->validation->test checkpoint_selection=validation test_metrics=observation_only test_steps=$TEST_STEPS test_seed=$TEST_SEED"

python - "$CONFIG_PATH" "$OUTPUT_ROOT" <<'PY2'
import json, os, sys
cp,out=sys.argv[1:3]; cfg=json.load(open(cp,encoding='utf-8')); exp=cfg['experiment']
eid=exp['id']; stem=os.path.splitext(os.path.basename(cp))[0]; role=str(exp.get('diagnostic_role','')).lower()
if not (eid==stem==os.path.basename(os.path.normpath(out))): raise SystemExit(f'identity mismatch: {eid} / {stem} / {out}')
if exp.get('protocol')!='formal_train_val_test': raise SystemExit('formal_train_val_test required')
if role!='r83_full_outer_coordinate_exposure_from_epoch0': raise SystemExit(f'R83 full-exposure role required, got {role!r}')
if exp.get('parent')!='R82_R05_ABX_R81_TORQUE_TANGENT_SINGLEPAIR_ANALYTIC3R_TVT_U02': raise SystemExit('R83 must use exact R82 as parent identity')
if str(exp.get('initialization','')).lower()!='scratch': raise SystemExit('R83 must start from scratch')
if int(cfg['model']['architecture']['iter_round']) != 3: raise SystemExit('R83 requires exactly 3 refinement rounds')
sp=cfg['model']['representation']['single_pair']; pc=sp['pair_coordinate']; cc=sp['coordinate_controller']; pa=sp['physical_authority']; rs=sp.get('round_state_conditioning',{})
if pc.get('mode')!='direct_shared' or cc.get('mode')!='egnn_prenorm_raw': raise SystemExit('inherited Pair/controller contract changed')
if not bool(sp.get('time_embed',True)): raise SystemExit('explicit Single/Pair time must remain on')
if pa.get('mode')!='carrier_primary_analytic' or int(pa.get('physical_dof',0))!=1: raise SystemExit('carrier-primary single field required')
if pa.get('geometric_operator_mode')!='latent_native_workspace': raise SystemExit('R77 latent native workspace must be retained')
if pa.get('refinement_supervision')!='final_round_only_inherited_from_AbFlow': raise SystemExit('final-round-only supervision must remain unchanged')
if pa.get('structure_supervision_mask')!='paratope_only' or pa.get('fixed_context_writeback')!='paratope_only': raise SystemExit('H3 authority closure required')
if not bool(rs.get('enabled',False)) or rs.get('geometry_source')!='current_authoritative_state': raise SystemExit('round-state Single/Pair factor missing')
if rs.get('round0_state')!='outer_Xt' or rs.get('later_round_state')!='previous_analytic_endpoint': raise SystemExit('R83 retains outer_Xt -> previous_analytic_endpoint recurrence')
if rs.get('relational_coordinate_frame')!='common_raw_complex': raise SystemExit('R83 retains common_raw_complex relational frame')
if rs.get('relational_coordinate_units')!='angstrom': raise SystemExit('R83 retains raw Angstrom relational geometry')
if any(bool(rs.get(k,False)) for k in ('prev_seq','prev_pair','prev_pos')): raise SystemExit('prev_* recycle is forbidden')
if not bool(rs.get('pair_static_within_round',False)): raise SystemExit('Pair must stay static inside each EGNN round')
if bool(rs.get('detach_between_rounds',True)): raise SystemExit('R83 keeps differentiable physical recurrence')
if rs.get('task_atom_observation_source')!='current_state_ca_fill' or not bool(rs.get('native_task_observation_mask_forbidden',False)): raise SystemExit('clean H3 observation barrier required')
pt=sp.get('pose_torque_actuation',{})
if not bool(pt.get('enabled',False)) or pt.get('mode')!='pair_conditioned_centroid_rodrigues': raise SystemExit('R83 must retain exact R82 torque tangent')
if pt.get('pivot')!='h3_ca_centroid' or pt.get('apply_to')!='carrier_after_egnn': raise SystemExit('R82 torque semantics changed')
oe=sp.get('outer_state_exposure',{})
if not bool(oe.get('enabled',False)) or oe.get('mode')!='detached_one_step_f01_coordinate': raise SystemExit('R83 detached one-step coordinate exposure required')
if 'warmup_epochs' in oe: raise SystemExit('R83 formal protocol forbids outer_state_exposure.warmup_epochs')
for k in ('period','min_t','max_t'):
    if k in oe: raise SystemExit(f'R83 full-exposure protocol forbids outer_state_exposure.{k}')
if abs(float(oe.get('step_dt',-1))-0.1)>1e-12: raise SystemExit('R83 exposure step_dt changed')
if oe.get('scope')!='every_training_graph' or oe.get('start')!='epoch0': raise SystemExit('R83 full exposure must cover every training graph from epoch0')
if oe.get('prepass')!='eval_no_grad' or oe.get('main_target')!='recompute_from_exposed_state': raise SystemExit('R83 detached/target consistency contract changed')
if oe.get('sequence_state')!='analytic_at_main_time': raise SystemExit('R83 must remain coordinate-only exposure')
ex=sp.get('execution',{})
if int(ex.get('triangle_chunk_size',64)) != 64 or int(ex.get('pair_distance_chunk_size',8)) != 8: raise SystemExit('R83 keeps matched train chunks 64/8')
if int(ex.get('triangle_chunk_size_eval',64)) != 64 or int(ex.get('pair_distance_chunk_size_eval',8)) != 8: raise SystemExit('R83 keeps matched eval chunks 64/8')
if not bool(ex.get('static_layout_cache',False)): raise SystemExit('R83 retains execution-only static layout reuse')
if float(cfg['loss']['interface']) != 1.0: raise SystemExit('interface weight must remain 1.0')
if float(cfg['loss']['distogram'].get('weight',0)) != 0 or float(cfg['loss']['smooth_lddt'].get('weight',0)) != 0: raise SystemExit('no auxiliary loss may be introduced')
gen=cfg['generation']
if gen.get('terminal_coordinate_authority')!='integrated_carrier_direct_h3' or bool(gen.get('terminal_kabsch_fusion',True)): raise SystemExit('integrated carrier terminal semantics required')
if int(gen.get('n_steps',0))!=10 or int(gen.get('seed',-1))!=2023: raise SystemExit('formal Test protocol changed')
print(f'[ExperimentIdentity] PASS id={eid} parent={exp["parent"]} role={role}')
print('[SingleFieldContract] authority=carrier_primary_analytic physical_dof=1 structure=paratope_only writeback=paratope_only terminal=integrated_carrier')
print('[RoundStateSinglePairContract] refresh=once_per_macro_round round0=outer_Xt later=previous_analytic_endpoint relational_frame=common_raw_complex pair_static_within_round=1 prev_seq=0 prev_pair=0 prev_pos=0 detach=0')
print('[PoseActuationContract] inherited_R82=pair_conditioned_centroid_rodrigues pivot=h3_ca_centroid exact_rigid=1')
print('[OuterStateExposureContract] mode=detached_one_step_f01_coordinate main_time=same_t start=epoch0 scope=every_training_graph dt=0.1 prepass=eval_no_grad target=recomputed sequence=analytic_main_time')
print('[LeakageBarrier] task=H3 native_task_xloss_mask=BLOCKED task_atom_observation=current_state_ca_fill fixed_context_observation=ALLOWED')
print('[ExecutionContract] static_layout_once_per_outer=1 SinglePair_refreshes=3 train_chunks=64/8 eval_chunks=64/8 validation_bridge_capture=off equations=UNCHANGED train_val_test=UNCHANGED epoch_test=ON')
print('[ScientificDeltaContract] versus_exact_R82=full_detached_one_step_outer_coordinate_state_exposure_from_epoch0_only new_loss=0 new_sampler=0 new_head=0 new_cartesian_field=0 sequence_rollout=0 mixture=0 warmup=0')
print('[DiagnosticContract] inner_refinement=compact outer_exposure_target_identity=exact_hybrid_delta matched_rollout_oracle=epochs_0_25_50_100_150_199')
print('[RelationalGeometryContract] frame=common_raw_complex units=angstrom pair_distance_divisor_A=10 local_coordinate_scale_A=0.1')
print('[DiagnosticsContract] Validation=lr+seq+AAR+struct+interface+edge InnerRefinement=compact OuterStateExposure=state+target+identity TestFieldTrajectory=compact TestStateExposure=sparse_epoch_gated')
PY2

python - "$PROJECT_ROOT" <<'PY2'
import hashlib, os, sys
root=sys.argv[1]
expected={
 'models/AbFlow/abflow_components.py':'3621235a1670d156',
 'utils/nn_utils.py':'17facac7fc7cc85b',
 'models/modules/am_enc.py':'0bed7e9c826490e0',
 'models/modules/am_egnn.py':'134e23b1c5e68bf4',
 'train.py':'b7bf85b7779af7fc',
}
for rel,prefix in expected.items():
 p=os.path.join(root,rel)
 if not os.path.isfile(p): raise SystemExit(f'R82 parent dependency missing: {rel}')
 got=hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
 if got!=prefix:
  raise SystemExit(f'R82 parent dependency SHA mismatch: {rel} expected={prefix} got={got}. Do not overwrite the running R82 parent with a stale attachment.')
print('[ParentDependencySHA] PASS ' + ' '.join(f'{k}={v}' for k,v in expected.items()))
PY2
grep -Fq 'Fixed = M & (~Design)' "$PROJECT_ROOT/models/AbFlow/abflow_components.py" || { echo 'H3 donor information barrier missing' >&2; exit 2; }
grep -Fq 'def prepare_layout(' "$PROJECT_ROOT/models/AbFlow/abflow_components.py" || { echo 'static NativeTrunk layout optimization missing' >&2; exit 2; }
grep -Fq 'def forward_prepared(' "$PROJECT_ROOT/models/AbFlow/abflow_components.py" || { echo 'prepared NativeTrunk forward missing' >&2; exit 2; }
grep -Fq 'def _relational_common_raw_coordinates(' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'common raw relational frame missing' >&2; exit 2; }
grep -Fq 'def _apply_pair_torque_actuation(' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'R82 torque tangent missing' >&2; exit 2; }
grep -Fq 'def _same_noise_analytic_state_at_previous_time(' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'R83 outer exposure helper missing' >&2; exit 2; }
grep -Fq 'recompute_from_exposed_state' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'R83 target consistency guard missing' >&2; exit 2; }
grep -Fq 'outer_exposure_target_identity_err_A' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'R83 target identity diagnostic missing' >&2; exit 2; }
grep -Fq 'def _outer_state_exposure_active' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'R83 full exposure activation missing' >&2; exit 2; }
if grep -Eq 'outer_state_exposure_period|_outer_state_exposure_scheduled|outer_state_exposure_min_t|outer_state_exposure_max_t' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"; then
  echo 'Rejected R83 mixture/window scheduling code is still present' >&2
  exit 2
fi
grep -Fq 'current_relational_h3_native = authority_endpoint_native' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'analytic-endpoint round recurrence missing' >&2; exit 2; }
grep -Fq "'[InnerRefinement] '" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo 'R83 compact inner-refinement diagnostics missing' >&2; exit 2; }
grep -Fq "'[OuterStateExposure] '" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo 'R83 outer exposure diagnostics missing' >&2; exit 2; }
python -m py_compile "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" "$PROJECT_ROOT/models/AbFlow/abflow_components.py" "$PROJECT_ROOT/utils/nn_utils.py" "$PROJECT_ROOT/models/modules/am_enc.py" "$PROJECT_ROOT/models/modules/am_egnn.py" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" "$PROJECT_ROOT/train.py"
grep -Fq 'X[paratope_mask] = authority_endpoint_native' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'paratope-only endpoint supervision/writeback missing' >&2; exit 2; }
grep -Fq 'gen_X[paratope_mask] = interface_X_final' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'integrated carrier terminal missing' >&2; exit 2; }
grep -Fq 'return 0.5 * (cos(step / self.max_step * pi) + 1) * 0.9' "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo 'context_ratio changed unexpectedly' >&2; exit 2; }
echo '[Preflight] py_compile=PASS resume_supported=PASS parent_R82_contract=PASS common_pair_frame=PASS h3_native_observation_blocked=PASS torque_tangent=PASS outer_state_exposure_full_epoch0=PASS mixture=ABSENT time_window=ABSENT target_recompute=PASS target_identity_diag=PASS train_val_test=UNCHANGED sampler=UNCHANGED single_field=PASS'

python - "$PROJECT_ROOT" <<'PY2'
import hashlib,os,sys
root=sys.argv[1]
for rel in ['models/AbFlow/AbFlow_model.py','models/AbFlow/abflow_components.py','utils/nn_utils.py','models/modules/am_enc.py','models/modules/am_egnn.py','trainer/AbFlow_trainer.py','train.py','scripts/train/run_R83_outer_state_exposure_v257.sh']:
 p=os.path.join(root,rel)
 if os.path.isfile(p): print(f'[SourceSHA256] {rel} {hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]}')
PY2

set +e
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$PROJECT_ROOT/train.py" --config "$CONFIG_PATH"
STATUS=$?
set -e
echo "[RunComplete] experiment=$EXP_ID version=$VERSION exit_code=$STATUS configured_max_epoch=$MAX_EPOCH"
exit "$STATUS"
