#!/usr/bin/env bash
set -euo pipefail

# R82: clean R81 relational geometry + one torque-tangent actuator.
# Scientific delta: Pair-conditioned exact Rodrigues rotation about the current
# H3 CA centroid, applied to the existing carrier after the mature R05 EGNN.
# No new loss, sampler, coordinate field, prev_* recycle, clipping, tanh or
# manual physical step scale. Train -> Validation -> Test remains unchanged.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CONFIG_PATH=${1:-}
shift || true
GPU_CSV=""; MASTER_PORT=""; MASTER_ADDR=""

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/train/run_R82_torque_tangent_singlepair_v253.sh <R82-config.json> \
    --gpus 2,3,4,5,6,7 --port 29786

R82 contract:
  - origin = clean R81 scratch run; strict same-version resume is supported;
  - round0 relational state = outer Xt; rounds1/2 = previous analytic endpoint;
  - dense Single/Pair geometry uses one common raw-complex Angstrom frame;
  - no prev_seq/prev_pair/prev_pos, no new loss/sampler/controller/Cartesian field;
  - H3 native xloss_mask remains blocked; Pair stays static inside each round;
  - sole scientific delta = Pair-conditioned centroid-preserving Rodrigues torque tangent;
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
resume=str(tr.get('schedule',{}).get('resume_checkpoint','') or '').strip()
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
 'RESUME_CHECKPOINT':abspath(resume) if resume else '',
}
for k,v in vals.items(): print(f'{k}='+shlex.quote(str(v)))
PY2
)"

MASTER_ADDR=${MASTER_ADDR:-$CFG_MASTER_ADDR}
[[ -n "$MASTER_PORT" && "$MASTER_PORT" =~ ^[0-9]+$ ]] || { echo "--port is required" >&2; exit 2; }
[[ "$TEST_STEPS" == "10" ]] || { echo "formal observational Test requires n_steps=10" >&2; exit 2; }
EFFECTIVE_GLOBAL_TRAIN_BATCH=$((PER_GPU_BATCH_SIZE * NPROC_PER_NODE))

mkdir -p "$OUTPUT_ROOT" || { echo "Failed to create OUTPUT_ROOT: $OUTPUT_ROOT" >&2; exit 2; }

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || {
    echo "resume_checkpoint not found: $RESUME_CHECKPOINT" >&2
    exit 2
  }

  CKPT_DIR=$(dirname "$RESUME_CHECKPOINT")
  [[ "$(basename "$CKPT_DIR")" == "checkpoint" ]] || {
    echo "resume_checkpoint must live under version_N/checkpoint/: $RESUME_CHECKPOINT" >&2
    exit 2
  }

  RUN_DIR=$(dirname "$CKPT_DIR")
  VERSION_BASE=$(basename "$RUN_DIR")
  [[ "$VERSION_BASE" =~ ^version_([0-9]+)$ ]] || {
    echo "resume_checkpoint must live under version_N/checkpoint/: $RESUME_CHECKPOINT" >&2
    exit 2
  }
  VERSION="${BASH_REMATCH[1]}"

  RESUME_ROOT=$(realpath "$(dirname "$RUN_DIR")")
  OUTPUT_ROOT_REAL=$(realpath "$OUTPUT_ROOT")
  [[ "$RESUME_ROOT" == "$OUTPUT_ROOT_REAL" ]] || {
    echo "resume_checkpoint does not belong to this R82 experiment:" >&2
    echo "  checkpoint root: $RESUME_ROOT" >&2
    echo "  config output:   $OUTPUT_ROOT_REAL" >&2
    exit 2
  }

  export ABFLOW_FIXED_VERSION="$VERSION"
  export ABFLOW_EXPECTED_RUN_DIR="$RUN_DIR"
  export ABFLOW_REQUIRE_SCRATCH=0
  export ABFLOW_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
  RUN_MODE="resume"
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
  RUN_MODE="scratch"
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
export ABFLOW_STATE_EXPOSURE_EPOCHS="${ABFLOW_STATE_EXPOSURE_EPOCHS:-77,80,90,100,125,150,175,199}"
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
  echo "[RunInit] mode=resume experiment=R82 checkpoint=$RESUME_CHECKPOINT version=$VERSION optimizer=resume ema=resume epoch=resume best_val=resume"
else
  echo "[RunInit] mode=scratch parent=R81_clean optimizer=reset ema=reset epoch=0 best_val=reset"
fi
echo "[TrainingHorizon] source=json max_epoch=$MAX_EPOCH launcher_epoch_override=none"
echo "[TrainValTestContract] order=train->validation->test checkpoint_selection=validation test_metrics=observation_only test_steps=$TEST_STEPS test_seed=$TEST_SEED"
echo "[StateExposureAuditContract] epochs=$ABFLOW_STATE_EXPOSURE_EPOCHS steps=$ABFLOW_STATE_EXPOSURE_STEPS isolate=coordinate_state sequence_context=same_rollout_St observer_only=1 training_effect=0 rng_restored=1 optimizer_untouched=1 sampler_unchanged=1"

python - "$CONFIG_PATH" "$OUTPUT_ROOT" <<'PY2'
import json, os, sys
cp,out=sys.argv[1:3]; cfg=json.load(open(cp,encoding='utf-8')); exp=cfg['experiment']
eid=exp['id']; stem=os.path.splitext(os.path.basename(cp))[0]; role=str(exp.get('diagnostic_role','')).lower()
if not (eid==stem==os.path.basename(os.path.normpath(out))): raise SystemExit(f'identity mismatch: {eid} / {stem} / {out}')
if exp.get('protocol')!='formal_train_val_test': raise SystemExit('formal_train_val_test required')
if role!='r82_pair_torque_tangent_actuation_only': raise SystemExit(f'R82 role required, got {role!r}')
if exp.get('parent')!='R81_R05_ABX_R79_COMMON_FRAME_SINGLEPAIR_ANALYTIC3R_TVT_U02': raise SystemExit('R82 parent must be clean R81')
if str(exp.get('initialization','')).lower()!='scratch': raise SystemExit('R82 must start from scratch')
resume=str(cfg['training']['schedule'].get('resume_checkpoint','') or '').strip()
print(f'[ResumeContract] mode={"resume" if resume else "scratch"} checkpoint={resume if resume else "none"}')
sp=cfg['model']['representation']['single_pair']; pc=sp['pair_coordinate']; cc=sp['coordinate_controller']; pa=sp['physical_authority']; rs=sp['round_state_conditioning']; tq=sp.get('pose_torque_actuation',{})
if int(cfg['model']['architecture']['iter_round']) != 3: raise SystemExit('R82 requires exactly 3 refinement rounds')
if pc.get('mode')!='direct_shared' or cc.get('mode')!='egnn_prenorm_raw': raise SystemExit('R81 Pair/base EGNN controller must remain unchanged')
if pa.get('mode')!='carrier_primary_analytic' or int(pa.get('physical_dof',0))!=1: raise SystemExit('carrier-primary single field required')
if pa.get('geometric_operator_mode')!='latent_native_workspace': raise SystemExit('R81 latent native workspace must remain unchanged')
if pa.get('refinement_supervision')!='final_round_only_inherited_from_AbFlow': raise SystemExit('final-round-only supervision required')
if rs.get('round0_state')!='outer_Xt' or rs.get('later_round_state')!='previous_analytic_endpoint': raise SystemExit('clean R81 recurrence required')
if rs.get('relational_coordinate_frame')!='common_raw_complex' or rs.get('relational_coordinate_units')!='angstrom': raise SystemExit('R81 common raw Angstrom geometry required')
if any(bool(rs.get(k,False)) for k in ('prev_seq','prev_pair','prev_pos')): raise SystemExit('prev_* recycle forbidden')
if rs.get('task_atom_observation_source')!='current_state_ca_fill' or not bool(rs.get('native_task_observation_mask_forbidden',False)): raise SystemExit('clean H3 observation barrier required')
if not bool(tq.get('enabled',False)) or tq.get('mode')!='pair_conditioned_centroid_rodrigues': raise SystemExit('R82 torque tangent missing')
if tq.get('pivot')!='h3_ca_centroid' or tq.get('apply_to')!='carrier_after_egnn': raise SystemExit('R82 torque action semantics changed')
if not bool(tq.get('zero_init',False)): raise SystemExit('R82 requires zero-init matched parent boundary')
if bool(tq.get('manual_scale',False)) or bool(tq.get('coordinate_clip',False)) or bool(tq.get('coordinate_tanh',False)): raise SystemExit('R82 forbids manual scale/clip/tanh')
if any(float(cfg['loss'][k]) != 1.0 for k in ('sequence','structure','interface','edge')): raise SystemExit('base losses changed')
if float(cfg['loss']['distogram'].get('weight',0)) != 0 or float(cfg['loss']['smooth_lddt'].get('weight',0)) != 0: raise SystemExit('auxiliary loss introduced')
gen=cfg['generation']
if int(gen.get('n_steps',0))!=10 or int(gen.get('seed',-1))!=2023: raise SystemExit('formal Test protocol changed')
if gen.get('terminal_coordinate_authority')!='integrated_carrier_direct_h3' or bool(gen.get('terminal_kabsch_fusion',True)): raise SystemExit('carrier terminal semantics changed')
print(f'[ExperimentIdentity] PASS id={eid} parent={exp["parent"]} role={role}')
print('[SingleFieldContract] authority=carrier_primary_analytic physical_dof=1 terminal=integrated_carrier')
print('[RelationalGeometryContract] frame=common_raw_complex units=angstrom anti_leak=on rounds=3 later=previous_analytic_endpoint')
print('[PoseActuationContract] base=R05_pair_radial torque=pair_conditioned centroid=h3_ca exact_rodrigues=1 translation_preserved=1 rigid_internal_geometry_preserved=1 zero_init=1')
print('[ScientificDeltaContract] versus_R81=pair_conditioned_centroid_preserving_torque_tangent_only new_loss=0 new_sampler=0 new_coordinate_field=0 prev_recycle=0')
print('[DiagnosticsContract] RoundTransportValidation=compact TestFieldTrajectory=compact TestSequenceTrajectory=compact StateExposureAudit=epoch_gated_same_sequence_context')
PY2
grep -Fq 'current_state_observed = _abflow_ca_fill_observed_mask(' "$PROJECT_ROOT/models/AbFlow/abflow_components.py" || { echo 'H3 observation barrier missing' >&2; exit 2; }
grep -Fq 'def prepare_layout(' "$PROJECT_ROOT/models/AbFlow/abflow_components.py" || { echo 'static NativeTrunk layout optimization missing' >&2; exit 2; }
grep -Fq 'def forward_prepared(' "$PROJECT_ROOT/models/AbFlow/abflow_components.py" || { echo 'prepared NativeTrunk forward missing' >&2; exit 2; }
grep -Fq 'def _relational_common_raw_coordinates(' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'R81 common raw relational frame missing' >&2; exit 2; }
grep -Fq 'def _apply_pair_torque_actuation(' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'R82 torque tangent missing' >&2; exit 2; }
grep -Fq 'pair_conditioned_centroid_rodrigues' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'R82 torque mode missing' >&2; exit 2; }
grep -Fq 'current_relational_h3_native = authority_endpoint_native' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'clean R79 endpoint recurrence missing' >&2; exit 2; }
grep -Fq "'[RoundTransportValidation] '" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo 'transport diagnostics missing' >&2; exit 2; }
grep -Fq "'[TestSequenceTrajectory] '" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo 'sequence trajectory diagnostics missing' >&2; exit 2; }
grep -Fq "'oracle_gain_seq_nll'" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'P0/P1 coupling diagnostics missing' >&2; exit 2; }
python -m py_compile "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" "$PROJECT_ROOT/models/AbFlow/abflow_components.py" "$PROJECT_ROOT/utils/nn_utils.py" "$PROJECT_ROOT/models/modules/am_enc.py" "$PROJECT_ROOT/models/modules/am_egnn.py" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" "$PROJECT_ROOT/train.py"
grep -Fq 'X[paratope_mask] = authority_endpoint_native' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'paratope-only endpoint supervision/writeback missing' >&2; exit 2; }
grep -Fq 'gen_X[paratope_mask] = interface_X_final' "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo 'integrated carrier terminal missing' >&2; exit 2; }
grep -Fq 'return 0.5 * (cos(step / self.max_step * pi) + 1) * 0.9' "$PROJECT_ROOT/trainer/AbFlow_trainer.py" || { echo 'context_ratio changed unexpectedly' >&2; exit 2; }
echo "[Preflight] py_compile=PASS resume_supported=PASS run_mode=$RUN_MODE p0_p1_diagnostics=PASS common_pair_frame=PASS h3_native_observation_blocked=PASS torque_tangent=PASS zero_init=PASS train_val_test=UNCHANGED epoch_test=ON single_field=PASS sampler_unchanged=PASS"

python - "$PROJECT_ROOT" <<'PY2'
import hashlib,os,sys
root=sys.argv[1]
for rel in ['models/AbFlow/AbFlow_model.py','models/AbFlow/abflow_components.py','utils/nn_utils.py','models/modules/am_enc.py','models/modules/am_egnn.py','trainer/AbFlow_trainer.py','train.py','scripts/train/run_R82_torque_tangent_singlepair_v253.sh']:
 p=os.path.join(root,rel)
 if os.path.isfile(p): print(f'[SourceSHA256] {rel} {hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]}')
PY2

set +e
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$PROJECT_ROOT/train.py" --config "$CONFIG_PATH"
STATUS=$?
set -e
echo "[RunComplete] experiment=$EXP_ID version=$VERSION exit_code=$STATUS configured_max_epoch=$MAX_EPOCH"
exit "$STATUS"
