#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CONFIG_PATH=${1:-}
shift || true
GPU_CSV=""; MASTER_PORT=""; MASTER_ADDR=""; RESUME_CHECKPOINT=""

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/train/run_R67_R68_v225.sh <R67-causal.json> --gpus 0,1,2,3 --port 29767

Accepted formal R67 causal roles:
  r67_direct_shared_prenorm_raw
  r67_direct_shared_prenorm_unit
GPU ids are execution resources and must stay on the CLI, never in scientific JSON.
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
[[ -n "$GPU_CSV" ]] || { echo "--gpus is required" >&2; exit 2; }
[[ "$GPU_CSV" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid --gpus '$GPU_CSV'" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
NPROC=${#GPU_IDS[@]}
[[ $(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l) -eq $NPROC ]] || { echo "duplicate GPU id" >&2; exit 2; }
if [[ "$CONFIG_PATH" != /* ]]; then CONFIG_PATH="$PROJECT_ROOT/${CONFIG_PATH#./}"; fi
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 2; }

eval "$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import json, os, shlex, sys
cfg=json.load(open(sys.argv[1],encoding='utf-8')); root=sys.argv[2]
if 'gpus' in cfg.get('runtime',{}): raise SystemExit('runtime.gpus is forbidden; use --gpus')
tr=cfg['training']; out=tr['output_dir']
if not os.path.isabs(out): out=os.path.abspath(os.path.join(root,out))
resume=tr.get('schedule',{}).get('resume_checkpoint','') or ''
if resume and not os.path.isabs(resume): resume=os.path.abspath(os.path.join(root,resume))
vals={'OUTPUT_ROOT':out,'GLOBAL_BATCH_SIZE':tr['loader']['batch_size'],'CFG_RESUME_CHECKPOINT':resume,
      'CFG_MASTER_ADDR':cfg.get('runtime',{}).get('master_addr','127.0.0.1'),
      'CFG_MASTER_PORT':cfg.get('runtime',{}).get('master_port',''),'NNODES':cfg.get('runtime',{}).get('nnodes',1),
      'OMP_THREADS':cfg.get('runtime',{}).get('omp_num_threads',2),'CUDA_ALLOC':cfg.get('runtime',{}).get('cuda_allocator','max_split_size_mb:128')}
for k,v in vals.items(): print(f'{k}='+shlex.quote(str(v)))
PY
)"
MASTER_ADDR=${MASTER_ADDR:-$CFG_MASTER_ADDR}; MASTER_PORT=${MASTER_PORT:-$CFG_MASTER_PORT}
[[ -n "$MASTER_PORT" && "$MASTER_PORT" =~ ^[0-9]+$ ]] || { echo "--port is required" >&2; exit 2; }
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-$CFG_RESUME_CHECKPOINT}
(( GLOBAL_BATCH_SIZE % NPROC == 0 )) || { echo "global batch $GLOBAL_BATCH_SIZE not divisible by $NPROC" >&2; exit 2; }
LOCAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE/NPROC))
cd "$PROJECT_ROOT"; mkdir -p "$OUTPUT_ROOT"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "resume checkpoint not found" >&2; exit 2; }
  RUN_DIR=$(dirname "$(dirname "$RESUME_CHECKPOINT")"); VERSION_BASE=$(basename "$RUN_DIR")
  [[ "$VERSION_BASE" =~ ^version_([0-9]+)$ ]] || { echo "resume must live under version_N/checkpoint" >&2; exit 2; }
  VERSION=${BASH_REMATCH[1]}; [[ "$(realpath "$(dirname "$RUN_DIR")")" == "$(realpath "$OUTPUT_ROOT")" ]] || { echo "cross-experiment resume forbidden" >&2; exit 2; }
  unset ABFLOW_FIXED_VERSION || true
else
  VERSION=0; while ! mkdir "$OUTPUT_ROOT/version_$VERSION" 2>/dev/null; do VERSION=$((VERSION+1)); done
  RUN_DIR="$OUTPUT_ROOT/version_$VERSION"; export ABFLOW_FIXED_VERSION="$VERSION"
fi
RUN_LOG="$RUN_DIR/run_time.log"; LATEST_LOG="$OUTPUT_ROOT/run_time.log"; mkdir -p "$RUN_DIR"; ln -sfn "version_$VERSION/run_time.log" "$LATEST_LOG"
export ABFLOW_PROJECT_ROOT="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES="$GPU_CSV" ABFLOW_NPROC_PER_NODE="$NPROC" ABFLOW_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
export OMP_NUM_THREADS="$OMP_THREADS" PYTHONUNBUFFERED=1; [[ -n "$CUDA_ALLOC" ]] && export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC"

# Observation-only diagnostics. They never alter optimizer, loss, sampler or checkpoint selection.
export ABFLOW_GEOMETRY_FORENSICS="${ABFLOW_GEOMETRY_FORENSICS:-on}"
export ABFLOW_SAMPLE_FORENSICS="${ABFLOW_SAMPLE_FORENSICS:-on}"
export ABFLOW_SAMPLE_STEP_RMS_ALERT_A="${ABFLOW_SAMPLE_STEP_RMS_ALERT_A:-10}"
export ABFLOW_COORD_AUDIT_INTERVAL="${ABFLOW_COORD_AUDIT_INTERVAL:-20}"
export ABFLOW_COORD_AUDIT_FIRST_STEPS="${ABFLOW_COORD_AUDIT_FIRST_STEPS:-2}"
export ABFLOW_GEOMETRY_AUTHORITY_INTERVAL=0
export ABFLOW_GEOMETRY_AUTHORITY_RMS_ALERT="${ABFLOW_GEOMETRY_AUTHORITY_RMS_ALERT:-5}"
export ABFLOW_GEOMETRY_AUTHORITY_ALERT_MAX_PER_EPOCH="${ABFLOW_GEOMETRY_AUTHORITY_ALERT_MAX_PER_EPOCH:-1}"
export ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD="${ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD:-10000}"
export ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH="${ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH:-1}"
export ABFLOW_GRAD_OVERFLOW_LOG_LIMIT="${ABFLOW_GRAD_OVERFLOW_LOG_LIMIT:-3}"
export ABFLOW_TQDM="${ABFLOW_TQDM:-on}" ABFLOW_SCI_LOG_FIRST_STEPS="${ABFLOW_SCI_LOG_FIRST_STEPS:-1}" ABFLOW_SCI_LOG_INTERVAL="${ABFLOW_SCI_LOG_INTERVAL:-40}" ABFLOW_RUNTIME_GUARD_STEPS="${ABFLOW_RUNTIME_GUARD_STEPS:-1}"
export ABFLOW_EPOCH_TEST_FAIL_FAST="on" ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY="record_and_continue" ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS="${ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS:-on}"

exec > >(stdbuf -oL -eL tee -a "$RUN_LOG") 2>&1
echo "[RunLog] canonical=$RUN_LOG latest=$LATEST_LOG"
echo "[RunVersion] fixed_version=$VERSION dir=$RUN_DIR"
echo "[RunConfig] config=$CONFIG_PATH"
echo "[RunResources] physical_gpus=$GPU_CSV nproc=$NPROC global_batch=$GLOBAL_BATCH_SIZE local_batch=$LOCAL_BATCH_SIZE master_addr=$MASTER_ADDR port=$MASTER_PORT nnodes=$NNODES omp=$OMP_THREADS"
echo "[RunResume] checkpoint=${RESUME_CHECKPOINT:-scratch}"
echo "[Logging] train_progress=$ABFLOW_TQDM val_progress=$ABFLOW_TQDM test_progress=$ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS path_audit_interval=$ABFLOW_COORD_AUDIT_INTERVAL diagnostics=direct_shared_basis_causal"
echo "[TrainValTestContract] order=train->validation->test checkpoint_selection=validation test_metrics=observation_only test_steps=10 test_seed=2023 infra_fail_fast=on model_invalid=record_and_continue"

python - "$CONFIG_PATH" "$OUTPUT_ROOT" <<'PY'
import json, os, sys
cp, out=sys.argv[1:3]; cfg=json.load(open(cp,encoding='utf-8')); exp=cfg['experiment']; eid=exp['id']; stem=os.path.splitext(os.path.basename(cp))[0]
if not (eid==stem==os.path.basename(os.path.normpath(out))): raise SystemExit(f'formal identity mismatch: {eid} / {stem} / {out}')
if exp.get('protocol')!='formal_train_val_test': raise SystemExit('formal run requires formal_train_val_test')
if os.path.basename(cfg['data']['test']['set'])!='test.json': raise SystemExit('formal run requires observational test.json')
print(f"[ExperimentIdentity] PASS id={eid} protocol={exp['protocol']} test={cfg['data']['test']['set']}")
PY
python -m py_compile "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" "$PROJECT_ROOT/models/modules/am_enc.py" "$PROJECT_ROOT/models/modules/am_egnn.py" "$PROJECT_ROOT/trainer/AbFlow_trainer.py" "$PROJECT_ROOT/train.py"
grep -q "GradientNormOverflowRecovered" "$PROJECT_ROOT/trainer/abs_trainer.py" || { echo "stable gradient-norm guard missing" >&2; exit 2; }
echo "[Preflight] py_compile=PASS stable_grad_norm=PASS config_gpu_authority=CLI"
python - "$PROJECT_ROOT" <<'PY'
import hashlib, os, sys
root=sys.argv[1]
for rel in ['models/AbFlow/AbFlow_model.py','models/modules/am_enc.py','models/modules/am_egnn.py','trainer/AbFlow_trainer.py','train.py','scripts/train/run_R67_R68_v225.sh']:
 p=os.path.join(root,rel)
 if os.path.isfile(p): print(f"[SourceSHA256] {rel} {hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]}")
PY
python - "$CONFIG_PATH" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1],encoding='utf-8'))
sp=cfg['model']['representation']['single_pair']; st=sp['coordinate_state']; pc=sp['pair_coordinate']; cc=sp['coordinate_controller']; bridge=sp.get('bridge',{}); loss=cfg['loss']; d=loss['distogram']; exp=cfg['experiment']
role=str(exp.get('diagnostic_role','')).lower(); eid=exp['id']; mode=str(st.get('mode','')).lower(); bridge_on=bool(bridge.get('enabled',True)); ctrl=str(cc.get('mode','')).lower()
allowed={'r67_direct_shared_prenorm_raw','r67_direct_shared_prenorm_unit'}
if role not in allowed: raise SystemExit(f'R67 launcher requires diagnostic_role in {sorted(allowed)}, got {role!r}')
if mode!='r05_recurrent_carrier': raise SystemExit(f'R67 run requires coordinate_state.mode=r05_recurrent_carrier, got {mode!r}')
if st.get('carrier_recurrence')!='egnn_output_exact_cross_round': raise SystemExit('requires exact EGNN carrier cross-round recurrence')
if st.get('structural_recurrence')!='pred_X_cross_round': raise SystemExit('requires pred_X structural cross-round recurrence')
pair_mode=str(pc.get('mode','')).lower()
if pair_mode!='direct_shared': raise SystemExit('causal runs require pair_coordinate.mode=direct_shared so Pair remains one first-class EGNN edge message for both state and geometry')
if not bool(pc.get('adapter_prenorm',False)): raise SystemExit('causal runs keep Pair-state PreNorm ON in both arms')
if bool(pc.get('semantic_gate',False)): raise SystemExit('causal runs forbid semantic gate')
if not bool(pc.get('same_edge_message_for_state_and_geometry',False)): raise SystemExit('direct-shared causal contract requires same_edge_message_for_state_and_geometry=true')
if int(cfg['model']['architecture']['iter_round'])!=3: raise SystemExit('requires 3 physical rounds')
if not bool(loss.get('generated_region_only',False)): raise SystemExit('requires generated_region_only=true')
if float(loss.get('coarse_anchor_distance',{}).get('weight',0.0))!=0.0: raise SystemExit('causal runs disable coarse anchor')
if float(loss.get('smooth_lddt',{}).get('weight',0.0))!=0.0: raise SystemExit('causal runs disable smooth-lDDT')
dw=float(d.get('weight',0.0)); scope=str(d.get('pair_scope','')).lower(); red=str(d.get('reduction','')).lower()
if dw != 0.0: raise SystemExit('causal runs require Distogram weight=0')
if bool(cc.get('spectral_norm_control', False)): raise SystemExit('spectral control is forbidden in this attribution run')
if not eid.startswith('R67_R05_'): raise SystemExit('R67 id mismatch')
if not bridge_on: raise SystemExit('formal causal runs require bridge.enabled=true')
expected_ctrl={
  'r67_direct_shared_prenorm_raw':'egnn_prenorm_raw',
  'r67_direct_shared_prenorm_unit':'egnn_prenorm_unit_direction',
}[role]
if ctrl!=expected_ctrl: raise SystemExit(f'{role} requires coordinate_controller.mode={expected_ctrl}, got {ctrl!r}')
if str(cc.get('action_input_normalization','')).lower()!='layer_norm': raise SystemExit('both causal arms must keep coordinate-head PreNorm identical')
max_epoch=int(cfg['training']['schedule'].get('max_epoch',0))
if max_epoch <= 0: raise SystemExit('training.schedule.max_epoch must be a positive integer in JSON')
basis='raw_relative_vector' if role.endswith('_raw') else 'unit_direction'
print(f'[DiagnosticIdentity] role={role} bridge=1 pair_mode=direct_shared coord_controller={ctrl} basis={basis} spectral_norm_control=0')
print(f"[TrainingHorizon] source=json max_epoch={max_epoch} patience={cfg['training']['schedule'].get('patience')} launcher_epoch_override=none resume_supported=1")
print('[PairCoordinateContract] pair_prenorm=layer_norm route=direct_shared same_pair_message_for_state_and_geometry=1 zero_init_shared_adapter=1 gate=none')
print(f'[CoordinateControllerContract] coordinate_head_prenorm=1 scalar_activation=identity scalar_bound=none basis={basis} coordinate_aggregation=mean clipping=0 trust_radius=0 manual_scale=0')
print('[CausalQuestion] failed_R67=gated_action+raw; arm_A=direct_shared+raw; arm_B=direct_shared+unit. A-vs-failed isolates Pair routing; B-vs-A isolates raw-distance Cartesian leverage.')
print('[NativeStageContract] telemetry=base/state/direct_pair,raw_distance,basis_norm,dx_over_alpha,dx,spectral_W1_W2 observation_only')
print('[RecurrentGeometryContract] rounds=3 carrier_feedback=egnn_output_exact_cross_round structural_feedback=pred_X_cross_round full_cross_round_gradient=1 no_projection=1 no_kabsch_fusion=1')
print(f'[DistogramContract] weight={dw:g} scope={scope} reduction={red}')
PY

torchrun --nnodes="$NNODES" --nproc_per_node="$NPROC" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$PROJECT_ROOT/train.py" --config "$CONFIG_PATH"
