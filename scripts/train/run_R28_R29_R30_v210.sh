#!/usr/bin/env bash
set -euo pipefail
# V210_1_GOLD_STANDARD_RUN_DIR_CLOSURE
#
# Scientific JSON stays authoritative.  The launcher is allowed to change only
# run bookkeeping: save_dir, scratch resume state, fixed version and log paths.

MODE=${1:-}
EXP=${2:-}
GPUS=${3:-}
BASE_CONFIG=${4:-}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CFG_ROOT=${CFG_ROOT:-scripts/train/configs/R05_ABX_NATIVE_V210_GOLD_STANDARD}
RUN_ROOT="$PROJECT_ROOT/results_module"

[[ "$MODE" == "train" ]] || {
  echo "usage: $0 train EXP GPU_A,GPU_B [config]" >&2
  exit 2
}

case "$EXP" in
  R28_R05_ABX_NATIVE_PAIR_EGNN_U02) DEFAULT_GPU=2,3; DEFAULT_PORT=29828 ;;
  R29_R05_ABX_NATIVE_PAIR_EGNN_DISTOGRAM_U02) DEFAULT_GPU=4,5; DEFAULT_PORT=29829 ;;
  R30_R05_MF_DONOR_SMOOTH_LDDT_U02) DEFAULT_GPU=6,7; DEFAULT_PORT=29830 ;;
  *) echo "Unknown EXP=$EXP" >&2; exit 2 ;;
esac

GPUS=${GPUS:-$DEFAULT_GPU}
PORT=${PORT:-$DEFAULT_PORT}
BASE_CONFIG=${BASE_CONFIG:-$CFG_ROOT/$EXP.json}
[[ -f "$BASE_CONFIG" ]] || { echo "missing config: $BASE_CONFIG" >&2; exit 2; }

# ---------------------------------------------------------------------------
# 1. Validate the immutable scientific/task contract before creating a run.
# ---------------------------------------------------------------------------
python - "$BASE_CONFIG" "$EXP" <<'PY'
import json, sys
p, exp = sys.argv[1:3]
c = json.load(open(p, encoding="utf-8"))
m = c.get("_experiment", {})
if m.get("exp_id") != exp:
    raise SystemExit(f"exp_id mismatch: {m.get('exp_id')!r} != {exp!r}")
if c.get("cdr") != ["H3"] or c.get("paratope") != ["H3"]:
    raise SystemExit(
        f"task contract failed: cdr={c.get('cdr')!r} paratope={c.get('paratope')!r}"
    )
if int(c.get("batch_size", 0)) <= 0:
    raise SystemExit("batch_size missing/invalid")
e = m.get("runtime_env", {})
for k in (
    "ABFLOW_LOSS_SEQUENCE_WEIGHT",
    "ABFLOW_LOSS_STRUCTURE_WEIGHT",
    "ABFLOW_LOSS_INTERFACE_WEIGHT",
    "ABFLOW_LOSS_EDGE_WEIGHT",
    "ABFLOW_LOSS_DISTOGRAM_WEIGHT",
    "ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT",
):
    if k not in e:
        raise SystemExit(f"missing JSON loss weight: {k}")
if e.get("ABFLOW_EPOCH_TEST") != "on":
    raise SystemExit("formal Train->Val->Test requires ABFLOW_EPOCH_TEST=on")
if e.get("ABFLOW_EPOCH_TEST_INTERVAL") != "1":
    raise SystemExit("formal Test interval must be 1")
if e.get("ABFLOW_EPOCH_TEST_N_STEPS") != "10":
    raise SystemExit("formal Test n_steps must be 10")

if exp.startswith("R28_"):
    expected=("on","off","off")
elif exp.startswith("R29_"):
    expected=("on","on","off")
else:
    expected=("off","off","on")
actual=(
    e.get("ABFLOW_ABX_NATIVE_REPR"),
    e.get("ABFLOW_ABX_DISTOGRAM"),
    e.get("ABFLOW_MF_SMOOTH_LDDT"),
)
if actual != expected:
    raise SystemExit(f"scientific factor mismatch: actual={actual} expected={expected}")

print(
    f"[V210.1PreflightPASS] {exp} task={c['cdr']} batch={c['batch_size']} "
    f"pair={actual[0]} dist={actual[1]} slddt={actual[2]}"
)
PY

# ---------------------------------------------------------------------------
# 2. Reserve version_N atomically.  This restores the established AbFlow
#    launcher contract and prevents two DDP launches from sharing a directory.
# ---------------------------------------------------------------------------
EXP_ROOT="$RUN_ROOT/$EXP"
mkdir -p "$EXP_ROOT"
RUN_VERSION=0
while ! mkdir "$EXP_ROOT/version_${RUN_VERSION}" 2>/dev/null; do
  RUN_VERSION=$((RUN_VERSION + 1))
done
RUN_VERSION_DIR="$EXP_ROOT/version_${RUN_VERSION}"
export ABFLOW_FIXED_VERSION="$RUN_VERSION"

RUN_TIME_LOG="$RUN_VERSION_DIR/run_time.log"
LATEST_RUN_LOG="$EXP_ROOT/run_time.log"
export ABFLOW_RUN_TIME_LOG="$RUN_TIME_LOG"
: > "$RUN_TIME_LOG"

if [[ -e "$LATEST_RUN_LOG" && ! -L "$LATEST_RUN_LOG" ]]; then
  mv "$LATEST_RUN_LOG" \
    "$EXP_ROOT/run_time.pre_v210_1.$(date +%Y%m%d_%H%M%S).log"
fi
ln -sfn "version_${RUN_VERSION}/run_time.log" "$LATEST_RUN_LOG"

export PYTHONUNBUFFERED=1
exec > >(stdbuf -oL -eL tee -a "$RUN_TIME_LOG") 2>&1

echo "[RunLog] canonical=$RUN_TIME_LOG latest=$LATEST_RUN_LOG"
echo "[RunVersion] EXP=$EXP fixed_version=$ABFLOW_FIXED_VERSION dir=$RUN_VERSION_DIR"

# ---------------------------------------------------------------------------
# 3. Generate the exact runtime JSON.  Only save_dir/resume are launcher-owned.
#    Keep _experiment intact so train.sh can export all ABFLOW_* scientific env.
# ---------------------------------------------------------------------------
GEN_DIR="$EXP_ROOT/generated_configs"
mkdir -p "$GEN_DIR"
RUN_CONFIG="$GEN_DIR/${EXP}.version_${RUN_VERSION}.json"

python - "$BASE_CONFIG" "$RUN_CONFIG" "$RUN_VERSION_DIR" "${ABFLOW_FORCE_SCRATCH:-0}" <<'PY'
import json, sys
src, dst, run_dir, force_scratch = sys.argv[1:5]
cfg = json.load(open(src, encoding="utf-8"))

# The only formal runtime mutation: output destination.
cfg["save_dir"] = run_dir

# Scratch is a launch choice, not a scientific factor.
if force_scratch == "1":
    cfg.pop("resume_checkpoint", None)
else:
    if cfg.get("resume_checkpoint") is None:
        cfg.pop("resume_checkpoint", None)

json.dump(cfg, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open(dst, "a", encoding="utf-8").write("\n")

check = json.load(open(dst, encoding="utf-8"))
if check.get("save_dir") != run_dir:
    raise SystemExit("generated save_dir contract failed")
if check.get("cdr") != ["H3"] or check.get("paratope") != ["H3"]:
    raise SystemExit("generated task contract failed")
if "_experiment" not in check or "runtime_env" not in check["_experiment"]:
    raise SystemExit("generated config lost scientific runtime_env")

print(
    f"[V210.1GeneratedConfigPASS] save_dir={check['save_dir']} "
    f"cdr={check['cdr']} paratope={check['paratope']}"
)
PY

echo "[V210.1SourceSHA] model=$(sha256sum "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" | awk '{print $1}')"
echo "[V210.1SourceSHA] trainer=$(sha256sum "$PROJECT_ROOT/trainer/AbFlow_trainer.py" | awk '{print $1}')"
echo "[V210.1SourceSHA] train.sh=$(sha256sum "$PROJECT_ROOT/scripts/train/train.sh" | awk '{print $1}')"
echo "[V210.1SourceSHA] launcher=$(sha256sum "$PROJECT_ROOT/scripts/train/run_R28_R29_R30_v210.sh" | awk '{print $1}')"
echo "[V210.1RunConfig] base=$BASE_CONFIG generated=$RUN_CONFIG"
echo "[V210.1Run] EXP=$EXP GPUs=$GPUS PORT=$PORT Train->Val->Test"

cd "$PROJECT_ROOT"
GPU="$GPUS" PORT="$PORT" bash scripts/train/train.sh "$RUN_CONFIG"
