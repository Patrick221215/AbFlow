#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-}
RESULT_DIR=${2:-}
TEST_JSON=${3:-/home/data3/cjm/project/AbFlow/datasets/RAbD/test.json}
GPU_ID=${ABFLOW_SF2M_SDE_GPU:-0}
INFER_G=${ABFLOW_SF2M_INFER_G:-}
LAUNCHER=${ABFLOW_V71_LAUNCHER:-scripts/train/run_F01_F02_F03_sf2m_r3_v71.sh}
EXP=F03_PCS_RC_LC_R1_SF2M_R3_DUALFIELD_SCORE_FLOW

if [[ -z "$CKPT" || -z "$RESULT_DIR" || -z "$INFER_G" ]]; then
  echo "Usage: bash $0 <F03_CKPT> <RESULT_DIR> [TEST_JSON]"
  echo "Required: choose ABFLOW_SF2M_INFER_G once using validation-only diagnostics; do NOT tune it on test."
  echo "Example syntax: ABFLOW_SF2M_INFER_G=<fixed_g> ABFLOW_SF2M_SDE_GPU=0 bash $0 ..."
  exit 2
fi

# This is a SAME-CHECKPOINT sampler diagnostic, not checkpoint selection.
# g is an inference-only time-homogeneous diffusion amplitude permitted by
# separate probability-flow/score parameterization. It must be reported as a
# diagnostic setting, not tuned on the test set.
ABFLOW_SF2M_F03_SAMPLER_MODE=sf2m_sde \
ABFLOW_SF2M_INFER_G="$INFER_G" \
ABFLOW_SAMPLE_N_STEPS=${ABFLOW_SAMPLE_N_STEPS:-50} \
bash "$LAUNCHER" test "$EXP" "$GPU_ID" "$CKPT" "$RESULT_DIR" "$TEST_JSON"
