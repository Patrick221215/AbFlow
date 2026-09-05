#!/usr/bin/env bash
###
 # @Author: Patrick221215 1427584833@qq.com
 # @Date: 2026-08-20 19:44:07
 # @LastEditors: Patrick221215 1427584833@qq.com
 # @LastEditTime: 2026-08-20 19:44:11
 # @FilePath: /cjm/project/AbFlow/scripts/train/run_F01_F02_F03_sf2m_r3_v71 copy.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
set -euo pipefail

PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-/home/data3/cjm/project/AbFlow}
RUN_ROOT=${ABFLOW_RUN_ROOT:-${PROJECT_ROOT}/results_module}
LAUNCHER=${ABFLOW_V71_LAUNCHER:-scripts/train/run_F01_F02_F03_sf2m_r3_v71.sh}

EXP1=F01_PCS_RC_LC_R1_SF2M_R3_CANONICAL_FLOW
EXP2=F02_PCS_RC_LC_R1_SF2M_R3_TIED_SCORE_CONTROL
EXP3=F03_PCS_RC_LC_R1_SF2M_R3_DUALFIELD_SCORE_FLOW
CFG1=scripts/train/configs/${EXP1}.json
CFG2=scripts/train/configs/${EXP2}.json
CFG3=scripts/train/configs/${EXP3}.json

GPU1=${ABFLOW_V71_GPU1:-2,3}
GPU2=${ABFLOW_V71_GPU2:-4,5}
GPU3=${ABFLOW_V71_GPU3:-6,7}
PORT1=${ABFLOW_V71_PORT1:-29661}
PORT2=${ABFLOW_V71_PORT2:-29662}
PORT3=${ABFLOW_V71_PORT3:-29663}

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/_launcher_logs"
LOG1="${RUN_ROOT}/_launcher_logs/${EXP1}.log"
LOG2="${RUN_ROOT}/_launcher_logs/${EXP2}.log"
LOG3="${RUN_ROOT}/_launcher_logs/${EXP3}.log"
echo "============================================================"
echo "AbFlow v71: three matched SF2M-R3 configurations"
echo "${EXP1} -> ${RUN_ROOT}/${EXP1} | GPU=${GPU1}"
echo "${EXP2} -> ${RUN_ROOT}/${EXP2} | GPU=${GPU2}"
echo "${EXP3} -> ${RUN_ROOT}/${EXP3} | GPU=${GPU3}"
echo "============================================================"

# Disable live watchers during parallel training so three jobs do not compete
# for the same evaluation GPU. Final evaluation is sequential below.
(
  cd "${PROJECT_ROOT}"
  PORT="${PORT1}" ABFLOW_RUN_ROOT="${RUN_ROOT}" ABFLOW_AUTO_TOPK_EVAL=off \
  bash "${LAUNCHER}" train "${EXP1}" "${GPU1}" "${CFG1}"
) > "${LOG1}" 2>&1 & P1=$!
(
  cd "${PROJECT_ROOT}"
  PORT="${PORT2}" ABFLOW_RUN_ROOT="${RUN_ROOT}" ABFLOW_AUTO_TOPK_EVAL=off \
  bash "${LAUNCHER}" train "${EXP2}" "${GPU2}" "${CFG2}"
) > "${LOG2}" 2>&1 & P2=$!
(
  cd "${PROJECT_ROOT}"
  PORT="${PORT3}" ABFLOW_RUN_ROOT="${RUN_ROOT}" ABFLOW_AUTO_TOPK_EVAL=off \
  bash "${LAUNCHER}" train "${EXP3}" "${GPU3}" "${CFG3}"
) > "${LOG3}" 2>&1 & P3=$!

echo "PIDs: ${P1} ${P2} ${P3}"
echo "tail -f ${LOG1}"
echo "tail -f ${LOG2}"
echo "tail -f ${LOG3}"
STATUS=0
wait "${P1}" || STATUS=1
wait "${P2}" || STATUS=1
wait "${P3}" || STATUS=1
# Archive launcher logs under their final experiment folders only AFTER the
# launchers have created/validated fresh-run directories.
for pair in "${EXP1}:${LOG1}" "${EXP2}:${LOG2}" "${EXP3}:${LOG3}"; do
  EXP="${pair%%:*}"; LOG="${pair#*:}"
  if [[ -d "${RUN_ROOT}/${EXP}" && -f "${LOG}" ]]; then
    cp -f "${LOG}" "${RUN_ROOT}/${EXP}/launcher.log"
  fi
done
if [[ "${STATUS}" -ne 0 ]]; then
  echo "At least one training run failed; final TopK evaluation skipped."
  exit "${STATUS}"
fi

if [[ "${ABFLOW_V71_FINAL_TOPK_EVAL:-on}" == "on" ]]; then
  EVAL_GPU=${ABFLOW_V71_EVAL_GPU:-0}
  TEST_JSON=${ABFLOW_TOPK_TEST_JSON:-${PROJECT_ROOT}/datasets/RAbD/test.json}
  EVAL_SCRIPT=${ABFLOW_TOPK_EVAL_SCRIPT:-scripts/test/evaluate_topk_map.py}
  if [[ -f "${PROJECT_ROOT}/${EVAL_SCRIPT}" && -f "${TEST_JSON}" ]]; then
    for EXP in "${EXP1}" "${EXP2}" "${EXP3}"; do
      echo "[v71] sequential final TopK: ${EXP} on GPU ${EVAL_GPU}"
      (
        cd "${PROJECT_ROOT}"
        python "${EVAL_SCRIPT}" \
          --exp-id "${EXP}" \
          --run-dir "${RUN_ROOT}/${EXP}" \
          --test-json "${TEST_JSON}" \
          --gpu-ids "${EVAL_GPU}" \
          --project-root "${PROJECT_ROOT}" \
          --launcher "${LAUNCHER}"
      ) >> "${RUN_ROOT}/${EXP}/auto_topk_eval_final.log" 2>&1 || true
    done
  else
    echo "[v71] evaluator or test JSON missing; training completed, final eval skipped."
  fi
fi
