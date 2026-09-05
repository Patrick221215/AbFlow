#!/usr/bin/env bash
###
 # @Author: Patrick221215 1427584833@qq.com
 # @Date: 2026-09-02 23:42:09
 # @LastEditors: Patrick221215 1427584833@qq.com
 # @LastEditTime: 2026-09-03 00:11:36
 # @FilePath: /cjm/project/AbFlow/scripts/train/run_v126_batched_runtime.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export ABFLOW_MFDESIGN_BATCHED_RUNTIME="${ABFLOW_MFDESIGN_BATCHED_RUNTIME:-on}"
export ABFLOW_V127_FULL_LOGITS_MB="${ABFLOW_V127_FULL_LOGITS_MB:-2048}"

# Same run:
# step0: numerical parity gate
# step0-2: gradient-contract gate
# step0-7: focused RuntimePerf
# then continues automatically to epoch 200 with profiling disabled.
exec bash scripts/train/run_MFSC_v111_full.sh train "${1:-2,3}"
