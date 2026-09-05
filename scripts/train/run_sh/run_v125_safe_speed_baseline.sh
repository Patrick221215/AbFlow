#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export ABFLOW_V125_FULL_LOGITS_MB="${ABFLOW_V125_FULL_LOGITS_MB:-448}"

exec bash scripts/train/run_MFSC_v111_full.sh flowtest "${1:-2,3}"
