#!/usr/bin/env bash
set -euo pipefail

# V211 formal launcher: the selected JSON is the sole experiment authority.
# This wrapper contains no experiment-id/GPU/port/loss switch table. It only
# validates the chosen JSON and delegates to the universal JSON-driven runner.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
CONFIG_PATH=${1:-}

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: bash $0 <experiment.json>" >&2
  exit 2
fi
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$PROJECT_ROOT/${CONFIG_PATH#./}"
fi
[[ -f "$CONFIG_PATH" ]] || {
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
}

python "$PROJECT_ROOT/scripts/train/validate_R28_R29_R30_v211.py" \
  --project-root "$PROJECT_ROOT" \
  --config "$CONFIG_PATH"

exec bash "$PROJECT_ROOT/scripts/train/train.sh" "$CONFIG_PATH"
