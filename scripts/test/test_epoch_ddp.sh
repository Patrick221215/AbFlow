#!/bin/bash
set -euo pipefail
# Backward-compatible alias for the formal RAbD DDP test path.
CODE_DIR=$(realpath "$(dirname "$0")/../..")
exec bash "$CODE_DIR/scripts/test/test.sh" "$@"
