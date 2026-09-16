#!/usr/bin/env bash
set -euo pipefail

# Backwards-compatible entry point for the unified fairness runner.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec env FRAME_COUNTS="100 300" bash "$SCRIPT_DIR/run_scannet50_fairness.sh" "$@"
