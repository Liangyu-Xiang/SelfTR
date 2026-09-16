#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point: execute the 1000-frame FastVGGT fairness protocol.
# SCANNET_GT_ROOT is required when the complete frame pool and ScanNet meshes
# reside in separate locations.
CHECKPOINT=${1:?"usage: $0 CHECKPOINT [OUTPUT_ROOT]"}
OUTPUT_ROOT=${2:-outputs/scannet50_fairness_v3}
SCANNET_DATA_ROOT=${SCANNET_DATA_ROOT:?"set SCANNET_DATA_ROOT to the complete ScanNet frame extraction"}
SCANNET_GT_ROOT=${SCANNET_GT_ROOT:-"$SCANNET_DATA_ROOT/extracted/scannet-dataset"}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec env FRAME_COUNTS="1000" bash "$SCRIPT_DIR/run_scannet50_fairness.sh" \
  "$CHECKPOINT" "$SCANNET_DATA_ROOT" "$SCANNET_GT_ROOT" "$GPU_LIST" "$OUTPUT_ROOT"
