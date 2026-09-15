#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate SelfTR only, without --resume, so an optimized implementation
# replaces prior per-scene results.  Scene shards are disjoint across GPUs.
# Usage: bash scripts/rerun_scannet50_selftr.sh CHECKPOINT DATA_ROOT GT_ROOT NUM_FRAMES [GPU_LIST] [OUTPUT_DIR]
CHECKPOINT=${1:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT NUM_FRAMES [GPU_LIST] [OUTPUT_DIR]"}
DATA_ROOT=${2:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT NUM_FRAMES [GPU_LIST] [OUTPUT_DIR]"}
GT_ROOT=${3:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT NUM_FRAMES [GPU_LIST] [OUTPUT_DIR]"}
NUM_FRAMES=${4:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT NUM_FRAMES [GPU_LIST] [OUTPUT_DIR]"}
GPU_LIST=${5:-0,1,2,3,4,5,6}
OUTPUT_DIR=${6:-outputs/scannet50_selftr_${NUM_FRAMES}}
PYTHON_BIN=${EVAL_PYTHON:-python}

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
mapfile -t SCENES < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'scene*' -printf '%f\n' | sort)
(( ${#SCENES[@]} == 50 )) || { echo "expected 50 ScanNet scenes under $DATA_ROOT, found ${#SCENES[@]}" >&2; exit 2; }

run_worker() {
  local gpu=$1 worker=$2
  local selected=()
  for index in "${!SCENES[@]}"; do
    if (( index % ${#GPUS[@]} == worker )); then selected+=("${SCENES[index]}"); fi
  done
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" scripts/eval_scannet50.py \
    --method selftr --checkpoint "$CHECKPOINT" --dataset-root "$DATA_ROOT" --gt-root "$GT_ROOT" \
    --num-frames "$NUM_FRAMES" --require-exact-frames --scenes "${selected[@]}" --skip-summary \
    --device cuda:0 --output-dir "$OUTPUT_DIR"
}

PIDS=()
for worker in "${!GPUS[@]}"; do
  run_worker "${GPUS[worker]}" "$worker" & PIDS+=("$!")
done
wait "${PIDS[@]}"
"$PYTHON_BIN" scripts/summarize_scannet50.py --output-dir "$OUTPUT_DIR" --method selftr --num-frames "$NUM_FRAMES"
