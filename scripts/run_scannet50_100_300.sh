#!/usr/bin/env bash
set -euo pipefail

# Run DenseVGGT, FastVGGT, and SelTR at 100 and 300 frames over all ScanNet50 scenes.
# Usage: bash scripts/run_scannet50_100_300.sh CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]
CHECKPOINT=${1:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
DATA_ROOT=${2:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
GT_ROOT=${3:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
GPU_LIST=${4:-0,1,2,3,4,5,6}
OUTPUT_ROOT=${5:-outputs/scannet50_100_300}
PYTHON_BIN=${EVAL_PYTHON:-python}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
mapfile -t SCENES < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'scene*' -printf '%f\n' | sort)
(( ${#SCENES[@]} == 50 )) || { echo "expected 50 ScanNet scenes under $DATA_ROOT, found ${#SCENES[@]}" >&2; exit 2; }

run_worker() {
  local gpu=$1 worker=$2
  for spec in densevggt:100 fastvggt:100 selftr:100 densevggt:300 fastvggt:300 selftr:300; do
    local method=${spec%%:*} frames=${spec##*:} output="$OUTPUT_ROOT/${method}_${frames}"
    local selected=()
    for index in "${!SCENES[@]}"; do
      if (( index % ${#GPUS[@]} == worker )); then selected+=("${SCENES[index]}"); fi
    done
    if ! CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" scripts/eval_scannet50.py \
      --method "$method" --checkpoint "$CHECKPOINT" --dataset-root "$DATA_ROOT" --gt-root "$GT_ROOT" \
      --num-frames "$frames" --require-exact-frames --scenes "${selected[@]}" --skip-summary --resume \
      --device cuda:0 --output-dir "$output"; then
      echo "GPU $gpu: $method/$frames completed with failed scenes; continuing remaining configurations" >&2
    fi
  done
}

PIDS=()
for worker in "${!GPUS[@]}"; do run_worker "${GPUS[worker]}" "$worker" & PIDS+=("$!"); done
wait "${PIDS[@]}"
for spec in densevggt:100 fastvggt:100 selftr:100 densevggt:300 fastvggt:300 selftr:300; do
  method=${spec%%:*}; frames=${spec##*:}
  "$PYTHON_BIN" scripts/summarize_scannet50.py --output-dir "$OUTPUT_ROOT/${method}_${frames}" --method "$method" --num-frames "$frames"
done
