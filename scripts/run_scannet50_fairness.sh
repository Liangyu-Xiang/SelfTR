#!/usr/bin/env bash
set -euo pipefail

# Unified FastVGGT-protocol comparison for DenseVGGT, FastVGGT, and SelfTR.
# Usage: bash scripts/run_scannet50_fairness.sh CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]
# Set FRAME_COUNTS='100 300' to run only a subset while debugging.
CHECKPOINT=${1:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
DATA_ROOT=${2:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
GT_ROOT=${3:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
GPU_LIST=${4:-0,1,2,3,4,5,6}
OUTPUT_ROOT=${5:-outputs/scannet50_fairness_v3}
FRAME_COUNTS=${FRAME_COUNTS:-"100 300 500 1000"}
PYTHON_BIN=${EVAL_PYTHON:-python}

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
read -r -a FRAMES <<< "$FRAME_COUNTS"
mapfile -t SCENES < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'scene*' -printf '%f\n' | sort)
(( ${#SCENES[@]} == 50 )) || { echo "expected 50 ScanNet scenes under $DATA_ROOT, found ${#SCENES[@]}" >&2; exit 2; }

run_worker() {
  local gpu=$1 worker=$2
  local selected=()
  for index in "${!SCENES[@]}"; do
    if (( index % ${#GPUS[@]} == worker )); then selected+=("${SCENES[index]}"); fi
  done
  for frames in "${FRAMES[@]}"; do
    for method in densevggt fastvggt selftr; do
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" scripts/eval_scannet50.py \
        --method "$method" --checkpoint "$CHECKPOINT" --dataset-root "$DATA_ROOT" --gt-root "$GT_ROOT" \
        --num-frames "$frames" --require-exact-frames --scenes "${selected[@]}" --skip-summary --resume \
        --fairness-fastvggt-protocol --device cuda:0 --output-dir "$OUTPUT_ROOT/${method}_${frames}"
    done
  done
}

pids=()
for worker in "${!GPUS[@]}"; do
  run_worker "${GPUS[worker]}" "$worker" & pids+=("$!")
done
wait "${pids[@]}"

for frames in "${FRAMES[@]}"; do
  for method in densevggt fastvggt selftr; do
    "$PYTHON_BIN" scripts/summarize_scannet50.py --output-dir "$OUTPUT_ROOT/${method}_${frames}" \
      --method "$method" --num-frames "$frames" --fairness-fastvggt-protocol
  done
done
"$PYTHON_BIN" scripts/validate_scannet50_fairness.py --output-root "$OUTPUT_ROOT" --frames "${FRAMES[@]}"
