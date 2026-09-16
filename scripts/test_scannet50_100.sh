#!/usr/bin/env bash
set -euo pipefail

# Two-scene, 100-frame FastVGGT-fairness smoke test.
# Usage: bash scripts/test_scannet50_100.sh CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]
# Every scene executes DenseVGGT, FastVGGT, and SelfTR in sequence on one GPU.
CHECKPOINT=${1:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
DATA_ROOT=${2:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
GT_ROOT=${3:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
GPU_LIST=${4:-0,1,2,3,4,5,6}
OUTPUT_ROOT=${5:-outputs/scannet50_fairness_smoke_100_2scene}
PYTHON_BIN=${EVAL_PYTHON:-python}
SCENES=(scene0000_00 scene0013_02)

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
(( ${#GPUS[@]} >= 1 )) || { echo "GPU_LIST must contain at least one GPU ID" >&2; exit 2; }
for scene in "${SCENES[@]}"; do
  [[ -d "$DATA_ROOT/$scene" ]] || { echo "missing scene directory: $DATA_ROOT/$scene" >&2; exit 2; }
  [[ -f "$GT_ROOT/$scene/${scene}_vh_clean_2.ply" ]] || { echo "missing GT mesh for $scene under $GT_ROOT" >&2; exit 2; }
done

run_scene() {
  local gpu=$1 scene=$2
  for method in densevggt fastvggt selftr; do
    echo "[smoke] GPU ${gpu}: ${scene}, ${method}, 100 frames" >&2
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/eval_scannet50.py \
      --method "$method" --checkpoint "$CHECKPOINT" --dataset-root "$DATA_ROOT" --gt-root "$GT_ROOT" \
      --num-frames 100 --require-exact-frames --scenes "$scene" --skip-summary --resume \
      --fairness-fastvggt-protocol --device cuda:0 --output-dir "$OUTPUT_ROOT/${method}_100"
  done
}

pids=()
for index in "${!SCENES[@]}"; do
  # Scene jobs are assigned round-robin across every GPU supplied by the user.
  # With the default two-scene smoke test, only up to two GPUs can be busy.
  gpu=${GPUS[index % ${#GPUS[@]}]}
  run_scene "$gpu" "${SCENES[index]}" & pids+=("$!")
done
wait "${pids[@]}"

"$PYTHON_BIN" scripts/report_scannet50_fairness_smoke.py \
  --output-root "$OUTPUT_ROOT" \
  --reference-file configs/scannet50_fairness_smoke_100_references.json
