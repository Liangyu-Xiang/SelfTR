#!/usr/bin/env bash
set -euo pipefail

# Unified FastVGGT-protocol comparison for DenseVGGT, FastVGGT, and SelfTR.
# Usage: bash scripts/run_scannet50_fairness.sh CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]
# Set FRAME_COUNTS='100 300' to run only a subset while debugging. All 50
# scenes are assigned round-robin across every GPU in GPU_LIST.
CHECKPOINT=${1:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
DATA_ROOT=${2:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
GT_ROOT=${3:?"usage: $0 CHECKPOINT DATA_ROOT GT_ROOT [GPU_LIST] [OUTPUT_ROOT]"}
GPU_LIST=${4:-0,1,2,3,4,5,6}
OUTPUT_ROOT=${5:-outputs/scannet50_fairness_v3}
FRAME_COUNTS=${FRAME_COUNTS:-"100 300 500 1000"}
PYTHON_BIN=${EVAL_PYTHON:-python}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-5}

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
        --fairness-fastvggt-protocol --save-visualizations --device cuda:0 --output-dir "$OUTPUT_ROOT/${method}_${frames}"
    done
  done
}

completed_scene_jobs() {
  local count=0 frames method
  for frames in "${FRAMES[@]}"; do
    for method in densevggt fastvggt selftr; do
      if [[ -d "$OUTPUT_ROOT/${method}_${frames}" ]]; then
        count=$((count + $(find "$OUTPUT_ROOT/${method}_${frames}" -mindepth 2 -maxdepth 2 -path '*/scene*/metrics.json' -type f | wc -l)))
      fi
    done
  done
  printf '%s' "$count"
}

progress_monitor() {
  local total=$((50 * ${#FRAMES[@]} * 3)) start now done elapsed eta width=36 filled percent live pid
  start=$(date +%s)
  while true; do
    done=$(completed_scene_jobs)
    now=$(date +%s)
    elapsed=$((now - start))
    percent=$((100 * done / total))
    filled=$((width * done / total))
    if (( done > 0 && done < total )); then
      eta=$((elapsed * (total - done) / done))
      printf '\r[%-*s] %3d%%  %d/%d scene-jobs  elapsed %02d:%02d  ETA %02d:%02d' \
        "$width" "$(printf '%*s' "$filled" '' | tr ' ' '#')" "$percent" "$done" "$total" \
        $((elapsed / 60)) $((elapsed % 60)) $((eta / 60)) $((eta % 60)) >&2
    else
      printf '\r[%-*s] %3d%%  %d/%d scene-jobs  elapsed %02d:%02d' \
        "$width" "$(printf '%*s' "$filled" '' | tr ' ' '#')" "$percent" "$done" "$total" \
        $((elapsed / 60)) $((elapsed % 60)) >&2
    fi
    live=0
    for pid in "$@"; do
      if kill -0 "$pid" 2>/dev/null; then live=1; break; fi
    done
    (( live )) || break
    sleep "$PROGRESS_INTERVAL"
  done
  printf '\n' >&2
}

pids=()
for worker in "${!GPUS[@]}"; do
  run_worker "${GPUS[worker]}" "$worker" & pids+=("$!")
done
progress_monitor "${pids[@]}" & progress_pid=$!
trap 'kill "$progress_pid" 2>/dev/null || true' EXIT
wait "${pids[@]}"
wait "$progress_pid"
trap - EXIT

for frames in "${FRAMES[@]}"; do
  for method in densevggt fastvggt selftr; do
    "$PYTHON_BIN" scripts/summarize_scannet50.py --output-dir "$OUTPUT_ROOT/${method}_${frames}" \
      --method "$method" --num-frames "$frames" --fairness-fastvggt-protocol
  done
done
"$PYTHON_BIN" scripts/validate_scannet50_fairness.py --output-root "$OUTPUT_ROOT" --frames "${FRAMES[@]}"
