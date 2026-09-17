#!/usr/bin/env python3
"""Merge per-scene ScanNet50 worker outputs into one method/configuration report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def compatible_frame_count(item: dict[str, Any], requested: int) -> bool:
    """Accept legacy full-length results and new short-sequence results."""
    actual = item.get("frames")
    recorded_requested = item.get("requested_frames", actual)
    frame_ids = item.get("frame_ids")
    return (isinstance(actual, int) and 1 <= actual <= requested
            and recorded_requested == requested
            and isinstance(frame_ids, list) and len(frame_ids) == actual)


def numeric_mean(items: list[dict[str, Any]]) -> dict[str, float]:
    keys = set().union(*(item.keys() for item in items))
    return {key: float(np.mean([item[key] for item in items if isinstance(item.get(key), (float, int, np.number))]))
            for key in keys if any(isinstance(item.get(key), (float, int, np.number)) for item in items)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--fairness-fastvggt-protocol", action="store_true")
    args = parser.parse_args()
    scenes, failures = [], []
    for path in sorted(args.output_dir.glob("scene*/metrics.json")):
        scenes.append(json.loads(path.read_text()))
    for path in sorted(args.output_dir.glob("scene*/failure.json")):
        # A successful resumed run can coexist with a stale failure file left
        # by an older evaluator.  The successful scene result is authoritative.
        if not path.with_name("metrics.json").exists():
            failures.append(json.loads(path.read_text()))
    if args.fairness_fastvggt_protocol and (len(scenes) != 50 or failures):
        raise RuntimeError(
            f"fairness summaries require all 50 scenes with no failures; got {len(scenes)} successful and {len(failures)} failed"
        )
    if not scenes:
        payload = {"method": args.method, "num_frames_requested": args.num_frames, "scene_count": 0,
                   "failed_scene_count": len(failures), "scenes": [], "failures": failures,
                   "error": "no scene completed successfully"}
        (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output_dir / "metrics.json"), "scenes": 0, "failures": len(failures)}))
        return
    token = [item["efficiency"]["token_retention"] for item in scenes]
    if args.method == "selftr":
        retention = {"policy": "selftr_stagewise", "stages": [
            {"stage": index + 1, "mean_retention_percent": float(np.mean(values)) if values else None}
            for index, values in ((i, [entry["stages"][i]["retention_percent"] for entry in token if len(entry.get("stages", [])) > i]) for i in range(3))]}
    else:
        values = [entry.get("retention_percent") for entry in token if entry.get("retention_percent") is not None]
        retention = {"policy": "fixed_once", "retention_percent": float(np.mean(values)) if values else None}
    if args.fairness_fastvggt_protocol:
        invalid = [item.get("scene") for item in scenes
                   if item.get("protocol_id") != "fastvggt_fairness_v3"
                   or item.get("method") != args.method or not compatible_frame_count(item, args.num_frames)]
        if invalid:
            raise RuntimeError(f"fairness summary contains stale/incompatible results, e.g. {invalid[:3]}")
    actual_frame_counts = [item["frames"] for item in scenes]
    payload = {"method": args.method, "num_frames_requested": args.num_frames,
               "protocol": {"experiment_kind": "fairness" if args.fairness_fastvggt_protocol else "main",
                            "sampler": ("released FastVGGT RGB/pose integer-stride selection" if args.fairness_fastvggt_protocol
                                        else "project stride-3 RGB/pose/depth selection")},
               "scene_count": len(scenes),
               "failed_scene_count": len(failures), "scenes": scenes, "failures": failures,
               "effective_frame_counts": {"min": min(actual_frame_counts), "max": max(actual_frame_counts),
                                          "short_scene_count": sum(count < args.num_frames for count in actual_frame_counts)},
               "mean_pose": numeric_mean([item["pose"] for item in scenes]),
               "mean_fastvggt_pose": numeric_mean([item["fastvggt_pose"] for item in scenes]),
               "mean_reconstruction": numeric_mean([item["reconstruction"] for item in scenes]),
               "mean_fastvggt_reconstruction": numeric_mean([item.get("fastvggt_reconstruction", {}) for item in scenes]),
               "mean_depth": numeric_mean([item["depth"] for item in scenes]),
               "mean_efficiency": numeric_mean([item["efficiency"] for item in scenes]),
               "token_retention": retention}
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir / "metrics.json"), "scenes": len(scenes), "failures": len(failures)}))


if __name__ == "__main__":
    main()
