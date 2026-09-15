#!/usr/bin/env python3
"""Merge per-scene ScanNet50 worker outputs into one method/configuration report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def numeric_mean(items: list[dict[str, Any]]) -> dict[str, float]:
    keys = set().union(*(item.keys() for item in items))
    return {key: float(np.mean([item[key] for item in items if isinstance(item.get(key), (float, int, np.number))]))
            for key in keys if any(isinstance(item.get(key), (float, int, np.number)) for item in items)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    args = parser.parse_args()
    scenes, failures = [], []
    for path in sorted(args.output_dir.glob("scene*/metrics.json")):
        scenes.append(json.loads(path.read_text()))
    for path in sorted(args.output_dir.glob("scene*/failure.json")):
        failures.append(json.loads(path.read_text()))
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
    payload = {"method": args.method, "num_frames_requested": args.num_frames, "scene_count": len(scenes),
               "failed_scene_count": len(failures), "scenes": scenes, "failures": failures,
               "mean_pose": numeric_mean([item["pose"] for item in scenes]),
               "mean_fastvggt_pose": numeric_mean([item["fastvggt_pose"] for item in scenes]),
               "mean_reconstruction": numeric_mean([item["reconstruction"] for item in scenes]),
               "mean_depth": numeric_mean([item["depth"] for item in scenes]),
               "mean_efficiency": numeric_mean([item["efficiency"] for item in scenes]),
               "token_retention": retention}
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir / "metrics.json"), "scenes": len(scenes), "failures": len(failures)}))


if __name__ == "__main__":
    main()
