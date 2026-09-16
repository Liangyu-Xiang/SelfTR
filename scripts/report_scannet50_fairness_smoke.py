#!/usr/bin/env python3
"""Create a transparent two-scene ScanNet fairness smoke-test comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METHODS = ("densevggt", "fastvggt", "selftr")
SCENES = ("scene0000_00", "scene0013_02")
ACTUAL_METRICS = {
    "CD (FastVGGT pipeline, m)": ("fastvggt_reconstruction", "cd_m"),
    "F1@5cm (FastVGGT pipeline)": ("fastvggt_reconstruction", "f1_at_0.05m"),
    "NC (FastVGGT pipeline)": ("fastvggt_reconstruction", "nc"),
    "ATE (FastVGGT, m)": ("fastvggt_pose", "fastvggt_ate_m"),
    "ARE (FastVGGT, deg)": ("fastvggt_pose", "fastvggt_are_deg"),
    "RPE-rot (FastVGGT, deg)": ("fastvggt_pose", "fastvggt_rpe_rot_deg"),
    "RPE-trans (FastVGGT, m)": ("fastvggt_pose", "fastvggt_rpe_trans_m"),
    "Latency (model-only, s)": ("efficiency", "latency_s"),
    "FPS (model-only)": ("efficiency", "fps"),
    "VRAM allocated (GiB)": ("efficiency", "peak_vram_allocated_gib"),
}
LOCAL_FIELD_MAP = {
    "CD (FastVGGT pipeline, m)": "reconstruction.cd_m",
    "F1@5cm (FastVGGT pipeline)": "reconstruction.f1_at_0.05m",
    "NC (FastVGGT pipeline)": "reconstruction.nc",
    "ATE (FastVGGT, m)": "fastvggt_pose.fastvggt_ate_m",
    "ARE (FastVGGT, deg)": "fastvggt_pose.fastvggt_are_deg",
    "RPE-rot (FastVGGT, deg)": "fastvggt_pose.fastvggt_rpe_rot_deg",
    "RPE-trans (FastVGGT, m)": "fastvggt_pose.fastvggt_rpe_trans_m",
    "Latency (model-only, s)": "efficiency.latency_s",
    "FPS (model-only)": "efficiency.fps",
    "VRAM allocated (GiB)": "efficiency.peak_vram_allocated_gib",
}


def mean(items: list[float]) -> float:
    return sum(items) / len(items)


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value):.4f}"


def metric_value(item: dict[str, Any], section: str, field: str) -> float:
    value = item.get(section, {}).get(field)
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"missing numeric metric {section}.{field} in {item.get('scene')}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference_file.read_text())
    actual: dict[str, dict[str, float]] = {}
    for method in METHODS:
        results = []
        for scene in SCENES:
            path = args.output_root / f"{method}_100" / scene / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing smoke-test output: {path}")
            item = json.loads(path.read_text())
            if item.get("protocol_id") != "fastvggt_fairness_v3" or item.get("frames") != 100:
                raise RuntimeError(f"{path} is not a current 100-frame fairness result")
            if item.get("scene") != scene or item.get("method") != method:
                raise RuntimeError(f"{path} has incompatible scene/method metadata")
            results.append(item)
        actual[method] = {
            name: mean([metric_value(item, *source) for item in results])
            for name, source in ACTUAL_METRICS.items()
        }

    rows = []
    lines = [
        "# ScanNet-50 fairness smoke test (100 frames, two scenes)", "",
        "Scenes: `scene0000_00`, `scene0013_02`. Actual values are arithmetic means over these two scenes.", "",
        "The **local reference** is a historical two-scene local run using the older main-protocol evaluator, so it is diagnostic only—not a numerical pass/fail target for the current FastVGGT fairness protocol.",
        "The **FastVGGT paper reference** is the published ScanNet-50 mean over 50 scenes; it is reported only for DenseVGGT and FastVGGT and likewise is not directly comparable to a two-scene smoke test.", "",
    ]
    for method in METHODS:
        lines.extend([f"## {method}", "", "| Metric | Actual fairness | Local historical reference | Δ actual−local | FastVGGT paper reference | Δ actual−paper |", "|---|---:|---:|---:|---:|---:|"])
        local = reference["local_historical"]["methods"].get(method, {})
        paper = reference["fastvggt_paper"]["methods"].get(method, {})
        for metric, value in actual[method].items():
            local_value = local.get(LOCAL_FIELD_MAP[metric])
            paper_value = paper.get(metric)
            row = {
                "method": method, "metric": metric, "actual": value,
                "local_reference": local_value, "paper_reference": paper_value,
                "delta_actual_local": None if local_value is None else value - local_value,
                "delta_actual_paper": None if paper_value is None else value - paper_value,
            }
            rows.append(row)
            lines.append(f"| {metric} | {fmt(value)} | {fmt(local_value)} | {fmt(row['delta_actual_local'])} | {fmt(paper_value)} | {fmt(row['delta_actual_paper'])} |")
        lines.append("")
    payload = {
        "protocol": "fastvggt_fairness_v3", "num_frames": 100, "scenes": list(SCENES),
        "actual_two_scene_mean": actual, "references": reference, "comparison_rows": rows,
        "notes": [
            "Local historical values use a pre-v3 main-protocol evaluator and are diagnostic only.",
            "Paper values are 50-scene means and use the paper's timing boundary; only CD and latency are published for the 100-frame ScanNet-50 reconstruction table.",
        ],
    }
    json_path = args.output_root / "smoke_comparison.json"
    md_path = args.output_root / "smoke_comparison.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Smoke comparison: {md_path}")
    print(f"Machine-readable comparison: {json_path}")


if __name__ == "__main__":
    main()
