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
# Each row names one protocol explicitly.  A local reference is shown only for
# the project reconstruction path, because the historical run predates the
# FastVGGT full-cloud evaluator.  The paper publishes only its FastVGGT-path CD
# and timing values for the 100-frame ScanNet-50 table.
METRICS = (
    ("Project Acc (m)", "reconstruction", "acc_m", "reconstruction.acc_m", None),
    ("Project Comp (m)", "reconstruction", "comp_m", "reconstruction.comp_m", None),
    ("Project Overall (m)", "reconstruction", "overall_m", "reconstruction.overall_m", None),
    ("Project CD (m)", "reconstruction", "cd_m", "reconstruction.cd_m", None),
    ("Project Precision@5cm", "reconstruction", "precision_at_0.05m", "reconstruction.precision_at_0.05m", None),
    ("Project Recall@5cm", "reconstruction", "recall_at_0.05m", "reconstruction.recall_at_0.05m", None),
    ("Project F1@5cm", "reconstruction", "f1_at_0.05m", "reconstruction.f1_at_0.05m", None),
    ("Project NC", "reconstruction", "nc", "reconstruction.nc", None),
    ("FastVGGT Acc (m)", "fastvggt_reconstruction", "acc_m", None, None),
    ("FastVGGT Comp (m)", "fastvggt_reconstruction", "comp_m", None, None),
    ("FastVGGT Overall (m)", "fastvggt_reconstruction", "overall_m", None, None),
    ("FastVGGT CD (m)", "fastvggt_reconstruction", "cd_m", None, "CD (FastVGGT pipeline, m)"),
    ("FastVGGT Precision@5cm", "fastvggt_reconstruction", "precision_at_0.05m", None, None),
    ("FastVGGT Recall@5cm", "fastvggt_reconstruction", "recall_at_0.05m", None, None),
    ("FastVGGT F1@5cm", "fastvggt_reconstruction", "f1_at_0.05m", None, None),
    ("FastVGGT NC", "fastvggt_reconstruction", "nc", None, None),
    ("ATE (FastVGGT, m)", "fastvggt_pose", "fastvggt_ate_m", "fastvggt_pose.fastvggt_ate_m", None),
    ("ARE (FastVGGT, deg)", "fastvggt_pose", "fastvggt_are_deg", "fastvggt_pose.fastvggt_are_deg", None),
    ("RPE-rot (FastVGGT, deg)", "fastvggt_pose", "fastvggt_rpe_rot_deg", "fastvggt_pose.fastvggt_rpe_rot_deg", None),
    ("RPE-trans (FastVGGT, m)", "fastvggt_pose", "fastvggt_rpe_trans_m", "fastvggt_pose.fastvggt_rpe_trans_m", None),
    ("Latency (model-only, s)", "efficiency", "latency_s", "efficiency.latency_s", "Latency (model-only, s)"),
    ("FPS (model-only)", "efficiency", "fps", "efficiency.fps", None),
    ("VRAM allocated (GiB)", "efficiency", "peak_vram_allocated_gib", "efficiency.peak_vram_allocated_gib", None),
)


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
        actual[method] = {name: mean([metric_value(item, section, field) for item in results])
                          for name, section, field, _, _ in METRICS}

    rows = []
    lines = [
        "# ScanNet-50 fairness smoke test (100 frames, two scenes)", "",
        "Scenes: `scene0000_00`, `scene0013_02`. Actual values are arithmetic means over these two scenes.", "",
        "Both reconstruction protocols are reported separately: `Project` uses this repository's reservoir/bbox/voxel route, while `FastVGGT` uses full-cloud bbox alignment followed by FastVGGT's deterministic 100k sample and voxelization.",
        "The local historical reference is populated only for the Project protocol and pose/efficiency fields. The paper reference is populated only for the published FastVGGT CD and latency. Blank cells mean that no same-protocol reference exists.",
        "The local and paper references remain diagnostic rather than pass/fail thresholds because the current run is a two-scene fairness smoke test.", "",
    ]
    for method in METHODS:
        lines.extend([f"## {method}", "", "| Metric | Actual fairness | Local historical reference | Δ actual−local | FastVGGT paper reference | Δ actual−paper |", "|---|---:|---:|---:|---:|---:|"])
        local = reference["local_historical"]["methods"].get(method, {})
        paper = reference["fastvggt_paper"]["methods"].get(method, {})
        for metric, _, _, local_key, paper_key in METRICS:
            value = actual[method][metric]
            local_value = None if local_key is None else local.get(local_key)
            paper_value = None if paper_key is None else paper.get(paper_key)
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
