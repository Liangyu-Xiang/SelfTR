#!/usr/bin/env python3
"""Build the cross-method ScanNet50 fairness table used by the paper.

``summarize_scannet50.py`` remains deliberately single-method.  This script
only consumes its completed 50-scene summaries, so the speed-up denominator is
always the DenseVGGT mean measured under the same frame-count/protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("densevggt", "fastvggt", "selftr")
PROTOCOL_ID = "fastvggt_fairness_v3"


def read_metric(payload: dict[str, Any], section: str, key: str) -> float | None:
    value = payload.get(section, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def retention(payload: dict[str, Any]) -> dict[str, Any]:
    token = payload.get("token_retention", {})
    fixed = token.get("retention_percent")
    if isinstance(fixed, (int, float)):
        return {"kind": "fixed_once", "mean_percent": float(fixed), "stages_percent": []}
    stages = [float(stage["mean_retention_percent"]) for stage in token.get("stages", [])
              if isinstance(stage.get("mean_retention_percent"), (int, float))]
    return {"kind": "selftr_stagewise", "mean_percent": float(np.mean(stages)) if stages else None,
            "stages_percent": stages}


def method_row(payload: dict[str, Any], method: str, frames: int, dense_latency: float) -> dict[str, Any]:
    latency = read_metric(payload, "mean_efficiency", "latency_s")
    if latency is None or latency <= 0:
        raise RuntimeError(f"{method}_{frames}: missing positive mean_efficiency.latency_s")
    return {
        "frames": frames,
        "method": method,
        "AUC@3 (%)": read_metric(payload, "mean_pose", "auc_at_3_percent"),
        "AUC@30 (%)": read_metric(payload, "mean_pose", "auc_at_30_percent"),
        # Fairness reconstruction values deliberately come from the released
        # FastVGGT cloud path, not the project's diagnostic reference path.
        "Acc (m)": read_metric(payload, "mean_fastvggt_reconstruction", "acc_m"),
        "Comp (m)": read_metric(payload, "mean_fastvggt_reconstruction", "comp_m"),
        "NC": read_metric(payload, "mean_fastvggt_reconstruction", "normal_consistency"),
        "CD (m)": read_metric(payload, "mean_fastvggt_reconstruction", "cd_m"),
        "Latency (s)": latency,
        "FPS": read_metric(payload, "mean_efficiency", "fps"),
        "Peak VRAM allocated (GiB)": read_metric(payload, "mean_efficiency", "peak_vram_allocated_gib"),
        "Token retention": retention(payload),
        "Spd. (x)": dense_latency / latency,
    }


def display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def display_retention(value: dict[str, Any]) -> str:
    stages = value["stages_percent"]
    if stages:
        return "/".join(f"{stage:.1f}" for stage in stages) + "%"
    return display(value["mean_percent"], 1) + "%"


def validate_summary(payload: dict[str, Any], path: Path, method: str, frames: int) -> None:
    if payload.get("method") != method or payload.get("num_frames_requested") != frames:
        raise RuntimeError(f"stale summary at {path}: expected {method}, {frames} frames")
    if payload.get("scene_count") != 50 or payload.get("failed_scene_count") != 0:
        raise RuntimeError(f"incomplete fairness summary at {path}: expected 50 successful scenes")
    scenes = payload.get("scenes", [])
    if len(scenes) != 50 or any(scene.get("protocol_id") != PROTOCOL_ID for scene in scenes):
        raise RuntimeError(f"invalid fairness protocol scenes at {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for frames in args.frames:
        summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            path = args.output_root / f"{method}_{frames}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            validate_summary(payload, path, method, frames)
            summaries[method] = payload
        dense_latency = read_metric(summaries["densevggt"], "mean_efficiency", "latency_s")
        if dense_latency is None or dense_latency <= 0:
            raise RuntimeError(f"densevggt_{frames}: missing positive latency")
        all_rows.extend(method_row(summaries[method], method, frames, dense_latency) for method in METHODS)

    json_path = args.output_root / "fairness_paper_report.json"
    json_path.write_text(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "speed_definition": "DenseVGGT mean latency / method mean latency at the same frame count",
        "reconstruction_protocol": "released FastVGGT full-cloud bbox-aligned reconstruction path",
        "rows": all_rows,
    }, indent=2, sort_keys=True) + "\n")

    columns = ("Frames", "Method", "AUC@3", "AUC@30", "Acc", "Comp", "NC", "CD", "Latency (s)",
               "FPS", "VRAM (GiB)", "Retention", "Spd.")
    markdown = ["# ScanNet50 FastVGGT fairness report", "",
                "Reconstruction columns use `mean_fastvggt_reconstruction`; `Spd.` is DenseVGGT mean latency divided by the method mean latency at the same frame count.",
                "", "| " + " | ".join(columns) + " |",
                "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in all_rows:
        markdown.append("| " + " | ".join((
            str(row["frames"]), row["method"], display(row["AUC@3 (%)"]), display(row["AUC@30 (%)"]),
            display(row["Acc (m)"]), display(row["Comp (m)"]), display(row["NC"]), display(row["CD (m)"]),
            display(row["Latency (s)"]), display(row["FPS"]), display(row["Peak VRAM allocated (GiB)"]),
            display_retention(row["Token retention"]), display(row["Spd. (x)"]),
        )) + " |")
    markdown_path = args.output_root / "fairness_paper_report.md"
    markdown_path.write_text("\n".join(markdown) + "\n")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "rows": len(all_rows)}))


if __name__ == "__main__":
    main()
