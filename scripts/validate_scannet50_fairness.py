#!/usr/bin/env python3
"""Reject incomplete or non-identical inputs in a ScanNet-50 fairness run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SCANNET50_SCENES = {
    "scene0000_00", "scene0013_02", "scene0029_01", "scene0042_02", "scene0056_00", "scene0071_00", "scene0084_01", "scene0096_00", "scene0109_00", "scene0121_01",
    "scene0136_01", "scene0150_00", "scene0164_01", "scene0177_01", "scene0194_00", "scene0207_01", "scene0221_01", "scene0238_00", "scene0254_01", "scene0267_00",
    "scene0280_00", "scene0294_02", "scene0309_00", "scene0325_01", "scene0340_01", "scene0353_02", "scene0367_01", "scene0380_02", "scene0395_00", "scene0409_01",
    "scene0421_02", "scene0435_03", "scene0451_01", "scene0466_01", "scene0477_00", "scene0493_01", "scene0509_01", "scene0525_00", "scene0540_02", "scene0555_00",
    "scene0571_00", "scene0582_02", "scene0593_00", "scene0606_01", "scene0619_00", "scene0631_01", "scene0648_00", "scene0663_01", "scene0675_00", "scene0691_00",
}
PROTOCOL_ID = "fastvggt_fairness_v3"


def compatible_frame_count(item: dict, requested: int) -> bool:
    """Accept an all-available short scene while retaining legacy full runs."""
    actual = item.get("frames")
    recorded_requested = item.get("requested_frames", actual)
    frame_ids = item.get("frame_ids")
    return (isinstance(actual, int) and 1 <= actual <= requested
            and recorded_requested == requested
            and isinstance(frame_ids, list) and len(frame_ids) == actual)


def load_run(directory: Path, method: str, frames: int) -> dict[str, dict]:
    failures = list(directory.glob("scene*/failure.json"))
    metrics_paths = sorted(directory.glob("scene*/metrics.json"))
    items = {path.parent.name: json.loads(path.read_text()) for path in metrics_paths}
    if failures:
        raise RuntimeError(f"{directory}: {len(failures)} failed scene(s), e.g. {failures[0]}")
    if set(items) != SCANNET50_SCENES:
        missing, extra = sorted(SCANNET50_SCENES - set(items)), sorted(set(items) - SCANNET50_SCENES)
        raise RuntimeError(f"{directory}: expected exactly 50 ScanNet scenes; missing={missing[:3]}, extra={extra[:3]}")
    for scene, item in items.items():
        if (item.get("protocol_id") != PROTOCOL_ID or item.get("method") != method
                or not compatible_frame_count(item, frames)):
            raise RuntimeError(f"{directory}/{scene}: stale or incompatible per-scene result")
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[100, 300, 500, 1000])
    parser.add_argument("--methods", nargs="+", default=["densevggt", "fastvggt", "selftr"])
    args = parser.parse_args()

    checked = {}
    for frames in args.frames:
        reference_ids = None
        for method in args.methods:
            items = load_run(args.output_root / f"{method}_{frames}", method, frames)
            frame_ids = {scene: item["frame_ids"] for scene, item in items.items()}
            if reference_ids is None:
                reference_ids = frame_ids
            elif frame_ids != reference_ids:
                mismatch = next(scene for scene in SCANNET50_SCENES if frame_ids[scene] != reference_ids[scene])
                raise RuntimeError(f"{frames} frames: {method} has different input frame IDs for {mismatch}")
            checked[f"{method}_{frames}"] = len(items)
    payload = {"protocol_id": PROTOCOL_ID, "validated": checked}
    output = args.output_root / "fairness_validation.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "validated_runs": len(checked)}))


if __name__ == "__main__":
    main()
