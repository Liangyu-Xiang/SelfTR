"""Regression tests for the browser-compatible GLB exporter."""

import numpy as np
import pytest

pytest.importorskip("trimesh")
from visual_util import predictions_to_glb


def test_predictions_to_glb_accepts_batched_vggt_predictions():
    """Original VGGT's ``world_points`` output exports without Omega shims."""
    points = np.array(
        [
            [
                [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
                [[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]]],
            ]
        ],
        dtype=np.float32,
    )
    predictions = {
        "world_points": points,
        "world_points_conf": np.ones((1, 2, 2, 2), dtype=np.float32),
        "images": np.full((1, 2, 3, 2, 2), 0.5, dtype=np.float32),
        "extrinsic": np.array(
            [[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
              [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]],
            dtype=np.float32,
        ),
    }

    scene = predictions_to_glb(predictions, conf_thres=0, filter_depth_edges=False, max_points=0)
    point_cloud = next(geometry for geometry in scene.geometry.values() if geometry.__class__.__name__ == "PointCloud")

    assert len(point_cloud.vertices) == 8
    assert len(scene.geometry) == 3  # One point cloud plus one frustum per frame.
    assert scene.export(file_type="glb").startswith(b"glTF")
