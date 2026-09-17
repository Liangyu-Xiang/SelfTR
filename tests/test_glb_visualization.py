"""Regression tests for the browser-compatible GLB exporter."""

import numpy as np
import pytest

pytest.importorskip("trimesh")
from visual_util import point_cloud_to_glb, predictions_to_glb


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


def test_point_cloud_to_glb_supports_a_bounded_evaluation_sample():
    scene = point_cloud_to_glb(
        vertices=np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float32),
        colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        camera_to_world=np.eye(4, dtype=np.float32)[None],
    )
    assert len(scene.geometry) == 2
    assert scene.export(file_type="glb").startswith(b"glTF")
