"""Regression coverage for the released FastVGGT ScanNet sampling/preprocessing path."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("open3d")
pytest.importorskip("evo")
torch = pytest.importorskip("torch")

from scripts.eval_scannet50 import fastvggt_frame_indices, load_fastvggt_images


def released_build_frame_selection(length: int, requested: int) -> np.ndarray:
    """The relevant released FastVGGT rule over an already valid RGB/pose pool."""
    valid = list(range(length))
    if len(valid) <= requested:
        return np.asarray(valid, dtype=np.int64)
    first, remaining = valid[0], valid[1:]
    stride = max(1, len(remaining) // (requested - 1))
    return np.asarray([first, *remaining[::stride]][:requested], dtype=np.int64)


@pytest.mark.parametrize("length,requested", ((350, 100), (700, 300), (1200, 500), (2500, 1000), (350, 1000)))
def test_fairness_sampler_matches_released_fastvggt(length: int, requested: int):
    actual = fastvggt_frame_indices(length, requested)
    expected = released_build_frame_selection(length, requested)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == 0
    assert len(actual) == min(length, requested)


def test_fairness_preprocessing_applies_fastvggt_center_crop(tmp_path: Path):
    """Tall RGB inputs become FastVGGT's centred 518x518 crop after resize."""
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    image[..., 0] = np.arange(200, dtype=np.uint8)[:, None]
    path = tmp_path / "0.jpg"
    Image.fromarray(image).save(path)

    fairness = load_fastvggt_images([path], fairness_fastvggt_protocol=True)
    project = load_fastvggt_images([path], fairness_fastvggt_protocol=False)
    assert tuple(fairness.shape) == (1, 1, 3, 518, 518)
    assert tuple(project.shape) == (1, 1, 3, 1036, 518)
    # Centre crop starts at row 259 of the resized image, hence it must differ
    # from the native top row while preserving a non-zero vertical gradient.
    assert not torch.allclose(fairness[0, 0, :, 0], project[0, 0, :, 0])
