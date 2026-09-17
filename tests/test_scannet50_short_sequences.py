"""Compatibility checks for ScanNet scenes shorter than the requested count."""

from scripts.summarize_scannet50 import compatible_frame_count


def test_short_sequence_is_valid_when_it_records_the_requested_count():
    item = {"frames": 713, "requested_frames": 1000, "frame_ids": list(range(713))}
    assert compatible_frame_count(item, requested=1000)


def test_legacy_full_length_result_remains_resume_compatible():
    # Existing completed runs predate ``requested_frames`` but have frames=N.
    item = {"frames": 500, "frame_ids": list(range(500))}
    assert compatible_frame_count(item, requested=500)


def test_invalid_short_sequence_metadata_is_rejected():
    item = {"frames": 713, "requested_frames": 500, "frame_ids": list(range(713))}
    assert not compatible_frame_count(item, requested=1000)
