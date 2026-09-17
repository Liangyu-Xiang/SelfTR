"""Regression coverage for paper-table field names."""

from scripts.report_scannet50_fairness import method_row


def test_paper_report_reads_fastvggt_normal_consistency_from_nc():
    payload = {
        "mean_pose": {"auc_at_3_percent": 10.0, "auc_at_30_percent": 20.0},
        "mean_fastvggt_reconstruction": {
            "acc_m": 0.1, "comp_m": 0.2, "nc": 0.75, "cd_m": 0.3,
        },
        "mean_efficiency": {"latency_s": 2.0, "fps": 50.0, "peak_vram_allocated_gib": 4.0},
        "token_retention": {"policy": "fixed_once", "retention_percent": 100.0},
    }

    row = method_row(payload, "densevggt", 100, dense_latency=2.0)

    assert row["NC"] == 0.75
