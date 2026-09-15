import pytest
import torch

from vggt.models.um_triton import fused_um_edge_cost


def test_fused_um_edge_cost_falls_back_on_cpu():
    features = torch.eye(2, dtype=torch.float32)
    assert fused_um_edge_cost(
        features,
        torch.ones(2),
        torch.arange(2),
        torch.zeros(2),
        features,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([True]),
        prefer_best_parent=True,
    ) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_um_edge_cost_matches_pytorch_reference(monkeypatch):
    monkeypatch.setenv("SELFTR_UM_TRITON", "1")
    generator = torch.Generator(device="cuda").manual_seed(7)
    groups, dim, edges = 37, 1024, 257
    features = torch.nn.functional.normalize(
        torch.randn(groups, dim, generator=generator, device="cuda"), dim=-1
    )
    weights = torch.rand(groups, generator=generator, device="cuda") * 4.0 + 1.0
    sums = features * weights[:, None]
    representatives = torch.arange(groups, device="cuda")
    errors = weights - (sums * features).sum(dim=-1)
    left = torch.randint(groups, (edges,), generator=generator, device="cuda")
    right = torch.randint(groups, (edges,), generator=generator, device="cuda")
    valid = left != right

    actual = fused_um_edge_cost(
        sums, weights, representatives, errors, features, left, right, valid,
        prefer_best_parent=True,
    )
    assert actual is not None
    merged_sum = sums[left] + sums[right]
    merged_weight = weights[left] + weights[right]
    left_error = merged_weight - (merged_sum * features[representatives[left]]).sum(dim=-1)
    right_error = merged_weight - (merged_sum * features[representatives[right]]).sum(dim=-1)
    expected = (
        torch.minimum(left_error, right_error) - errors[left] - errors[right]
    ).masked_fill(~valid, float("inf"))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
