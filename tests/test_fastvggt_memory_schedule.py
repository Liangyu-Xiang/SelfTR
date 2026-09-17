"""Regression coverage for FastVGGT's memory-only execution changes."""

import torch
from torch import nn
import torch.nn.functional as F

from vggt.models.acceleration import (
    _fastvggt_reference_bipartite_merge,
    fastvggt_reference_attention,
)


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.head_dim = 4
        self.qkv = nn.Linear(8, 24, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.rope = None
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(8, 8, bias=False)
        self.proj_drop = nn.Dropout(0.0)


def _pre_cleanup_reference(attention, x, *, width, height, merge_ratio):
    """The prior implementation, retained here solely for output comparison."""
    batch, token_count, channels = x.shape
    qkv = attention.qkv(x).reshape(
        batch, token_count, 3, attention.num_heads, attention.head_dim
    ).permute(2, 0, 3, 1, 4)
    query, key, value = attention.q_norm(qkv[0]), attention.k_norm(qkv[1]), qkv[2]
    generator = torch.Generator(device=x.device)
    generator.manual_seed(33)
    merge, unmerge, _ = _fastvggt_reference_bipartite_merge(
        x, width, height, int(token_count * merge_ratio), generator
    )
    query_in = query.permute(0, 2, 1, 3).reshape(batch, token_count, channels)
    key_in = key.permute(0, 2, 1, 3).reshape(batch, token_count, channels)
    value_in = value.permute(0, 2, 1, 3).reshape(batch, token_count, channels)
    query_out, key_out, value_out = merge(
        query_in, mode="mean", extra_tensors=key_in, extra_tensors_2=value_in
    )
    query = query_out.reshape(batch, -1, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    key = key_out.reshape(batch, -1, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    value = value_out.reshape(batch, -1, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    output = attention.proj_drop(attention.proj(output.transpose(1, 2).reshape(batch, -1, channels)))
    return unmerge(output)


def test_fastvggt_cleanup_schedule_is_numerically_identical():
    torch.manual_seed(7)
    attention = _TinyAttention().eval()
    # Three 2x2 images: (2 * 2 + 5) * 3 = 27 tokens.
    x = torch.randn(1, 27, 8)

    expected = _pre_cleanup_reference(attention, x, width=2, height=2, merge_ratio=0.5)
    actual = fastvggt_reference_attention(
        attention, x, None, patch_width=2, patch_height=2, merge_ratio=0.5
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
