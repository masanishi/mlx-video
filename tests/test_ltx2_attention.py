"""Tests for LTX attention helpers and query chunking."""

import mlx.core as mx
import numpy as np

from mlx_video.models.ltx_2.attention import Attention, scaled_dot_product_attention
from mlx_video.models.ltx_2.config import LTXRopeType
from mlx_video.models.ltx_2.rope import precompute_freqs_cis


def _create_positions(batch_size: int, tokens: int) -> mx.array:
    coords = np.arange(tokens, dtype=np.float32)
    starts = np.stack([coords, np.zeros_like(coords), np.zeros_like(coords)], axis=0)
    ends = starts.copy()
    ends[0] += 1
    ends[1] += 1
    ends[2] += 1
    latent_coords = np.stack([starts, ends], axis=-1)
    latent_coords = np.tile(latent_coords[None, ...], (batch_size, 1, 1, 1))
    scale_factors = np.array([8, 32, 32], dtype=np.float32).reshape(1, 3, 1, 1)
    pixel_coords = latent_coords * scale_factors
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / 24.0
    return mx.array(pixel_coords, dtype=mx.float32)


def test_scaled_dot_product_attention_query_chunking_matches_full_attention():
    mx.random.seed(0)
    q = mx.random.normal((1, 7, 16), dtype=mx.float32)
    k = mx.random.normal((1, 7, 16), dtype=mx.float32)
    v = mx.random.normal((1, 7, 16), dtype=mx.float32)

    full = scaled_dot_product_attention(q, k, v, heads=4)
    chunked = scaled_dot_product_attention(q, k, v, heads=4, query_chunk_size=3)
    mx.eval(full, chunked)

    np.testing.assert_allclose(np.array(chunked), np.array(full), rtol=1e-5, atol=1e-5)


def test_scaled_dot_product_attention_query_chunking_matches_with_query_mask():
    mx.random.seed(1)
    q = mx.random.normal((1, 6, 8), dtype=mx.float32)
    k = mx.random.normal((1, 6, 8), dtype=mx.float32)
    v = mx.random.normal((1, 6, 8), dtype=mx.float32)

    mask = mx.zeros((1, 1, 6, 6), dtype=mx.float32)
    mask[:, :, 4:, :] = -1e9

    full = scaled_dot_product_attention(q, k, v, heads=2, mask=mask)
    chunked = scaled_dot_product_attention(
        q, k, v, heads=2, mask=mask, query_chunk_size=2
    )
    mx.eval(full, chunked)

    np.testing.assert_allclose(np.array(chunked), np.array(full), rtol=1e-5, atol=1e-5)


def test_attention_module_uses_query_chunk_size_without_changing_output_shape():
    attn = Attention(query_dim=8, heads=2, dim_head=4)
    attn.query_chunk_size = 2

    x = mx.random.normal((1, 5, 8), dtype=mx.float32)
    out = attn(x)
    mx.eval(out)

    assert out.shape == (1, 5, 8)


def test_attention_module_query_chunking_matches_full_output():
    mx.random.seed(7)
    attn = Attention(query_dim=8, heads=2, dim_head=4)
    x = mx.random.normal((1, 6, 8), dtype=mx.float32)

    full = attn(x)
    attn.query_chunk_size = 2
    chunked = attn(x)
    mx.eval(full, chunked)

    np.testing.assert_allclose(np.array(chunked), np.array(full), rtol=1e-5, atol=1e-5)


def test_attention_module_query_chunking_matches_full_output_with_split_rope():
    mx.random.seed(9)
    attn = Attention(query_dim=8, heads=2, dim_head=4, rope_type=LTXRopeType.SPLIT)
    x = mx.random.normal((1, 6, 8), dtype=mx.float32)
    positions = _create_positions(batch_size=1, tokens=6)
    pe = precompute_freqs_cis(
        positions,
        dim=8,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        num_attention_heads=2,
        rope_type=LTXRopeType.SPLIT,
        double_precision=True,
    )

    full = attn(x, pe=pe)
    attn.query_chunk_size = 2
    chunked = attn(x, pe=pe)
    mx.eval(full, chunked)

    np.testing.assert_allclose(np.array(chunked), np.array(full), rtol=1e-5, atol=1e-5)
