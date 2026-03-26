"""Tests for LTX attention helpers and query chunking."""

import mlx.core as mx
import numpy as np

from mlx_video.models.ltx_2.attention import Attention, scaled_dot_product_attention


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
