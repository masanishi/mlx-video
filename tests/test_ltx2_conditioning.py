"""Unit tests for LTX latent conditioning helpers."""

import mlx.core as mx
import numpy as np

from mlx_video.models.ltx_2.conditioning.latent import (
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
)


def test_apply_conditioning_replaces_only_target_frame_span():
    latent = mx.zeros((1, 2, 4, 1, 1), dtype=mx.float32)
    clean = mx.full((1, 2, 4, 1, 1), -1.0, dtype=mx.float32)
    mask = mx.ones((1, 1, 4, 1, 1), dtype=mx.float32)
    state = LatentState(latent=latent, clean_latent=clean, denoise_mask=mask)

    cond_latent = mx.full((1, 2, 1, 1, 1), 5.0, dtype=mx.float32)
    conditioned = apply_conditioning(
        state,
        [VideoConditionByLatentIndex(latent=cond_latent, frame_idx=2, strength=0.25)],
    )

    conditioned_latent = np.array(conditioned.latent)
    conditioned_clean = np.array(conditioned.clean_latent)
    conditioned_mask = np.array(conditioned.denoise_mask)

    assert conditioned_latent.shape == (1, 2, 4, 1, 1)
    assert np.allclose(conditioned_latent[:, :, 0:2], 0.0)
    assert np.allclose(conditioned_latent[:, :, 2:3], 5.0)
    assert np.allclose(conditioned_latent[:, :, 3:4], 0.0)

    assert np.allclose(conditioned_clean[:, :, 0:2], -1.0)
    assert np.allclose(conditioned_clean[:, :, 2:3], 5.0)
    assert np.allclose(conditioned_clean[:, :, 3:4], -1.0)

    assert np.allclose(conditioned_mask[:, :, 0:2], 1.0)
    assert np.allclose(conditioned_mask[:, :, 2:3], 0.75)
    assert np.allclose(conditioned_mask[:, :, 3:4], 1.0)


def test_apply_conditioning_truncates_conditioning_past_sequence_end():
    latent = mx.zeros((1, 1, 3, 1, 1), dtype=mx.float32)
    clean = mx.zeros((1, 1, 3, 1, 1), dtype=mx.float32)
    mask = mx.ones((1, 1, 3, 1, 1), dtype=mx.float32)
    state = LatentState(latent=latent, clean_latent=clean, denoise_mask=mask)

    cond_latent = mx.array([[[[[1.0]], [[2.0]]]]], dtype=mx.float32)
    conditioned = apply_conditioning(
        state,
        [VideoConditionByLatentIndex(latent=cond_latent, frame_idx=2, strength=1.0)],
    )

    conditioned_latent = np.array(conditioned.latent)
    conditioned_mask = np.array(conditioned.denoise_mask)

    assert conditioned_latent.shape == (1, 1, 3, 1, 1)
    assert np.allclose(conditioned_latent[:, :, 0:2], 0.0)
    assert np.allclose(conditioned_latent[:, :, 2:3], 1.0)
    assert np.allclose(conditioned_mask[:, :, 2:3], 0.0)
