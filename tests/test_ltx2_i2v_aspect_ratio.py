"""Regression tests for LTX-2 I2V aspect-ratio-preserving image preprocessing."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_video.utils import load_image, prepare_image_for_encoding


def _make_source_image() -> Image.Image:
    return Image.new("RGB", (200, 100), color=(255, 128, 64))


def test_load_image_letterboxes_to_target_box(tmp_path):
    source_path = Path(tmp_path) / "source.png"
    _make_source_image().save(source_path)

    image = load_image(source_path, height=128, width=128)
    image_np = np.array(image)

    assert image_np.shape == (128, 128, 3)
    assert np.all(image_np[:32] == 0)
    assert np.all(image_np[-32:] == 0)

    content_mask = image_np.sum(axis=-1) > 0
    ys, xs = np.where(content_mask)
    assert ys.min() == 32
    assert ys.max() == 95
    assert xs.min() == 0
    assert xs.max() == 127


def test_prepare_image_for_encoding_letterboxes_to_target_box():
    source_image = np.array(_make_source_image()).astype(np.float32) / 255.0
    encoded = prepare_image_for_encoding(mx.array(source_image), 128, 128)
    encoded_np = np.array(encoded)

    assert encoded_np.shape == (1, 3, 1, 128, 128)
    content = encoded_np[0, :, 0]

    assert np.allclose(content[:, :32, :], -1.0)
    assert np.allclose(content[:, -32:, :], -1.0)
    assert np.any(content[:, 32:96, :] > -1.0)
