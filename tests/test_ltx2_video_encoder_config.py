"""Tests for LTX-2 VideoEncoder config inference and stale-config recovery."""

import json

import mlx.core as mx
import numpy as np

from mlx_video.models.ltx_2.config import VideoEncoderModelConfig
from mlx_video.models.ltx_2.convert import infer_vae_encoder_config
from mlx_video.models.ltx_2.video_vae.video_vae import VideoEncoder


ACTUAL_ENCODER_BLOCKS = [
    ("res_x", {"num_layers": 4}),
    ("compress_space_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 4}),
    ("compress_all_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
    ("compress_all_res", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
]

BROKEN_ENCODER_BLOCKS = [
    ("res_x", {"num_layers": 4}),
    ("compress_space_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_all_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
    ("compress_all_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
]


def _flatten_params(params, prefix=""):
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _build_encoder() -> VideoEncoder:
    mx.random.seed(42)
    model = VideoEncoder(VideoEncoderModelConfig(encoder_blocks=ACTUAL_ENCODER_BLOCKS))
    mx.eval(model.parameters())
    return model


def test_infer_vae_encoder_config_matches_weight_layout():
    model = _build_encoder()
    weights = _flatten_params(dict(model.parameters()))

    config = infer_vae_encoder_config(weights)

    assert config["encoder_blocks"] == [list(block) for block in ACTUAL_ENCODER_BLOCKS]


def test_video_encoder_from_pretrained_repairs_stale_config(tmp_path):
    model = _build_encoder()
    weights = _flatten_params(dict(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    broken_config = VideoEncoderModelConfig(
        encoder_blocks=BROKEN_ENCODER_BLOCKS
    ).to_dict()
    (tmp_path / "config.json").write_text(json.dumps(broken_config), encoding="utf-8")

    repaired = VideoEncoder.from_pretrained(tmp_path)

    sample = mx.random.normal((1, 3, 1, 64, 64))
    mx.eval(sample)

    expected = model(sample)
    actual = repaired(sample)
    mx.eval(expected, actual)

    np.testing.assert_allclose(
        np.array(actual),
        np.array(expected),
        rtol=1e-5,
        atol=1e-5,
    )