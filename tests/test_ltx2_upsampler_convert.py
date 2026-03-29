"""Tests for LTX-2 upsampler conversion and loading."""

from pathlib import Path

import mlx.core as mx

import mlx_video.models.ltx_2.convert as convert_module
import mlx_video.models.ltx_2.upsampler as upsampler_module


class TestSanitizeUpsamplerWeights:
    def test_transposes_and_renames_pytorch_layout(self):
        weights = {
            "upsampler.0.weight": mx.zeros((128, 32, 3, 3), dtype=mx.float32),
            "res_blocks.0.conv1.weight": mx.zeros(
                (32, 32, 3, 3, 3), dtype=mx.float32
            ),
            "upsampler.blur_down.kernel": mx.zeros((1, 1, 5, 5), dtype=mx.float32),
            "res_blocks.0.conv1.bias": mx.zeros((32,), dtype=mx.float32),
        }

        out = upsampler_module.sanitize_upsampler_weights(weights)

        assert "upsampler.conv.weight" in out
        assert out["upsampler.conv.weight"].shape == (128, 3, 3, 32)
        assert out["res_blocks.0.conv1.weight"].shape == (32, 3, 3, 3, 32)
        assert out["upsampler.blur_down.kernel"].shape == (1, 5, 5, 1)
        assert out["res_blocks.0.conv1.bias"].shape == (32,)

    def test_is_idempotent_for_mlx_layout(self):
        weights = {
            "upsampler.conv.weight": mx.zeros((128, 3, 3, 32), dtype=mx.float32),
            "res_blocks.0.conv1.weight": mx.zeros(
                (32, 3, 3, 3, 32), dtype=mx.float32
            ),
            "upsampler.blur_down.kernel": mx.zeros((1, 5, 5, 1), dtype=mx.float32),
        }

        out = upsampler_module.sanitize_upsampler_weights(weights)

        assert out["upsampler.conv.weight"].shape == (128, 3, 3, 32)
        assert out["res_blocks.0.conv1.weight"].shape == (32, 3, 3, 3, 32)
        assert out["upsampler.blur_down.kernel"].shape == (1, 5, 5, 1)


class TestLoadUpsampler:
    def test_load_upsampler_sanitizes_pytorch_layout_at_runtime(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "res_blocks.0.conv1.weight": mx.zeros(
                    (32, 32, 3, 3, 3), dtype=mx.float32
                ),
                "upsampler.0.weight": mx.zeros((128, 32, 3, 3), dtype=mx.float32),
            },
        )

        captured = {}

        class FakeUpsampler:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            def load_weights(self, items, strict=False):
                captured["weights"] = dict(items)
                captured["strict"] = strict

        monkeypatch.setattr(upsampler_module, "LatentUpsampler", FakeUpsampler)

        model, spatial_scale = upsampler_module.load_upsampler(str(path))

        assert isinstance(model, FakeUpsampler)
        assert spatial_scale == 2.0
        assert captured["init"]["mid_channels"] == 32
        assert captured["weights"]["upsampler.conv.weight"].shape == (128, 3, 3, 32)
        assert captured["weights"]["res_blocks.0.conv1.weight"].shape == (
            32,
            3,
            3,
            3,
            32,
        )
        assert captured["strict"] is False

    def test_load_upsampler_skips_runtime_sanitize_for_mlx_layout(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "res_blocks.0.conv1.weight": mx.zeros(
                    (32, 3, 3, 3, 32), dtype=mx.float32
                ),
                "upsampler.conv.weight": mx.zeros((128, 3, 3, 32), dtype=mx.float32),
            },
        )

        captured = {}

        class FakeUpsampler:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            def load_weights(self, items, strict=False):
                captured["weights"] = dict(items)
                captured["strict"] = strict

        monkeypatch.setattr(upsampler_module, "LatentUpsampler", FakeUpsampler)
        monkeypatch.setattr(
            upsampler_module,
            "sanitize_upsampler_weights",
            lambda _weights: (_ for _ in ()).throw(
                AssertionError("sanitize_upsampler_weights should not run")
            ),
        )

        _, spatial_scale = upsampler_module.load_upsampler(str(path))

        assert spatial_scale == 2.0
        assert captured["init"]["mid_channels"] == 32
        assert captured["weights"]["upsampler.conv.weight"].shape == (128, 3, 3, 32)
        assert captured["strict"] is False

    def test_load_upsampler_accepts_offline_converted_weights(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "res_blocks.0.conv1.weight": mx.zeros(
                    (32, 3, 3, 3, 32), dtype=mx.float32
                ),
                "upsampler.conv.weight": mx.zeros((128, 3, 3, 32), dtype=mx.float32),
            },
        )

        captured = {}

        class FakeUpsampler:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            def load_weights(self, items, strict=False):
                captured["weights"] = dict(items)
                captured["strict"] = strict

        monkeypatch.setattr(upsampler_module, "LatentUpsampler", FakeUpsampler)

        _, spatial_scale = upsampler_module.load_upsampler(str(path))

        assert spatial_scale == 2.0
        assert captured["init"]["mid_channels"] == 32
        assert captured["weights"]["upsampler.conv.weight"].shape == (128, 3, 3, 32)
        assert captured["weights"]["res_blocks.0.conv1.weight"].shape == (
            32,
            3,
            3,
            3,
            32,
        )
        assert captured["strict"] is False


class TestUpsamplerSanitizeDetection:
    def test_detects_raw_pytorch_layout(self):
        weights = {
            "upsampler.0.weight": mx.zeros((128, 32, 3, 3), dtype=mx.float32),
            "res_blocks.0.conv1.weight": mx.zeros(
                (32, 32, 3, 3, 3), dtype=mx.float32
            ),
        }

        assert upsampler_module.upsampler_weights_need_sanitization(weights) is True

    def test_detects_converted_mlx_layout(self):
        weights = {
            "upsampler.conv.weight": mx.zeros((128, 3, 3, 32), dtype=mx.float32),
            "res_blocks.0.conv1.weight": mx.zeros(
                (32, 3, 3, 3, 32), dtype=mx.float32
            ),
        }

        assert upsampler_module.upsampler_weights_need_sanitization(weights) is False


class TestConvertUpscalerArtifacts:
    def test_convert_sanitizes_upscaler_file_before_saving(self, tmp_path, monkeypatch):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        monolithic = source_dir / "ltx-2.3-22b-distilled.safetensors"
        monolithic.touch()
        upscaler = source_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        upscaler.touch()

        saved = {}

        def fake_load(path: str):
            path_obj = Path(path)
            if path_obj == monolithic:
                return {
                    "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": mx.zeros(
                        (1, 1), dtype=mx.float32
                    )
                }
            if path_obj == upscaler:
                return {
                    "res_blocks.0.conv1.weight": mx.zeros(
                        (32, 32, 3, 3, 3), dtype=mx.float32
                    ),
                    "upsampler.0.weight": mx.zeros(
                        (128, 32, 3, 3), dtype=mx.float32
                    ),
                }
            raise AssertionError(f"Unexpected load path: {path}")

        def fake_save_safetensors(path: str, weights):
            saved[Path(path).name] = weights

        monkeypatch.setattr(convert_module, "resolve_source", lambda *_args, **_kwargs: monolithic)
        monkeypatch.setattr(convert_module.mx, "load", fake_load)
        monkeypatch.setattr(convert_module.mx, "save_safetensors", fake_save_safetensors)
        monkeypatch.setattr(convert_module, "save_sharded", lambda *_args, **_kwargs: 1)
        monkeypatch.setattr(convert_module, "save_single", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(convert_module, "save_config", lambda *_args, **_kwargs: None)

        convert_module.convert(str(source_dir), tmp_path / "out", variant="distilled")

        converted = saved[upscaler.name]
        assert converted["upsampler.conv.weight"].shape == (128, 3, 3, 32)
        assert converted["res_blocks.0.conv1.weight"].shape == (32, 3, 3, 3, 32)