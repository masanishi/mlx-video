"""Regression tests for runtime transformer quantization wiring."""

from pathlib import Path
import sys
import types

import pytest

import mlx_video.models.ltx_2.generate as generate_module
from mlx_video.models.ltx_2.generate import PipelineType


class _FakeTextEncoder:
    def __init__(self, has_prompt_adaln: bool = False):
        self.has_prompt_adaln = has_prompt_adaln

    def load(self, model_path, text_encoder_path):
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path

    def parameters(self):
        import mlx.core as mx

        return mx.array(0.0, dtype=mx.float32)

    def __call__(self, prompt, return_audio_embeddings=False):
        import mlx.core as mx

        video_embeddings = mx.zeros((1, 4, 8), dtype=mx.float32)
        if return_audio_embeddings:
            audio_embeddings = mx.zeros((1, 4, 8), dtype=mx.float32)
            return video_embeddings, audio_embeddings
        return video_embeddings


def _prepare_fake_model_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "transformer" / "config.json").write_text(
        '{"has_prompt_adaln": true}', encoding="utf-8"
    )
    (root / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    return root


def test_generate_video_passes_runtime_transformer_quantization(tmp_path, monkeypatch):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    captured = {}

    class _StopAfterTransformerLoad(Exception):
        pass

    monkeypatch.setattr(generate_module, "get_model_path", lambda _repo: model_path)
    monkeypatch.setattr(
        generate_module,
        "resolve_text_encoder_path",
        lambda model_path, text_encoder_repo: model_path,
    )
    monkeypatch.setattr(
        "mlx_video.models.ltx_2.text_encoder.LTX2TextEncoder", _FakeTextEncoder
    )

    def fake_from_pretrained(
        *, model_path, strict, quantization=None, weight_preprocessor=None
    ):
        captured["quantization"] = quantization
        captured["weight_preprocessor"] = weight_preprocessor
        raise _StopAfterTransformerLoad

    monkeypatch.setattr(
        generate_module,
        "LTXModel",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    with pytest.raises(_StopAfterTransformerLoad):
        generate_module.generate_video(
            model_repo="dummy/repo",
            text_encoder_repo=None,
            prompt="A scenic ocean",
            pipeline=PipelineType.DISTILLED,
            height=64,
            width=64,
            num_frames=9,
            output_path=str(tmp_path / "out.mp4"),
            tiling="none",
            verbose=False,
            transformer_quantization_bits=8,
            transformer_quantization_group_size=32,
            transformer_quantization_mode="mxfp8",
        )

    assert captured["quantization"] == {
        "bits": 8,
        "group_size": 32,
        "mode": "mxfp8",
    }
    assert captured["weight_preprocessor"] is None


def test_resolve_transformer_quantization_defaults():
    assert generate_module.resolve_transformer_quantization(
        bits=8,
        group_size=None,
        mode="affine",
    ) == {"bits": 8, "group_size": 64, "mode": "affine"}
    assert generate_module.resolve_transformer_quantization(
        bits=4,
        group_size=None,
        mode="affine",
    ) == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert generate_module.resolve_transformer_quantization(
        bits=8,
        group_size=None,
        mode="mxfp8",
    ) == {"bits": 8, "group_size": 32, "mode": "mxfp8"}
    assert generate_module.resolve_transformer_quantization(
        bits=8,
        group_size=None,
        mode="mxfp8",
        quantize_input=True,
    ) == {
        "bits": 8,
        "group_size": 32,
        "mode": "mxfp8",
        "quantize_input": True,
    }


def test_resolve_transformer_quantization_rejects_invalid_mxfp8_bits():
    with pytest.raises(ValueError, match="bits=8"):
        generate_module.resolve_transformer_quantization(
            bits=4,
            group_size=None,
            mode="mxfp8",
        )


def test_resolve_transformer_quantization_rejects_affine_input_quantization():
    with pytest.raises(ValueError, match="Activation quantization"):
        generate_module.resolve_transformer_quantization(
            bits=8,
            group_size=None,
            mode="affine",
            quantize_input=True,
        )


def test_main_parses_transformer_quantization_flags(monkeypatch):
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(generate_module, "generate_video", fake_generate_video)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_video.ltx_2.generate",
            "--prompt",
            "A scenic ocean",
            "--transformer-quantization-bits",
            "8",
            "--transformer-quantization-mode",
            "mxfp8",
            "--transformer-quantization-group-size",
            "32",
            "--transformer-quantize-inputs",
        ],
    )

    generate_module.main()

    assert captured["transformer_quantization_bits"] == 8
    assert captured["transformer_quantization_group_size"] == 32
    assert captured["transformer_quantization_mode"] == "mxfp8"
    assert captured["transformer_quantize_inputs"] is True


def test_generate_video_passes_mxfp8_input_quantization(tmp_path, monkeypatch):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    captured = {}

    class _StopAfterTransformerLoad(Exception):
        pass

    monkeypatch.setattr(generate_module, "get_model_path", lambda _repo: model_path)
    monkeypatch.setattr(
        generate_module,
        "resolve_text_encoder_path",
        lambda model_path, text_encoder_repo: model_path,
    )
    monkeypatch.setattr(
        "mlx_video.models.ltx_2.text_encoder.LTX2TextEncoder", _FakeTextEncoder
    )

    def fake_from_pretrained(
        *, model_path, strict, quantization=None, weight_preprocessor=None
    ):
        captured["quantization"] = quantization
        raise _StopAfterTransformerLoad

    monkeypatch.setattr(
        generate_module,
        "LTXModel",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    with pytest.raises(_StopAfterTransformerLoad):
        generate_module.generate_video(
            model_repo="dummy/repo",
            text_encoder_repo=None,
            prompt="A scenic ocean",
            pipeline=PipelineType.DISTILLED,
            height=64,
            width=64,
            num_frames=9,
            output_path=str(tmp_path / "out.mp4"),
            tiling="none",
            verbose=False,
            transformer_quantization_bits=8,
            transformer_quantization_group_size=32,
            transformer_quantization_mode="mxfp8",
            transformer_quantize_inputs=True,
        )

    assert captured["quantization"] == {
        "bits": 8,
        "group_size": 32,
        "mode": "mxfp8",
        "quantize_input": True,
    }


def test_main_rejects_invalid_mxfp8_group_size(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_video.ltx_2.generate",
            "--prompt",
            "A scenic ocean",
            "--transformer-quantization-bits",
            "8",
            "--transformer-quantization-mode",
            "mxfp8",
            "--transformer-quantization-group-size",
            "64",
        ],
    )

    with pytest.raises(SystemExit):
        generate_module.main()

    captured = capsys.readouterr()
    assert "group_size=32" in captured.err
