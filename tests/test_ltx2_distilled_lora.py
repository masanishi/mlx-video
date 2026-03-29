"""Regression tests for distilled LoRA handling in LTX-2.3."""

from pathlib import Path
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_video.models.ltx_2.generate as generate_module
from mlx_video.models.ltx_2.config import LTXModelConfig, LTXModelType
from mlx_video.models.ltx_2.generate import PipelineType
from mlx_video.models.ltx_2.ltx_2 import LTXModel
from mlx_video.quantization import QQLinearWithBias, quantize_modules


class _FakeTextEncoder:
    def __init__(self, has_prompt_adaln: bool = False):
        self.has_prompt_adaln = has_prompt_adaln

    def load(self, model_path, text_encoder_path):
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path

    def parameters(self):
        return mx.array(0.0, dtype=mx.float32)

    def __call__(self, prompt, return_audio_embeddings=False):
        video_embeddings = mx.zeros((1, 4, 8), dtype=mx.float32)
        if return_audio_embeddings:
            audio_embeddings = mx.zeros((1, 4, 8), dtype=mx.float32)
            return video_embeddings, audio_embeddings
        return video_embeddings


class _FakeTransformer:
    def __init__(self):
        self.config = types.SimpleNamespace(has_prompt_adaln=True)

    def parameters(self):
        return mx.array(0.0, dtype=mx.float32)


class _FakeUpsampler:
    def parameters(self):
        return mx.array(0.0, dtype=mx.float32)


class _FakeVideoDecoder:
    def __init__(self):
        self.per_channel_statistics = types.SimpleNamespace(
            mean=mx.zeros((1,), dtype=mx.float32),
            std=mx.ones((1,), dtype=mx.float32),
        )

    def parameters(self):
        return mx.array(0.0, dtype=mx.float32)


class _DirectMergeModel:
    def __init__(self, weights):
        self._weights = weights
        self.loaded_batches = []

    def parameters(self):
        return self._weights

    def load_weights(self, batch, strict=False):
        self.loaded_batches.append(list(batch))
        for key, value in batch:
            parts = key.split(".")
            target = self._weights
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value


def _build_quantized_ltx_model(
    *,
    bits: int = 8,
    group_size: int = 64,
    mode: str = "affine",
    quantize_input: bool = False,
) -> LTXModel:
    config = LTXModelConfig(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=1,
        attention_head_dim=64,
        in_channels=64,
        out_channels=64,
        num_layers=1,
        cross_attention_dim=64,
        caption_channels=64,
        positional_embedding_max_pos=[1, 1, 1],
        has_prompt_adaln=False,
    )
    model = LTXModel(config)
    quantize_modules(
        model,
        group_size=group_size,
        bits=bits,
        mode=mode,
        quantize_input=quantize_input,
        class_predicate=lambda path, module: hasattr(module, "to_quantized")
        and (
            not hasattr(module, "weight")
            or module.weight.shape[-1] % group_size == 0
        ),
    )
    mx.eval(model.parameters())
    return model


def _prepare_fake_model_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "transformer" / "config.json").write_text(
        '{"has_prompt_adaln": true}', encoding="utf-8"
    )
    # Distilled always expects a spatial upscaler during the stage-1 -> stage-2 handoff.
    (root / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    return root


def _install_common_fakes(monkeypatch, model_path: Path):
    monkeypatch.setattr(generate_module, "get_model_path", lambda _repo: model_path)
    monkeypatch.setattr(
        generate_module,
        "resolve_text_encoder_path",
        lambda model_path, text_encoder_repo: model_path,
    )
    monkeypatch.setattr(
        "mlx_video.models.ltx_2.text_encoder.LTX2TextEncoder", _FakeTextEncoder
    )
    monkeypatch.setattr(
        generate_module,
        "LTXModel",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _FakeTransformer()
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (_FakeUpsampler(), 2.0),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: latents,
    )
    monkeypatch.setattr(
        generate_module,
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _FakeVideoDecoder()
        ),
    )


def test_distilled_lora_is_merged_when_explicitly_passed(tmp_path, monkeypatch):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    _install_common_fakes(monkeypatch, model_path)

    (tmp_path / "explicit-lora.safetensors").touch()

    class _StopAfterFirstDenoise(Exception):
        pass

    captured = {"merge_calls": 0, "call_order": []}

    def fake_denoise_distilled(latents, *_args, audio_latents=None, **_kwargs):
        captured["call_order"].append("denoise")
        raise _StopAfterFirstDenoise

    def fake_merge(model, lora_path, strength):
        captured["call_order"].append("merge")
        captured["path"] = Path(lora_path)
        captured["strength"] = strength
        captured["merge_calls"] += 1

    monkeypatch.setattr(generate_module, "denoise_distilled", fake_denoise_distilled)
    monkeypatch.setattr(generate_module, "load_and_merge_lora", fake_merge)

    with pytest.raises(_StopAfterFirstDenoise):
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
            lora_path=str(tmp_path / "explicit-lora.safetensors"),
            lora_strength=0.75,
        )

    assert captured["path"].name == "explicit-lora.safetensors"
    assert captured["strength"] == 0.75
    assert captured["merge_calls"] == 1
    assert captured["call_order"] == ["merge", "denoise"]


def test_distilled_without_lora_does_not_merge(tmp_path, monkeypatch):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    _install_common_fakes(monkeypatch, model_path)

    class _StopAfterStage2Denoise(Exception):
        pass

    calls = {"merge": 0, "denoise": 0}

    def fake_denoise_distilled(latents, *_args, audio_latents=None, **_kwargs):
        calls["denoise"] += 1
        if calls["denoise"] == 2:
            raise _StopAfterStage2Denoise
        return latents, audio_latents

    def fake_merge(*_args, **_kwargs):
        calls["merge"] += 1
        raise AssertionError("distilled pipeline should not merge LoRA when lora_path is None")

    monkeypatch.setattr(generate_module, "denoise_distilled", fake_denoise_distilled)
    monkeypatch.setattr(generate_module, "load_and_merge_lora", fake_merge)

    with pytest.raises(_StopAfterStage2Denoise):
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
            lora_path=None,
        )

    assert calls["merge"] == 0
    assert calls["denoise"] == 2


def test_load_and_merge_lora_preserves_base_dtype_and_reports_full_coverage(
    tmp_path, monkeypatch
):
    lora_path = tmp_path / "direct-merge.safetensors"
    lora_path.touch()

    model = _DirectMergeModel(
        {
            "transformer_blocks": {
                "0": {
                    "attn1": {
                        "to_q": {
                            "weight": mx.array(
                                [[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16
                            )
                        }
                    }
                }
            }
        }
    )
    printed = []

    monkeypatch.setattr(
        generate_module,
        "console",
        types.SimpleNamespace(print=lambda message: printed.append(str(message))),
    )
    monkeypatch.setattr(
        generate_module.mx,
        "load",
        lambda _path: {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.array(
                [[1.0, -1.0]], dtype=mx.float32
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.array(
                [[2.0], [3.0]], dtype=mx.float32
            ),
        },
    )

    generate_module.load_and_merge_lora(model, str(lora_path), strength=0.5)

    merged_weight = model._weights["transformer_blocks"]["0"]["attn1"]["to_q"][
        "weight"
    ]
    expected_delta = np.array([[1.0, -1.0], [1.5, -1.5]], dtype=np.float32)
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32) + expected_delta

    assert merged_weight.dtype == mx.bfloat16
    np.testing.assert_allclose(
        np.array(merged_weight.astype(mx.float32)), expected, rtol=1e-3, atol=1e-3
    )
    assert any("Merged 1/1 LoRA pairs" in line for line in printed)


def test_build_runtime_quantized_lora_preprocessor_applies_delta(tmp_path):
    lora_path = tmp_path / "premerge-lora.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.ones(
                (1, 64), dtype=mx.float32
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.ones(
                (64, 1), dtype=mx.float32
            ),
        },
    )

    preprocess, resolved = generate_module._build_runtime_quantized_lora_preprocessor(
        str(lora_path),
        strength=0.5,
    )
    weights = {
        "transformer_blocks.0.attn1.to_q.weight": mx.zeros((64, 64), dtype=mx.float32)
    }

    processed = preprocess(weights)

    assert resolved == lora_path
    assert not mx.array_equal(
        processed["transformer_blocks.0.attn1.to_q.weight"],
        weights["transformer_blocks.0.attn1.to_q.weight"],
    ).item()


def test_generate_video_uses_mxfp8_lora_preprocessor_during_transformer_load(
    tmp_path, monkeypatch
):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    lora_path = tmp_path / "runtime-premerge-lora.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.ones(
                (1, 64), dtype=mx.float32
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.ones(
                (64, 1), dtype=mx.float32
            ),
        },
    )
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
        *,
        model_path,
        strict,
        quantization=None,
        weight_preprocessor=None,
    ):
        captured["quantization"] = quantization
        captured["has_preprocessor"] = weight_preprocessor is not None
        sample = {
            "transformer_blocks.0.attn1.to_q.weight": mx.zeros(
                (64, 64), dtype=mx.float32
            )
        }
        processed = weight_preprocessor(sample)
        captured["changed"] = not mx.array_equal(
            processed["transformer_blocks.0.attn1.to_q.weight"],
            sample["transformer_blocks.0.attn1.to_q.weight"],
        ).item()
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
            lora_path=str(lora_path),
            transformer_quantization_bits=8,
            transformer_quantization_group_size=32,
            transformer_quantization_mode="mxfp8",
        )

    assert captured["quantization"] == {
        "bits": 8,
        "group_size": 32,
        "mode": "mxfp8",
    }
    assert captured["has_preprocessor"] is True
    assert captured["changed"] is True


def test_load_and_merge_lora_reports_unmatched_pairs(tmp_path, monkeypatch):
    lora_path = tmp_path / "partial-merge.safetensors"
    lora_path.touch()

    model = _DirectMergeModel(
        {
            "transformer_blocks": {
                "0": {
                    "attn1": {
                        "to_q": {"weight": mx.array([[1.0]], dtype=mx.float32)}
                    }
                }
            }
        }
    )
    printed = []

    monkeypatch.setattr(
        generate_module,
        "console",
        types.SimpleNamespace(print=lambda message: printed.append(str(message))),
    )
    monkeypatch.setattr(
        generate_module.mx,
        "load",
        lambda _path: {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.array(
                [[1.0]], dtype=mx.float32
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.array(
                [[1.0]], dtype=mx.float32
            ),
            "diffusion_model.nonexistent.lora_A.weight": mx.array(
                [[1.0]], dtype=mx.float32
            ),
            "diffusion_model.nonexistent.lora_B.weight": mx.array(
                [[1.0]], dtype=mx.float32
            ),
        },
    )

    generate_module.load_and_merge_lora(model, str(lora_path), strength=1.0)

    assert any("Skipped 1 unmatched LoRA pairs" in line for line in printed)
    assert any("Merged 1/2 LoRA pairs" in line for line in printed)


def test_load_and_merge_lora_requantizes_quantized_linear(tmp_path, monkeypatch):
    lora_path = tmp_path / "quantized-merge.safetensors"
    lora_path.touch()

    model = _build_quantized_ltx_model()

    monkeypatch.setattr(
        generate_module.mx,
        "load",
        lambda _path: {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.array(
                [[1.0] * 64], dtype=mx.float32
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.array(
                [[0.5], [0.25]] + [[0.0]] * 62,
                dtype=mx.float32,
            ),
        },
    )

    target_before = model.transformer_blocks[0].attn1.to_q
    assert isinstance(target_before, nn.QuantizedLinear)

    generate_module.load_and_merge_lora(model, str(lora_path), strength=1.0)

    target_after = model.transformer_blocks[0].attn1.to_q
    assert isinstance(target_after, nn.QuantizedLinear)
    assert target_after.mode == "affine"
    assert target_after.group_size == 64


def test_load_and_merge_lora_requantizes_mxfp4_quantized_linear(
    tmp_path, monkeypatch
):
    lora_path = tmp_path / "quantized-merge-mxfp4.safetensors"
    lora_path.touch()

    model = _build_quantized_ltx_model(bits=4, group_size=32, mode="mxfp4")

    monkeypatch.setattr(
        generate_module.mx,
        "load",
        lambda _path: {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.array(
                [[1.0] * 64], dtype=mx.float32
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.array(
                [[0.5], [0.25]] + [[0.0]] * 62,
                dtype=mx.float32,
            ),
        },
    )

    generate_module.load_and_merge_lora(model, str(lora_path), strength=1.0)

    target_after = model.transformer_blocks[0].attn1.to_q
    assert isinstance(target_after, nn.QuantizedLinear)
    assert target_after.mode == "mxfp4"
    assert target_after.bits == 4
    assert target_after.group_size == 32
    assert target_after.biases is None


def test_load_and_merge_lora_requantizes_mxfp8_using_float32_merge(
    tmp_path, monkeypatch
):
    lora_path = tmp_path / "quantized-merge-mxfp8.safetensors"
    lora_path.touch()

    model = _build_quantized_ltx_model(bits=8, group_size=32, mode="mxfp8")
    captured = {}
    original_quantize = generate_module.mx.quantize

    monkeypatch.setattr(
        generate_module.mx,
        "load",
        lambda _path: {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.array(
                [[1.0] * 64], dtype=mx.float32
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.array(
                [[0.5], [0.25]] + [[0.0]] * 62,
                dtype=mx.float32,
            ),
        },
    )

    def capture_quantize(weight, *args, **kwargs):
        if kwargs.get("mode") == "mxfp8":
            captured["dtype"] = weight.dtype
        return original_quantize(weight, *args, **kwargs)

    monkeypatch.setattr(generate_module.mx, "quantize", capture_quantize)

    generate_module.load_and_merge_lora(model, str(lora_path), strength=1.0)

    target_after = model.transformer_blocks[0].attn1.to_q
    assert isinstance(target_after, nn.QuantizedLinear)
    assert target_after.mode == "mxfp8"
    assert target_after.bits == 8
    assert target_after.group_size == 32
    assert target_after.biases is None
    assert captured["dtype"] == mx.float32


def test_load_and_merge_lora_preserves_mxfp8_input_quantized_linear(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "mlx_video.quantization.activation_quantized_matmul_supported",
        lambda mode: mode == "mxfp8",
    )
    lora_path = tmp_path / "quantized-merge-mxfp8-inputs.safetensors"
    lora_path.touch()

    model = _build_quantized_ltx_model(
        bits=8,
        group_size=32,
        mode="mxfp8",
        quantize_input=True,
    )

    monkeypatch.setattr(
        generate_module.mx,
        "load",
        lambda _path: {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.array(
                [[1.0] * 64], dtype=mx.float32
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.array(
                [[0.5], [0.25]] + [[0.0]] * 62,
                dtype=mx.float32,
            ),
        },
    )

    target_before = model.transformer_blocks[0].attn1.to_q
    assert isinstance(target_before, QQLinearWithBias)

    generate_module.load_and_merge_lora(model, str(lora_path), strength=1.0)

    target_after = model.transformer_blocks[0].attn1.to_q
    assert isinstance(target_after, QQLinearWithBias)
    assert target_after.mode == "mxfp8"
    assert target_after.bits == 8
    assert target_after.group_size == 32
    assert target_after.biases is None
