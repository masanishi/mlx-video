"""Regression tests for LTX-2 model/text-encoder loading."""

import json
from pathlib import Path
import types

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_video.models.ltx_2.generate as generate_module
from mlx_video.models.ltx_2.config import LTXModelConfig, LTXModelType
from mlx_video.models.ltx_2.ltx_2 import LTXModel
from mlx_video.models.ltx_2.text_encoder import LanguageModel
from mlx_video.quantization import QQLinearWithBias, quantize_modules
from mlx_video.utils import get_model_path


def _flatten_params(params, prefix=""):
    flat = {}

    if isinstance(params, dict):
        items = params.items()
    elif isinstance(params, (list, tuple)):
        items = enumerate(params)
    else:
        if hasattr(params, "dtype"):
            flat[prefix] = params
        return flat

    for key, value in items:
        key_str = str(key)
        full_key = f"{prefix}.{key_str}" if prefix else key_str
        flat.update(_flatten_params(value, full_key))

    return flat


def _build_tiny_ltx_model(quantizable: bool = False) -> LTXModel:
    head_dim = 64 if quantizable else 8
    channels = 64 if quantizable else 4
    context_dim = 64 if quantizable else 8
    config = LTXModelConfig(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=1,
        attention_head_dim=head_dim,
        in_channels=channels,
        out_channels=channels,
        num_layers=1,
        cross_attention_dim=context_dim,
        caption_channels=context_dim,
        positional_embedding_max_pos=[1, 1, 1],
        has_prompt_adaln=False,
    )
    model = LTXModel(config)
    mx.eval(model.parameters())
    return model


def test_ltx_model_from_pretrained_respects_index_json(tmp_path, monkeypatch):
    model = _build_tiny_ltx_model()
    weights = _flatten_params(dict(model.parameters()))

    canonical_name = "model-00000-of-00002.safetensors"
    extra_name = "model-00001-of-00008.safetensors"

    (tmp_path / "config.json").write_text(
        json.dumps(model.config.to_dict()), encoding="utf-8"
    )
    (tmp_path / canonical_name).touch()
    (tmp_path / extra_name).touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {key: canonical_name for key in weights},
            }
        ),
        encoding="utf-8",
    )

    load_calls = []

    def fake_load(path: str):
        name = Path(path).name
        load_calls.append(name)
        if name == canonical_name:
            return weights
        if name == extra_name:
            raise AssertionError("index外の shard を読み込んではいけません")
        raise AssertionError(f"Unexpected load path: {path}")

    monkeypatch.setattr("mlx_video.models.ltx_2.ltx_2.mx.load", fake_load)

    loaded = LTXModel.from_pretrained(tmp_path)

    assert isinstance(loaded, LTXModel)
    assert load_calls == [canonical_name]


def test_ltx_model_from_pretrained_runtime_quantizes_transformer(tmp_path):
    model = _build_tiny_ltx_model(quantizable=True)
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(model.config.to_dict()), encoding="utf-8"
    )

    loaded = LTXModel.from_pretrained(
        tmp_path,
        quantization={"bits": 8, "group_size": 64, "mode": "affine"},
    )

    assert isinstance(loaded.patchify_proj, nn.Linear)
    assert isinstance(loaded.transformer_blocks[0].attn1.to_q, nn.QuantizedLinear)
    assert loaded.transformer_blocks[0].attn1.to_q.mode == "affine"


def test_ltx_model_from_pretrained_runtime_quantizes_transformer_mxfp4(tmp_path):
    model = _build_tiny_ltx_model(quantizable=True)
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(model.config.to_dict()), encoding="utf-8"
    )

    loaded = LTXModel.from_pretrained(
        tmp_path,
        quantization={"bits": 4, "mode": "mxfp4"},
    )

    assert isinstance(loaded.transformer_blocks[0].attn1.to_q, nn.QuantizedLinear)
    assert loaded.transformer_blocks[0].attn1.to_q.mode == "mxfp4"
    assert loaded.transformer_blocks[0].attn1.to_q.group_size == 32
    assert loaded.transformer_blocks[0].attn1.to_q.biases is None


def test_ltx_model_from_pretrained_runtime_quantizes_transformer_mxfp8(tmp_path):
    model = _build_tiny_ltx_model(quantizable=True)
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(model.config.to_dict()), encoding="utf-8"
    )

    loaded = LTXModel.from_pretrained(
        tmp_path,
        quantization={"bits": 8, "mode": "mxfp8"},
    )

    assert isinstance(loaded.transformer_blocks[0].attn1.to_q, nn.QuantizedLinear)
    assert loaded.transformer_blocks[0].attn1.to_q.mode == "mxfp8"
    assert loaded.transformer_blocks[0].attn1.to_q.group_size == 32
    assert loaded.transformer_blocks[0].attn1.to_q.biases is None


def test_ltx_model_from_pretrained_runtime_quantizes_transformer_mxfp8_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "mlx_video.quantization.activation_quantized_matmul_supported",
        lambda mode: mode == "mxfp8",
    )
    model = _build_tiny_ltx_model(quantizable=True)
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(model.config.to_dict()), encoding="utf-8"
    )

    loaded = LTXModel.from_pretrained(
        tmp_path,
        quantization={"bits": 8, "mode": "mxfp8", "quantize_input": True},
    )

    target = loaded.transformer_blocks[0].attn1.to_q
    assert isinstance(target, QQLinearWithBias)
    assert target.mode == "mxfp8"
    assert target.group_size == 32
    assert target.biases is None
    assert "bias" in target
    out = target(mx.zeros((1, 2, 64), dtype=mx.bfloat16))
    assert out.shape == (1, 2, 64)


def test_ltx_model_from_pretrained_runtime_quantizes_transformer_mxfp8_inputs_falls_back(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "mlx_video.quantization.activation_quantized_matmul_supported",
        lambda _mode: False,
    )
    model = _build_tiny_ltx_model(quantizable=True)
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(model.config.to_dict()), encoding="utf-8"
    )

    loaded = LTXModel.from_pretrained(
        tmp_path,
        quantization={"bits": 8, "mode": "mxfp8", "quantize_input": True},
    )

    target = loaded.transformer_blocks[0].attn1.to_q
    assert isinstance(target, nn.QuantizedLinear)
    assert not isinstance(target, QQLinearWithBias)
    assert getattr(loaded, "_activation_quantization_enabled", False) is False
    assert getattr(loaded, "_activation_quantization_fallback", False) is True


def test_ltx_model_from_pretrained_loads_saved_quantized_transformer(tmp_path):
    model = _build_tiny_ltx_model(quantizable=True)
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        mode="affine",
        class_predicate=lambda path, module: hasattr(module, "to_quantized")
        and (not hasattr(module, "weight") or module.weight.shape[-1] % 64 == 0),
    )
    mx.eval(model.parameters())
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **model.config.to_dict(),
                "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
            }
        ),
        encoding="utf-8",
    )

    loaded = LTXModel.from_pretrained(tmp_path)

    assert isinstance(loaded.patchify_proj, nn.QuantizedLinear)
    assert isinstance(loaded.transformer_blocks[0].attn1.to_q, nn.QuantizedLinear)
    assert loaded.transformer_blocks[0].attn1.to_q.mode == "affine"


def test_ltx_model_from_pretrained_loads_saved_mxfp4_transformer(tmp_path):
    model = _build_tiny_ltx_model(quantizable=True)
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        mode="mxfp4",
        class_predicate=lambda path, module: hasattr(module, "to_quantized")
        and (not hasattr(module, "weight") or module.weight.shape[-1] % 32 == 0),
    )
    mx.eval(model.parameters())
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **model.config.to_dict(),
                "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
            }
        ),
        encoding="utf-8",
    )

    loaded = LTXModel.from_pretrained(tmp_path)

    assert isinstance(loaded.transformer_blocks[0].attn1.to_q, nn.QuantizedLinear)
    assert loaded.transformer_blocks[0].attn1.to_q.mode == "mxfp4"
    assert loaded.transformer_blocks[0].attn1.to_q.group_size == 32
    assert loaded.transformer_blocks[0].attn1.to_q.biases is None


def test_ltx_model_from_pretrained_loads_saved_mxfp8_transformer(tmp_path):
    model = _build_tiny_ltx_model(quantizable=True)
    nn.quantize(
        model,
        group_size=32,
        bits=8,
        mode="mxfp8",
        class_predicate=lambda path, module: hasattr(module, "to_quantized")
        and (not hasattr(module, "weight") or module.weight.shape[-1] % 32 == 0),
    )
    mx.eval(model.parameters())
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **model.config.to_dict(),
                "quantization": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
            }
        ),
        encoding="utf-8",
    )

    loaded = LTXModel.from_pretrained(tmp_path)

    assert isinstance(loaded.transformer_blocks[0].attn1.to_q, nn.QuantizedLinear)
    assert loaded.transformer_blocks[0].attn1.to_q.mode == "mxfp8"
    assert loaded.transformer_blocks[0].attn1.to_q.group_size == 32
    assert loaded.transformer_blocks[0].attn1.to_q.biases is None


def test_ltx_model_from_pretrained_loads_saved_mxfp8_input_quantized_transformer(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "mlx_video.quantization.activation_quantized_matmul_supported",
        lambda mode: mode == "mxfp8",
    )
    model = _build_tiny_ltx_model(quantizable=True)
    quantize_modules(
        model,
        group_size=32,
        bits=8,
        mode="mxfp8",
        quantize_input=True,
        class_predicate=lambda path, module: hasattr(module, "to_quantized")
        and (not hasattr(module, "weight") or module.weight.shape[-1] % 32 == 0),
    )
    mx.eval(model.parameters())
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **model.config.to_dict(),
                "quantization": {
                    "bits": 8,
                    "group_size": 32,
                    "mode": "mxfp8",
                    "quantize_input": True,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = LTXModel.from_pretrained(tmp_path)

    target = loaded.transformer_blocks[0].attn1.to_q
    assert isinstance(target, QQLinearWithBias)
    assert target.mode == "mxfp8"
    assert target.group_size == 32
    assert target.biases is None
    assert "bias" in target


def test_ltx_model_runtime_quantization_skips_sensitive_top_level_layers(tmp_path):
    model = _build_tiny_ltx_model(quantizable=True)
    weights = _flatten_params(dict(model.parameters()))

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(model.config.to_dict()), encoding="utf-8"
    )

    loaded = LTXModel.from_pretrained(
        tmp_path,
        quantization={"bits": 8, "group_size": 64, "mode": "affine"},
    )

    assert isinstance(loaded.transformer_blocks[0].attn1.to_q, nn.QuantizedLinear)
    assert isinstance(loaded.patchify_proj, nn.Linear)
    assert isinstance(loaded.proj_out, nn.Linear)


def test_language_model_from_pretrained_requires_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {}}), encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="No \\.safetensors weights found"):
        LanguageModel.from_pretrained(tmp_path)


def test_load_video_decoder_statistics_reads_directly_from_weights(
    tmp_path, monkeypatch
):
    decoder_dir = tmp_path / "vae" / "decoder"
    decoder_dir.mkdir(parents=True)
    mx.save_safetensors(
        str(decoder_dir / "model.safetensors"),
        {
            "per_channel_statistics.mean": mx.array([1.0, 2.0], dtype=mx.float32),
            "per_channel_statistics.std": mx.array([3.0, 4.0], dtype=mx.float32),
            "conv_in.conv.weight": mx.zeros((1, 1, 1, 1, 1), dtype=mx.float32),
        },
    )

    generate_module._VIDEO_DECODER_STATS_CACHE.clear()
    monkeypatch.setattr(
        generate_module,
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("full decoder load should not be used")
            )
        ),
    )

    mean, std = generate_module.load_video_decoder_statistics(tmp_path)

    assert mx.array_equal(mean, mx.array([1.0, 2.0], dtype=mx.float32))
    assert mx.array_equal(std, mx.array([3.0, 4.0], dtype=mx.float32))


def test_load_video_decoder_statistics_handles_bfloat16_fast_path(
    tmp_path, monkeypatch
):
    decoder_dir = tmp_path / "vae" / "decoder"
    decoder_dir.mkdir(parents=True)
    mx.save_safetensors(
        str(decoder_dir / "model.safetensors"),
        {
            "per_channel_statistics.mean": mx.array([1.0, 2.0], dtype=mx.bfloat16),
            "per_channel_statistics.std": mx.array([3.0, 4.0], dtype=mx.bfloat16),
            "conv_in.conv.weight": mx.zeros((1, 1, 1, 1, 1), dtype=mx.bfloat16),
        },
    )

    generate_module._VIDEO_DECODER_STATS_CACHE.clear()
    monkeypatch.setattr(
        generate_module,
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("full decoder load should not be used")
            )
        ),
    )

    mean, std = generate_module.load_video_decoder_statistics(tmp_path)

    assert mean.dtype == mx.bfloat16
    assert std.dtype == mx.bfloat16
    assert mx.array_equal(mean, mx.array([1.0, 2.0], dtype=mx.bfloat16))
    assert mx.array_equal(std, mx.array([3.0, 4.0], dtype=mx.bfloat16))


def test_load_video_decoder_statistics_falls_back_and_caches_results(
    tmp_path, monkeypatch
):
    decoder_dir = tmp_path / "vae" / "decoder"
    decoder_dir.mkdir(parents=True)
    mx.save_safetensors(
        str(decoder_dir / "model.safetensors"),
        {
            "conv_in.conv.weight": mx.zeros((1, 1, 1, 1, 1), dtype=mx.float32),
        },
    )

    calls = {"count": 0}

    class _FakeDecoder:
        def __init__(self):
            self.per_channel_statistics = types.SimpleNamespace(
                mean=mx.array([5.0, 6.0], dtype=mx.float32),
                std=mx.array([7.0, 8.0], dtype=mx.float32),
            )

    def fake_from_pretrained(_path):
        calls["count"] += 1
        return _FakeDecoder()

    generate_module._VIDEO_DECODER_STATS_CACHE.clear()
    monkeypatch.setattr(
        generate_module,
        "VideoDecoder",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    mean1, std1 = generate_module.load_video_decoder_statistics(tmp_path)
    mean2, std2 = generate_module.load_video_decoder_statistics(tmp_path)

    assert calls["count"] == 1
    assert mx.array_equal(mean1, mean2)
    assert mx.array_equal(std1, std2)
    assert mx.array_equal(mean1, mx.array([5.0, 6.0], dtype=mx.float32))
    assert mx.array_equal(std1, mx.array([7.0, 8.0], dtype=mx.float32))


def test_get_model_path_redownloads_incomplete_cache(tmp_path, monkeypatch):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}", encoding="utf-8")

    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "config.json").write_text("{}", encoding="utf-8")
    (complete / "model.safetensors").touch()

    calls = []

    def fake_snapshot_download(repo_id: str, local_files_only: bool, **kwargs):
        calls.append(
            {
                "repo_id": repo_id,
                "local_files_only": local_files_only,
                "allow_patterns": kwargs.get("allow_patterns"),
            }
        )
        return str(partial if local_files_only else complete)

    monkeypatch.setattr("mlx_video.utils.snapshot_download", fake_snapshot_download)

    resolved = get_model_path("dummy/repo")

    assert resolved == complete
    assert [call["local_files_only"] for call in calls] == [True, False]


def test_generate_video_falls_back_to_default_distilled_repo_when_local_ltx23_dir_is_missing(
    tmp_path, monkeypatch
):
    missing_local_repo = tmp_path / "LTX-2.3-distilled"
    captured = []

    class _StopAfterModelResolve(Exception):
        pass

    def fake_get_model_path(repo: str):
        captured.append(repo)
        raise _StopAfterModelResolve

    monkeypatch.setattr(generate_module, "get_model_path", fake_get_model_path)

    with pytest.raises(_StopAfterModelResolve):
        generate_module.generate_video(
            model_repo=str(missing_local_repo),
            text_encoder_repo=None,
            prompt="demo prompt",
            pipeline=generate_module.PipelineType.DISTILLED,
            verbose=False,
            return_video=False,
        )

    assert captured == [generate_module.DEFAULT_DISTILLED_MODEL_REPO]


def test_generate_video_keeps_existing_local_distilled_repo_path(tmp_path, monkeypatch):
    local_repo = tmp_path / "LTX-2.3-distilled"
    local_repo.mkdir()
    captured = []

    class _StopAfterModelResolve(Exception):
        pass

    def fake_get_model_path(repo: str):
        captured.append(repo)
        raise _StopAfterModelResolve

    monkeypatch.setattr(generate_module, "get_model_path", fake_get_model_path)

    with pytest.raises(_StopAfterModelResolve):
        generate_module.generate_video(
            model_repo=str(local_repo),
            text_encoder_repo=None,
            prompt="demo prompt",
            pipeline=generate_module.PipelineType.DISTILLED,
            verbose=False,
            return_video=False,
        )

    assert captured == [str(local_repo)]
