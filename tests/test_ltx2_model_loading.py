"""Regression tests for LTX-2 model/text-encoder loading."""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_video.models.ltx_2.config import LTXModelConfig, LTXModelType
from mlx_video.models.ltx_2.ltx_2 import LTXModel
from mlx_video.models.ltx_2.text_encoder import LanguageModel
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


def _build_tiny_ltx_model() -> LTXModel:
    config = LTXModelConfig(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=1,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        cross_attention_dim=8,
        caption_channels=8,
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


def test_language_model_from_pretrained_requires_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {}}), encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="No \\.safetensors weights found"):
        LanguageModel.from_pretrained(tmp_path)


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