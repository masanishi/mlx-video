"""Regression tests for LTX-2.3 image-to-video with synchronized audio."""

import sys
import types
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_video.models.ltx_2.generate as generate_module
from mlx_video.generate_dev import generate_video_dev
from mlx_video.models.ltx_2.config import LTXModelType
from mlx_video.models.ltx_2.generate import PipelineType


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
        attn = types.SimpleNamespace(query_chunk_size=None)
        block = types.SimpleNamespace(
            attn1=attn,
            attn2=types.SimpleNamespace(query_chunk_size=None),
            audio_attn1=types.SimpleNamespace(query_chunk_size=None),
            audio_attn2=types.SimpleNamespace(query_chunk_size=None),
            audio_to_video_attn=types.SimpleNamespace(query_chunk_size=None),
            video_to_audio_attn=types.SimpleNamespace(query_chunk_size=None),
        )
        self.transformer_blocks = {0: block}

    def parameters(self):
        return mx.array(0.0, dtype=mx.float32)


class _FakeVideoEncoder:
    def __call__(self, image_tensor):
        _, _, _, height, width = image_tensor.shape
        return mx.zeros((1, 128, 1, height // 32, width // 32), dtype=image_tensor.dtype)


class _FakeVideoDecoder:
    def __init__(self):
        self.per_channel_statistics = types.SimpleNamespace(
            mean=mx.zeros((1,), dtype=mx.float32),
            std=mx.ones((1,), dtype=mx.float32),
        )

    def parameters(self):
        return mx.array(0.0, dtype=mx.float32)

    def __call__(self, latents):
        latent_frames = latents.shape[2]
        num_frames = 1 + (latent_frames - 1) * 8
        height = latents.shape[3] * 32
        width = latents.shape[4] * 32
        return mx.zeros((1, 3, num_frames, height, width), dtype=mx.float32)

    def decode_tiled(self, latents, **_kwargs):
        return self(latents)


class _FakeAudioDecoder:
    def parameters(self):
        return mx.array(0.0, dtype=mx.float32)

    def __call__(self, audio_latents):
        audio_frames = audio_latents.shape[2]
        return mx.zeros((1, 2, audio_frames * 4, 64), dtype=mx.float32)


class _FakeVocoder:
    output_sampling_rate = 24000

    def parameters(self):
        return mx.array(0.0, dtype=mx.float32)

    def __call__(self, mel_spectrogram):
        return mx.zeros((2, 1024), dtype=mx.float32)


class _FakeVideoWriter:
    def __init__(self, path, *_args, **_kwargs):
        self.path = Path(path)

    def write(self, _frame):
        return None

    def release(self):
        self.path.write_bytes(b"fake-video")


_FAKE_CV2 = types.SimpleNamespace(
    COLOR_RGB2BGR=0,
    VideoWriter_fourcc=lambda *_args: 0,
    VideoWriter=_FakeVideoWriter,
    cvtColor=lambda frame, _flag: frame,
)


def _prepare_fake_model_dir(root: Path) -> Path:
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "transformer" / "config.json").write_text(
        '{"has_prompt_adaln": true}', encoding="utf-8"
    )
    return root


def test_generate_video_dev_wrapper_forces_dev_pipeline(monkeypatch):
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("mlx_video.generate_dev.generate_video", fake_generate_video)

    generate_video_dev(prompt="demo prompt")

    assert captured["pipeline"] == PipelineType.DEV
    assert captured["tiling"] == "none"


def test_generate_video_dev_wrapper_forwards_low_memory(monkeypatch):
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("mlx_video.generate_dev.generate_video", fake_generate_video)

    generate_video_dev(prompt="demo prompt", low_memory=True)

    assert captured["pipeline"] == PipelineType.DEV
    assert captured["low_memory"] is True


def test_generate_video_dev_wrapper_forwards_profile_memory(monkeypatch):
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("mlx_video.generate_dev.generate_video", fake_generate_video)

    generate_video_dev(prompt="demo prompt", profile_memory=True)

    assert captured["pipeline"] == PipelineType.DEV
    assert captured["profile_memory"] is True


def test_main_accepts_image_plus_generated_audio(monkeypatch):
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(generate_module, "generate_video", fake_generate_video)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_video.ltx_2.generate",
            "--prompt",
            "demo prompt",
            "--image",
            "start.png",
            "--audio",
        ],
    )

    generate_module.main()

    assert captured["image"] == "start.png"
    assert captured["audio"] is True
    assert captured["audio_file"] is None


def test_main_accepts_low_memory(monkeypatch):
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(generate_module, "generate_video", fake_generate_video)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_video.ltx_2.generate",
            "--prompt",
            "demo prompt",
            "--low-memory",
        ],
    )

    generate_module.main()

    assert captured["low_memory"] is True


def test_main_accepts_profile_memory(monkeypatch):
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(generate_module, "generate_video", fake_generate_video)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_video.ltx_2.generate",
            "--prompt",
            "demo prompt",
            "--profile-memory",
        ],
    )

    generate_module.main()

    assert captured["profile_memory"] is True


def test_main_accepts_skip_stage2_refinement(monkeypatch):
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(generate_module, "generate_video", fake_generate_video)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_video.ltx_2.generate",
            "--prompt",
            "demo prompt",
            "--skip-stage2-refinement",
        ],
    )

    generate_module.main()

    assert captured["skip_stage2_refinement"] is True


def test_main_accepts_stage2_refinement_steps(monkeypatch):
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(generate_module, "generate_video", fake_generate_video)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_video.ltx_2.generate",
            "--prompt",
            "demo prompt",
            "--stage2-refinement-steps",
            "1",
        ],
    )

    generate_module.main()

    assert captured["stage2_refinement_steps"] == 1


def test_low_memory_configures_adaptive_transformer_query_chunking(tmp_path, monkeypatch):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    configured = {"chunks": []}

    class _StopAfterStage2ChunkConfig(Exception):
        pass

    def fake_set_attention_query_chunk_size(_model, query_chunk_size):
        configured["chunks"].append(query_chunk_size)
        if len(configured["chunks"]) >= 2:
            raise _StopAfterStage2ChunkConfig

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
        "set_attention_query_chunk_size",
        fake_set_attention_query_chunk_size,
    )
    monkeypatch.setattr(
        generate_module,
        "denoise_distilled",
        lambda latents, *_args, audio_latents=None, **_kwargs: (latents, audio_latents),
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (
            types.SimpleNamespace(parameters=lambda: mx.array(0.0, dtype=mx.float32)),
            2.0,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "load_video_decoder_statistics",
        lambda *_args, **_kwargs: (
            mx.zeros((1,), dtype=mx.float32),
            mx.ones((1,), dtype=mx.float32),
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: mx.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3] * 2,
                latents.shape[4] * 2,
            ),
            dtype=latents.dtype,
        ),
    )
    with pytest.raises(_StopAfterStage2ChunkConfig):
        generate_module.generate_video(
            model_repo="dummy/repo",
            text_encoder_repo=None,
            prompt="demo prompt",
            pipeline=PipelineType.DISTILLED,
            height=1344,
            width=768,
            num_frames=97,
            output_path=str(tmp_path / "out.mp4"),
            verbose=False,
            low_memory=True,
            audio=False,
            tiling="none",
        )

    assert configured["chunks"][:2] == [None, 4096]


def test_strip_transformer_to_video_only_removes_audio_modules():
    simple_preprocessor = types.SimpleNamespace(
        patchify_proj=object(),
        adaln=object(),
        caption_projection=object(),
        inner_dim=128,
        max_pos=[20, 2048, 2048],
        num_attention_heads=32,
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1000,
        positional_embedding_theta=10000.0,
        rope_type="interleaved",
        double_precision_rope=False,
        prompt_adaln=object(),
    )
    block = types.SimpleNamespace(
        audio_attn1=object(),
        audio_attn2=object(),
        audio_ff=object(),
        audio_scale_shift_table=object(),
        audio_prompt_scale_shift_table=object(),
        audio_to_video_attn=object(),
        video_to_audio_attn=object(),
        scale_shift_table_a2v_ca_audio=object(),
        scale_shift_table_a2v_ca_video=object(),
    )
    transformer = types.SimpleNamespace(
        model_type=LTXModelType.AudioVideo,
        config=types.SimpleNamespace(model_type=LTXModelType.AudioVideo),
        video_args_preprocessor=types.SimpleNamespace(simple_preprocessor=simple_preprocessor),
        audio_patchify_proj=object(),
        audio_adaln_single=object(),
        audio_prompt_adaln_single=object(),
        audio_caption_projection=object(),
        audio_scale_shift_table=object(),
        audio_norm_out=object(),
        audio_proj_out=object(),
        audio_args_preprocessor=object(),
        av_ca_video_scale_shift_adaln_single=object(),
        av_ca_audio_scale_shift_adaln_single=object(),
        av_ca_a2v_gate_adaln_single=object(),
        av_ca_v2a_gate_adaln_single=object(),
        av_ca_timestep_scale_multiplier=1000,
        transformer_blocks={0: block},
    )

    generate_module.strip_transformer_to_video_only(transformer)

    assert transformer.model_type == LTXModelType.VideoOnly
    assert transformer.config.model_type == LTXModelType.VideoOnly
    assert transformer.video_args_preprocessor.__class__.__name__ == "TransformerArgsPreprocessor"
    assert not hasattr(transformer, "audio_args_preprocessor")
    assert not hasattr(transformer, "audio_proj_out")
    assert not hasattr(block, "audio_attn1")
    assert not hasattr(block, "audio_to_video_attn")


def test_generate_video_allows_image_plus_audio_generation(tmp_path, monkeypatch):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    output_path = tmp_path / "i2v-audio.mp4"
    denoise_calls = {}
    mux_calls = {}

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
        "VideoEncoder",
        types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _FakeVideoEncoder()),
    )
    monkeypatch.setattr(
        generate_module,
        "VideoDecoder",
        types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _FakeVideoDecoder()),
    )
    monkeypatch.setattr(
        generate_module,
        "load_image",
        lambda _image, height, width, dtype: mx.zeros((height, width, 3), dtype=dtype),
    )
    monkeypatch.setattr(
        generate_module,
        "prepare_image_for_encoding",
        lambda _image, height, width, dtype: mx.zeros(
            (1, 3, 1, height, width), dtype=dtype
        ),
    )

    def fake_denoise_dev_av(
        latents,
        audio_latents,
        video_positions,
        audio_positions,
        *_args,
        video_state=None,
        audio_frozen=False,
        **_kwargs,
    ):
        denoise_calls["video_state"] = video_state
        denoise_calls["audio_shape"] = audio_latents.shape
        denoise_calls["video_positions_shape"] = video_positions.shape
        denoise_calls["audio_positions_shape"] = audio_positions.shape
        denoise_calls["audio_frozen"] = audio_frozen
        return latents, audio_latents

    monkeypatch.setattr(generate_module, "denoise_dev_av", fake_denoise_dev_av)
    monkeypatch.setattr(
        generate_module,
        "load_audio_decoder",
        lambda *_args, **_kwargs: _FakeAudioDecoder(),
    )
    monkeypatch.setattr(
        generate_module,
        "load_vocoder_model",
        lambda *_args, **_kwargs: _FakeVocoder(),
    )
    monkeypatch.setattr(
        generate_module,
        "save_audio",
        lambda audio, path, sample_rate: Path(path).write_bytes(
            f"sr={sample_rate},shape={audio.shape}".encode("utf-8")
        ),
    )

    def fake_mux(video_path, audio_path, final_output_path):
        mux_calls["video_path"] = Path(video_path)
        mux_calls["audio_path"] = Path(audio_path)
        mux_calls["output_path"] = Path(final_output_path)
        Path(final_output_path).write_bytes(b"muxed")
        return True

    monkeypatch.setattr(generate_module, "mux_video_audio", fake_mux)
    monkeypatch.setitem(sys.modules, "cv2", _FAKE_CV2)

    video_np, audio_np = generate_module.generate_video(
        model_repo="dummy/repo",
        text_encoder_repo=None,
        prompt="A singer on stage",
        pipeline=PipelineType.DEV,
        image="start.png",
        audio=True,
        height=64,
        width=64,
        num_frames=9,
        output_path=str(output_path),
        tiling="none",
        verbose=False,
    )

    assert denoise_calls["video_state"] is not None
    assert denoise_calls["audio_shape"][2] > 0
    assert denoise_calls["audio_frozen"] is False
    assert mux_calls["video_path"].name == "i2v-audio.temp.mp4"
    assert mux_calls["audio_path"].name == "i2v-audio.wav"
    assert mux_calls["output_path"] == output_path
    assert output_path.exists()
    assert output_path.with_suffix(".wav").exists()
    assert video_np.shape == (9, 64, 64, 3)
    assert audio_np is not None
    assert audio_np.shape == (2, 1024)


def test_generate_video_auto_selects_latest_x2_upscaler(tmp_path, monkeypatch):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors").touch()
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()

    selected = {}

    class _StopAfterUpscalerSelection(Exception):
        pass

    def fake_denoise_distilled(latents, *_args, audio_latents=None, **_kwargs):
        return latents, audio_latents

    def fake_load_upsampler(path):
        selected["filename"] = Path(path).name
        raise _StopAfterUpscalerSelection()

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
        generate_module, "denoise_distilled", fake_denoise_distilled
    )
    monkeypatch.setattr(generate_module, "load_upsampler", fake_load_upsampler)

    with pytest.raises(_StopAfterUpscalerSelection):
        generate_module.generate_video(
            model_repo="dummy/repo",
            text_encoder_repo=None,
            prompt="A sunset over the sea",
            pipeline=PipelineType.DISTILLED,
            height=64,
            width=64,
            num_frames=9,
            output_path=str(tmp_path / "out.mp4"),
            tiling="none",
            verbose=False,
        )

    assert selected["filename"] == "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"


def test_distilled_i2v_defers_stage2_image_encoding_until_after_stage1(
    tmp_path, monkeypatch
):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()

    encoder_calls = []
    denoise_encoder_counts = []

    class _RecordingVideoEncoder(_FakeVideoEncoder):
        def __call__(self, image_tensor):
            encoder_calls.append(tuple(image_tensor.shape))
            return super().__call__(image_tensor)

    class _StopAfterStage2Denoise(Exception):
        pass

    def fake_denoise_distilled(latents, *_args, audio_latents=None, **_kwargs):
        denoise_encoder_counts.append(len(encoder_calls))
        if len(denoise_encoder_counts) == 2:
            raise _StopAfterStage2Denoise
        return latents, audio_latents

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
        "VideoEncoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _RecordingVideoEncoder()
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _FakeVideoDecoder()
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "load_image",
        lambda _image, height, width, dtype: mx.zeros((height, width, 3), dtype=dtype),
    )
    monkeypatch.setattr(
        generate_module,
        "prepare_image_for_encoding",
        lambda _image, height, width, dtype: mx.zeros(
            (1, 3, 1, height, width), dtype=dtype
        ),
    )
    monkeypatch.setattr(
        generate_module, "denoise_distilled", fake_denoise_distilled
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (types.SimpleNamespace(parameters=lambda: mx.array(0.0, dtype=mx.float32)), 2.0),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: mx.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3] * 2,
                latents.shape[4] * 2,
            ),
            dtype=latents.dtype,
        ),
    )

    with pytest.raises(_StopAfterStage2Denoise):
        generate_module.generate_video(
            model_repo="dummy/repo",
            text_encoder_repo=None,
            prompt="A scenic ocean",
            pipeline=PipelineType.DISTILLED,
            image="start.png",
            height=64,
            width=64,
            num_frames=9,
            output_path=str(tmp_path / "out.mp4"),
            tiling="none",
            verbose=False,
        )

    assert denoise_encoder_counts == [1, 2]


def test_distilled_low_memory_stage2_skips_audio_refinement_but_keeps_audio_output(
    tmp_path, monkeypatch
):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    output_path = tmp_path / "low-memory-audio.mp4"

    calls = {"denoise_audio_args": [], "decoded_audio_mean": None, "events": []}

    class _RecordingAudioDecoder(_FakeAudioDecoder):
        def __call__(self, audio_latents):
            calls["decoded_audio_mean"] = float(mx.mean(audio_latents).item())
            return super().__call__(audio_latents)

    def fake_denoise_distilled(latents, *_args, audio_latents=None, **_kwargs):
        calls["denoise_audio_args"].append(audio_latents)
        if len(calls["denoise_audio_args"]) == 1:
            assert audio_latents is not None
            preserved = mx.full(audio_latents.shape, 7.0, dtype=audio_latents.dtype)
            return latents, preserved
        return latents, audio_latents

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
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _FakeVideoDecoder()
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "denoise_distilled",
        fake_denoise_distilled,
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (
            types.SimpleNamespace(parameters=lambda: mx.array(0.0, dtype=mx.float32)),
            2.0,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: (
            calls["events"].append("upsample")
            or mx.zeros(
                (
                    latents.shape[0],
                    latents.shape[1],
                    latents.shape[2],
                    latents.shape[3] * 2,
                    latents.shape[4] * 2,
                ),
                dtype=latents.dtype,
            )
        ),
    )

    def fake_strip_transformer_to_video_only(_model):
        calls["model_trimmed"] = calls.get("model_trimmed", 0) + 1
        calls["events"].append("trim")

    monkeypatch.setattr(
        generate_module,
        "strip_transformer_to_video_only",
        fake_strip_transformer_to_video_only,
    )
    monkeypatch.setattr(
        generate_module,
        "load_audio_decoder",
        lambda *_args, **_kwargs: _RecordingAudioDecoder(),
    )
    monkeypatch.setattr(
        generate_module,
        "load_vocoder_model",
        lambda *_args, **_kwargs: _FakeVocoder(),
    )
    monkeypatch.setattr(
        generate_module,
        "save_audio",
        lambda audio, path, sample_rate: Path(path).write_bytes(
            f"sr={sample_rate},shape={audio.shape}".encode("utf-8")
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "mux_video_audio",
        lambda _video, _audio, final_output: Path(final_output).write_bytes(b"muxed")
        or True,
    )
    monkeypatch.setitem(sys.modules, "cv2", _FAKE_CV2)

    video_np, audio_np = generate_module.generate_video(
        model_repo="dummy/repo",
        text_encoder_repo=None,
        prompt="A singer on stage",
        pipeline=PipelineType.DISTILLED,
        audio=True,
        low_memory=True,
        height=64,
        width=64,
        num_frames=9,
        output_path=str(output_path),
        tiling="none",
        verbose=False,
    )

    assert len(calls["denoise_audio_args"]) == 2
    assert calls["denoise_audio_args"][0] is not None
    assert calls["denoise_audio_args"][1] is None
    assert calls["model_trimmed"] == 1
    assert calls["events"][:2] == ["trim", "upsample"]
    assert calls["decoded_audio_mean"] == 7.0
    assert output_path.exists()
    assert video_np.shape == (9, 64, 64, 3)
    assert audio_np is not None


def test_distilled_skip_stage2_refinement_decodes_upsampled_stage1_result(
    tmp_path, monkeypatch
):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    output_path = tmp_path / "skip-stage2.mp4"
    calls = {"denoise": 0}

    def fake_denoise_distilled(latents, *_args, audio_latents=None, **_kwargs):
        calls["denoise"] += 1
        return latents, audio_latents

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
        "denoise_distilled",
        fake_denoise_distilled,
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (
            types.SimpleNamespace(parameters=lambda: mx.array(0.0, dtype=mx.float32)),
            2.0,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: mx.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3] * 2,
                latents.shape[4] * 2,
            ),
            dtype=latents.dtype,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _FakeVideoDecoder()
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", _FAKE_CV2)

    video_np = generate_module.generate_video(
        model_repo="dummy/repo",
        text_encoder_repo=None,
        prompt="demo prompt",
        pipeline=PipelineType.DISTILLED,
        height=64,
        width=64,
        num_frames=9,
        output_path=str(output_path),
        verbose=False,
        tiling="none",
        skip_stage2_refinement=True,
    )

    assert calls["denoise"] == 1
    assert video_np.shape == (9, 64, 64, 3)


def test_distilled_partial_stage2_refinement_uses_shortened_sigma_schedule(
    tmp_path, monkeypatch
):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    output_path = tmp_path / "stage2-lite.mp4"
    calls = {"sigmas": []}

    def fake_denoise_distilled(latents, _positions, _embeddings, _transformer, sigmas, *_args, audio_latents=None, **_kwargs):
        calls["sigmas"].append(list(sigmas))
        return latents, audio_latents

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
        "denoise_distilled",
        fake_denoise_distilled,
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (
            types.SimpleNamespace(parameters=lambda: mx.array(0.0, dtype=mx.float32)),
            2.0,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "load_video_decoder_statistics",
        lambda *_args, **_kwargs: (
            mx.zeros((1,), dtype=mx.float32),
            mx.ones((1,), dtype=mx.float32),
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: mx.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3] * 2,
                latents.shape[4] * 2,
            ),
            dtype=latents.dtype,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _FakeVideoDecoder()
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", _FAKE_CV2)

    video_np = generate_module.generate_video(
        model_repo="dummy/repo",
        text_encoder_repo=None,
        prompt="demo prompt",
        pipeline=PipelineType.DISTILLED,
        height=64,
        width=64,
        num_frames=9,
        output_path=str(output_path),
        verbose=False,
        tiling="none",
        stage2_refinement_steps=1,
    )

    assert calls["sigmas"][0] == generate_module.STAGE_1_SIGMAS
    assert calls["sigmas"][1] == [0.909375, 0.0]
    assert video_np.shape == (9, 64, 64, 3)


def test_low_memory_large_short_decode_prefers_spatial_tiling(tmp_path, monkeypatch):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    output_path = tmp_path / "decode-escalated.mp4"
    decode_calls = {}

    class _RecordingDecoder(_FakeVideoDecoder):
        def decode_tiled(self, latents, **kwargs):
            decode_calls["tiling_mode"] = kwargs.get("tiling_mode")
            decode_calls["tiling_config"] = kwargs.get("tiling_config")
            return super().decode_tiled(latents, **kwargs)

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
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _RecordingDecoder()
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "denoise_distilled",
        lambda latents, *_args, audio_latents=None, **_kwargs: (latents, audio_latents),
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (
            types.SimpleNamespace(parameters=lambda: mx.array(0.0, dtype=mx.float32)),
            2.0,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: mx.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3] * 2,
                latents.shape[4] * 2,
            ),
            dtype=latents.dtype,
        ),
    )
    monkeypatch.setattr(generate_module, "strip_transformer_to_video_only", lambda _m: None)
    monkeypatch.setitem(sys.modules, "cv2", _FAKE_CV2)

    video_np = generate_module.generate_video(
        model_repo="dummy/repo",
        text_encoder_repo=None,
        prompt="A large scenic landscape",
        pipeline=PipelineType.DISTILLED,
        low_memory=True,
        height=1344,
        width=768,
        num_frames=97,
        output_path=str(output_path),
        tiling="conservative",
        verbose=False,
    )

    assert decode_calls["tiling_mode"] == "spatial"
    assert decode_calls["tiling_config"].spatial_config.tile_size_in_pixels == 768
    assert decode_calls["tiling_config"].temporal_config is None
    assert video_np.shape == (97, 1344, 768, 3)


def test_low_memory_tiled_decode_can_stream_without_returning_video(
    tmp_path, monkeypatch
):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    output_path = tmp_path / "streamed-low-memory.mp4"
    decode_calls = {}

    class _RecordingDecoder(_FakeVideoDecoder):
        def decode_tiled(self, latents, **kwargs):
            decode_calls["return_output"] = kwargs.get("return_output")
            on_frames_ready = kwargs.get("on_frames_ready")
            frames = mx.zeros((1, 3, 9, 64, 64), dtype=mx.float32)
            on_frames_ready(frames, 0)
            return None

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
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _RecordingDecoder()
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "denoise_distilled",
        lambda latents, *_args, audio_latents=None, **_kwargs: (latents, audio_latents),
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (
            types.SimpleNamespace(parameters=lambda: mx.array(0.0, dtype=mx.float32)),
            2.0,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "load_video_decoder_statistics",
        lambda *_args, **_kwargs: (
            mx.zeros((1,), dtype=mx.float32),
            mx.ones((1,), dtype=mx.float32),
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: mx.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3] * 2,
                latents.shape[4] * 2,
            ),
            dtype=latents.dtype,
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", _FAKE_CV2)

    video_np = generate_module.generate_video(
        model_repo="dummy/repo",
        text_encoder_repo=None,
        prompt="demo prompt",
        pipeline=PipelineType.DISTILLED,
        height=64,
        width=64,
        num_frames=9,
        output_path=str(output_path),
        verbose=False,
        tiling="conservative",
        low_memory=True,
        return_video=False,
    )

    assert decode_calls["return_output"] is False
    assert video_np is None
    assert output_path.exists()


def test_low_memory_streaming_short_clip_spatial_decode_adds_temporal_tiling(
    tmp_path, monkeypatch
):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    output_path = tmp_path / "streamed-short-clip.mp4"
    decode_calls = {}

    class _RecordingDecoder(_FakeVideoDecoder):
        def decode_tiled(self, latents, **kwargs):
            decode_calls["tiling_config"] = kwargs.get("tiling_config")
            decode_calls["return_output"] = kwargs.get("return_output")
            on_frames_ready = kwargs.get("on_frames_ready")
            frames = mx.zeros((1, 3, 73, 64, 64), dtype=mx.float32)
            on_frames_ready(frames[:, :, :49, :, :], 0)
            on_frames_ready(frames[:, :, 49:, :, :], 49)
            return None

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
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _RecordingDecoder()
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "denoise_distilled",
        lambda latents, *_args, audio_latents=None, **_kwargs: (latents, audio_latents),
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (
            types.SimpleNamespace(parameters=lambda: mx.array(0.0, dtype=mx.float32)),
            2.0,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "load_video_decoder_statistics",
        lambda *_args, **_kwargs: (
            mx.zeros((1,), dtype=mx.float32),
            mx.ones((1,), dtype=mx.float32),
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: mx.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3] * 2,
                latents.shape[4] * 2,
            ),
            dtype=latents.dtype,
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", _FAKE_CV2)

    video_np = generate_module.generate_video(
        model_repo="dummy/repo",
        text_encoder_repo=None,
        prompt="demo prompt",
        pipeline=PipelineType.DISTILLED,
        height=1344,
        width=768,
        num_frames=73,
        output_path=str(output_path),
        verbose=False,
        tiling="conservative",
        low_memory=True,
        return_video=False,
    )

    assert decode_calls["return_output"] is False
    assert decode_calls["tiling_config"].spatial_config.tile_size_in_pixels == 768
    assert decode_calls["tiling_config"].temporal_config.tile_size_in_frames == 64
    assert video_np is None


def test_low_memory_streaming_longer_short_clip_uses_smaller_temporal_tiles(
    tmp_path, monkeypatch
):
    model_path = _prepare_fake_model_dir(tmp_path / "fake-model")
    (model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
    output_path = tmp_path / "streamed-longer-short-clip.mp4"
    decode_calls = {}

    class _RecordingDecoder(_FakeVideoDecoder):
        def decode_tiled(self, latents, **kwargs):
            decode_calls["tiling_config"] = kwargs.get("tiling_config")
            decode_calls["return_output"] = kwargs.get("return_output")
            on_frames_ready = kwargs.get("on_frames_ready")
            frames = mx.zeros((1, 3, 121, 64, 64), dtype=mx.float32)
            on_frames_ready(frames[:, :, :25, :, :], 0)
            on_frames_ready(frames[:, :, 25:57, :, :], 25)
            on_frames_ready(frames[:, :, 57:89, :, :], 57)
            on_frames_ready(frames[:, :, 89:, :, :], 89)
            return None

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
        "VideoDecoder",
        types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _RecordingDecoder()
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "denoise_distilled",
        lambda latents, *_args, audio_latents=None, **_kwargs: (latents, audio_latents),
    )
    monkeypatch.setattr(
        generate_module,
        "load_upsampler",
        lambda *_args, **_kwargs: (
            types.SimpleNamespace(parameters=lambda: mx.array(0.0, dtype=mx.float32)),
            2.0,
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "load_video_decoder_statistics",
        lambda *_args, **_kwargs: (
            mx.zeros((1,), dtype=mx.float32),
            mx.ones((1,), dtype=mx.float32),
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "upsample_latents",
        lambda latents, *_args, **_kwargs: mx.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                latents.shape[2],
                latents.shape[3] * 2,
                latents.shape[4] * 2,
            ),
            dtype=latents.dtype,
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", _FAKE_CV2)

    video_np = generate_module.generate_video(
        model_repo="dummy/repo",
        text_encoder_repo=None,
        prompt="demo prompt",
        pipeline=PipelineType.DISTILLED,
        height=1344,
        width=768,
        num_frames=121,
        output_path=str(output_path),
        verbose=False,
        tiling="conservative",
        low_memory=True,
        return_video=False,
    )

    assert decode_calls["return_output"] is False
    assert decode_calls["tiling_config"].spatial_config.tile_size_in_pixels == 768
    assert decode_calls["tiling_config"].temporal_config.tile_size_in_frames == 32
    assert decode_calls["tiling_config"].temporal_config.tile_overlap_in_frames == 8
    assert video_np is None
