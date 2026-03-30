"""Regression tests for distilled MLX compile helpers."""

import mlx.core as mx

import mlx_video.models.ltx_2.generate as generate_module
from mlx_video.models.ltx_2.conditioning.latent import LatentState


class _FakeDistilledTransformer:
    def __init__(self, velocity_value: float = 0.0):
        self.velocity_value = velocity_value
        self.calls = 0

    def __call__(self, video=None, audio=None):
        self.calls += 1
        video_velocity = mx.zeros_like(video.latent) + self.velocity_value
        audio_velocity = (
            mx.zeros_like(audio.latent) + self.velocity_value if audio is not None else None
        )
        return video_velocity, audio_velocity


def test_denoise_distilled_reuses_cached_forward_per_mode_without_mx_compile(
    monkeypatch,
):
    monkeypatch.setattr(
        generate_module.mx,
        "compile",
        lambda _fn: (_ for _ in ()).throw(AssertionError("mx.compile should not run")),
    )

    transformer = _FakeDistilledTransformer()
    latents = mx.zeros((1, 2, 1, 1, 1), dtype=mx.float32)
    positions = mx.zeros((1, 3, 1, 2), dtype=mx.float32)
    text_embeddings = mx.zeros((1, 1, 4), dtype=mx.float32)

    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
    )
    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
    )

    assert hasattr(transformer, "_compiled_distilled_video")

    audio_latents = mx.zeros((1, 2, 1, 2), dtype=mx.float32)
    audio_positions = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    audio_embeddings = mx.zeros((1, 1, 4), dtype=mx.float32)

    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
    )
    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
    )

    assert hasattr(transformer, "_compiled_distilled_av")


def test_denoise_distilled_preserves_i2v_clean_tokens():
    transformer = _FakeDistilledTransformer(velocity_value=1.0)
    latents = mx.zeros((1, 1, 2, 1, 1), dtype=mx.float32)
    positions = mx.zeros((1, 3, 2, 2), dtype=mx.float32)
    text_embeddings = mx.zeros((1, 1, 4), dtype=mx.float32)

    state = LatentState(
        latent=mx.array([[[[[3.0]], [[5.0]]]]], dtype=mx.float32),
        clean_latent=mx.array([[[[[9.0]], [[7.0]]]]], dtype=mx.float32),
        denoise_mask=mx.array([[[[[1.0]], [[0.0]]]]], dtype=mx.float32),
    )

    output, audio_output = generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        state=state,
    )

    expected = mx.array([[[[[2.0]], [[7.0]]]]], dtype=mx.float32)
    assert mx.allclose(output, expected)
    assert audio_output is None


def test_denoise_distilled_compiles_stage2_outer_function_once_per_mode(monkeypatch):
    compiled = []

    def fake_compile(fn):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(generate_module.mx, "compile", fake_compile)

    transformer = _FakeDistilledTransformer()
    latents = mx.zeros((1, 2, 1, 1, 1), dtype=mx.float32)
    positions = mx.zeros((1, 3, 1, 2), dtype=mx.float32)
    text_embeddings = mx.zeros((1, 1, 4), dtype=mx.float32)

    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        compile_outer_transformer=True,
        compile_cache_scope="stage2",
    )
    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        compile_outer_transformer=True,
        compile_cache_scope="stage2",
    )

    assert hasattr(transformer, "_compiled_distilled_stage2_video")
    assert len(compiled) == 1

    audio_latents = mx.zeros((1, 2, 1, 2), dtype=mx.float32)
    audio_positions = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    audio_embeddings = mx.zeros((1, 1, 4), dtype=mx.float32)

    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
        compile_outer_transformer=True,
        compile_cache_scope="stage2",
    )
    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
        compile_outer_transformer=True,
        compile_cache_scope="stage2",
    )

    assert hasattr(transformer, "_compiled_distilled_stage2_av")
    assert len(compiled) == 2


def test_denoise_distilled_compiles_stage1_outer_function_once_per_mode(monkeypatch):
    compiled = []

    def fake_compile(fn):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(generate_module.mx, "compile", fake_compile)

    transformer = _FakeDistilledTransformer()
    latents = mx.zeros((1, 2, 1, 1, 1), dtype=mx.float32)
    positions = mx.zeros((1, 3, 1, 2), dtype=mx.float32)
    text_embeddings = mx.zeros((1, 1, 4), dtype=mx.float32)

    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        compile_outer_transformer=True,
    )
    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        compile_outer_transformer=True,
    )

    assert hasattr(transformer, "_compiled_distilled_stage1_video")
    assert len(compiled) == 1

    audio_latents = mx.zeros((1, 2, 1, 2), dtype=mx.float32)
    audio_positions = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    audio_embeddings = mx.zeros((1, 1, 4), dtype=mx.float32)

    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
        compile_outer_transformer=True,
    )
    generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [1.0, 0.0],
        verbose=False,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
        compile_outer_transformer=True,
    )

    assert hasattr(transformer, "_compiled_distilled_stage1_av")
    assert len(compiled) == 2


def test_denoise_distilled_applies_same_euler_schedule_to_audio(monkeypatch):
    monkeypatch.setattr(generate_module.mx, "compile", lambda fn: fn)

    transformer = _FakeDistilledTransformer(velocity_value=1.0)
    latents = mx.full((1, 1, 1, 1, 1), 10.0, dtype=mx.float32)
    positions = mx.zeros((1, 3, 1, 2), dtype=mx.float32)
    text_embeddings = mx.zeros((1, 1, 4), dtype=mx.float32)
    audio_latents = mx.full((1, 1, 1, 1), 10.0, dtype=mx.float32)
    audio_positions = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    audio_embeddings = mx.zeros((1, 1, 4), dtype=mx.float32)

    output, audio_output = generate_module.denoise_distilled(
        latents,
        positions,
        text_embeddings,
        transformer,
        [2.0, 1.0, 0.0],
        verbose=False,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_embeddings=audio_embeddings,
    )

    expected_video = mx.full((1, 1, 1, 1, 1), 8.0, dtype=mx.float32)
    expected_audio = mx.full((1, 1, 1, 1), 8.0, dtype=mx.float32)
    assert mx.allclose(output, expected_video)
    assert audio_output is not None
    assert mx.allclose(audio_output, expected_audio)
