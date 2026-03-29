"""Tests for LTX memory profiling helpers."""

from mlx_video.models.ltx_2.generate import (
    MemoryProfiler,
    resolve_attention_query_chunk_size,
    resolve_decode_tiling_mode,
    resolve_stage2_sigma_schedule,
)


class _FakeConsole:
    def __init__(self):
        self.messages = []

    def print(self, message):
        self.messages.append(message)


def test_memory_profiler_disabled_uses_global_peak(monkeypatch):
    monkeypatch.setattr("mlx_video.models.ltx_2.generate.mx.get_peak_memory", lambda: 123)
    profiler = MemoryProfiler(enabled=False, console=_FakeConsole())
    assert profiler.final_peak() == 123


def test_memory_profiler_tracks_phase_peak(monkeypatch):
    console = _FakeConsole()
    peak_values = iter([300, 500])
    active_values = iter([100, 150])
    cache_values = iter([50, 70])

    monkeypatch.setattr("mlx_video.models.ltx_2.generate.mx.eval", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlx_video.models.ltx_2.generate.mx.reset_peak_memory", lambda: None)
    monkeypatch.setattr(
        "mlx_video.models.ltx_2.generate.mx.get_peak_memory",
        lambda: next(peak_values),
    )
    monkeypatch.setattr(
        "mlx_video.models.ltx_2.generate.mx.get_active_memory",
        lambda: next(active_values),
    )
    monkeypatch.setattr(
        "mlx_video.models.ltx_2.generate.mx.get_cache_memory",
        lambda: next(cache_values),
    )

    profiler = MemoryProfiler(enabled=True, console=console)
    profiler.start("stage")
    profiler.log("stage done")

    assert profiler.final_peak() == 500
    assert console.messages
    assert "stage done" in console.messages[0]


def test_resolve_decode_tiling_mode_prefers_spatial_for_large_short_low_memory_decode():
    assert (
        resolve_decode_tiling_mode(
            "conservative",
            low_memory=True,
            height=1344,
            width=768,
            num_frames=97,
        )
        == "spatial"
    )


def test_resolve_decode_tiling_mode_keeps_small_decode_conservative():
    assert (
        resolve_decode_tiling_mode(
            "conservative",
            low_memory=True,
            height=512,
            width=512,
            num_frames=33,
        )
        == "conservative"
    )


def test_resolve_decode_tiling_mode_keeps_aggressive_for_long_low_memory_decode():
    assert (
        resolve_decode_tiling_mode(
            "conservative",
            low_memory=True,
            height=1088,
            width=1920,
            num_frames=481,
        )
        == "aggressive"
    )


def test_resolve_attention_query_chunk_size_disables_for_small_token_counts():
    assert resolve_attention_query_chunk_size(low_memory=True, num_tokens=4_096) is None


def test_resolve_attention_query_chunk_size_uses_medium_chunk_for_mid_sized_inputs():
    assert (
        resolve_attention_query_chunk_size(low_memory=True, num_tokens=20_000)
        == 4_096
    )


def test_resolve_attention_query_chunk_size_uses_small_chunk_for_large_inputs():
    assert (
        resolve_attention_query_chunk_size(low_memory=True, num_tokens=70_000)
        == 2_048
    )


def test_resolve_attention_query_chunk_size_prefers_memory_for_stage2_even_without_low_memory():
    assert (
        resolve_attention_query_chunk_size(
            low_memory=False,
            num_tokens=20_000,
            prefer_memory=True,
        )
        == 4_096
    )


def test_resolve_stage2_sigma_schedule_for_one_step():
    assert resolve_stage2_sigma_schedule(1) == [0.909375, 0.0]


def test_resolve_stage2_sigma_schedule_for_two_steps():
    assert resolve_stage2_sigma_schedule(2) == [0.909375, 0.421875, 0.0]
