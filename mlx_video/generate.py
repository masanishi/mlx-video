"""Backward-compatible LTX generation entrypoint.

Prefer `mlx_video.ltx_2.generate`, but keep the legacy import/module path working.
"""

from mlx_video.models.ltx_2.generate import (
    AUDIO_LATENTS_PER_SECOND,
    AUDIO_SAMPLE_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    PipelineType,
    cfg_delta,
    compute_audio_frames,
    create_audio_position_grid,
    create_position_grid,
    generate_video,
    ltx2_scheduler,
    main,
)

__all__ = [
    "AUDIO_LATENTS_PER_SECOND",
    "AUDIO_SAMPLE_RATE",
    "DEFAULT_NEGATIVE_PROMPT",
    "PipelineType",
    "cfg_delta",
    "compute_audio_frames",
    "create_audio_position_grid",
    "create_position_grid",
    "generate_video",
    "ltx2_scheduler",
    "main",
]


if __name__ == "__main__":
    main()
