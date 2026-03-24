"""Backward-compatible helpers for the legacy dev-only LTX entrypoint."""

from typing import Optional

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


def generate_video_dev(
    model_repo: str = "prince-canuma/LTX-2.3-dev",
    text_encoder_repo: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    num_inference_steps: int = 40,
    cfg_scale: float = 4.0,
    audio_cfg_scale: float = 7.0,
    cfg_rescale: float = 0.0,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output.mp4",
    save_frames: bool = False,
    verbose: bool = True,
    enhance_prompt: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image: Optional[str] = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    tiling: str = "none",
    stream: bool = False,
    audio: bool = False,
    output_audio_path: Optional[str] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 1.0,
    stg_blocks: Optional[list] = None,
    modality_scale: float = 3.0,
    lora_path: Optional[str] = None,
    lora_strength: float = 1.0,
    lora_strength_stage_1: Optional[float] = None,
    lora_strength_stage_2: Optional[float] = None,
    audio_file: Optional[str] = None,
    audio_start_time: float = 0.0,
    spatial_upscaler: Optional[str] = None,
    low_memory: bool = False,
    profile_memory: bool = False,
):
    """Compatibility wrapper for the historical dev-only API.

    The modern implementation lives in `mlx_video.models.ltx_2.generate.generate_video`.
    This wrapper keeps the old import path working while forcing the `dev` pipeline
    and preserving the old `tiling="none"` default expected by existing callers.
    """

    return generate_video(
        model_repo=model_repo,
        text_encoder_repo=text_encoder_repo,
        prompt=prompt,
        pipeline=PipelineType.DEV,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        audio_cfg_scale=audio_cfg_scale,
        cfg_rescale=cfg_rescale,
        seed=seed,
        fps=fps,
        output_path=output_path,
        save_frames=save_frames,
        verbose=verbose,
        enhance_prompt=enhance_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        image=image,
        image_strength=image_strength,
        image_frame_idx=image_frame_idx,
        tiling=tiling,
        stream=stream,
        audio=audio,
        output_audio_path=output_audio_path,
        use_apg=use_apg,
        apg_eta=apg_eta,
        apg_norm_threshold=apg_norm_threshold,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        modality_scale=modality_scale,
        lora_path=lora_path,
        lora_strength=lora_strength,
        lora_strength_stage_1=lora_strength_stage_1,
        lora_strength_stage_2=lora_strength_stage_2,
        audio_file=audio_file,
        audio_start_time=audio_start_time,
        spatial_upscaler=spatial_upscaler,
        low_memory=low_memory,
        profile_memory=profile_memory,
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
    "generate_video_dev",
    "ltx2_scheduler",
    "main",
]


if __name__ == "__main__":
    main()
