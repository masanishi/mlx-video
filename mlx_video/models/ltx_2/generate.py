"""Unified video and audio-video generation pipeline for LTX-2 / LTX-2.3.

Supports both distilled (two-stage with upsampling) and dev (single-stage with CFG) pipelines.
"""

import argparse
import gc
import math
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from safetensors import safe_open

# Rich console for styled output
console = Console()


from mlx_video.models.ltx_2.conditioning import (
    VideoConditionByLatentIndex,
    apply_conditioning,
)
from mlx_video.models.ltx_2.conditioning.latent import LatentState, apply_denoise_mask
from mlx_video.models.ltx_2.config import LTXModelType
from mlx_video.models.ltx_2.ltx_2 import LTXModel, TransformerArgsPreprocessor
from mlx_video.models.ltx_2.transformer import Modality
from mlx_video.models.ltx_2.upsampler import load_upsampler, upsample_latents
from mlx_video.models.ltx_2.video_vae import VideoEncoder
from mlx_video.models.ltx_2.video_vae.decoder import VideoDecoder
from mlx_video.models.ltx_2.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from mlx_video.quantization import (
    dequantize_linear_weight,
    is_quantized_linear_module,
    requantize_linear_module_like,
)
from mlx_video.utils import (
    get_model_path,
    load_image,
    normalize_quantization_config,
    prepare_image_for_encoding,
    resolve_safetensor_files,
)


class PipelineType(Enum):
    """Pipeline type selector."""

    DISTILLED = "distilled"  # Two-stage with upsampling, fixed sigmas, no CFG
    DEV = "dev"  # Single-stage, dynamic sigmas, CFG
    DEV_TWO_STAGE = (
        "dev-two-stage"  # Two-stage: dev (half res, CFG) + distilled LoRA (full res)
    )
    DEV_TWO_STAGE_HQ = "dev-two-stage-hq"  # Two-stage: res_2s sampler, LoRA both stages


# Distilled model sigma schedules
STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
OFFICIAL_DEV_TWO_STAGE_STAGE_2_SIGMAS = [0.85, 0.7250, 0.4219, 0.0]

# Dev model scheduling constants
BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

# Audio constants
AUDIO_SAMPLE_RATE = 24000  # Output audio sample rate
AUDIO_LATENT_SAMPLE_RATE = 16000  # VAE internal sample rate
AUDIO_HOP_LENGTH = 160
AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
AUDIO_LATENT_CHANNELS = 8  # Latent channels before patchifying
AUDIO_MEL_BINS = 16
AUDIO_LATENTS_PER_SECOND = (
    AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR
)  # 25

# Default external text encoder repo for reformatted model repos without
# embedded Gemma weights (e.g. LTX-2.3 pre-converted checkpoints).
DEFAULT_DISTILLED_MODEL_REPO = "prince-canuma/LTX-2.3-distilled"
DEFAULT_TEXT_ENCODER_REPO = "google/gemma-3-12b-it"
TEXT_ENCODER_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.model"]
LORA_MERGE_BATCH_SIZE = 8
TRANSFORMER_QUANTIZATION_MODES = ("affine", "mxfp8")

# Default negative prompt for CFG (dev pipeline)
# Matches PyTorch LTX-2 reference DEFAULT_NEGATIVE_PROMPT from constants.py
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

_VIDEO_DECODER_STATS_CACHE: dict[Path, tuple[mx.array, mx.array]] = {}


def resolve_lora_file(lora_path: str) -> Path:
    """Resolve a LoRA input path to a concrete safetensors file."""
    lora_file = Path(lora_path)
    if lora_file.is_file():
        return lora_file
    if lora_file.is_dir():
        candidates = sorted(lora_file.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No .safetensors files found in {lora_path}")
        lora_candidates = [c for c in candidates if "distilled-lora" in c.name]
        resolved = lora_candidates[0] if lora_candidates else candidates[0]
        console.print(f"[dim]Using LoRA file: {resolved.name}[/]")
        return resolved

    lora_dir = get_model_path(lora_path)
    candidates = sorted(lora_dir.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors files found in {lora_dir}")
    lora_candidates = [c for c in candidates if "distilled-lora" in c.name]
    resolved = lora_candidates[0] if lora_candidates else candidates[0]
    console.print(f"[dim]Using LoRA from repo: {lora_path} ({resolved.name})[/]")
    return resolved


def _build_runtime_quantized_lora_preprocessor(lora_path: str, strength: float):
    """Build a shard-local LoRA preprocessor for runtime quantization loaders."""
    from mlx_video.lora.apply import apply_loras_to_weights
    from mlx_video.lora.loader import load_multiple_loras
    from mlx_video.lora.types import LoRAConfig

    resolved_lora = resolve_lora_file(lora_path)
    module_to_loras = load_multiple_loras(
        [LoRAConfig(path=resolved_lora, strength=strength)]
    )

    def preprocess(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return apply_loras_to_weights(
            weights,
            module_to_loras,
            quantization=None,
            report=False,
        )

    return preprocess, resolved_lora


def load_and_merge_lora(
    model: LTXModel,
    lora_path: str,
    strength: float = 1.0,
) -> None:
    """Load LoRA weights and merge them into the transformer model in-place.

    Supports two formats:
    - Raw PyTorch: keys like diffusion_model.{module}.lora_A.weight (needs sanitization)
    - Pre-converted MLX: keys like {module}.lora_A.weight (already sanitized)

    Merge formula: weight += (lora_B * strength) @ lora_A

    Args:
        model: The LTXModel transformer to merge into
        lora_path: Path to the LoRA safetensors file or directory containing one
        strength: LoRA strength/coefficient (default 1.0)
    """
    # Resolve path: local file/dir or HuggingFace repo
    lora_file = resolve_lora_file(lora_path)

    # Load LoRA weights
    lora_weights = mx.load(str(lora_file))

    # Detect format: raw PyTorch has 'diffusion_model.' prefix
    has_prefix = any(k.startswith("diffusion_model.") for k in lora_weights)

    # Group into A/B pairs by module name, storing source keys so weights can be
    # dropped from memory as soon as they are merged.
    lora_pairs = {}
    for key in lora_weights:
        module_key = key
        if has_prefix:
            if not key.startswith("diffusion_model."):
                continue
            module_key = key.replace("diffusion_model.", "")

        if module_key.endswith(".lora_A.weight"):
            base_key = module_key.replace(".lora_A.weight", "")
            lora_pairs.setdefault(base_key, {})["A_key"] = key
        elif module_key.endswith(".lora_B.weight"):
            base_key = module_key.replace(".lora_B.weight", "")
            lora_pairs.setdefault(base_key, {})["B_key"] = key
        elif module_key.endswith(".alpha"):
            base_key = module_key.replace(".alpha", "")
            lora_pairs.setdefault(base_key, {})["alpha_key"] = key

    # Apply key sanitization only for raw PyTorch format
    # Replacements handle both mid-string and end-of-string positions
    # since LoRA base keys end at the module name without trailing dot
    _LORA_KEY_REPLACEMENTS = [
        (".to_out.0", ".to_out"),
        (".ff.net.0.proj", ".ff.proj_in"),
        (".ff.net.2", ".ff.proj_out"),
        (".audio_ff.net.0.proj", ".audio_ff.proj_in"),
        (".audio_ff.net.2", ".audio_ff.proj_out"),
        (".linear_1", ".linear1"),
        (".linear_2", ".linear2"),
    ]
    if has_prefix:
        sanitized_pairs = {}
        for key, pair in lora_pairs.items():
            new_key = key
            for old, new in _LORA_KEY_REPLACEMENTS:
                if new_key.endswith(old):
                    new_key = new_key[: -len(old)] + new
                else:
                    new_key = new_key.replace(old + ".", new + ".")
            sanitized_pairs[new_key] = pair
    else:
        sanitized_pairs = lora_pairs

    # Get current model weights as a flat dict (references, not copies)
    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, full_key))
            else:
                flat[full_key] = v
        return flat

    def resolve_module(module_key: str):
        parts = module_key.split(".")
        parent = model
        try:
            for part in parts[:-1]:
                parent = (
                    getattr(parent, part) if not part.isdigit() else parent[int(part)]
                )
            leaf_name = parts[-1]
            target = (
                getattr(parent, leaf_name)
                if not leaf_name.isdigit()
                else parent[int(leaf_name)]
            )
        except (AttributeError, IndexError, TypeError, KeyError):
            return None, None, None
        return parent, leaf_name, target

    def merge_into_quantized_linear(
        module_key: str,
        lora_a: mx.array,
        lora_b: mx.array,
        alpha: float,
    ) -> bool:
        parent, leaf_name, target = resolve_module(module_key)
        if not is_quantized_linear_module(target):
            return False

        # 日本語概要:
        # 量子化済み layer は 1 layer ずつ dequantize -> LoRA merge -> 再量子化する。
        # これなら常駐メモリをほぼ増やさず、runtime で余計な LoRA matmul も発生しない。
        base_weight = dequantize_linear_weight(target)
        base_weight_f32 = base_weight.astype(mx.float32)
        delta = (lora_b * (strength * (alpha / lora_a.shape[0]))) @ lora_a
        merged_weight = base_weight_f32 + delta.astype(mx.float32)
        requantized = requantize_linear_module_like(target, merged_weight)
        if "bias" in target:
            requantized.bias = target.bias
        if leaf_name.isdigit():
            parent[int(leaf_name)] = requantized
        else:
            setattr(parent, leaf_name, requantized)
        return True

    flat_weights = flatten_params(dict(model.parameters()))

    # Merge LoRA deltas in small batches to avoid large transient spikes.
    merged_count = 0
    total_complete_pairs = 0
    skipped_pairs = []
    batch = []
    batch_size = LORA_MERGE_BATCH_SIZE

    for module_key, pair in sanitized_pairs.items():
        if "A_key" not in pair or "B_key" not in pair:
            continue
        total_complete_pairs += 1

        weight_key = f"{module_key}.weight"
        if weight_key not in flat_weights:
            skipped_pairs.append(module_key)
            lora_weights.pop(pair["A_key"], None)
            lora_weights.pop(pair["B_key"], None)
            if "alpha_key" in pair:
                lora_weights.pop(pair["alpha_key"], None)
            continue

        lora_a = lora_weights.pop(pair["A_key"]).astype(mx.float32)  # (rank, in_features)
        lora_b = lora_weights.pop(pair["B_key"]).astype(mx.float32)  # (out_features, rank)
        alpha = (
            float(lora_weights.pop(pair["alpha_key"]).item())
            if "alpha_key" in pair
            else float(lora_a.shape[0])
        )

        if merge_into_quantized_linear(module_key, lora_a, lora_b, alpha):
            flat_weights.pop(weight_key, None)
            del lora_a, lora_b
            merged_count += 1
            continue

        # delta = (lora_B * strength) @ lora_A
        delta = (lora_b * (strength * (alpha / lora_a.shape[0]))) @ lora_a

        base_weight = flat_weights.get(weight_key)
        if base_weight is None:
            skipped_pairs.append(module_key)
            del lora_a, lora_b, delta
            continue

        if base_weight.ndim == 2 and base_weight.shape == delta.shape:
            base_weight = flat_weights.pop(weight_key)
            merged_weight = base_weight + delta.astype(base_weight.dtype)
            batch.append((weight_key, merged_weight))
            del lora_a, lora_b, delta, base_weight
            merged_count += 1
        else:
            raise ValueError(
                f"LoRA shape mismatch for {module_key}: "
                f"base={tuple(base_weight.shape)}, delta={tuple(delta.shape)}. "
                "This LoRA does not match the loaded transformer weights."
            )

        if len(batch) >= batch_size:
            model.load_weights(batch, strict=False)
            mx.eval(model.parameters())
            batch.clear()
            gc.collect()
            mx.clear_cache()

    if batch:
        model.load_weights(batch, strict=False)
        mx.eval(model.parameters())
        batch.clear()
        gc.collect()
        mx.clear_cache()

    del flat_weights, lora_weights
    mx.clear_cache()
    if skipped_pairs:
        sample = ", ".join(skipped_pairs[:5])
        extra = "..." if len(skipped_pairs) > 5 else ""
        console.print(
            f"[yellow]⚠[/] Skipped {len(skipped_pairs)} unmatched LoRA pairs: {sample}{extra}[/]"
        )
    console.print(
        f"[green]✓[/] Merged {merged_count}/{total_complete_pairs} LoRA pairs (strength={strength})"
    )
    invalidate_distilled_transformer_compile_cache(model)


def find_distilled_lora_file(model_path: Path) -> Optional[Path]:
    """Return the first distilled LoRA safetensors file found in a model directory."""
    lora_files = sorted(model_path.glob("*distilled-lora*.safetensors"))
    return lora_files[0] if lora_files else None


def cfg_delta(cond: mx.array, uncond: mx.array, scale: float) -> mx.array:
    """Compute CFG delta for classifier-free guidance.

    Args:
        cond: Conditional prediction
        uncond: Unconditional prediction
        scale: CFG guidance scale

    Returns:
        Delta to add to unconditional for CFG: (scale - 1) * (cond - uncond)
    """
    return (scale - 1.0) * (cond - uncond)


def apg_delta(
    cond: mx.array,
    uncond: mx.array,
    scale: float,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
) -> mx.array:
    """Compute APG (Adaptive Projected Guidance) delta.

    Decomposes guidance into parallel and orthogonal components relative to
    the conditional prediction, providing more stable guidance for I2V.

    Based on: https://arxiv.org/abs/2407.12173

    Args:
        cond: Conditional prediction (x0_pos)
        uncond: Unconditional prediction (x0_neg)
        scale: Guidance strength (same as CFG scale)
        eta: Weight for parallel component (1.0 = keep full parallel)
        norm_threshold: Clamp guidance norm to this value (0 = no clamping)

    Returns:
        Delta to add to unconditional for APG guidance
    """
    guidance = cond - uncond

    # Optionally clamp guidance norm for stability
    if norm_threshold > 0:
        guidance_norm = mx.sqrt(
            mx.sum(guidance**2, axis=(-1, -2, -3), keepdims=True) + 1e-8
        )
        scale_factor = mx.minimum(
            mx.ones_like(guidance_norm), norm_threshold / guidance_norm
        )
        guidance = guidance * scale_factor

    # Project guidance onto cond direction
    batch_size = cond.shape[0]
    cond_flat = mx.reshape(cond, (batch_size, -1))
    guidance_flat = mx.reshape(guidance, (batch_size, -1))

    # Projection coefficient: (guidance · cond) / (cond · cond)
    dot_product = mx.sum(guidance_flat * cond_flat, axis=1, keepdims=True)
    squared_norm = mx.sum(cond_flat**2, axis=1, keepdims=True) + 1e-8
    proj_coeff = dot_product / squared_norm

    # Reshape back and compute parallel/orthogonal components
    proj_coeff = mx.reshape(proj_coeff, (batch_size,) + (1,) * (cond.ndim - 1))
    g_parallel = proj_coeff * cond
    g_orth = guidance - g_parallel

    # Combine with eta weighting parallel component
    g_apg = g_parallel * eta + g_orth

    return g_apg * (scale - 1.0)


def should_apply_extra_guidance(step_index: int, skip_step: int) -> bool:
    """Return whether an optional STG/modality pass should run on this step.

    Args:
        step_index: Zero-based denoising step index.
        skip_step: Interval for running the extra guidance pass.
            1 = every step, 2 = every other step, etc.
    """
    if skip_step < 1:
        raise ValueError(f"skip_step must be >= 1, got {skip_step}")
    return step_index % skip_step == 0


def ltx2_scheduler(
    steps: int,
    num_tokens: Optional[int] = None,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> mx.array:
    """LTX-2 scheduler for sigma generation (dev model).

    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.

    Args:
        steps: Number of inference steps
        num_tokens: Number of latent tokens (F*H*W). If None, uses MAX_SHIFT_ANCHOR
        max_shift: Maximum shift factor
        base_shift: Base shift factor
        stretch: Whether to stretch sigmas to terminal value
        terminal: Terminal sigma value for stretching

    Returns:
        Array of sigma values of shape (steps + 1,)
    """
    tokens = num_tokens if num_tokens is not None else MAX_SHIFT_ANCHOR
    sigmas = np.linspace(1.0, 0.0, steps + 1)

    # Compute shift based on token count
    x1 = BASE_SHIFT_ANCHOR
    x2 = MAX_SHIFT_ANCHOR
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = tokens * mm + b

    # Apply shift transformation
    power = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        sigmas = np.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

    # Stretch sigmas to terminal value
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    return mx.array(sigmas, dtype=mx.float32)


def create_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> mx.array:
    """Create position grid for RoPE in pixel space.

    Args:
        batch_size: Batch size
        num_frames: Number of frames (latent)
        height: Height (latent)
        width: Width (latent)
        temporal_scale: VAE temporal scale factor (default 8)
        spatial_scale: VAE spatial scale factor (default 32)
        fps: Frames per second (default 24.0)
        causal_fix: Apply causal fix for first frame (default True)

    Returns:
        Position grid of shape (B, 3, num_patches, 2) in pixel space
        where dim 2 is [start, end) bounds for each patch
    """
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(
        3, 1, 1, 1
    )
    patch_ends = patch_starts + patch_size_delta

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(
        1, 3, 1, 1
    )
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale, a_min=0, a_max=None
        )

    # Divide temporal coords by fps
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

    # Cast entire position grid through bfloat16 to match PyTorch's behavior.
    # PyTorch does: positions = positions.to(bfloat16) on ALL coordinates before
    # passing to the transformer/RoPE. This quantization is what the model was
    # trained with, so we must replicate it for numerical fidelity.
    positions_bf16 = mx.array(pixel_coords, dtype=mx.bfloat16)
    mx.eval(positions_bf16)
    return positions_bf16.astype(mx.float32)


def create_audio_position_grid(
    batch_size: int,
    audio_frames: int,
    sample_rate: int = AUDIO_LATENT_SAMPLE_RATE,
    hop_length: int = AUDIO_HOP_LENGTH,
    downsample_factor: int = AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    is_causal: bool = True,
) -> mx.array:
    """Create temporal position grid for audio RoPE."""

    def get_audio_latent_time_in_sec(start_idx: int, end_idx: int) -> np.ndarray:
        latent_frame = np.arange(start_idx, end_idx, dtype=np.float32)
        mel_frame = latent_frame * downsample_factor
        if is_causal:
            mel_frame = np.clip(mel_frame + 1 - downsample_factor, 0, None)
        return mel_frame * hop_length / sample_rate

    start_times = get_audio_latent_time_in_sec(0, audio_frames)
    end_times = get_audio_latent_time_in_sec(1, audio_frames + 1)

    positions = np.stack([start_times, end_times], axis=-1)
    positions = positions[np.newaxis, np.newaxis, :, :]
    positions = np.tile(positions, (batch_size, 1, 1, 1))

    # Cast through bfloat16 to match PyTorch's precision behavior
    positions_bf16 = mx.array(positions, dtype=mx.bfloat16)
    mx.eval(positions_bf16)
    return positions_bf16.astype(mx.float32)


def compute_audio_frames(num_video_frames: int, fps: float) -> int:
    """Compute number of audio latent frames given video duration."""
    duration = num_video_frames / fps
    return round(duration * AUDIO_LATENTS_PER_SECOND)


# =============================================================================
# Distilled Pipeline Denoising (no CFG, fixed sigmas)
# =============================================================================


def invalidate_distilled_transformer_compile_cache(
    transformer: Optional[LTXModel],
) -> None:
    """Drop cached distilled forward wrappers after the model structure changes."""
    if transformer is None:
        return

    for attr in ("_compiled_distilled_video", "_compiled_distilled_av"):
        if hasattr(transformer, attr):
            delattr(transformer, attr)


def resolve_transformer_quantization(
    bits: Optional[int],
    group_size: Optional[int],
    mode: str = "affine",
    quantize_input: bool = False,
) -> Optional[dict]:
    # 日本語概要:
    # CLI から明示指定された時だけ、未量子化 transformer を runtime で量子化する。
    if bits is None:
        return None
    if mode not in TRANSFORMER_QUANTIZATION_MODES:
        raise ValueError(
            f"Unsupported transformer quantization mode: {mode!r}. "
            f"Expected one of {', '.join(TRANSFORMER_QUANTIZATION_MODES)}."
        )

    quantization = normalize_quantization_config(
        {
            "bits": bits,
            "group_size": group_size,
            "mode": mode,
            "quantize_input": quantize_input,
        }
    )
    if quantization["mode"] == "affine" and quantization["bits"] not in (4, 8):
        raise ValueError("Affine transformer quantization only supports 4-bit or 8-bit.")
    if not quantization.get("quantize_input"):
        quantization.pop("quantize_input", None)
    return quantization


def video_latents_to_tokens(
    latents: mx.array,
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    """Convert video latents from (B, C, F, H, W) to (B, T, C)."""
    b, c, *_ = latents.shape
    tokens = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
    return tokens.astype(dtype) if dtype is not None else tokens


def video_tokens_to_latents(
    tokens: mx.array,
    latent_shape: tuple[int, int, int, int, int],
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    """Convert token-major video latents from (B, T, C) back to (B, C, F, H, W)."""
    b, c, f, h, w = latent_shape
    latents = mx.reshape(mx.transpose(tokens, (0, 2, 1)), (b, c, f, h, w))
    return latents.astype(dtype) if dtype is not None else latents


def audio_latents_to_tokens(
    audio_latents: mx.array,
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    """Convert audio latents from (B, C, T, F) to (B, T, C*F)."""
    ab, ac, at, af = audio_latents.shape
    tokens = mx.reshape(mx.transpose(audio_latents, (0, 2, 1, 3)), (ab, at, ac * af))
    return tokens.astype(dtype) if dtype is not None else tokens


def audio_tokens_to_latents(
    tokens: mx.array,
    audio_shape: tuple[int, int, int, int],
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    """Convert token-major audio latents from (B, T, C*F) back to (B, C, T, F)."""
    ab, ac, at, af = audio_shape
    audio_latents = mx.reshape(tokens, (ab, at, ac, af))
    audio_latents = mx.transpose(audio_latents, (0, 2, 1, 3))
    return audio_latents.astype(dtype) if dtype is not None else audio_latents


def video_denoise_mask_to_tokens(
    denoise_mask: mx.array,
    latent_shape: tuple[int, int, int, int, int],
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    """Expand a per-frame denoise mask to token-major shape (B, T, 1)."""
    b, _, f, _, _ = denoise_mask.shape
    _, _, _, h, w = latent_shape
    expanded = mx.broadcast_to(denoise_mask, (b, 1, f, h, w))
    tokens = mx.transpose(mx.reshape(expanded, (b, 1, -1)), (0, 2, 1))
    return tokens.astype(dtype) if dtype is not None else tokens


def get_distilled_transformer_forward(
    transformer: LTXModel,
    *,
    enable_audio: bool,
):
    """Return a cached distilled forward specialized for video-only or A/V."""
    attr = "_compiled_distilled_av" if enable_audio else "_compiled_distilled_video"
    compiled = getattr(transformer, attr, None)
    if compiled is not None:
        return compiled

    if enable_audio:
        # 日本語概要:
        # Mode ごとの wrapper だけを使い回す。
        # `mx.compile()` は 22B transformer では初回 trace のメモリピークが大きく、
        # Stage 1 で unified memory / VRAM を押し上げやすいため使わない。
        def forward(
            video_latent: mx.array,
            video_timesteps: mx.array,
            video_positions: mx.array,
            video_context: mx.array,
            video_sigma: mx.array,
            audio_latent: mx.array,
            audio_timesteps: mx.array,
            audio_positions: mx.array,
            audio_context: mx.array,
            audio_sigma: mx.array,
        ) -> tuple[mx.array, mx.array]:
            video_modality = Modality(
                latent=video_latent,
                timesteps=video_timesteps,
                positions=video_positions,
                context=video_context,
                context_mask=None,
                enabled=True,
                sigma=video_sigma,
            )
            audio_modality = Modality(
                latent=audio_latent,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_context,
                context_mask=None,
                enabled=True,
                sigma=audio_sigma,
            )
            return transformer(video=video_modality, audio=audio_modality)
    else:
        # 日本語概要:
        # video-only でも compile は避け、軽い wrapper だけをキャッシュする。
        def forward(
            video_latent: mx.array,
            video_timesteps: mx.array,
            video_positions: mx.array,
            video_context: mx.array,
            video_sigma: mx.array,
        ) -> tuple[mx.array, Optional[mx.array]]:
            video_modality = Modality(
                latent=video_latent,
                timesteps=video_timesteps,
                positions=video_positions,
                context=video_context,
                context_mask=None,
                enabled=True,
                sigma=video_sigma,
            )
            return transformer(video=video_modality, audio=None)

    setattr(transformer, attr, forward)
    return forward


def denoise_distilled(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list,
    verbose: bool = True,
    state: Optional[LatentState] = None,
    audio_latents: Optional[mx.array] = None,
    audio_positions: Optional[mx.array] = None,
    audio_embeddings: Optional[mx.array] = None,
    audio_frozen: bool = False,
) -> tuple[mx.array, Optional[mx.array]]:
    """Run denoising loop for distilled pipeline (no CFG)."""
    dtype = latents.dtype
    enable_audio = audio_latents is not None

    if state is not None:
        latents = state.latent

    # 日本語概要:
    # denoise loop の内部表現を token-major に寄せて、毎 step の 5D <-> 3D 変換を
    # 最小化する。最終的に戻り値だけ元の latent layout に戻す。
    video_shape = tuple(latents.shape)
    video_tokens_f32 = video_latents_to_tokens(latents.astype(mx.float32))

    audio_shape = None
    audio_tokens_f32 = None
    if enable_audio:
        audio_shape = tuple(audio_latents.shape)
        audio_tokens_f32 = audio_latents_to_tokens(audio_latents.astype(mx.float32))

    clean_video_tokens_f32 = None
    denoise_mask_tokens_f32 = None
    denoise_mask_tokens = None
    if state is not None:
        clean_video_tokens_f32 = video_latents_to_tokens(state.clean_latent.astype(mx.float32))
        denoise_mask_tokens_f32 = video_denoise_mask_to_tokens(
            state.denoise_mask.astype(mx.float32),
            video_shape,
            dtype=mx.float32,
        )
        denoise_mask_tokens = mx.squeeze(denoise_mask_tokens_f32, axis=-1).astype(dtype)

    desc = "[cyan]Denoising A/V[/]" if enable_audio else "[cyan]Denoising[/]"
    num_steps = len(sigmas) - 1
    compiled_forward = get_distilled_transformer_forward(
        transformer,
        enable_audio=enable_audio,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task(desc, total=num_steps)

        for i in range(num_steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]

            b, _, f, h, w = video_shape
            num_tokens = f * h * w
            latents_flat = video_tokens_f32.astype(dtype)
            sigma_scalar = mx.array(sigma, dtype=dtype)
            sigma_array = mx.full((b,), sigma, dtype=dtype)

            if state is not None:
                timesteps = sigma_scalar * denoise_mask_tokens
            else:
                timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

            if enable_audio:
                ab, _, at, _ = audio_shape
                audio_flat = audio_tokens_f32.astype(dtype)
                # A2V: frozen audio uses timesteps=0 (tells model audio is clean)
                a_ts = (
                    mx.zeros((ab, at), dtype=dtype)
                    if audio_frozen
                    else mx.full((ab, at), sigma, dtype=dtype)
                )
                a_sig = (
                    mx.zeros((ab,), dtype=dtype)
                    if audio_frozen
                    else mx.full((ab,), sigma, dtype=dtype)
                )
                velocity, audio_velocity = compiled_forward(
                    latents_flat,
                    timesteps,
                    positions,
                    text_embeddings,
                    sigma_array,
                    audio_flat,
                    a_ts,
                    audio_positions,
                    audio_embeddings,
                    a_sig,
                )
            else:
                velocity, audio_velocity = compiled_forward(
                    latents_flat,
                    timesteps,
                    positions,
                    text_embeddings,
                    sigma_array,
                )

            # 日本語概要:
            # x0 計算と Euler 更新は token-major のまま進めて、
            # step の最後だけ eval して MLX の lazy graph をまとめて実行する。
            sigma_f32 = mx.array(sigma, dtype=mx.float32)
            timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
            x0_f32 = video_tokens_f32 - timesteps_f32 * velocity.astype(mx.float32)

            if state is not None:
                x0_f32 = (
                    x0_f32 * denoise_mask_tokens_f32
                    + clean_video_tokens_f32 * (1.0 - denoise_mask_tokens_f32)
                )

            audio_x0_f32 = None
            if enable_audio and audio_velocity is not None and not audio_frozen:
                audio_x0_f32 = audio_tokens_f32 - sigma_f32 * audio_velocity.astype(
                    mx.float32
                )

            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                video_tokens_f32 = x0_f32 + sigma_next_f32 * (
                    video_tokens_f32 - x0_f32
                ) / sigma_f32
                if enable_audio and audio_x0_f32 is not None and not audio_frozen:
                    audio_tokens_f32 = audio_x0_f32 + sigma_next_f32 * (
                        audio_tokens_f32 - audio_x0_f32
                    ) / sigma_f32
            else:
                video_tokens_f32 = x0_f32
                if enable_audio and audio_x0_f32 is not None and not audio_frozen:
                    audio_tokens_f32 = audio_x0_f32

            if enable_audio:
                mx.eval(video_tokens_f32, audio_tokens_f32)
            else:
                mx.eval(video_tokens_f32)

            progress.advance(task)

    latents = video_tokens_to_latents(video_tokens_f32, video_shape, dtype=dtype)
    if enable_audio and audio_tokens_f32 is not None and audio_shape is not None:
        audio_latents = audio_tokens_to_latents(
            audio_tokens_f32,
            audio_shape,
            dtype=dtype,
        )
    else:
        audio_latents = None

    return latents, audio_latents


# =============================================================================
# Dev Pipeline Denoising (with CFG, dynamic sigmas)
# =============================================================================


def denoise_dev(
    latents: mx.array,
    positions: mx.array,
    text_embeddings_pos: mx.array,
    text_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    cfg_rescale: float = 0.0,
    verbose: bool = True,
    state: Optional[LatentState] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 0.0,
    stg_blocks: Optional[list] = None,
) -> mx.array:
    """Run denoising loop for dev pipeline with CFG/APG and optional STG guidance.

    Args:
        cfg_rescale: Rescale factor for CFG (0.0-1.0). Normalizes guided prediction
                     variance relative to conditional prediction to reduce over-saturation.
                     PyTorch default is 0.7. Set to 0.0 to disable.
        use_apg: Use Adaptive Projected Guidance instead of standard CFG.
                 APG decomposes guidance into parallel/orthogonal components
                 for more stable I2V generation.
        apg_eta: APG parallel component weight (1.0 = keep full parallel)
        apg_norm_threshold: APG guidance norm clamp (0 = no clamping)
        stg_scale: STG (Spatiotemporal Guidance) scale. 0.0 = disabled.
        stg_blocks: Transformer block indices for STG perturbation.
    """
    from mlx_video.models.ltx_2.rope import precompute_freqs_cis

    dtype = latents.dtype
    if state is not None:
        latents = state.latent

    # Keep latents in float32 throughout the denoising loop to avoid
    # quantization noise accumulation over many steps.
    # Model input is cast to model dtype; all denoising math stays in float32.
    latents = latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    use_stg = stg_scale != 0.0 and stg_blocks is not None
    num_steps = len(sigmas_list) - 1

    # Precompute RoPE once
    precomputed_rope = precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_rope)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        passes = ["CFG"] if use_cfg else []
        if use_stg:
            passes.append("STG")
        label = "+".join(passes) if passes else "uncond"
        task = progress.add_task(f"[cyan]Denoising ({label})[/]", total=num_steps)

        for i in range(num_steps):
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]

            b, c, f, h, w = latents.shape
            num_tokens = f * h * w
            # Cast to model dtype for transformer input
            latents_flat = mx.transpose(
                mx.reshape(latents, (b, c, -1)), (0, 2, 1)
            ).astype(dtype)

            if state is not None:
                denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
                denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
                denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
                timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
            else:
                timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

            sigma_array = mx.full((b,), sigma, dtype=dtype)

            # Positive conditioning pass
            video_modality_pos = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings_pos,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_rope,
                sigma=sigma_array,
            )
            velocity_pos, _ = transformer(video=video_modality_pos, audio=None)

            # Convert velocity to x0 (denoised) using per-token timesteps
            # Matches PyTorch's X0Model: x0 = latent - timestep * velocity
            # For conditioned tokens (timestep=0): x0 = latent (correct regardless of velocity)
            # For unconditioned tokens (timestep=sigma): x0 = latent - sigma * velocity
            latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
            timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
            x0_pos_f32 = latents_flat_f32 - timesteps_f32 * velocity_pos.astype(
                mx.float32
            )

            # Start with positive prediction
            x0_guided_f32 = x0_pos_f32

            if use_cfg:
                # Negative conditioning pass
                video_modality_neg = Modality(
                    latent=latents_flat,
                    timesteps=timesteps,
                    positions=positions,
                    context=text_embeddings_neg,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_rope,
                    sigma=sigma_array,
                )
                velocity_neg, _ = transformer(video=video_modality_neg, audio=None)

                # Convert negative velocity to x0 using per-token timesteps
                x0_neg_f32 = latents_flat_f32 - timesteps_f32 * velocity_neg.astype(
                    mx.float32
                )

                # Apply guidance to x0 predictions
                # For conditioned tokens: x0_pos = x0_neg = latent, so delta = 0
                if use_apg:
                    # APG: decompose into parallel/orthogonal components for stability
                    x0_guided_f32 = x0_pos_f32 + apg_delta(
                        x0_pos_f32,
                        x0_neg_f32,
                        cfg_scale,
                        eta=apg_eta,
                        norm_threshold=apg_norm_threshold,
                    )
                else:
                    # Standard CFG
                    x0_guided_f32 = x0_pos_f32 + (cfg_scale - 1.0) * (
                        x0_pos_f32 - x0_neg_f32
                    )

            # STG pass: skip self-attention at specified blocks
            if use_stg:
                velocity_ptb, _ = transformer(
                    video=video_modality_pos,
                    audio=None,
                    stg_video_blocks=stg_blocks,
                )
                mx.eval(velocity_ptb)

                x0_ptb_f32 = latents_flat_f32 - timesteps_f32 * velocity_ptb.astype(
                    mx.float32
                )
                x0_guided_f32 = x0_guided_f32 + stg_scale * (x0_pos_f32 - x0_ptb_f32)

            # Apply CFG rescale if enabled (std-ratio rescaling to reduce over-saturation)
            # factor = rescale * (cond_std / pred_std) + (1 - rescale)
            # pred = pred * factor
            if cfg_rescale > 0.0 and (use_cfg or use_stg):
                v_factor = x0_pos_f32.std() / (x0_guided_f32.std() + 1e-8)
                v_factor = cfg_rescale * v_factor + (1.0 - cfg_rescale)
                x0_guided_f32 = x0_guided_f32 * v_factor

            # Reshape x0 from token space (b, tokens, c) to spatial (b, c, f, h, w)
            denoised = mx.reshape(
                mx.transpose(x0_guided_f32, (0, 2, 1)), (b, c, f, h, w)
            )

            sigma_f32 = mx.array(sigma, dtype=mx.float32)

            if state is not None:
                denoised = apply_denoise_mask(
                    denoised, state.clean_latent.astype(mx.float32), state.denoise_mask
                )

            # Euler step in float32 (latents stay in float32)
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
            else:
                latents = denoised

            mx.eval(latents)
            progress.advance(task)

    return latents.astype(dtype)


def denoise_dev_av(
    video_latents: mx.array,
    audio_latents: mx.array,
    video_positions: mx.array,
    audio_positions: mx.array,
    video_embeddings_pos: mx.array,
    video_embeddings_neg: mx.array,
    audio_embeddings_pos: mx.array,
    audio_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    audio_cfg_scale: float = 7.0,
    cfg_rescale: float = 0.0,
    verbose: bool = True,
    video_state: Optional[LatentState] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 0.0,
    stg_video_blocks: Optional[list] = None,
    stg_audio_blocks: Optional[list] = None,
    modality_scale: float = 1.0,
    stg_skip_step: int = 1,
    modality_skip_step: int = 1,
    audio_frozen: bool = False,
) -> tuple[mx.array, mx.array]:
    """Run denoising loop for dev pipeline with CFG/APG, STG, modality guidance, and audio.

    Args:
        audio_cfg_scale: Separate CFG scale for audio (PyTorch default: 7.0).
        cfg_rescale: Rescale factor for CFG (0.0-1.0). Normalizes guided prediction
                     variance to reduce artifacts. Default 0.0 means no rescaling.
        use_apg: Use Adaptive Projected Guidance instead of standard CFG for video.
        apg_eta: APG parallel component weight (1.0 = keep full parallel)
        apg_norm_threshold: APG guidance norm clamp (0 = no clamping)
        stg_scale: STG (Spatiotemporal Guidance) scale. 0.0 = disabled.
        stg_video_blocks: Transformer block indices for video STG perturbation.
        stg_audio_blocks: Transformer block indices for audio STG perturbation.
        modality_scale: Cross-modal guidance scale. 1.0 = disabled.
        stg_skip_step: Run STG perturbation every Nth denoising step.
                       1 keeps the current behavior (every step).
        modality_skip_step: Run cross-modal isolation every Nth denoising step.
                            1 keeps the current behavior (every step).
    """
    from mlx_video.models.ltx_2.rope import precompute_freqs_cis

    if stg_skip_step < 1:
        raise ValueError(f"stg_skip_step must be >= 1, got {stg_skip_step}")
    if modality_skip_step < 1:
        raise ValueError(
            f"modality_skip_step must be >= 1, got {modality_skip_step}"
        )

    dtype = video_latents.dtype
    if video_state is not None:
        video_latents = video_state.latent

    # Keep latents in float32 throughout the denoising loop for precision.
    video_latents = video_latents.astype(mx.float32)
    audio_latents = audio_latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    use_stg = stg_scale != 0.0 and stg_video_blocks is not None
    use_modality = modality_scale != 1.0
    num_steps = len(sigmas_list) - 1

    # Precompute video RoPE
    precomputed_video_rope = precompute_freqs_cis(
        video_positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )

    # Precompute audio RoPE
    precomputed_audio_rope = precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_video_rope, precomputed_audio_rope)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        passes = ["CFG"] if use_cfg else []
        if use_stg:
            passes.append("STG")
        if use_modality:
            passes.append("Mod")
        label = "+".join(passes) if passes else "uncond"
        task = progress.add_task(f"[cyan]Denoising A/V ({label})[/]", total=num_steps)

        for i in range(num_steps):
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]
            apply_stg = use_stg and should_apply_extra_guidance(i, stg_skip_step)
            apply_modality = use_modality and should_apply_extra_guidance(
                i, modality_skip_step
            )

            # Flatten video latents (cast to model dtype for transformer input)
            b, c, f, h, w = video_latents.shape
            num_video_tokens = f * h * w
            video_flat_f32 = mx.transpose(mx.reshape(video_latents, (b, c, -1)), (0, 2, 1))
            video_flat = video_flat_f32.astype(dtype)

            # Flatten audio latents (cast to model dtype for transformer input)
            ab, ac, at, af = audio_latents.shape
            audio_flat_f32 = mx.reshape(
                mx.transpose(audio_latents, (0, 2, 1, 3)), (ab, at, ac * af)
            )
            audio_flat = audio_flat_f32.astype(dtype)

            # Compute timesteps
            if video_state is not None:
                denoise_mask_flat = mx.reshape(
                    video_state.denoise_mask, (b, 1, f, 1, 1)
                )
                denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
                denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
                video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
            else:
                video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)

            # A2V: frozen audio uses timesteps=0 (tells model audio is clean)
            audio_timesteps = (
                mx.zeros((ab, at), dtype=dtype)
                if audio_frozen
                else mx.full((ab, at), sigma, dtype=dtype)
            )

            # Positive conditioning pass
            sigma_array = mx.full((b,), sigma, dtype=dtype)
            audio_sigma_array = (
                mx.zeros((ab,), dtype=dtype)
                if audio_frozen
                else mx.full((ab,), sigma, dtype=dtype)
            )
            video_modality_pos = Modality(
                latent=video_flat,
                timesteps=video_timesteps,
                positions=video_positions,
                context=video_embeddings_pos,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_video_rope,
                sigma=sigma_array,
            )
            audio_modality_pos = Modality(
                latent=audio_flat,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_embeddings_pos,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
                sigma=audio_sigma_array,
            )
            video_vel_pos, audio_vel_pos = transformer(
                video=video_modality_pos, audio=audio_modality_pos
            )

            # Convert velocity to denoised (x0) using per-token timesteps
            # This matches PyTorch's X0ModelWrapper: x0 = latent - timestep * velocity
            # For conditioned tokens (timestep=0): x0 = latent (velocity is irrelevant)
            # For unconditioned tokens (timestep=sigma): x0 = latent - sigma * velocity
            video_timesteps_f32 = mx.expand_dims(
                video_timesteps.astype(mx.float32), axis=-1
            )
            audio_timesteps_f32 = mx.expand_dims(
                audio_timesteps.astype(mx.float32), axis=-1
            )

            video_x0_pos_f32 = (
                video_flat_f32 - video_timesteps_f32 * video_vel_pos.astype(mx.float32)
            )
            audio_x0_pos_f32 = (
                audio_flat_f32 - audio_timesteps_f32 * audio_vel_pos.astype(mx.float32)
            )

            # Start with positive prediction
            video_x0_guided_f32 = video_x0_pos_f32
            audio_x0_guided_f32 = audio_x0_pos_f32

            # Pass 2: CFG (negative conditioning)
            if use_cfg:
                video_modality_neg = Modality(
                    latent=video_flat,
                    timesteps=video_timesteps,
                    positions=video_positions,
                    context=video_embeddings_neg,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_video_rope,
                    sigma=sigma_array,
                )
                audio_modality_neg = Modality(
                    latent=audio_flat,
                    timesteps=audio_timesteps,
                    positions=audio_positions,
                    context=audio_embeddings_neg,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_audio_rope,
                    sigma=audio_sigma_array,
                )
                video_vel_neg, audio_vel_neg = transformer(
                    video=video_modality_neg, audio=audio_modality_neg
                )

                video_x0_neg_f32 = (
                    video_flat_f32
                    - video_timesteps_f32 * video_vel_neg.astype(mx.float32)
                )
                audio_x0_neg_f32 = (
                    audio_flat_f32
                    - audio_timesteps_f32 * audio_vel_neg.astype(mx.float32)
                )

                if use_apg:
                    video_x0_guided_f32 = video_x0_pos_f32 + apg_delta(
                        video_x0_pos_f32,
                        video_x0_neg_f32,
                        cfg_scale,
                        eta=apg_eta,
                        norm_threshold=apg_norm_threshold,
                    )
                else:
                    video_x0_guided_f32 = video_x0_pos_f32 + (cfg_scale - 1.0) * (
                        video_x0_pos_f32 - video_x0_neg_f32
                    )
                audio_x0_guided_f32 = audio_x0_pos_f32 + (audio_cfg_scale - 1.0) * (
                    audio_x0_pos_f32 - audio_x0_neg_f32
                )

            # Pass 3: STG (self-attention perturbation at specified blocks)
            if apply_stg:
                video_vel_ptb, audio_vel_ptb = transformer(
                    video=video_modality_pos,
                    audio=audio_modality_pos,
                    stg_video_blocks=stg_video_blocks,
                    stg_audio_blocks=stg_audio_blocks,
                )

                video_x0_ptb_f32 = (
                    video_flat_f32
                    - video_timesteps_f32 * video_vel_ptb.astype(mx.float32)
                )
                audio_x0_ptb_f32 = (
                    audio_flat_f32
                    - audio_timesteps_f32 * audio_vel_ptb.astype(mx.float32)
                )

                video_x0_guided_f32 = video_x0_guided_f32 + stg_scale * (
                    video_x0_pos_f32 - video_x0_ptb_f32
                )
                audio_x0_guided_f32 = audio_x0_guided_f32 + stg_scale * (
                    audio_x0_pos_f32 - audio_x0_ptb_f32
                )

            # Pass 4: Modality isolation (skip all cross-modal attention)
            if apply_modality:
                video_vel_iso, audio_vel_iso = transformer(
                    video=video_modality_pos,
                    audio=audio_modality_pos,
                    skip_cross_modal=True,
                )

                video_x0_iso_f32 = (
                    video_flat_f32
                    - video_timesteps_f32 * video_vel_iso.astype(mx.float32)
                )
                audio_x0_iso_f32 = (
                    audio_flat_f32
                    - audio_timesteps_f32 * audio_vel_iso.astype(mx.float32)
                )

                video_x0_guided_f32 = video_x0_guided_f32 + (modality_scale - 1.0) * (
                    video_x0_pos_f32 - video_x0_iso_f32
                )
                audio_x0_guided_f32 = audio_x0_guided_f32 + (modality_scale - 1.0) * (
                    audio_x0_pos_f32 - audio_x0_iso_f32
                )

            # Apply CFG rescale (std-ratio rescaling to reduce over-saturation)
            if cfg_rescale > 0.0 and (use_cfg or apply_stg or apply_modality):
                v_factor = video_x0_pos_f32.std() / (video_x0_guided_f32.std() + 1e-8)
                v_factor = cfg_rescale * v_factor + (1.0 - cfg_rescale)
                video_x0_guided_f32 = video_x0_guided_f32 * v_factor
                a_factor = audio_x0_pos_f32.std() / (audio_x0_guided_f32.std() + 1e-8)
                a_factor = cfg_rescale * a_factor + (1.0 - cfg_rescale)
                audio_x0_guided_f32 = audio_x0_guided_f32 * a_factor

            # Reshape x0 from token space (b, tokens, c) to spatial (b, c, f, h, w)
            video_denoised_f32 = mx.reshape(
                mx.transpose(video_x0_guided_f32, (0, 2, 1)), (b, c, f, h, w)
            )
            audio_denoised_f32 = mx.reshape(audio_x0_guided_f32, (ab, at, ac, af))
            audio_denoised_f32 = mx.transpose(audio_denoised_f32, (0, 2, 1, 3))

            # Post-process: blend denoised with clean latent using mask
            # Matches PyTorch's post_process_latent: denoised * mask + clean * (1 - mask)
            sigma_f32 = mx.array(sigma, dtype=mx.float32)

            if video_state is not None:
                clean_f32 = video_state.clean_latent.astype(mx.float32)
                mask_f32 = video_state.denoise_mask.astype(mx.float32)
                video_denoised_f32 = video_denoised_f32 * mask_f32 + clean_f32 * (
                    1.0 - mask_f32
                )

            # Euler step: sample + velocity * dt (float32)
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                dt_f32 = sigma_next_f32 - sigma_f32

                video_velocity_f32 = (video_latents - video_denoised_f32) / sigma_f32
                video_latents = video_latents + video_velocity_f32 * dt_f32

                if not audio_frozen:
                    audio_velocity_f32 = (
                        audio_latents - audio_denoised_f32
                    ) / sigma_f32
                    audio_latents = audio_latents + audio_velocity_f32 * dt_f32
            else:
                video_latents = video_denoised_f32
                if not audio_frozen:
                    audio_latents = audio_denoised_f32

            # Avoid per-pass synchronization so MLX can schedule the full step graph,
            # then materialize the updated latents once at the loop boundary.
            mx.eval(video_latents, audio_latents)
            progress.advance(task)

    return video_latents, audio_latents


def denoise_res2s_av(
    video_latents: mx.array,
    audio_latents: mx.array,
    video_positions: mx.array,
    audio_positions: mx.array,
    video_embeddings_pos: mx.array,
    video_embeddings_neg: mx.array,
    audio_embeddings_pos: mx.array,
    audio_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 3.0,
    audio_cfg_scale: float = 7.0,
    cfg_rescale: float = 0.45,
    audio_cfg_rescale: Optional[float] = None,
    verbose: bool = True,
    video_state: Optional[LatentState] = None,
    stg_scale: float = 0.0,
    stg_video_blocks: Optional[list] = None,
    stg_audio_blocks: Optional[list] = None,
    modality_scale: float = 1.0,
    noise_seed: int = 42,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    audio_frozen: bool = False,
) -> tuple[mx.array, mx.array]:
    """Run res_2s second-order denoising loop with CFG/STG/modality guidance.

    Two model evaluations per step (current point + midpoint), with SDE noise
    injection and optional bong iteration for anchor refinement.

    Args:
        audio_cfg_rescale: Separate rescale for audio. If None, uses cfg_rescale.
        noise_seed: Seed for SDE noise generators.
        bongmath: Enable iterative anchor refinement.
        bongmath_max_iter: Max bong iterations per step.
    """
    from mlx_video.models.ltx_2.rope import precompute_freqs_cis
    from mlx_video.models.ltx_2.samplers import (
        get_new_noise,
        get_res2s_coefficients,
        sde_noise_step,
    )

    if audio_cfg_rescale is None:
        audio_cfg_rescale = cfg_rescale

    dtype = video_latents.dtype
    if video_state is not None:
        video_latents = video_state.latent

    video_latents = video_latents.astype(mx.float32)
    audio_latents = audio_latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    use_stg = stg_scale != 0.0 and stg_video_blocks is not None
    use_modality = modality_scale != 1.0
    n_full_steps = len(sigmas_list) - 1

    # Pad sigmas if last is 0 (avoid division by zero in RK steps)
    if sigmas_list[-1] == 0:
        sigmas_list = sigmas_list[:-1] + [0.0011, 0.0]

    # Compute step sizes in log-space for the main loop steps only.
    # After padding, sigmas_list may have an extra [0.0011, 0.0] tail;
    # we only need hs for the n_full_steps pairs the loop actually uses.
    hs = [-math.log(sigmas_list[i + 1] / sigmas_list[i]) for i in range(n_full_steps)]

    # Precompute RoPE
    precomputed_video_rope = precompute_freqs_cis(
        video_positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    precomputed_audio_rope = precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_video_rope, precomputed_audio_rope)

    phi_cache = {}
    c2 = 0.5

    # Noise key management: step noise and substep noise use different keys
    step_noise_key = mx.random.key(noise_seed)
    substep_noise_key = mx.random.key(noise_seed + 10000)

    def _eval_guided_denoise(v_latents, a_latents, sigma):
        """Run all guidance passes and return (video_denoised, audio_denoised) in float32 spatial format."""
        b, c, f, h, w = v_latents.shape
        num_video_tokens = f * h * w
        video_flat_f32 = mx.transpose(mx.reshape(v_latents, (b, c, -1)), (0, 2, 1))
        video_flat = video_flat_f32.astype(dtype)

        ab, ac, at, af = a_latents.shape
        audio_flat_f32 = mx.reshape(
            mx.transpose(a_latents, (0, 2, 1, 3)), (ab, at, ac * af)
        )
        audio_flat = audio_flat_f32.astype(dtype)

        # Timesteps
        if video_state is not None:
            denoise_mask_flat = mx.reshape(video_state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
            video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)
        audio_timesteps = (
            mx.zeros((ab, at), dtype=dtype)
            if audio_frozen
            else mx.full((ab, at), sigma, dtype=dtype)
        )

        sigma_array = mx.full((b,), sigma, dtype=dtype)
        audio_sigma_array = (
            mx.zeros((ab,), dtype=dtype)
            if audio_frozen
            else mx.full((ab,), sigma, dtype=dtype)
        )

        # Pass 1: Positive conditioning
        video_modality_pos = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_video_rope,
            sigma=sigma_array,
        )
        audio_modality_pos = Modality(
            latent=audio_flat,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_audio_rope,
            sigma=audio_sigma_array,
        )
        video_vel_pos, audio_vel_pos = transformer(
            video=video_modality_pos, audio=audio_modality_pos
        )
        mx.eval(video_vel_pos, audio_vel_pos)

        # Convert velocity to x0
        video_ts_f32 = mx.expand_dims(video_timesteps.astype(mx.float32), axis=-1)
        audio_ts_f32 = mx.expand_dims(audio_timesteps.astype(mx.float32), axis=-1)

        video_x0_pos = video_flat_f32 - video_ts_f32 * video_vel_pos.astype(mx.float32)
        audio_x0_pos = audio_flat_f32 - audio_ts_f32 * audio_vel_pos.astype(mx.float32)

        video_x0_guided = video_x0_pos
        audio_x0_guided = audio_x0_pos

        # Pass 2: CFG
        if use_cfg:
            video_modality_neg = Modality(
                latent=video_flat,
                timesteps=video_timesteps,
                positions=video_positions,
                context=video_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_video_rope,
                sigma=sigma_array,
            )
            audio_modality_neg = Modality(
                latent=audio_flat,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
                sigma=audio_sigma_array,
            )
            video_vel_neg, audio_vel_neg = transformer(
                video=video_modality_neg, audio=audio_modality_neg
            )
            mx.eval(video_vel_neg, audio_vel_neg)

            video_x0_neg = video_flat_f32 - video_ts_f32 * video_vel_neg.astype(
                mx.float32
            )
            audio_x0_neg = audio_flat_f32 - audio_ts_f32 * audio_vel_neg.astype(
                mx.float32
            )

            video_x0_guided = video_x0_pos + (cfg_scale - 1.0) * (
                video_x0_pos - video_x0_neg
            )
            audio_x0_guided = audio_x0_pos + (audio_cfg_scale - 1.0) * (
                audio_x0_pos - audio_x0_neg
            )

        # Pass 3: STG
        if use_stg:
            video_vel_ptb, audio_vel_ptb = transformer(
                video=video_modality_pos,
                audio=audio_modality_pos,
                stg_video_blocks=stg_video_blocks,
                stg_audio_blocks=stg_audio_blocks,
            )
            mx.eval(video_vel_ptb, audio_vel_ptb)

            video_x0_ptb = video_flat_f32 - video_ts_f32 * video_vel_ptb.astype(
                mx.float32
            )
            audio_x0_ptb = audio_flat_f32 - audio_ts_f32 * audio_vel_ptb.astype(
                mx.float32
            )

            video_x0_guided = video_x0_guided + stg_scale * (
                video_x0_pos - video_x0_ptb
            )
            audio_x0_guided = audio_x0_guided + stg_scale * (
                audio_x0_pos - audio_x0_ptb
            )

        # Pass 4: Modality isolation
        if use_modality:
            video_vel_iso, audio_vel_iso = transformer(
                video=video_modality_pos,
                audio=audio_modality_pos,
                skip_cross_modal=True,
            )
            mx.eval(video_vel_iso, audio_vel_iso)

            video_x0_iso = video_flat_f32 - video_ts_f32 * video_vel_iso.astype(
                mx.float32
            )
            audio_x0_iso = audio_flat_f32 - audio_ts_f32 * audio_vel_iso.astype(
                mx.float32
            )

            video_x0_guided = video_x0_guided + (modality_scale - 1.0) * (
                video_x0_pos - video_x0_iso
            )
            audio_x0_guided = audio_x0_guided + (modality_scale - 1.0) * (
                audio_x0_pos - audio_x0_iso
            )

        # Rescale (separate factors for video and audio)
        if cfg_rescale > 0.0 and (use_cfg or use_stg or use_modality):
            v_factor = video_x0_pos.std() / (video_x0_guided.std() + 1e-8)
            v_factor = cfg_rescale * v_factor + (1.0 - cfg_rescale)
            video_x0_guided = video_x0_guided * v_factor
        if audio_cfg_rescale > 0.0 and (use_cfg or use_stg or use_modality):
            a_factor = audio_x0_pos.std() / (audio_x0_guided.std() + 1e-8)
            a_factor = audio_cfg_rescale * a_factor + (1.0 - audio_cfg_rescale)
            audio_x0_guided = audio_x0_guided * a_factor

        # Reshape to spatial
        video_denoised = mx.reshape(
            mx.transpose(video_x0_guided, (0, 2, 1)), (b, c, f, h, w)
        )
        audio_denoised = mx.reshape(audio_x0_guided, (ab, at, ac, af))
        audio_denoised = mx.transpose(audio_denoised, (0, 2, 1, 3))

        # Post-process with mask
        if video_state is not None:
            clean_f32 = video_state.clean_latent.astype(mx.float32)
            mask_f32 = video_state.denoise_mask.astype(mx.float32)
            video_denoised = video_denoised * mask_f32 + clean_f32 * (1.0 - mask_f32)

        mx.eval(video_denoised, audio_denoised)
        return video_denoised, audio_denoised

    # Main res_2s loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        passes = ["res2s"]
        if use_cfg:
            passes.append("CFG")
        if use_stg:
            passes.append("STG")
        if use_modality:
            passes.append("Mod")
        label = "+".join(passes)
        task = progress.add_task(
            f"[cyan]Denoising A/V ({label})[/]", total=n_full_steps
        )

        for step_idx in range(n_full_steps):
            sigma = sigmas_list[step_idx]
            sigma_next = sigmas_list[step_idx + 1]
            h = hs[step_idx]

            # Initialize anchor
            x_anchor_video = video_latents
            x_anchor_audio = audio_latents

            # ============================================================
            # Stage 1: Evaluate denoiser at current sigma
            # ============================================================
            denoised_video_1, denoised_audio_1 = _eval_guided_denoise(
                video_latents, audio_latents, sigma
            )

            # RK coefficients
            a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2)

            # Substep sigma (geometric midpoint for c2=0.5)
            sub_sigma = math.sqrt(sigma * sigma_next)

            # Compute midpoint
            eps_1_video = denoised_video_1 - x_anchor_video
            x_mid_video = x_anchor_video + h * a21 * eps_1_video

            if not audio_frozen:
                eps_1_audio = denoised_audio_1 - x_anchor_audio
                x_mid_audio = x_anchor_audio + h * a21 * eps_1_audio
            else:
                eps_1_audio = None
                x_mid_audio = audio_latents  # frozen: pass through unchanged

            # SDE noise injection at substep
            substep_noise_key, key1, key2 = mx.random.split(substep_noise_key, 3)
            substep_noise_v = get_new_noise(video_latents.shape, key1)

            x_mid_video = sde_noise_step(
                x_anchor_video, x_mid_video, sigma, sub_sigma, substep_noise_v
            )
            if not audio_frozen:
                substep_noise_a = get_new_noise(audio_latents.shape, key2)
                x_mid_audio = sde_noise_step(
                    x_anchor_audio, x_mid_audio, sigma, sub_sigma, substep_noise_a
                )
            mx.eval(x_mid_video, x_mid_audio)

            # ============================================================
            # Bong iteration: refine anchor (pure arithmetic, no model calls)
            # ============================================================
            if bongmath and h < 0.5 and sigma > 0.03:
                for _ in range(bongmath_max_iter):
                    x_anchor_video = x_mid_video - h * a21 * eps_1_video
                    eps_1_video = denoised_video_1 - x_anchor_video
                    if not audio_frozen:
                        x_anchor_audio = x_mid_audio - h * a21 * eps_1_audio
                        eps_1_audio = denoised_audio_1 - x_anchor_audio
                if audio_frozen:
                    mx.eval(x_anchor_video, eps_1_video)
                else:
                    mx.eval(x_anchor_video, x_anchor_audio, eps_1_video, eps_1_audio)

            # ============================================================
            # Stage 2: Evaluate denoiser at midpoint sigma
            # ============================================================
            denoised_video_2, denoised_audio_2 = _eval_guided_denoise(
                x_mid_video.astype(mx.float32),
                x_mid_audio.astype(mx.float32),
                sub_sigma,
            )

            # ============================================================
            # Final combination with RK coefficients
            # ============================================================
            eps_2_video = denoised_video_2 - x_anchor_video
            x_next_video = x_anchor_video + h * (b1 * eps_1_video + b2 * eps_2_video)

            # SDE noise injection at step level
            step_noise_key, key1, key2 = mx.random.split(step_noise_key, 3)
            step_noise_v = get_new_noise(video_latents.shape, key1)
            x_next_video = sde_noise_step(
                x_anchor_video, x_next_video, sigma, sigma_next, step_noise_v
            )

            video_latents = x_next_video.astype(mx.float32)
            if not audio_frozen:
                eps_2_audio = denoised_audio_2 - x_anchor_audio
                x_next_audio = x_anchor_audio + h * (
                    b1 * eps_1_audio + b2 * eps_2_audio
                )
                step_noise_a = get_new_noise(audio_latents.shape, key2)
                x_next_audio = sde_noise_step(
                    x_anchor_audio, x_next_audio, sigma, sigma_next, step_noise_a
                )
                audio_latents = x_next_audio.astype(mx.float32)

            mx.eval(video_latents, audio_latents)
            progress.advance(task)

    # Final clean step if original schedule ended at 0
    if sigmas.tolist()[-1] == 0:
        denoised_video, denoised_audio = _eval_guided_denoise(
            video_latents, audio_latents, sigmas_list[n_full_steps]
        )
        video_latents = denoised_video
        if not audio_frozen:
            audio_latents = denoised_audio
        mx.eval(video_latents, audio_latents)

    return video_latents, audio_latents


# =============================================================================
# Audio Loading and Processing
# =============================================================================


def load_audio_decoder(model_path: Path, pipeline: PipelineType):
    """Load audio VAE decoder."""
    from mlx_video.models.ltx_2.audio_vae import AudioDecoder

    decoder = AudioDecoder.from_pretrained(model_path / "audio_vae" / "decoder")

    return decoder


def load_vocoder_model(model_path: Path, pipeline: PipelineType):
    """Load vocoder for mel to waveform conversion.

    Automatically detects HiFi-GAN (LTX-2) or BigVGAN+BWE (LTX-2.3).
    """
    from mlx_video.models.ltx_2.audio_vae.vocoder import load_vocoder as _load_vocoder

    return _load_vocoder(model_path / "vocoder")


def encode_conditioning_image_latent(
    model_path: Path,
    image_path: str,
    height: int,
    width: int,
    dtype: mx.Dtype,
    vae_encoder: Optional["VideoEncoder"] = None,
) -> mx.array:
    """Encode a single conditioning image into video latent space.

    Loads the VAE encoder on demand, performs aspect-ratio-preserving preprocessing,
    encodes the image, and releases the encoder immediately afterwards to keep
    peak memory lower during multi-stage generation.
    """
    owns_encoder = vae_encoder is None
    if vae_encoder is None:
        vae_encoder = VideoEncoder.from_pretrained(model_path / "vae" / "encoder")

    input_image = load_image(image_path, height=height, width=width, dtype=dtype)
    image_tensor = prepare_image_for_encoding(input_image, height, width, dtype=dtype)
    image_latent = vae_encoder(image_tensor)
    mx.eval(image_latent)

    if owns_encoder:
        del vae_encoder
        mx.clear_cache()
    return image_latent


def load_video_decoder_statistics(model_path: Path) -> tuple[mx.array, mx.array]:
    """Load per-channel latent statistics with a cheap safetensors fast-path.

    Two-stage pipelines only need the decoder's latent normalization stats during
    upsampling, so loading the full decoder module here is unnecessary overhead.
    Prefer reading the tensors directly from the decoder weights and fall back to
    instantiating the decoder only for legacy layouts.
    """
    decoder_path = (model_path / "vae" / "decoder").resolve()
    cached = _VIDEO_DECODER_STATS_CACHE.get(decoder_path)
    if cached is not None:
        return cached

    stat_keys = {
        "mean": (
            "per_channel_statistics.mean",
            "vae.per_channel_statistics.mean-of-means",
        ),
        "std": (
            "per_channel_statistics.std",
            "vae.per_channel_statistics.std-of-means",
        ),
    }

    try:
        found = {"mean": None, "std": None}
        for weight_file in resolve_safetensor_files(decoder_path):
            with safe_open(str(weight_file), framework="np") as f:
                keys = set(f.keys())
                fallback_keys = {}
                for name, candidates in stat_keys.items():
                    if found[name] is not None:
                        continue
                    for key in candidates:
                        if key in keys:
                            fallback_keys[name] = key
                            try:
                                found[name] = mx.array(f.get_tensor(key))
                            except TypeError:
                                # safetensors' numpy path cannot materialize BF16.
                                # Fall back to MLX's loader for this shard only.
                                shard_weights = mx.load(str(weight_file))
                                for fallback_name, fallback_key in fallback_keys.items():
                                    if found[fallback_name] is None:
                                        found[fallback_name] = shard_weights[fallback_key]
                                del shard_weights
                            break
                if found["mean"] is not None and found["std"] is not None:
                    break

        if found["mean"] is not None and found["std"] is not None:
            mean = found["mean"]
            std = found["std"]
            mx.eval(mean, std)
            _VIDEO_DECODER_STATS_CACHE[decoder_path] = (mean, std)
            return mean, std
    except (FileNotFoundError, OSError, TypeError, ValueError):
        pass

    vae_decoder = VideoDecoder.from_pretrained(str(decoder_path))
    mean = vae_decoder.per_channel_statistics.mean
    std = vae_decoder.per_channel_statistics.std
    mx.eval(mean, std)
    del vae_decoder
    mx.clear_cache()
    _VIDEO_DECODER_STATS_CACHE[decoder_path] = (mean, std)
    return mean, std


def set_attention_query_chunk_size(model: LTXModel, query_chunk_size: Optional[int]) -> None:
    """Configure query-chunking across all transformer attention modules.

    This reduces attention workspace by evaluating Q in smaller slices while reusing
    the same K/V tensors. Useful for long full-resolution Stage 2 passes.
    """
    transformer_blocks = getattr(model, "transformer_blocks", None)
    if transformer_blocks is None:
        return

    attention_names = (
        "attn1",
        "attn2",
        "audio_attn1",
        "audio_attn2",
        "audio_to_video_attn",
        "video_to_audio_attn",
    )
    for block in transformer_blocks.values():
        for name in attention_names:
            attention = getattr(block, name, None)
            if attention is not None:
                attention.query_chunk_size = query_chunk_size


def resolve_attention_query_chunk_size(
    *,
    low_memory: bool,
    num_tokens: int,
    prefer_memory: bool = False,
) -> Optional[int]:
    """Choose an attention query chunk size from the active token count.

    Short clips can still produce large token counts at higher resolutions, so the
    decision should be based on tokens rather than clip duration alone.
    """
    if not (low_memory or prefer_memory):
        return None

    if num_tokens <= 8_192:
        return None
    if num_tokens <= 32_768:
        return 4_096
    return 2_048


def configure_attention_query_chunking(
    model: LTXModel,
    *,
    low_memory: bool,
    latent_frames: int,
    latent_h: int,
    latent_w: int,
    prefer_memory: bool = False,
) -> Optional[int]:
    """Apply token-aware attention chunking to the transformer in-place."""
    num_tokens = latent_frames * latent_h * latent_w
    query_chunk_size = resolve_attention_query_chunk_size(
        low_memory=low_memory,
        num_tokens=num_tokens,
        prefer_memory=prefer_memory,
    )
    set_attention_query_chunk_size(model, query_chunk_size)
    return query_chunk_size


def strip_transformer_to_video_only(model: LTXModel) -> None:
    """Drop audio/cross-modal modules so Stage 2 can run with a smaller resident model.

    This is intended for low-memory two-stage execution after Stage 1 has finished and
    audio latents are no longer refined. The function mutates the loaded transformer
    in-place, preserving the video path while deleting audio-only and A/V cross-modal
    modules to release unified memory before Stage 2.
    """
    video_preprocessor = getattr(model, "video_args_preprocessor", None)
    simple_preprocessor = getattr(video_preprocessor, "simple_preprocessor", None)
    if simple_preprocessor is not None:
        model.video_args_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=simple_preprocessor.patchify_proj,
            adaln=simple_preprocessor.adaln,
            caption_projection=simple_preprocessor.caption_projection,
            inner_dim=simple_preprocessor.inner_dim,
            max_pos=simple_preprocessor.max_pos,
            num_attention_heads=simple_preprocessor.num_attention_heads,
            use_middle_indices_grid=simple_preprocessor.use_middle_indices_grid,
            timestep_scale_multiplier=simple_preprocessor.timestep_scale_multiplier,
            positional_embedding_theta=simple_preprocessor.positional_embedding_theta,
            rope_type=simple_preprocessor.rope_type,
            double_precision_rope=simple_preprocessor.double_precision_rope,
            prompt_adaln=simple_preprocessor.prompt_adaln,
        )

    top_level_attrs = (
        "audio_positional_embedding_max_pos",
        "audio_num_attention_heads",
        "audio_inner_dim",
        "audio_cross_attention_dim",
        "audio_patchify_proj",
        "audio_adaln_single",
        "audio_prompt_adaln_single",
        "audio_caption_projection",
        "audio_scale_shift_table",
        "audio_norm_out",
        "audio_proj_out",
        "audio_args_preprocessor",
        "av_ca_video_scale_shift_adaln_single",
        "av_ca_audio_scale_shift_adaln_single",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_v2a_gate_adaln_single",
        "av_ca_timestep_scale_multiplier",
    )
    for attr in top_level_attrs:
        if hasattr(model, attr):
            delattr(model, attr)

    block_attrs = (
        "audio_attn1",
        "audio_attn2",
        "audio_ff",
        "audio_scale_shift_table",
        "audio_prompt_scale_shift_table",
        "audio_to_video_attn",
        "video_to_audio_attn",
        "scale_shift_table_a2v_ca_audio",
        "scale_shift_table_a2v_ca_video",
    )
    transformer_blocks = getattr(model, "transformer_blocks", {})
    for block in transformer_blocks.values():
        for attr in block_attrs:
            if hasattr(block, attr):
                delattr(block, attr)

    if hasattr(model, "model_type"):
        model.model_type = LTXModelType.VideoOnly
    if hasattr(model, "config"):
        model.config.model_type = LTXModelType.VideoOnly

    # 日本語概要:
    # 低メモリ化で transformer の構造を変更したあとは、
    # 以前の cached forward wrapper を再利用しないように明示的に破棄する。
    invalidate_distilled_transformer_compile_cache(model)

    gc.collect()
    mx.clear_cache()


def prepare_distilled_stage2_transformer(
    model: Optional[LTXModel],
    *,
    video_only: bool,
) -> None:
    """Prepare the distilled transformer for full-resolution Stage 2 execution."""
    if model is None:
        return

    if video_only:
        strip_transformer_to_video_only(model)
        return

    # 日本語概要:
    # Stage 1 用の cached forward wrapper は full-res Stage 2 では不要なので、
    # Stage 2 に入る前に破棄して shape ごとの参照を持ち越さないようにする。
    invalidate_distilled_transformer_compile_cache(model)
    gc.collect()
    mx.clear_cache()


class MemoryProfiler:
    """Phase-local MLX memory profiler using active/cache/peak counters."""

    def __init__(self, enabled: bool, console: Console):
        self.enabled = enabled
        self.console = console
        self.current_phase: Optional[str] = None
        self.phase_start_active = 0
        self.phase_start_cache = 0
        self.overall_peak = 0

    @staticmethod
    def _gib(value: int) -> float:
        return value / (1024**3)

    def _update_overall_peak(self, peak: int) -> None:
        self.overall_peak = max(self.overall_peak, peak)

    def start(self, phase: str) -> None:
        if not self.enabled:
            return
        mx.eval()
        self.current_phase = phase
        mx.reset_peak_memory()
        self.phase_start_active = mx.get_active_memory()
        self.phase_start_cache = mx.get_cache_memory()

    def capture(self, phase: Optional[str] = None) -> Optional[str]:
        if not self.enabled:
            return None
        mx.eval()
        label = phase or self.current_phase or "phase"
        active = mx.get_active_memory()
        cache = mx.get_cache_memory()
        peak = mx.get_peak_memory()
        self._update_overall_peak(peak)
        delta_active = active - self.phase_start_active
        delta_cache = cache - self.phase_start_cache
        return (
            "[dim]🧠 {label}: active={active:.2f}GB (Δ{delta_active:+.2f}), "
            "cache={cache:.2f}GB (Δ{delta_cache:+.2f}), peak={peak:.2f}GB[/]".format(
                label=label,
                active=self._gib(active),
                delta_active=self._gib(delta_active),
                cache=self._gib(cache),
                delta_cache=self._gib(delta_cache),
                peak=self._gib(peak),
            )
        )

    def log(self, phase: Optional[str] = None) -> None:
        message = self.capture(phase)
        if message is not None:
            self.console.print(message)

    def final_peak(self) -> int:
        if not self.enabled:
            return mx.get_peak_memory()
        self._update_overall_peak(mx.get_peak_memory())
        return self.overall_peak


def resolve_decode_tiling_mode(
    tiling: str,
    *,
    low_memory: bool,
    height: int,
    width: int,
    num_frames: int,
) -> str:
    """Choose the decode tiling mode for low-memory decode.

    For short clips, temporal tiling is often unnecessary overhead, so prefer
    spatial-only tiling once the output is large enough. Fall back to aggressive
    tiling for longer clips where temporal tiling still helps control memory.
    """
    if not low_memory or tiling != "conservative":
        return tiling

    output_pixels = height * width * num_frames
    if output_pixels < 75_000_000:
        return tiling

    if num_frames <= 241:
        return "spatial"

    if output_pixels >= 75_000_000:
        return "aggressive"

    return tiling


def resolve_stage2_sigma_schedule(
    stage2_refinement_steps: int,
    base_sigmas: Optional[list[float]] = None,
) -> list[float]:
    """Return the Stage 2 distilled sigma schedule for 0-3 refinement steps.

    The full distilled Stage 2 schedule is 3 steps. Fewer steps keep the same
    initial sigma and terminal 0.0 endpoint while skipping intermediate sigmas.
    """
    sigmas = base_sigmas or STAGE_2_SIGMAS
    schedules = {
        0: [],
        1: [sigmas[0], sigmas[-1]],
        2: [sigmas[0], sigmas[-2], sigmas[-1]],
        3: sigmas,
    }
    if stage2_refinement_steps not in schedules:
        raise ValueError(
            f"stage2_refinement_steps must be between 0 and 3, got {stage2_refinement_steps}"
        )
    return schedules[stage2_refinement_steps]


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = AUDIO_SAMPLE_RATE):
    """Save audio to WAV with a high-fidelity path when soundfile is available."""
    if audio.ndim == 2:
        audio = audio.T

    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)

    try:
        import soundfile as sf

        sf.write(str(path), audio, sample_rate, subtype="PCM_24")
        return
    except Exception:
        pass

    import wave

    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2 if audio_int16.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def mux_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_bitrate: str = "320k",
):
    """Combine video and audio into final output using ffmpeg."""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]FFmpeg error: {e.stderr.decode()}[/]")
        return False
    except FileNotFoundError:
        console.print("[red]FFmpeg not found. Please install ffmpeg.[/]")
        return False


# =============================================================================
# Unified Generate Function
# =============================================================================


def resolve_text_encoder_path(
    model_path: Path,
    text_encoder_repo: Optional[str],
) -> Path:
    """Resolve the text encoder path for LTX generation.

    Monolithic/original model repos include the text encoder at the repo root.
    Reformatted repos such as LTX-2.3 distilled/dev ship transformer + connector
    weights separately and require a standalone Gemma repo.
    """
    if text_encoder_repo is not None:
        return get_model_path(
            text_encoder_repo,
            allow_patterns=TEXT_ENCODER_ALLOW_PATTERNS,
            required_patterns=["*.safetensors", "tokenizer.json"],
        )

    has_embedded_text_encoder = (model_path / "config.json").exists() or (
        model_path / "text_encoder"
    ).is_dir()
    if has_embedded_text_encoder:
        return model_path

    console.print(
        "[dim]No embedded text encoder found; "
        f"using {DEFAULT_TEXT_ENCODER_REPO}[/]"
    )
    return get_model_path(
        DEFAULT_TEXT_ENCODER_REPO,
        allow_patterns=TEXT_ENCODER_ALLOW_PATTERNS,
        required_patterns=["*.safetensors", "tokenizer.json"],
    )


def resolve_generation_model_repo(
    model_repo: str,
    pipeline: PipelineType,
) -> str:
    """Resolve the effective model repo/path used for generation.

    The README examples often use a local `./LTX-2.3-distilled` checkout. When
    that directory is missing, the distilled pipeline should gracefully fall
    back to the canonical Hugging Face repo so the weights are auto-downloaded.
    """
    normalized = model_repo.strip()
    if pipeline != PipelineType.DISTILLED:
        return normalized

    if not normalized:
        console.print(
            f"[dim]No distilled model repo provided; using {DEFAULT_DISTILLED_MODEL_REPO}[/]"
        )
        return DEFAULT_DISTILLED_MODEL_REPO

    candidate_path = Path(normalized).expanduser()
    if candidate_path.exists():
        return str(candidate_path)

    if (
        normalized != DEFAULT_DISTILLED_MODEL_REPO
        and candidate_path.name == Path(DEFAULT_DISTILLED_MODEL_REPO).name
    ):
        console.print(
            "[dim]Local distilled model not found at "
            f"{model_repo}; using {DEFAULT_DISTILLED_MODEL_REPO}[/]"
        )
        return DEFAULT_DISTILLED_MODEL_REPO

    return normalized


def generate_video(
    model_repo: str,
    text_encoder_repo: Optional[str],
    prompt: str,
    pipeline: PipelineType = PipelineType.DISTILLED,
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
    tiling: str = "auto",
    stream: bool = False,
    audio: bool = False,
    output_audio_path: Optional[str] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 1.0,
    stg_blocks: Optional[list] = None,
    modality_scale: float = 3.0,
    stg_skip_step: int = 1,
    modality_skip_step: int = 1,
    lora_path: Optional[str] = None,
    lora_strength: float = 1.0,
    lora_strength_stage_1: Optional[float] = None,
    lora_strength_stage_2: Optional[float] = None,
    custom_lora_path: Optional[str] = None,
    custom_lora_strength: float = 1.0,
    distilled_lora_path: Optional[str] = None,
    distilled_lora_strength_stage_1: Optional[float] = None,
    distilled_lora_strength_stage_2: Optional[float] = None,
    audio_file: Optional[str] = None,
    audio_start_time: float = 0.0,
    spatial_upscaler: Optional[str] = None,
    low_memory: bool = False,
    profile_memory: bool = False,
    skip_stage2_refinement: bool = False,
    stage2_refinement_steps: Optional[int] = None,
    disable_stage2_audio_refinement: bool = False,
    preserve_stage2_audio_refinement: bool = False,
    transformer_quantization_bits: Optional[int] = None,
    transformer_quantization_group_size: Optional[int] = None,
    transformer_quantization_mode: str = "affine",
    transformer_quantize_inputs: bool = False,
    dev_two_stage_sigma_preset: str = "default",
    return_video: bool = True,
    audio_bitrate: str = "320k",
):
    """Generate video using LTX-2 / LTX-2.3 models.

    Supports four pipelines:
    - DISTILLED: Two-stage generation with upsampling, fixed sigma schedules, no CFG
    - DEV: Single-stage generation with dynamic sigmas and CFG
    - DEV_TWO_STAGE: Stage 1 dev (half res, CFG) + upsample + stage 2 distilled with LoRA (full res, no CFG)
    - DEV_TWO_STAGE_HQ: res_2s sampler, LoRA both stages (0.25/0.5), lower rescale

    Args:
        model_repo: Model repository ID
        text_encoder_repo: Optional separate text encoder repository ID
        prompt: Text description of the video to generate
        pipeline: Pipeline type (DISTILLED or DEV)
        negative_prompt: Negative prompt for CFG (dev pipeline only)
        height: Output video height (must be divisible by 32/64)
        width: Output video width (must be divisible by 32/64)
        num_frames: Number of frames (must be 1 + 8*k)
        num_inference_steps: Number of denoising steps (dev pipeline only)
        cfg_scale: Guidance scale for CFG (dev pipeline only)
        seed: Random seed for reproducibility
        fps: Frames per second for output video
        output_path: Path to save the output video
        save_frames: Whether to save individual frames as images
        verbose: Whether to print progress
        enhance_prompt: Whether to enhance prompt using Gemma
        max_tokens: Max tokens for prompt enhancement
        temperature: Temperature for prompt enhancement
        image: Path to conditioning image for I2V
        image_strength: Conditioning strength for I2V
        image_frame_idx: Frame index to condition for I2V
        tiling: Tiling mode for VAE decoding
        stream: Stream frames to output as they're decoded
        audio: Enable synchronized audio generation
        output_audio_path: Path to save audio file
        use_apg: Use Adaptive Projected Guidance instead of CFG (more stable for I2V)
        apg_eta: APG parallel component weight (1.0 = keep full parallel)
        apg_norm_threshold: APG guidance norm clamp (0 = no clamping)
        stg_skip_step: Run STG perturbation every Nth denoising step for
            `dev` / `dev-two-stage`. 1 keeps the current behavior.
        modality_skip_step: Run cross-modal isolation every Nth denoising
            step for `dev` / `dev-two-stage`. 1 keeps the current behavior.
        custom_lora_path: Optional custom/user LoRA for dev-two-stage. When split
            distilled LoRA controls are enabled, legacy --lora-path is also
            accepted as an alias for this path.
        custom_lora_strength: Merge strength for the custom/user LoRA.
        distilled_lora_path: Optional distilled LoRA path for dev-two-stage.
            If omitted while stage strengths are provided, auto-detect from the
            loaded model directory.
        distilled_lora_strength_stage_1: Distilled LoRA strength for Stage 1 in
            dev-two-stage split-LoRA mode.
        distilled_lora_strength_stage_2: Distilled LoRA strength for Stage 2 in
            dev-two-stage split-LoRA mode. Must be >= Stage 1 because LoRA
            merging is additive in-place.
        low_memory: Apply safer low-memory defaults for Apple Silicon execution.
        profile_memory: Print phase-local MLX memory measurements.
        skip_stage2_refinement: Experimental two-stage shortcut that decodes the
            upsampled Stage 1 result without running Stage 2 refinement.
        stage2_refinement_steps: Experimental Stage 2 step count override for
            two-stage pipelines. Valid values are 0-3.
        disable_stage2_audio_refinement: Run Stage 2 in video-only mode for the
            distilled pipeline and reuse Stage 1 audio latents unchanged.
        preserve_stage2_audio_refinement: Keep Stage 2 audio refinement enabled
            even when --low-memory is active. This raises peak memory.
        transformer_quantization_bits: Runtime transformer quantization bit-width.
            Use 8 to reduce memory on unquantized LTX-2.3 checkpoints.
        transformer_quantization_group_size: Group size for runtime transformer
            quantization. Defaults depend on mode: 64 for affine, 32 for mxfp8.
        transformer_quantization_mode: Runtime transformer quantization mode.
            Supports "affine" and experimental "mxfp8".
        transformer_quantize_inputs: Quantize transformer activations on the
            fly in addition to weights. Currently only supported for mxfp8.
        dev_two_stage_sigma_preset: Sigma schedule preset for dev-two-stage.
            "default" keeps the existing dynamic dev Stage 1 scheduler and
            legacy Stage 2 sigmas; "official" matches the official 2.3
            two-stage distilled workflow schedules (Stage 1 fixed 8-step
            manual sigma schedule, Stage 2 official 0.85-based schedule).
        return_video: Whether to return the decoded video array to the caller.
            CLI execution can disable this to avoid a final full-video memory spike.
        audio_bitrate: AAC bitrate used when muxing audio back into the MP4.
    """
    start_time = time.time()
    memory_profiler = MemoryProfiler(enabled=profile_memory, console=console)

    # Validate dimensions
    is_two_stage = pipeline in (
        PipelineType.DISTILLED,
        PipelineType.DEV_TWO_STAGE,
        PipelineType.DEV_TWO_STAGE_HQ,
    )
    divisor = 64 if is_two_stage else 32
    assert height % divisor == 0, f"Height must be divisible by {divisor}, got {height}"
    assert width % divisor == 0, f"Width must be divisible by {divisor}, got {width}"

    if (skip_stage2_refinement or stage2_refinement_steps is not None) and not is_two_stage:
        raise ValueError(
            "Stage 2 refinement controls are only supported for two-stage pipelines."
        )
    if stg_skip_step < 1:
        raise ValueError(f"stg_skip_step must be >= 1, got {stg_skip_step}")
    if modality_skip_step < 1:
        raise ValueError(
            f"modality_skip_step must be >= 1, got {modality_skip_step}"
        )
    if (
        (stg_skip_step != 1 or modality_skip_step != 1)
        and pipeline not in (PipelineType.DEV, PipelineType.DEV_TWO_STAGE)
    ):
        raise ValueError(
            "Guidance skip-step controls are only supported for --pipeline dev and dev-two-stage."
        )
    if skip_stage2_refinement and stage2_refinement_steps not in (None, 0):
        raise ValueError(
            "--skip-stage2-refinement cannot be combined with --stage2-refinement-steps > 0."
        )
    if disable_stage2_audio_refinement and preserve_stage2_audio_refinement:
        raise ValueError(
            "--disable-stage2-audio-refinement cannot be combined with "
            "--preserve-stage2-audio-refinement."
        )

    split_dev_two_stage_lora_controls = any(
        value is not None
        for value in (
            custom_lora_path,
            distilled_lora_path,
            distilled_lora_strength_stage_1,
            distilled_lora_strength_stage_2,
        )
    ) or custom_lora_strength != 1.0
    if split_dev_two_stage_lora_controls and pipeline != PipelineType.DEV_TWO_STAGE:
        raise ValueError(
            "Custom/distilled split LoRA controls are only supported for --pipeline dev-two-stage."
        )
    if dev_two_stage_sigma_preset != "default" and pipeline != PipelineType.DEV_TWO_STAGE:
        raise ValueError(
            "--dev-two-stage-sigma-preset is only supported for --pipeline dev-two-stage."
        )

    stage2_sigma_base = (
        OFFICIAL_DEV_TWO_STAGE_STAGE_2_SIGMAS
        if pipeline == PipelineType.DEV_TWO_STAGE
        and dev_two_stage_sigma_preset == "official"
        else STAGE_2_SIGMAS
    )

    effective_stage2_steps = (
        0
        if skip_stage2_refinement
        else (
            stage2_refinement_steps
            if stage2_refinement_steps is not None
            else len(STAGE_2_SIGMAS) - 1
        )
    )
    stage2_sigmas = resolve_stage2_sigma_schedule(
        effective_stage2_steps,
        base_sigmas=stage2_sigma_base,
    )
    skip_stage2_refinement = effective_stage2_steps == 0

    if num_frames % 8 != 1:
        adjusted_num_frames = round((num_frames - 1) / 8) * 8 + 1
        console.print(
            f"[yellow]⚠️  Number of frames must be 1 + 8*k. Using: {adjusted_num_frames}[/]"
        )
        num_frames = adjusted_num_frames

    is_i2v = image is not None
    is_a2v = audio_file is not None
    if is_a2v and audio:
        raise ValueError(
            "Cannot use both --audio-file (A2V) and --audio (generate audio). Choose one."
        )
    # A2V implicitly enables audio path through the transformer
    if is_a2v:
        audio = True
    if preserve_stage2_audio_refinement and not is_two_stage:
        raise ValueError(
            "--preserve-stage2-audio-refinement is only supported for two-stage pipelines."
        )
    if preserve_stage2_audio_refinement and not audio:
        raise ValueError(
            "--preserve-stage2-audio-refinement requires --audio or --audio-file."
        )

    low_memory_adjustments = []
    stage2_video_only = False
    if low_memory:
        if tiling in ("auto", "none", "default"):
            tiling = "conservative"
            low_memory_adjustments.append("tiling=conservative")
        if pipeline in (
            PipelineType.DEV,
            PipelineType.DEV_TWO_STAGE,
            PipelineType.DEV_TWO_STAGE_HQ,
        ):
            if stg_scale != 0.0:
                stg_scale = 0.0
                low_memory_adjustments.append("stg=off")
            if modality_scale != 1.0:
                modality_scale = 1.0
                low_memory_adjustments.append("modality-guidance=off")
        low_memory_adjustments.append("query-chunk=adaptive")

    if pipeline in (PipelineType.DISTILLED, PipelineType.DEV_TWO_STAGE):
        preserve_stage2_audio = preserve_stage2_audio_refinement and audio
        stage2_video_only = (not audio) or disable_stage2_audio_refinement or (
            low_memory and audio and not preserve_stage2_audio
        )
        if (
            audio
            and stage2_video_only
            and "stage2-audio-refine=off" not in low_memory_adjustments
        ):
            low_memory_adjustments.append("stage2-audio-refine=off")
        elif (
            low_memory
            and preserve_stage2_audio
            and "stage2-audio-refine=on" not in low_memory_adjustments
        ):
            low_memory_adjustments.append("stage2-audio-refine=on")

    # 既存の two-stage 分岐が使っている名前を残して互換性を保つ。
    low_memory_stage2_video_only = stage2_video_only

    mode_str = "I2V" if is_i2v else "T2V"
    if is_a2v:
        mode_str = "A2V" + ("+I2V" if is_i2v else "")
    elif audio:
        mode_str += "+Audio"

    pipeline_names = {
        PipelineType.DISTILLED: "DISTILLED",
        PipelineType.DEV: "DEV",
        PipelineType.DEV_TWO_STAGE: "DEV-TWO-STAGE",
        PipelineType.DEV_TWO_STAGE_HQ: "DEV-TWO-STAGE-HQ",
    }
    pipeline_name = pipeline_names[pipeline]
    header = f"[bold cyan]🎬 [{pipeline_name}] [{mode_str}] {width}x{height} • {num_frames} frames[/]"
    console.print(Panel(header, expand=False))
    console.print(f"[dim]Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}[/]")
    if low_memory:
        adjustments = ", ".join(low_memory_adjustments) if low_memory_adjustments else "none"
        console.print(f"[dim]Low-memory mode: on ({adjustments})[/]")
    if profile_memory:
        console.print("[dim]Memory profiling: on[/]")
    if pipeline == PipelineType.DEV_TWO_STAGE and dev_two_stage_sigma_preset == "official":
        console.print(
            "[dim]DEV two-stage sigma preset: official two-stage distilled "
            "(Stage 1 fixed 8-step manual sigmas, Stage 2 official 0.85-based schedule)[/]"
        )
    if skip_stage2_refinement:
        console.print("[dim]Stage 2 refinement: skipped after upsample (experimental)[/]")
    elif is_two_stage and effective_stage2_steps != len(STAGE_2_SIGMAS) - 1:
        console.print(
            f"[dim]Stage 2 refinement: {effective_stage2_steps}/{len(STAGE_2_SIGMAS) - 1} steps (experimental)[/]"
        )
        console.print(
            "[yellow]Motion quality note: fewer Stage 2 steps can soften motion detail and increase blur.[/]"
        )
    if disable_stage2_audio_refinement and stage2_video_only:
        console.print(
            "[dim]Stage 2 audio refinement: off (reuse Stage 1 audio latents, lower memory)[/]"
        )
    elif low_memory and audio and not stage2_video_only:
        console.print(
            "[yellow]Low-memory override: keeping Stage 2 audio refinement enabled may increase peak memory.[/]"
        )

    if pipeline in (
        PipelineType.DEV,
        PipelineType.DEV_TWO_STAGE,
        PipelineType.DEV_TWO_STAGE_HQ,
    ):
        steps_display = str(num_inference_steps)
        if pipeline == PipelineType.DEV_TWO_STAGE and dev_two_stage_sigma_preset == "official":
            steps_display = "official manual 8-step preset"
        audio_cfg_info = f", Audio CFG: {audio_cfg_scale}" if audio else ""
        stg_skip_info = (
            f" every {stg_skip_step} steps" if stg_scale != 0.0 and stg_skip_step > 1 else ""
        )
        mod_skip_info = (
            f" every {modality_skip_step} steps"
            if modality_scale != 1.0 and modality_skip_step > 1
            else ""
        )
        stg_info = (
            f", STG: {stg_scale} blocks={stg_blocks}{stg_skip_info}"
            if stg_scale != 0.0
            else ""
        )
        mod_info = (
            f", Modality: {modality_scale}{mod_skip_info}"
            if modality_scale != 1.0
            else ""
        )
        console.print(
            f"[dim]Steps: {steps_display}, CFG: {cfg_scale}{audio_cfg_info}, Rescale: {cfg_rescale}{stg_info}{mod_info}[/]"
        )

    if is_i2v:
        console.print(
            f"[dim]Image: {image} (strength={image_strength}, frame={image_frame_idx})[/]"
        )

    # Always compute audio frames - PyTorch distilled pipeline unconditionally
    # generates audio alongside video (model was trained with joint audio-video).
    # The --audio flag only controls whether audio is decoded and saved to output.
    audio_frames = compute_audio_frames(num_frames, fps)
    if audio:
        console.print(
            f"[dim]Audio: {audio_frames} latent frames @ {AUDIO_SAMPLE_RATE}Hz[/]"
        )

    # Get model path
    model_repo = resolve_generation_model_repo(model_repo, pipeline)
    model_path = get_model_path(model_repo)
    text_encoder_path = resolve_text_encoder_path(
        model_path=model_path,
        text_encoder_repo=text_encoder_repo,
    )
    split_stage_custom_lora_path = None
    split_stage_custom_lora_strength = custom_lora_strength
    split_stage_distilled_lora_path = None
    split_stage_distilled_lora_strength_s1 = distilled_lora_strength_stage_1
    split_stage_distilled_lora_strength_s2 = distilled_lora_strength_stage_2
    use_split_dev_two_stage_loras = (
        pipeline == PipelineType.DEV_TWO_STAGE and split_dev_two_stage_lora_controls
    )
    if use_split_dev_two_stage_loras:
        split_stage_custom_lora_path = custom_lora_path or lora_path
        if split_stage_custom_lora_path == lora_path and custom_lora_strength == 1.0:
            split_stage_custom_lora_strength = lora_strength

        if (
            split_stage_distilled_lora_strength_s1 is None
            and split_stage_distilled_lora_strength_s2 is None
            and distilled_lora_path is not None
        ):
            split_stage_distilled_lora_strength_s1 = 0.5
            split_stage_distilled_lora_strength_s2 = 0.5
        elif (
            split_stage_distilled_lora_strength_s1 is None
            and split_stage_distilled_lora_strength_s2 is not None
        ):
            split_stage_distilled_lora_strength_s1 = split_stage_distilled_lora_strength_s2
        elif (
            split_stage_distilled_lora_strength_s2 is None
            and split_stage_distilled_lora_strength_s1 is not None
        ):
            split_stage_distilled_lora_strength_s2 = split_stage_distilled_lora_strength_s1

        if (
            split_stage_distilled_lora_strength_s1 is not None
            and split_stage_distilled_lora_strength_s2 is not None
            and split_stage_distilled_lora_strength_s2
            < split_stage_distilled_lora_strength_s1
        ):
            raise ValueError(
                "Distilled LoRA Stage 2 strength must be >= Stage 1 strength for dev-two-stage."
            )

        if distilled_lora_path is not None:
            split_stage_distilled_lora_path = distilled_lora_path
        elif (
            (split_stage_distilled_lora_strength_s1 or 0.0) > 0.0
            or (split_stage_distilled_lora_strength_s2 or 0.0) > 0.0
        ):
            distilled_lora_file = find_distilled_lora_file(model_path)
            if distilled_lora_file is None:
                raise FileNotFoundError(
                    "No distilled LoRA file found in the model directory. "
                    "Pass --distilled-lora-path explicitly."
                )
            split_stage_distilled_lora_path = str(distilled_lora_file)
            console.print(
                f"[dim]Auto-detected distilled LoRA: {distilled_lora_file.name}[/]"
            )

    # Resolve spatial upscaler path for two-stage pipelines
    upscaler_path = None
    upscaler_scale = 2.0
    if is_two_stage:
        if spatial_upscaler is not None:
            # User-specified upscaler file
            upscaler_path = (
                model_path / spatial_upscaler
                if not Path(spatial_upscaler).is_absolute()
                else Path(spatial_upscaler)
            )
            if not upscaler_path.exists():
                # Try as a filename within model_path
                upscaler_path = model_path / spatial_upscaler
            # Detect scale from filename
            if "x1.5" in str(upscaler_path):
                upscaler_scale = 1.5
            elif "x2" in str(upscaler_path):
                upscaler_scale = 2.0
        else:
            # Auto-detect: prefer the newest x2 upscaler variant
            upscaler_files = sorted(
                model_path.glob("*spatial-upscaler-x2*.safetensors")
            )
            if upscaler_files:
                upscaler_path = upscaler_files[-1]
                upscaler_scale = 2.0

    # Calculate latent dimensions
    if is_two_stage:
        # Stage 1 always at half resolution (matches PyTorch)
        stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
        # Stage 2 resolution = stage 1 * upscaler scale
        stage2_h = int(stage1_h * upscaler_scale)
        stage2_w = int(stage1_w * upscaler_scale)
    else:
        latent_h, latent_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    mx.random.seed(seed)

    # Read transformer config to detect model version
    import json

    transformer_config_path = model_path / "transformer" / "config.json"
    has_prompt_adaln = False
    if transformer_config_path.exists():
        with open(transformer_config_path) as f:
            has_prompt_adaln = json.load(f).get("has_prompt_adaln", False)

    # Load text encoder
    memory_profiler.start("text encoder")
    with console.status("[blue]📝 Loading text encoder...[/]", spinner="dots"):
        from mlx_video.models.ltx_2.text_encoder import LTX2TextEncoder

        text_encoder = LTX2TextEncoder(has_prompt_adaln=has_prompt_adaln)
        text_encoder.load(model_path=model_path, text_encoder_path=text_encoder_path)
        mx.eval(text_encoder.parameters())
    console.print("[green]✓[/] Text encoder loaded")

    # Optionally enhance the prompt
    if enhance_prompt:
        console.print("[bold magenta]✨ Enhancing prompt[/]")
        prompt = text_encoder.enhance_t2v(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            verbose=verbose,
        )
        console.print(
            f"[dim]Enhanced: {prompt[:150]}{'...' if len(prompt) > 150 else ''}[/]"
        )

    # Encode prompts - always get audio embeddings since the model was trained
    # with joint audio-video processing (PyTorch unconditionally generates audio)
    if pipeline in (
        PipelineType.DEV,
        PipelineType.DEV_TWO_STAGE,
        PipelineType.DEV_TWO_STAGE_HQ,
    ):
        # Dev/dev-two-stage pipelines need positive and negative embeddings for CFG
        video_embeddings_pos, audio_embeddings_pos = text_encoder(
            prompt, return_audio_embeddings=True
        )
        video_embeddings_neg, audio_embeddings_neg = text_encoder(
            negative_prompt, return_audio_embeddings=True
        )
        model_dtype = video_embeddings_pos.dtype
        mx.eval(
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
        )
        # For dev-two-stage, stage 2 uses single positive embedding (no CFG)
        if pipeline in (PipelineType.DEV_TWO_STAGE, PipelineType.DEV_TWO_STAGE_HQ):
            text_embeddings = video_embeddings_pos
    else:
        # Distilled pipeline - single embedding
        text_embeddings, audio_embeddings = text_encoder(
            prompt, return_audio_embeddings=True
        )
        mx.eval(text_embeddings, audio_embeddings)
        model_dtype = text_embeddings.dtype

    del text_encoder
    mx.clear_cache()
    memory_profiler.log("text embeddings ready")

    # Load transformer
    transformer_quantization = resolve_transformer_quantization(
        bits=transformer_quantization_bits,
        group_size=transformer_quantization_group_size,
        mode=transformer_quantization_mode,
        quantize_input=transformer_quantize_inputs,
    )
    runtime_quantized_lora_preprocessor = None
    runtime_quantized_lora_premerged = False
    if (
        pipeline == PipelineType.DISTILLED
        and lora_path is not None
        and transformer_quantization is not None
        and transformer_quantization["mode"] == "mxfp8"
    ):
        runtime_quantized_lora_preprocessor, resolved_lora_file = (
            _build_runtime_quantized_lora_preprocessor(lora_path, lora_strength)
        )
        lora_path = str(resolved_lora_file)
        runtime_quantized_lora_premerged = True
    transformer_desc = f"🤖 Loading {pipeline_name.lower()} transformer{' (A/V mode)' if audio else ''}..."
    memory_profiler.start("transformer load")
    with console.status(f"[blue]{transformer_desc}[/]", spinner="dots"):
        transformer = LTXModel.from_pretrained(
            model_path=model_path / "transformer",
            strict=True,
            quantization=transformer_quantization,
            weight_preprocessor=runtime_quantized_lora_preprocessor,
        )

    console.print("[green]✓[/] Transformer loaded")
    if transformer_quantization is not None:
        activation_quantized = getattr(
            transformer,
            "_activation_quantization_enabled",
            transformer_quantization.get("quantize_input", False),
        )
        console.print(
            "[dim]"
            f"Transformer runtime quantization: {transformer_quantization['mode']} "
            f"{transformer_quantization['bits']}-bit "
            f"(group_size={transformer_quantization['group_size']}, "
            f"input_quantized={'yes' if activation_quantized else 'no'})"
            "[/]"
        )
        if transformer_quantization.get("quantize_input") and getattr(
            transformer, "_activation_quantization_fallback", False
        ):
            console.print(
                "[yellow]Activation quantization is not available on this MLX/Metal runtime; "
                "falling back to weight-only mxfp8.[/]"
            )
        if runtime_quantized_lora_premerged:
            console.print(
                "[dim]LoRA pre-merged into raw weights before runtime mxfp8 quantization[/]"
            )
    memory_profiler.log("transformer loaded")

    # Auto-detect stg_blocks from transformer config if not explicitly provided.
    # LTX-2.3 (has_prompt_adaln=True) uses block 28; LTX-2 uses block 29.
    if stg_blocks is None and stg_scale != 0.0:
        if transformer.config.has_prompt_adaln:
            stg_blocks = [28]
        else:
            stg_blocks = [29]
        console.print(
            f"[dim]Auto-detected STG blocks: {stg_blocks} (model={'2.3' if transformer.config.has_prompt_adaln else '2'})[/]"
        )

    # ==========================================================================
    # A2V: Encode input audio to frozen latents
    # ==========================================================================
    a2v_audio_latents = None
    a2v_waveform = None
    a2v_sr = None
    if is_a2v:
        from mlx_video.models.ltx_2.audio_vae import AudioEncoder
        from mlx_video.models.ltx_2.audio_vae.audio_processor import (
            ensure_stereo,
            load_audio,
            waveform_to_mel,
        )
        from mlx_video.models.ltx_2.utils import convert_audio_encoder

        memory_profiler.start("a2v encode")
        with console.status(
            "[blue]Loading and encoding input audio (A2V)...[/]", spinner="dots"
        ):
            video_duration = num_frames / fps

            # Load audio
            waveform, sr = load_audio(
                audio_file,
                target_sr=AUDIO_LATENT_SAMPLE_RATE,
                start_time=audio_start_time,
                max_duration=video_duration,
            )
            waveform = ensure_stereo(waveform)
            a2v_waveform = waveform.copy()
            a2v_sr = sr

            # Compute mel-spectrogram
            mel = waveform_to_mel(
                waveform,
                sample_rate=sr,
                n_fft=1024,
                hop_length=AUDIO_HOP_LENGTH,
                n_mels=64,
            )

            # Convert audio encoder weights if needed, then load
            encoder_dir = convert_audio_encoder(
                model_path, source_repo="Lightricks/LTX-2"
            )
            audio_encoder = AudioEncoder.from_pretrained(encoder_dir)
            mx.eval(audio_encoder.parameters())

            # Encode: (1, 2, time, 64) -> normalized latents
            encoded = audio_encoder(mel)
            mx.eval(encoded)

            # encoded is in MLX format (B, T', mel_bins', z_channels) = (1, T', 16, 8)
            # Convert to PyTorch-style format for consistency: (B, C, T, mel_bins)
            a2v_audio_latents = mx.transpose(encoded, (0, 3, 1, 2)).astype(model_dtype)

            # Trim/pad to match expected audio_frames
            t_encoded = a2v_audio_latents.shape[2]
            if t_encoded > audio_frames:
                a2v_audio_latents = a2v_audio_latents[:, :, :audio_frames, :]
            elif t_encoded < audio_frames:
                pad_size = audio_frames - t_encoded
                padding = mx.zeros(
                    (1, AUDIO_LATENT_CHANNELS, pad_size, AUDIO_MEL_BINS),
                    dtype=model_dtype,
                )
                a2v_audio_latents = mx.concatenate([a2v_audio_latents, padding], axis=2)
            mx.eval(a2v_audio_latents)

            del audio_encoder
            mx.clear_cache()

        console.print(
            f"[green]✓[/] Audio encoded ({a2v_audio_latents.shape[2]} frames from {audio_file})"
        )
        memory_profiler.log("a2v encoded")

    # ==========================================================================
    # Pipeline-specific generation logic
    # ==========================================================================

    if pipeline == PipelineType.DISTILLED:
        # ======================================================================
        # DISTILLED PIPELINE: Two-stage with upsampling
        # ======================================================================

        # Load VAE encoder for I2V
        reuse_conditioning_encoder = (
            is_i2v and not low_memory and not skip_stage2_refinement
        )
        conditioning_vae_encoder = None
        stage1_image_latent = None
        if is_i2v:
            with console.status(
                "[blue]🖼️  Encoding Stage 1 image conditioning...[/]", spinner="dots"
            ):
                s1_h, s1_w = stage1_h * 32, stage1_w * 32
                if reuse_conditioning_encoder and conditioning_vae_encoder is None:
                    conditioning_vae_encoder = VideoEncoder.from_pretrained(
                        model_path / "vae" / "encoder"
                    )
                stage1_image_latent = encode_conditioning_image_latent(
                    model_path=model_path,
                    image_path=image,
                    height=s1_h,
                    width=s1_w,
                    dtype=model_dtype,
                    vae_encoder=conditioning_vae_encoder,
                )
            console.print("[green]✓[/] Stage 1 image conditioning encoded")

        # Merge distilled LoRA before stage 1 so it affects the full distilled
        # pipeline (stage 1 generation + stage 2 refinement) with a single merge.
        if lora_path is not None and not runtime_quantized_lora_premerged:
            with console.status(
                f"[blue]🔧 Merging distilled LoRA (stages 1+2, strength={lora_strength})...[/]",
                spinner="dots",
            ):
                load_and_merge_lora(transformer, lora_path, strength=lora_strength)

        # Stage 1
        console.print(
            f"\n[bold yellow]⚡ Stage 1:[/] Generating at {stage1_w*32}x{stage1_h*32} (8 steps)"
        )
        memory_profiler.start("stage 1 denoise")
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        # Init audio latents/positions: use encoded A2V latents or random
        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS)
            ).astype(model_dtype)
        )
        mx.eval(audio_positions, audio_latents)

        # Apply I2V conditioning
        state1 = None
        if is_i2v and stage1_image_latent is not None:
            latent_shape = (1, 128, latent_frames, stage1_h, stage1_w)
            state1 = LatentState(
                latent=mx.zeros(latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditioning = VideoConditionByLatentIndex(
                latent=stage1_image_latent,
                frame_idx=image_frame_idx,
                strength=image_strength,
            )
            state1 = apply_conditioning(state1, [conditioning])

            noise = mx.random.normal(latent_shape, dtype=model_dtype)
            noise_scale = mx.array(STAGE_1_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask
                + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(
                (1, 128, latent_frames, stage1_h, stage1_w), dtype=model_dtype
            )
            mx.eval(latents)

        configure_attention_query_chunking(
            transformer,
            low_memory=low_memory,
            latent_frames=latent_frames,
            latent_h=stage1_h,
            latent_w=stage1_w,
        )

        latents, audio_latents = denoise_distilled(
            latents,
            positions,
            text_embeddings,
            transformer,
            STAGE_1_SIGMAS,
            verbose=verbose,
            state=state1,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings,
            audio_frozen=is_a2v,
        )

        if state1 is not None:
            del state1
        if stage1_image_latent is not None:
            del stage1_image_latent
        mx.clear_cache()
        memory_profiler.log("stage 1 complete")

        if not skip_stage2_refinement and transformer is not None:
            memory_profiler.start("stage 2 prepare")
            if stage2_video_only:
                console.print(
                    "[dim]Stage 2: stripping transformer audio/cross-modal modules before upsampling[/]"
                )
            prepare_distilled_stage2_transformer(
                transformer,
                video_only=stage2_video_only,
            )
            memory_profiler.log("stage 2 transformer ready")
        elif skip_stage2_refinement and transformer is not None:
            del transformer
            transformer = None
            mx.clear_cache()

        # Upsample latents
        memory_profiler.start("upsample")
        with console.status(
            f"[magenta]🔍 Upsampling latents {upscaler_scale}x...[/]", spinner="dots"
        ):
            if upscaler_path is None or not upscaler_path.exists():
                raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
            upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
            mx.eval(upsampler.parameters())

            vae_mean, vae_std = load_video_decoder_statistics(model_path)

            latents = upsample_latents(
                latents,
                upsampler,
                vae_mean,
                vae_std,
            )
            mx.eval(latents)

            del upsampler, vae_mean, vae_std
            mx.clear_cache()
        console.print("[green]✓[/] Latents upsampled")
        memory_profiler.log("upsample complete")

        if skip_stage2_refinement:
            console.print(
                "[dim]Stage 2 refinement skipped; decoding upsampled Stage 1 latents[/]"
            )

        # Stage 2
        if not skip_stage2_refinement:
            console.print(
                f"\n[bold yellow]⚡ Stage 2:[/] Refining at {stage2_w*32}x{stage2_h*32} ({effective_stage2_steps} steps)"
            )
            if stage2_video_only:
                console.print(
                    "[dim]Stage 2 audio refinement disabled; reusing Stage 1 audio latents[/]"
                )
            positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
            mx.eval(positions)

            state2 = None
            if is_i2v:
                memory_profiler.start("stage 2 conditioning")
                with console.status(
                    "[blue]🖼️  Encoding Stage 2 image conditioning...[/]", spinner="dots"
                ):
                    s2_h, s2_w = stage2_h * 32, stage2_w * 32
                    stage2_image_latent = encode_conditioning_image_latent(
                        model_path=model_path,
                        image_path=image,
                        height=s2_h,
                        width=s2_w,
                        dtype=model_dtype,
                        vae_encoder=conditioning_vae_encoder,
                    )
                console.print("[green]✓[/] Stage 2 image conditioning encoded")

                state2 = LatentState(
                    latent=latents,
                    clean_latent=mx.zeros_like(latents),
                    denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
                )
                conditioning = VideoConditionByLatentIndex(
                    latent=stage2_image_latent,
                    frame_idx=image_frame_idx,
                    strength=image_strength,
                )
                state2 = apply_conditioning(state2, [conditioning])

                noise = mx.random.normal(latents.shape).astype(model_dtype)
                noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
                scaled_mask = state2.denoise_mask * noise_scale
                state2 = LatentState(
                    latent=noise * scaled_mask
                    + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                    clean_latent=state2.clean_latent,
                    denoise_mask=state2.denoise_mask,
                )
                latents = state2.latent
                mx.eval(latents)
                del stage2_image_latent
                memory_profiler.log("stage 2 conditioning ready")
                if conditioning_vae_encoder is not None:
                    del conditioning_vae_encoder
                    conditioning_vae_encoder = None
                    mx.clear_cache()
            else:
                noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
                one_minus_scale = mx.array(1.0 - stage2_sigmas[0], dtype=model_dtype)
                noise = mx.random.normal(latents.shape).astype(model_dtype)
                latents = noise * noise_scale + latents * one_minus_scale
                mx.eval(latents)

            configure_attention_query_chunking(
                transformer,
                low_memory=low_memory,
                latent_frames=latent_frames,
                latent_h=stage2_h,
                latent_w=stage2_w,
                # 日本語概要:
                # Stage 2 の query chunking はメモリ削減には効くが、video-only 化した
                # distilled 経路では速度低下のほうが目立ちやすい。
                # そのため自動 chunking は joint A/V Stage 2 に限定し、video-only は
                # low_memory 指定時だけ有効にする。
                prefer_memory=not stage2_video_only,
            )

            memory_profiler.start("stage 2 denoise")

            # Re-noise audio at sigma=0.909375 for joint refinement (matches PyTorch)
            if audio_latents is not None and not is_a2v and not stage2_video_only:
                audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
                audio_noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
                audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                    mx.array(1.0, dtype=model_dtype) - audio_noise_scale
                )
                mx.eval(audio_latents)

            # Joint video + audio refinement (no CFG, positive embeddings only)
            latents, refined_audio_latents = denoise_distilled(
                latents,
                positions,
                text_embeddings,
                transformer,
                stage2_sigmas,
                verbose=verbose,
                state=state2,
                audio_latents=None if stage2_video_only else audio_latents,
                audio_positions=None if stage2_video_only else audio_positions,
                audio_embeddings=None if stage2_video_only else audio_embeddings,
                audio_frozen=is_a2v,
            )
            if refined_audio_latents is not None:
                audio_latents = refined_audio_latents
            memory_profiler.log("stage 2 complete")

    elif pipeline == PipelineType.DEV:
        # ======================================================================
        # DEV PIPELINE: Single-stage with CFG
        # ======================================================================

        # Load VAE encoder for I2V
        image_latent = None
        if is_i2v:
            with console.status(
                "[blue]🖼️  Loading VAE encoder and encoding image...[/]", spinner="dots"
            ):
                vae_encoder = VideoEncoder.from_pretrained(
                    model_path / "vae" / "encoder"
                )

                input_image = load_image(
                    image, height=height, width=width, dtype=model_dtype
                )
                image_tensor = prepare_image_for_encoding(
                    input_image, height, width, dtype=model_dtype
                )
                image_latent = vae_encoder(image_tensor)
                mx.eval(image_latent)

                del vae_encoder
                mx.clear_cache()
            console.print("[green]✓[/] VAE encoder loaded and image encoded")

        # Generate sigma schedule with token-count-dependent shifting
        sigmas = ltx2_scheduler(steps=num_inference_steps)
        mx.eval(sigmas)
        console.print(
            f"[dim]Sigma schedule: {sigmas[0].item():.4f} → {sigmas[-2].item():.4f} → {sigmas[-1].item():.4f}[/]"
        )

        console.print(
            f"\n[bold yellow]⚡ Generating:[/] {width}x{height} ({num_inference_steps} steps, CFG={cfg_scale}, rescale={cfg_rescale})"
        )
        memory_profiler.start("dev denoise")
        mx.random.seed(seed)

        video_positions = create_position_grid(1, latent_frames, latent_h, latent_w)
        mx.eval(video_positions)

        # Always init audio latents/positions - PyTorch unconditionally generates audio
        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
        )
        mx.eval(audio_positions, audio_latents)

        # Initialize latents with optional I2V conditioning
        video_state = None
        video_latent_shape = (1, 128, latent_frames, latent_h, latent_w)
        if is_i2v and image_latent is not None:
            video_state = LatentState(
                latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditioning = VideoConditionByLatentIndex(
                latent=image_latent, frame_idx=image_frame_idx, strength=image_strength
            )
            video_state = apply_conditioning(video_state, [conditioning])

            noise = mx.random.normal(video_latent_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = video_state.denoise_mask * noise_scale
            video_state = LatentState(
                latent=noise * scaled_mask
                + video_state.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=video_state.clean_latent,
                denoise_mask=video_state.denoise_mask,
            )
            latents = video_state.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(video_latent_shape, dtype=model_dtype)
            mx.eval(latents)

        configure_attention_query_chunking(
            transformer,
            low_memory=low_memory,
            latent_frames=latent_frames,
            latent_h=latent_h,
            latent_w=latent_w,
        )

        # Always use A/V denoising - PyTorch always processes audio+video jointly
        latents, audio_latents = denoise_dev_av(
            latents,
            audio_latents,
            video_positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
            transformer,
            sigmas,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            cfg_rescale=cfg_rescale,
            verbose=verbose,
            video_state=video_state,
            use_apg=use_apg,
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
            stg_scale=stg_scale,
            stg_video_blocks=stg_blocks,
            stg_audio_blocks=stg_blocks,
            modality_scale=modality_scale,
            stg_skip_step=stg_skip_step,
            modality_skip_step=modality_skip_step,
            audio_frozen=is_a2v,
        )
        memory_profiler.log("dev complete")

        # Load VAE decoder (for dev pipeline, loaded here instead of during upsampling)
        vae_decoder = VideoDecoder.from_pretrained(str(model_path / "vae" / "decoder"))

    elif pipeline == PipelineType.DEV_TWO_STAGE:
        # ======================================================================
        # DEV TWO-STAGE PIPELINE:
        #   Stage 1: Dev denoising at half resolution with CFG
        #   Upsample: 2x spatial via LatentUpsampler
        #   Stage 2: Distilled denoising at full resolution with LoRA, no CFG
        # ======================================================================

        # Load VAE encoder for I2V
        reuse_conditioning_encoder = (
            is_i2v and not low_memory and not skip_stage2_refinement
        )
        conditioning_vae_encoder = None
        stage1_image_latent = None
        if is_i2v:
            with console.status(
                "[blue]🖼️  Encoding Stage 1 image conditioning...[/]", spinner="dots"
            ):
                s1_h, s1_w = stage1_h * 32, stage1_w * 32
                if reuse_conditioning_encoder and conditioning_vae_encoder is None:
                    conditioning_vae_encoder = VideoEncoder.from_pretrained(
                        model_path / "vae" / "encoder"
                    )
                stage1_image_latent = encode_conditioning_image_latent(
                    model_path=model_path,
                    image_path=image,
                    height=s1_h,
                    width=s1_w,
                    dtype=model_dtype,
                    vae_encoder=conditioning_vae_encoder,
                )
            console.print("[green]✓[/] Stage 1 image conditioning encoded")

        if use_split_dev_two_stage_loras and split_stage_custom_lora_path is not None:
            with console.status(
                f"[blue]🔧 Merging custom LoRA (stages 1+2, strength={split_stage_custom_lora_strength})...[/]",
                spinner="dots",
            ):
                load_and_merge_lora(
                    transformer,
                    split_stage_custom_lora_path,
                    strength=split_stage_custom_lora_strength,
                )

        if (
            use_split_dev_two_stage_loras
            and split_stage_distilled_lora_path is not None
            and (split_stage_distilled_lora_strength_s1 or 0.0) > 0.0
        ):
            with console.status(
                f"[blue]🔧 Merging distilled LoRA (stage 1, strength={split_stage_distilled_lora_strength_s1})...[/]",
                spinner="dots",
            ):
                load_and_merge_lora(
                    transformer,
                    split_stage_distilled_lora_path,
                    strength=split_stage_distilled_lora_strength_s1,
                )

        # Stage 1: Dev denoising at reduced resolution with CFG
        if dev_two_stage_sigma_preset == "official":
            sigmas = mx.array(STAGE_1_SIGMAS, dtype=mx.float32)
        else:
            sigmas = ltx2_scheduler(steps=num_inference_steps)
        mx.eval(sigmas)
        stage1_steps = len(sigmas) - 1
        console.print(
            f"[dim]Stage 1 sigma schedule: {sigmas[0].item():.4f} → {sigmas[-2].item():.4f} → {sigmas[-1].item():.4f}[/]"
        )

        console.print(
            f"\n[bold yellow]⚡ Stage 1:[/] Dev generating at {stage1_w*32}x{stage1_h*32} ({stage1_steps} steps, CFG={cfg_scale}, rescale={cfg_rescale})"
        )
        memory_profiler.start("stage 1 denoise")
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        # Always init audio latents/positions - PyTorch unconditionally generates audio
        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
        )
        mx.eval(audio_positions, audio_latents)

        # Apply I2V conditioning for stage 1
        state1 = None
        stage1_shape = (1, 128, latent_frames, stage1_h, stage1_w)
        if is_i2v and stage1_image_latent is not None:
            state1 = LatentState(
                latent=mx.zeros(stage1_shape, dtype=model_dtype),
                clean_latent=mx.zeros(stage1_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditioning = VideoConditionByLatentIndex(
                latent=stage1_image_latent,
                frame_idx=image_frame_idx,
                strength=image_strength,
            )
            state1 = apply_conditioning(state1, [conditioning])

            noise = mx.random.normal(stage1_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask
                + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(stage1_shape, dtype=model_dtype)
            mx.eval(latents)

        configure_attention_query_chunking(
            transformer,
            low_memory=low_memory,
            latent_frames=latent_frames,
            latent_h=stage1_h,
            latent_w=stage1_w,
        )

        # Stage 1: Always use joint AV denoising (matches PyTorch)
        latents, audio_latents = denoise_dev_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
            transformer,
            sigmas,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            cfg_rescale=cfg_rescale,
            verbose=verbose,
            video_state=state1,
            use_apg=use_apg,
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
            stg_scale=stg_scale,
            stg_video_blocks=stg_blocks,
            stg_audio_blocks=stg_blocks,
            modality_scale=modality_scale,
            stg_skip_step=stg_skip_step,
            modality_skip_step=modality_skip_step,
            audio_frozen=is_a2v,
        )

        if state1 is not None:
            del state1
        if stage1_image_latent is not None:
            del stage1_image_latent
        mx.clear_cache()
        memory_profiler.log("stage 1 complete")

        if skip_stage2_refinement:
            del transformer
            transformer = None
            mx.clear_cache()

        # Upsample latents
        memory_profiler.start("upsample")
        with console.status(
            f"[magenta]🔍 Upsampling latents {upscaler_scale}x...[/]", spinner="dots"
        ):
            if upscaler_path is None or not upscaler_path.exists():
                raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
            upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
            mx.eval(upsampler.parameters())

            vae_mean, vae_std = load_video_decoder_statistics(model_path)

            latents = upsample_latents(
                latents,
                upsampler,
                vae_mean,
                vae_std,
            )
            mx.eval(latents)

            del upsampler, vae_mean, vae_std
            mx.clear_cache()
        console.print("[green]✓[/] Latents upsampled")
        memory_profiler.log("upsample complete")

        if skip_stage2_refinement:
            console.print(
                "[dim]Stage 2 refinement skipped; decoding upsampled Stage 1 latents[/]"
            )

        # Merge LoRA weights for stage 2 distilled refinement.
        if not skip_stage2_refinement and use_split_dev_two_stage_loras:
            additional_distilled_strength = (
                (split_stage_distilled_lora_strength_s2 or 0.0)
                - (split_stage_distilled_lora_strength_s1 or 0.0)
            )
            if (
                split_stage_distilled_lora_path is not None
                and additional_distilled_strength > 0.0
            ):
                with console.status(
                    f"[blue]🔧 Adjusting distilled LoRA (stage 2, total={split_stage_distilled_lora_strength_s2})...[/]",
                    spinner="dots",
                ):
                    load_and_merge_lora(
                        transformer,
                        split_stage_distilled_lora_path,
                        strength=additional_distilled_strength,
                    )
        else:
            if not skip_stage2_refinement and lora_path is None:
                distilled_lora_file = find_distilled_lora_file(model_path)
                if distilled_lora_file is not None:
                    lora_path = str(distilled_lora_file)
                    console.print(f"[dim]Auto-detected LoRA: {distilled_lora_file.name}[/]")
                else:
                    console.print(
                        "[yellow]⚠️  No LoRA file found. Stage 2 will use base weights.[/]"
                    )

            if not skip_stage2_refinement and lora_path is not None:
                with console.status(
                    "[blue]🔧 Merging distilled LoRA weights...[/]", spinner="dots"
                ):
                    load_and_merge_lora(transformer, lora_path, strength=lora_strength)

        if not skip_stage2_refinement and low_memory_stage2_video_only:
            memory_profiler.start("stage 2 model trim")
            console.print(
                "[dim]Low-memory: stripping transformer audio/cross-modal modules before Stage 2[/]"
            )
            strip_transformer_to_video_only(transformer)
            memory_profiler.log("stage 2 model trimmed")

        # Stage 2: Distilled refinement at full resolution (no CFG)
        # Matches PyTorch: re-noise audio at sigma=0.909375, then jointly refine
        # both video and audio through the distilled schedule using the LoRA-merged model.
        if not skip_stage2_refinement:
            console.print(
                f"\n[bold yellow]⚡ Stage 2:[/] Distilled refining at {width}x{height} ({effective_stage2_steps} steps, no CFG)"
            )
            if low_memory_stage2_video_only:
                console.print(
                    "[dim]Low-memory: Stage 2 audio refinement disabled; reusing Stage 1 audio latents[/]"
                )
            positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
            mx.eval(positions)

            state2 = None
            if is_i2v:
                memory_profiler.start("stage 2 conditioning")
                with console.status(
                    "[blue]🖼️  Encoding Stage 2 image conditioning...[/]", spinner="dots"
                ):
                    s2_h, s2_w = stage2_h * 32, stage2_w * 32
                    stage2_image_latent = encode_conditioning_image_latent(
                        model_path=model_path,
                        image_path=image,
                        height=s2_h,
                        width=s2_w,
                        dtype=model_dtype,
                        vae_encoder=conditioning_vae_encoder,
                    )
                console.print("[green]✓[/] Stage 2 image conditioning encoded")

                state2 = LatentState(
                    latent=latents,
                    clean_latent=mx.zeros_like(latents),
                    denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
                )
                conditioning = VideoConditionByLatentIndex(
                    latent=stage2_image_latent,
                    frame_idx=image_frame_idx,
                    strength=image_strength,
                )
                state2 = apply_conditioning(state2, [conditioning])

                noise = mx.random.normal(latents.shape).astype(model_dtype)
                noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
                scaled_mask = state2.denoise_mask * noise_scale
                state2 = LatentState(
                    latent=noise * scaled_mask
                    + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                    clean_latent=state2.clean_latent,
                    denoise_mask=state2.denoise_mask,
                )
                latents = state2.latent
                mx.eval(latents)
                del stage2_image_latent
                memory_profiler.log("stage 2 conditioning ready")
                if conditioning_vae_encoder is not None:
                    del conditioning_vae_encoder
                    conditioning_vae_encoder = None
                    mx.clear_cache()
            else:
                noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
                one_minus_scale = mx.array(1.0 - stage2_sigmas[0], dtype=model_dtype)
                noise = mx.random.normal(latents.shape).astype(model_dtype)
                latents = noise * noise_scale + latents * one_minus_scale
                mx.eval(latents)

            configure_attention_query_chunking(
                transformer,
                low_memory=low_memory,
                latent_frames=latent_frames,
                latent_h=stage2_h,
                latent_w=stage2_w,
            )

            memory_profiler.start("stage 2 denoise")

            # Re-noise audio at sigma=0.909375 for joint refinement (matches PyTorch)
            if audio_latents is not None and not is_a2v and not low_memory_stage2_video_only:
                audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
                audio_noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
                audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                    mx.array(1.0, dtype=model_dtype) - audio_noise_scale
                )
                mx.eval(audio_latents)

            # Joint video + audio refinement (no CFG, positive embeddings only)
            latents, refined_audio_latents = denoise_distilled(
                latents,
                positions,
                text_embeddings,
                transformer,
                stage2_sigmas,
                verbose=verbose,
                state=state2,
                audio_latents=None if low_memory_stage2_video_only else audio_latents,
                audio_positions=None if low_memory_stage2_video_only else audio_positions,
                audio_embeddings=None if low_memory_stage2_video_only else audio_embeddings_pos,
                audio_frozen=is_a2v,
            )
            if refined_audio_latents is not None:
                audio_latents = refined_audio_latents
            memory_profiler.log("stage 2 complete")

    elif pipeline == PipelineType.DEV_TWO_STAGE_HQ:
        # ======================================================================
        # DEV TWO-STAGE HQ PIPELINE:
        #   Stage 1: res_2s denoising at half resolution with CFG + LoRA@0.25
        #   Upsample: 2x spatial via LatentUpsampler
        #   Stage 2: res_2s refinement at full resolution with LoRA@0.5, no CFG
        # ======================================================================

        # HQ defaults: STG disabled, lower rescale, fewer steps (PyTorch LTX_2_3_HQ_PARAMS)
        hq_lora_strength_s1 = (
            lora_strength_stage_1 if lora_strength_stage_1 is not None else 0.25
        )
        hq_lora_strength_s2 = (
            lora_strength_stage_2 if lora_strength_stage_2 is not None else 0.5
        )
        hq_cfg_rescale = (
            cfg_rescale if cfg_rescale != 0.7 else 0.45
        )  # Override default 0.7 → 0.45
        hq_steps = (
            num_inference_steps if num_inference_steps != 30 else 15
        )  # Override default 30 → 15
        hq_stg_scale = (
            stg_scale if stg_scale != 1.0 else 0.0
        )  # Override default 1.0 → 0.0

        # Load VAE encoder for I2V
        reuse_conditioning_encoder = (
            is_i2v and not low_memory and not skip_stage2_refinement
        )
        conditioning_vae_encoder = None
        stage1_image_latent = None
        if is_i2v:
            with console.status(
                "[blue]🖼️  Encoding Stage 1 image conditioning...[/]", spinner="dots"
            ):
                s1_h, s1_w = stage1_h * 32, stage1_w * 32
                if reuse_conditioning_encoder and conditioning_vae_encoder is None:
                    conditioning_vae_encoder = VideoEncoder.from_pretrained(
                        model_path / "vae" / "encoder"
                    )
                stage1_image_latent = encode_conditioning_image_latent(
                    model_path=model_path,
                    image_path=image,
                    height=s1_h,
                    width=s1_w,
                    dtype=model_dtype,
                    vae_encoder=conditioning_vae_encoder,
                )
            console.print("[green]✓[/] Stage 1 image conditioning encoded")

        # Auto-detect and merge LoRA for stage 1 (strength 0.25)
        if lora_path is None:
            lora_files = sorted(model_path.glob("*distilled-lora*.safetensors"))
            if lora_files:
                lora_path = str(lora_files[0])
                console.print(f"[dim]Auto-detected LoRA: {Path(lora_path).name}[/]")
            else:
                console.print(
                    "[yellow]Warning: No LoRA file found. HQ pipeline works best with distilled LoRA.[/]"
                )

        if lora_path is not None:
            with console.status(
                f"[blue]Merging distilled LoRA (stage 1, strength={hq_lora_strength_s1})...[/]",
                spinner="dots",
            ):
                load_and_merge_lora(
                    transformer, lora_path, strength=hq_lora_strength_s1
                )

        # Stage 1: res_2s denoising at reduced resolution with CFG
        # HQ passes actual token count to scheduler (unlike regular dev-two-stage)
        num_tokens = latent_frames * stage1_h * stage1_w
        sigmas = ltx2_scheduler(steps=hq_steps, num_tokens=num_tokens)
        mx.eval(sigmas)
        console.print(
            f"[dim]Stage 1 sigma schedule: {sigmas[0].item():.4f} -> {sigmas[-2].item():.4f} -> {sigmas[-1].item():.4f} (tokens={num_tokens})[/]"
        )

        console.print(
            f"\n[bold yellow]Stage 1:[/] res_2s at {stage1_w*32}x{stage1_h*32} ({hq_steps} steps, CFG={cfg_scale}, rescale={hq_cfg_rescale})"
        )
        memory_profiler.start("stage 1 denoise")
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
        )
        mx.eval(audio_positions, audio_latents)

        # Apply I2V conditioning for stage 1
        state1 = None
        stage1_shape = (1, 128, latent_frames, stage1_h, stage1_w)
        if is_i2v and stage1_image_latent is not None:
            state1 = LatentState(
                latent=mx.zeros(stage1_shape, dtype=model_dtype),
                clean_latent=mx.zeros(stage1_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditioning = VideoConditionByLatentIndex(
                latent=stage1_image_latent,
                frame_idx=image_frame_idx,
                strength=image_strength,
            )
            state1 = apply_conditioning(state1, [conditioning])

            noise = mx.random.normal(stage1_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask
                + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(stage1_shape, dtype=model_dtype)
            mx.eval(latents)

        configure_attention_query_chunking(
            transformer,
            low_memory=low_memory,
            latent_frames=latent_frames,
            latent_h=stage1_h,
            latent_w=stage1_w,
        )

        # Stage 1: res_2s with CFG (STG disabled for HQ by default)
        latents, audio_latents = denoise_res2s_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
            transformer,
            sigmas,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            cfg_rescale=hq_cfg_rescale,
            audio_cfg_rescale=1.0,
            verbose=verbose,
            video_state=state1,
            stg_scale=hq_stg_scale,
            stg_video_blocks=stg_blocks,
            stg_audio_blocks=stg_blocks,
            modality_scale=modality_scale,
            noise_seed=seed,
            audio_frozen=is_a2v,
        )

        if state1 is not None:
            del state1
        if stage1_image_latent is not None:
            del stage1_image_latent
        mx.clear_cache()
        memory_profiler.log("stage 1 complete")

        if skip_stage2_refinement:
            del transformer
            transformer = None
            mx.clear_cache()

        # Upsample latents
        memory_profiler.start("upsample")
        with console.status(
            f"[magenta]Upsampling latents {upscaler_scale}x...[/]", spinner="dots"
        ):
            if upscaler_path is None or not upscaler_path.exists():
                raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
            upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
            mx.eval(upsampler.parameters())

            vae_mean, vae_std = load_video_decoder_statistics(model_path)

            latents = upsample_latents(
                latents,
                upsampler,
                vae_mean,
                vae_std,
            )
            mx.eval(latents)

            del upsampler, vae_mean, vae_std
            mx.clear_cache()
        console.print("[green]✓[/] Latents upsampled")
        memory_profiler.log("upsample complete")

        if skip_stage2_refinement:
            console.print(
                "[dim]Stage 2 refinement skipped; decoding upsampled Stage 1 latents[/]"
            )

        # Merge additional LoRA for stage 2 (additive: 0.25 + 0.25 = 0.5 total)
        if not skip_stage2_refinement and lora_path is not None:
            additional_strength = hq_lora_strength_s2 - hq_lora_strength_s1
            if additional_strength > 0:
                with console.status(
                    f"[blue]Adjusting LoRA (stage 2, total={hq_lora_strength_s2})...[/]",
                    spinner="dots",
                ):
                    load_and_merge_lora(
                        transformer, lora_path, strength=additional_strength
                    )

        # Stage 2: res_2s refinement at full resolution (no CFG)
        if not skip_stage2_refinement:
            console.print(
                f"\n[bold yellow]Stage 2:[/] res_2s refining at {stage2_w*32}x{stage2_h*32} ({effective_stage2_steps} steps, no CFG)"
            )
            positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
            mx.eval(positions)

            state2 = None
            if is_i2v:
                memory_profiler.start("stage 2 conditioning")
                with console.status(
                    "[blue]🖼️  Encoding Stage 2 image conditioning...[/]", spinner="dots"
                ):
                    s2_h, s2_w = stage2_h * 32, stage2_w * 32
                    stage2_image_latent = encode_conditioning_image_latent(
                        model_path=model_path,
                        image_path=image,
                        height=s2_h,
                        width=s2_w,
                        dtype=model_dtype,
                        vae_encoder=conditioning_vae_encoder,
                    )
                console.print("[green]✓[/] Stage 2 image conditioning encoded")

                state2 = LatentState(
                    latent=latents,
                    clean_latent=mx.zeros_like(latents),
                    denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
                )
                conditioning = VideoConditionByLatentIndex(
                    latent=stage2_image_latent,
                    frame_idx=image_frame_idx,
                    strength=image_strength,
                )
                state2 = apply_conditioning(state2, [conditioning])

                noise = mx.random.normal(latents.shape).astype(model_dtype)
                noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
                scaled_mask = state2.denoise_mask * noise_scale
                state2 = LatentState(
                    latent=noise * scaled_mask
                    + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                    clean_latent=state2.clean_latent,
                    denoise_mask=state2.denoise_mask,
                )
                latents = state2.latent
                mx.eval(latents)
                del stage2_image_latent
                memory_profiler.log("stage 2 conditioning ready")
                if conditioning_vae_encoder is not None:
                    del conditioning_vae_encoder
                    conditioning_vae_encoder = None
                    mx.clear_cache()
            else:
                noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
                one_minus_scale = mx.array(1.0 - stage2_sigmas[0], dtype=model_dtype)
                noise = mx.random.normal(latents.shape).astype(model_dtype)
                latents = noise * noise_scale + latents * one_minus_scale
                mx.eval(latents)

            configure_attention_query_chunking(
                transformer,
                low_memory=low_memory,
                latent_frames=latent_frames,
                latent_h=stage2_h,
                latent_w=stage2_w,
            )

            memory_profiler.start("stage 2 denoise")

            # Re-noise audio at sigma=0.909375 for joint refinement
            if audio_latents is not None and not is_a2v:
                audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
                audio_noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
                audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                    mx.array(1.0, dtype=model_dtype) - audio_noise_scale
                )
                mx.eval(audio_latents)

            # Stage 2: res_2s with no CFG (positive embeddings only)
            latents, audio_latents = denoise_res2s_av(
                latents,
                audio_latents,
                positions,
                audio_positions,
                video_embeddings_pos,
                video_embeddings_pos,  # both pos (no neg for stage 2)
                audio_embeddings_pos,
                audio_embeddings_pos,
                transformer,
                mx.array(stage2_sigmas, dtype=mx.float32),
                cfg_scale=1.0,  # no CFG
                audio_cfg_scale=1.0,
                cfg_rescale=0.0,
                verbose=verbose,
                video_state=state2,
                noise_seed=seed + 1,
                audio_frozen=is_a2v,
            )
            memory_profiler.log("stage 2 complete")

    if transformer is not None:
        del transformer
    mx.clear_cache()

    if pipeline != PipelineType.DEV:
        vae_decoder = VideoDecoder.from_pretrained(str(model_path / "vae" / "decoder"))

    # ==========================================================================
    # Decode and save outputs (common to both pipelines)
    # ==========================================================================

    console.print("\n[blue]🎞️  Decoding video...[/]")
    memory_profiler.start("video decode")

    decode_tiling_mode = resolve_decode_tiling_mode(
        tiling,
        low_memory=low_memory,
        height=height,
        width=width,
        num_frames=num_frames,
    )
    if decode_tiling_mode != tiling:
        console.print(
            f"[dim]Low-memory decode tiling escalated: {tiling} → {decode_tiling_mode}[/]"
        )

    # Select tiling configuration
    if decode_tiling_mode == "none":
        tiling_config = None
    elif decode_tiling_mode == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif decode_tiling_mode == "default":
        tiling_config = TilingConfig.default()
    elif decode_tiling_mode == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif decode_tiling_mode == "conservative":
        tiling_config = TilingConfig.conservative()
    elif decode_tiling_mode == "spatial":
        spatial_tile_size = 768 if low_memory and num_frames <= 241 else 512
        tiling_config = TilingConfig.spatial_only(tile_size=spatial_tile_size)
    elif decode_tiling_mode == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        console.print(
            f"[yellow]  Unknown tiling mode '{decode_tiling_mode}', using auto[/]"
        )
        tiling_config = TilingConfig.auto(height, width, num_frames)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream mode / no-return tiled flush mode
    video_writer = None
    stream_progress = None
    captured_frame_chunks = None
    stream_stats = None
    video_decode_report = None

    stream_tiled_output = tiling_config is not None and (
        stream or (not return_video and not save_frames)
    )

    if stream_tiled_output and not stream:
        console.print(
            "[dim]  Streaming tiled decode to disk because return_video=False[/]"
        )

    if (
        stream_tiled_output
        and decode_tiling_mode == "spatial"
        and tiling_config is not None
        and tiling_config.temporal_config is None
        and num_frames > 65
    ):
        # Keep the larger temporal window for as long as possible. Smaller
        # temporal tiles reduce memory a bit more, but they also make motion
        # noticeably softer because each tile has less temporal context and
        # more overlap blending. Only drop to 32f for very long clips.
        temporal_tile_size = 32 if num_frames > 241 else 64
        temporal_overlap = 8 if temporal_tile_size == 32 else 24
        tiling_config = TilingConfig(
            spatial_config=tiling_config.spatial_config,
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=temporal_tile_size,
                tile_overlap_in_frames=temporal_overlap,
            ),
        )

    if audio:
        temp_video_path = output_path.with_suffix(".temp.mp4")
        save_path = temp_video_path
    else:
        save_path = output_path

    if stream_tiled_output:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        stream_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        stream_progress.start()
        stream_task = stream_progress.add_task(
            "[cyan]Streaming frames[/]", total=num_frames
        )
        stream_stats = {
            "callbacks": 0,
            "frames": 0,
            "max_chunk": 0,
            "min_chunk": None,
            "first_start": None,
            "last_start": None,
        }
        if return_video or save_frames:
            captured_frame_chunks = []

        def on_frames_ready(frames: mx.array, start_idx: int):
            frames = mx.squeeze(frames, axis=0)
            frames = mx.transpose(frames, (1, 2, 3, 0))
            frames = mx.clip((frames + 1.0) / 2.0, 0.0, 1.0)
            frames = (frames * 255).astype(mx.uint8)
            frames_np = np.array(frames)
            chunk_frames = int(frames_np.shape[0])

            stream_stats["callbacks"] += 1
            stream_stats["frames"] += chunk_frames
            stream_stats["max_chunk"] = max(stream_stats["max_chunk"], chunk_frames)
            stream_stats["min_chunk"] = (
                chunk_frames
                if stream_stats["min_chunk"] is None
                else min(stream_stats["min_chunk"], chunk_frames)
            )
            if stream_stats["first_start"] is None:
                stream_stats["first_start"] = start_idx
            stream_stats["last_start"] = start_idx

            if memory_profiler.enabled:
                active = mx.get_active_memory()
                cache = mx.get_cache_memory()
                peak = mx.get_peak_memory()
                console.print(
                    "[dim]    Stream callback #{idx}: start={start}, frames={frames}, "
                    "active={active:.2f}GB, cache={cache:.2f}GB, peak={peak:.2f}GB[/]".format(
                        idx=stream_stats["callbacks"],
                        start=start_idx,
                        frames=chunk_frames,
                        active=memory_profiler._gib(active),
                        cache=memory_profiler._gib(cache),
                        peak=memory_profiler._gib(peak),
                    )
                )

            if captured_frame_chunks is not None:
                captured_frame_chunks.append(frames_np)

            for frame in frames_np:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                stream_progress.advance(stream_task)

    else:
        on_frames_ready = None

    if tiling_config is not None:
        spatial_info = (
            f"{tiling_config.spatial_config.tile_size_in_pixels}px"
            if tiling_config.spatial_config
            else "none"
        )
        temporal_info = (
            f"{tiling_config.temporal_config.tile_size_in_frames}f"
            if tiling_config.temporal_config
            else "none"
        )
        console.print(
            f"[dim]  Tiling ({decode_tiling_mode}): spatial={spatial_info}, temporal={temporal_info}[/]"
        )
        video = vae_decoder.decode_tiled(
            latents,
            tiling_config=tiling_config,
            tiling_mode=decode_tiling_mode,
            debug=verbose,
            on_frames_ready=on_frames_ready,
            return_output=not stream_tiled_output,
        )
    else:
        console.print("[dim]  Tiling: disabled[/]")
        video = vae_decoder(latents)
    if video is not None:
        mx.eval(video)
    mx.clear_cache()
    video_decode_report = memory_profiler.capture("video decoded")

    # Close stream writer
    if video_writer is not None:
        if video is not None:
            video = mx.squeeze(video, axis=0)
            video = mx.transpose(video, (1, 2, 3, 0))
            video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
            video = (video * 255).astype(mx.uint8)

            if captured_frame_chunks is not None:
                returned_frames_np = np.array(video)
                captured_frame_chunks.append(returned_frames_np)
                frames_iter = returned_frames_np
            else:
                frames_iter = None

            if frames_iter is None:
                for frame_idx in range(video.shape[0]):
                    frame_np = np.array(video[frame_idx])
                    video_writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                    stream_progress.advance(stream_task)
            else:
                for frame in frames_iter:
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    stream_progress.advance(stream_task)

        video_writer.release()
        if stream_progress is not None:
            stream_progress.stop()
        console.print(f"[green]✅ Streamed video to[/] {save_path}")
        if stream_stats is not None and stream_stats["callbacks"] > 0:
            avg_chunk = stream_stats["frames"] / stream_stats["callbacks"]
            console.print(
                "[dim]  Stream stats: callbacks={callbacks}, frames={frames}, "
                "chunk=min {min_chunk} / avg {avg_chunk:.1f} / max {max_chunk}, "
                "start range={first_start}→{last_start}[/]".format(
                    callbacks=stream_stats["callbacks"],
                    frames=stream_stats["frames"],
                    min_chunk=stream_stats["min_chunk"],
                    avg_chunk=avg_chunk,
                    max_chunk=stream_stats["max_chunk"],
                    first_start=stream_stats["first_start"],
                    last_start=stream_stats["last_start"],
                )
            )
        if video_decode_report is not None:
            console.print(video_decode_report)
        video_np = (
            np.concatenate(captured_frame_chunks, axis=0)
            if captured_frame_chunks is not None and captured_frame_chunks
            else None
        )
    else:
        video = mx.squeeze(video, axis=0)
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255).astype(mx.uint8)
        video_np = np.array(video) if (return_video or save_frames) else None

        try:
            import cv2

            h, w = int(video.shape[1]), int(video.shape[2])
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
            if video_np is not None:
                for frame in video_np:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                for frame_idx in range(video.shape[0]):
                    frame_np = np.array(video[frame_idx])
                    out.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
            out.release()
            if not audio:
                console.print(f"[green]✅ Saved video to[/] {output_path}")
        except Exception as e:
            console.print(f"[red]❌ Could not save video: {e}[/]")
        if video_decode_report is not None:
            console.print(video_decode_report)

    # Decode and save audio if enabled
    audio_np = None
    vocoder_sample_rate = AUDIO_SAMPLE_RATE
    if audio and audio_latents is not None:
        if is_a2v and a2v_waveform is not None:
            # A2V: use original input audio waveform (no VAE decoding needed)
            audio_np = a2v_waveform
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
            vocoder_sample_rate = a2v_sr or AUDIO_LATENT_SAMPLE_RATE
            console.print("[green]✓[/] Using original input audio (A2V)")
        else:
            memory_profiler.start("audio decode")
            with console.status("[blue]Decoding audio...[/]", spinner="dots"):
                audio_decoder = load_audio_decoder(model_path, pipeline)
                vocoder = load_vocoder_model(model_path, pipeline)
                mx.eval(audio_decoder.parameters(), vocoder.parameters())

                mel_spectrogram = audio_decoder(audio_latents)
                mx.eval(mel_spectrogram)
                console.print(
                    f"[dim]  Mel spectrogram: shape={mel_spectrogram.shape}, std={mel_spectrogram.std().item():.4f}, mean={mel_spectrogram.mean().item():.4f}[/]"
                )

                audio_waveform = vocoder(mel_spectrogram)
                mx.eval(audio_waveform)

                audio_np = np.array(audio_waveform.astype(mx.float32))
                if audio_np.ndim == 3:
                    audio_np = audio_np[0]

                # Get sample rate from vocoder (dynamic: 24kHz for LTX-2, 48kHz for LTX-2.3 BWE)
                vocoder_sample_rate = getattr(
                    vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE
                )

                del audio_decoder, vocoder
                mx.clear_cache()
            console.print("[green]✓[/] Audio decoded")
            memory_profiler.log("audio decoded")

        audio_path = (
            Path(output_audio_path)
            if output_audio_path
            else output_path.with_suffix(".wav")
        )
        save_audio(audio_np, audio_path, vocoder_sample_rate)
        console.print(f"[green]✅ Saved audio to[/] {audio_path}")

        with console.status("[blue]🎬 Combining video and audio...[/]", spinner="dots"):
            temp_video_path = output_path.with_suffix(".temp.mp4")
            success = mux_video_audio(
                temp_video_path,
                audio_path,
                output_path,
                audio_bitrate=audio_bitrate,
            )
        if success:
            console.print(f"[green]✅ Saved video with audio to[/] {output_path}")
            temp_video_path.unlink()
        else:
            temp_video_path.rename(output_path)
            console.print(f"[yellow]⚠️  Saved video without audio to[/] {output_path}")

    del vae_decoder
    mx.clear_cache()

    if save_frames:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(video_np):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        console.print(f"[green]✅ Saved {len(video_np)} frames to {frames_dir}[/]")

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    time_str = f"{int(minutes)}m {seconds:.1f}s" if minutes >= 1 else f"{seconds:.1f}s"
    console.print(
        Panel(
            f"[bold green]🎉 Done![/] Generated in {time_str} ({elapsed/num_frames:.2f}s/frame)\n"
            f"[bold green]✨ Peak memory:[/] {memory_profiler.final_peak() / (1024 ** 3):.2f}GB",
            expand=False,
        )
    )

    if audio:
        return video_np, audio_np
    return video_np


def main():
    parser = argparse.ArgumentParser(
                description="Generate LTX-2 / LTX-2.3 videos with optional I2V, A2V, and synchronized audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distilled pipeline (two-stage, fast, no CFG)
    uv run mlx_video.ltx_2.generate --prompt "A cat walking on grass"
    uv run mlx_video.ltx_2.generate --prompt "Ocean waves" --pipeline distilled

  # Dev pipeline (single-stage, CFG, higher quality)
    uv run mlx_video.ltx_2.generate --prompt "A cat walking" --pipeline dev --cfg-scale 3.0
    uv run mlx_video.ltx_2.generate --prompt "Ocean waves" --pipeline dev --steps 40

  # Dev two-stage pipeline (dev + LoRA refinement)
    uv run mlx_video.ltx_2.generate --prompt "A cat walking" --pipeline dev-two-stage --cfg-scale 3.0

  # Image-to-Video (works with both pipelines)
    uv run mlx_video.ltx_2.generate --prompt "A person dancing" --image photo.jpg
    uv run mlx_video.ltx_2.generate --prompt "Waves crashing" --image beach.png --pipeline dev

    # Image-to-Video + synchronized audio generation
    uv run mlx_video.ltx_2.generate --prompt "A singer on stage" --image singer.png --audio

  # With Audio (works with both pipelines)
    uv run mlx_video.ltx_2.generate --prompt "Ocean waves crashing" --audio
    uv run mlx_video.ltx_2.generate --prompt "A jazz band playing" --audio --pipeline dev
        """,
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text description of the video to generate",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="distilled",
        choices=["distilled", "dev", "dev-two-stage", "dev-two-stage-hq"],
        help="Pipeline type: distilled (fast), dev (CFG), dev-two-stage (dev + LoRA), dev-two-stage-hq (res_2s + LoRA both stages)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for CFG (dev pipeline only)",
    )
    parser.add_argument(
        "--height", "-H", type=int, default=512, help="Output video height"
    )
    parser.add_argument(
        "--width", "-W", type=int, default=512, help="Output video width"
    )
    parser.add_argument(
        "--num-frames", "-n", type=int, default=33, help="Number of frames"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (dev pipeline only, default 30)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="CFG guidance scale for video (dev pipeline only, default 3.0)",
    )
    parser.add_argument(
        "--audio-cfg-scale",
        type=float,
        default=7.0,
        help="CFG guidance scale for audio (default 7.0, PyTorch default)",
    )
    parser.add_argument(
        "--cfg-rescale",
        type=float,
        default=0.7,
        help="CFG rescale factor (0.0-1.0). Normalizes guided prediction variance to reduce artifacts (dev pipeline only, default 0.7)",
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument(
        "--output-path", "-o", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--save-frames", action="store_true", help="Save individual frames as images"
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=DEFAULT_DISTILLED_MODEL_REPO,
        help="Model repository",
    )
    parser.add_argument(
        "--text-encoder-repo",
        type=str,
        default=None,
        help="Text encoder repository (auto-detects google/gemma-3-12b-it if needed)",
    )
    parser.add_argument(
        "--transformer-quantization-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help="Runtime-quantize the LTX transformer to 4-bit or 8-bit before inference",
    )
    parser.add_argument(
        "--transformer-quantization-mode",
        type=str,
        default="affine",
        choices=list(TRANSFORMER_QUANTIZATION_MODES),
        help='Runtime transformer quantization mode: "affine" (recommended) or experimental "mxfp8"',
    )
    parser.add_argument(
        "--transformer-quantization-group-size",
        type=int,
        default=None,
        help="Group size for runtime transformer quantization (defaults to 64 for affine, 32 for mxfp8)",
    )
    parser.add_argument(
        "--transformer-quantize-inputs",
        action="store_true",
        help="Experimental: quantize transformer activations on the fly (mxfp8 only)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--enhance-prompt", action="store_true", help="Enhance the prompt using Gemma"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens for prompt enhancement"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for prompt enhancement",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to conditioning image for I2V / I2V+Audio",
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for I2V",
    )
    parser.add_argument(
        "--image-frame-idx",
        type=int,
        default=0,
        help="Frame index to condition for I2V",
    )
    parser.add_argument(
        "--tiling",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "default",
            "aggressive",
            "conservative",
            "spatial",
            "temporal",
        ],
        help="Tiling mode for VAE decoding",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream frames to output as they're decoded",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Apply safer low-memory settings (conservative tiling, reduced extra guidance passes)",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Print phase-local MLX memory measurements for profiling/debugging",
    )
    parser.add_argument(
        "--skip-stage2-refinement",
        action="store_true",
        help="Experimental: decode the upsampled Stage 1 result without running Stage 2 refinement",
    )
    parser.add_argument(
        "--stage2-refinement-steps",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Experimental: override Stage 2 refinement step count for two-stage pipelines (0-3)",
    )
    parser.add_argument(
        "--disable-stage2-audio-refinement",
        action="store_true",
        help="Experimental: run distilled Stage 2 in video-only mode and reuse Stage 1 audio latents",
    )
    parser.add_argument(
        "--preserve-stage2-audio-refinement",
        action="store_true",
        help="Keep Stage 2 audio refinement enabled even in --low-memory mode (higher peak memory)",
    )
    parser.add_argument(
        "--audio",
        "-a",
        action="store_true",
        help="Enable synchronized audio generation (can be combined with --image)",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to audio file for A2V (audio-to-video) conditioning; mutually exclusive with --audio",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds for audio file (default: 0.0)",
    )
    parser.add_argument(
        "--output-audio", type=str, default=None, help="Output audio path"
    )
    parser.add_argument(
        "--audio-bitrate",
        type=str,
        default="320k",
        help="AAC bitrate for muxed MP4 audio (default: 320k)",
    )
    parser.add_argument(
        "--apg",
        action="store_true",
        help="Use Adaptive Projected Guidance instead of CFG (more stable for I2V)",
    )
    parser.add_argument(
        "--apg-eta",
        type=float,
        default=1.0,
        help="APG parallel component weight (1.0 = keep full parallel)",
    )
    parser.add_argument(
        "--apg-norm-threshold",
        type=float,
        default=0.0,
        help="APG guidance norm clamp (0 = no clamping)",
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        default=1.0,
        help="STG (Spatiotemporal Guidance) scale (default 1.0, 0.0 = disabled)",
    )
    parser.add_argument(
        "--stg-blocks",
        type=int,
        nargs="+",
        default=None,
        help="Transformer block indices for STG perturbation (default: [29] for LTX-2, [28] for LTX-2.3)",
    )
    parser.add_argument(
        "--stg-skip-step",
        type=int,
        default=1,
        help="Run STG guidance every Nth step (default 1 = every step)",
    )
    parser.add_argument(
        "--modality-scale",
        type=float,
        default=3.0,
        help="Cross-modal guidance scale (default 3.0, 1.0 = disabled)",
    )
    parser.add_argument(
        "--modality-skip-step",
        type=int,
        default=1,
        help="Run cross-modal guidance every Nth step (default 1 = every step)",
    )
    parser.add_argument(
        "--dev-two-stage-sigma-preset",
        type=str,
        default="default",
        choices=["default", "official"],
        help='Sigma schedule preset for dev-two-stage: "default" keeps the current dynamic scheduler, "official" matches the official LTX-2.3 two-stage distilled workflow manual sigmas.',
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA safetensors file. Legacy dev-two-stage behavior treats this as the distilled Stage 2 LoRA; when using --distilled-lora-* controls it is reused as the custom/user LoRA path.",
    )
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=1.0,
        help="LoRA merge strength (distilled / dev-two-stage pipeline, default 1.0)",
    )
    parser.add_argument(
        "--lora-strength-stage-1",
        type=float,
        default=0.25,
        help="LoRA strength for HQ stage 1 (default 0.25)",
    )
    parser.add_argument(
        "--lora-strength-stage-2",
        type=float,
        default=0.5,
        help="LoRA strength for HQ stage 2 (default 0.5)",
    )
    parser.add_argument(
        "--custom-lora-path",
        type=str,
        default=None,
        help="Custom/user LoRA path for dev-two-stage split-LoRA mode",
    )
    parser.add_argument(
        "--custom-lora-strength",
        type=float,
        default=1.0,
        help="Custom/user LoRA strength for dev-two-stage split-LoRA mode (default 1.0)",
    )
    parser.add_argument(
        "--distilled-lora-path",
        type=str,
        default=None,
        help="Distilled LoRA path for dev-two-stage split-LoRA mode. Auto-detected from --model-repo when omitted and stage strengths are provided.",
    )
    parser.add_argument(
        "--distilled-lora-strength-stage-1",
        type=float,
        default=None,
        help="Distilled LoRA Stage 1 strength for dev-two-stage split-LoRA mode",
    )
    parser.add_argument(
        "--distilled-lora-strength-stage-2",
        type=float,
        default=None,
        help="Distilled LoRA Stage 2 strength for dev-two-stage split-LoRA mode",
    )
    parser.add_argument(
        "--spatial-upscaler",
        type=str,
        default=None,
        help="Spatial upscaler filename (e.g. ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors). Auto-detects the latest x2 variant by default (prefers v1.1 over v1.0).",
    )
    args = parser.parse_args()
    try:
        resolve_transformer_quantization(
            bits=args.transformer_quantization_bits,
            group_size=args.transformer_quantization_group_size,
            mode=args.transformer_quantization_mode,
            quantize_input=args.transformer_quantize_inputs,
        )
    except ValueError as exc:
        parser.error(str(exc))

    pipeline_map = {
        "distilled": PipelineType.DISTILLED,
        "dev": PipelineType.DEV,
        "dev-two-stage": PipelineType.DEV_TWO_STAGE,
        "dev-two-stage-hq": PipelineType.DEV_TWO_STAGE_HQ,
    }
    pipeline = pipeline_map[args.pipeline]

    generate_video(
        model_repo=args.model_repo,
        text_encoder_repo=args.text_encoder_repo,
        prompt=args.prompt,
        pipeline=pipeline,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        audio_cfg_scale=args.audio_cfg_scale,
        cfg_rescale=args.cfg_rescale,
        seed=args.seed,
        fps=args.fps,
        output_path=args.output_path,
        save_frames=args.save_frames,
        verbose=args.verbose,
        enhance_prompt=args.enhance_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image=args.image,
        image_strength=args.image_strength,
        image_frame_idx=args.image_frame_idx,
        tiling=args.tiling,
        stream=args.stream,
        audio=args.audio,
        output_audio_path=args.output_audio,
        use_apg=args.apg,
        apg_eta=args.apg_eta,
        apg_norm_threshold=args.apg_norm_threshold,
        stg_scale=args.stg_scale,
        stg_blocks=args.stg_blocks,
        stg_skip_step=args.stg_skip_step,
        modality_scale=args.modality_scale,
        modality_skip_step=args.modality_skip_step,
        lora_path=args.lora_path,
        lora_strength=args.lora_strength,
        lora_strength_stage_1=args.lora_strength_stage_1,
        lora_strength_stage_2=args.lora_strength_stage_2,
        custom_lora_path=args.custom_lora_path,
        custom_lora_strength=args.custom_lora_strength,
        distilled_lora_path=args.distilled_lora_path,
        distilled_lora_strength_stage_1=args.distilled_lora_strength_stage_1,
        distilled_lora_strength_stage_2=args.distilled_lora_strength_stage_2,
        audio_file=args.audio_file,
        audio_start_time=args.audio_start_time,
        spatial_upscaler=args.spatial_upscaler,
        low_memory=args.low_memory,
        profile_memory=args.profile_memory,
        skip_stage2_refinement=args.skip_stage2_refinement,
        stage2_refinement_steps=args.stage2_refinement_steps,
        disable_stage2_audio_refinement=args.disable_stage2_audio_refinement,
        preserve_stage2_audio_refinement=args.preserve_stage2_audio_refinement,
        transformer_quantization_bits=args.transformer_quantization_bits,
        transformer_quantization_group_size=args.transformer_quantization_group_size,
        transformer_quantization_mode=args.transformer_quantization_mode,
        transformer_quantize_inputs=args.transformer_quantize_inputs,
        dev_two_stage_sigma_preset=args.dev_two_stage_sigma_preset,
        return_video=args.save_frames,
        audio_bitrate=args.audio_bitrate,
    )


if __name__ == "__main__":
    main()
