import json
import math
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

from mlx_video.quantization import quantize_modules


DEFAULT_MODEL_ALLOW_PATTERNS = ["*.safetensors", "*.json"]
_QUANTIZATION_MODE_DEFAULTS = {
    "affine": {"group_size": 64},
    "mxfp4": {"group_size": 32, "bits": 4},
    "mxfp8": {"group_size": 32, "bits": 8},
}


def _path_has_required_files(path: Path, required_patterns: list[str]) -> bool:
    """Check whether a downloaded model path contains all required file patterns."""
    return all(any(path.rglob(pattern)) for pattern in required_patterns)


def resolve_safetensor_files(model_path: Union[str, Path]) -> List[Path]:
    """Resolve the canonical list of safetensors files for a model directory.

    If a `*.safetensors.index.json` file is present, only the shard files listed in
    its `weight_map` are returned. This avoids accidentally loading stray or stale
    shard files that may coexist in the same directory.
    """
    model_path = Path(model_path)

    index_files = sorted(model_path.glob("*.safetensors.index.json"))
    if index_files:
        with open(index_files[0], "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})

        shard_names = []
        seen = set()
        for shard_name in weight_map.values():
            if shard_name not in seen:
                seen.add(shard_name)
                shard_names.append(shard_name)

        shard_paths = [model_path / shard_name for shard_name in shard_names]
        missing = [path for path in shard_paths if not path.exists()]
        if missing:
            missing_str = ", ".join(path.name for path in missing)
            raise FileNotFoundError(
                f"Missing safetensor shard(s) referenced by index: {missing_str}"
            )

        return shard_paths

    return sorted(model_path.glob("*.safetensors"))


def get_model_path(
    model_repo: str,
    allow_patterns: Optional[list[str]] = None,
    required_patterns: Optional[list[str]] = None,
):
    """Get or download LTX-2 model path."""
    allow_patterns = allow_patterns or DEFAULT_MODEL_ALLOW_PATTERNS
    required_patterns = required_patterns or ["*.safetensors"]

    try:
        if Path(model_repo).exists():
            return Path(model_repo)

        path = Path(
            snapshot_download(
                repo_id=model_repo,
                local_files_only=True,
                allow_patterns=allow_patterns,
            )
        )

        if _path_has_required_files(path, required_patterns):
            return path

        print("Model cache is incomplete. Downloading missing files...")
    except Exception:
        print("Downloading model weights...")

    path = Path(
        snapshot_download(
            repo_id=model_repo,
            local_files_only=False,
            resume_download=True,
            allow_patterns=allow_patterns,
        )
    )

    if not _path_has_required_files(path, required_patterns):
        required_str = ", ".join(required_patterns)
        raise FileNotFoundError(
            f"Downloaded model cache is still incomplete for {model_repo!r}. "
            f"Required pattern(s): {required_str}"
        )

    return path


def normalize_quantization_config(quantization: Optional[dict]) -> Optional[dict]:
    if quantization is None:
        return None

    normalized = dict(quantization)
    mode = normalized.get("mode", "affine")
    normalized["mode"] = mode

    if mode in _QUANTIZATION_MODE_DEFAULTS:
        defaults = _QUANTIZATION_MODE_DEFAULTS[mode]
        if "group_size" in defaults and normalized.get("group_size") is None:
            normalized["group_size"] = defaults["group_size"]
        if "bits" in defaults and normalized.get("bits") is None:
            normalized["bits"] = defaults["bits"]
    elif "group_size" not in normalized or "bits" not in normalized:
        raise ValueError(
            "Quantization config must define bits and group_size for non-default modes."
        )

    bits = normalized.get("bits")
    group_size = normalized.get("group_size")

    if bits is None:
        raise ValueError("Quantization config must define bits.")
    if group_size is None:
        raise ValueError("Quantization config must define group_size.")

    if mode == "mxfp4":
        if bits != 4:
            raise ValueError("MXFP4 quantization requires bits=4.")
        if group_size != 32:
            raise ValueError("MXFP4 quantization requires group_size=32.")
    elif mode == "mxfp8":
        if bits != 8:
            raise ValueError("MXFP8 quantization requires bits=8.")
        if group_size != 32:
            raise ValueError("MXFP8 quantization requires group_size=32.")

    if normalized.get("quantize_input") and mode not in ("mxfp8", "nvfp4"):
        raise ValueError(
            "Activation quantization is only supported for mxfp8 and nvfp4."
        )

    return normalized


def supports_quantized_weight_shape(module: nn.Module, group_size: int) -> bool:
    if not hasattr(module, "weight"):
        return True
    return module.weight.shape[-1] % group_size == 0


def apply_quantization(model: nn.Module, weights: mx.array, quantization: dict):
    if quantization is not None:
        quantization = normalize_quantization_config(quantization)
        force_quantization = quantization.get("force", False)
        predicate_override = quantization.get("class_predicate")

        def get_class_predicate(p, m):
            if predicate_override is not None:
                return predicate_override(p, m)
            # Handle custom per layer quantizations
            if p in quantization:
                return quantization[p]
            if not hasattr(m, "to_quantized"):
                return False
            if not supports_quantized_weight_shape(m, quantization["group_size"]):
                return False
            # 保存済みの量子化重みがない通常モデルでも、runtime 指定時はここで新規量子化する。
            if force_quantization:
                return True
            # Handle legacy models which may not have everything quantized
            return f"{p}.scales" in weights

        quantize_modules(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            quantize_input=quantization.get("quantize_input", False),
            class_predicate=get_class_predicate,
        )


@partial(mx.compile, shapeless=True)
def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, mx.ones((x.shape[-1],), dtype=x.dtype), eps)


@partial(mx.compile, shapeless=True)
def to_denoised(
    noisy: mx.array, velocity: mx.array, sigma: mx.array | float
) -> mx.array:
    """Convert velocity prediction to denoised output.

    Given noisy input x_t and velocity prediction v, compute denoised x_0:
    x_0 = x_t - sigma * v

    Uses float32 for computation precision (matching PyTorch behavior),
    then converts back to input dtype.

    Args:
        noisy: Noisy input tensor x_t
        velocity: Velocity prediction v
        sigma: Noise level (scalar or per-sample)

    Returns:
        Denoised tensor x_0
    """
    original_dtype = noisy.dtype

    # Cast to float32 for precision (PyTorch uses calc_dtype=torch.float32)
    noisy_f32 = noisy.astype(mx.float32)
    velocity_f32 = velocity.astype(mx.float32)

    if isinstance(sigma, (int, float)):
        sigma_f32 = mx.array(sigma, dtype=mx.float32)
    else:
        sigma_f32 = sigma.astype(mx.float32)
        while sigma_f32.ndim < velocity_f32.ndim:
            sigma_f32 = mx.expand_dims(sigma_f32, axis=-1)

    result = noisy_f32 - sigma_f32 * velocity_f32
    return result.astype(original_dtype)


def repeat_interleave(x: mx.array, repeats: int, axis: int = -1) -> mx.array:
    """Repeat elements of tensor along an axis, similar to torch.repeat_interleave.

    Args:
        x: Input tensor
        repeats: Number of repetitions for each element
        axis: The axis along which to repeat values

    Returns:
        Tensor with repeated values
    """
    # Handle negative axis
    if axis < 0:
        axis = x.ndim + axis

    # Get shape
    shape = list(x.shape)

    # Expand dims, repeat, then reshape
    x = mx.expand_dims(x, axis=axis + 1)

    # Create tile pattern
    tile_pattern = [1] * x.ndim
    tile_pattern[axis + 1] = repeats

    x = mx.tile(x, tile_pattern)

    # Reshape to merge the repeated dimension
    new_shape = shape.copy()
    new_shape[axis] *= repeats

    return mx.reshape(x, new_shape)


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x / mx.sqrt(mx.mean(x * x, axis=1, keepdims=True) + self.eps)


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1D tensor of timesteps
        embedding_dim: Dimension of the embeddings to create
        flip_sin_to_cos: If True, flip sin and cos ordering
        downscale_freq_shift: Frequency shift factor
        scale: Scale factor for timesteps
        max_period: Maximum period for the sinusoids

    Returns:
        Tensor of shape (len(timesteps), embedding_dim)
    """
    assert timesteps.ndim == 1, "Timesteps should be 1D"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mx.exp(exponent)
    emb = (timesteps[:, None].astype(mx.float32) * scale) * emb[None, :]

    # Compute sin and cos embeddings
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    # Zero pad if odd embedding dimension
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])

    return emb


def _resize_image_with_letterbox(
    image: Image.Image,
    target_height: int,
    target_width: int,
) -> Image.Image:
    """Resize an image to fit inside a box while preserving aspect ratio."""
    orig_w, orig_h = image.size
    scale = min(target_width / orig_w, target_height / orig_h)

    resized_w = max(1, min(target_width, int(round(orig_w * scale))))
    resized_h = max(1, min(target_height, int(round(orig_h * scale))))

    resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    offset_x = (target_width - resized_w) // 2
    offset_y = (target_height - resized_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def load_image(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Load and preprocess an image for I2V conditioning.

    Args:
        image_path: Path to the image file
        height: Target height (must be divisible by 32). If None, uses original.
        width: Target width (must be divisible by 32). If None, uses original.

    Returns:
        Image tensor of shape (H, W, 3) in range [0, 1]
    """
    image = Image.open(image_path).convert("RGB")

    # Resize if dimensions specified
    if height is not None and width is not None:
        image = _resize_image_with_letterbox(image, height, width)
    elif height is not None or width is not None:
        # If only one dimension specified, resize preserving aspect ratio
        orig_w, orig_h = image.size
        if height is not None:
            scale = height / orig_h
            new_w = int(orig_w * scale)
            new_w = (new_w // 32) * 32  # Round to nearest 32
            image = image.resize((new_w, height), Image.Resampling.LANCZOS)
        else:
            scale = width / orig_w
            new_h = int(orig_h * scale)
            new_h = (new_h // 32) * 32  # Round to nearest 32
            image = image.resize((width, new_h), Image.Resampling.LANCZOS)
    else:
        # Round to nearest 32
        orig_w, orig_h = image.size
        new_w = (orig_w // 32) * 32
        new_h = (orig_h // 32) * 32
        if new_w != orig_w or new_h != orig_h:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Convert to numpy then MLX
    image_np = np.array(image).astype(np.float32) / 255.0
    return mx.array(image_np, dtype=dtype)


def resize_image_aspect_ratio(
    image: mx.array,
    long_side: int = 512,
) -> mx.array:
    """Resize image preserving aspect ratio, making long side = long_side.

    Args:
        image: Image tensor of shape (H, W, 3)
        long_side: Target size for the longer dimension

    Returns:
        Resized image tensor
    """
    h, w = image.shape[:2]

    if h > w:
        new_h = long_side
        new_w = int(w * long_side / h)
    else:
        new_w = long_side
        new_h = int(h * long_side / w)

    # Round to nearest 32
    new_h = (new_h // 32) * 32
    new_w = (new_w // 32) * 32

    # Use PIL for high-quality resize
    image_np = np.array(image)
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return mx.array(np.array(pil_image).astype(np.float32) / 255.0)


def prepare_image_for_encoding(
    image: mx.array,
    target_height: int,
    target_width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Prepare image for VAE encoding by resizing and normalizing.

    Args:
        image: Image tensor of shape (H, W, 3) in range [0, 1]
        target_height: Target height for the video
        target_width: Target width for the video

    Returns:
        Image tensor ready for encoding, shape (1, 3, 1, H, W) in range [-1, 1]
    """
    h, w = image.shape[:2]

    # Resize if needed
    if h != target_height or w != target_width:
        image_np = np.array(image)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image = _resize_image_with_letterbox(
            pil_image, target_height, target_width
        )
        image = mx.array(np.array(pil_image).astype(np.float32) / 255.0)

    # Normalize to [-1, 1]
    image = image * 2.0 - 1.0

    # Convert to (B, C, 1, H, W)
    image = mx.transpose(image, (2, 0, 1))  # (3, H, W)
    image = mx.expand_dims(image, axis=0)  # (1, 3, H, W)
    image = mx.expand_dims(image, axis=2)  # (1, 3, 1, H, W)

    return image.astype(dtype)
