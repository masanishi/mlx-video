"""Convert LTX-2 safetensors to MLX directory layout.

Converts from the single-file format (e.g. Lightricks/LTX-2/ltx-2-19b-distilled.safetensors)
to the modular directory structure:

    output/
    ├── transformer/          # DiT transformer weights (sharded)
    │   ├── config.json
    │   ├── model-00001-of-N.safetensors
    │   └── model.safetensors.index.json
    ├── vae/
    │   ├── decoder/          # Video VAE decoder
    │   │   ├── config.json
    │   │   └── model.safetensors
    │   └── encoder/          # Video VAE encoder
    │       ├── config.json
    │       └── model.safetensors
    ├── audio_vae/            # Audio VAE decoder
    │   ├── config.json
    │   └── model.safetensors
    ├── vocoder/              # Audio vocoder
    │   ├── config.json
    │   └── model.safetensors
    └── text_projections/     # Text projection connectors
        └── model.safetensors

Usage:
    # From HF repo ID
    python -m mlx_video.models.ltx.convert --source Lightricks/LTX-2 --output LTX-2-distilled --variant distilled
    python -m mlx_video.models.ltx.convert --source Lightricks/LTX-2 --output LTX-2-dev --variant dev

    # From local folder containing the monolithic safetensors
    python -m mlx_video.models.ltx.convert --source ./Lightricks-LTX-2/ --output LTX-2-distilled --variant distilled

    # From a direct safetensors file path
    python -m mlx_video.models.ltx.convert --source ./ltx-2-19b-distilled.safetensors --output LTX-2-distilled --variant distilled
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict

import mlx.core as mx


# ─── Component configs ────────────────────────────────────────────────────────

TRANSFORMER_CONFIG = {
    "attention_head_dim": 128,
    "attention_type": "default",
    "audio_attention_head_dim": 64,
    "audio_caption_channels": 3840,
    "audio_cross_attention_dim": 2048,
    "audio_in_channels": 128,
    "audio_num_attention_heads": 32,
    "audio_out_channels": 128,
    "audio_positional_embedding_max_pos": [20],
    "av_ca_timestep_scale_multiplier": 1000,
    "caption_channels": 3840,
    "cross_attention_dim": 4096,
    "double_precision_rope": True,
    "in_channels": 128,
    "model_type": "ltx av model",
    "norm_eps": 1e-06,
    "num_attention_heads": 32,
    "num_layers": 48,
    "out_channels": 128,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "positional_embedding_theta": 10000.0,
    "rope_type": "split",
    "timestep_scale_multiplier": 1000,
    "use_middle_indices_grid": True,
}

VAE_DECODER_CONFIG_DISTILLED = {
    "ch": 128,
    "ch_mult": [1, 2, 4],
    "dropout": 0.0,
    "num_res_blocks": 2,
    "out_ch": 2,
    "resolution": 256,
    "timestep_conditioning": False,
    "z_channels": 8,
}

VAE_DECODER_CONFIG_DEV = {
    "ch": 128,
    "ch_mult": [1, 2, 4],
    "dropout": 0.0,
    "num_res_blocks": 2,
    "out_ch": 2,
    "resolution": 256,
    "timestep_conditioning": True,
    "z_channels": 8,
}

VAE_ENCODER_CONFIG = {
    "convolution_dimensions": 3,
    "encoder_blocks": [
        ["res_x", {"num_layers": 4}],
        ["compress_space_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 6}],
        ["compress_time_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 6}],
        ["compress_all_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 2}],
        ["compress_all_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 2}],
    ],
    "encoder_spatial_padding_mode": "zeros",
    "in_channels": 3,
    "latent_log_var": "uniform",
    "norm_layer": "pixel_norm",
    "out_channels": 128,
    "patch_size": 4,
}

AUDIO_VAE_CONFIG = {
    "attn_resolutions": [],
    "attn_type": "vanilla",
    "causality_axis": "height",
    "ch": 128,
    "ch_mult": [1, 2, 4],
    "dropout": 0.0,
    "give_pre_end": False,
    "is_causal": True,
    "mel_bins": 64,
    "mel_hop_length": 160,
    "mid_block_add_attention": False,
    "norm_type": "pixel",
    "num_res_blocks": 2,
    "out_ch": 2,
    "resamp_with_conv": True,
    "resolution": 256,
    "sample_rate": 16000,
    "tanh_out": False,
    "z_channels": 8,
}

VOCODER_CONFIG = {
    "output_sample_rate": 24000,
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "resblock_kernel_sizes": [3, 7, 11],
    "stereo": True,
    "upsample_initial_channel": 1024,
    "upsample_kernel_sizes": [16, 15, 8, 4, 4],
    "upsample_rates": [6, 5, 2, 2, 2],
}


# ─── Key prefix routing ──────────────────────────────────────────────────────

TRANSFORMER_PREFIX = "model.diffusion_model."
VAE_DECODER_PREFIX = "vae.decoder."
VAE_ENCODER_PREFIX = "vae.encoder."
VAE_STATS_PREFIX = "vae.per_channel_statistics."
AUDIO_DECODER_PREFIX = "audio_vae.decoder."
AUDIO_STATS_PREFIX = "audio_vae.per_channel_statistics."
VOCODER_PREFIX = "vocoder."
TEXT_AGG_KEY = "text_embedding_projection.aggregate_embed.weight"
VIDEO_CONNECTOR_PREFIX = "model.diffusion_model.video_embeddings_connector."
AUDIO_CONNECTOR_PREFIX = "model.diffusion_model.audio_embeddings_connector."


# ─── Sanitization functions ──────────────────────────────────────────────────


def sanitize_transformer(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize transformer keys: strip prefix, rename layers, cast to bfloat16."""
    sanitized = {}
    for key, value in weights.items():
        if not key.startswith(TRANSFORMER_PREFIX):
            continue
        # Skip connector weights (they go to text_projections)
        if "audio_embeddings_connector" in key or "video_embeddings_connector" in key:
            continue

        new_key = key[len(TRANSFORMER_PREFIX):]
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
        new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
        new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
        new_key = new_key.replace(".linear_1.", ".linear1.")
        new_key = new_key.replace(".linear_2.", ".linear2.")

        # Cast all weights to bfloat16 (matches MLX model loading behavior)
        if value.dtype != mx.bfloat16:
            value = value.astype(mx.bfloat16)

        sanitized[new_key] = value
    return sanitized


def sanitize_vae_decoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE decoder keys: strip prefix, transpose Conv3d, wrap .conv."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if key.startswith(VAE_STATS_PREFIX):
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics.mean"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics.std"
            else:
                continue
        elif key.startswith(VAE_DECODER_PREFIX):
            new_key = key[len(VAE_DECODER_PREFIX):]
        else:
            continue

        # Conv3d weight transpose: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
        if ".conv.weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Wrap .conv.weight -> .conv.conv.weight (CausalConv3d wrapper)
        if ".conv.weight" in new_key or ".conv.bias" in new_key:
            if ".conv.conv.weight" not in new_key and ".conv.conv.bias" not in new_key:
                new_key = new_key.replace(".conv.weight", ".conv.conv.weight")
                new_key = new_key.replace(".conv.bias", ".conv.conv.bias")

        sanitized[new_key] = value
    return sanitized


def sanitize_vae_encoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE encoder keys: strip prefix, transpose Conv3d/Conv2d."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if "position_ids" in key:
            continue

        if key.startswith(VAE_STATS_PREFIX):
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics.mean"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics.std"
            else:
                continue
            # Per-channel statistics must stay float32 for precision
            if value.dtype != mx.float32:
                value = value.astype(mx.float32)
        elif key.startswith(VAE_ENCODER_PREFIX):
            new_key = key[len(VAE_ENCODER_PREFIX):]
        else:
            continue

        # Conv3d: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_audio_decoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize audio VAE decoder keys: strip prefix, transpose Conv2d."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if key.startswith(AUDIO_DECODER_PREFIX):
            new_key = key[len(AUDIO_DECODER_PREFIX):]
        elif key.startswith(AUDIO_STATS_PREFIX):
            if "mean-of-means" in key:
                new_key = "per_channel_statistics.mean_of_means"
            elif "std-of-means" in key:
                new_key = "per_channel_statistics.std_of_means"
            else:
                continue
        else:
            continue

        # Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_vocoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize vocoder keys: strip prefix, transpose Conv1d/ConvTranspose1d."""
    sanitized = {}
    for key, value in weights.items():
        if not key.startswith(VOCODER_PREFIX):
            continue

        new_key = key[len(VOCODER_PREFIX):]

        # Handle Conv1d/ConvTranspose1d weight shape conversion
        if "weight" in new_key and value.ndim == 3:
            if "ups" in new_key:
                # ConvTranspose1d: PyTorch (in_ch, out_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                value = mx.transpose(value, (1, 2, 0))
            else:
                # Conv1d: PyTorch (out_ch, in_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                value = mx.transpose(value, (0, 2, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_connector_key(key: str) -> str:
    """Sanitize connector sub-key names."""
    key = key.replace(".ff.net.0.proj.", ".ff.proj_in.")
    key = key.replace(".ff.net.2.", ".ff.proj_out.")
    key = key.replace(".to_out.0.", ".to_out.")
    return key


def extract_text_projections(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Extract text projection weights (aggregate_embed + connectors)."""
    extracted = {}

    # aggregate_embed
    if TEXT_AGG_KEY in weights:
        extracted["aggregate_embed.weight"] = weights[TEXT_AGG_KEY]

    # video_embeddings_connector
    for key, value in weights.items():
        if key.startswith(VIDEO_CONNECTOR_PREFIX):
            suffix = key[len(VIDEO_CONNECTOR_PREFIX):]
            new_key = "video_embeddings_connector." + sanitize_connector_key(suffix)
            extracted[new_key] = value

    # audio_embeddings_connector
    for key, value in weights.items():
        if key.startswith(AUDIO_CONNECTOR_PREFIX):
            suffix = key[len(AUDIO_CONNECTOR_PREFIX):]
            new_key = "audio_embeddings_connector." + sanitize_connector_key(suffix)
            extracted[new_key] = value

    return extracted


# ─── Saving utilities ─────────────────────────────────────────────────────────


def save_sharded(
    weights: Dict[str, mx.array],
    output_dir: Path,
    max_shard_size_bytes: int = 5 * 1024 * 1024 * 1024,  # 5GB per shard
):
    """Save weights as sharded safetensors with an index file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort keys for deterministic output
    sorted_keys = sorted(weights.keys())

    # Calculate total size
    total_size = sum(weights[k].nbytes for k in sorted_keys)

    # Determine sharding
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted_keys:
        tensor = weights[key]
        tensor_size = tensor.nbytes

        if current_size + tensor_size > max_shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}

    for i, shard in enumerate(shards):
        if num_shards == 1:
            filename = "model.safetensors"
        else:
            filename = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"

        mx.save_safetensors(str(output_dir / filename), shard)

        for key in shard:
            weight_map[key] = filename

    # Write index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    return num_shards


def save_single(weights: Dict[str, mx.array], output_dir: Path):
    """Save weights as a single safetensors file with an index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)

    # Also write index for consistency
    total_size = sum(v.nbytes for v in weights.values())
    weight_map = {k: "model.safetensors" for k in sorted(weights.keys())}
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def save_config(config: dict, output_dir: Path):
    """Save config.json to a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


# ─── Source resolution ─────────────────────────────────────────────────────────

VARIANT_FILE_PATTERNS = {
    "distilled": "ltx-2-19b-distilled.safetensors",
    "dev": "ltx-2-19b-dev.safetensors",
}

# Matches upscaler files like ltx-2-spatial-upscaler-x2-1.0.safetensors,
# ltx-2.3-spatial-upscaler-x2-1.0.safetensors, etc.
UPSCALER_PATTERN = re.compile(r"^ltx-[\d.]+-(?:spatial|temporal)-upscaler-.+\.safetensors$")


def resolve_source(source: str, variant: str) -> Path:
    """Resolve source to a monolithic safetensors file path.

    Args:
        source: HF repo ID (e.g. "Lightricks/LTX-2"), local directory, or direct file path.
        variant: Model variant ("distilled" or "dev") to select the right file.

    Returns:
        Path to the monolithic safetensors file.
    """
    source_path = Path(source)

    # Direct file path
    if source_path.is_file():
        return source_path

    # Local directory — look for the variant's safetensors file
    if source_path.is_dir():
        target = VARIANT_FILE_PATTERNS.get(variant)
        if target:
            candidate = source_path / target
            if candidate.is_file():
                return candidate

        # Fallback: glob for ltx-2-19b-*.safetensors
        matches = sorted(source_path.glob("ltx-2-19b-*.safetensors"))
        if matches:
            if len(matches) == 1:
                return matches[0]
            # Multiple matches — pick by variant keyword
            for m in matches:
                if variant in m.name:
                    return m
            return matches[0]

        raise FileNotFoundError(
            f"No ltx-2-19b-*.safetensors found in {source_path}. "
            f"Expected {target} for variant '{variant}'."
        )

    # HF repo ID — download via huggingface_hub
    if "/" in source and not source_path.exists():
        from huggingface_hub import hf_hub_download

        filename = VARIANT_FILE_PATTERNS.get(variant)
        if not filename:
            raise ValueError(f"Unknown variant '{variant}'. Expected 'distilled' or 'dev'.")

        print(f"Downloading {filename} from {source}...")
        local_path = hf_hub_download(repo_id=source, filename=filename)
        return Path(local_path)

    raise FileNotFoundError(
        f"Source not found: {source}. Provide an HF repo ID, local directory, or file path."
    )


# ─── Main ─────────────────────────────────────────────────────────────────────


def convert(source: str, output_path: Path, variant: str = "distilled"):
    """Convert monolithic safetensors to modular directory layout.

    Args:
        source: HF repo ID (e.g. "Lightricks/LTX-2"), local directory, or file path.
        output_path: Output directory for the modular layout.
        variant: "distilled" or "dev".
    """
    source_path = resolve_source(source, variant)

    print(f"Loading monolithic weights from {source_path.name}...")
    all_weights = mx.load(str(source_path))
    total_keys = len(all_weights)
    print(f"  Loaded {total_keys} keys")

    # Route keys to components
    print("\nExtracting components...")

    # 1. Transformer
    print("  [1/6] Transformer...")
    transformer_weights = sanitize_transformer(all_weights)
    num_shards = save_sharded(transformer_weights, output_path / "transformer")
    save_config(TRANSFORMER_CONFIG, output_path / "transformer")
    t_params = sum(v.size for v in transformer_weights.values())
    print(f"    {len(transformer_weights)} keys, {t_params:,} params, {num_shards} shards")

    # 2. VAE Decoder
    print("  [2/6] VAE Decoder...")
    vae_decoder_weights = sanitize_vae_decoder(all_weights)
    save_single(vae_decoder_weights, output_path / "vae" / "decoder")
    decoder_config = VAE_DECODER_CONFIG_DISTILLED if variant == "distilled" else VAE_DECODER_CONFIG_DEV
    save_config(decoder_config, output_path / "vae" / "decoder")
    d_params = sum(v.size for v in vae_decoder_weights.values())
    print(f"    {len(vae_decoder_weights)} keys, {d_params:,} params")

    # 3. VAE Encoder
    print("  [3/6] VAE Encoder...")
    vae_encoder_weights = sanitize_vae_encoder(all_weights)
    save_single(vae_encoder_weights, output_path / "vae" / "encoder")
    save_config(VAE_ENCODER_CONFIG, output_path / "vae" / "encoder")
    e_params = sum(v.size for v in vae_encoder_weights.values())
    print(f"    {len(vae_encoder_weights)} keys, {e_params:,} params")

    # 4. Audio VAE Decoder
    print("  [4/6] Audio VAE Decoder...")
    audio_decoder_weights = sanitize_audio_decoder(all_weights)
    save_single(audio_decoder_weights, output_path / "audio_vae")
    save_config(AUDIO_VAE_CONFIG, output_path / "audio_vae")
    a_params = sum(v.size for v in audio_decoder_weights.values())
    print(f"    {len(audio_decoder_weights)} keys, {a_params:,} params")

    # 5. Vocoder
    print("  [5/6] Vocoder...")
    vocoder_weights = sanitize_vocoder(all_weights)
    save_single(vocoder_weights, output_path / "vocoder")
    save_config(VOCODER_CONFIG, output_path / "vocoder")
    v_params = sum(v.size for v in vocoder_weights.values())
    print(f"    {len(vocoder_weights)} keys, {v_params:,} params")

    # 6. Text Projections
    print("  [6/6] Text Projections...")
    text_proj_weights = extract_text_projections(all_weights)
    tp_dir = output_path / "text_projections"
    tp_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(tp_dir / "model.safetensors"), text_proj_weights)
    tp_params = sum(v.size for v in text_proj_weights.values())
    print(f"    {len(text_proj_weights)} keys, {tp_params:,} params")

    # 7. Copy upscaler files
    print("\nCopying upscaler files...")
    source_dir = source_path.parent
    is_hf_repo = "/" in source and not Path(source).exists()
    upscaler_files = []

    if is_hf_repo:
        from huggingface_hub import list_repo_files

        upscaler_files = [
            f for f in list_repo_files(source) if UPSCALER_PATTERN.match(f)
        ]
    else:
        upscaler_files = [
            f.name for f in source_dir.iterdir()
            if f.is_file() and UPSCALER_PATTERN.match(f.name)
        ]

    if not upscaler_files:
        print("  No upscaler files found")

    for upscaler_file in sorted(upscaler_files):
        dest = output_path / upscaler_file
        if dest.exists():
            print(f"  {upscaler_file}: already exists, skipping")
            continue

        local_candidate = source_dir / upscaler_file
        if local_candidate.is_file():
            shutil.copy2(str(local_candidate), str(dest))
            print(f"  {upscaler_file}: copied")
        elif is_hf_repo:
            from huggingface_hub import hf_hub_download

            print(f"  {upscaler_file}: downloading from {source}...")
            downloaded = hf_hub_download(repo_id=source, filename=upscaler_file)
            shutil.copy2(downloaded, str(dest))
            print(f"  {upscaler_file}: done")
        else:
            print(f"  {upscaler_file}: not found, skipping")

    # Summary
    all_converted = (
        len(transformer_weights)
        + len(vae_decoder_weights)
        + len(vae_encoder_weights)
        + len(audio_decoder_weights)
        + len(vocoder_weights)
        + len(text_proj_weights)
    )
    print(f"\nDone! Converted {all_converted}/{total_keys} keys")
    if all_converted < total_keys:
        # Find unconverted keys
        converted_prefixes = set()
        for key in all_weights:
            if key.startswith(TRANSFORMER_PREFIX):
                converted_prefixes.add(key)
            elif key.startswith(VAE_DECODER_PREFIX) or key.startswith(VAE_STATS_PREFIX):
                converted_prefixes.add(key)
            elif key.startswith(VAE_ENCODER_PREFIX):
                converted_prefixes.add(key)
            elif key.startswith(AUDIO_DECODER_PREFIX) or key.startswith(AUDIO_STATS_PREFIX):
                converted_prefixes.add(key)
            elif key.startswith(VOCODER_PREFIX):
                converted_prefixes.add(key)
            elif key == TEXT_AGG_KEY:
                converted_prefixes.add(key)
            elif key.startswith(VIDEO_CONNECTOR_PREFIX) or key.startswith(AUDIO_CONNECTOR_PREFIX):
                converted_prefixes.add(key)

        skipped = set(all_weights.keys()) - converted_prefixes
        if skipped:
            print(f"  Skipped {len(skipped)} keys:")
            for k in sorted(skipped)[:20]:
                print(f"    {k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert monolithic LTX-2 safetensors to modular MLX layout"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="HF repo ID (e.g. Lightricks/LTX-2), local directory, or direct safetensors file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for modular layout",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["distilled", "dev"],
        default="distilled",
        help="Model variant (affects VAE decoder config and which file to download)",
    )
    args = parser.parse_args()

    convert(args.source, Path(args.output), variant=args.variant)
