# mlx-video

MLX-Video is the best package for inference and finetuning of Image-Video-Audio generation models on your Mac using MLX.

## Installation

### Option 1: Install with pip (requires git):
```bash
pip install git+https://github.com/Blaizzy/mlx-video.git
```

### Option 2: Install with uv (ultra-fast package manager, optional):
```bash
uv pip install git+https://github.com/Blaizzy/mlx-video.git
```

## Supported Models

### LTX-2

[LTX-2](https://huggingface.co/Lightricks/LTX-2) is a 19B parameter video generation model from Lightricks. See the full [LTX-2 model card](mlx_video/models/ltx_2/README.md) for detailed usage, CLI options, pipeline descriptions, and architecture.

**Features:**
- Text-to-Video (T2V), Image-to-Video (I2V), and Audio-to-Video (A2V)
- Four pipelines: Distilled (fast), Dev (CFG), Dev Two-Stage (LoRA), Dev Two-Stage HQ (highest quality)
- Synchronized audio-video generation (experimental)
- LoRA support (local files or HuggingFace repos)
- Prompt enhancement via Gemma
- 2x spatial upscaling for images and videos

**Quick start:**

```bash
# Text-to-Video (distilled, fastest)
uv run mlx_video.generate --prompt "Two dogs wearing sunglasses, cinematic, sunset" -n 97 --width 768

# Image-to-Video
uv run mlx_video.generate --prompt "A person dancing" --image photo.jpg

# Audio-to-Video
uv run mlx_video.generate --audio-file music.wav --prompt "A band playing music"

# Dev pipeline with CFG (higher quality)
uv run mlx_video.generate --pipeline dev --prompt "A cinematic scene" --cfg-scale 3.0

# Dev two-stage HQ (highest quality)
uv run mlx_video.generate --pipeline dev-two-stage-hq \
    --prompt "A cinematic scene of ocean waves at golden hour" \
    --model-repo prince-canuma/LTX-2-dev
```

<img src="https://github.com/Blaizzy/mlx-video/raw/main/examples/poodles.gif" width="512" alt="Poodles demo">

**Converting weights:**

Pre-converted weights are available on HuggingFace ([LTX-2-distilled](https://huggingface.co/prince-canuma/LTX-2-distilled), [LTX-2-dev](https://huggingface.co/prince-canuma/LTX-2-dev), [LTX-2.3-distilled](https://huggingface.co/prince-canuma/LTX-2.3-distilled), [LTX-2.3-dev](https://huggingface.co/prince-canuma/LTX-2.3-dev)), or convert from the original Lightricks checkpoint:

```bash
uv run python -m mlx_video.models.ltx_2.convert \
    --source Lightricks/LTX-2 --output ./LTX-2-distilled --variant distilled
```

## Requirements

- macOS with Apple Silicon
- Python >= 3.11
- MLX >= 0.22.0

## License

MIT
