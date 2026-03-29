#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
MLX_REPO_URL="${MLX_REPO_URL:-https://github.com/ml-explore/mlx.git}"
MLX_REF="${MLX_REF:-0ff1115a46c77d7a99c075b2e8376b0cbf91f781}"
MLX_PATCH="${MLX_PATCH:-$ROOT_DIR/patches/mlx-mxfp8-qqmm.patch}"
MLX_BUILD_ROOT="${MLX_BUILD_ROOT:-$ROOT_DIR/.mlx-build}"
MLX_SRC_DIR="${MLX_SRC_DIR:-$MLX_BUILD_ROOT/mlx-src-${MLX_REF:0:7}}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing virtualenv python at $VENV_PYTHON" >&2
  echo "Create the repo .venv first." >&2
  exit 1
fi

for cmd in git uv xcrun xcodebuild; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if [[ ! -f "$MLX_PATCH" ]]; then
  echo "Missing patch file: $MLX_PATCH" >&2
  exit 1
fi

mkdir -p "$MLX_BUILD_ROOT"

if ! xcrun metal -v >/dev/null 2>&1; then
  echo "Installing Metal Toolchain..."
  xcodebuild -downloadComponent MetalToolchain
fi

if [[ ! -d "$MLX_SRC_DIR/.git" ]]; then
  echo "Cloning MLX into $MLX_SRC_DIR"
  git clone "$MLX_REPO_URL" "$MLX_SRC_DIR"
fi

CURRENT_HEAD="$(git -C "$MLX_SRC_DIR" rev-parse HEAD 2>/dev/null || true)"
if [[ "$CURRENT_HEAD" != "$MLX_REF" ]]; then
  if [[ -n "$(git -C "$MLX_SRC_DIR" status --porcelain 2>/dev/null)" ]]; then
    echo "MLX source tree has local changes: $MLX_SRC_DIR" >&2
    echo "Remove it or point MLX_SRC_DIR at a clean checkout." >&2
    exit 1
  fi
  echo "Checking out MLX commit $MLX_REF"
  git -C "$MLX_SRC_DIR" fetch --tags origin
  git -C "$MLX_SRC_DIR" checkout --detach "$MLX_REF"
fi

if git -C "$MLX_SRC_DIR" apply --reverse --check "$MLX_PATCH" >/dev/null 2>&1; then
  echo "MLX qqmm patch already applied"
else
  echo "Applying MLX qqmm patch"
  git -C "$MLX_SRC_DIR" apply "$MLX_PATCH"
fi

echo "Installing build prerequisites into repo venv"
uv pip install --python "$VENV_PYTHON" setuptools wheel typing_extensions numpy

echo "Replacing stock MLX in repo venv"
uv pip uninstall --python "$VENV_PYTHON" mlx >/dev/null 2>&1 || true
env -u DEBUG uv pip install --python "$VENV_PYTHON" -e "$MLX_SRC_DIR" --no-build-isolation

echo "Verifying mxfp8 activation quantization support"
"$VENV_PYTHON" - <<'PY'
import mlx.core as mx
import mlx_video.quantization as q

version = mx.__version__
probe = q.activation_quantized_matmul_supported("mxfp8")
core_file = getattr(mx, "__file__", "<unknown>")
print(f"mlx={version}")
print(f"mxfp8_activation_quantization={probe}")
print(f"core={core_file}")
if not probe:
    raise SystemExit("Patched MLX installed, but qqmm probe still failed.")
PY

cat <<EOF

Patched MLX install complete.

Use one of these commands for LTX runs:
  uv run --no-sync mlx_video.ltx_2.generate ...
  .venv/bin/mlx_video.ltx_2.generate ...

Do not use plain 'uv run ...' here. It will resync uv.lock and restore stock mlx.
EOF
