#!/usr/bin/env bash
# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file; see NOTICE.
#
# One-time setup on Linux + NVIDIA: venv with MLX's CUDA backend, then the Qwen3 MLX 4-bit weights the tests use.
# usage: bash scripts/cuda/setup.sh [venv-dir]      (default: .venv-cuda in the repo root)
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV=${1:-.venv-cuda}
if command -v uv >/dev/null; then
  [ -d "$VENV" ] || uv venv -p 3.12 "$VENV"
  . "$VENV/bin/activate"
  uv pip install -r requirements-cuda.txt
else
  [ -d "$VENV" ] || python3.12 -m venv "$VENV"
  . "$VENV/bin/activate"
  pip install -r requirements-cuda.txt
fi
python -c "import mlx.core as mx; print('mlx', mx.__version__, 'gpu available:', mx.is_available(mx.gpu))"
for m in Qwen/Qwen3-0.6B-MLX-4bit Qwen/Qwen3-1.7B-MLX-4bit Qwen/Qwen3-4B-MLX-4bit; do
  python -c "from huggingface_hub import snapshot_download as s; print(s('$m'))"
done
echo "SETUP_OK  (activate with: . $VENV/bin/activate)"
