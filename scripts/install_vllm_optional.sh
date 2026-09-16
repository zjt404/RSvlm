#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/qwenvl}"
cd "${PROJECT_ROOT}"
source "${VENV_DIR}/bin/activate"

echo "The server currently uses NVIDIA driver 535. Verify the vLLM wheel CUDA requirement before accepting dependency changes."
uv pip install --python "${VENV_DIR}/bin/python" "vllm>=0.11,<0.12"
python -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)'

