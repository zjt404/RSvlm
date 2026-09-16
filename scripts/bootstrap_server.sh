#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/qwenvl}"

cd "${PROJECT_ROOT}"
source "${VENV_DIR}/bin/activate"

# ms-swift 4.5 imports the FSDP2 API added in PyTorch 2.6. Driver 535 is
# compatible with the CUDA 11.8 wheel and does not meet the CUDA 12.4 driver
# requirement, so use the official cu118 build.
uv pip install --python "${VENV_DIR}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.6.0" "torchvision==0.21.0"

uv pip install --python "${VENV_DIR}/bin/python" -e '.[train,app,dev]'
python scripts/make_smoke_data.py

python - <<'PY'
import torch
import transformers
import swift
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("ms_swift", getattr(swift, "__version__", "unknown"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY

pytest -q
