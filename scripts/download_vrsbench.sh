#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
TARGET_DIR="${VRSBENCH_ROOT:-${PROJECT_ROOT}/data/raw/VRSBench}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT
mkdir -p "${TARGET_DIR}"

hf download xiang709/VRSBench \
  --repo-type dataset \
  --local-dir "${TARGET_DIR}"

if [[ "${EXTRACT:-1}" == "1" ]]; then
  python -m zipfile -e "${TARGET_DIR}/Images_train.zip" "${TARGET_DIR}/Images_train"
  python -m zipfile -e "${TARGET_DIR}/Images_val.zip" "${TARGET_DIR}/Images_val"
fi

echo "Downloaded VRSBench to ${TARGET_DIR}. Review its non-commercial image restrictions before redistribution."
