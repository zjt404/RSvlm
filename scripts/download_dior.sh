#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
TARGET_DIR="${DIOR_ROOT:-${PROJECT_ROOT}/data/raw/DIOR}"
DATASET_ID="${DIOR_DATASET_ID:-wokaikaixinxin/DIOR}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_qwenvl.sh"
mkdir -p "${TARGET_DIR}"

modelscope download --dataset "${DATASET_ID}" \
  Annotations.zip \
  ImageSets.zip \
  JPEGImages-trainval.zip \
  JPEGImages-test.zip \
  --local_dir "${TARGET_DIR}" \
  --max-workers "${MAX_WORKERS:-4}"

for archive in Annotations.zip ImageSets.zip JPEGImages-trainval.zip JPEGImages-test.zip; do
  python -m zipfile -t "${TARGET_DIR}/${archive}"
done

if [[ "${EXTRACT:-1}" == "1" ]]; then
  for archive in Annotations.zip ImageSets.zip JPEGImages-trainval.zip JPEGImages-test.zip; do
    python -m zipfile -e "${TARGET_DIR}/${archive}" "${TARGET_DIR}"
  done
fi

echo "Downloaded and verified DIOR at ${TARGET_DIR}."
