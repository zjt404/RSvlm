#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
VRSBENCH_ROOT="${VRSBENCH_ROOT:-${PROJECT_ROOT}/data/raw/VRSBench}"
INTERIM_DIR="${INTERIM_DIR:-${PROJECT_ROOT}/data/interim}"
WAIT_PID="${WAIT_PID:-}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_qwenvl.sh"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

for archive in Images_train.zip Images_val.zip; do
  test -s "${VRSBENCH_ROOT}/${archive}"
  python -m zipfile -t "${VRSBENCH_ROOT}/${archive}"
  python -m zipfile -e "${VRSBENCH_ROOT}/${archive}" "${VRSBENCH_ROOT}"
done

mkdir -p "${INTERIM_DIR}"

rs-vlm-data vrsbench \
  --annotations "${VRSBENCH_ROOT}/VRSBench_train.json" \
  --images-root "${VRSBENCH_ROOT}" \
  --source-split train \
  --task auto \
  --output "${INTERIM_DIR}/vrsbench_train.jsonl"

rs-vlm-data vrsbench \
  --annotations "${VRSBENCH_ROOT}/VRSBench_EVAL_vqa.json" \
  --images-root "${VRSBENCH_ROOT}" \
  --source-split test \
  --task vqa \
  --output "${INTERIM_DIR}/vrsbench_test_vqa.jsonl"

rs-vlm-data vrsbench \
  --annotations "${VRSBENCH_ROOT}/VRSBench_EVAL_referring.json" \
  --images-root "${VRSBENCH_ROOT}" \
  --source-split test \
  --task grounding \
  --output "${INTERIM_DIR}/vrsbench_test_grounding.jsonl"

rs-vlm-data validate \
  --inputs \
    "${INTERIM_DIR}/vrsbench_train.jsonl" \
    "${INTERIM_DIR}/vrsbench_test_vqa.jsonl" \
    "${INTERIM_DIR}/vrsbench_test_grounding.jsonl" \
  --require-images
