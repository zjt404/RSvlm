#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
VRSBENCH_ROOT="${VRSBENCH_ROOT:-${PROJECT_ROOT}/data/raw/VRSBench}"
DIOR_ROOT="${DIOR_ROOT:-${PROJECT_ROOT}/data/raw/DIOR}"
INTERIM_DIR="${INTERIM_DIR:-${PROJECT_ROOT}/data/interim}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/processed}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_qwenvl.sh"
mkdir -p "${INTERIM_DIR}" "${OUTPUT_DIR}"

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

rs-vlm-data dior \
  --root "${DIOR_ROOT}" \
  --output "${INTERIM_DIR}/dior.jsonl" \
  --max-spatial-per-image 8

rs-vlm-data assemble \
  --inputs \
    "${INTERIM_DIR}/vrsbench_train.jsonl" \
    "${INTERIM_DIR}/vrsbench_test_vqa.jsonl" \
    "${INTERIM_DIR}/vrsbench_test_grounding.jsonl" \
    "${INTERIM_DIR}/dior.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --grpo-size 8000

rs-vlm-data validate \
  --inputs "${OUTPUT_DIR}/sft_train.jsonl" "${OUTPUT_DIR}/sft_val.jsonl" "${OUTPUT_DIR}/sft_test.jsonl" \
  --require-images
