#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}}"
TEST_DATA="${TEST_DATA:-${PROJECT_ROOT}/data/processed_fixed/sft_test.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/outputs/evaluation}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_qwenvl.sh"
mkdir -p "${RESULTS_DIR}"

run_eval() {
  local name="$1"
  local adapter="$2"
  local adapter_args=()
  if [[ -n "${adapter}" ]]; then
    adapter_args+=(--adapter "${adapter}")
  fi
  python -m rs_vlm.predict \
    --dataset "${TEST_DATA}" \
    --model "${MODEL_PATH}" \
    --output "${RESULTS_DIR}/${name}.jsonl" \
    "${adapter_args[@]}"
  rs-vlm-eval "${RESULTS_DIR}/${name}.jsonl" --output "${RESULTS_DIR}/${name}_metrics.json"
}

run_eval base ""
if [[ -n "${SFT_ADAPTER:-}" ]]; then
  run_eval sft "${SFT_ADAPTER}"
fi
if [[ -n "${GRPO_ADAPTER:-}" ]]; then
  run_eval grpo "${GRPO_ADAPTER}"
  python -m rs_vlm.predict \
    --dataset "${TEST_DATA}" \
    --model "${MODEL_PATH}" \
    --adapter "${GRPO_ADAPTER}" \
    --shuffle-images \
    --output "${RESULTS_DIR}/grpo_shuffled.jsonl"
  rs-vlm-eval "${RESULTS_DIR}/grpo_shuffled.jsonl" --output "${RESULTS_DIR}/grpo_shuffled_metrics.json"
fi
