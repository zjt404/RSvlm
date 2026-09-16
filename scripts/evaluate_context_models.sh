#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}}"
RELATION_TEST="${RELATION_TEST:-${PROJECT_ROOT}/data/processed_stage2_context/sft_context_test.jsonl}"
ORIGINAL_TEST="${ORIGINAL_TEST:-${PROJECT_ROOT}/data/processed_fixed/sft_test.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/outputs/evaluation_context_stage2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SFT_ADAPTER="${SFT_ADAPTER:-${PROJECT_ROOT}/outputs/sft_corrected/v1-20260827-154532/checkpoint-4880}"
STAGE2_ADAPTER="${STAGE2_ADAPTER:?Set STAGE2_ADAPTER to the completed Stage-2 checkpoint}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_qwenvl.sh"
mkdir -p "${RESULTS_DIR}"

run_relation() {
  local name="$1"
  local adapter="$2"
  local adapter_args=()
  if [[ -n "${adapter}" ]]; then
    adapter_args+=(--adapter "${adapter}")
  fi
  python -m rs_vlm.predict \
    --dataset "${RELATION_TEST}" \
    --model "${MODEL_PATH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --output "${RESULTS_DIR}/${name}_relation.jsonl" \
    "${adapter_args[@]}"
  python scripts/evaluate_context.py \
    "${RESULTS_DIR}/${name}_relation.jsonl" \
    --output "${RESULTS_DIR}/${name}_relation_metrics.json"
}

run_relation base ""
run_relation sft "${SFT_ADAPTER}"
run_relation stage2 "${STAGE2_ADAPTER}"

python -m rs_vlm.predict \
  --dataset "${ORIGINAL_TEST}" \
  --model "${MODEL_PATH}" \
  --adapter "${STAGE2_ADAPTER}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "${RESULTS_DIR}/stage2_original.jsonl"
rs-vlm-eval "${RESULTS_DIR}/stage2_original.jsonl" \
  --output "${RESULTS_DIR}/stage2_original_metrics.json"
