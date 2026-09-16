#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}}"
SFT_ADAPTER="${SFT_ADAPTER:?Set SFT_ADAPTER to the selected SFT checkpoint}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed_fixed/grpo_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/grpo}"
SMOKE="${SMOKE:-0}"
USE_VLLM="${USE_VLLM:-0}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_qwenvl.sh"

EXTRA_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  TRAIN_DATA="${PROJECT_ROOT}/data/fixtures/grpo_smoke.jsonl"
  OUTPUT_DIR="${PROJECT_ROOT}/outputs/grpo_smoke"
  EXTRA_ARGS+=(--max_steps 1 --save_steps 1 --logging_steps 1)
fi
if [[ "${USE_VLLM}" == "1" ]]; then
  EXTRA_ARGS+=(
    --use_vllm true
    --vllm_mode colocate
    --vllm_gpu_memory_utilization 0.35
    --vllm_max_model_len 4224
    --vllm_enable_lora true
    --vllm_max_lora_rank 16
    --sleep_level 1
  )
else
  EXTRA_ARGS+=(--use_vllm false)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_PIXELS="${MAX_PIXELS:-589824}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --model_type qwen3_vl \
  --adapters "${SFT_ADAPTER}" \
  --dataset "${TRAIN_DATA}" \
  --external_plugins "${PROJECT_ROOT}/src/rs_vlm/rewards.py" \
  --reward_funcs rs_task_reward \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.05 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --attn_impl sdpa \
  --max_length 4096 \
  --max_completion_length 64 \
  --num_generations 4 \
  --temperature 0.8 \
  --top_p 0.95 \
  --beta 0.01 \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 2 \
  --log_completions true \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  "${EXTRA_ARGS[@]}"
