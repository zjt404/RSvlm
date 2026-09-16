#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed_fixed/sft_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed_fixed/sft_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sft_visual_lora}"
SMOKE="${SMOKE:-0}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_qwenvl.sh"

GRADIENT_ACCUMULATION_STEPS=16
EVAL_STRATEGY=epoch
SAVE_STRATEGY=epoch
LOGGING_STEPS=20
EXTRA_ARGS=()

if [[ "${SMOKE}" == "1" ]]; then
  TRAIN_DATA="${PROJECT_ROOT}/data/fixtures/sft_smoke.jsonl"
  VAL_DATA="${PROJECT_ROOT}/data/fixtures/sft_smoke.jsonl"
  OUTPUT_DIR="${PROJECT_ROOT}/outputs/sft_visual_lora_smoke"
  GRADIENT_ACCUMULATION_STEPS=1
  EVAL_STRATEGY=steps
  SAVE_STRATEGY=steps
  LOGGING_STEPS=1
  EXTRA_ARGS+=(--max_steps 1 --eval_steps 1 --save_steps 1)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_PIXELS="${MAX_PIXELS:-589824}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen3_vl \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate 1e-4 \
  --vit_lr 1e-5 \
  --aligner_lr 5e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit false \
  --freeze_aligner false \
  --gradient_checkpointing true \
  --attn_impl sdpa \
  --max_length 4096 \
  --eval_strategy "${EVAL_STRATEGY}" \
  --save_strategy "${SAVE_STRATEGY}" \
  --save_total_limit 2 \
  --logging_steps "${LOGGING_STEPS}" \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  "${EXTRA_ARGS[@]}"
