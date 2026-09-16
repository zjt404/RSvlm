#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}}"
BASE_ADAPTER="${BASE_ADAPTER:-${PROJECT_ROOT}/outputs/sft_v5_all_boxes_language/v0-20260908-175838/checkpoint-1412}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed_context_v6_dense_sft/sft_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed_context_v6_dense_sft/sft_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sft_context_v6_dense}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_qwenvl.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_PIXELS="${MAX_PIXELS:-589824}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen3_vl \
  --adapters "${BASE_ADAPTER}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
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
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 10 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard
