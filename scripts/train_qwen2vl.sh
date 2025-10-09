#!/usr/bin/env bash
# Fine-tune Qwen2-VL using ms-swift and the custom driving-scene dataset.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/dataset/train_qwen.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/qwen2vl_lora}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-2B-Instruct}"
TRAIN_TYPE="${TRAIN_TYPE:-lora}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

mkdir -p "${OUTPUT_DIR}"

export MAX_PIXELS

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found at ${DATASET_PATH}" >&2
  exit 1
fi

swift sft \
  --model "${MODEL_NAME}" \
  --dataset "${DATASET_PATH}" \
  --train_type "${TRAIN_TYPE}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --target_modules "${TARGET_MODULES:-all-linear}" \
  --freeze_vit "${FREEZE_VIT:-true}" \
  --max_length "${MAX_LENGTH}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --weight_decay "${WEIGHT_DECAY:-0}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --eval_steps "${EVAL_STEPS:-500}" \
  --load_from_cache_file true \
  --split_dataset_ratio "${SPLIT_DATASET_RATIO:-0.02}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --dataset_num_proc "${DATASET_NUM_PROC:-4}" \
  --output_dir "${OUTPUT_DIR}"
