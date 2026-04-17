#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

detect_nproc_per_node() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    python - <<'PY'
import os
devices = [d.strip() for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d.strip()]
print(len(devices))
PY
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus | wc -l | tr -d ' '
    return
  fi

  echo 1
}

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/helpers/llama3.2_1B_lora.yaml}"
DATA_FILE="${DATA_FILE:-$ROOT_DIR/data/preference_Llama-3.2-1B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca.json}"
CACHE_DIR="${CACHE_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/adv_dpo_1b}"

NPROC_PER_NODE="${NPROC_PER_NODE:-$(detect_nproc_per_node)}"
LR="${LR:-1e-4}"
ATTACKER_LR="${ATTACKER_LR:-5e-4}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

ENABLE_ATTACK_INNER_LOOP="${ENABLE_ATTACK_INNER_LOOP:-True}"
ATTACK_INNER_STEPS="${ATTACK_INNER_STEPS:-3}"
RESET_ATTACK_TOKENS_EACH_BATCH="${RESET_ATTACK_TOKENS_EACH_BATCH:-True}"
ATTACKER_REFERENCE_FREE="${ATTACKER_REFERENCE_FREE:-False}"

LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
LOG_PEAK_MEMORY_STATS="${LOG_PEAK_MEMORY_STATS:-True}"

if [[ -z "$CACHE_DIR" ]]; then
  echo "CACHE_DIR is required and should point to the local model snapshot directory." >&2
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Dataset file not found: $DATA_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

CMD=(
  torchrun
  --nproc_per_node "$NPROC_PER_NODE"
  "$ROOT_DIR/lora_dpo_distributed.py"
  --config "$CONFIG_FILE"
  "output_dir=$OUTPUT_DIR"
  "cache_dir=$CACHE_DIR"
  "dataset.data_files=$DATA_FILE"
  "optimizer.lr=$LR"
  "attacker_optimizer.lr=$ATTACKER_LR"
  "epochs=$EPOCHS"
  "max_steps_per_epoch=$MAX_STEPS_PER_EPOCH"
  "batch_size=$BATCH_SIZE"
  "gradient_accumulation_steps=$GRAD_ACCUM"
  "enable_attack_inner_loop=$ENABLE_ATTACK_INNER_LOOP"
  "attack_inner_steps=$ATTACK_INNER_STEPS"
  "reset_attack_tokens_each_batch=$RESET_ATTACK_TOKENS_EACH_BATCH"
  "attacker_reference_free=$ATTACKER_REFERENCE_FREE"
  "log_every_n_steps=$LOG_EVERY_N_STEPS"
  "log_peak_memory_stats=$LOG_PEAK_MEMORY_STATS"
)

if (($# > 0)); then
  CMD+=("$@")
fi

echo "Launching adversarial DPO training"
echo "ROOT_DIR=$ROOT_DIR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "DATA_FILE=$DATA_FILE"
echo "CACHE_DIR=$CACHE_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LR=$LR"
echo "ATTACKER_LR=$ATTACKER_LR"
echo "EPOCHS=$EPOCHS"
echo "MAX_STEPS_PER_EPOCH=$MAX_STEPS_PER_EPOCH"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "GRAD_ACCUM=$GRAD_ACCUM"
echo "ENABLE_ATTACK_INNER_LOOP=$ENABLE_ATTACK_INNER_LOOP"
echo "ATTACK_INNER_STEPS=$ATTACK_INNER_STEPS"
echo "RESET_ATTACK_TOKENS_EACH_BATCH=$RESET_ATTACK_TOKENS_EACH_BATCH"
echo "ATTACKER_REFERENCE_FREE=$ATTACKER_REFERENCE_FREE"
echo
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"
