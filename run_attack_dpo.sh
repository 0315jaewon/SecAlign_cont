#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_FILE="${CONFIG_FILE:-helpers/llama3.1_8B_lora.yaml}"
DATA_FILE="${DATA_FILE:-data/preference_Llama-3.1-8B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca.json}"
CACHE_DIR="${CACHE_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/attack_dpo_8b_lr1e-4}"
LR="${LR:-1e-4}"

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

NPROC_PER_NODE="${NPROC_PER_NODE:-$(detect_nproc_per_node)}"

if [[ -z "$CACHE_DIR" ]]; then
  echo "CACHE_DIR is required and should point at the local Meta-Llama-3.1-8B-Instruct snapshot." >&2
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

echo "Launching adversarial DPO training"
echo "ROOT_DIR=$ROOT_DIR"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "DATA_FILE=$DATA_FILE"
echo "CACHE_DIR=$CACHE_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LR=$LR"

torchrun \
  --nproc_per_node "$NPROC_PER_NODE" \
  "$ROOT_DIR/lora_dpo_distributed.py" \
  --config "$CONFIG_FILE" \
  output_dir="$OUTPUT_DIR" \
  dataset.data_files="$DATA_FILE" \
  optimizer.lr="$LR" \
  cache_dir="$CACHE_DIR"
