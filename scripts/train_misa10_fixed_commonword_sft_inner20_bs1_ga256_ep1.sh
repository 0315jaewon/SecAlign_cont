#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p misa10_fixed_commonword_train_outputs

torchrun --nproc_per_node 1 "$ROOT_DIR/lora_dpo_distributed.py" \
  --config "$ROOT_DIR/configs/training/llama3.2_1b_misa10_fixed_commonword_sft_inner20_bs1_ga256_ep1.yaml" \
  > misa10_fixed_commonword_train_outputs/train.out 2>&1

echo "Finished MISA-10 SFT-attacker training."
echo "Log: $ROOT_DIR/misa10_fixed_commonword_train_outputs/train.out"
echo "Output: /tmp/secalign_runs/misa10_fixed_commonword_sft_inner20_bs1_ga256_epoch1"
