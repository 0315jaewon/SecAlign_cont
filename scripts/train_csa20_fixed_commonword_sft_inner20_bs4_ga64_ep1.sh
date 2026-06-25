#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p csa20_fixed_commonword_train_outputs

torchrun --nproc_per_node 1 "$ROOT_DIR/lora_dpo_distributed.py" \
  --config "$ROOT_DIR/configs/training/llama3.2_1b_csa20_fixed_commonword_sft_inner20_bs4_ga64_ep1.yaml" \
  > csa20_fixed_commonword_train_outputs/train.out 2>&1

echo "Finished CSA-20 fixed common-word SFT-attacker training."
echo "Log: $ROOT_DIR/csa20_fixed_commonword_train_outputs/train.out"
echo "Output: /tmp/secalign_runs/csa20_fixed_commonword_sft_inner20_bs4_ga64_epoch1"
