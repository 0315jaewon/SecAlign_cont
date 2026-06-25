#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p csa10_may3_dpo_commonword_train_outputs

torchrun --nproc_per_node 1 "$ROOT_DIR/lora_dpo_distributed.py" \
  --config "$ROOT_DIR/configs/training/llama3.2_1b_csa10_fixed_commonword_dpo_inner20_bs4_ga64_ep1.yaml" \
  > csa10_may3_dpo_commonword_train_outputs/train.out 2>&1

echo "Finished DPO-attacker CSA-10 fixed common-word training."
echo "Log: $ROOT_DIR/csa10_may3_dpo_commonword_train_outputs/train.out"
echo "Output: /tmp/secalign_runs/csa10_may3_dpo_commonword_inner20_bs4_ga64_epoch1"
