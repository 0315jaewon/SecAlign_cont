#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-/home/gcpuser/models/Llama-3.2-1B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/gcpuser/models/csa10-checkpoint-probing/csa10_checkpointed_banginit_sft_rejected_inner20_epoch1}"
DATA_FILE="${DATA_FILE:-data/preference_Llama-3.2-1B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca.json}"
OUT_DIR="${OUT_DIR:-alpaca_csa10_checkpoint_margin_probe_outputs}"
STEPS="${STEPS:-15 30 45 60 75 90 105 120 135 150}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
START_INDEX="${START_INDEX:-0}"
ATTACK_STEPS="${ATTACK_STEPS:-100}"
GENERATE_EVERY="${GENERATE_EVERY:-10}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-256}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
DTYPE="${DTYPE:-bf16}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$BASE_MODEL/config.json" ]]; then
  echo "Missing base model at $BASE_MODEL/config.json"
  echo "Run: huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir $BASE_MODEL"
  exit 1
fi

for step in $STEPS; do
  model_dir="$CHECKPOINT_ROOT/epoch_step_${step}"
  if [[ ! -f "$model_dir/adapter_model.safetensors" ]]; then
    echo "Missing checkpoint adapter: $model_dir/adapter_model.safetensors"
    exit 1
  fi

  echo
  echo "Running Alpaca CSA-10 margin probe for checkpoint step ${step}"
  python probe_alpaca_checkpoint_csa_margin.py \
    --model "$model_dir" \
    --base_model "$BASE_MODEL" \
    --checkpoint_label "step_${step}" \
    --data "$DATA_FILE" \
    --output_jsonl "$OUT_DIR/step_${step}.jsonl" \
    --num_samples "$NUM_SAMPLES" \
    --start_index "$START_INDEX" \
    --num_attack_tokens 10 \
    --attack_steps "$ATTACK_STEPS" \
    --generate_every "$GENERATE_EVERY" \
    --generation_max_new_tokens "$GENERATION_MAX_NEW_TOKENS" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --dtype "$DTYPE" \
    --resume \
    > "$OUT_DIR/step_${step}.out" 2>&1
done

python summarize_alpaca_checkpoint_csa_margin.py \
  --input_dir "$OUT_DIR" \
  --output_tsv "$OUT_DIR/summary.tsv" \
  --max_steps "$ATTACK_STEPS"

echo "Done. Summary: $OUT_DIR/summary.tsv"
