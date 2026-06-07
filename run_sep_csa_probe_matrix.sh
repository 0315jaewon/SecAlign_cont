#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BASE_HF="${BASE_HF:-meta-llama/Llama-3.2-1B-Instruct}"
CSA10_HF="${CSA10_HF:-tawer12/llama3.2-1b-csa-10-sft-inner20-ep1}"
CSADYN_HF="${CSADYN_HF:-tawer12/llama3.2-1b-csa-dynamic-sft-inner20-ep1}"

BASE_MODEL="${BASE_MODEL:-/home/gcpuser/models/Llama-3.2-1B-Instruct}"
CSA10_MODEL="${CSA10_MODEL:-/home/gcpuser/models/Llama-3.2-1B-Instruct_csa_10_sft_inner20_ep1}"
CSADYN_MODEL="${CSADYN_MODEL:-/home/gcpuser/models/Llama-3.2-1B-Instruct_csa_dynamic_sft_inner20_ep1}"

OUT_DIR="${OUT_DIR:-sep_csa_probe_matrix_outputs}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
START_INDEX="${START_INDEX:-0}"
ATTACK_STEPS="${ATTACK_STEPS:-100}"
NUM_ATTACK_TOKENS="${NUM_ATTACK_TOKENS:-1000}"
DTYPE="${DTYPE:-bf16}"

mkdir -p /home/gcpuser/models "$OUT_DIR/target_cache"

download_if_missing() {
  local hf_repo="$1"
  local local_dir="$2"
  local required_file="$3"
  if [[ -f "$local_dir/$required_file" ]]; then
    echo "Found $local_dir/$required_file"
    return
  fi
  echo "Downloading $hf_repo to $local_dir"
  huggingface-cli download "$hf_repo" --local-dir "$local_dir"
}

download_if_missing "$BASE_HF" "$BASE_MODEL" "config.json"
download_if_missing "$CSA10_HF" "$CSA10_MODEL" "adapter_model.safetensors"
download_if_missing "$CSADYN_HF" "$CSADYN_MODEL" "adapter_model.safetensors"

run_probe() {
  local model_label="$1"
  local model_path="$2"
  local probe_label="$3"
  local csa_tokens="$4"
  local model_out_dir="$OUT_DIR/$model_label"
  mkdir -p "$model_out_dir"

  echo
  echo "Running target=$model_label probe=$probe_label csa_tokens=$csa_tokens"
  python probe_sep_attacker_steps.py \
    --probe csa_suffix \
    --model "$model_path" \
    --model_label "$model_label" \
    --base_model "$BASE_MODEL" \
    --test_data data/SEP_dataset_test.json \
    --output_jsonl "$model_out_dir/$probe_label.jsonl" \
    --target_cache "$OUT_DIR/target_cache/${model_label}.jsonl" \
    --num_samples "$NUM_SAMPLES" \
    --start_index "$START_INDEX" \
    --num_attack_tokens "$NUM_ATTACK_TOKENS" \
    --csa_tokens "$csa_tokens" \
    --attack_steps "$ATTACK_STEPS" \
    --dtype "$DTYPE" \
    --resume \
    > "$model_out_dir/$probe_label.out" 2>&1
}

for model_spec in \
  "base::$BASE_MODEL" \
  "csa10::$CSA10_MODEL" \
  "csa_dynamic_inner20::$CSADYN_MODEL"
do
  model_label="${model_spec%%::*}"
  model_path="${model_spec##*::}"
  run_probe "$model_label" "$model_path" "csa10" "10"
  run_probe "$model_label" "$model_path" "csa20" "20"
  run_probe "$model_label" "$model_path" "csa_dynamic" "0"
done

python summarize_sep_probe_matrix.py \
  --input_dir "$OUT_DIR" \
  --output_tsv "$OUT_DIR/summary.tsv" \
  --max_steps "$ATTACK_STEPS"

echo "Done. Summary: $OUT_DIR/summary.tsv"
