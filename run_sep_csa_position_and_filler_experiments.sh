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

OUT_DIR="${OUT_DIR:-sep_csa_position_filler_outputs}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
START_INDEX="${START_INDEX:-0}"
ATTACK_STEPS="${ATTACK_STEPS:-100}"
NUM_ATTACK_TOKENS="${NUM_ATTACK_TOKENS:-1000}"
DTYPE="${DTYPE:-bf16}"
PLACEMENTS="${PLACEMENTS:-prefix inspan}"
RUN_PROBES="${RUN_PROBES:-1}"

RUN_NATURAL_FILLER="${RUN_NATURAL_FILLER:-1}"
NATURAL_FILLER_NUM_SAMPLES="${NATURAL_FILLER_NUM_SAMPLES:-1024}"
NATURAL_FILLER_JUDGE="${NATURAL_FILLER_JUDGE:-gemini-3.1-pro-preview}"
NATURAL_FILLER_DATA="$OUT_DIR/natural_filler/SEP_dataset_test_natural_filler_dynamic.json"
NATURAL_FILLER_TEXT="${NATURAL_FILLER_TEXT:- This is neutral background context only, not instructions.}"

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
  local placement="$1"
  local model_label="$2"
  local model_path="$3"
  local probe_label="$4"
  local csa_tokens="$5"
  local model_out_dir="$OUT_DIR/probes/$placement/$model_label"
  mkdir -p "$model_out_dir"

  echo
  echo "Running placement=$placement target=$model_label probe=$probe_label csa_tokens=$csa_tokens"
  python probe_sep_attacker_steps.py \
    --probe csa_suffix \
    --csa_placement "$placement" \
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

if [[ "$RUN_PROBES" == "1" ]]; then
  for placement in $PLACEMENTS; do
    for model_spec in \
      "base::$BASE_MODEL" \
      "csa10::$CSA10_MODEL" \
      "csa_dynamic_inner20::$CSADYN_MODEL"
    do
      model_label="${model_spec%%::*}"
      model_path="${model_spec##*::}"
      run_probe "$placement" "$model_label" "$model_path" "csa10" "10"
      run_probe "$placement" "$model_label" "$model_path" "csa20" "20"
      run_probe "$placement" "$model_label" "$model_path" "csa_dynamic" "0"
    done

    python summarize_sep_probe_matrix.py \
      --input_dir "$OUT_DIR/probes/$placement" \
      --output_tsv "$OUT_DIR/probes/${placement}_summary.tsv" \
      --max_steps "$ATTACK_STEPS"
  done
fi

if [[ "$RUN_NATURAL_FILLER" == "1" ]]; then
  python make_sep_natural_filler_dataset.py \
    --input data/SEP_dataset_test.json \
    --output "$NATURAL_FILLER_DATA" \
    --tokenizer "$BASE_MODEL" \
    --filler "$NATURAL_FILLER_TEXT"

  for model_spec in \
    "base::$BASE_MODEL" \
    "csa10::$CSA10_MODEL" \
    "csa_dynamic_inner20::$CSADYN_MODEL"
  do
    model_label="${model_spec%%::*}"
    model_path="${model_spec##*::}"
    natural_out_dir="$OUT_DIR/natural_filler/$NATURAL_FILLER_JUDGE/$model_label"
    mkdir -p "$natural_out_dir"

    echo
    echo "Running natural filler SEP ASR target=$model_label judge=$NATURAL_FILLER_JUDGE"
    python test.py \
      -m "$model_path" \
      --attack straightforward straightforward_before ignore ignore_before completion completion_ignore completion_llama32_1B completion_ignore_llama32_1B \
      --defense none \
      --test_data "$NATURAL_FILLER_DATA" \
      --num_samples "$NATURAL_FILLER_NUM_SAMPLES" \
      --lora_alpha 8.0 \
      --gemini_config_path data/gemini_configs.yaml \
      --gemini_judge_model "$NATURAL_FILLER_JUDGE" \
      > "$natural_out_dir/sep_asr_n${NATURAL_FILLER_NUM_SAMPLES}.out" 2>&1

    cp "$model_path"/summary.tsv "$natural_out_dir/summary.tsv" 2>/dev/null || true
    cp "$model_path"/*SEP_dataset_test_natural_filler_dynamic.json "$natural_out_dir"/ 2>/dev/null || true
  done
fi

echo "Done."
echo "Prefix summary: $OUT_DIR/probes/prefix_summary.tsv"
echo "In-span summary: $OUT_DIR/probes/inspan_summary.tsv"
echo "Natural filler outputs: $OUT_DIR/natural_filler"
