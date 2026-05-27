#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL=/home/gcpuser/models/Llama-3.2-1B-Instruct
CSA_MODEL=/home/gcpuser/models/Llama-3.2-1B-Instruct_secalign_adapter
MISA_MODEL=/home/gcpuser/models/Llama-3.2-1B-Instruct_misa_sft_tokens1000_inner20_epoch1
META_MODEL=/home/gcpuser/models/Llama-3.2-1B-Instruct_epoch1_adapter

CSA_DATASET=data/optimized_sep_csa10_base_llama32_1b_n1024
MISA_DATASET=data/optimized_sep_misa1000_base_llama32_1b_n1024

OUT_DIR=optimized_sep_eval_outputs
SUMMARY_TSV="${OUT_DIR}/summary.tsv"
mkdir -p "${OUT_DIR}"

check_file() {
  local label="$1"
  local path="$2"
  if [ ! -e "$path" ]; then
    echo "ERROR: missing ${label}: ${path}"
    exit 1
  fi
  echo "Found ${label}: ${path}"
}

check_file "base model config" "${BASE_MODEL}/config.json"
check_file "CSA adapter config" "${CSA_MODEL}/adapter_config.json"
check_file "MISA adapter config" "${MISA_MODEL}/adapter_config.json"
check_file "Meta SecAlign adapter config" "${META_MODEL}/adapter_config.json"
check_file "CSA optimized records" "${CSA_DATASET}/records.jsonl"
check_file "CSA optimized manifest" "${CSA_DATASET}/manifest.json"
check_file "MISA optimized records" "${MISA_DATASET}/records.jsonl"
check_file "MISA optimized manifest" "${MISA_DATASET}/manifest.json"

run_eval() {
  local model_label="$1"
  local model_path="$2"
  local dataset_label="$3"
  local dataset_path="$4"
  local extra_args=()
  if [ "$model_path" != "$BASE_MODEL" ]; then
    extra_args+=(--base_model "$BASE_MODEL")
  fi

  echo "Running model=${model_label} dataset=${dataset_label}"
  python evaluate_optimized_sep_dataset.py \
    --dataset_dir "$dataset_path" \
    --model "$model_path" \
    --model_label "$model_label" \
    --attack_label "$dataset_label" \
    --output_json "${OUT_DIR}/${model_label}_${dataset_label}.json" \
    --summary_tsv "$SUMMARY_TSV" \
    --max_new_tokens 512 \
    --dtype bf16 \
    --gemini_judge_model gemini-3.1-pro-preview \
    "${extra_args[@]}" \
    > "${OUT_DIR}/${model_label}_${dataset_label}.out" 2>&1
}

run_eval base "$BASE_MODEL" csa10_optimized "$CSA_DATASET"
run_eval meta_secalign "$META_MODEL" csa10_optimized "$CSA_DATASET"
run_eval csa10 "$CSA_MODEL" csa10_optimized "$CSA_DATASET"
run_eval misa1000 "$MISA_MODEL" csa10_optimized "$CSA_DATASET"

run_eval base "$BASE_MODEL" misa1000_optimized "$MISA_DATASET"
run_eval meta_secalign "$META_MODEL" misa1000_optimized "$MISA_DATASET"
run_eval csa10 "$CSA_MODEL" misa1000_optimized "$MISA_DATASET"
run_eval misa1000 "$MISA_MODEL" misa1000_optimized "$MISA_DATASET"

python summarize_optimized_sep_eval.py \
  --summary_tsv "$SUMMARY_TSV" \
  --output_dir "$OUT_DIR" \
  --models base meta_secalign csa10 misa1000 \
  --attacks csa10_optimized misa1000_optimized \
  > "${OUT_DIR}/combined_summary.out" 2>&1

echo "Optimized SEP eval grid complete. Summary: ${SUMMARY_TSV}"
