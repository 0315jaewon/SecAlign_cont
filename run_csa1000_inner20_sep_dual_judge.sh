#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HF_MODEL="tawer12/llama3.2-1b-csa-1000-sft-inner20-ep1"
MODEL="/home/gcpuser/models/Llama-3.2-1B-Instruct_csa_1000_sft_inner20_ep1"
OUT_DIR="csa1000_inner20_sep_eval_outputs"
SEP_N="1024"
GEMINI_PRO_MODEL="gemini-3.1-pro-preview"
GEMINI_FLASH_MODEL="gemini-2.5-flash"

SEP_ATTACKS=(
  straightforward
  straightforward_before
  ignore
  ignore_before
  completion
  completion_ignore
  completion_llama32_1B
  completion_ignore_llama32_1B
)

check_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
  echo "Found ${label}: ${path}"
}

mkdir -p /home/gcpuser/models
mkdir -p "$OUT_DIR/$GEMINI_PRO_MODEL" "$OUT_DIR/$GEMINI_FLASH_MODEL"

check_file "base model config" "/home/gcpuser/models/Llama-3.2-1B-Instruct/config.json"
check_file "Gemini config" "data/gemini_configs.yaml"
check_file "SEP dataset" "data/SEP_dataset_test.json"

echo "Downloading ${HF_MODEL} to ${MODEL}"
rm -rf "$MODEL"
huggingface-cli download "$HF_MODEL" --local-dir "$MODEL"

check_file "CSA-1000 inner20 adapter config" "$MODEL/adapter_config.json"
check_file "CSA-1000 inner20 adapter weights" "$MODEL/adapter_model.safetensors"
check_file "CSA-1000 inner20 tokenizer config" "$MODEL/tokenizer_config.json"

echo "Step 1/2: Running SEP n=${SEP_N} with ${GEMINI_PRO_MODEL}"
python test.py \
  -m "$MODEL" \
  --attack "${SEP_ATTACKS[@]}" \
  --defense none \
  --test_data data/SEP_dataset_test.json \
  --num_samples "$SEP_N" \
  --lora_alpha 8.0 \
  --gemini_config_path data/gemini_configs.yaml \
  --gemini_judge_model "$GEMINI_PRO_MODEL" \
  > "$OUT_DIR/$GEMINI_PRO_MODEL/sep_asr_n${SEP_N}.out" 2>&1

cp "$MODEL/summary.tsv" "$OUT_DIR/$GEMINI_PRO_MODEL/summary.tsv"
cp "$MODEL"/*SEP_dataset_test.json "$OUT_DIR/$GEMINI_PRO_MODEL/"

echo "Step 2/2: Rejudging cached SEP outputs with ${GEMINI_FLASH_MODEL}"
python rejudge_sep_outputs.py \
  -m "$MODEL" \
  --test_data data/SEP_dataset_test.json \
  --num_samples "$SEP_N" \
  --attacks "${SEP_ATTACKS[@]}" \
  --lora_alpha 8.0 \
  --gemini_config_path data/gemini_configs.yaml \
  --gemini_judge_model "$GEMINI_FLASH_MODEL" \
  --output_dir "$OUT_DIR/$GEMINI_FLASH_MODEL" \
  > "$OUT_DIR/$GEMINI_FLASH_MODEL/sep_rejudge_n${SEP_N}.out" 2>&1

echo "CSA-1000 inner20 SEP dual-judge eval complete."
echo "Pro summary: $OUT_DIR/$GEMINI_PRO_MODEL/summary.tsv"
echo "Flash summary: $OUT_DIR/$GEMINI_FLASH_MODEL/summary.tsv"
