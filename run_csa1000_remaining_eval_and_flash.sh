#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/home/gcpuser/models/Llama-3.2-1B-Instruct_csa_variable_1000_banginit_sft_rejected_inner20_epoch1}"
OUT_DIR="${OUT_DIR:-csa1000_eval_outputs}"
SEP_N="${SEP_N:-1024}"
GEMINI_FLASH_MODEL="${GEMINI_FLASH_MODEL:-gemini-2.5-flash}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-512}"

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

ALPACAFARM_ATTACKS=(
  none
  straightforward
  straightforward_before
  ignore
  ignore_before
  completion
  completion_ignore
)

mkdir -p "$OUT_DIR"

check_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
  echo "Found ${label}: ${path}"
}

check_file "CSA-1000 adapter config" "$MODEL/adapter_config.json"
check_file "CSA-1000 adapter weights" "$MODEL/adapter_model.safetensors"
check_file "Gemini config" "data/gemini_configs.yaml"
check_file "OpenAI config" "data/openai_configs.yaml"
check_file "SEP dataset" "data/SEP_dataset_test.json"
check_file "AlpacaFarm dataset" "data/davinci_003_outputs.json"

echo "Step 1/3: Rejudging cached CSA-1000 SEP outputs with ${GEMINI_FLASH_MODEL}"
python rejudge_sep_outputs.py \
  -m "$MODEL" \
  --test_data data/SEP_dataset_test.json \
  --num_samples "$SEP_N" \
  --attacks "${SEP_ATTACKS[@]}" \
  --lora_alpha 8.0 \
  --gemini_config_path data/gemini_configs.yaml \
  --gemini_judge_model "$GEMINI_FLASH_MODEL" \
  --output_dir "$OUT_DIR/sep_rejudge_${GEMINI_FLASH_MODEL}_n${SEP_N}" \
  > "$OUT_DIR/sep_rejudge_${GEMINI_FLASH_MODEL}_n${SEP_N}.out" 2>&1

echo "Step 2/3: Running CSA-1000 AlpacaFarm ASR and AlpacaEval utility"
python test.py \
  -m "$MODEL" \
  --attack "${ALPACAFARM_ATTACKS[@]}" \
  --defense none \
  --test_data data/davinci_003_outputs.json \
  --lora_alpha 8.0 \
  --openai_config_path data/openai_configs.yaml \
  > "$OUT_DIR/alpacafarm.out" 2>&1

echo "Step 3/3: Running CSA-1000 lm-eval utility"
python test_lm_eval.py \
  -m "$MODEL" \
  --lora_alpha 8.0 \
  --tasks all \
  --batch_size "$LM_EVAL_BATCH_SIZE" \
  > "$OUT_DIR/lm_eval.out" 2>&1

echo "CSA-1000 remaining eval pipeline complete."
echo "Model summary: $MODEL/summary.tsv"
echo "Flash SEP summary: $OUT_DIR/sep_rejudge_${GEMINI_FLASH_MODEL}_n${SEP_N}/summary.tsv"
