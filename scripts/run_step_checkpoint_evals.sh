#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-sandbox}"
if [[ -z "${CONDA_PREFIX:-}" || "$(basename "$CONDA_PREFIX")" != "$CONDA_ENV" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
fi

BASE_MODEL="${BASE_MODEL:-/home/$USER/models/Meta-Llama-3.1-8B-Instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/secalign_runs/llama31_8b_position_diverse10_commonword_sft_inner20_bs1_ga32_8gpu_epoch1/epoch_step_60}"
SNAPSHOT_SUFFIX="${SNAPSHOT_SUFFIX:-positiondiverse_step60_snapshot}"
EVAL_MODEL="${EVAL_MODEL:-${BASE_MODEL}_${SNAPSHOT_SUFFIX}}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LORA_ALPHA="${LORA_ALPHA:-8}"
GENERAL_UTILITY_BATCH_SIZE="${GENERAL_UTILITY_BATCH_SIZE:-128}"
OPENAI_CONFIG_PATH="${OPENAI_CONFIG_PATH:-data/openai_configs.yaml}"
GEMINI_CONFIG_PATH="${GEMINI_CONFIG_PATH:-data/gemini_configs.yaml}"
GEMINI_JUDGE_MODEL="${GEMINI_JUDGE_MODEL:-gemini-2.5-flash}"
REFRESH_SNAPSHOT="${REFRESH_SNAPSHOT:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

ALPACA_DATA="${ALPACA_DATA:-data/davinci_003_outputs.json}"
SEP_DATA="${SEP_DATA:-data/SEP_dataset_test.json}"
SEP_REFERENCE_DATA="${SEP_REFERENCE_DATA:-data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json}"

ALPACA_ATTACKS=(
  straightforward
  straightforward_before
  ignore
  ignore_before
  completion
  completion_ignore
  completion_llama31_8B
  completion_ignore_llama31_8B
)

SEP_ATTACKS=(
  straightforward
  straightforward_before
  ignore
  ignore_before
  completion
  completion_ignore
  completion_llama31_8B
  completion_ignore_llama31_8B
)

num_samples_args=()
if [[ -n "$NUM_SAMPLES" ]]; then
  num_samples_args=(--num_samples "$NUM_SAMPLES")
fi

required_files=(
  "$BASE_MODEL/config.json"
  "$BASE_MODEL/tokenizer.json"
  "$CHECKPOINT_DIR/adapter_model.safetensors"
  "$CHECKPOINT_DIR/adapter_config.json"
  "$CHECKPOINT_DIR/tokenizer.json"
  "$CHECKPOINT_DIR/tokenizer_config.json"
  "$OPENAI_CONFIG_PATH"
  "$GEMINI_CONFIG_PATH"
  "$ALPACA_DATA"
  "$SEP_DATA"
  "$SEP_REFERENCE_DATA"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done

if [[ "$EVAL_MODEL" != "${BASE_MODEL}_"* ]]; then
  echo "EVAL_MODEL must begin with BASE_MODEL followed by an underscore." >&2
  echo "The eval code infers the base model with model_name_or_path.split('_')[0]." >&2
  echo "BASE_MODEL=$BASE_MODEL" >&2
  echo "EVAL_MODEL=$EVAL_MODEL" >&2
  exit 1
fi

if [[ "$REFRESH_SNAPSHOT" == "1" ]]; then
  echo "Refreshing checkpoint snapshot:"
  echo "  from: $CHECKPOINT_DIR"
  echo "  to:   $EVAL_MODEL"
  rm -rf "$EVAL_MODEL"
  cp -aL "$CHECKPOINT_DIR" "$EVAL_MODEL"
else
  echo "Using existing checkpoint snapshot: $EVAL_MODEL"
fi

find "$EVAL_MODEL" -maxdepth 2 -type f \( \
  -name adapter_model.safetensors -o \
  -name adapter_config.json -o \
  -name tokenizer.json -o \
  -name tokenizer_config.json \
\) -print

echo "[1/4] AlpacaFarm utility"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python test.py \
  --model_name_or_path "$EVAL_MODEL" \
  --attack none \
  --defense none \
  --test_data "$ALPACA_DATA" \
  --openai_config_path "$OPENAI_CONFIG_PATH" \
  --lora_alpha "$LORA_ALPHA" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  "${num_samples_args[@]}"

echo "[2/4] AlpacaFarm ASR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python test.py \
  --model_name_or_path "$EVAL_MODEL" \
  --attack "${ALPACA_ATTACKS[@]}" \
  --defense none \
  --test_data "$ALPACA_DATA" \
  --openai_config_path "$OPENAI_CONFIG_PATH" \
  --gemini_config_path "$GEMINI_CONFIG_PATH" \
  --gemini_judge_model "$GEMINI_JUDGE_MODEL" \
  --lora_alpha "$LORA_ALPHA" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  "${num_samples_args[@]}"

echo "[3/4] SEP ASR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python test.py \
  --model_name_or_path "$EVAL_MODEL" \
  --attack "${SEP_ATTACKS[@]}" \
  --defense none \
  --test_data "$SEP_DATA" \
  --openai_config_path "$OPENAI_CONFIG_PATH" \
  --gemini_config_path "$GEMINI_CONFIG_PATH" \
  --gemini_judge_model "$GEMINI_JUDGE_MODEL" \
  --lora_alpha "$LORA_ALPHA" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  "${num_samples_args[@]}"

echo "[4/4] General utility"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python test_lm_eval.py \
  --model_name_or_path "$EVAL_MODEL" \
  --lora_alpha "$LORA_ALPHA" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --batch_size "$GENERAL_UTILITY_BATCH_SIZE"

echo "All evals finished."
cat "$EVAL_MODEL/summary.tsv"
