#!/usr/bin/env bash
set -euo pipefail

mkdir -p /home/gcpuser/models
mkdir -p probe_outputs

require_hf() {
  if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "ERROR: Neither hf nor huggingface-cli is available. Run: conda activate secalign"
    exit 1
  fi
}

hf_download() {
  local repo_id="$1"
  local local_dir="$2"
  require_hf
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo_id" --local-dir "$local_dir"
  else
    huggingface-cli download "$repo_id" --local-dir "$local_dir"
  fi
}

ensure_model_dir() {
  local label="$1"
  local path="$2"
  local repo_id="$3"
  local required_file="$4"

  if [ -e "$path/$required_file" ]; then
    echo "Found $label at $path"
    return 0
  fi

  if [ -z "$repo_id" ]; then
    echo "ERROR: Missing $label at $path and no fallback repo is configured."
    echo "Expected file: $path/$required_file"
    exit 1
  fi

  echo "Missing $label at $path; downloading $repo_id ..."
  mkdir -p "$path"
  hf_download "$repo_id" "$path"

  if [ ! -e "$path/$required_file" ]; then
    echo "ERROR: Downloaded $repo_id but still missing $path/$required_file"
    exit 1
  fi
}

echo "Checking required model directories..."

ensure_model_dir \
  "base Llama-3.2-1B-Instruct" \
  "/home/gcpuser/models/Llama-3.2-1B-Instruct" \
  "meta-llama/Llama-3.2-1B-Instruct" \
  "config.json"

ensure_model_dir \
  "CSA-10 continuous suffix adapter" \
  "/home/gcpuser/models/Llama-3.2-1B-Instruct_secalign_adapter" \
  "tawer12/llama-3.2-1b-secalign-adapter" \
  "adapter_config.json"

ensure_model_dir \
  "MISA-1000 adapter" \
  "/home/gcpuser/models/Llama-3.2-1B-Instruct_misa_sft_tokens1000_inner20_epoch1" \
  "tawer12/Llama-3.2-1B-Instruct_misa_sft_tokens1000_inner20_epoch1" \
  "adapter_config.json"

ensure_model_dir \
  "Meta SecAlign 1B adapter" \
  "/home/gcpuser/models/Llama-3.2-1B-Instruct_epoch1_adapter" \
  "tawer12/secalign-adv-dpo-1b" \
  "adapter_config.json"

echo "Model checks complete."
echo "Starting SEP attacker-step probe grid."

python probe_sep_attacker_steps.py \
  --probe csa_suffix \
  --model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --model_label base \
  --target_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --target_cache probe_outputs/sep_base_targets_n8.jsonl \
  --output_jsonl probe_outputs/base_csa10_probe.jsonl \
  --num_samples 8 \
  --num_attack_tokens 10 \
  --csa_tokens 10 \
  --attack_steps 100 \
  --attacker_lr 5e-4 \
  --attack_init_token "!" \
  --dtype bf16 \
  > probe_outputs/base_csa10_probe.out 2>&1

python probe_sep_attacker_steps.py \
  --probe csa_suffix \
  --model /home/gcpuser/models/Llama-3.2-1B-Instruct_secalign_adapter \
  --base_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --model_label csa10 \
  --target_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --target_cache probe_outputs/sep_base_targets_n8.jsonl \
  --output_jsonl probe_outputs/csa10_csa10_probe.jsonl \
  --num_samples 8 \
  --num_attack_tokens 10 \
  --csa_tokens 10 \
  --attack_steps 100 \
  --attacker_lr 5e-4 \
  --attack_init_token "!" \
  --dtype bf16 \
  > probe_outputs/csa10_csa10_probe.out 2>&1

python probe_sep_attacker_steps.py \
  --probe csa_suffix \
  --model /home/gcpuser/models/Llama-3.2-1B-Instruct_misa_sft_tokens1000_inner20_epoch1 \
  --base_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --model_label misa1000 \
  --target_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --target_cache probe_outputs/sep_base_targets_n8.jsonl \
  --output_jsonl probe_outputs/misa1000_csa10_probe.jsonl \
  --num_samples 8 \
  --num_attack_tokens 10 \
  --csa_tokens 10 \
  --attack_steps 100 \
  --attacker_lr 5e-4 \
  --attack_init_token "!" \
  --dtype bf16 \
  > probe_outputs/misa1000_csa10_probe.out 2>&1

python probe_sep_attacker_steps.py \
  --probe csa_suffix \
  --model /home/gcpuser/models/Llama-3.2-1B-Instruct_epoch1_adapter \
  --base_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --model_label meta_secalign \
  --target_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --target_cache probe_outputs/sep_base_targets_n8.jsonl \
  --output_jsonl probe_outputs/meta_secalign_csa10_probe.jsonl \
  --num_samples 8 \
  --num_attack_tokens 10 \
  --csa_tokens 10 \
  --attack_steps 100 \
  --attacker_lr 5e-4 \
  --attack_init_token "!" \
  --dtype bf16 \
  > probe_outputs/meta_secalign_csa10_probe.out 2>&1

python probe_sep_attacker_steps.py \
  --probe misa_span \
  --model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --model_label base \
  --target_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --target_cache probe_outputs/sep_base_targets_n8.jsonl \
  --output_jsonl probe_outputs/base_misa1000_probe.jsonl \
  --num_samples 8 \
  --num_attack_tokens 1000 \
  --attack_steps 100 \
  --attacker_lr 5e-4 \
  --dtype bf16 \
  > probe_outputs/base_misa1000_probe.out 2>&1

python probe_sep_attacker_steps.py \
  --probe misa_span \
  --model /home/gcpuser/models/Llama-3.2-1B-Instruct_secalign_adapter \
  --base_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --model_label csa10 \
  --target_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --target_cache probe_outputs/sep_base_targets_n8.jsonl \
  --output_jsonl probe_outputs/csa10_misa1000_probe.jsonl \
  --num_samples 8 \
  --num_attack_tokens 1000 \
  --attack_steps 100 \
  --attacker_lr 5e-4 \
  --dtype bf16 \
  > probe_outputs/csa10_misa1000_probe.out 2>&1

python probe_sep_attacker_steps.py \
  --probe misa_span \
  --model /home/gcpuser/models/Llama-3.2-1B-Instruct_misa_sft_tokens1000_inner20_epoch1 \
  --base_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --model_label misa1000 \
  --target_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --target_cache probe_outputs/sep_base_targets_n8.jsonl \
  --output_jsonl probe_outputs/misa1000_misa1000_probe.jsonl \
  --num_samples 8 \
  --num_attack_tokens 1000 \
  --attack_steps 100 \
  --attacker_lr 5e-4 \
  --dtype bf16 \
  > probe_outputs/misa1000_misa1000_probe.out 2>&1

python probe_sep_attacker_steps.py \
  --probe misa_span \
  --model /home/gcpuser/models/Llama-3.2-1B-Instruct_epoch1_adapter \
  --base_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --model_label meta_secalign \
  --target_model /home/gcpuser/models/Llama-3.2-1B-Instruct \
  --target_cache probe_outputs/sep_base_targets_n8.jsonl \
  --output_jsonl probe_outputs/meta_secalign_misa1000_probe.jsonl \
  --num_samples 8 \
  --num_attack_tokens 1000 \
  --attack_steps 100 \
  --attacker_lr 5e-4 \
  --dtype bf16 \
  > probe_outputs/meta_secalign_misa1000_probe.out 2>&1

echo "Probe grid complete."
