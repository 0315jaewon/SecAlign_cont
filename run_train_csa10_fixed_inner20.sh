#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p csa10_fixed_train_outputs

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/helpers/llama3.2_1B_lora.yaml}" \
DATA_FILE="${DATA_FILE:-$ROOT_DIR/data/preference_Llama-3.2-1B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca.json}" \
CACHE_DIR="${CACHE_DIR:-/home/gcpuser/models/Llama-3.2-1B-Instruct}" \
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/secalign_runs/csa10_fixed_sft_inner20_epoch1}" \
EPOCHS="${EPOCHS:-1}" \
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-150}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
GRAD_ACCUM="${GRAD_ACCUM:-128}" \
NUM_ATTACK_TOKENS="10" \
ATTACK_TOKENS_PER_SAMPLE="10" \
ATTACK_TOKEN_MODE="suffix" \
ATTACK_INNER_STEPS="${ATTACK_INNER_STEPS:-20}" \
RESET_ATTACK_TOKENS_EACH_BATCH="True" \
ATTACKER_OBJECTIVE="${ATTACKER_OBJECTIVE:-sft_rejected}" \
LOG_ATTACK_TOKEN_DIAGNOSTICS="${LOG_ATTACK_TOKEN_DIAGNOSTICS:-True}" \
bash run_attack_dpo.sh
