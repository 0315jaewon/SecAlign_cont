#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
EFFECTIVE_BATCH=$((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM))

CONFIG_FILE="${CONFIG_FILE:-configs/training/llama3.1_8b_position_diverse10_commonword_sft_inner20_projectedr0p12_bs1_ga64_4gpu_ep1.yaml}"
CACHE_DIR="${CACHE_DIR:-/home/$USER/models/Meta-Llama-3.1-8B-Instruct}"
DATA_FILE="${DATA_FILE:-$ROOT_DIR/data/preference_Llama-3.1-8B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/secalign_runs/llama31_8b_position_diverse10_commonword_sft_inner20_projectedr0p12_bs1_ga64_4gpu_epoch1}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/position_diverse10_commonword_8b_inner20_projectedr0p12_train_outputs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train.out}"

EPOCHS="${EPOCHS:-1}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-null}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
LR="${LR:-0.00016}"
ATTACKER_LR="${ATTACKER_LR:-0.0005}"
ATTACK_INNER_STEPS="${ATTACK_INNER_STEPS:-20}"
ATTACK_L2_RADIUS="${ATTACK_L2_RADIUS:-0.12}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-10}"
CHECKPOINT_KEEP_LAST_N_STEPS="${CHECKPOINT_KEEP_LAST_N_STEPS:-1}"

if [[ "$EFFECTIVE_BATCH" -ne 256 ]]; then
  echo "Expected effective batch 256, got ${EFFECTIVE_BATCH}." >&2
  echo "effective_batch = NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM" >&2
  exit 1
fi

for required_file in \
  "$CACHE_DIR/model-00001-of-00004.safetensors" \
  "$CACHE_DIR/model-00002-of-00004.safetensors" \
  "$CACHE_DIR/model-00003-of-00004.safetensors" \
  "$CACHE_DIR/model-00004-of-00004.safetensors" \
  "$CACHE_DIR/tokenizer.json" \
  "$CACHE_DIR/original/tokenizer.model" \
  "$DATA_FILE" \
  "$CONFIG_FILE"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "Launching Llama-3.1-8B position-diverse projected SFT-attacker training, inner20 radius=0.12"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "CACHE_DIR=$CACHE_DIR"
echo "DATA_FILE=$DATA_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "GRAD_ACCUM=$GRAD_ACCUM"
echo "EFFECTIVE_BATCH=$EFFECTIVE_BATCH"
echo "ATTACKER_LR=$ATTACKER_LR"
echo "ATTACK_INNER_STEPS=$ATTACK_INNER_STEPS"
echo "ATTACK_L2_RADIUS=$ATTACK_L2_RADIUS"
echo "MAX_STEPS_PER_EPOCH=$MAX_STEPS_PER_EPOCH"
echo "CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS"
echo "CHECKPOINT_KEEP_LAST_N_STEPS=$CHECKPOINT_KEEP_LAST_N_STEPS"
echo

python -m torchtune._cli.tune run \
  --nnodes 1 \
  --nproc_per_node "$NPROC_PER_NODE" \
  lora_dpo_distributed.py \
  --config "$CONFIG_FILE" \
  "cache_dir=$CACHE_DIR" \
  "checkpointer.checkpoint_dir=$CACHE_DIR" \
  "dataset.data_files=$DATA_FILE" \
  "output_dir=$OUTPUT_DIR" \
  "epochs=$EPOCHS" \
  "max_steps_per_epoch=$MAX_STEPS_PER_EPOCH" \
  "batch_size=$BATCH_SIZE" \
  "gradient_accumulation_steps=$GRAD_ACCUM" \
  "tokenizer.max_seq_len=$MAX_SEQ_LEN" \
  "optimizer.lr=$LR" \
  "attacker_optimizer.lr=$ATTACKER_LR" \
  "attack_inner_steps=$ATTACK_INNER_STEPS" \
  "attack_l2_radius=$ATTACK_L2_RADIUS" \
  "checkpoint_every_n_steps=$CHECKPOINT_EVERY_N_STEPS" \
  "checkpoint_keep_last_n_steps=$CHECKPOINT_KEEP_LAST_N_STEPS" \
  "save_adapter_weights_only=True" \
  "enable_attack_tokens=True" \
  "enable_attack_inner_loop=True" \
  "attacker_objective=sft_rejected" \
  "attack_token_mode=random_injection_gaps" \
  "attack_tokens_per_sample=null" \
  "log_attack_token_diagnostics=False" \
  "init_lora_before_fsdp=True" \
  "load_base_model_before_fsdp=True" \
  "keep_attack_embedding_unsharded=True" \
  2>&1 | tee "$LOG_FILE"

echo "Finished Llama-3.1-8B position-diverse projected SFT-attacker inner20 radius=0.12 training."
echo "Log: $LOG_FILE"
echo "Output: $OUTPUT_DIR"
