# inner10_epoch0

Llama-3.1-8B position-diverse commonword SFT-attacker LoRA, 10 attacker inner steps, final epoch-0 checkpoint.

- Model snapshot: `/home/jaewonc/models/Meta-Llama-3.1-8B-Instruct_positiondiverse_inner10_epoch0_snapshot`
- Training Script: `repro/train_llama31_8b_position_diverse10_commonword_sft_inner10_bs1_ga64_4gpu_ep1.sh`
- Training Config: `repro/llama3.1_8b_position_diverse10_commonword_sft_inner10_bs1_ga64_4gpu_ep1.yaml`

## Example Launch

```bash
NPROC_PER_NODE=8 GRAD_ACCUM=32 CACHE_DIR=/home/$USER/models/Meta-Llama-3.1-8B-Instruct OUTPUT_DIR=/tmp/secalign_runs/llama31_8b_position_diverse10_commonword_sft_inner10_bs1_ga32_8gpu_epoch1 LOG_DIR=$PWD/position_diverse10_commonword_8b_inner10_train_outputs nohup scripts/train_llama31_8b_position_diverse10_commonword_sft_inner10_bs1_ga64_4gpu_ep1.sh > position_diverse10_commonword_8b_inner10_train_outputs/nohup.out 2>&1 &
```

The launch script enforces effective batch size 256 via `NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM` and records the exact runtime overrides in its training log.
