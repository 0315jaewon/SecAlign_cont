import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl_by_index(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["sample_index"])] = record
    return records


def attack_tokens(prefix: str, count: int) -> List[str]:
    return [f"{prefix}{idx}>" for idx in range(count)]


def encode(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def strip_eot(text: str) -> str:
    return text.replace("<|eot_id|>", "").strip()


def find_rejected_span(prompt: str, rejected_input_whole: str):
    if rejected_input_whole in prompt:
        start = prompt.find(rejected_input_whole)
        return start, start + len(rejected_input_whole)
    pattern = re.escape(rejected_input_whole.strip()).replace(r"\ ", r"\s+")
    match = re.search(pattern, prompt)
    if match is not None:
        return match.start(), match.end()
    return None


def build_attacked_prompt(sample: dict, active_tokens: List[str]) -> tuple[str, str]:
    prompt = sample["prompt"]
    rejected_span = sample.get("rejected_input_whole") or sample.get("rejected_input")
    if not isinstance(rejected_span, str):
        raise ValueError("Sample is missing rejected_input_whole/rejected_input.")

    suffix = " " + " ".join(active_tokens)
    span = find_rejected_span(prompt, rejected_span)
    if span is None:
        # Fallback mirrors suffix attack semantics but records that exact insertion failed.
        return prompt + suffix, "append_to_prompt_fallback"
    _, end = span
    return prompt[:end] + suffix + prompt[end:], "after_rejected_span"


def load_model_and_tokenizer(model_path: str, base_model: str, args):
    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    tokenizer_path = model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": attack_tokens(
                args.attack_token_prefix, args.num_attack_tokens
            )
        }
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype_map[args.dtype],
    )
    model.resize_token_embeddings(len(tokenizer))

    if is_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading LoRA checkpoint adapters requires peft.") from exc
        model = PeftModel.from_pretrained(model, model_path)

    model.to("cuda")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def build_inputs_embeds(
    input_ids: torch.Tensor,
    embedding,
    attack_token_id_to_offset: Dict[int, int],
    attack_embeds: torch.Tensor,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    ids = input_ids[0]
    hidden_size = embedding.weight.shape[1]
    embeds = torch.empty(
        ids.shape[0],
        hidden_size,
        device=ids.device,
        dtype=model_dtype,
    )

    normal_positions = []
    normal_ids = []
    attack_positions = []
    attack_offsets = []
    for pos, token_id in enumerate(ids.tolist()):
        offset = attack_token_id_to_offset.get(token_id)
        if offset is None:
            normal_positions.append(pos)
            normal_ids.append(token_id)
        else:
            attack_positions.append(pos)
            attack_offsets.append(offset)

    if normal_positions:
        embeds[
            torch.tensor(normal_positions, device=ids.device)
        ] = embedding(torch.tensor(normal_ids, device=ids.device)).to(model_dtype)
    if attack_positions:
        embeds[
            torch.tensor(attack_positions, device=ids.device)
        ] = attack_embeds[torch.tensor(attack_offsets, device=ids.device)].to(model_dtype)
    return embeds.unsqueeze(0)


def response_nll_per_token(
    model,
    tokenizer,
    prompt_ids: List[int],
    response_text: str,
    attack_token_id_to_offset: Dict[int, int],
    attack_embeds: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, int, bool]:
    response_ids = encode(tokenizer, response_text)
    if not response_ids:
        raise ValueError("Response tokenized to zero tokens.")

    truncated = False
    available = max_seq_len - len(prompt_ids)
    if available <= 0:
        raise ValueError(
            f"Prompt length {len(prompt_ids)} exceeds max_seq_len={max_seq_len}."
        )
    if len(response_ids) > available:
        response_ids = response_ids[:available]
        truncated = True

    full_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    input_ids = torch.tensor([full_ids], device=model.device, dtype=torch.long)
    labels_tensor = torch.tensor([labels], device=model.device, dtype=torch.long)

    embedding = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
    inputs_embeds = build_inputs_embeds(
        input_ids,
        embedding,
        attack_token_id_to_offset,
        attack_embeds,
        model_dtype,
    )
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones_like(input_ids),
    )
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels_tensor[:, 1:].contiguous()
    nll_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    token_count = int((shift_labels != -100).sum().detach().cpu())
    if token_count <= 0:
        raise RuntimeError("No response tokens contributed to NLL.")
    return nll_sum / token_count, token_count, truncated


@torch.no_grad()
def generate_with_attack_embeddings(
    model,
    tokenizer,
    prompt_ids: List[int],
    attack_token_id_to_offset: Dict[int, int],
    attack_embeds: torch.Tensor,
    max_new_tokens: int,
) -> str:
    embedding = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
    current_ids = torch.tensor([prompt_ids], device=model.device, dtype=torch.long)
    generated_ids = []

    for _ in range(max_new_tokens):
        inputs_embeds = build_inputs_embeds(
            current_ids,
            embedding,
            attack_token_id_to_offset,
            attack_embeds,
            model_dtype,
        )
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(current_ids),
        )
        next_id = int(torch.argmax(outputs.logits[0, -1]).detach().cpu())
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_id)
        current_ids = torch.cat(
            [current_ids, torch.tensor([[next_id]], device=current_ids.device)],
            dim=1,
        )
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def probe_one_sample(model, tokenizer, sample: dict, idx: int, args):
    embedding = model.get_input_embeddings()
    attack_token_ids = [
        tokenizer.convert_tokens_to_ids(tok)
        for tok in attack_tokens(args.attack_token_prefix, args.num_attack_tokens)
    ]
    active_tokens = attack_tokens(args.attack_token_prefix, args.num_attack_tokens)
    attacked_prompt, insertion_mode = build_attacked_prompt(sample, active_tokens)
    prompt_ids = encode(tokenizer, attacked_prompt)

    init_token_ids = encode(tokenizer, args.attack_init_token)
    if len(init_token_ids) != 1:
        raise RuntimeError(
            f"--attack_init_token must tokenize to one token, got {init_token_ids}."
        )
    init_token_id = init_token_ids[0]
    init_rows = (
        embedding.weight[init_token_id]
        .detach()
        .float()
        .repeat(args.num_attack_tokens, 1)
    )
    attack_embeds = torch.nn.Parameter(
        init_rows.to(device=model.device, dtype=torch.float32)
    )
    attack_token_id_to_offset = {
        token_id: offset for offset, token_id in enumerate(attack_token_ids)
    }
    optimizer = torch.optim.AdamW([attack_embeds], lr=args.attacker_lr, weight_decay=0.0)

    records = []
    generations = []
    first_margin_flip_step = None
    chosen_text = sample["chosen"]
    rejected_text = sample["rejected"]

    for step in range(args.attack_steps + 1):
        with torch.no_grad():
            chosen_nll, chosen_tokens, chosen_truncated = response_nll_per_token(
                model,
                tokenizer,
                prompt_ids,
                chosen_text,
                attack_token_id_to_offset,
                attack_embeds.detach(),
                args.max_seq_len,
            )
        rejected_nll, rejected_tokens, rejected_truncated = response_nll_per_token(
            model,
            tokenizer,
            prompt_ids,
            rejected_text,
            attack_token_id_to_offset,
            attack_embeds,
            args.max_seq_len,
        )
        margin = chosen_nll.detach() - rejected_nll.detach()
        if first_margin_flip_step is None and float(margin.cpu()) > 0:
            first_margin_flip_step = step

        record = {
            "step": step,
            "chosen_nll_per_token": float(chosen_nll.detach().cpu()),
            "rejected_nll_per_token": float(rejected_nll.detach().cpu()),
            "margin_chosen_minus_rejected": float(margin.cpu()),
            "chosen_tokens": chosen_tokens,
            "rejected_tokens": rejected_tokens,
            "chosen_truncated": chosen_truncated,
            "rejected_truncated": rejected_truncated,
            "margin_flipped": bool(float(margin.cpu()) > 0),
        }
        records.append(record)

        if step % args.generate_every == 0:
            generated = generate_with_attack_embeddings(
                model,
                tokenizer,
                prompt_ids,
                attack_token_id_to_offset,
                attack_embeds.detach(),
                args.generation_max_new_tokens,
            )
            generations.append({"step": step, "output": generated})

        print(
            f"sample={idx} step={step}/{args.attack_steps} "
            f"chosen_nll={record['chosen_nll_per_token']:.6f} "
            f"rejected_nll={record['rejected_nll_per_token']:.6f} "
            f"margin={record['margin_chosen_minus_rejected']:.6f} "
            f"flip={record['margin_flipped']}",
            flush=True,
        )

        if step == args.attack_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        rejected_nll.backward()
        optimizer.step()

    return {
        "sample_index": idx,
        "checkpoint_label": args.checkpoint_label,
        "model": args.model,
        "insertion_mode": insertion_mode,
        "num_attack_tokens": args.num_attack_tokens,
        "attack_steps": args.attack_steps,
        "generate_every": args.generate_every,
        "attacker_lr": args.attacker_lr,
        "prompt": sample["prompt"],
        "chosen_input": sample.get("chosen_input"),
        "rejected_input": sample.get("rejected_input"),
        "rejected_input_whole": sample.get("rejected_input_whole"),
        "chosen_reference": strip_eot(sample["chosen"]),
        "rejected_reference": strip_eot(sample["rejected"]),
        "attacked_prompt": attacked_prompt,
        "prompt_tokens": len(prompt_ids),
        "first_margin_flip_step": first_margin_flip_step,
        "final_margin": records[-1]["margin_chosen_minus_rejected"],
        "records": records,
        "generations": generations,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint_label", required=True)
    parser.add_argument(
        "--data",
        default="data/preference_Llama-3.2-1B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca.json",
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_attack_tokens", type=int, default=10)
    parser.add_argument("--attack_steps", type=int, default=100)
    parser.add_argument("--generate_every", type=int, default=10)
    parser.add_argument("--generation_max_new_tokens", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--attacker_lr", type=float, default=5e-4)
    parser.add_argument("--attack_token_prefix", default="<ATTACK_")
    parser.add_argument("--attack_init_token", default="!")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_json(args.data)
    data = data[args.start_index : args.start_index + args.num_samples]
    output_path = Path(args.output_jsonl)
    completed = read_jsonl_by_index(output_path) if args.resume else {}

    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model, args)

    for local_idx, sample in enumerate(
        tqdm(data, desc=f"alpaca probe {args.checkpoint_label}")
    ):
        idx = args.start_index + local_idx
        if idx in completed:
            print(f"Skipping completed sample_index={idx}", flush=True)
            continue
        result = probe_one_sample(model, tokenizer, sample, idx, args)
        append_jsonl(output_path, result)


if __name__ == "__main__":
    main()
