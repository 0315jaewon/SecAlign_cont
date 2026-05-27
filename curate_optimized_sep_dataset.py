import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def completed_indices(records_path: Path) -> set[int]:
    if not records_path.exists():
        return set()
    done = set()
    with open(records_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            done.add(json.loads(line)["sample_index"])
    return done


def read_jsonl_by_index(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    records = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["sample_index"]] = record
    return records


def build_chat_prompt(tokenizer, instruction: str, input_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": instruction},
            {"role": "input", "content": input_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def encode(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def attack_tokens(prefix: str, count: int) -> List[str]:
    return [f"{prefix}{idx}>" for idx in range(count)]


def prepare_csa_suffix(
    item: dict,
    tokenizer,
    embedding,
    args,
    attack_token_ids: List[int],
    init_token_id: int,
) -> Tuple[str, List[int], torch.Tensor]:
    active_count = min(args.num_attack_tokens, args.csa_tokens)
    active_tokens = attack_tokens(args.attack_token_prefix, active_count)
    optimized_input = (
        item["input"].rstrip()
        + " "
        + item["injection"].strip()
        + " "
        + " ".join(active_tokens)
    ).strip()
    prompt = build_chat_prompt(tokenizer, item["instruction"], optimized_input)
    prompt_ids = encode(tokenizer, prompt)
    init_rows = embedding.weight[init_token_id].detach().float().repeat(active_count, 1)
    return optimized_input, prompt_ids, init_rows


def prepare_misa_span(
    item: dict,
    tokenizer,
    embedding,
    args,
    attack_token_ids: List[int],
) -> Tuple[str, List[int], torch.Tensor]:
    injection_ids = encode(tokenizer, item["injection"])
    active_count = min(args.num_attack_tokens, len(injection_ids))
    active_tokens = attack_tokens(args.attack_token_prefix, active_count)
    optimized_input = (item["input"].rstrip() + " " + " ".join(active_tokens)).strip()
    prompt = build_chat_prompt(tokenizer, item["instruction"], optimized_input)
    prompt_ids = encode(tokenizer, prompt)
    init_ids = torch.tensor(injection_ids[:active_count], device=embedding.weight.device)
    init_rows = embedding(init_ids).detach().float()
    return optimized_input, prompt_ids, init_rows


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
        normal_pos_tensor = torch.tensor(normal_positions, device=ids.device)
        normal_id_tensor = torch.tensor(normal_ids, device=ids.device)
        embeds[normal_pos_tensor] = embedding(normal_id_tensor).to(model_dtype)

    if attack_positions:
        attack_pos_tensor = torch.tensor(attack_positions, device=ids.device)
        attack_offset_tensor = torch.tensor(attack_offsets, device=ids.device)
        embeds[attack_pos_tensor] = attack_embeds[attack_offset_tensor].to(model_dtype)

    return embeds.unsqueeze(0)


def sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, args) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=args.target_max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_injection_only_prompt(tokenizer, item: dict) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": item["injection"]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def get_or_create_target_record(
    cache_path: Path | None,
    target_cache: Dict[int, dict],
    model,
    tokenizer,
    idx: int,
    item: dict,
    args,
) -> dict:
    existing = target_cache.get(idx)
    if (
        existing is not None
        and existing.get("injection_response")
        and existing.get("benign_response")
    ):
        return existing

    injection_response = (
        existing.get("injection_response") if existing is not None else None
    )
    benign_response = existing.get("benign_response") if existing is not None else None

    if not injection_response:
        injection_response = generate_response(
            model, tokenizer, build_injection_only_prompt(tokenizer, item), args
        )
    if not benign_response:
        benign_response = generate_response(
            model,
            tokenizer,
            build_chat_prompt(tokenizer, item["instruction"], item["input"]),
            args,
        )

    record = {
        "sample_index": idx,
        "instruction": item["instruction"],
        "input": item["input"],
        "injection": item["injection"],
        "witness": item.get("witness"),
        "injection_response": injection_response,
        "benign_response": benign_response,
        "model": args.model,
        "target_max_new_tokens": args.target_max_new_tokens,
    }
    target_cache[idx] = record
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(cache_path, record)
    return record


def optimize_one(
    model,
    tokenizer,
    prompt_ids: List[int],
    target_text: str,
    init_rows: torch.Tensor,
    attack_token_ids: List[int],
    args,
) -> Tuple[torch.Tensor, List[float]]:
    device = next(model.parameters()).device
    embedding = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype

    target_ids = encode(tokenizer, target_text)
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

    full_ids = prompt_ids + target_ids
    if len(full_ids) > args.max_seq_len:
        raise RuntimeError(
            f"Sequence length {len(full_ids)} exceeds --max_seq_len={args.max_seq_len}."
        )

    labels = [-100] * len(prompt_ids) + target_ids
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    labels_tensor = torch.tensor([labels], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    active_attack_ids = attack_token_ids[: init_rows.shape[0]]
    attack_token_id_to_offset = {
        token_id: offset for offset, token_id in enumerate(active_attack_ids)
    }
    attack_embeds = torch.nn.Parameter(init_rows.to(device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([attack_embeds], lr=args.attacker_lr, weight_decay=0.0)

    losses = []
    for _ in range(args.attack_steps + 1):
        inputs_embeds = build_inputs_embeds(
            input_ids,
            embedding,
            attack_token_id_to_offset,
            attack_embeds,
            model_dtype,
        )
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        loss = sequence_loss(outputs.logits, labels_tensor)
        losses.append(float(loss.detach().cpu()))

        if len(losses) > args.attack_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return attack_embeds.detach().cpu().to(torch.float16), losses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["csa_suffix", "misa_span"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_data", default="data/SEP_dataset_test.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--num_attack_tokens", type=int, default=1000)
    parser.add_argument("--csa_tokens", type=int, default=10)
    parser.add_argument("--attack_steps", type=int, default=20)
    parser.add_argument("--attacker_lr", type=float, default=5e-4)
    parser.add_argument("--attack_token_prefix", default="<ATTACK_")
    parser.add_argument("--attack_init_token", default="!")
    parser.add_argument(
        "--target_mode",
        choices=["injection_response", "field"],
        default="injection_response",
    )
    parser.add_argument("--target_field", default="witness")
    parser.add_argument("--target_cache")
    parser.add_argument("--target_max_new_tokens", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    emb_dir = out_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
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
        args.model,
        torch_dtype=dtype_map[args.dtype],
    ).to("cuda")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    embedding = model.get_input_embeddings()
    attack_token_ids = [
        tokenizer.convert_tokens_to_ids(tok)
        for tok in attack_tokens(args.attack_token_prefix, args.num_attack_tokens)
    ]
    init_token_ids = encode(tokenizer, args.attack_init_token)
    if len(init_token_ids) != 1:
        raise RuntimeError(
            f"--attack_init_token must tokenize to one token, got {init_token_ids}."
        )
    init_token_id = init_token_ids[0]

    data = load_json(args.test_data)
    if args.num_samples > 0:
        data = data[: args.num_samples]

    done = completed_indices(records_path)
    target_cache_path = Path(args.target_cache) if args.target_cache else None
    target_cache = read_jsonl_by_index(target_cache_path) if target_cache_path else {}
    manifest = {
        "mode": args.mode,
        "model": args.model,
        "test_data": args.test_data,
        "num_samples": len(data),
        "num_attack_tokens": args.num_attack_tokens,
        "csa_tokens": args.csa_tokens,
        "attack_steps": args.attack_steps,
        "attacker_lr": args.attacker_lr,
        "target_mode": args.target_mode,
        "target_field": args.target_field,
        "target_cache": args.target_cache,
        "target_max_new_tokens": args.target_max_new_tokens,
        "records": str(records_path),
        "embeddings_dir": str(emb_dir),
        "format": "jsonl plus per-example torch .pt files",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for idx, item in enumerate(tqdm(data, desc=f"curating {args.mode}")):
        if idx in done:
            continue

        if args.mode == "csa_suffix":
            optimized_input, prompt_ids, init_rows = prepare_csa_suffix(
                item, tokenizer, embedding, args, attack_token_ids, init_token_id
            )
        else:
            optimized_input, prompt_ids, init_rows = prepare_misa_span(
                item, tokenizer, embedding, args, attack_token_ids
            )

        target_record = None
        if args.target_mode == "injection_response":
            target_record = get_or_create_target_record(
                target_cache_path,
                target_cache,
                model,
                tokenizer,
                idx,
                item,
                args,
            )
            target_text = target_record["injection_response"]
        else:
            target_text = str(item[args.target_field])

        if not target_text:
            raise RuntimeError(f"Empty target text for sample_index={idx}.")

        optimized_embeddings, losses = optimize_one(
            model,
            tokenizer,
            prompt_ids,
            target_text,
            init_rows,
            attack_token_ids,
            args,
        )

        emb_path = emb_dir / f"{idx:06d}.pt"
        torch.save(
            {
                "sample_index": idx,
                "mode": args.mode,
                "optimized_attack_embeddings": optimized_embeddings,
                "attack_tokens": attack_tokens(args.attack_token_prefix, init_rows.shape[0]),
                "attack_token_ids": attack_token_ids[: init_rows.shape[0]],
                "target_text": target_text,
                "loss_trace": losses,
            },
            emb_path,
        )

        record = dict(item)
        record.update(
            {
                "sample_index": idx,
                "optimized_input": optimized_input,
                "attack_mode": args.mode,
                "attack_active_token_count": int(init_rows.shape[0]),
                "attack_steps": args.attack_steps,
                "attacker_lr": args.attacker_lr,
                "target_mode": args.target_mode,
                "target_field": args.target_field,
                "target_cache": args.target_cache,
                "target_text": target_text,
                "benign_response": (
                    target_record["benign_response"] if target_record is not None else None
                ),
                "loss_initial": losses[0],
                "loss_final": losses[-1],
                "attack_embedding_path": str(emb_path.relative_to(out_dir)),
            }
        )
        append_jsonl(records_path, record)

        if args.log_every > 0 and (idx + 1) % args.log_every == 0:
            print(
                f"processed={idx + 1} loss_initial={losses[0]:.4f} "
                f"loss_final={losses[-1]:.4f} active_tokens={init_rows.shape[0]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
