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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


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


def attack_tokens(prefix: str, count: int) -> List[str]:
    return [f"{prefix}{idx}>" for idx in range(count)]


def encode(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_chat_prompt(tokenizer, instruction: str, input_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": instruction},
            {"role": "input", "content": input_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_injection_only_prompt(tokenizer, item: dict) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": item["injection"]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_csa_inspan_injection(
    tokenizer,
    injection: str,
    active_tokens: List[str],
) -> str:
    injection_ids = encode(tokenizer, injection.strip())
    if not injection_ids:
        return " ".join(active_tokens)

    pieces = [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in injection_ids
    ]
    output = []
    for idx, piece in enumerate(pieces):
        output.append(piece)
        if idx < len(active_tokens):
            output.append(" " + active_tokens[idx])
    return "".join(output).strip()


def infer_base_model_path(model_path: str) -> str:
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        base_path = model_path.split("_")[0]
        if os.path.exists(base_path):
            return base_path
    return model_path


def load_model_and_tokenizer(model_path: str, args):
    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    base_path = args.base_model or infer_base_model_path(model_path)
    tokenizer_path = model_path if os.path.exists(model_path) else base_path
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
        base_path,
        torch_dtype=dtype_map[args.dtype],
    )
    model.resize_token_embeddings(len(tokenizer))

    if is_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Loading LoRA adapters requires peft. Install peft in this env."
            ) from exc
        model = PeftModel.from_pretrained(model, model_path)

    model.to("cuda")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def get_target_text(model, tokenizer, item: dict, idx: int, args, cache: Dict[int, dict]):
    if args.target_cache:
        cached = cache.get(idx)
        if cached is not None and cached.get("injection_response"):
            return cached["injection_response"]

    target = generate_response(
        model,
        tokenizer,
        build_injection_only_prompt(tokenizer, item),
        args.target_max_new_tokens,
    )
    if args.target_cache:
        record = {
            "sample_index": idx,
            "instruction": item["instruction"],
            "input": item["input"],
            "injection": item["injection"],
            "witness": item.get("witness"),
            "injection_response": target,
            "target_model": args.target_model or args.model,
            "target_max_new_tokens": args.target_max_new_tokens,
        }
        append_jsonl(Path(args.target_cache), record)
        cache[idx] = record
    return target


def prepare_probe_prompt(
    item: dict,
    tokenizer,
    embedding,
    args,
    attack_token_ids: List[int],
    init_token_id: int,
) -> Tuple[str, List[int], torch.Tensor, int]:
    if args.probe == "csa_suffix":
        if args.csa_tokens <= 0:
            active_count = min(args.num_attack_tokens, len(encode(tokenizer, item["injection"])))
        else:
            active_count = min(args.num_attack_tokens, args.csa_tokens)
        if args.csa_placement == "inspan":
            active_count = min(active_count, len(encode(tokenizer, item["injection"])))
        active_tokens = attack_tokens(args.attack_token_prefix, active_count)
        if args.csa_placement == "suffix":
            attacked_input = (
                item["input"].rstrip()
                + " "
                + item["injection"].strip()
                + " "
                + " ".join(active_tokens)
            ).strip()
        elif args.csa_placement == "prefix":
            attacked_input = (
                item["input"].rstrip()
                + " "
                + " ".join(active_tokens)
                + " "
                + item["injection"].strip()
            ).strip()
        elif args.csa_placement == "inspan":
            attacked_input = (
                item["input"].rstrip()
                + " "
                + build_csa_inspan_injection(tokenizer, item["injection"], active_tokens)
            ).strip()
        else:
            raise ValueError(f"Unsupported CSA placement: {args.csa_placement}")
        init_rows = (
            embedding.weight[init_token_id].detach().float().repeat(active_count, 1)
        )
    else:
        injection_ids = encode(tokenizer, item["injection"])
        active_count = min(args.num_attack_tokens, len(injection_ids))
        active_tokens = attack_tokens(args.attack_token_prefix, active_count)
        attacked_input = (item["input"].rstrip() + " " + " ".join(active_tokens)).strip()
        init_ids = torch.tensor(injection_ids[:active_count], device=embedding.weight.device)
        init_rows = embedding(init_ids).detach().float()

    prompt = build_chat_prompt(tokenizer, item["instruction"], attacked_input)
    prompt_ids = encode(tokenizer, prompt)
    return attacked_input, prompt_ids, init_rows, active_count


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
def generate_with_attack_embeddings(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attack_token_id_to_offset: Dict[int, int],
    attack_embeds: torch.Tensor,
    max_new_tokens: int,
) -> str:
    embedding = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
    generated_ids = []
    current_ids = input_ids.clone()

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
        next_id_tensor = torch.tensor([[next_id]], device=current_ids.device)
        current_ids = torch.cat([current_ids, next_id_tensor], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def probe_one_sample(model, tokenizer, item: dict, idx: int, target_text: str, args):
    device = next(model.parameters()).device
    embedding = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
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

    attacked_input, prompt_ids, init_rows, active_count = prepare_probe_prompt(
        item, tokenizer, embedding, args, attack_token_ids, init_token_id
    )
    target_ids = encode(tokenizer, target_text)
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    full_ids = prompt_ids + target_ids
    if len(full_ids) > args.max_seq_len:
        target_ids = target_ids[: max(1, args.max_seq_len - len(prompt_ids))]
        full_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    generation_input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    labels_tensor = torch.tensor([labels], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    active_attack_ids = attack_token_ids[:active_count]
    attack_token_id_to_offset = {
        token_id: offset for offset, token_id in enumerate(active_attack_ids)
    }
    attack_embeds = torch.nn.Parameter(init_rows.to(device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([attack_embeds], lr=args.attacker_lr, weight_decay=0.0)

    records = []
    for step in range(args.attack_steps + 1):
        inputs_embeds = build_inputs_embeds(
            input_ids,
            embedding,
            attack_token_id_to_offset,
            attack_embeds,
            model_dtype,
        )
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        loss = sequence_loss(outputs.logits, labels_tensor)

        generated = generate_with_attack_embeddings(
            model,
            tokenizer,
            generation_input_ids,
            attack_token_id_to_offset,
            attack_embeds.detach(),
            args.generation_max_new_tokens,
        )
        witness_present = item["witness"].lower() in generated.lower()
        record = {
            "sample_index": idx,
            "step": step,
            "loss": float(loss.detach().cpu()),
            "witness_present": bool(witness_present),
            "output": generated,
        }
        records.append(record)

        print("\n" + "=" * 100, flush=True)
        print(
            f"model={args.model_label or args.model} probe={args.probe} "
            f"placement={args.csa_placement if args.probe == 'csa_suffix' else 'NA'} "
            f"sample={idx} step={step}/{args.attack_steps} "
            f"active_tokens={active_count} loss={record['loss']:.6f} "
            f"witness={item['witness']!r} witness_present={witness_present}",
            flush=True,
        )
        print(f"instruction: {item['instruction']}", flush=True)
        print(f"original_input: {item['input']}", flush=True)
        print(f"injection: {item['injection']}", flush=True)
        print(f"attacked_input: {attacked_input}", flush=True)
        print("output:", flush=True)
        print(generated, flush=True)

        if step == args.attack_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return {
        "sample_index": idx,
        "model": args.model,
        "model_label": args.model_label,
        "probe": args.probe,
        "csa_placement": args.csa_placement if args.probe == "csa_suffix" else None,
        "instruction": item["instruction"],
        "input": item["input"],
        "injection": item["injection"],
        "witness": item["witness"],
        "target_text": target_text,
        "attacked_input": attacked_input,
        "active_attack_tokens": active_count,
        "records": records,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=["csa_suffix", "misa_span"], required=True)
    parser.add_argument(
        "--csa_placement",
        choices=["suffix", "prefix", "inspan"],
        default="suffix",
        help="Where CSA attack tokens are placed relative to the SEP injection.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_label")
    parser.add_argument("--base_model")
    parser.add_argument("--target_model")
    parser.add_argument("--test_data", default="data/SEP_dataset_test.json")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--target_cache")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_attack_tokens", type=int, default=1000)
    parser.add_argument(
        "--csa_tokens",
        type=int,
        default=10,
        help="Number of CSA suffix tokens. Use <=0 for dynamic injection-token length.",
    )
    parser.add_argument("--attack_steps", type=int, default=100)
    parser.add_argument("--attacker_lr", type=float, default=5e-4)
    parser.add_argument("--attack_token_prefix", default="<ATTACK_")
    parser.add_argument("--attack_init_token", default="!")
    parser.add_argument("--target_max_new_tokens", type=int, default=512)
    parser.add_argument("--generation_max_new_tokens", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_json(args.test_data)
    data = data[args.start_index : args.start_index + args.num_samples]

    model, tokenizer = load_model_and_tokenizer(args.model, args)
    target_model = model
    target_tokenizer = tokenizer
    if args.target_model and args.target_model != args.model:
        target_args = argparse.Namespace(**vars(args))
        target_args.model = args.target_model
        target_args.base_model = None
        target_model, target_tokenizer = load_model_and_tokenizer(args.target_model, target_args)

    target_cache = read_jsonl_by_index(Path(args.target_cache)) if args.target_cache else {}
    output_path = Path(args.output_jsonl)
    completed = read_jsonl_by_index(output_path) if args.resume else {}

    for local_idx, item in enumerate(tqdm(data, desc=f"probing {args.model_label or args.model} {args.probe}")):
        idx = args.start_index + local_idx
        if idx in completed:
            print(
                f"Skipping completed sample_index={idx} for "
                f"model={args.model_label or args.model} probe={args.probe}",
                flush=True,
            )
            continue
        target_text = get_target_text(
            target_model,
            target_tokenizer,
            item,
            idx,
            args,
            target_cache,
        )
        result = probe_one_sample(model, tokenizer, item, idx, target_text, args)
        append_jsonl(output_path, result)


if __name__ == "__main__":
    main()
