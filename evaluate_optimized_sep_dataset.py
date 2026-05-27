import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import judge_injection_following_multi_stage


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def append_tsv(path: Path, row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a") as f:
        if not exists:
            f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")


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


def infer_base_model_path(model_path: str) -> str:
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        base_path = model_path.split("_")[0]
        if os.path.exists(base_path):
            return base_path
    return model_path


def load_model_and_tokenizer(model_path: str, base_model: str | None, dtype: str, num_attack_tokens: int, prefix: str):
    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    base_path = base_model or infer_base_model_path(model_path)
    tokenizer_path = model_path if os.path.exists(model_path) else base_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": attack_tokens(prefix, num_attack_tokens)}
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype_map[dtype],
    )
    model.resize_token_embeddings(len(tokenizer))

    if is_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading LoRA adapters requires peft.") from exc
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


@torch.no_grad()
def generate_with_optimized_embeddings(
    model,
    tokenizer,
    prompt_ids: List[int],
    attack_token_id_to_offset: Dict[int, int],
    attack_embeds: torch.Tensor,
    max_new_tokens: int,
) -> str:
    device = next(model.parameters()).device
    embedding = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype
    current_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    attack_embeds = attack_embeds.to(device=device, dtype=torch.float32)
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
            [current_ids, torch.tensor([[next_id]], device=device)], dim=1
        )

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_outputs(records: List[dict], args) -> List[bool]:
    witness_appears = [
        str(record.get("witness", "")).lower() in record["output"].lower()
        for record in records
    ]
    witness_indices = [idx for idx, appears in enumerate(witness_appears) if appears]
    attack_success = [False] * len(records)

    if args.skip_judge or not witness_indices:
        for idx in witness_indices:
            attack_success[idx] = True
        return attack_success

    judge_results = judge_injection_following_multi_stage(
        injections=[records[i]["injection"] for i in witness_indices],
        responses=[records[i]["output"] for i in witness_indices],
        gemini_config_path=args.gemini_config_path,
        instructions=[records[i]["instruction"] for i in witness_indices],
        perfect_examples=[records[i].get("target_text", "") for i in witness_indices],
        failed_examples=[records[i].get("benign_response", "") for i in witness_indices],
        model_name=args.gemini_judge_model,
    )
    for idx, judge_result in zip(witness_indices, judge_results):
        attack_success[idx] = bool(judge_result)
    return attack_success


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--summary_tsv", required=True)
    parser.add_argument("--attack_label")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--attack_token_prefix", default="<ATTACK_")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gemini_config_path", default="data/gemini_configs.yaml")
    parser.add_argument("--gemini_judge_model", default="gemini-3.1-pro-preview")
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    records_path = dataset_dir / "records.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    output_path = Path(args.output_json)

    if output_path.exists() and not args.force:
        outputs = json.load(open(output_path))
        if outputs and all("attack_success" in record for record in outputs):
            metric = sum(bool(r["attack_success"]) for r in outputs) / len(outputs)
            print(f"Reusing {output_path}: ASR={metric * 100:.2f}%")
            return

    records = load_jsonl(records_path)
    if args.num_samples > 0:
        records = records[: args.num_samples]
    manifest = json.load(open(manifest_path))
    attack_label = args.attack_label or manifest.get("mode", dataset_dir.name)
    num_attack_tokens = int(manifest["num_attack_tokens"])

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.base_model,
        args.dtype,
        num_attack_tokens,
        args.attack_token_prefix,
    )
    attack_token_ids = [
        tokenizer.convert_tokens_to_ids(tok)
        for tok in attack_tokens(args.attack_token_prefix, num_attack_tokens)
    ]

    outputs = []
    for record in tqdm(records, desc=f"evaluating {args.model_label} {attack_label}"):
        emb_payload = torch.load(
            dataset_dir / record["attack_embedding_path"],
            map_location="cpu",
        )
        attack_embeds = emb_payload["optimized_attack_embeddings"].float()
        active_attack_ids = attack_token_ids[: attack_embeds.shape[0]]
        attack_token_id_to_offset = {
            token_id: offset for offset, token_id in enumerate(active_attack_ids)
        }
        prompt = build_chat_prompt(
            tokenizer,
            record["instruction"],
            record["optimized_input"],
        )
        output = generate_with_optimized_embeddings(
            model,
            tokenizer,
            encode(tokenizer, prompt),
            attack_token_id_to_offset,
            attack_embeds,
            args.max_new_tokens,
        )
        out_record = dict(record)
        out_record.update(
            {
                "output": output,
                "generator": args.model,
                "model_label": args.model_label,
                "evaluated_attack_label": attack_label,
            }
        )
        outputs.append(out_record)

    attack_success = evaluate_outputs(outputs, args)
    for record, success in zip(outputs, attack_success):
        record["attack_success"] = bool(success)
        record["witness_present"] = (
            str(record.get("witness", "")).lower() in record["output"].lower()
        )

    write_json(output_path, outputs)
    metric = sum(attack_success) / len(attack_success) if attack_success else 0.0
    witness_rate = (
        sum(bool(record["witness_present"]) for record in outputs) / len(outputs)
        if outputs
        else 0.0
    )
    append_tsv(
        Path(args.summary_tsv),
        {
            "model": args.model_label,
            "attack": attack_label,
            "ASR": f"{metric * 100:.2f}%",
            "witness_rate": f"{witness_rate * 100:.2f}%",
            "num_samples": str(len(outputs)),
            "dataset_dir": str(dataset_dir),
            "output_json": str(output_path),
            "judge_model": "witness_only" if args.skip_judge else args.gemini_judge_model,
        },
    )
    print(
        f"model={args.model_label} attack={attack_label} "
        f"ASR={metric * 100:.2f}% witness_rate={witness_rate * 100:.2f}% "
        f"n={len(outputs)}"
    )


if __name__ == "__main__":
    main()
