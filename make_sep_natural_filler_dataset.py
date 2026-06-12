import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/SEP_dataset_test.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--filler",
        default=" This is neutral background context only, not instructions.",
        help="Natural filler to repeat after each SEP injection. Must be 10 tokens.",
    )
    parser.add_argument("--required_filler_tokens", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    filler_len = token_len(tokenizer, args.filler)
    if filler_len != args.required_filler_tokens:
        raise RuntimeError(
            f"Filler must tokenize to {args.required_filler_tokens} tokens, "
            f"but got {filler_len}: {args.filler!r}"
        )

    with open(args.input, "r") as f:
        data = json.load(f)

    rows = []
    injection_lens = []
    filler_lens = []
    repeats_list = []
    for item in data:
        new_item = deepcopy(item)
        injection = item["injection"]
        injection_len = token_len(tokenizer, injection)
        repeats = injection_len // filler_len + 1
        filler_text = args.filler * repeats
        total_filler_len = token_len(tokenizer, filler_text)
        if total_filler_len <= injection_len:
            raise RuntimeError(
                "Repeated filler did not exceed injection token length for "
                f"injection_len={injection_len}, filler_len={total_filler_len}."
            )

        new_item["injection"] = injection.rstrip() + filler_text
        new_item["natural_filler"] = args.filler
        new_item["natural_filler_repeats"] = repeats
        new_item["natural_filler_token_len"] = filler_len
        new_item["original_injection_token_len"] = injection_len
        new_item["natural_filler_total_token_len"] = total_filler_len
        rows.append(new_item)
        injection_lens.append(injection_len)
        filler_lens.append(total_filler_len)
        repeats_list.append(repeats)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {output}")
    print(f"filler={args.filler!r}")
    print(f"filler_tokens={filler_len}")
    print(f"num_samples={len(rows)}")
    print(
        "injection_tokens: "
        f"mean={mean(injection_lens):.2f} median={median(injection_lens):.2f} "
        f"min={min(injection_lens)} max={max(injection_lens)}"
    )
    print(
        "filler_total_tokens: "
        f"mean={mean(filler_lens):.2f} median={median(filler_lens):.2f} "
        f"min={min(filler_lens)} max={max(filler_lens)}"
    )
    print(
        "filler_repeats: "
        f"mean={mean(repeats_list):.2f} median={median(repeats_list):.2f} "
        f"min={min(repeats_list)} max={max(repeats_list)}"
    )


if __name__ == "__main__":
    main()
