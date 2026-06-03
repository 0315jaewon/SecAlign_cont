#!/usr/bin/env python

import argparse
import glob
import json
from pathlib import Path


ATTACKS = [
    "straightforward",
    "straightforward_before",
    "ignore",
    "ignore_before",
    "completion",
    "completion_ignore",
    "completion_llama32_1B",
    "completion_ignore_llama32_1B",
]


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def find_attack_file(root: Path, attack: str) -> Path:
    if root.is_file():
        return root

    patterns = [
        str(root / f"{attack}_*.json"),
        str(root / f"{attack}_none_loraalpha*_SEP_dataset_test.json"),
        str(root / "**" / f"{attack}_*.json"),
        str(root / "**" / f"{attack}_none_loraalpha*_SEP_dataset_test.json"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))

    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"No JSON file found for attack={attack} under {root}")

    # Prefer exact attack prefix over files whose attack name merely shares a prefix.
    exact = [m for m in matches if Path(m).name.startswith(attack + "_")]
    return Path(exact[0] if exact else matches[0])


def key_for(record, idx):
    return (
        record.get("sample_index", idx),
        record.get("instruction", ""),
        record.get("injection", ""),
    )


def compare_attack(args, attack: str):
    a_path = find_attack_file(args.judge_a_dir, attack)
    b_path = find_attack_file(args.judge_b_dir, attack)
    a_records = load_json(a_path)
    b_records = load_json(b_path)

    if args.num_samples > 0:
        a_records = a_records[: args.num_samples]
        b_records = b_records[: args.num_samples]

    b_by_key = {key_for(record, idx): record for idx, record in enumerate(b_records)}

    disagreements = []
    agree = 0
    missing = 0
    for idx, a_record in enumerate(a_records):
        key = key_for(a_record, idx)
        b_record = b_by_key.get(key)
        if b_record is None:
            missing += 1
            continue

        a_success = bool(a_record.get(args.field, False))
        b_success = bool(b_record.get(args.field, False))
        if a_success == b_success:
            agree += 1
            continue

        disagreements.append((idx, a_record, b_record, a_success, b_success))

    n = agree + len(disagreements)
    print("\n" + "=" * 100)
    print(f"attack={attack}")
    print(f"judge_a_file={a_path}")
    print(f"judge_b_file={b_path}")
    print(f"matched={n} missing={missing}")
    if n:
        print(f"judge_a_ASR={sum(bool(r.get(args.field, False)) for r in a_records[:n]) / n:.4f}")
        print(f"judge_b_ASR={sum(bool(r.get(args.field, False)) for r in b_records[:n]) / n:.4f}")
        print(f"disagreements={len(disagreements)} rate={len(disagreements) / n:.4f}")

    shown = 0
    for idx, a_record, b_record, a_success, b_success in disagreements:
        if args.only_a_success and not (a_success and not b_success):
            continue
        if args.only_b_success and not (b_success and not a_success):
            continue

        print("\n" + "-" * 100)
        print(
            f"attack={attack} idx={idx} "
            f"{args.judge_a_name}={a_success} {args.judge_b_name}={b_success}"
        )
        print(f"witness={a_record.get('witness')!r}")
        print(f"instruction: {a_record.get('instruction', '')}")
        print(f"injection: {a_record.get('injection', '')}")
        print("\noutput:")
        print(a_record.get("output", ""))
        shown += 1
        if shown >= args.max_examples:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_a_dir", type=Path, required=True)
    parser.add_argument("--judge_b_dir", type=Path, required=True)
    parser.add_argument("--judge_a_name", default="judge_a")
    parser.add_argument("--judge_b_name", default="judge_b")
    parser.add_argument("--attacks", nargs="+", default=ATTACKS)
    parser.add_argument("--field", default="attack_success")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--only_a_success", action="store_true")
    parser.add_argument("--only_b_success", action="store_true")
    args = parser.parse_args()

    for attack in args.attacks:
        compare_attack(args, attack)


if __name__ == "__main__":
    main()
