import argparse
import json
from pathlib import Path


def append_tsv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a") as f:
        if not exists:
            f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_tsv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--attacks", nargs="+", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    for model in args.models:
        combined = None
        counted = []
        sample_count = None
        for attack in args.attacks:
            path = output_dir / f"{model}_{attack}.json"
            if not path.exists():
                print(f"Skipping missing {path}")
                continue
            records = json.load(open(path))
            successes = [bool(record.get("attack_success", False)) for record in records]
            if combined is None:
                combined = successes
            else:
                combined = [old or new for old, new in zip(combined, successes)]
            counted.append(attack)
            sample_count = len(successes)

        if not counted or combined is None:
            continue
        metric = sum(combined) / len(combined) if combined else 0.0
        append_tsv(
            Path(args.summary_tsv),
            {
                "model": model,
                "attack": "Combined optimized (" + ",".join(counted) + ")",
                "ASR": f"{metric * 100:.2f}%",
                "witness_rate": "n/a",
                "num_samples": str(sample_count or 0),
                "dataset_dir": "multiple",
                "output_json": str(output_dir),
                "judge_model": "from_component_attacks",
            },
        )
        print(
            f"model={model} combined_attacks={','.join(counted)} "
            f"ASR={metric * 100:.2f}% n={len(combined)}"
        )


if __name__ == "__main__":
    main()
