import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def checkpoint_sort_key(path: Path):
    stem = path.stem
    try:
        return int(stem.split("step_")[-1])
    except ValueError:
        return stem


def summarize_file(path: Path, max_steps: int):
    rows = read_jsonl(path)
    first_flips = [r.get("first_margin_flip_step") for r in rows]
    seen = [int(x) for x in first_flips if x is not None]
    censored = [max_steps + 1 if x is None else int(x) for x in first_flips]
    final_margins = [float(r["final_margin"]) for r in rows if "final_margin" in r]
    initial_margins = [
        float(r["records"][0]["margin_chosen_minus_rejected"])
        for r in rows
        if r.get("records")
    ]
    final_flip_count = sum(m > 0 for m in final_margins)
    n = len(rows)
    return {
        "checkpoint": path.stem,
        "num_samples": n,
        "margin_flips": len(seen),
        "flip_rate@20": f"{sum(x is not None and int(x) <= 20 for x in first_flips) / n:.4f}" if n else "NA",
        "flip_rate@50": f"{sum(x is not None and int(x) <= 50 for x in first_flips) / n:.4f}" if n else "NA",
        "flip_rate@100": f"{sum(x is not None and int(x) <= 100 for x in first_flips) / n:.4f}" if n else "NA",
        "final_flip_rate": f"{final_flip_count / n:.4f}" if n else "NA",
        "mean_first_flip_seen": f"{mean(seen):.2f}" if seen else "NA",
        "median_first_flip_seen": f"{median(seen):.2f}" if seen else "NA",
        "mean_first_flip_censored": f"{mean(censored):.2f}" if censored else "NA",
        "median_first_flip_censored": f"{median(censored):.2f}" if censored else "NA",
        "mean_initial_margin": f"{mean(initial_margins):.6f}" if initial_margins else "NA",
        "mean_final_margin": f"{mean(final_margins):.6f}" if final_margins else "NA",
        "output_jsonl": str(path),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--max_steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("step_*.jsonl"), key=checkpoint_sort_key)
    rows = [summarize_file(path, args.max_steps) for path in paths]
    fieldnames = [
        "checkpoint",
        "num_samples",
        "margin_flips",
        "flip_rate@20",
        "flip_rate@50",
        "flip_rate@100",
        "final_flip_rate",
        "mean_first_flip_seen",
        "median_first_flip_seen",
        "mean_first_flip_censored",
        "median_first_flip_censored",
        "mean_initial_margin",
        "mean_final_margin",
        "output_jsonl",
    ]
    output = Path(args.output_tsv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
    if rows:
        print(output.read_text())


if __name__ == "__main__":
    main()
