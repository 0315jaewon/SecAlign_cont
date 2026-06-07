import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def read_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    with path.open("r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def first_witness_step(record: dict):
    for step_record in record.get("records", []):
        if step_record.get("witness_present"):
            return int(step_record["step"])
    return None


def summarize_file(path: Path, model_label: str, probe_label: str, max_steps: int):
    records = read_jsonl(path)
    first_steps = [first_witness_step(record) for record in records]
    seen_steps = [step for step in first_steps if step is not None]
    censored = [max_steps + 1 if step is None else step for step in first_steps]
    active_counts = [
        int(record.get("active_attack_tokens", 0))
        for record in records
        if "active_attack_tokens" in record
    ]

    n = len(records)
    row = {
        "target_model": model_label,
        "probe": probe_label,
        "num_samples": n,
        "successes": len(seen_steps),
        "ASR@20": f"{sum(step is not None and step <= 20 for step in first_steps) / n:.4f}" if n else "NA",
        "ASR@50": f"{sum(step is not None and step <= 50 for step in first_steps) / n:.4f}" if n else "NA",
        "ASR@100": f"{sum(step is not None and step <= 100 for step in first_steps) / n:.4f}" if n else "NA",
        "mean_first_step_seen": f"{mean(seen_steps):.2f}" if seen_steps else "NA",
        "median_first_step_seen": f"{median(seen_steps):.2f}" if seen_steps else "NA",
        "mean_first_step_censored": f"{mean(censored):.2f}" if censored else "NA",
        "median_first_step_censored": f"{median(censored):.2f}" if censored else "NA",
        "mean_active_tokens": f"{mean(active_counts):.2f}" if active_counts else "NA",
        "min_active_tokens": min(active_counts) if active_counts else "NA",
        "max_active_tokens": max(active_counts) if active_counts else "NA",
        "output_jsonl": str(path),
    }
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="sep_csa_probe_matrix_outputs")
    parser.add_argument("--output_tsv", default="sep_csa_probe_matrix_outputs/summary.tsv")
    parser.add_argument("--max_steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = []
    for path in sorted(input_dir.glob("*/*.jsonl")):
        if path.parent.name == "target_cache":
            continue
        model_label = path.parent.name
        probe_label = path.stem
        rows.append(summarize_file(path, model_label, probe_label, args.max_steps))

    fieldnames = [
        "target_model",
        "probe",
        "num_samples",
        "successes",
        "ASR@20",
        "ASR@50",
        "ASR@100",
        "mean_first_step_seen",
        "median_first_step_seen",
        "mean_first_step_censored",
        "median_first_step_censored",
        "mean_active_tokens",
        "min_active_tokens",
        "max_active_tokens",
        "output_jsonl",
    ]
    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")
    if rows:
        with output_path.open("r") as f:
            print(f.read())


if __name__ == "__main__":
    main()
