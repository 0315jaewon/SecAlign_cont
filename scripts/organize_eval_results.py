#!/usr/bin/env python3
"""Reorder SecAlign eval summaries and collect result CSVs in one directory."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ATTACK_ORDER = [
    "none",
    "straightforward",
    "straightforward_before",
    "ignore",
    "ignore_before",
    "completion",
    "completion_ignore",
    "completion_llama31_8B",
    "completion_ignore_llama31_8B",
    "Combined non-adaptive (straightforward,straightforward_before,ignore,ignore_before,completion,completion_ignore)",
    "Combined adaptive (completion_llama31_8B,completion_ignore_llama31_8B)",
]

TEST_DATA_ORDER = [
    "data/davinci_003_outputs.json",
    "data/SEP_dataset_test.json",
    "meta_gpqa_cot",
    "meta_ifeval",
    "meta_bbh",
    "meta_mmlu_0shot_instruct",
    "meta_mmlu_pro_instruct",
]


DEFAULT_MODELS = {
    "inner20_step60": "~/models/Meta-Llama-3.1-8B-Instruct_positiondiverse_step60_snapshot",
    "inner10_epoch0": "~/models/Meta-Llama-3.1-8B-Instruct_positiondiverse_inner10_epoch0_snapshot",
}


def expand_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def row_key(row: dict[str, str]) -> tuple[int, int, str, str]:
    attack = row.get("attack", "")
    test_data = row.get("test_data", "")
    return (
        TEST_DATA_ORDER.index(test_data) if test_data in TEST_DATA_ORDER else 999,
        ATTACK_ORDER.index(attack) if attack in ATTACK_ORDER else 999,
        test_data,
        attack,
    )


def reorder_summary(model_dir: Path, dest_dir: Path) -> None:
    summary = model_dir / "summary.tsv"
    if not summary.exists():
        print(f"summary.tsv missing, skipping: {summary}")
        return

    backup = model_dir / "summary.tsv.bak_before_reorder"
    if not backup.exists():
        shutil.copy2(summary, backup)

    with summary.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            print(f"empty summary.tsv, skipping: {summary}")
            return
        rows = list(reader)

    rows.sort(key=row_key)

    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in [summary, dest_dir / "summary.tsv"]:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    print(f"reordered summary: {summary}")
    print(f"copied summary to: {dest_dir / 'summary.tsv'}")


def copy_results_csvs(model_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for csv_path in sorted(model_dir.rglob("results.csv")):
        rel_name = "__".join(csv_path.relative_to(model_dir).parts)
        target = dest_dir / rel_name
        shutil.copy2(csv_path, target)
        copied += 1
        print(f"copied {csv_path} -> {target}")
    if copied == 0:
        print(f"no results.csv files found under: {model_dir}")


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model entries must look like label=/path/to/model")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("model label cannot be empty")
    return label, expand_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorder summary.tsv files and collect nested results.csv files."
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_arg,
        help=(
            "Model directory to organize, as label=/path. "
            "May be passed more than once. Defaults to the inner20 and inner10 snapshots."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="eval_results/llama31_8b_position_diverse",
        help="Directory where organized summaries and result CSVs are copied.",
    )
    parser.add_argument(
        "--no-reorder",
        action="store_true",
        help="Only collect results.csv files; do not rewrite summary.tsv ordering.",
    )

    args = parser.parse_args()
    models = dict(args.model or [(k, expand_path(v)) for k, v in DEFAULT_MODELS.items()])
    out_root = expand_path(args.out_dir)

    for label, model_dir in models.items():
        if not model_dir.exists():
            print(f"model directory missing, skipping {label}: {model_dir}")
            continue

        dest_dir = out_root / label
        if not args.no_reorder:
            reorder_summary(model_dir, dest_dir)
        copy_results_csvs(model_dir, dest_dir)

    print(f"organized results root: {out_root}")


if __name__ == "__main__":
    main()
