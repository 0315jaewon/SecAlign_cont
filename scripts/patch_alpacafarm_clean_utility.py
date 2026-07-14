#!/usr/bin/env python3
"""Patch AlpacaFarm clean utility rows in SecAlign summary.tsv files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_PATCHES = {
    "~/models/Meta-Llama-3.1-8B-Instruct_positiondiverse_inner10_epoch0_snapshot": "13.74%",
    "~/models/Meta-Llama-3.1-8B-Instruct_positiondiverse_inner20_epoch0_snapshot": "7.28%",
}

TARGET_ATTACK = "none"
TARGET_TEST_DATA = "data/davinci_003_outputs.json"


def normalize_percent(value: str) -> str:
    value = value.strip()
    return value if value.endswith("%") else f"{value}%"


def patch_summary(model_dir: Path, utility: str) -> None:
    summary = model_dir / "summary.tsv"
    if not summary.exists():
        raise FileNotFoundError(f"summary.tsv not found: {summary}")

    backup = model_dir / "summary.tsv.bak_before_clean_utility_patch"
    if not backup.exists():
        backup.write_text(summary.read_text())

    with summary.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"empty summary.tsv: {summary}")
        rows = list(reader)

    patched = False
    for row in rows:
        if row.get("attack") == TARGET_ATTACK and row.get("test_data") == TARGET_TEST_DATA:
            old = row.get("ASR/Utility", "")
            row["ASR/Utility"] = utility
            patched = True
            print(f"{summary}: {old} -> {utility}")

    if not patched:
        rows.insert(
            0,
            {
                "attack": TARGET_ATTACK,
                "ASR/Utility": utility,
                "defense": "none",
                "instruction_hierarchy": "True",
                "lora_alpha": "8.0",
                "test_data": TARGET_TEST_DATA,
            },
        )
        print(f"{summary}: inserted clean utility row {utility}")

    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_patch(value: str) -> tuple[Path, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("patch must look like /path/to/model=13.74%")
    model, utility = value.split("=", 1)
    return Path(model).expanduser(), normalize_percent(utility)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch the AlpacaFarm clean utility row in one or more summary.tsv files."
    )
    parser.add_argument(
        "--patch",
        action="append",
        type=parse_patch,
        help=(
            "Patch entry as /path/to/model=utility_percent. May be repeated. "
            "Defaults to inner10=13.74%% and inner20=7.28%% snapshots."
        ),
    )
    args = parser.parse_args()

    patches = args.patch or [
        (Path(path).expanduser(), normalize_percent(utility))
        for path, utility in DEFAULT_PATCHES.items()
    ]

    for model_dir, utility in patches:
        patch_summary(model_dir, utility)


if __name__ == "__main__":
    main()
