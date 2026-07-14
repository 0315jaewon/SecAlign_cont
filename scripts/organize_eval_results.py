#!/usr/bin/env python3
"""Reorder SecAlign eval summaries and collect reproducibility artifacts."""

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

RUN_METADATA = {
    "inner20_step60": {
        "description": "Llama-3.1-8B position-diverse commonword SFT-attacker LoRA, 20 attacker inner steps, step-60 snapshot.",
        "training_script": "scripts/train_llama31_8b_position_diverse10_commonword_sft_inner20_bs1_ga64_4gpu_ep1.sh",
        "training_config": "configs/training/llama3.1_8b_position_diverse10_commonword_sft_inner20_bs1_ga64_4gpu_ep1.yaml",
        "example_launch": (
            "NPROC_PER_NODE=8 GRAD_ACCUM=32 "
            "CACHE_DIR=/home/$USER/models/Meta-Llama-3.1-8B-Instruct "
            "OUTPUT_DIR=/tmp/secalign_runs/llama31_8b_position_diverse10_commonword_sft_inner20_bs1_ga32_8gpu_epoch1 "
            "LOG_DIR=$PWD/position_diverse10_commonword_8b_train_outputs "
            "nohup scripts/train_llama31_8b_position_diverse10_commonword_sft_inner20_bs1_ga64_4gpu_ep1.sh "
            "> position_diverse10_commonword_8b_train_outputs/nohup.out 2>&1 &"
        ),
    },
    "inner10_epoch0": {
        "description": "Llama-3.1-8B position-diverse commonword SFT-attacker LoRA, 10 attacker inner steps, final epoch-0 checkpoint.",
        "training_script": "scripts/train_llama31_8b_position_diverse10_commonword_sft_inner10_bs1_ga64_4gpu_ep1.sh",
        "training_config": "configs/training/llama3.1_8b_position_diverse10_commonword_sft_inner10_bs1_ga64_4gpu_ep1.yaml",
        "example_launch": (
            "NPROC_PER_NODE=8 GRAD_ACCUM=32 "
            "CACHE_DIR=/home/$USER/models/Meta-Llama-3.1-8B-Instruct "
            "OUTPUT_DIR=/tmp/secalign_runs/llama31_8b_position_diverse10_commonword_sft_inner10_bs1_ga32_8gpu_epoch1 "
            "LOG_DIR=$PWD/position_diverse10_commonword_8b_inner10_train_outputs "
            "nohup scripts/train_llama31_8b_position_diverse10_commonword_sft_inner10_bs1_ga64_4gpu_ep1.sh "
            "> position_diverse10_commonword_8b_inner10_train_outputs/nohup.out 2>&1 &"
        ),
    },
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


def copy_run_artifacts(label: str, model_dir: Path, dest_dir: Path, repo_root: Path) -> None:
    metadata = RUN_METADATA.get(label)
    if metadata is None:
        print(f"no run metadata configured for: {label}")
        return

    repro_dir = dest_dir / "repro"
    repro_dir.mkdir(parents=True, exist_ok=True)

    copied_paths = []
    for key in ["training_script", "training_config"]:
        rel_path = Path(metadata[key])
        src = repo_root / rel_path
        if not src.exists():
            print(f"configured {key} missing, skipping: {src}")
            continue
        target = repro_dir / rel_path.name
        shutil.copy2(src, target)
        copied_paths.append((key, target.relative_to(dest_dir)))
        print(f"copied {key}: {src} -> {target}")

    manifest = dest_dir / "training_manifest.md"
    lines = [
        f"# {label}",
        "",
        metadata["description"],
        "",
        f"- Model snapshot: `{model_dir}`",
    ]
    for key, rel_target in copied_paths:
        label_text = key.replace("_", " ").title()
        lines.append(f"- {label_text}: `{rel_target}`")
    lines.extend(
        [
            "",
            "## Example Launch",
            "",
            "```bash",
            metadata["example_launch"],
            "```",
            "",
            "The launch script enforces effective batch size 256 via "
            "`NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM` and records the exact "
            "runtime overrides in its training log.",
            "",
        ]
    )
    manifest.write_text("\n".join(lines))
    print(f"wrote training manifest: {manifest}")


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
        description=(
            "Reorder summary.tsv files and collect nested results.csv files plus "
            "training reproducibility artifacts."
        )
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
    parser.add_argument(
        "--no-run-artifacts",
        action="store_true",
        help="Do not copy training scripts/configs or write training manifests.",
    )

    args = parser.parse_args()
    models = dict(args.model or [(k, expand_path(v)) for k, v in DEFAULT_MODELS.items()])
    out_root = expand_path(args.out_dir)
    repo_root = Path(__file__).resolve().parents[1]

    for label, model_dir in models.items():
        if not model_dir.exists():
            print(f"model directory missing, skipping {label}: {model_dir}")
            continue

        dest_dir = out_root / label
        if not args.no_reorder:
            reorder_summary(model_dir, dest_dir)
        copy_results_csvs(model_dir, dest_dir)
        if not args.no_run_artifacts:
            copy_run_artifacts(label, model_dir, dest_dir, repo_root)

    print(f"organized results root: {out_root}")


if __name__ == "__main__":
    main()
