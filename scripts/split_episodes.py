#!/usr/bin/env python3
"""Split episode JSONL data into train, validation, and test files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split coin challenge episodes.")
    parser.add_argument("--input", default="episodes_train.jsonl", help="Source JSONL file.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--train-output", default="episodes_train_split.jsonl")
    parser.add_argument("--val-output", default="episodes_val.jsonl")
    parser.add_argument("--test-output", default="episodes_test.jsonl")
    parser.add_argument("--manifest-output", default="split_manifest.json")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    source = Path(args.input)
    rows = load_jsonl(source)
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    total = len(rows)
    train_count = round(total * args.train_ratio)
    val_count = round(total * args.val_ratio)

    splits = {
        args.train_output: rows[:train_count],
        args.val_output: rows[train_count : train_count + val_count],
        args.test_output: rows[train_count + val_count :],
    }

    for output_path, split_rows in splits.items():
        write_jsonl(Path(output_path), split_rows)

    manifest = {
        "source": str(source),
        "seed": args.seed,
        "split_ratio": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "counts": {output_path: len(split_rows) for output_path, split_rows in splits.items()},
        "note": "Original source file is preserved; image paths remain relative to repository root.",
    }
    Path(args.manifest_output).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
