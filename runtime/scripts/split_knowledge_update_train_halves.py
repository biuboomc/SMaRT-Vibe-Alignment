#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a knowledge-update training JSONL into two deterministic halves."
    )
    parser.add_argument("--input", required=True, help="Input training JSONL path.")
    parser.add_argument("--output-a", required=True, help="Output JSONL path for half A.")
    parser.add_argument("--output-b", required=True, help="Output JSONL path for half B.")
    parser.add_argument("--metadata-output", default="", help="Optional split metadata JSON path.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    records: list[dict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    if len(records) < 2:
        raise ValueError(f"Need at least 2 rows to split, got {len(records)}")

    rng = random.Random(args.seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    midpoint = len(indices) // 2
    half_a_indices = set(indices[:midpoint])

    output_a = Path(args.output_a)
    output_b = Path(args.output_b)
    output_a.parent.mkdir(parents=True, exist_ok=True)
    output_b.parent.mkdir(parents=True, exist_ok=True)

    count_a = 0
    count_b = 0
    with output_a.open("w", encoding="utf-8") as handle_a, output_b.open("w", encoding="utf-8") as handle_b:
        for idx, record in enumerate(records):
            payload = json.dumps(record, ensure_ascii=False)
            if idx in half_a_indices:
                handle_a.write(payload + "\n")
                count_a += 1
            else:
                handle_b.write(payload + "\n")
                count_b += 1

    if args.metadata_output:
        metadata_path = Path(args.metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input": str(input_path),
            "output_a": str(output_a),
            "output_b": str(output_b),
            "total_rows": len(records),
            "count_a": count_a,
            "count_b": count_b,
            "seed": args.seed,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "total_rows": len(records),
                "count_a": count_a,
                "count_b": count_b,
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
