#!/usr/bin/env python3
"""Verify hashes, counts, split membership, and privacy of the Reader bundle."""

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PRIVATE_PATH = re.compile(r"(?i)(?:/mnt/shared-storage[^/]*/|[a-z]:\\Users\\[^\\]+\\)")
SOURCE_NAMES = (
    "reader_source_all_19121.jsonl.gz",
    "reader_source_train_17184.jsonl.gz",
    "reader_source_test_200.jsonl.gz",
)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_private_text(value):
    if isinstance(value, str):
        return PRIVATE_PATH.search(value) is not None
    if isinstance(value, dict):
        return any(has_private_text(child) for child in value.values())
    if isinstance(value, list):
        return any(has_private_text(child) for child in value)
    return False


def inspect_jsonl(path, expected):
    digest = hashlib.sha256()
    ids = set()
    counts = Counter()
    rows = variants = 0
    with gzip.open(path, "rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if has_private_text(row):
                raise ValueError(f"{path.name}: private path in row {rows}")
            if "context" in row or "title" in row or "original_answer" in row:
                raise ValueError(f"{path.name}: unneeded source field in row {rows}")
            for variant in row.get("variants", []):
                if any(key.startswith("original_") or key in {"context", "title"}
                       for key in variant):
                    raise ValueError(f"{path.name}: raw benchmark field in row {rows}")
            rid = row.get("reader_id") or row.get("knowledge_id")
            if not rid or rid in ids:
                raise ValueError(f"{path.name}: missing or duplicated Reader ID")
            ids.add(rid)
            counts[str(row.get("update_type", "unknown"))] += 1
            rows += 1
            variants += len(row.get("variants", []))
    if rows != expected["rows"] or variants != expected["variants"]:
        raise ValueError(f"{path.name}: row or variant count mismatch")
    if dict(sorted(counts.items())) != expected["update_types"]:
        raise ValueError(f"{path.name}: item type count mismatch")
    if digest.hexdigest() != expected["content_sha256"]:
        raise ValueError(f"{path.name}: uncompressed hash mismatch")
    return ids


def inspect_other_gzip(path, expected):
    digest = hashlib.sha256()
    lines = 0
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            if path.name.endswith(".jsonl.gz"):
                lines += chunk.count(b"\n")
    if digest.hexdigest() != expected["content_sha256"]:
        raise ValueError(f"{path.name}: uncompressed hash mismatch")
    if path.name.endswith(".jsonl.gz") and lines != expected["rows"]:
        raise ValueError(f"{path.name}: line count mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    root = args.data_root
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{name}: missing file or size mismatch")
        expected_hash = expected.get("compressed_sha256") or expected.get("sha256")
        if digest_file(path) != expected_hash:
            raise ValueError(f"{name}: file hash mismatch")

    ids = {name: inspect_jsonl(root / name, files[name]) for name in SOURCE_NAMES}
    all_ids = ids[SOURCE_NAMES[0]]
    train_ids = ids[SOURCE_NAMES[1]]
    test_ids = ids[SOURCE_NAMES[2]]
    if not train_ids <= all_ids or not test_ids <= all_ids or train_ids & test_ids:
        raise ValueError("Training and test IDs are not disjoint subsets of all items")

    query_ids = inspect_jsonl(root / "reader_test_queries_200.jsonl.gz",
                              files["reader_test_queries_200.jsonl.gz"])
    if query_ids != test_ids:
        raise ValueError("Fixed test queries do not match the test item IDs")

    for name in ("reader_teacher_train_68736.parquet.gz",
                 "reader_teacher_test_800.parquet.gz"):
        inspect_other_gzip(root / name, files[name])
    prompts = json.loads((root / "metaqueries224.json").read_text(encoding="utf-8"))
    if len(prompts["prompts"]) != 224:
        raise ValueError("Meta-query pool is not 224 prompts")
    schema = json.loads((root / "reader_schema_v3.json").read_text(encoding="utf-8"))
    if has_private_text(schema) or has_private_text(manifest):
        raise ValueError("Metadata contains a private machine path")
    print("Reader data release verified: 19,121 final items; "
          "17,184 training items; 200 test items; 224 meta-queries.")


if __name__ == "__main__":
    main()
