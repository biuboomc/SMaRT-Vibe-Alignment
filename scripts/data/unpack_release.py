#!/usr/bin/env python3
"""Expand Reader training and test inputs without overwriting existing files."""

import argparse
import gzip
import shutil
from pathlib import Path


FILES = {
    "reader_source_train_17184.jsonl.gz": "train/reader_source_train_17184.jsonl",
    "reader_teacher_train_68736.parquet.gz": "train/reader_teacher_train_68736.parquet",
    "reader_source_test_200.jsonl.gz": "test/reader_source_test_200.jsonl",
    "reader_teacher_test_800.parquet.gz": "test/reader_teacher_test_800.parquet",
    "reader_test_queries_200.jsonl.gz": "test/reader_test_queries_200.jsonl",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-all", action="store_true",
                        help="Also expand the 19,121-item pre-split source file.")
    args = parser.parse_args()
    files = dict(FILES)
    if args.include_all:
        files["reader_source_all_19121.jsonl.gz"] = "reader_source_all_19121.jsonl"
    targets = [args.output_dir / relative for relative in files.values()]
    targets += [args.output_dir / name for name in (
        "metaqueries224.json", "reader_schema_v3.json", "release_manifest.json"
    )]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {existing[0]}")
    for name, relative in files.items():
        source = args.data_root / name
        target = args.output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(source, "rb") as input_stream, target.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
        print(target)
    for name in ("metaqueries224.json", "reader_schema_v3.json",
                 "release_manifest.json"):
        target = args.output_dir / name
        shutil.copyfile(args.data_root / name, target)
        print(target)


if __name__ == "__main__":
    main()
