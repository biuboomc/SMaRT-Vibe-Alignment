#!/usr/bin/env python3
"""Build the public-safe Reader data bundle from the frozen run artifacts."""

import argparse
import gzip
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path


SOURCE_FILES = {
    "reader_source_all_19121.jsonl.gz": "reader_source_all_19121.jsonl.gz",
    "reader_source_train_17184.jsonl.gz": "reader_source_train_balanced24_17184_maxanswer512.jsonl.gz",
    "reader_source_test_200.jsonl.gz": "reader_source_test_200.jsonl.gz",
    "reader_test_queries_200.jsonl.gz": "test200_one_query.jsonl.gz",
}
VARIANT_FIELDS = (
    "question",
    "answer",
    "rewrite_variant_id",
    "behavior_variant_id",
    "sample_hash",
    "rewrite_sample_hash",
)
DROP_FIELDS = {"title", "context", "original_answer"}
PRIVATE_PATH = re.compile(r"(?i)(?:/mnt/shared-storage[^/]*/|[a-z]:\\Users\\[^\\]+\\)")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_private(value):
    if isinstance(value, str):
        return PRIVATE_PATH.search(value) is not None
    if isinstance(value, dict):
        return any(scan_private(child) for child in value.values())
    if isinstance(value, list):
        return any(scan_private(child) for child in value)
    return False


def clean_row(row):
    result = {key: value for key, value in row.items() if key not in DROP_FIELDS}
    if "variants" in result:
        variants = []
        for variant in result["variants"]:
            clean = {key: variant[key] for key in VARIANT_FIELDS if key in variant}
            if not clean.get("question") or not clean.get("answer"):
                raise ValueError("Variant lacks a question or answer")
            variants.append(clean)
        result["variants"] = variants
    if scan_private(result):
        raise ValueError("Release row contains a private machine path")
    return result


def gzip_writer(path):
    raw = path.open("wb")
    return raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0)


def build_jsonl(source, target):
    source_hash = hashlib.sha256()
    content_hash = hashlib.sha256()
    counts = Counter()
    rows = 0
    variants = 0
    raw, zipped = gzip_writer(target)
    try:
        with gzip.open(source, "rb") as input_stream:
            for source_line in input_stream:
                source_hash.update(source_line)
                row = clean_row(json.loads(source_line))
                line = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                zipped.write(line)
                content_hash.update(line)
                counts[str(row.get("update_type", "unknown"))] += 1
                rows += 1
                variants += len(row.get("variants", []))
    finally:
        zipped.close()
        raw.close()
    return {
        "rows": rows,
        "variants": variants,
        "update_types": dict(sorted(counts.items())),
        "original_sha256": source_hash.hexdigest(),
        "content_sha256": content_hash.hexdigest(),
        "compressed_sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def repackage_gzip(source, target, rows=None):
    content_hash = hashlib.sha256()
    count = 0
    raw, zipped = gzip_writer(target)
    try:
        with gzip.open(source, "rb") as input_stream:
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                zipped.write(chunk)
                content_hash.update(chunk)
                if rows is not None:
                    count += chunk.count(b"\n")
    finally:
        zipped.close()
        raw.close()
    if rows is not None and count != rows:
        raise ValueError(f"{source.name}: expected {rows} rows, found {count}")
    return {
        "rows": count if rows is not None else rows,
        "content_sha256": content_hash.hexdigest(),
        "compressed_sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging_root
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)

    files = {}
    for name, source_name in SOURCE_FILES.items():
        files[name] = build_jsonl(staging / source_name, output / name)

    extras = {
        "reader_teacher_train_68736.parquet.gz":
            ("reader_teacher_train_68736.sanitized.parquet.gz", None),
        "reader_teacher_test_800.parquet.gz":
            ("reader_teacher_test_800.sanitized.parquet.gz", None),
    }
    for name, (source_name, expected_rows) in extras.items():
        files[name] = repackage_gzip(staging / source_name, output / name, expected_rows)
    files["reader_teacher_train_68736.parquet.gz"]["rows"] = 68736
    files["reader_teacher_test_800.parquet.gz"]["rows"] = 800

    prompts = staging / "metaqueries224.json"
    prompt_data = json.loads(prompts.read_text(encoding="utf-8"))
    if len(prompt_data["prompts"]) != 224:
        raise ValueError("Expected 224 meta-queries")
    shutil.copyfile(prompts, output / "metaqueries224.json")
    files["metaqueries224.json"] = {
        "rows": 224,
        "sha256": sha256_file(output / "metaqueries224.json"),
        "bytes": (output / "metaqueries224.json").stat().st_size,
    }

    with gzip.open(staging / "reader_schema_v3.json.gz", "rt", encoding="utf-8") as stream:
        schema = json.load(stream)
    (output / "reader_schema_v3.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    files["reader_schema_v3.json"] = {
        "sha256": sha256_file(output / "reader_schema_v3.json"),
        "bytes": (output / "reader_schema_v3.json").stat().st_size,
    }

    original = json.loads((staging / "manifest_reader_v3.json").read_text(encoding="utf-8"))
    balanced = json.loads((staging / "manifest_balanced24_4epoch.json").read_text(encoding="utf-8"))
    maxanswer = json.loads((staging / "maxanswer512_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "release_schema": "mart-reader-data-v1",
        "reader_schema": original["schema_version"],
        "reported_reader_checkpoint_step": 2800,
        "base_model": "Qwen/Qwen3-14B",
        "contents": "Reader construction, training, and test inputs only; no model weights or intervention data",
        "source_counts": original["stats"]["all"]["update_type_counts"],
        "source_benchmarks": original["stats"]["knowledge"]["knowledge_benchmark_counts"],
        "behavior_categories": original["stats"]["behavior"]["behavior_category_counts"],
        "balanced_training": {
            "update_types": balanced["stats"]["balanced"]["update_type_counts"],
            "batch_mix": {key: balanced["settings"][key] for key in (
                "knowledge_per_batch", "behavior_per_batch", "no_op_per_batch",
                "random_per_batch", "batch_size"
            )},
            "answer_token_limit": maxanswer["answer_token_limit"],
            "variants_removed_by_answer_limit": maxanswer["removed_variant_count"],
        },
        "test_items": original["stats"]["test"]["update_type_counts"],
        "meta_query_pool_size": 224,
        "split_settings": {
            key: original["settings"][key] for key in (
                "split_seed", "split_strata", "test_seed", "test_selection",
                "test_per_type", "prompt_seed", "max_query_prompts"
            )
        },
        "original_input_sha256": {
            "knowledge": original["inputs"]["knowledge_jsonl"]["sha256"],
            "behavior": original["inputs"]["behavior_jsonl"]["sha256"],
            "metaqueries": original["inputs"]["prompt_file"]["sha256"],
            "train_teacher": balanced["artifacts"]["balanced_teacher"]["sha256"],
            "test_teacher": original["artifacts"]["teachers"]["test"]["sha256"],
        },
        "files": files,
    }
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: info.get("rows") for name, info in files.items()}, indent=2))


if __name__ == "__main__":
    main()
