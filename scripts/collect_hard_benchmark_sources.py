from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


FRONTIERMATH_URL = "https://epoch.ai/frontiermath/benchmark-problems"


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        data = data.strip()
        if data:
            self.parts.append(data)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def collect_openbookqa() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        path = hf_hub_download(
            repo_id="allenai/openbookqa",
            filename=f"additional/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        for index, row in df.iterrows():
            raw = {k: jsonable(v) for k, v in row.to_dict().items()}
            rows.append(
                {
                    "benchmark": "OpenBookQA",
                    "source_id": f"openbookqa_{split}_{raw.get('id', index)}",
                    "source_index": len(rows),
                    "split": split,
                    "question": raw.get("question_stem"),
                    "choices": raw.get("choices"),
                    "answer": raw.get("answerKey"),
                    "fact1": raw.get("fact1"),
                    "humanScore": raw.get("humanScore"),
                    "clarity": raw.get("clarity"),
                }
            )
    return rows


def collect_theoremqa() -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id="TIGER-Lab/TheoremQA",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        raw = {k: jsonable(v) for k, v in row.to_dict().items()}
        rows.append(
            {
                "benchmark": "TheoremQA",
                "source_id": f"theoremqa_{index:04d}",
                "source_index": index,
                "question": raw.get("Question"),
                "answer": raw.get("Answer"),
                "answer_type": raw.get("Answer_type"),
                "picture_present": raw.get("Picture") is not None,
            }
        )
    return rows


def collect_scicode() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, filename in [("dev", "problems_dev.jsonl"), ("test", "problems_test.jsonl")]:
        path = hf_hub_download(repo_id="Zilinghan/scicode", filename=filename, repo_type="dataset")
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                rows.append(
                    {
                        "benchmark": "SciCode",
                        "source_id": f"scicode_{split}_{raw.get('problem_id', len(rows))}",
                        "source_index": len(rows),
                        "split": split,
                        "problem_name": raw.get("problem_name"),
                        "problem_id": raw.get("problem_id"),
                        "problem_description_main": raw.get("problem_description_main"),
                        "problem_io": raw.get("problem_io"),
                        "required_dependencies": raw.get("required_dependencies"),
                        "sub_steps": raw.get("sub_steps"),
                        "general_solution": raw.get("general_solution"),
                        "general_tests": raw.get("general_tests"),
                    }
                )
    return rows


def collect_scibench() -> list[dict[str, Any]]:
    files = [
        "atkins.json",
        "atkins_sol.json",
        "calculus.json",
        "calculus_sol.json",
        "chemmc.json",
        "chemmc_sol.json",
        "class.json",
        "class_sol.json",
        "diff.json",
        "diff_sol.json",
        "fund.json",
        "fund_sol.json",
        "matter.json",
        "matter_sol.json",
        "quan.json",
        "quan_sol.json",
        "stat.json",
        "stat_sol.json",
        "thermo.json",
        "thermo_sol.json",
    ]
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for filename in files:
        path = hf_hub_download(repo_id="xw27/scibench", filename=filename, repo_type="dataset")
        data = json.load(open(path, "r", encoding="utf-8"))
        for raw in data:
            key = (str(raw.get("source")), str(raw.get("problemid")), str(raw.get("problem_text"))[:200])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "benchmark": "SciBench",
                    "source_id": f"scibench_{raw.get('source','unknown')}_{len(rows):04d}",
                    "source_index": len(rows),
                    "file": filename,
                    "problem_text": raw.get("problem_text"),
                    "answer_latex": raw.get("answer_latex"),
                    "answer_number": raw.get("answer_number"),
                    "unit": raw.get("unit"),
                    "source": raw.get("source"),
                    "problemid": raw.get("problemid"),
                    "comment": raw.get("comment"),
                    "solution": raw.get("solution"),
                }
            )
    return rows


def collect_frontiermath_public() -> list[dict[str, Any]]:
    html = urllib.request.urlopen(FRONTIERMATH_URL, timeout=90).read().decode("utf-8", "replace")
    parser = TextParser()
    parser.feed(html)
    lines = [x for x in parser.parts if x.strip()]
    text = "\n".join(lines)
    # Public page has 12 sample problems. Splitting on Answer markers is robust enough for source chunks;
    # the generator will extract clean QA from each chunk.
    anchors = [m.start() for m in re.finditer(r"\nAnswer:?\n", text)]
    chunks: list[str] = []
    last = 0
    for pos in anchors:
        start = max(0, text.rfind("\nTier ", 0, pos))
        if start < last:
            start = last
        end_candidates = [m.start() for m in re.finditer(r"\nAnswer:?\n", text[pos + 1 :])]
        chunks.append(text[start : pos + 6000])
        last = pos + 1
    # Deduplicate heavily overlapping chunks.
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk = re.sub(r"\n{3,}", "\n\n", chunk).strip()
        if len(chunk) < 500:
            continue
        key = chunk[:1000]
        if key in seen:
            continue
        seen.add(key)
        tier_match = re.search(r"Tier\s+([1-4])", chunk)
        out.append(
            {
                "benchmark": "FrontierMathPublic",
                "source_id": f"frontiermath_public_{len(out):02d}",
                "source_index": len(out),
                "url": FRONTIERMATH_URL,
                "tier": tier_match.group(1) if tier_match else None,
                "raw_public_problem_chunk": chunk[:30000],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for collector in [
        collect_frontiermath_public,
        collect_scicode,
        collect_openbookqa,
        collect_scibench,
        collect_theoremqa,
    ]:
        got = collector()
        print(json.dumps({"event": "collected", "collector": collector.__name__, "rows": len(got)}, ensure_ascii=False), flush=True)
        rows.extend(got)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output)
    print(json.dumps({"event": "done", "rows": len(rows), "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
