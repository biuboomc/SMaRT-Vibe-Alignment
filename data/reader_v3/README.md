# Reader v3 Data

This directory contains the final MaRT Reader construction data, the balanced
training subset used for the reported step-2,800 checkpoint, and the held-out
200-item test set. It does not contain a per-step sampling schedule, generated
test responses, judge outputs, model weights, or MetaEdit intervention data.
The omission affects exact batch-order replay, not the released training
examples or teacher targets.

## Files

| File | Items or rows | Purpose |
| --- | ---: | --- |
| `reader_source_all_19121.jsonl.gz` | 19,121 | Audited knowledge and behavior items before balanced selection |
| `reader_source_train_17184.jsonl.gz` | 17,184 | Actual balanced training items, after the 512-token answer filter |
| `reader_teacher_train_68736.parquet.gz` | 68,736 | Four teacher-supervised meta-query rows per training item |
| `reader_source_test_200.jsonl.gz` | 200 | Held-out item descriptions and eight QA variants per item |
| `reader_teacher_test_800.parquet.gz` | 800 | Four teacher rows per test item |
| `reader_test_queries_200.jsonl.gz` | 200 | Fixed one-query-per-item test input |
| `metaqueries224.json` | 224 | Anchor-free meta-query pool |
| `reader_schema_v3.json` | - | Reader data schema |
| `release_manifest.json` | - | Counts, seeds, file sizes, and SHA-256 checksums |

All `.gz` files use ordinary gzip compression. Run
`python scripts/data/verify_release.py data/reader_v3` from the repository
root, then `python scripts/data/unpack_release.py data/reader_v3
--output-dir ./reader_data` to expand the train and test inputs.

## Construction and Splits

The audited pool has 9,148 knowledge items and 9,973 synthetic behavior
items. Balanced training uses 8,592 of each type. The held-out test has 100
of each type. A further 451 knowledge and 495 behavior items were placed in
the screening/validation split. From the remaining training pool, 5 knowledge
and 786 behavior items were reserved when balancing the two types. These
unused items remain in `reader_source_all_19121.jsonl.gz`; they are not in
the balanced training subset or held-out test. Selection seeds and source
hashes are in `release_manifest.json`.

The final training source removed five QA variants whose answers exceeded
the 512-token filter; it retains 137,467 QA variants. This is the source
variant used by the run that produced the reported Reader checkpoint.
Teacher rows and the 224 meta-queries are provided directly so that users
do not need to call a data-generation API to reconstruct the training
inputs.

## Source Attribution

Knowledge items were extracted from Humanity's Last Exam, SimpleQA,
FrontierScience, OpenBookQA, SciBench, TheoremQA, SciCode, and publicly
released FrontierMath examples. The 9,973 behavior items were synthesized
and audited across seven behavior categories. Per-benchmark and
per-category counts are recorded in `release_manifest.json`.

This bundle is a **content-preserving projection for Reader training**, not
a byte-for-byte copy of the private raw files. Explicit `original_*`
fields, source-context/title fields, and generation-workflow metadata were
omitted. The QA variants used by the Qwen QA
adapter builder, canonical targets, meta-query assignments, teacher
messages, teacher traces, row order, and IDs are retained.
The original input checksums and the release checksums are both recorded
for provenance.

The repository software licenses do not grant rights to third-party
benchmark content. This bundle is intended for research reproduction;
users should review the upstream terms before redistributing derived
records or combining them with raw benchmark files.
