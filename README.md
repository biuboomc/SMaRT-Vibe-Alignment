# Imprint Reader

[![Paper](https://img.shields.io/badge/arXiv-2609.35261-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.35261) [![Hugging Face](https://img.shields.io/badge/Imprint%20Reader%20v1.0--0928-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/quantumfr/imprint-reader-v1.0-0928)

Imprint Reader studies whether a model can describe changes carried by a
weight update. Semantic Mount-and-Read Tuning (SMaRT) freezes an item-specific update,
mounts it onto a Reader with aligned parameter coordinates, and trains the
Reader to describe the induced factual knowledge or behavioral tendency.
No-change and nonzero random-update episodes provide control targets.

This repository contains the final Reader training and held-out test data,
data preparation scripts, and an installable verl-derived Reader runtime. The
paper reports the Reader checkpoint after 2,800 updates. MetaEdit
interventions are outside the scope of this release.

## Overview

- **Data:** 9,148 knowledge and 9,973 behavior items after audit, including
  the 17,184-item balanced training subset and 200 held-out test items.
- **Training:** 24 knowledge, 24 behavior, 8 no-change, and 8 random-update
  episodes per global batch of 64.
- **Readout:** four anchor-free meta-queries per item, drawn from a
  224-query pool. Teacher targets are canonical fact or behavior statements.

## Repository Map

| Path | Contents |
| --- | --- |
| [`data/reader_v3/`](data/reader_v3/README.md) | Final Reader data, test set, prompt pool, and checksums |
| `scripts/data/` | Build, verify, and unpack the data release |
| `scripts/knowledge_update/` | Knowledge/behavior preparation and Reader materialization |
| [`runtime/`](runtime/README.md) | Full installable Reader runtime source |
| `overlays/verl/` | Focused Reader replacements and CPU tests |
| `configs/reader_v3_balanced24.reference.yaml` | Reference training settings |

## Getting Started

The data tools use only the Python standard library. Verify the checked-in
files before using them:

```bash
python scripts/data/verify_release.py data/reader_v3
python scripts/data/unpack_release.py data/reader_v3 --output-dir ./reader_data
```

The second command expands the actual balanced train source and teacher
table, the held-out test source and teacher table, and the fixed
one-query-per-item test input. Add `--include-all` to expand the complete
19,121-item pre-split source. Existing files are never overwritten.
`scripts/data/build_release.py` is a maintainer tool that expects the private
frozen input snapshot; ordinary users only need the verify and unpack steps.

For training in a prepared GPU environment, install the included runtime:

```bash
pip install -e ./runtime
```

The runtime requires the appropriate PyTorch/CUDA and inference-engine stack;
see [`runtime/README.md`](runtime/README.md) for its scope.

## Data

| Split or artifact | Knowledge | Behavior | Total |
| --- | ---: | ---: | ---: |
| Audited source items | 9,148 | 9,973 | 19,121 |
| Balanced training items | 8,592 | 8,592 | 17,184 |
| Held-out test items | 100 | 100 | 200 |
| Training teacher rows | 34,368 | 34,368 | 68,736 |

The final training source is the answer-length-filtered version used by the
reported Reader run. Five overlong behavior QA variants were removed, leaving
137,467 training variants. The reported checkpoint is step 2,800.
The per-step item-index table is deliberately not distributed; the released
items and teacher targets are complete, but an exact replay of batch order
requires regenerating that table from the documented seeds.

The release retains the QA variants, canonical targets, meta-queries,
split IDs, and teacher targets needed for Reader training and testing. It
omits explicit source-context and `original_*` metadata fields and
machine-specific paths. [The data card](data/reader_v3/README.md) describes every file, its
provenance, and the resulting difference from the private raw snapshot.

## Reader Runtime

`runtime/` is a source-only copy of the complete verl-derived Reader runtime;
`overlays/verl/` keeps the focused replacement files and CPU tests visible.
The adapter builder is the version used by the reported run, including an
effective nonzero random LoRA control. The reference configuration records
Qwen3-14B, BF16, supervised cross-entropy, learning rate `1e-4`, a cosine
schedule, and a `0.1` warmup ratio. Site-specific distributed launch
settings, Qwen3-14B weights, and GPU resources are still required for a full
training replay. No checkpoint, API credentials, logs, raw third-party
benchmark dumps, or intervention pipelines are included.

## License and Acknowledgements

Original scripts and documentation are MIT-licensed; the full verl-derived
runtime and overlay retain Apache-2.0 licensing. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
The data bundle is not covered by the software licenses. It contains
model-generated derivatives of the cited source benchmarks; users must respect
the applicable upstream dataset terms. We thank the verl project and the
benchmark creators.

Before changing repository visibility, review the
[`public-release checklist`](docs/RELEASE_CHECKLIST.md).
