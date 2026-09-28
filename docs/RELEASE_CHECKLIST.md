# Before Making This Repository Public

The local Reader release contains source code and the final training and
held-out test inputs. It is intentionally not pushed by the packaging tools.

1. Run `python scripts/data/verify_release.py data/reader_v3` and the
   standard-library tests in `tests/test_data_release.py`.
2. Review the upstream terms for the eight benchmarks listed in
   `data/reader_v3/README.md`. The repository's MIT and Apache-2.0
   software licenses do not license benchmark-derived records.
3. Inspect the diff for paths, identities, credentials, generated logs,
   and unintended files. The release excludes raw benchmark context fields,
   checkpoints, test generations, judge outputs, and MetaEdit applications.
4. Check the rendered README and verify every compressed file remains below
   GitHub's 100 MB single-file limit. The current largest file is under
   50 MiB.
5. Do not simply change the visibility of the existing private GitHub
   repository while double-blind review matters. Its existing Git commit
   records an identifiable author, and a deleted source manifest remains
   recoverable from Git history. Publish a clean snapshot in a fresh history
   when appropriate, or review and intentionally rewrite history first.
6. Decide when to update the private GitHub repository. Its existing
   anonymous mirror may reflect a push before the repository itself is
   made public.

The step-2,800 Reader checkpoint, full training launch configuration for a
particular cluster, and intervention pipelines are not part of this source
and data release.
