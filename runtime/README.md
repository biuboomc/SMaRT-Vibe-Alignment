# Reader Runtime

This directory is a source-only snapshot of the verl-derived runtime used for
MaRT Reader training. It includes the full `verl` Python package and the
active adapter-builder script, so it can be installed without locating a
separate private fork:

```bash
pip install -e ./runtime
```

Install the appropriate PyTorch, CUDA, and inference-engine dependencies for
your machine first. The released data can be verified and unpacked with the
standard-library tools in `../scripts/data/`; GPU dependencies are not needed
for those steps.

The active Reader entry points include
`scripts/build_ephemeral_lora_adapters.py` and the Reader trainer under
`verl/trainer/ppo/`. The training mix and hyperparameters are recorded in
`../configs/reader_v3_balanced24.reference.yaml`. That YAML is a reference,
not a cluster-specific launcher. Site-specific distributed launch settings,
model files, and checkpoints are not included. The exact per-step item-index
table is also omitted from the data release and must be regenerated before
using the schedule-strict trainer configuration.

The no-change control zeros the trainable LoRA parameters. The independent
random control explicitly reinitializes them to a nonzero update: matrix
parameters use Kaiming-uniform initialization and one-dimensional parameters
use a zero-mean normal distribution with standard deviation 0.02. This
control is not norm-matched to a knowledge or behavior update.

The runtime derives from verl and remains under Apache-2.0. See `LICENSE`
and the repository's `THIRD_PARTY_NOTICES.md`.
