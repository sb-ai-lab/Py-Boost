# Delicious GPU benchmark

The default `pyboost-delicious-paper-v1-k5` compares library revisions on the real
**Delicious** multilabel dataset from the
[SketchBoost paper, Appendix B](https://arxiv.org/html/2211.12858v1#A2).
It follows the paper's evaluation protocol and chosen hyperparameters, without
rerunning hyperparameter search or competing libraries. The full comparison can take days on one GPU, especially the unsketched baseline
with separate memory fits; the old ten-minute synthetic budget no longer applies.

## Frozen data and protocol

Delicious has **16105 rows, 500 features and 983 binary labels**. Download the
original 1.3 MB archive selected by the
[paper's download script](https://github.com/sb-ai-lab/SketchBoost-paper/blob/e479fde7a0dd5f624fd9efd654e2d192a12d325d/get_data.sh),
not Delicious-200K. The harness checks the archive and every input file's SHA-256
before parsing dense float32 arrays. The NPZ cache is loaded with
`allow_pickle=False`; no external pickle is loaded. Preparation is outside timing.

Following the [reference preprocessing](https://github.com/sb-ai-lab/SketchBoost-paper/blob/e479fde7a0dd5f624fd9efd654e2d192a12d325d/preprocess.py),
use column zero of the official split files, preserving order and converting
one-based indices. There are **12920 train / 3185 test** rows. Following the
[reference runner](https://github.com/sb-ai-lab/SketchBoost-paper/blob/e479fde7a0dd5f624fd9efd654e2d192a12d325d/runner.py),
apply `KFold(5, shuffle=True, random_state=42)` only to train: every fold has
**10336 train / 2584 validation** rows. All five models are independently evaluated
on the untouched 3185-row test set. Do not average their predictions into an
ensemble. Model seed is 42; CPU thread count is eight.

Archive SHA-256:
`9a4e79e9bd1835225e59eef09ba0ee29c5430f785045a9245943511836542806`.
Normalized float32 data and ordered split SHA-256:
`a53ca49cdce00c38f90225994ba01e6b352e9445d121df35fcd177a1cbd5c676`.
Every worker verifies the normalized cache and records a fold checksum covering
the actual dataset plus ordered train, validation and test indices.

The frozen manifest is `suite-delicious-paper-v1-k5.json`:

| Parameter | Value |
|---|---:|
| Loss / validation metric | binary cross-entropy over all labels and rows |
| Maximum iterations | 20000 |
| Learning rate | 0.015 |
| Early stopping patience | 500 |
| Maximum depth | 12 |
| Maximum bins | 256 |
| Minimum data in leaf | 76 |
| Row sampling | 0.623613094287341 |
| L2 leaf regularization | 13.057198025783617 |
| Column sampling | 1.0 |
| `use_hess` | false |

The unrounded parameters are copied into plain JSON from the Delicious/CatBoost
entry of the paper's
[`runs/baselines_and_params.pkl`](https://github.com/sb-ai-lab/SketchBoost-paper/blob/e479fde7a0dd5f624fd9efd654e2d192a12d325d/runs/baselines_and_params.pkl).
Sharing CatBoost's parameters with SketchBoost and disabling Hessians in split
scoring follows the experiment code. The tested checkout supplies the actual
sketch implementations: this compares library revisions, rather than pinning the
2022 library. No hyperparameter search runs during PR checks.

| Method | Outputs K | Sketch width k | Coverage |
|---|---:|---:|---|
| Full GradientBoosting, single output group | 983 | all 983 | Paper baseline: no sketch |
| SketchBoost/topk | 983 | 5 | Paper: Top Outputs |
| SketchBoost/rand | 983 | 5 | Paper: Random Sampling |
| SketchBoost/proj | 983 | 5 | Paper: Random Projections |
| SketchBoost/filter | 983 | 5 | Additional library coverage, outside the paper |

There are **25 cases**: five methods × five folds. `sketch_outputs=5` affects only
structure search; leaves and predictions retain all 983 labels. `topk` chooses
five distinct labels; `rand`/`filter` make five draws with replacement (at most
five distinct labels); `proj` creates five components. The paper also sweeps other
k values; this required profile fixes k=5 as requested.

## Run and resume

Use Linux, Python 3.12 and CUDA 12 with the pinned dependency closure. The harness
is independent of tested checkouts, selected by `--source`.

```bash
uv venv --python 3.12 .venv-benchmark
uv pip install --python .venv-benchmark/bin/python -r benchmarks/requirements-py312-cu12.txt
uv pip install --python .venv-benchmark/bin/python --no-deps -e .
.venv-benchmark/bin/python benchmarks/run.py prepare-data
.venv-benchmark/bin/python benchmarks/run.py run \
  --source /path/to/base --source /path/to/candidate \
  --device 1 --output .benchmark-results/delicious-comparison
```

For offline preparation, use `prepare-data --archive /path/to/Delicious.zip`.
A different archive is rejected. Pass `--data-dir /path/to/cache` to preparation
and run when using a nondefault cache. Workers never download data implicitly.

Exactly **one physical GPU** runs both versions, all methods/folds and measurement
phases sequentially. Select an index or UUID; only that UUID is exposed as logical
CUDA device zero. Multiple visible GPUs or other compute processes on the GPU
are rejected. Reports include model, physical/CUDA-visible VRAM, UUID, PCI bus,
compute capability, SM count, memory bus width, power, maximum SM clock and
NVIDIA/CUDA driver/runtime versions. Use the same idle GPU and environment for
both versions. GitHub Actions does not need a GPU worker.

Commit library changes before running. Reports record source SHAs, imported paths
and source checksums; cases alternate version order. Each method/fold is saved in
`<output>/<commit>/results.json` alongside raw worker jobs, results and logs.
Default timeout is **72 hours per worker**, including warmup, timing and memory.
Failed cases, OOM and timeouts are failures, never skipped passes.

Repeat the same command with `--resume` to skip completed cases and retry unfinished
or failed cases. It refuses different source SHAs, suite, harness, environment,
case selection or repeat counts. Stop the original run before resuming. Without
`--resume`, output must be empty. Per-worker logs show progress every 500 iterations.

`--case delicious-proj-fold0` is a diagnostic subset and never fulfills the full
requirement. `--repeats` changes repetitions **within each fold**; the default is
**one**, not five. All five CV folds are required.

## Measurement and publication

Each version/method/fold runs in a fresh subprocess. Warmup uses 32 iterations on
that fold and is excluded from timing. It does not run 20000 warmup iterations.

1. Fit one fresh model with up to 20000 iterations and validation early stopping.
   Time fit and test prediction separately using CUDA synchronization and
   `perf_counter`. Fit includes quantization, transfers and validation.
2. Score test predictions outside timers: binary cross-entropy averaged over
   labels/rows, clipping probabilities to `[1e-7, 1-1e-7]` as in the paper. Use
   bounded float64 accumulation. Test data never control stopping.
3. Fit and predict with another fresh model for memory measurement. Hooks and RSS
   sampling are disabled during timing. Record peak live rounded CuPy allocations
   (including Py-Boost's private fit pool) and CPU RSS sampled every 5 ms. GPU peaks
   exclude context, free pool cache and external allocations; they are not total
   process VRAM. CPU RSS includes resident input data and can miss short peaks.

The summary reports **five-fold means ± sample standard deviations (`ddof=1`)** for fit,
prediction and test log loss, plus maximum fit GPU allocation across folds. JSON
preserves each fold's timing/memory, test quality, executed iterations and retained
trees for both phases. Folds are different datasets, not five repeats of one fit.
The per-fold table and iteration/tree counts follow the summary. Faster convergence
must be assessed together with work performed and quality.

Two-source runs generate `comparison.md`. Saved reports can be compared with:

```bash
.venv-benchmark/bin/python benchmarks/run.py compare \
  .benchmark-results/run/BASE_SHA/results.json \
  .benchmark-results/run/CANDIDATE_SHA/results.json \
  --output .benchmark-results/run/comparison.md
```

Comparison rejects mismatched workloads, datasets, environments and failed runs.
Partial runs are marked `full_suite: false`. Measurements are not automatic
statistical verdicts or a claim of exactly matching the paper's scores/timings on
different hardware/software. Repeated runs can vary with GPU clocks and temperature.

Publish the table and run configuration in a PR comment. Keep raw JSON, logs,
archives and cached data locally in ignored directories. Do not create result
branches or upload artifacts unless explicitly requested. The runner performs no
network writes; only `prepare-data` downloads the pinned archive.

Legacy synthetic workloads remain selectable with `--suite
benchmarks/suite-v1-k5.json` or `--suite benchmarks/suite-v1.json` (k=8); see
[SYNTHETIC.md](SYNTHETIC.md). They do not fulfill the Delicious requirement. Frozen
workloads, data, splits or measurement changes require a new suite ID and rerunning
both library versions with the same final harness.
