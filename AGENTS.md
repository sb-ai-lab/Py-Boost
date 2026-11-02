# Repository instructions

уточняй непонятные вопросы перед изменением кода

## Required local GPU benchmark

Before declaring a PR ready, run the complete `pyboost-delicious-paper-v1-k5`
benchmark described in `benchmarks/README.md` against the PR's exact base and
candidate commits, using the same harness, pinned environment and idle GPU.
Use all 25 method/fold cases: five methods, five shuffled train folds, the official
held-out test set, all 983 labels and `sketch_outputs=5` in every sketch case.
Use the paper's parameters, up to 20000 iterations and early stopping after 500
rounds. Each fold has one timing run after a 32-iteration warmup and a separate
full memory run. Five CV folds are not five timing repetitions. Do not substitute
synthetic data, truncate the iteration limit or drop methods/folds in a required
run. This protocol can take days; there is no ten-minute limit.

- Commit library changes first. Record both code SHAs and the benchmark/data
  checksums. Verify that the imported library comes from the intended checkout.
- Run `benchmarks/run.py run` with both source paths, an explicit device and a new
  output directory. A missing GPU, failed case, timeout or OOM is not a pass.
- Use exactly one physical GPU. Never run benchmark cases or versions in parallel
  on multiple GPUs. Publish its model, VRAM, UUID, compute capability and driver
  versions in the report, plus full output count K and sketch width k per case.
- Use `benchmarks/run.py compare` to produce the report; do not compare mismatched
  environments/workloads. Inspect time, peak live CuPy bytes, CPU RSS, holdout
  quality, executed iterations and retained tree counts.
- When PR creation/update is authorized, publish the comparison table and run
  configuration in a PR comment. Keep raw JSON, logs and archives locally in the
  ignored output directory. Do not create result branches or upload artifacts
  unless explicitly requested. For local-only work, provide saved report paths.
- After changing the library code, rerun against the new candidate SHA. An earlier
  report does not verify later code changes. For a change to the harness itself,
  rerun both library versions with the final harness implementation.
- Explicitly report regressions, failures, omitted cases and unavailable hardware.
  Preserve completed work and evidence if a run cannot finish; do not describe
  such a PR as benchmark-verified. Request an explicit user waiver when needed.

The frozen benchmark is for version-to-version comparisons. Keep its cases,
data, splits and measurement semantics stable; introduce a new suite version when
these semantics change. GPU unit tests remain required and do not replace it.
