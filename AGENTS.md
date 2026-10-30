# Repository instructions

уточняй непонятные вопросы перед изменением кода

## Required local GPU benchmark

Before declaring a PR ready, run the complete `pyboost-gpu-v1-k5` benchmark described
in `benchmarks/README.md` against the PR's exact base and candidate commits, using
the same harness, pinned environment and idle GPU. Use all 12 cases, one warmup
and five timing repetitions, with `sketch_outputs=5` in every sketch case; keep
memory runs separate. Do not replace the suite
with a single sketch method, a small-output case, or an improvised timing script.

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
generation and measurement semantics stable; introduce a new suite version when
these semantics change. GPU unit tests remain required and do not replace it.
