# NVFP4 optimization harness

The agent uses `scripts/nvfp4_harness.py` to create a kernel-only worktree,
evaluate it, and compare results. The driver does not invoke an LLM or start
an unbounded search. A supervising agent chooses edits and a time budget.
Every hardware invocation goes through `scripts/run_safe_pytest.sh`.

## Source and runtime isolation

`prepare` makes a sparse, detached Git worktree containing only the operation's
kernel directory. Existing CLAUDE/AGENTS instruction files are symlinked;
no instruction files, Python environments, toolchains, or runtime binaries
are copied. Each worker has its own generated JIT cache. The shared checkout
supplies the compiled runtime, Python code, tests, and safe runner.

The candidate may edit C++ kernel sources/headers under
`ttnn/cpp/ttnn/operations/experimental/dequantize_nvfp4/device/kernels`.
Changes outside that directory are rejected. Host packing, public API,
program factory, and the evaluator stay fixed. Kernel symlinks are rejected
in candidates. These are operational scope checks, not a process sandbox.

Kernel lookup checks the process working directory before
`TT_METAL_KERNEL_PATH`. The pytest plugin enters the sparse worktree before
test/device fixtures run, while `TT_METAL_RUNTIME_ROOT` remains the shared
runtime root. The evaluator checks the candidate header hash against the
accuracy report and verifies the measured JIT translation unit includes the
candidate's absolute compute-source path. It also checks source/evaluator
hashes did not change during evaluation.

## Evaluation stages

1. **Accuracy and functionality.** Run the reconstruction survey and operation
   tests with `--dev`. The SFPU path uses the agreed max/mean/median ULP and
   SNR gate. Its older exact tie-away assertion has been replaced with that
   gate. The FPU path retains its separate tolerance. Packing, cache, trace,
   streaming, API validation, and ordinary output layout remain checked.
   The six documented UINT8/BF16-DEST exclusions are the only permitted skips.
2. **Production L1-to-L1 timing.** Run the actual TTNN operation through Tracy,
   without watcher instrumentation. The fixed workload has BF16 DEST/output,
   UINT16 packed input, tiled I/O, interleaved L1, tensor scale 0.8125, and
   2048x4096 decoded values. There are eight calls; discard the first two.
3. **Optional isolated-math diagnostic.** In a private source overlay, repeat
   the complete four-plane/four-face decode sweep 128 times after one scale
   expansion. This preserves scale-register state carried between faces;
   repeating an individual face is invalid for such candidates. Reuse candidate
   headers/readers/writers by symlink. The production candidate is not edited.
   Divide the math-thread duration by repetitions and output tiles per core.
   This is a `MATH_ISOLATE`-style amortized DEST-to-DEST diagnostic using Tracy,
   not the separate LLK harness's literal `PerfRunType::MATH_ISOLATE`. Input
   staging, scale expansion, transpose and packing are amortized, not absent.
   Repetition can change compiler constant-hoisting/scheduling; only the
   production L1-to-L1 measurement ranks candidates.

All tests use the trusted checkout. JUnit XML must show actual successful
execution, because Tracy can mask a pytest failure with exit code zero.
Missing/empty results, unexpected skips, missing profile operations, and
missing clock metadata reject the evaluation. Accuracy failures stop before
performance profiling. Each evaluation needs a new result directory so stale
artifacts cannot satisfy validation.

Pipeline cycles are derived from the profiler-reported clock and divided by
the largest per-core **output-tile** assignment. Four 32x32 BF16 output tiles
come from each packed UINT16 tile. The full pipeline includes reader/writer
NoC transfers, scale-row broadcasts, format changes, output transpose, pack,
and synchronization. Offline host preparation and JIT compilation are outside
the reported kernel duration.

Results include a JSON manifest, source/evaluator hashes, candidate patch,
accuracy reports, JUnit XML, commands, logs, raw profiler reports, measured
compute-source path, and math-thread disassembly. The compiler/runtime are
shared with production. The safe runner provides device serialization and
operation-hang recovery; a stage timeout also limits compiler/host stalls.
If forcibly terminated, its dirty marker lets the next safe invocation reset
the device before reuse.

## Commands

Run from the trusted checkout with its existing build/environment:

```bash
source source-env.sh
python scripts/nvfp4_harness.py prepare --path /tmp/nvfp4-baseline
python scripts/nvfp4_harness.py prepare --path /tmp/nvfp4-candidate

python scripts/nvfp4_harness.py evaluate \
    --candidate /tmp/nvfp4-baseline \
    --output generated/nvfp4_loop/baseline --math-isolate

# The agent edits only kernel files in /tmp/nvfp4-candidate.
python scripts/nvfp4_harness.py evaluate \
    --candidate /tmp/nvfp4-candidate \
    --output generated/nvfp4_loop/candidate-001 --math-isolate

python scripts/nvfp4_harness.py compare \
    --baseline generated/nvfp4_loop/baseline/result.json \
    --candidate generated/nvfp4_loop/candidate-001/result.json
```

`--accuracy-only` validates a candidate without profiling. The timeout defaults
to 600 seconds per safe-runner invocation and can be set with
`--timeout-seconds`. Hardware performance runs deliberately clear inherited
simulator, slow-dispatch, watcher, and counter settings. Parallel simulator
verification is separate; simulator timer values are not performance scores.

## Agent loop and promotion

Record a hypothesis, edit the worker, evaluate, and retain or reject the patch.
Keep the independent evaluator unchanged during a run. Store failed results,
not only winners. Compare uses matching evaluator hashes, workload metadata,
and device frequency. It reports a screening improvement only if saved cycles
exceed both 1% of baseline time and three times either run's median absolute
deviation. This is a conservative screening heuristic, not a confidence test.

Confirm promising candidates with independent paired hardware remeasurement
(baseline, candidate, baseline and, when needed, another candidate run) before
promoting them. The CLI never merges or commits a candidate automatically.
An agent should stop when its explicit time/candidate budget is reached and
report the best confirmed passing result; it should leave the runtime branch
and evaluator intact when an experiment fails.

Before searching, verify an unchanged worktree passes and a deliberately bad
candidate is rejected before profiling. Hardware-free parser checks live in
`tests/scripts/test_nvfp4_harness.py`; use the safe runner for those too.


The [October 29 run report](OPTIMIZATION_2026_10_29.md) records the first
nine-candidate search, canary rejection, paired measurements, and retained
implementation. Its generated evidence is under `generated/nvfp4_loop/`.
