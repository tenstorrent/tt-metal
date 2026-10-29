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
   without watcher instrumentation. BF16 DEST/output, UINT16 packed input,
   tiled I/O, interleaved L1, and tensor scale 0.8125 are fixed. By default,
   size the workload against free L1, reserving 64 KiB per bank for CBs,
   alignment, and runtime headroom. On the measured 130-core Blackhole this
   gives 16640x4096 values: every core handles 128 packed input tiles and
   512 output tiles. There are eight calls; discard the first two.
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

Capacity is queried before allocation; peak tensor allocation/free space is
recorded with the first output live. Each packed tile requires 2,048 B of input,
512 B of prepared scales, and 8,192 B of output. The measured tensor footprint
is 1,376,256 of 1,460,864 bytes per bank (94.2%), leaving 84,608 B. Current CBs
use 25,664 B per core. Work remains four output tiles per DEST acquire/init;
the larger workload increases the number of these batches per invocation.

Use `NVFP4_BENCH_PACKED_TILES_PER_CORE=64` for a smaller balanced workload, or
`NVFP4_BENCH_LEGACY=1` for the historical 2048x4096 workload. Explicit sizes
exceeding the budget fail before allocation. Shape, assignment, capacity, and
peak allocation are recorded as workload metadata, so unlike workloads cannot
be compared silently. The changed default requires fresh baseline measurements.
See [capacity measurements](BENCHMARK_L1_CAPACITY.md) for the size sweep and
variability. Sampled output checks/download happen after converter timing.

All tests use the trusted checkout. JUnit XML must show actual successful
execution, because Tracy can mask a pytest failure with exit code zero.
Missing/empty results, unexpected skips, missing profile operations, and
missing clock metadata reject the evaluation. Accuracy failures stop before
performance profiling. Each evaluation needs a new result directory so stale
artifacts cannot satisfy validation.

### Screens, repeated measurements, and loop accounting

`evaluate --screen` checks the complete SFPU reconstruction corpus and original
twelve tensor scales, but omits the precise/FPU/storage-baseline survey and broad
regressions. Its successful status is `screened`, distinct from `passed` full
validation. Full validation includes 140 operation cases: 134 passes and the six
documented exclusions, including the transpose/consumer suite. Screens cannot
replace full validation before promotion. Both paths share `TENSOR_SCALES` and
the same numerical thresholds.

Use `--perf-repeats 4` for confirmation: four eight-call benchmark tests produce
32 recorded operations. Discard the first two calls of **each** group, retaining
24 samples. Repetition count becomes workload metadata, so incompatible runs
cannot be compared silently. Actual clock/core-count and evaluator/source checks
remain mandatory; a different reported frequency rejects the comparison.

Each stage writes `timing.json` and `pytest_metrics.json`. Safe-runner timestamps
separate lock wait, preflight, test execution, and cleanup. Pytest reports expose
collection and setup/call/teardown durations and actual case outcomes. Failed
experiments and resets remain in the logs/counts. JIT time overlaps these phases;
dedicated `TT_METAL_LOG_JIT_TELEMETRY=1` audits can distinguish cached/cold builds.

```bash
python scripts/nvfp4_loop_stats.py generated/nvfp4_second --output generated/loop-stats.json
```

The summary distinguishes evaluator runs, source/encoding fingerprints, and
worktrees; counts pytest outcomes separately from scalar gate comparisons; and
reports cumulative queue/phase time. Concurrent workers' summed waiting time
must not be interpreted as elapsed loop time. Partial live artifacts are listed
and skipped until complete. See the [layout and throughput report](OPTIMIZATION_2026_10_29_LAYOUT_AND_HARNESS.md).

For offline bit-layout research, a worker manifest may specify
`input_bit_permutation`, a sixteen-element bijection: entry i identifies the
source bit placed at output bit i. The trusted plugin applies it only after
fast packing, leaves scale data unchanged, and the evaluator verifies that it
remains fixed. Shipping a different input representation also requires its
packing contract/tests; the retained implementation uses ordinary nibbles.

Pipeline cycles are derived from the profiler-reported clock and divided by
the largest per-core **output-tile** assignment. Four 32x32 BF16 output tiles
come from each packed UINT16 tile. The full pipeline includes reader/writer
NoC transfers, compact-scale expansion, format changes, output transpose, pack,
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
