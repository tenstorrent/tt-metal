# Input-bit construction and evaluator throughput

Budget: October 29, 17:32:31–18:32:31 UTC, from `6a808fc2e6b`.
Three experimental workers investigated retained packed words, alternative input
bit arrangements, and programmable constants/instruction scheduling. The
coordinator instrumented the evaluator and validated/integrated the selection.
All workers were sparse git worktrees sharing repository objects, runtime,
compiler, and Python environment. Hardware access remained serialized through
`scripts/run_safe_pytest.sh`; JIT caches and artifacts were isolated by worker.

## Retained implementation

The input remains four ordinary E2M1 nibbles per UINT16. No packing/API change
is required. The experimental BF16-DEST decoder now:

1. Uses opaque `SFPLOAD.HI16` to place the packed word in bits 16–31.
2. Aligns a selected nibble's sign to bit 31, then arithmetic-shifts right six.
3. Masks with `0x81c00000`, keeping sign and the three magnitude bits together.
4. Adds exponent bias 127 and evaluates the existing exact E2M1 LUT.

These are bit operations until exponent rebiasing; synthetic subnormal bit
patterns are not subjected to floating-point arithmetic/FTZ. The mask lives in
programmable Creg12, initialized once per kernel invocation. Scale expansion
uses ordinary LRegs and does not overwrite it. No `SFPSETSGN` is needed.

Planes 0–2 replay four instructions: arithmetic shift, AND, exponent adjustment,
and LUT. Plane 3 needs no left alignment shift; its arithmetic shift moves
outside replay to fill the multiply dependency gap. It re-records/reuses three
instructions. Math replay slots 0–3 remain disjoint from transpose's 16–31.
Next-word and next-face prefetching remain. No LoadMacro is used.

The decoder executes 296 SFPU instructions per output tile, averaged over the
four-plane sweep, plus 0.5 coefficient initialization instructions per tile.
Ordinary vectors execute nine instructions for planes 0–2 and eight for plane
3. The complete-sweep diagnostic measures about 298.38 cycles/output tile,
versus 362.11 before this loop.

For ordinary TILE output in L1 the writer issues two writes before its barrier.
Per-core counts are multiples of two/four packed planes, so there is no tail.
CB capacity is four/eight pages; advancing two pages guarantees a contiguous
two-page source region even at ring wrap. The barrier precedes CB release.
Transposed output and DRAM output use single writes: profiling caught
regressions when pairing scattered transposed tile IDs and no DRAM benefit.
Row-major output also retains single
tile staging/writes. Fixed two-page batching removes the dynamic tail branch;
its timing was not distinguishable from variable batching, so this cleanup is
justified by the count invariant rather than a claimed incremental speedup.

## Experiments

Representative screening medians, 2048x4096 BF16 output, BF16 DEST, L1 I/O,
130 available cores, SFPU tensor scaling, global scale 0.8125. Cycles are per
1024-element output tile. These early eight-call screens are not a ranking of
small differences; subsequent confirmation uses 32 calls.

| Variant | L1 pipeline cycles/tile | Complete-sweep math cycles/tile |
| --- | ---: | ---: |
| Previous retained decoder | 567.72 | 362.11 |
| Programmable mask only | 567.72 | — |
| Mask-free two-shift construction | 566.91 | — |
| Retained word, incremental nibble shifts, cached plane scales | 550.54 | — |
| Retained word plus direct field placement | 539.94 | — |
| Retained HI16 word, signed construction, cached scales | 541.00 | 284.39 |
| Standard packing, direct magnitude/sign placement | 544.87 | 338.12 |
| Standard packing, HI16 plus arithmetic-shift construction | 538.30 | 298.37 |
| Two sign/magnitude pairs aligned by offline bit permutation | 541.28 | — |
| Three aligned pairs plus adjusted LUT | 542.76 | 277.11 |
| Literal mantissa/exponent/sign bitplanes | 674.10 | — |
| Signed construction, writer batch four | 538.14 | 298.37 |
| Signed construction, writer batch two | 518–526 across repeats | 298.38 |

The literal bitplane layout gathered separated magnitude bits at runtime and
cost ten arithmetic instructions/vector, versus 6.75 for the earlier decoder.
Its approximately 106-cycle pipeline regression agrees with that instruction
increase. Other permutations and retained-word variants improved isolated
math but did not establish a faster complete pipeline. LUT coefficient/mask
co-location, NOINLINE sweeps, shorter/longer replay, and two staggering controls
also failed to establish an additional pipeline improvement.

Two transaction-ID writer pipelines hung at NoC write barriers. The safe runner
detected each hang, collected triage, and reset the device while retaining the
lock. Those variants were rejected; their failed tests and time remain counted.
The selected writer uses ordinary acknowledged writes/barriers.

## Confirmation and accuracy

Final confirmation repeats the eight-call workload four times, retaining calls
3–8 from each group: 24 measured observations from 32 calls. Warmups are removed
per group, not just once. The evaluator checks all 32 operation records, actual
core counts, reported clock, source provenance, and matching evaluator hashes.
JUnit evidence is required because Tracy can mask pytest's exit status.

Matching-1350-MHz preliminary confirmation pairs were 567.71→525.28 and
567.97→524.38 cycles/tile. One earlier selected profile reported 1343 MHz; its
comparison against a 1350-MHz baseline was rejected and is not used to claim a
speedup. Pipeline samples remain more variable than the math diagnostic;
reports retain the samples and MAD rather than quoting only the best call.

Final specialized-writer confirmation: two matched-1350-MHz pairs measured
567.09→521.02 and 567.03→520.47 cycles/tile, **8.13% and 8.21% less time**.
The post-formatting source-identical confirmation additionally measured
567.13→524.36 cycles/tile, 7.54% less time. Across these repeats the observed
median improvement is approximately 7.5–8.2%. The retained ordinary-L1 kernel takes about 24.9 us for 8,388,608 elements
on the measured 130-core board. An ideally balanced 120-core projection is
about 316 GElem/s, 633 GB/s of BF16 output, or 198 GB/s of packed input plus
prepared scales (0.625 bytes/element). Output-equivalent bandwidth exceeding
512 GB/s does not prove overlap with a DRAM reader or matmul sharing the cores;
that needs a fused measurement. See `accepted-comparisons.json` and
`final-memory-{full,repeat}` for final source-matching evidence.

The accuracy corpus, original twelve tensor scales, FTZ reference, and limits
are unchanged. A shared `TENSOR_SCALES` constant prevents drift between screen
and full validation. Each scale case requires maximum ULP <= 2, mean <= 1,
median <= 1, and SNR > 46 dB; zero references use ULP only. Observed maxima are
1.4375 ULP, mean 0.8167679554, median 0.75, and minimum SNR 46.2356515 dB.

Full validation now includes all 140 operation cases: 134 passes and six
expected UINT8/BF16-DEST skips. This includes nonsquare/batched transpose,
cache/trace, all 65,536 packed UINT16 words in both input layouts, and native
matmul consumers with BF16/FP32 matmul DEST. Hardware-free harness/parser
checks also pass. Final source hashes are checked against the measured worker.

Native-consumer results: for X[128,2048] @ W[2048,4096], captured
decode-plus-matmul FW span changed from 81.787 to 79.211 us for ordinary L1
weights (**3.15% less time**). Transposed L1 was 80.785→80.533 us; ordinary
DRAM 131.339→132.043 us; transposed DRAM 130.929→131.505 us. The latter
differences are under 0.6% and do not establish an improvement. Ordinary L1
is faster than the optional transposed-weight path in this workload after
these optimizations. All four final cases passed exact layout/consumer
comparisons, with matching 1350-MHz clocks and verified source hashes.
The unconditional-batching profiles that exposed regressions are retained
as rejected policies in `consumer/`.

## Loop accounting

Previous run: ten algorithmic designs, twenty full evaluator runs, sixty pytest
processes, 2,280 passing cases, and 120 expected skips, excluding its separate
consumer/harness checks. Its whole-loop elapsed time was 34.9 minutes, not a
full hour. Old artifacts did not record exact lock/JIT breakdowns.

This run: 60 evaluator runs (12 full passes, 46 screened runs, two rejected failures), 36 source/encoding fingerprints across 32 worktrees. The 128 evaluator stages recorded 1,638 passes, two failed cases, and 72 expected skips, with 362,053,632 scalar reconstruction comparisons. Auxiliary JUnit files recorded 42 passes; the 16 native-consumer profiling cases are separate in their logs. There were 138 timestamped safe-runner processes, holding the device lock for 27.2 minutes. Cumulative worker wait was 39.8 minutes and overlaps between workers. Evaluator phases included Python startup, fixture/device setup, host/JIT/reference calculations, and profiler post-processing; the JSON preserves their individual totals. Separate evaluator epochs are recorded in `throughput-epochs.json`.

Counts distinguish evaluations/retests from unique source-and-encoding
fingerprints and distinct worktrees. Fingerprints include baseline and final
formatting/integration snapshots; they are not a count of different algorithms.
Scalar reconstruction comparisons are reported separately from pytest cases.
The clock-rejected measurement and the two failed writer experiments remain
in the artifacts. Auxiliary consumer/harness suites are listed separately.

An uncrowded candidate screen takes approximately 17–18 seconds:

| Component | Accuracy stage | Profiling stage |
| --- | ---: | ---: |
| Python/Tracy startup | 1.3 s | 2.4 s |
| Fixture/device setup | 3.4 s | 3.4 s |
| Test call, including host work/JIT | 3.5 s cold; 0.34 s cached | 0.3 s |
| Post-session processing | 0.6 s | 2.1 s |

Eight benchmark kernel executions total only about 0.2 milliseconds. Reducing
hardware repetitions would save little compared with process/device setup.
Dedicated telemetry audits measured 5.7 seconds of cold JIT wall time, 66 linked
binaries, versus 0.0 seconds rounded for a cached run. Parallel build-step
durations and JIT time overlap setup/test-call phases and must not be added to
them. A broad 118-case accuracy run held the lock for 54–60 seconds before
queueing; the expanded final suite has 140 cases.

Multiple workers can wait concurrently. Summed lock-wait seconds are cumulative
across workers, not wall time lost from the one-hour budget. Device lock time
also includes profiler post-processing. Stage timing JSON separates queueing,
preflight, startup, collection, pytest setup/call/teardown, and post-session
work. Retaining a device across batches of candidates is a future opportunity;
this loop preserves the safe runner's lock/reset behavior.

## Reproduce

```bash
source source-env.sh
python scripts/nvfp4_harness.py prepare --path /workspace/nvfp4-worker
python scripts/nvfp4_harness.py evaluate --candidate /workspace/nvfp4-worker \
    --output generated/nvfp4_trial --screen
# Before promotion, full regressions and repeated profiling:
python scripts/nvfp4_harness.py evaluate --candidate /workspace/nvfp4-worker \
    --output generated/nvfp4_confirm --perf-repeats 4 --math-isolate
python scripts/nvfp4_loop_stats.py generated/nvfp4_second --output generated/loop-stats.json
```

Screens retain the complete SFPU reconstruction corpus: 520,192 values for
each of twelve scales, 6,242,304 comparisons per screen. They omit recomputation
of unrelated precise/FPU/storage baselines and broad layout regressions, which
remain mandatory before promotion. Research bit permutations are strict
bijections recorded in worker manifests and applied by a trusted pack hook;
the retained product path uses the standard nibble layout.

Artifacts: `generated/nvfp4_second/`, including `loop-stats.json`, per-evaluation
`timing.json`, pytest phase reports, JUnit XML, source patches/hashes, disassembly,
JIT audit logs, and consumer profiles. Raw rejected experiments are preserved.
