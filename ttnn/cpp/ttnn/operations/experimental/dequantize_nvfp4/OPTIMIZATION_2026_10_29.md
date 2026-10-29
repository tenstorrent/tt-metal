# First bounded NVFP4 optimization run

The run used a 30-minute budget, sparse kernel-only worktrees, the shared
runtime/SFPI compiler, and `scripts/run_safe_pytest.sh` for every hardware
invocation. The independent accuracy gate and benchmark workload stayed
fixed during the search. The harness is committed as `978cc7cd67f`.

## Candidate history

All timings below use the fixed 2048x4096 BF16-output L1 workload, BF16
DEST, SFPU tensor scaling, 130 workers, and profiler-reported 1,350 MHz.
Each reported time is the median of calls 3 through 8, normalized by 64
output tiles on the busiest core. Math timings use 128 face repetitions.

| Candidate | Accuracy | Pipeline cycles/output tile | Repeated math cycles/output tile |
| --- | --- | ---: | ---: |
| Fresh baseline | pass | 830.40 | 590.22 |
| Fold tensor scale into two scale vectors, with extreme-scale fallback | pass | 806.28 | 538.31 |
| Hoist magnitude mask | pass | 774.02 | 538.31 |
| Dynamic magnitude/sign shift rewrite | **fail; not profiled** | — | — |
| Unroll plane and face loops | pass | 730.19 | 501.57 |
| Reuse unused first LUT segment's register | pass | 709.71 | 481.56 |
| Fixed magnitude shifts, retaining existing sign extraction | pass | 698.16 | 469.56 |
| Expand compact scales in DEST once | pass | 691.57 | 469.38 |
| Share one SFPU setup/teardown across all faces | pass | **676.72** | **441.38** |
| Batch tiled writer transfers | pass | 673.00 | 441.38 |

The writer-batching change saved less than the predeclared 1% screening
floor relative to the preceding candidate and was discarded. The smaller
scale-expansion change was retained as part of the shared-context combination,
which improved on the preceding confirmed implementation by more than 1%.
The failed shift variant's cause was not established; it was not retained.

Repeated-math binaries can hoist/schedule constants differently, explaining
why hoisting the mask changed production time while leaving the repeated
math estimate almost unchanged. These numbers are diagnostic; production
pipeline time is the selection criterion.

## Confirmation and retained change

Independent baseline remeasurement gave **830.47 cycles/tile**. The selected
candidate remeasured at **676.37 cycles/tile**, saving 154.10 cycles, or
**18.56%**, comfortably exceeding the 8.30-cycle screening floor.

After applying and formatting the retained code on the branch, a separate
hardware run measured **677.29 cycles/tile**, or **32.1085 us** per operation:
261.26 billion values/s and 522.52 GB/s counting BF16 output bytes only.
This is L1/NoC conversion throughput; no DRAM speedup is claimed by this run.

The retained implementation:

- Unpacks the compact scale tile once and uses `SFPTRANSP` to expand scales
  into the required lane groups in DEST. Descending plane order preserves
  compact inputs until they have been read.
- Shares SFPU setup and teardown across that expansion and all face decodes.
- Unrolls plane and face loops to reduce scalar address/shift construction.
- Uses fixed logical shifts for magnitude extraction and the established
  sign-copy instruction. The unreachable first LUT segment supplies a
  temporary register, reducing coefficient-register moves.
- Folds BF16 tensor scale into the two FP32 scale vectors for conservative
  normal-scale bounds. Products are exact there; extreme tensor scales use
  the original operation order and accepted FTZ behavior.

No public API, host packing, program-factory, reader, or writer change was
needed. Output is still ordinary BF16 tile/row-major data, ready for consumers.

## Verification and evidence

The final branch passed **112 tests, with six expected exclusions**, using
watcher/asserts through safe pytest. An additional check covered all 65,536
packed UINT16 patterns with TILE and ROW_MAJOR inputs and observed **zero
ULP error** for tensor scale one in both layouts. Gate extrema remain:
max 1.4375 ULP, mean 0.81677 ULP, median 0.75 ULP, minimum SNR 46.23565 dB.

Harness validation also included four hardware-free parser/comparison tests,
an unchanged worktree, and a zero-output canary. The canary failed accuracy
and generated no performance stage, proving the candidate sources were used
and rejected before scoring. Measured JIT translation units include absolute
worktree source paths; passing profiles include math-thread disassembly.

Generated evidence under `generated/nvfp4_loop/` includes:

- `experiment.json`: budget, candidate history, selected source hashes, and
  final verification summary.
- `baseline-final`, `baseline-confirm`, `winner-confirm-1`: per-stage XML,
  source hashes, patches, numerical reports, profiler CSVs, and disassembly.
- `confirmed-comparison.json`: the paired screening result.
- Individual candidate directories, including failed accuracy diagnostics.
- `all-words.xml` and `all-words-source.py`: additional exhaustive-word check.

The final formatted-branch profiler report is timestamped
`2026_10_29_13_04_26`. Generated artifacts are not committed; the code,
harness, fixed workload, and run method are committed for reproduction.
