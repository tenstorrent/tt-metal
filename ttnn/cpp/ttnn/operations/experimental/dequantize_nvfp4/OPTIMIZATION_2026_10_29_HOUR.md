# TT intrinsics, replay, and transposed linear weights

This extends the October 29 worktree loop from revision `6a73ff1ceef`, with a
one-hour maximum budget. Each candidate uses an isolated sparse worktree and
kernel cache. All hardware execution goes through `scripts/run_safe_pytest.sh`.
The trusted reconstruction corpus and numerical limits are unchanged.

## Decoder experiments

Blackhole hardware, profiler clock 1,350 MHz, BF16 DEST, BF16 output,
`experimental_fast=True, fpu_tensor_scale=False`, tensor scale 0.8125.
The fixed workload is 2048x4096 with L1 input, scales, and output, distributed
over the available 130 cores. The busiest core produces 64 output tiles.
Numbers are medians of six calls after two warmups, in cycles per 1024-element
output tile. Pipeline timing includes reader, unpack, scale preparation, decode,
DEST transpose, pack, and writer. The historical table below uses a
repeated-face math diagnostic. It is not a standalone LLK MATH_ISOLATE
measurement; the final stateful decoder uses complete-sweep repetition instead.

| Candidate | Complete L1 pipeline | Repeated-face math |
| --- | ---: | ---: |
| Existing SFPI baseline | 677.13 | 441.38 |
| SFPI programmable tensor-scale constant | 677.97 | 441.38 |
| SFPI persistent ordinary LUT registers | 667.53 | 433.38 |
| Explicit TT intrinsics and LReg assignments | 672.29 | 437.39 |
| TT intrinsics, persistent constants, five-instruction replay | 630.73 | 426.35 |
| TT intrinsics with next-vector dependency hiding | 658.06 | 370.18 |
| Dependency hiding plus five-instruction replay (TTI04) | 574.91 | 369.61 |
| Full vector-pair replay with DEST-address increments | 621.76 | 416.34 |
| TTI04 with programmable Creg12 tensor scale | 575.15 | 369.61 |
| TTI04 plus next-face prefetch hiding one dependency | 571.02 | Not applicable |
| Next-face prefetch hiding both dependencies | **566.82** | Not applicable |

Ten distinct candidates were evaluated, excluding baseline and a duplicate
measurement. The programmable-register trials passed but did not improve the
winning path. The one-dependency next-face trial saved less than the 1%
screening threshold and was superseded by the two-dependency version.
The selected implementation uses ordinary LRegs and an immediate BF16 tensor
multiplier. No LoadMacro is used. Scale expansion remains a separate SFPI
helper; the decoder is a separate helper containing only TT intrinsics.

The raw register assignments are:

| Register | Contents |
| --- | --- |
| L0 | Three-bit magnitude, then synthetic magnitude / unused first LUT segment |
| L1, L2 | Persistent LUT coefficients |
| L3 | Signed LUT input, then the next packed UINT16 vector |
| L4, L5 | Even/odd group-scale vectors |
| L6 | LUT result, product, and rounded output |
| L7 | Magnitude mask, integer 7 |

`SFPDIVP2` adds the exponent bias and `SFPSETSGN` inserts the sign. `SFPLUT`
writes L6, allowing the next input to load into L3 while the LUT result is
pending. After multiplying L6 by its group scale, extracting/masking the next
nibble fills the multiply dependency. On each face's final vector, the next
even scale can overwrite L4 before the old odd scale is consumed by MUL; the
next odd scale overwrites L5 after MUL has read it. These two loads fill both
remaining result dependencies. The new pair is scaled after the final output
store. This requires visiting every face in order, starting at plane 0/face 0. `SFPSTOCHRND` performs native nearest
BF16 rounding before the store. The even/odd scale choice is resolved during
loop unrolling to L4 or L5; there is no per-lane conditional selection.

Math-thread replay slots 0 through 4 hold magnitude shift, exponent adjustment,
sign shift, sign insertion, and LUT. They are recorded while executing the
first vector of plane 0/face 0 after each packed tile's SFPI scale expansion.
The other vectors reuse this sequence. Standard DEST transpose uses separate
math replay slots 16 through 31. Constants are established after scale
expansion because that helper uses all eight ordinary LRegs.

For ordinary tensor scales, scale products are exact in FP32 and remain normal.
The conservative folding interval is [2^-100, 2^100]. Extreme tensor scales
retain the original multiplication order, preserving flush-to-zero behavior.

Before next-face prefetch, fresh integrated baseline/TTI04 measurements were
676.81 versus 574.44 pipeline cycles/tile, 15.13% less time.

Repeating one face is invalid when L4/L5 carry scales between faces. The harness
now repeats all four planes and their four faces 128 times after one scale
expansion. It reinitializes the first face's scales/replay on each sweep. The
trusted tests and numerical gate are unchanged; a hardware-free regression
checks instrumentation boundaries and preservation of source/header isolation.
Fresh measurements with this evaluator are:

| Decoder | L1 pipeline cycles/tile | Complete-sweep math cycles/tile |
| --- | ---: | ---: |
| SFPI baseline | 677.69 | 470.92 |
| TTI04 | 574.94 | 369.61 |
| Selected next-face prefetch | **567.52** | **362.11** |

The selected pipeline saves **16.26%** against SFPI and **1.29%** against TTI04.
An independent paired repeat measured 676.27 versus 568.36 cycles/tile,
**15.96% less time**, with 470.92 versus 362.11 complete-sweep math cycles/tile.
Both pairs exceed the evaluator's 1%/three-MAD screening threshold.
The changed diagnostic can change compiler scheduling/hoisting, as seen for
SFPI; its values must not be directly compared with the old repeated-face
method. Only production pipeline timing ranks candidates. Artifacts, exact
patches, hashes, XML outcomes, and measured ELF disassemblies are under
`generated/nvfp4_hour/`. Final paired evidence is `sweep-baseline-{1,2}`,
`sweep-winner-{1,2}`, and `confirmed-comparisons.json`. The selected source
hashes match the branch kernels. The TTNN target build passed. Test results were 112 existing passes with six
expected skips, 22 transpose/consumer passes, and five harness passes.
Pre-commit checks passed.

## Accuracy contract

Each evaluated candidate passed 112 tests with six expected skips. The gate is
per tensor-scale corpus: maximum BF16 ULP <= 2, mean <= 1, median <= 1, and
SNR > 46 dB. Zero-reference cases use ULP only. FP32 subnormals flush to zero.
Observed statistics were identical to the baseline: maximum 1.4375 ULP,
maximum case mean 0.8167679554 ULP, maximum case median 0.75 ULP, minimum
SNR 46.2356515 dB. Boundary tests also cover extreme tensor scales and FTZ.

## Native transpose in the linear consumer

Logical RHS `transpose_b=True` can be implemented natively by matmul Reuse,
ReuseMultiCast1D, and ReuseMultiCast2D program configurations. The RHS maps to
hardware SrcA in these kernels (`SrcOrder::Reverse`). Some other configurations
materialize a separate transpose, so inspect the selected backend/profile.

Omitting `transpose_dest` alone produces a grid of individually transposed
tiles, not the full matrix transpose. The new opt-in `transpose_output=True`
keeps those tiles and exchanges their grid coordinates in the writer, within
each batch. Normal packing remains unchanged. For original W[K,N], output is
W.T[N,K], consumable as `matmul(X, decoded_Wt, transpose_b=True, ...)`.
The option requires fast mode and TILE output; ordinary output remains default.

All 22 production-path tests passed with watcher enabled. They cover rank
2/3/4, nonsquare dimensions, both input layouts, both tensor-scale engines,
program-cache reuse, trace replay, all 65,536 packed UINT16 words, and native
matmul consumers in L1/DRAM with BF16 and FP32 matmul DEST. The converter
always uses BF16 DEST in these tests. Transposed output is bit-exact to the last-two-axis
transpose of ordinary decoded output, so the same reconstruction gate applies.

For X[128,2048] @ W[2048,4096], measured with TTI04 before next-face prefetch:

| Memory | Output orientation | Decode (us) | Matmul (us) | Paired kernel sum (us) |
| --- | --- | ---: | ---: | ---: |
| L1 | Ordinary W | 27.869 | 53.514 | 81.410 |
| L1 | W.T plus native transpose_b | 26.473 | 53.566 | 79.994 |
| DRAM | Ordinary W | 68.296 | 62.350 | 130.907 |
| DRAM | W.T plus native transpose_b | 62.119 | 68.173 | 130.604 |

The paired sum is the median of each iteration's two kernel durations, not the
sum of the separate medians. These exclude host gaps and dispatch handoff.
L1 improves about 1.7%; the DRAM difference is too small to establish a gain.
Faster conversion can be offset by changed RHS memory-read strides. Profiles
contain no separate transpose operation. Keep this option workload-dependent.

For the selected next-face-prefetch kernel, captured decode-plus-matmul FW
spans include device handoff and exclude eager Python gaps. Median trace
replays 3 through 8, using a counterbalanced case order:

| Memory | Ordinary W | W.T plus native transpose_b | Time reduction |
| --- | ---: | ---: | ---: |
| L1 | 81.789 us | 80.745 us | 1.28% |
| DRAM | 132.451 us | 131.420 us | 0.78% |

Earlier independent TTI04 traces gave 1.94–2.10% L1 gains and 0.48–1.09% DRAM
gains. The small DRAM difference remains inconclusive. All projections were
bit-exact to the directly uploaded reconstructed-weight matmul oracle. This
checks layout/consumer equivalence; reconstruction remains the numerical gate.
No transpose operation materialized. Reproducers, source hashes, and parsed
trace comparisons are in `generated/nvfp4_hour/consumer_transpose/`.

## Bandwidth accounting

For C cycles per output tile, an ideal balanced 120-core steady state gives
`120 * 1.35e9 * 1024 / C` elements/s. This is a projection from measured per-core
pipeline throughput; the reported hardware workload uses 130 cores.

This prepared fast format reads 0.5 packed-data bytes plus 0.125 BF16-scale
bytes per output element, and writes 2 BF16 bytes. Distinguish output-equivalent
bandwidth (2 bytes/element), encoded-input bandwidth (0.625 bytes/element), and
total standalone traffic (2.625 bytes/element). At 567.52 cycles/tile these
120-core projections are 292.3 GElem/s, 584.6 GB/s of BF16 output, 182.7 GB/s
of encoded input plus prepared scales, or 767.3 GB/s of aggregate L1 traffic.
The actual finite 130-core workload measured 26.90 us and 311.8 GElem/s.

Exceeding 512 GB/s in BF16-output equivalents does not establish that a fused
consumer can sustain 512 GB/s of encoded DRAM input. That would require
819.2 GElem/s at 0.625 input bytes/element. A standalone read-and-write converter
has a different ideal DRAM ceiling: 512/2.625 = 195.05 GElem/s. Overlap with a
matmul sharing these cores must be measured in a fused implementation.
