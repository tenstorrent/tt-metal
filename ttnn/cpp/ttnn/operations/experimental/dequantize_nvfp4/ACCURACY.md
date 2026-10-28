# NVFP4 reconstruction accuracy and verification contract

Measurements use the Blackhole kernel at commit `e351a027125`, BF16 DEST,
ordinary tiled output, and interleaved DRAM. The numerical survey does not
measure performance. No decoder implementation was changed for this survey.

The important result is that the relaxed decoder is not uniformly as accurate
as the BFP8 baseline. At tensor scale `1.02734375`, its error energy is 1.05482
times BFP8's on the declared corpus (about 0.232 dB lower SNR), despite
remaining within one BF16 encoding step of exact nearest-even reconstruction.
The accepted gate below permits flush-to-zero behavior for subnormal operands
and results; the IEEE underflow differences remain diagnostic.

## Accepted accuracy gate

The gate applies to `experimental_fast=True, fpu_tensor_scale=False` with
BF16 DEST. All decoding and scale multiplication in this path use the SFPU;
FPU HiFi/LoFi fidelity settings do not change that arithmetic. The separate
`fpu_tensor_scale=True` survey uses **HiFi4** for its tensor-scale multiply.
Both paths still use the normal unpack, transpose, and pack pipeline.

For **each tensor-scale case** in the fixed verification corpus, require:

| Metric | Requirement |
| --- | ---: |
| Maximum continuous BF16-ULP error | <= 2 |
| Mean continuous BF16-ULP error | <= 1 |
| Median continuous BF16-ULP error | <= 1 |
| Aggregate SNR | > 46 dB |

The ULP requirements are inclusive; the SNR requirement is strict. Cases
must pass individually, rather than pooling their statistics. An exact
nonzero reconstruction has infinite SNR and passes. An all-zero reference
has undefined SNR and is evaluated using the same ULP limits, without an
additional SNR or exact-zero requirement. Existing routing,
layout, and encoding-correctness tests remain in force. BFP8 error and
encoding-distance/P99 metrics remain diagnostic; BFP8 dominance is not an
additional acceptance requirement.

The selected subnormal policy is **flush to zero**. The reference flushes
subnormal FP32 tensor-scale operands and subnormal reconstructed results
below `2**-126`, preserving normal results for accuracy scoring. FP8 scale
subnormal encodings remain supported: they decode to normal BF16 values
and are not flushed merely because their FP8 exponent field is zero.
Boundary tests explicitly check this FTZ behavior and expected infinities.
Signed zero is compared numerically in these tests.

The ordinary device survey writes the gate limits, per-case pass/fail and
failures, and overall extrema to its JSON artifact. It writes the artifact
before asserting the aggregate gate result, so failed threshold checks retain
numerical diagnostics. Performance promotion uses hardware measurements.

## Reference and supported values

For each datum, reconstruct `x = decode_e2m1(d) * decode_ue4m3(s) * g32`.
Decode the stored datum and block scale exactly; first cast the API tensor
scale to FP32, then perform the reconstruction in FP64. Apply the accepted
operand/result FTZ policy for gating; retain unflushed reconstruction for
IEEE boundary diagnostics. None of the ordinary survey's nonzero values
are subnormal, so the policy does not change the tables below. This evaluates the
mathematical value of the already-quantized NVFP4 tensor, not the unknown
original tensor before NVFP4 quantization. That earlier quantization loss
cannot be measured without the original tensor.

FP64 is an exact oracle here: E2M1 times E4M3 needs at most six significant
binary digits, and multiplication by the original FP32 tensor scale needs
at most 30. Direct FP64-to-BF16 rounding is calculated explicitly, avoiding
double rounding through FP32 at halfway cases.

The scalar space at a fixed tensor scale has only **2,032 encoded pairs**:
16 E2M1 codes times 127 finite nonnegative scale codes. Standard NVFP4 block
scales are unsigned E4M3 carried in a byte: `0x00..0x7e` are finite; `0x7f`
is NaN and the high bit is padding. Signed E4M3 negatives and NaNs are outside
this operation's contract. Test the remaining byte patterns as invalid-input
cases rather than including them in finite-value error statistics. The fast
host preparation rejects scale codes above 126; the raw precise kernel
assumes valid scale bytes and does not inspect their contents on the host.

Sources: [NVIDIA PTX format definitions](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats)
and [Transformer Engine NVFP4 reconstruction](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.16/user-guide/features/low_precision_training/nvfp4/nvfp4.html).
The custom E2M1 block-scale mode is separate; this survey evaluates standard
NVFP4 UE4M3 scales.

## Verification corpus

The device survey uses all 127 valid scale codes and **256 complete blocks
of 16 values per scale**, or 520,192 reported values per tensor scale:

- 16 constant-code blocks, one for each signed E2M1 encoding.
- 16 blocks containing every code, rotated through positions.
- 32 blocks with a single maximum or a single target among maximum values.
- 72 target-code/block-maximum combinations, including both data signs.
- 120 independent random blocks, with seed `20261027`.

An extra padding scale makes the upload tile-aligned; its values are excluded
from all statistics. There are 12 ordinary tensor scales: `0`, `1`, `0.8125`,
`0.7314453125`, FP32 `0.1`, `1.00390625`, `1.01171875`, `1.0234375`,
`1.02734375`, `1.03125`, `2**-40`, and `2**40`. The last two check exponent
scaling while keeping every nonzero result normal. Every decoder variant
therefore sees 6,242,304 reported values in this survey.

Random groups alone are insufficient. They often contain magnitude 6 and
can miss smaller block maxima and exact rounding ties. For aligned groups,
a BFP output depends on the target value and the shared maximum exponent;
the 72 target/maximum combinations cover those numerical contexts at each
selected tensor scale. Position and permutation tests remain necessary for
routing correctness. This reduction does not apply unchanged when a BFP
block mixes different NVFP4 block scales.

If testing packed bytes, enumerate all 256 byte patterns rather than drawing
256 random patterns. All 65,536 UINT16 words can also be enumerated in a
separate packing/extraction test at selected scales. There is no need to
cross every packed word with every scale merely to cover scalar arithmetic.
Existing operation tests exercise all UINT16 words and both I/O layouts.
The larger `16**16` space of complete block arrangements need not be
exhausted for a pointwise decoder, but structured arrangements are important
for group quantization and routing.

Tensor-scale sampling needs special care because the fast kernel rounds
`g32` to BF16. The companion CPU model evaluates all 128 BF16 significands
in `[1,2)`, their 128 rounding midpoints, neighboring FP32 values, and
product-specific output-rounding/binade thresholds. It checks **753,396
critical-point cases**, including sign symmetry. Its maximum normal-domain
SFPU-model error is **1.4375 BF16 ULP**, versus **0.5 ULP** when the tensor
scale is exactly BF16. The worst witness is also included in the hardware
survey: block scale code 5, datum 1.5, tensor scale `1.02734375`.

This CPU model describes `RNA_BF16(d*s*RNE_BF16(g32))`, not internal FPU
fidelity phases. Exponent scaling carries its results to other binades only
while tensor scales, intermediates, and outputs avoid underflow/overflow.
The critical-point mean and median are stress-test statistics, not a
probability distribution over model weights or all FP32 tensor scales.

## Metrics and format baseline

All paths retain exact FP64 reconstruction as truth. BFP8 is the quality
comparator, not a replacement for truth. The storage baselines use TTNN's
actual host packer from FP32-staged reconstruction, with ordinary horizontal
16-element groups matching the NVFP4 groups. BF16, BFP8, and BFP4 storage
results therefore describe that production conversion route; they do not
claim to model every device-packer rounding configuration. The JSON reports
also distinguish direct exact BF16 RNE from FP32-staged BF16 storage.

For normal nonzero `x`, the common error unit is
`q(x) = 2**(floor(log2(abs(x))) - 7)`. Clamp spacing to `2**-133` for BF16
subnormals and use the same spacing at zero. Report `abs(y-x)/q(x)` as a
continuous BF16-ULP error. This differs from counting BF16 encodings crossed:
the latter compares the BF16 output with exact reconstruction rounded to
BF16 nearest-even. Both quantities are included in the JSON; neither is an
FP32-ULP metric or a block-dependent BFP8 ULP.

The survey records max/mean/median/P99 ULP, exact fraction, RMSE, relative
error quantiles, zero counts, encoding distance, and worst-case witnesses.
Aggregate SNR is `10*log10(sum(x*x) / sum((y-x)**2))`. Per-datum SNR is
`20*log10(abs(x)/abs(y-x))`; it is infinite for exact nonzero values and
undefined for a zero signal. The artifact records those counts separately
and reports finite per-datum SNR statistics. It also records minimum
nonexact group SNR, group error ratios, and groups exact in BFP8 but not in
the candidate. Aggregate SNR can hide errors in low-energy groups, so it
must not be the only acceptance metric.

The following tables weight the declared structured/random corpus uniformly.
SNR sums energy across scale codes, so larger scales contribute more signal
energy. These are reproducible stress-corpus measurements, not estimates of
accuracy on a particular model's weight distribution.

### Tensor scale `1.0`

| Path/storage | Max BF16 ULP | Mean BF16 ULP | Median BF16 ULP | SNR (dB) | Error energy / BFP8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact BF16 RNE | 0.00000 | 0.00000 | 0.00000 | inf | 0.00000 |
| Precise decoder | 0.00000 | 0.00000 | 0.00000 | inf | 0.00000 |
| Fast, SFPU tensor scale | 0.00000 | 0.00000 | 0.00000 | inf | 0.00000 |
| Fast, FPU tensor scale | 0.00000 | 0.00000 | 0.00000 | inf | 0.00000 |
| BFP8 host storage | 16.00000 | 0.46605 | 0.00000 | 51.64272 | 1.00000 |
| BFP4 host storage | 240.00000 | 24.79255 | 16.00000 | 21.15275 | 1119.42857 |

### Tensor scale `0.7314453125`

| Path/storage | Max BF16 ULP | Mean BF16 ULP | Median BF16 ULP | SNR (dB) | Error energy / BFP8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact BF16 RNE | 0.45312 | 0.24012 | 0.25000 | 55.64245 | 0.15104 |
| Precise decoder | 0.45312 | 0.24012 | 0.25000 | 55.64245 | 0.15104 |
| Fast, SFPU tensor scale | 0.76562 | 0.33126 | 0.26562 | 52.04374 | 0.34591 |
| Fast, FPU tensor scale | 0.76562 | 0.33117 | 0.26562 | 52.23635 | 0.33090 |
| BFP8 host storage | 15.54688 | 1.29748 | 0.45312 | 47.43337 | 1.00000 |
| BFP4 host storage | 234.06250 | 24.02990 | 12.43750 | 21.95960 | 352.67667 |

### Tensor scale `1.02734375`

| Path/storage | Max BF16 ULP | Mean BF16 ULP | Median BF16 ULP | SNR (dB) | Error energy / BFP8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact BF16 RNE | 0.50000 | 0.24021 | 0.25000 | 55.08795 | 0.13739 |
| Precise decoder | 0.50000 | 0.24021 | 0.25000 | 55.08795 | 0.13739 |
| Fast, SFPU tensor scale | 1.43750 | 0.81677 | 0.75000 | 46.23565 | 1.05482 |
| Fast, FPU tensor scale | 1.43750 | 0.81677 | 0.75000 | 46.23565 | 1.05482 |
| BFP8 host storage | 11.18750 | 1.42032 | 0.56250 | 46.46743 | 1.00000 |
| BFP4 host storage | 246.56250 | 25.04468 | 10.31250 | 21.38468 | 322.31072 |

BF16 is exact for tensor scale 1 because `d*s` fits its significand. BFP8
can still lose precision: in a group with scale 1.625, data 0.5 and 6 decode
to 0.8125 and 9.75. The BFP8 quantum is 0.125, making the smaller value a
rounding tie. BFP4 has only three magnitude bits; small nonzero values can
be rounded to zero when another value determines a large group exponent.
For example, a group with scale 1.875 and data 0.5 and 6 contains 0.9375
and 11.25; BFP4's quantum is 2, so 0.9375 becomes zero. That is a 100%
relative error even though its absolute magnitude is small.

At tensor scale `1.02734375`, both fast variants have max 1.4375 continuous
BF16 ULP error and max encoding distance one from exact BF16 RNE. Their
aggregate error energy exceeds BFP8's by about 5.5%. Thus a hard requirement
of BFP8-or-better aggregate error at every tested tensor scale rejects the
current relaxation. At tensor scale `0.7314453125`, by contrast, the SFPU
variant has about 0.346 times BFP8's total error energy, while roughly 17.8%
of individual values have larger error than BFP8. These are different
acceptance questions; neither follows from the other.

## Extreme scales and accepted FTZ behavior

A separate device survey crosses all scalar encodings with tensor scales
`2**-149`, `2**-134`, `2**-133`, `2**-127`, `2**-126`, `2**120`, and `2**127`.
Each case has 4,064 reported values, including two placements per scalar
encoding. It compares result classifications with directly rounded BF16
reconstruction and records finite errors separately as IEEE diagnostics.
It also compares every decoder output with an FTZ reference and asserts
that the all-SFPU and precise paths have no mismatches for these boundary
cases. The HiFi4 FPU path remains diagnostic: at `g = 2**-126`, it differs
from that oracle in 180 normal-valued results, consistent with extra loss
in its internal multiplication phases near underflow. It is outside the
all-SFPU accuracy gate.

All three kernels, including the precise path, exhibit SFPU flush-to-zero.
At tensor scale `2**-127`, **3,456 of 4,064** results whose exact BF16
reconstruction is finite and nonzero are instead zero; at `2**-126`, the
count is **1,288**. Their maximum continuous BF16 ULP errors are 240 and
120 respectively. Thus the ordinary-rounding budgets do not describe the
full currently admitted global-scale range under IEEE semantics. These
flushes are accepted by the chosen FTZ policy. Choosing exact BF16 tensor
scales alone does not preserve subnormals. The tested large-scale cases produced
the expected infinities with no classification mismatches or NaNs; this is
not an exhaustive overflow-boundary test.

Under the chosen FTZ contract, a subnormal tensor-scale operand is treated
as zero before multiplication, even when multiplication by a large block
value could otherwise produce a normal result. Subnormal product results
also flush to zero. The boundary test retains IEEE differences in the JSON
and checks the FTZ reference separately; these are accepted flushes rather
than errors charged against the ordinary rounding gate.

## Agent evaluator contract

- Keep the FTZ-aware FP64 reference and independent storage-format baselines
  outside editable kernel files. Fix packing, scales, output ordering, and
  numerical zero-sign policy.
- Run exhaustive scalar, structured block, packing, layout, and tensor-scale
  boundary tests, then add held-out seeds and representative real tensors
  when available. The accepted gate applies per fixed tensor-scale case,
  with the corpus/seed recorded. Include the custom E2M1-scale mode separately
  if optimizing it; this survey covers standard UE4M3 block scales.
- Enforce max ULP <= 2, mean ULP <= 1, median ULP <= 1, and SNR > 46 dB
  for the all-SFPU decoder. Keep the FPU-scale path's results separate and
  labelled HiFi4. These thresholds replace a strict BFP8-or-better gate.
- Continue reporting BFP8 energy comparisons, P99 and relative-error tails,
  encoding distance, and per-group/element regressions as diagnostics.
  Do not pool tensor-scale cases in a way that hides a failed case.
- Check FTZ behavior and nonfinite classifications separately. Preserve
  invalid-scale and layout checks. Use hardware measurements to promote
  performance improvements after accuracy passes.

The existing all-SFPU path passes this gate on the declared corpus. Its
worst measured ordinary case has max 1.4375 ULP, mean 0.81677 ULP, median
0.75 ULP, and minimum SNR 46.23565 dB. These results characterize the fixed
corpus and do not guarantee the same aggregate statistics on every possible
input distribution. In particular, the SNR margin is about 0.236 dB.

## Simulator and performance evaluation

The current local simulator source (`/workspace/ttsim`, revision
`f7e37f78b71948dfc54057547c021ef8aea3b516`, October 21, 2026) was built for
Blackhole and passed three existing NVFP4 tests: SFPU tensor scales 1 and
0.7314453125, and FPU tensor scale 0.7314453125, all with BF16 DEST. This
establishes those numerical cases, not full survey equivalence. The older
prebuilt library at `/workspace/libttsim_bh.so` failed before kernel execution
with a PCI-configuration `UndefinedBehavior` error; the source build fixed it.

The source is still a functional model. `src/libttsim.cpp` can execute
multiple Tensix instructions within one global-clock tick, `src/tile.cpp`
returns that clock for timer reads, and `docs/unsupported_functionality.md`
lists a performance-modeling simulator as future work. Neither simulator
wall time nor these timer values should rank silicon performance. Use it
for parallel correctness/ISA-contract checks and use hardware profiling for
promotion decisions.

Static SFPU scheduling is useful for screening. Analyze emitted instructions
and LReg dependencies using the Blackhole ISA latency/issue tables. Without
LoadMacro, the SFPU accepts at most one external instruction per cycle;
arithmetic dependencies and multi-cycle instructions add stalls. Include
implicit register side effects, then calibrate estimates against isolated
hardware measurements. Scalar issue overhead, loops/MOP/replay, synchronization,
unpack/pack, and NoC traffic prevent a plain SFPU instruction count from
predicting complete L1-to-L1 time. No static estimator was implemented here.

The two investigation agents used read-only source access; simulator staging
used symlinks, with no copied venv, instruction files, or binaries. Future
parallel kernel-editing workers should use separate worktrees and isolated
JIT caches, reuse the built runtime/venv through explicit paths, and serialize
all hardware runs through `scripts/run_safe_pytest.sh`. Do not source a
hardcoded main-checkout environment without overriding worker source paths.

## Reproduction

From the main checkout, with its existing build and environment:

```bash
source source-env.sh
NVFP4_ACCURACY_REPORT=generated/nvfp4_accuracy/device_accuracy.json \
    scripts/run_safe_pytest.sh --dev \
    tests/ttnn/unit_tests/operations/test_nvfp4_accuracy.py -q
python tests/ttnn/unit_tests/operations/nvfp4_accuracy_model.py \
    --output generated/nvfp4_accuracy/global_scale_model.json
```

The device test emits the ordinary survey and a sibling
`device_accuracy_boundaries.json`; the CPU script emits the threshold survey.
These generated artifacts are not checked in. They contain all 12 ordinary
scale cases, both fast variants and the precise path, all storage baselines,
worst witnesses, and extreme-scale classifications.

To reproduce a simulator correctness smoke test, build Blackhole in the
existing simulator checkout, stage symlinks to its library and the matching
SoC descriptor in one private directory, then run:

```bash
source source-env.sh
TT_METAL_SIMULATOR=/tmp/nvfp4-ttsim-bh-agent/libttsim.so \
    timeout 120s scripts/run_safe_pytest.sh \
    'tests/ttnn/unit_tests/operations/test_dequantize_nvfp4.py::test_nvfp4_fast[fpu_tensor_scale=False-global_scale=1.0-e2m1=False-output_layout=Layout.TILE-input_layout=Layout.TILE]' -q
```

The safe runner sets slow dispatch and disables unsupported LoadMacro. In
simulator mode it deliberately provides no hang timer, so the outer timeout
is required. Hardware performance runs should use `--profile` without watcher
instrumentation, consume explicit test-result artifacts, and validate all
expected operation records: Tracy can mask a failing pytest exit status.
