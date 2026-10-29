# NVFP4 dequantization on Blackhole

`ttnn.experimental.dequantize_nvfp4` expands software-packed E2M1 data using
Blackhole's SFPU:

```
output = decode_e2m1(nibble) * decode(block_scale) * global_scale
```

Standard NVFP4 uses an E4M3FN scale for each 16 values and an FP32 global
**dequantization multiplier**. The FP4 magnitude codes 0 through 7 represent
`0, 0.5, 1, 1.5, 2, 3, 4, 6`; bit 3 supplies the sign. E4M3FN scale codes
0 through 126 are supported, including zero and subnormals. Negative and NaN
scale codes are outside the operation's contract. The kernel does not inspect
scale values on the host. The global scale must be finite and nonnegative.

An optional `Nvfp4ScaleFormat.E2M1` accepts nonnegative E2M1 scale codes 0
through 7, stored in one byte each. This is a custom scale variant, rather than
standard NVFP4. For a 16x16 block scale, repeat its code for each of its 16 rows
before packing. NVIDIA-swizzled scale buffers must be rearranged first.

[NVIDIA's format description](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
details the two scale levels.

## Reconstruction accuracy

See [the accuracy survey and accepted verification gate](ACCURACY.md)
for exhaustive scalar coverage, BF16/BFP8/BFP4 comparisons, and simulator
findings. The all-SFPU fast path must satisfy max BF16 ULP <= 2, mean and
median ULP <= 1, and SNR > 46 dB in each fixed tensor-scale test case.
Subnormal operands/results follow a flush-to-zero contract. BFP8 comparisons
remain diagnostic; the relaxed path can exceed BFP8's aggregate error on
targeted tensor scales. The separate FPU tensor-scale path uses HiFi4.

The [worktree optimization harness](HARNESS.md) evaluates kernel-only candidates
through the safe pytest runner, with Tracy L1-to-L1 timing and optional repeated
math diagnostics. It records accuracy, source provenance, profiles, and disassembly.

## Preparing and converting data

Prepare codes once when loading weights. `pack` rearranges codes; it does not
quantize floating-point values or choose scales.

```python
import torch
import ttnn

# Ordinary row-major codes, one code per FP4 value; one scale per 16 columns.
codes = torch.randint(0, 16, (64, 512), dtype=torch.int32)
scale_codes = torch.full((64, 32), 0x38, dtype=torch.uint8)  # E4M3FN 1.0
container = ttnn.uint16
packed, scales = ttnn.experimental.nvfp4.pack(codes, scale_codes, dtype=container)

packed_tt = ttnn.from_torch(
    packed, dtype=container, layout=ttnn.TILE_LAYOUT, device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
scales_tt = ttnn.from_torch(
    scales, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
output = ttnn.experimental.dequantize_nvfp4(
    packed_tt, scales_tt, global_scale=0.75,
    dtype=ttnn.bfloat16, output_layout=ttnn.TILE_LAYOUT,
    fp32_dest_acc_en=False,
)
```

For `uint8`, use `fp32_dest_acc_en=True`. The integer-8 datacopy path uses
32-bit DEST. `uint16` supports both BF16 and FP32 DEST and packs four values
per word. `dtype=ttnn.float32` also requires FP32 DEST. Either input or output
can use `ROW_MAJOR_LAYOUT`; choose tiled I/O for the faster path. Inputs and
output can reside in interleaved DRAM or L1. Sharding is not implemented.

The packed tensor accepts rank 2 through 4, with no padding and standard 32x32
tiles. Its last two dimensions must be positive multiples of 32. The output
has the same dimensions except for a last dimension multiplied by 2 (`uint8`)
or 4 (`uint16`). The host `pack` helper accepts 2D tensors; batch dimensions
can be flattened into height before preparation and restored before upload.
The unpacked width must be divisible by 64 or 128, respectively. Pad codes and
block scales before preparation if needed, then slice the decoded result.

## Physical packing contract

Nibbles are ordered least significant first. Each packed 32x32 tile produces
2 or 4 **adjacent output tiles**. Nibble `k` at packed tile position `(r, c)`
provides output tile `k` at `(r, c)`. Thus nibble packing is across tile planes,
rather than adjacent output columns. A row-major input stores the same logical
packed words as a tiled input; the reader stages them into faces.

Scales are a row-major UINT8 tensor shaped
`[packed_tile_count, 64 * values_per_word]`. Within each row:

1. Order by nibble plane.
2. For each plane, order by face: top left, top right, bottom left, bottom right.
3. For each face, store its 16 row-scale bytes.

Each scale byte applies to one horizontal run of 16 decoded values. This uses
exactly one E4M3 byte per 16 values, without expanding scales in device memory.
The `pack` helper implements both permutations.

## Kernel and fusion

The reader stages a packed tile and its compact scales. The unpacker copies
packed integers into DEST without floating-point interpretation. The SFPU
extracts nibbles, decodes values and scales, and applies both multipliers. It
reads compact scale words from L1 and selects each lane's scale with vector
shifts. Each plane writes to a separate DEST tile. Full DEST
synchronization accommodates the input plus four output tiles in FP32 mode.

Both DEST modes explicitly round BF16 output to nearest even in the SFPU,
after decoding and applying both multipliers in FP32 registers. This avoids
Blackhole's truncating SFPU store and the packer's ties-away rounding. FLOAT32
output retains the FP32 values in DEST until packing. As with
other SFPU arithmetic, FP32 subnormals may flush to zero. Extremely large
finite global scales may produce infinities in the output.

The standalone operation materializes the decoded output. Its decoder is a
building block for fusion, rather than a fused quantized matmul. A matmul
consumer would pack decoded tiles into a local circular buffer and unpack
them into SrcA/SrcB; FPU matmul cannot consume arbitrary decoded DEST tiles
directly. That integration needs its own scheduling and throughput tests.

Tiled I/O reads one packed page per 2/4 output tiles. Row-major I/O additionally
performs row transactions and face rearrangement on the data-movement RISCs.
The global scale is part of the cached program attributes; changing it selects
a different program. Buffer addresses are rebound by the descriptor framework
on cache hits. The device path performs no host reads, allocations of input
content, or host scale expansion, allowing device trace capture after warmup.

## Experimental column-scale path

Set `experimental_fast=True` in **both** `nvfp4.pack` and
`dequantize_nvfp4`. This path requires UINT16 input, BF16 output, and BF16
DEST. It supports the same row-major/tiled I/O and interleaved L1/DRAM options.
The default path and its packing contract remain unchanged.

```python
packed, scales = ttnn.experimental.nvfp4.pack(
    codes, scale_codes, dtype=ttnn.uint16, experimental_fast=True,
)
packed_tt = ttnn.from_torch(
    packed, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device,
)
scales_tt = ttnn.from_torch(
    scales, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
)
output = ttnn.experimental.dequantize_nvfp4(
    packed_tt, scales_tt, global_scale=0.75, experimental_fast=True,
    fpu_tensor_scale=True,  # Requested combination: apply tensor scale on the FPU.
)
# fpu_tensor_scale=False keeps tensor scaling in the SFPU; it benchmarks faster.
```

Offline preparation transposes each output 32x32 tile before packing its four
nibble planes. The 16 members of each quantization group then lie vertically
in a face. The helper decodes E4M3/E2M1 block scales exactly into BF16 and
exchanges the top-right/bottom-left scale faces to match that transpose.
Scales retain shape `[packed_tile_count, 256]`, with sixteen BF16 scale rows
per packed tile. Upload this prepared scale tensor as BF16 ROW_MAJOR. E2M1
scale preparation additionally takes `scale_format=Nvfp4ScaleFormat.E2M1`.

The reader fetches these compact rows into a separate CB. Each 32-byte CB
page is one 16-element scale row. Sixteen contiguous pages form a 16x16
BF16 tile, copied to DEST once per packed tile. An SFPU transpose expands
its rows into the even/odd-column vectors needed by each plane and face.
Two loads then supply all scales for a face, without further unpacker
broadcasts. Expansion runs from the last plane backwards to preserve the
compact source values. One SFPU context covers expansion and all face decodes.

A branch-free LUT decoder maps the magnitude bits to floating-point
`1,1.5,2,3,4,6,8,12`, then applies `x-1` below 2 and `x/2` otherwise. This
produces the exact E2M1 values, rather than an approximation. The sign is
retained. SFPSTOCHRND performs native nearest BF16 rounding, with ties away
from zero. An in-DEST full-tile transpose restores ordinary output tile order
before packing; output needs no subsequent permutation for matmul.

Both variants round the tensor scale to BF16 once. With
`fpu_tensor_scale=True`, a reusable BF16 scale tile is generated on the
reader RISC and HiFi4 FPU DEST reuse multiplies each decoded tile. With
`fpu_tensor_scale=False`, the SFPU applies that BF16 scalar before its native
rounding, avoiding the extra FPU multiply and operand reconfiguration. For
tensor scales in `[2**-100, 2**100]`, it folds the scalar into the two scale
vectors once per face. Those FP32 products are exact and normal; extreme
scales retain the original multiplication order to preserve FTZ behavior.
The FPU variant rounds during its BF16 accumulation phases: tests bound its
error by relative tolerance `1/128` against the exact product of the BF16
operands. The SFPU variant is checked exactly against native nearest,
ties-away rounding of that product. Neither variant promises the default
path's FP32 tensor-scale precision or ties-to-even result. HiFi2 was tested
and rejected because some BF16 scales exceeded the FPU variant's error bound.

Block scales occupy 128 bytes per decoded output tile instead of 64 bytes;
they are not expanded into full scale tiles in tensor memory. Packed data
plus block scales therefore uses 640 bytes per 1024 values (5 bits/value),
versus 576 bytes (4.5 bits/value) for the default path. Preparation is offline;
the device operation and runtime-generated tensor-scale tile support tracing.

## Validation

Run the focused tests through the repository's serialized runner:

```bash
scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/test_dequantize_nvfp4.py
```

Coverage includes all FP4 codes and valid E4M3 scale codes, zero/subnormal
scales, both storage widths, DEST modes, input/output layouts, DRAM and L1,
custom E2M1 scales, global scaling, validation failures, and buffer rebinding
on program-cache hits. Device performance must be measured after kernel JIT
warmup; host dispatch time is not an SFPU throughput measurement.

## Measured standalone throughput

On a Blackhole device with 130 worker cores, a 2048x4096 BF16 output, tiled
interleaved DRAM input/output, E4M3 scales, and global scale 0.8125:

| Packed storage | DEST | Device kernel time | Decoded values/s |
| --- | --- | ---: | ---: |
| UINT8 | FP32 | 87.3 us | 96.1 billion |
| UINT16 | BF16 | 89.0 us | 94.2 billion |
| UINT16 | FP32 | 89.1 us | 94.2 billion |

These are the medians of calls 3 through 8 from the device profiler's
`DEVICE KERNEL DURATION [ns]`, with 8,388,608 output values per call. They
include reader, compute, and writer execution; host preparation and kernel
JIT compilation are excluded. The final code includes explicit BF16
ties-to-even rounding in every case. Output throughput is approximately
188–192 GB/s; this counts decoded output bytes, rather than total DRAM traffic.

At the nominal 1.35 GHz Blackhole clock, the math-thread duration divided by
the largest per-core assignment (64 output tiles) is approximately 1,830–1,850
cycles per output tile. This includes synchronization and memory waits, and
is not an isolated SFPU instruction benchmark. The measurement supports the
standalone conversion's feasibility; it does not establish fused matmul
throughput. Reproduce the workload by using the preparation example above
with shape `[2048, 4096]`, uploading scales shaped `[2048, 256]`, and profiling
eight calls with each container/DEST combination after preparation.

### L1 pipeline and decoder cycles

A separate run with both inputs and output in interleaved L1, using the same
shape, scale format, global scale, BF16 output, and 130 cores, measured:

| Packed storage | DEST | L1-to-L1 pipeline cycles/output tile | Decoder cycles/output tile |
| --- | --- | ---: | ---: |
| UINT8 | FP32 | 1,807 | 1,715 |
| UINT16 | BF16 | 1,811 | 1,715 |
| UINT16 | FP32 | 1,811 | 1,715 |

Each output tile contains 32x32 = 1,024 decoded values. The L1 pipeline column
uses device kernel duration at the profiler-reported 1,350 MHz, normalized by
the largest per-core assignment of 64 output tiles. It includes interleaved-L1
NoC transfers, unpack, decode, pack, and synchronization; no DRAM accesses are
involved in the tensor data path.

For decoder isolation, a temporary benchmark repeated `nvfp4_decode_plane`
128 times per plane after a single unpack, overwriting the same DEST output
before packing once. The math-thread duration was divided by 128 times the
per-core output tile count. Thus this estimates DEST-to-DEST decoder cost,
including L1 scale loads, decoding, both multiplications, BF16 ties-to-even
rounding, and per-call setup; ordinary input/output waits are amortized. It
is not a count of SFPU arithmetic instructions alone. Both columns use the
median of calls 3 through 8. The production kernel was restored after the
measurement. Raw profiler reports are timestamped `2026_10_26_23_59_58`
(L1 pipeline) and `2026_10_27_00_00_49` (decoder isolation).

### Experimental-path measurements

Same 2048x4096 BF16 output and 130 cores; all three paths below use **BF16
DEST**. Timings are medians of calls 3 through 8. L1 cycles include all scale
broadcasts, format changes, the in-DEST transpose back to ordinary tiles,
packing, and NoC transfers. No final permutation is deferred to a consumer.

| Path | L1-to-L1 cycles/output tile | L1 device time | DRAM device time |
| --- | ---: | ---: | ---: |
| Default, FP32 tensor scale and ties-to-even | 1,811 | 85.9 us | 89.0 us |
| Column scales + LUT + native rounding + BF16 FPU tensor scale | 1,112 | 52.7 us | 89.9 us |
| Column scales + LUT + native rounding + BF16 SFPU tensor scale | **973** | **46.1 us** | **82.6 us** |

The complete FPU combination improves L1 conversion by about 1.63x. Keeping
the BF16 tensor scale in the SFPU is faster here: one SFPU multiply costs less
than the FPU DEST-reuse multiply plus operand reconfiguration. That variant
improves L1 conversion by about **1.86x**, delivering 181.9 billion decoded
values/s, or 363.7 GB/s of BF16 output. With DRAM I/O it delivers 101.6 billion
values/s, or 203.1 GB/s of BF16 output. The full FPU combination does not
improve DRAM time for this workload despite its lower compute cost.

Repeating only the face decoder 128 times after each scale-row broadcast
measured about **656 cycles per tile** for the SFPU portion of the FPU-scale
variant, and **718 cycles per tile** when tensor-scale multiplication is also
inside the SFPU. These decoder-only measurements amortize unpack, scale-row
broadcast, transpose, FPU work, and packing. Unlike the old decoder, block
scales are already in DEST for these measurements; use the L1-to-L1 column
above for a complete comparison. The production kernel was restored after
this temporary measurement. Profiler reports are timestamped
`2026_10_27_00_45_49` (complete pipeline) and `2026_10_27_00_46_51` (decoder).

For the fastest tested ordinary BF16 output, use the column packing with
`experimental_fast=True, fpu_tensor_scale=False`. The gains come from
hardware broadcast of compact scale rows, exact branch-free E2M1 lookup,
and native rounding. These measurements establish the best of the tested
variants, rather than an optimality proof. A fused consumer could further
reduce synchronization and intermediate traffic; that integration remains
separate from this standalone converter.


### Exponent and sign instruction optimization

The experimental decoder now uses `sfpi::addexp` (`SFPDIVP2`, exponent-add
mode) to rebias the synthetic LUT input, and `sfpi::copysgn` (`SFPSETSGN`)
to insert the E2M1 sign. These replace integer exponent/sign assembly and
remove redundant nibble/sign masks. The exponent adjustment operates on
raw bits: it must not be replaced by floating-point multiplication, which
would flush these synthetic subnormal bit patterns. Disassembly confirms
the two field instructions followed by the sign-retaining `SFPLUT`.

A direct decoder using `SFPSETMAN` and `SFPSETEXP` was also tested. Its
normal-value construction is exact, but handling E2M1 zero and subnormal
values needs predication and conversion. It passed the same 74 fast-path
tests, but was slower and was discarded.

Fresh measurements on the same Blackhole, shape, core count, and BF16 DEST
configuration compare the original LUT with the field-instruction variants:

| Decoder | Tensor scale | L1-to-L1 cycles/output tile | L1 device time | DRAM device time |
| --- | --- | ---: | ---: | ---: |
| Original LUT | FPU | 1,111 | 52.7 us | 88.9 us |
| Exponent/sign + LUT | FPU | **982** | **46.6 us** | **83.2 us** |
| Direct mantissa/exponent/sign | FPU | 1,295 | 61.4 us | 97.3 us |
| Original LUT | SFPU | 974 | 46.2 us | 82.0 us |
| Exponent/sign + LUT | SFPU | **843** | **40.0 us** | **77.1 us** |
| Direct mantissa/exponent/sign | SFPU | 1,157 | 54.8 us | 90.9 us |

Thus the selected version reduces L1 pipeline time by about 13.4% relative
to the previous fastest version. It delivers about **210.0 billion values/s**
or **419.9 GB/s of BF16 output** with L1 I/O, and **108.8 billion values/s**
or **217.5 GB/s of BF16 output** with DRAM I/O. These bandwidths count output
bytes only, excluding packed inputs and scale traffic.

With the same 128-repeat isolation method, the selected decoder measures
about **532 cycles/output tile** when tensor scaling uses the FPU (outside
that measurement), and **590 cycles/output tile** when tensor scaling uses
the SFPU. The earlier values were 656 and 718, respectively. The complete
pipeline table includes scale broadcast, transpose, and all other work.

Reports are timestamped `2026_10_27_16_31_43` (fresh original baseline),
`2026_10_27_16_32_37` (selected exponent/sign version),
`2026_10_27_16_34_13` (discarded direct field decoder), and
`2026_10_27_16_34_46` (selected decoder isolation). Each complete-pipeline
number is the median of calls 3 through 8, with 64 output tiles on the
busiest core and the profiler-reported 1,350 MHz clock. Temporary repeated
kernels were restored after measurement; only the selected decoder remains.


### Worktree-loop measurements, October 29

The [first bounded optimization run](OPTIMIZATION_2026_10_29.md) retained a
kernel-only candidate with shared SFPU setup, one compact-scale unpack,
scale expansion in DEST, unrolled plane/face loops, and reduced register
moves. Fresh paired measurements reduced complete L1-to-L1 conversion from
830.47 to 676.37 cycles per output tile (**18.6% less time**). A separate
run on the formatted branch measured 677.29 cycles/tile, 32.11 us, 261.3
billion values/s, or **522.5 GB/s of BF16 output**. The repeated math diagnostic
fell from 590.22 to about 441.38 cycles/tile; its extra staging is amortized
and its compiled scheduling may differ from the production binary.

The reconstruction gate statistics are unchanged: max 1.4375 ULP, worst
case mean 0.81677, median 0.75, and minimum SNR 46.23565 dB. The default
precise path and host packing contract are unchanged. The reader/writer
kernels are retained; a sub-threshold writer-batching trial was discarded.


The [TT-intrinsic/replay follow-up](OPTIMIZATION_2026_10_29_HOUR.md) measures
explicit register scheduling and native transposed-weight consumption. The
retained TT-intrinsic kernel uses five replay slots and prefetches the next
vector/face scales to hide dependencies. Independent pairs measure about
568 cycles/output tile versus 677 for the SFPI baseline, approximately 16%
less complete L1-to-L1 time, with the same reconstruction gate. The complete
DEST-sweep math diagnostic measures about 362 cycles/output tile. On 120
ideally balanced cores this projects to about 292 GElem/s, or 584 GB/s of
BF16 output; encoded input plus prepared scales accounts for about 183 GB/s.

### Latest bit-construction and harness follow-up

The [input-bit and evaluator-throughput loop](OPTIMIZATION_2026_10_29_LAYOUT_AND_HARNESS.md)
retains standard nibble packing. Opaque HI16 loads, arithmetic shifts, and a
programmable signed-magnitude mask remove separate sign insertion. The decoder
uses three/four replay instructions and measures about 298 cycles/output tile.
Ordinary tiled L1 output batches two writes; DRAM, transposed, and row-major
outputs retain single writes after consumer profiling exposed batching regressions.

Matched-clock 32-call confirmations measure about 521 versus 567 cycles/output
tile, approximately 7.5–8% less complete L1 conversion time, with unchanged numerical
gates. The tested ordinary-L1 linear projection improves about 3.2%; there is no
established DRAM gain. The harness now exposes screen/full modes, repeated timing,
per-stage/test counts, and device-lock/JIT overhead accounting.

### Transposed output for a linear consumer

`transpose_output=True` is an opt-in mode requiring `experimental_fast=True`
and `output_layout=ttnn.TILE_LAYOUT`. It returns the full transpose of the last
two decoded dimensions: packed data representing W[K,N] produces BF16 W.T[N,K].
Leading batch dimensions are preserved. Input preparation is unchanged:
`nvfp4.pack(codes_W, scales_W, experimental_fast=True)`.

The decoder keeps the column-oriented tiles in DEST and the writer exchanges
their tile-grid coordinates within each batch. Both the values inside each tile
and the tile grid are transposed; no separate device transpose is launched.
Use the result as `ttnn.linear(X, decoded_W_transposed, transpose_b=True, ...)`
or the equivalent matmul, selecting a native RHS-transpose program config such
as `MatmulMultiCoreReuseMultiCast1DProgramConfig`. Other backends may materialize
a transpose. The logical RHS maps to hardware SrcA in the matmul kernels.

Skipping the converter's DEST transpose can reduce conversion time, but RHS
memory-read strides also change. Measure the complete conversion-plus-consumer
workload: an L1 improvement does not imply a DRAM improvement. The default
`transpose_output=False` returns the ordinary decoded tensor and retains the
original output-layout contract.
