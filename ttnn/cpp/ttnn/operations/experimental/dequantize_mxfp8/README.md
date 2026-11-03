# MXFP8 dequantization on Blackhole

`ttnn.experimental.dequantize_mxfp8` decodes OCP MX FP8 with E4M3 elements:

```
output = E4M3(code) * 2^(E8M0(scale) - 127)
```

Each block of 32 horizontally adjacent values shares one E8M0 scale. Unlike
NVFP4 there is no tensor-wide scale, and no rounding: an E4M3 value has at most
four significant bits, so every product in the BF16/FP32 normal range is exact.

> **Status:** reference SFPU decoder (`device/kernels/compute/mxfp8_sfpu.hpp`).
> It builds each FP32 result with integer exponent arithmetic: exact, with
> overflow to ±inf and underflow flushed to zero. It is unoptimized: one
> datacopy plus one SFPU pass per output tile. `UINT8` input requires
> `fp32_dest_acc_en=True`, because its datacopy produces Integer-32 in DEST.

## Performance (reference SFPU decoder)

Measured 2026-11-03 with `scripts/quant_harness.py --format mxfp8 evaluate
--math-isolate --perf-repeats 4` on Blackhole p150b at 1350 MHz, 120 cores, in
the primary configuration: `UINT16` input, BF16 DEST and output, tiled,
interleaved L1. The workload is 26880x2048 values (224 packed tiles = 448
output tiles per core), with 24 retained samples.

| Metric | Value |
|---|---|
| L1-to-L1 pipeline | 1477 cycles per output tile (MAD 0.09) |
| Isolated SFPU decode sweep (×128) | 1422 cycles per output tile |
| One call, 55.05 M values | ≈ 490 µs ≈ 112 G values/s ≈ 225 GB/s of BF16 output |

The decode sweep is about 96% of the pipeline, so the kernel is compute-bound
in the SFPU. For comparison, the optimized NVFP4 kernel runs at about 530
cycles per output tile on the same device.

## Usage

```python
packed, scales = ttnn.experimental.mxfp8.pack(codes, scale_codes, dtype=ttnn.uint16)
data = ttnn.from_torch(packed, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
block_scales = ttnn.from_torch(scales, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
output = ttnn.experimental.dequantize_mxfp8(data, block_scales)  # BF16, TILE
```

`codes` holds E4M3 bit patterns `[H, W]` and `scale_codes` holds E8M0 codes `[H, W // 32]`.
H must be a multiple of 32. W must be a multiple of 32 for `uint8` and 64 for `uint16`.

## Host contract

| | |
|---|---|
| Data | `UINT8` (1 E4M3 byte/word; requires `fp32_dest_acc_en=True`) or `UINT16` (2 bytes/word, least significant first), TILE or ROW_MAJOR, rank 2-4, H and W multiples of 32, interleaved DRAM/L1 |
| Packing | Tile-plane packing as in NVFP4: byte *p* of every word of packed tile *k* is output tile `planes*k + p` of the same tile row. For `UINT8` this is the identity, so the tensor holds the E4M3 bytes in natural positions, byte-identical to an FP8_E4M3 tensor of the same layout. |
| Scales | `UINT8` ROW_MAJOR, raw E8M0 codes, shape `[packed_tiles, 32 * planes]`; per packed tile: byte plane, then output-tile row |
| Output | `BFLOAT16` (default) or `FLOAT32` (requires `fp32_dest_acc_en=True`), TILE (default) or ROW_MAJOR; last dim = packed last dim × planes |
| Numerics | Exact. Results ≥ 2^128 are signed ∞. Results < 2^-126 may flush to zero or be the correctly rounded subnormal. NaN encodings (E4M3 `0x7F`/`0xFF`, E8M0 `0xFF`) are outside the contract and `pack` rejects them. |

## Kernel contract (what a candidate kernel may assume)

The program factory and host API are fixed during kernel optimization. Kernels
may use these buffers however they like, but changing a buffer's format or size
is a host change (rebuild) outside the kernel-only harness.

| CB | Contents | Format | Pages |
|----|----------|--------|-------|
| c0 | packed data, one packed tile per page, face order | UInt8 / UInt16 | 2 |
| c1 | raw E8M0 bytes for one packed tile (32 × planes bytes, DRAM-aligned page) | UInt8 | 2 |
| c2 | reader staging for ROW_MAJOR input | UInt8 | 1 |
| c3 | kernel-owned workspace (e.g. prepared integer scales) | UInt16 32×32 | 2 × planes |
| c4 | kernel-owned workspace (e.g. prepared float scales) | Float16_b 32×32 | 2 × planes |
| c16 | decoded output tiles | output dtype | 2 × planes |
| c17 | writer staging for ROW_MAJOR output | UInt8 | 1 |

Compute kernel `compute/dequantize.cpp`:
- Compile-time args: `planes`, `uint8_input`, `bf16_output`, `fp32_dest_acc_en`.
- Runtime arg: the number of packed tiles for this core.
- `dst_full_sync_en=true`, HiFi4.

Reader compile-time args are `tile_h`, `tile_w`, `word_bytes`, `width_tiles`, `tiled`, `page_bytes`, `scale_bytes` and `staging_stride`, followed by the data and scale tensor accessors. Its runtime args are the data address, scale address, first packed tile and count.

Writer compile-time args are `tile_h`, `tile_w`, `element_bytes`, `output_width_tiles`, `tiled` and `page_bytes`, then the output accessor, then `planes`. Its runtime args are the output address, first output tile and count.

Keep `// MXFP8_MATH_SWEEP_BEGIN` / `// MXFP8_MATH_SWEEP_END` around the complete
per-packed-tile decode sweep; the harness's isolated-math diagnostic repeats it.

## Hardware notes relevant to kernel design

- Blackhole has no native MX unpack: the `MxFp8*` formats are Quasar-only.
- The unpacker reads `Fp8_e4m3` as FP16 (A-family, 5-bit exponent) into SrcA/DEST.
  Every E4M3 value fits FP16 exactly. The E8M0 range does not, so scaling must happen outside FP16.
- The packer can convert FP16 in DEST to an L1 BF16 CB (late conversion).
  However, `program.cpp` (BH/Quasar) rejects any compute kernel with an FP8 CB unless `fp32_dest_acc_en=True`.
  A native-unpack path therefore needs FP32 DEST and a CB declared as `Fp8_e4m3`, which is a host change.
- FP8_E4M3 tensors are currently ROW_MAJOR-only in ttnn.

## Tests

| File | Covers |
|------|--------|
| `tests/ttnn/unit_tests/operations/mxfp8_reference.py` | Independent decoder, exhaustive corpus, exact gate, metrics |
| `test_mxfp8_accuracy.py` | Exhaustive survey: every finite (code, scale) pair, both containers, all DEST/output modes. Writes `MXFP8_ACCURACY_REPORT` |
| `test_dequantize_mxfp8.py` | Packing contract (host), all layout/container/DEST combinations, ranks 3-4, program cache + L1, streaming, trace, matmul consumer, validation |
| `test_mxfp8_candidate_screen.py` | Exhaustive corpus, primary configuration only |
| `test_mxfp8_perf.py` | L1-capacity-sized benchmark in the primary configuration (`UINT16` input, BF16 DEST, BF16 TILE output) |

See [HARNESS.md](HARNESS.md) for the optimization loop.
