# Quantized routed rows (fp8 / int8 y on the combine's wire)

The routed experts' output rows can cross the combine's fabric quantized: `MoEBlockConfig.routed_y_quant`
(`RoutedRowFormat.Bf16` / `Fp8E4m3` / `Int8`) and `routed_y_quant_block` (values a scale block: 32, 64, 96, or 0 for
the down compute's pass width). In the model: `MoEBlockFfnOptions(routed_y_quant="int8", routed_y_quant_block=32)`.
The flat expert alone: `flat_routed_expert(..., y_quant=RoutedRowFormat.Int8, y_quant_block=0)`.

## Format (`common/kernels/quantized_row.hpp`)

`[H values, a byte each][a 16-byte cell a down core][32-byte trailer]`; a cell holds its core's exponent bytes, one per
16 values (a scale block repeats its byte). GLM-5.3: 6144 + 512 (+ 32) B against bf16's 12288 (+ 32).

* Int8: sign-magnitude magnitude x a power-of-two scale, exponent byte = log2(s) + 133: 16 values and their byte are a
  Bfp8_b face row, so the consumer unpacks a flat tile as a native Bfp8_b tile.
* Fp8E4m3: e4m3 x a power-of-two scale (e8m0 byte); subnormals flushed (the unpacker flushes them anyway).

Scales are powers of two: quantizing is an exponent shift, and the consumer's gate x scale x 1 / q stays exact, so the
combine's int32 snap keeps its sums bit-identical in any landing order.

## Producer: the flat expert's down cores (`device/kernels/y_quant.hpp`)

* The down compute runs fp32 DEST in **half sync** when y is quantized: a pass (3 columns for GLM) is quantized on the
  PACK thread's SFPU while the MATH thread runs the next pass's matmul.
* Per (token row, block): amax over the block's lanes, the scale, then every value's final byte (round to nearest even)
  as an int32 in DEST, packed as is through a UInt8 out CB (`pack_untilize_dest`); the packer's own fp32 -> e4m3
  conversion is not used (it truncates and gets subnormals wrong).
* Scales leave through DEST: the SFPU stores them in the half's spare tile, TRISC2 reads them through its RISC view of
  DEST into the row's cell (CB c_17); the y writer sends each row's values and its cell.
* Blackhole: `pack_untilize_dest` of an odd pass width into an 8-bit format writes one tile past the pass - the out CB's
  row stride is padded by a pass.

## Consumer: the combine's accumulators

* Reader (`quantized_row_stage.hpp`, `RowStager`): the row's values into the row CB (Int8: behind each Bfp8_b tile's
  exponent section), its cells into a scratch; once landed, Int8's exponents into the tiles, Fp8E4m3's FP16 factor
  tiles (a factor a DEST row).
* Compute (`quantized_row_weigh.hpp`, `RowWeigher`): Int8 a datacopy of the Bfp8_b tile; Fp8E4m3 an ELWMUL of the
  codes by the factor column (fp8 unpacks to FP16, and the FPU takes one exponent family); then the gate x 1 / q and
  the snap on the SFPU. Output rows stay bf16 (`out_row_bytes`).

## Measured (2026-11-25)

Flat expert alone, Galaxy shapes on a p150b (`test_flat_routed_expert_checks.py -k quant`): row error against the op's
bf16 y int8 1x96 1.0%, fp8 2.65% (PCC 0.99836 / 0.99806); device time int8 1x96 +3.5% / +4.2% (GLM 8x4 / 4x4 5120),
fp8 +8%.

Experts + combine, GLM-5.3 16 experts a chip, LoudBox 2x4 200G, the fabric packet one quantized row (6720 B):

| rows | M 320 | M 384 | PCC |
|---|---|---|---|
| bf16 | 1949 us | 2529 us | 0.99840 |
| int8 1x96 | 1500 | 1724 | 0.99836 |
| int8 1x32 | 1495 | 1726 | 0.99837 |
| fp8 1x96 | 1580 | 1754 | 0.99805 |

All bit-identical over 10 calls (uniform and zipf routing).

Tests: `test_flat_y_quant.py` (the quantize bit exact against `quantized_row_golden.py`),
`test_flat_routed_expert_checks.py -k quant` (accuracy, determinism, perf), `test_combine_quantized_row_bench.py` (the
accumulator on staged rows, within 1 bf16 ulp of an exact golden).
