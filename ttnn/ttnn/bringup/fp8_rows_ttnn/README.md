# ttnn.bringup.pack_fp8_rows (fp8_rows_ttnn)

## What it does
New op (Python ProgramDescriptor; deepseek_v41_flash_d_p P.39, INDEX_K_SERVED=fp8_row192). Packs the outputs of the
stock per-token fp8 cast into Blaze decode's 192-byte indexer key cache rows:

    e4m3 [.., M, 128] FP8_E4M3 ROW_MAJOR + scale [.., M, 1] FLOAT32 ROW_MAJOR
      (ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x, round_scale_to_power_of_two=False))
    -> out [.., M, 192] UINT8 ROW_MAJOR: [e4m3 (128 B) | scale fp32 little endian (4 B) | 60 zero bytes]

The layout of tt-blaze's `DsaIndexerDirectKCacheUpdate` (blaze/ops/dsa_indexer_direct_k_cache_update,
SCALED_ROW_BYTES 192). A row whose 128 values are all +-0 (a zeroed pad row) is written all zero, its scale too (the
cast writes the clamp floor 1e-4 / 448 there). Byte copies only: one data-movement kernel (NoC 0), rows split over
the cores by common runtime args (no per-core args), staged 16 rows at a time at a 64-byte aligned stride (the scale
read lands at row offset 128). Always dispatched through ttnn.bringup.prepare_program / run_prepared (built once per
shape, memory configs and grid; the three addresses patched a call, OV-23).

## Consider it instead of
- `ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache` (GLM's sparse-MLA rows: 512 B latent | 4 scales |
  64 bf16 RoPE, 656 B) when the row is a 128-value key with one scale padded to 192 B.
- host-side packing of fp8 rows.

## Options
- `output_tensor` (ttnn.Tensor, default None = the op allocates): the caller's buffer, exactly the output's spec
  (uint8 ROW_MAJOR DRAM interleaved [.., M, 192]); returned as is.
- `memory_config` (default None = DRAM interleaved): the allocated output's memory config.
- `max_grid` (ttnn.CoreCoord or (x, y), default None = the whole compute grid): the cores inside (0, 0) ..
  (max_grid - 1) (a Galaxy's 110-core cap, agent rule 18).
Realistic (use by default): every option (no arithmetic).

## Measured
- 8x4 Galaxy (tests/unit/test_pack_fp8_rows.py): M 320 / 640 / 32 / 96, random keys (a large channel, a tiny row,
  zero / -0 / half-zero rows), bf16 TILE through the stock cast: every byte equal to the cast's raw outputs packed on
  the host; the RM-input cast (tt-blaze's GLM oracle path) writes the same rows; vs tt-blaze's torch recipe
  (_quantize_scaled_keys) the scales are byte-equal and values differ by at most one e4m3 step (~1 % of values: the
  cast lands one step higher and keeps e4m3 subnormals the recipe flushes).
- Cores: ceil(M / 16) up to the grid: 20 / 40 a chip for deepseek_v41_flash_d_p's 320 / 640-row chunk slabs on 8x4.
- Device / host time: not measured by the agent (the orchestrator's A/B of P.39).

## Limits
- e4m3 width 128 and one scale a row only (the indexer key); DRAM interleaved inputs and output.
- The cast itself is the stock op: its TILE path runs one core per 32 x 128 block and takes no grid option (the
  caller keeps M / 32 <= its core cap: tt/compressor.py fp8_rows asserts it).

## How to call
```python
e4m3, scale = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(keys_tile, round_scale_to_power_of_two=False)
rows = ttnn.bringup.pack_fp8_rows(e4m3, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG, max_grid=ttnn.CoreCoord(11, 10))
# models/demos/deepseek_v41_flash_d_p/tt/compressor.py fp8_rows
```

## Used by
- deepseek_v41_flash_d_p: `tt/compressor.py` (`fp8_rows`, TtServedRows' writes of the served index keys,
  INDEX_K_SERVED=fp8_row192, P.39).

## History
CHANGELOG.md
