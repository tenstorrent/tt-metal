# fp8_rows_ttnn changelog

Source: none (new op; the row packing of ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache, another row
layout), deepseek_v41_flash_d_p P.39 (INDEX_K_SERVED=fp8_row192, owner 2026-11-06: Blaze decode compatibility).

## Changes

### Initial op (2026-11-06)
- What: `ttnn.bringup.pack_fp8_rows(e4m3, scale, output_tensor=None, memory_config=None, max_grid=None)`: the stock
  per_token_cast_to_fp8's e4m3 [.., M, 128] + fp32 scale [.., M, 1] -> uint8 [.., M, 192] rows [e4m3 | scale |
  zeros], all-zero for an all +-0 row; one prepared data-movement program (common runtime args only).
- Why: the served index keys of deepseek_v41_flash_d_p become Blaze decode's indexer key cache rows (192 B); the
  stock packer writes GLM's 656-byte rows (latent 512 | 4 scales | RoPE) only.
- Tests: tests/unit/test_pack_fp8_rows.py (8x4: 4 sizes byte-equal to the cast's raw outputs, the RM-input cast the
  same, Blaze's recipe within one e4m3 step with equal scales, pad rows zero, negative control, prepared on fresh
  buffers + a caller's buffer; refusals).
- Needed by: deepseek_v41_flash_d_p P.39
- Files: fp8_rows.py, __init__.py, kernels/pack_fp8_rows.cpp, tests/unit/test_pack_fp8_rows.py, README.md;
  ../__init__.py (PYTHON_OPS), ../INDEX.md

### unpack_fp8_rows (2026-11-18)
- What: `ttnn.bringup.unpack_fp8_rows(rows, out, row_lo, row_hi, batch=0, out_row0=None, max_grid=None)`: 192-byte
  rows [e4m3 x 128 | fp32 scale | ..] (uint8 ROW_MAJOR in DRAM, interleaved or ND-sharded) -> e4m3 x scale (fp32 on the
  SFPU) rounded to nearest even onto bfloat8_b (ckv_bfp4_ttnn's `ckv_round_bfp_face<7>`), in place into a bfloat8_b
  TILE tensor at a tile row. Reader: 32 rows a band read and shuffled into 4 FP8_E4M3 tiles (face order) + one fp32
  scale tile (the row's scale in every column); compute: copy (fp8 via srcA, the scale UnpackToDestFp32), SFPU
  multiply, SFPU round, pack; writer: 4 tiles a band. Prepared (OV-23): built once per (shapes, memory configs, core
  count); the range is 5 value slots a core; cores = min(grid, bands).
- Why: deepseek_v41_flash_d_p P.76 (owner 2026-11-18, INDEX_K_SOURCE=served): the prefill indexer scores the served
  rows instead of a private bfloat8_b copy; no device reader of the rows existed (the stock per_token_cast_back needs
  FP8_E4M3 + a separate scale and writes RM bf16 / fp32).
- Tests: tests/unit/test_unpack_fp8_rows.py (8x4: ND-sharded and interleaved, ranges, batches, an offset output row,
  rows outside untouched, determinism, 65536 rows a chip exact, refusals; timing printed).
- Gotcha: copy_init only checks the unpacker state (state_configure is the sentinel); switching between the fp8 and
  the fp32 CB needs reconfig_data_format_srca, else the scale tile is read in the fp8 format (garbage).
- Files: fp8_rows.py, __init__.py, kernels/unpack_fp8_rows_{reader,compute,writer}.cpp,
  tests/unit/test_unpack_fp8_rows.py, README.md; ../__init__.py (PYTHON_OPS), ../INDEX.md
