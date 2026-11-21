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

### unpack_fp8_rows `whole_grid` (2026-11-20)
- What: `unpack_fp8_rows(..., whole_grid=False)`: True runs every call on the whole grid (max_grid's cores; cores past
  the band count take 0 bands and return, as the kernels already handle), so ONE prepared program serves every row
  count. Default False keeps the behaviour before (cores = min(grid, bands): one program a band count below the grid).
  Python only (no kernel change).
- Why: deepseek_v41_flash_d_p MB.2 (OWNER 2026-11-20, rule 20: serving never builds a program after warm-up): the
  history prefetch decodes [0, c wr) rows, a band count that grows with the request's position, so short histories
  prepared a new program per band count on fresh traffic.
- Tests: tests/unit/test_unpack_fp8_rows.py::test_unpack_fp8_rows_whole_grid (1 / 3 / 37 bands, bytes equal to the
  default's, one whole-grid program of the current mesh). 8x4 (2026-11-20): the fork's suite (pack + unpack, option
  off) 10 passed as before; the new case passed (its first version counted the precompile pass's mesh too: fixed).
- Needed by: deepseek_v41_flash_d_p MB.2 (tt/index_keys.py `_decode`, under POS_DYNAMIC=on)
- Files: fp8_rows.py, tests/unit/test_unpack_fp8_rows.py, README.md, ../INDEX.md

### unpack_fp8_rows `common_values` (2026-11-21)
- What: `unpack_fp8_rows(..., common_values=False)`: True passes the per-run values (first row page, bands, out page)
  as COMMON runtime args (reader / writer args 1, 2; compute arg 0; kernel define UNPACK_COMMON_VALUES, no per-core
  args) through prepare_program's new common value slots, so a run patches five words a program instead of five a
  core. Default False keeps the behaviour (and the program) before. The same bytes.
- Why: deepseek_v41_flash_d_p PB.9X: under POS_DYNAMIC the index-key decodes run on the whole grid (whole_grid, MB.2);
  110 cores x 5 value slots x 32 chips cost ~0.33-0.36 ms of host a call (24 calls a packed chunk: ~8 ms of a
  host-bound glued chunk's enqueue, tests/packed/test_onecall_host.py PB3A_PROBE_OPNAMES).
- Tests: tests/unit/test_unpack_fp8_rows.py::test_unpack_fp8_rows_common_values (1 / 3 / 37 / 300 bands, bytes equal
  to the default's, one program; host ms a call printed).
- Needed by: deepseek_v41_flash_d_p PB.9X (tt/index_keys.py `_decode`, setting COMMON_VALUE_SLOTS)
- Files: fp8_rows.py, kernels/unpack_fp8_rows_{reader,writer,compute}.cpp, tests/unit/test_unpack_fp8_rows.py,
  README.md, ../INDEX.md; needs ../prepared_program's common value slots (same date)
