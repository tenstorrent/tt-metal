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
