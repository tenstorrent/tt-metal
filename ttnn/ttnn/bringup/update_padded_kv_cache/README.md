# ttnn.bringup.update_padded_kv_cache

## What it does
Fork of `ttnn.experimental.deepseek_prefill.update_padded_kv_cache`. Adds one thing: on the ROW_MAJOR copy path (no
RoPE / scales inputs) a chip's written stripe (input_seq / tp) and the cache depth need only be whole rows, not whole
32-row tiles, so a TP-deduped ROW_MAJOR cache can take stripes such as 80 rows (a ratio-2 compressed cache, 640 tokens
a mesh row over 4 columns). Everything else is the stock op.

## Consider it instead of
`ttnn.experimental.deepseek_prefill.update_padded_kv_cache`, when a TP-deduped ROW_MAJOR cache's per-chip stripe or
depth is not a multiple of 32 rows.

## Options
- `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.update_padded_kv_cache` (scalar and tensor / metadata forms): the op's cores inside that rectangle of the grid (FIX.15, a Galaxy's 110-core cap: 11 x 10); default the whole grid.
None (the relaxed check applies to ROW_MAJOR copy calls only; calls the stock op accepts run the same program).

## Measured
not measured (validation-only change; kernels unchanged). tests/unit/test_row_stripes.py: 8x4 exact vs torch.

## Limits
- The input seq stays a multiple of 32; the valid_global clamp still rounds to 32 page-rows (as the stock op).
- Row stripes are unit-tested at window-aligned chunk starts (scalar and metadata paths, valid clamp) on 8x4 only;
  non-aligned starts with row stripes run only in the model (DeepSeek-V4.1 contract tests).
- Source tests: not runnable on this tree (deepseek_v3_d_p tests conftest import error, see CHANGELOG).

## How to call
As the stock op: `ttnn.bringup.update_padded_kv_cache(cache, input, slot, kv_actual, valid_global=valid,
layer_idx=l, num_layers=L, cluster_axis=0, tp_axis=1)`; see tests/unit/test_row_stripes.py.

## Used by
deepseek_v41_flash_d_p (W.1): tt/compressor.py TtServedRows (compressed caches), tt/attention.py (window cache).

## History
CHANGELOG.md
