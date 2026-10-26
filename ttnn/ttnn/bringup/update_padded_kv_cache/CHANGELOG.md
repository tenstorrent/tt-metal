# update_padded_kv_cache (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache`
- Source SHA: `e3b3caa5c75e6deb0670edf8e9bc48f099b7e5c5`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: deepseek_v41_flash_d_p W.1
- Used by: deepseek_v41_flash_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_update_padded_kv_cache`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Restore the sibling namespace of the packed fp8 layout
- What: `namespace packed = ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache` (host and both
  kernels): fork_op.py had nested it under `bringup`, but the packed layout stays the source's shared sibling op.
- Why: build error "no namespace named 'pack_scaled_fp8_kv_cache' in namespace 'ttnn::operations::bringup'".
- Needed by: deepseek_v41_flash_d_p W.1
- Files: device/update_padded_kv_cache_device_operation.cpp, device/kernels/dataflow/{reader,writer}_update_padded_kv_cache.cpp

### ROW_MAJOR copy path: per-chip stripes and cache depth in whole rows, not whole tiles
- What: on the ROW_MAJOR copy path (no RoPE / scales inputs) the miss validator checks `(input_seq / 1) % tp_factor`
  and `cache_seq % 1` instead of 32-row units; the input seq itself stays a multiple of 32. The reader and writer were
  already row-granular there (writer_tile_height 1, page = one row), so no kernel, compile arg or program changes; TILE
  and split-input calls are validated as before. No option: only calls the stock op refused behave differently.
- Why: DeepSeek-V4.1-Flash on galaxy_8x4 at chunk 5120: its TP-deduped ratio-2 compressed cache writes 640 / 2 / 4 =
  80 rows a chip (TT_FATAL "input seq in tiles (10) must be divisible by tp_factor (4)"), and its cache at 56320 tokens
  is 11 x 80 = 880 rows a chip ("cache seq dim must be tile-aligned").
- Needed by: deepseek_v41_flash_d_p W.1
- Files: device/update_padded_kv_cache_device_operation.cpp, tests/unit/test_row_stripes.py
- Source tests: the source file's conftest (models/demos/deepseek_v3_d_p/tests/conftest.py) does not import on this
  tree (xing40_a4b_d_p/tt/settings.py passes the removed `max_precision` to Settings, F137), so the recorded baseline
  (tests/source_baseline.json, entry 0) is a collection error on both the original and the fork; re-record once it imports.
