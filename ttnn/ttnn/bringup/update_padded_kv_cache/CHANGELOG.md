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

### Tests: fork model-case suite with the deepseek_v41_flash_d_p cases (O.1, 2026-10-27)
- What: tests/cases.py, reference.py, test_update_padded_kv_cache.py: one random-input case for each of the 57 calls
  deepseek_v41_flash_d_p makes on 2x2 (capture 2026-10-27): 4_index_k_r1 (27 scalar chunk writes, SP-sharded,
  TP-replicated, 128 wide), 0_window_kv / 3_compress_kv_r1 (28 scalar writes, TP-dedup, 512 wide, one at the
  non-aligned start 53120), 1_compress_kv_r2 and 2_index_k_r2 (the metadata path, starts inside a stripe and valid
  ends inside a 32-row block). The cache in the served layout (ND-sharded one record per DRAM bank, the model's
  topology) with random prior content per chip; every chip's whole cache compared exactly (written rows vs the
  reference's block-cyclic placement of the rotated input rows, every other row unchanged); determinism with the cache
  restored before every call. reference.py's placement agrees with the kernels' own index arithmetic on 240 random
  starts / valid ends (2x2, 2x4, 8x4; CPU cross-check, not shipped). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: deepseek_v41_flash_d_p O.1
- Files: tests/cases.py, tests/reference.py, tests/test_update_padded_kv_cache.py

### Upstream sync to source @ 6b80daf43e3 (rebase onto llk_helper_library, owner: full sync, 2026-10-27)
- Taken: #61236 (an input row window: input_row_offset / input_rows, plain page copy only, default the whole input)
  and the header move (#60308). The window is unused by our models.
- Conflict: the device-operation entry point's signature (upstream's two new arguments taken). The fork's ROW_MAJOR
  row-granular validation (page_row_unit) merged clean next to upstream's windowed validator; checked by hand.

### max_grid: the op's cores inside a rectangle (deepseek_v41_flash_d_p FIX.15)
- What: option `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.update_padded_kv_cache` (scalar and tensor / metadata forms): every program of the op places its cores (and splits its work) inside (0, 0) .. (x - 1, y - 1) of the compute grid. Default: the device's whole grid, the call as before (same program, same hash).
- Why: agent rule 18 (owner 2026-10-30): a Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); no op may run on more than 110 of a chip's 120 Tensix cores. DeepSeek-V4.1 passes tt/cores.max_grid() = 11 x 10 on a Galaxy, None elsewhere.
- Measured: 8x4 Galaxy, one 5120-token chunk (update_padded_kv_cache): 120 -> 110 cores a chip (tests/perf/test_core_audit.py in deepseek_v41_flash_d_p); per-op device time in that model's BREADCRUMBS FIX.15.
- Needed by: deepseek_v41_flash_d_p FIX.15
- Files: device/update_padded_kv_cache_device_operation.cpp, device/update_padded_kv_cache_device_operation.hpp, update_padded_kv_cache.cpp, update_padded_kv_cache.hpp, update_padded_kv_cache_nanobind.cpp
