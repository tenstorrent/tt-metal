# block_select_ttnn changelog

Source: none (new ops; the per-row block gather `ttnn.gather` cannot do at 1M-key rows, see README), deepseek_v41_flash_d_p
PB.BM (OWNER 2026-11-21: indexer top-k by exact block-max pre-selection).

## Changes

### Initial ops (2026-11-21)
- What: `ttnn.bringup.block_gather`, `ttnn.bringup.block_keys`, `ttnn.bringup.block_max_bf16` (README): one
  data-movement kernel each, prepared programs, common runtime args only; block_max_bf16's length a value slot or a
  device tensor, both RISCs, double-buffered reads.
- Why: deepseek_v41_flash_d_p PB.BM: the index top-k over 512k-1M keys a row (3.2 ms a layer a chip at 1M) as a top-512
  of 16k block maxes + a top-512 of 16k gathered scores (~1.0 ms), exact up to 512th-value ties.
- Tests: tests/unit/test_block_select.py (byte-exact vs torch: block 1 / 2 / 8 / 16 / 32, none and out-of-range ids,
  repeated blocks, -inf tails, ties; prepared cache hit + a caller's buffer; block_max_bf16 factor 1 / 4 / 16 with n
  as a value and a tensor, nothing written past n).
- Needed by: deepseek_v41_flash_d_p PB.BM
- Files: block_select.py, __init__.py, kernels/block_gather.cpp, kernels/block_keys.cpp, kernels/block_max_bf16.cpp,
  tests/unit/test_block_select.py, README.md; ../__init__.py (PYTHON_OPS), ../INDEX.md

### block_gather on both RISCs (2026-11-21)
- What: block_gather's work unit = K / 4 blocks of a row (ids read at a 64-byte aligned offset), split evenly over
  both RISCs of every core at run time; the row's base address once a unit; every read of a unit before one barrier.
- Why: deepseek_v41_flash_d_p PB.BM: 256 rows on 110 cores left 36 cores with three whole rows (0.41-0.44 ms a call);
  now 0.30-0.34 ms. Same bytes (tests/unit/test_block_select.py).
- Files: kernels/block_gather.cpp, block_select.py, README.md

### Ported to model_bringup's single-request path (2026-11-22)
- What: no op change; the fork folder taken unchanged from the batching branch (e821084f82f) to model_bringup, with
  the PYTHON_OPS / INDEX.md entries. The model probe tests/perf/test_block_select_probe.py takes its rows from the
  spec's chunk (BSEL_ROWS, BSEL_BLOCKS).
- Why: deepseek_v41_flash_d_p MB.6 (OWNER 2026-11-21: block-max pre-selection on the shipped single 5120 path).
- Measured: 160 rows a chip, block 32: 0 mismatch rows at T 25600 .. 512000 (README, Measured).
- Files: README.md, CHANGELOG.md; ../__init__.py, ../INDEX.md (entries).
