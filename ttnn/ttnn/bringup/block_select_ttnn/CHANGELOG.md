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

### block_max_bf16 strip_major (deepseek_v41_flash_d_p PB.FBM, 2026-11-22; ported to model_bringup by MB.8)
- What: `block_max_bf16(..., strip_major=True, output_tensor=[.., 32 S, NB])`: the input is
  `ring_indexer_score_dsa(block_max_compute=True)`'s strip-major bf16 `[.., S NB / 8, 256]` (512-byte pages of 8
  blocks); each strip's [NB, 32] is transposed to [32, NB] (exact copies), the first n blocks (to 8). New kernel
  kernels/block_max_strip.cpp: work units of 128 blocks of a strip (16 page reads, a two-words-in two-words-out
  transpose, 32 row writes) over both RISCs of every core, double-buffered; n a value slot or a device tensor, as the
  fold. Prepared program.
- Why: PB.FBM's compute-kernel block maxes come out a strip at a time (a tile's 32 row maxes in one 64-byte run); the
  top-k needs [Sq, NB] rows. 8 MB a chip at 1M instead of the writer path's 64 MB of fp32 maxes.
- Measured (8x4, one chip 256 rows, model probe): 0.14-0.21 ms a call at 69632-73728 keys, 0.21 at 516096 (the fold
  of the writer path: 0.16-0.24 / 0.57).
- Tests: tests/unit/test_block_select.py check_block_select: 5 strip-major cases (n a value and a tensor, n not a
  multiple of 8, nothing written past n), byte-exact.
- Files: block_select.py, kernels/block_max_strip.cpp, tests/unit/test_block_select.py, README.md
