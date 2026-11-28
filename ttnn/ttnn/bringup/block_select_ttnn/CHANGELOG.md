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

### Native C++ ops (2026-11-28, deepseek_v41_flash_d_p NC.6)
- What: ttnn.bringup.block_gather / block_keys / block_max_bf16 are now the native C++ device ops ttnn.block_gather /
  ttnn.block_keys / ttnn.block_max_bf16 (ttnn/cpp/ttnn/operations/<op>: device operation, program factory, nanobind;
  kernels moved to each op's device/kernels with an args header; block_max_bf16 carries both modes, the factor fold
  and strip_major, n a value or n_tensor). Same kernels, cores, runtime args and signatures; the program cache keys on
  shapes / memory configs / block / grid (no addresses, no n), a hit rewrites the addresses (and n). The pre-port
  Python route kept for tests only in generic/ (with its kernels).
- Why: OWNER 2026-11-27 C++ port series (bringup/NATIVE_PORT_RULES.md): the Python prepared programs' wrapper, key and
  run_prepared patching cost host time a call (9 calls a chunk on the indexer's block path at long context).
- How: tt_ops_code_gen native_migrate (separate checkout, one entry a run: block_gather 20.9 min, block_keys 27.2,
  block_max_bf16 22.0, all ported; acceptance suites in tests/ttnn/unit_tests/operations/<op>/pre_migration, golden
  tests/unit/test_block_select.py no worse). After each run the sibling entries' kernels were put back (DS41-402).
- Measured (8x4, models/demos/deepseek_v41_flash_d_p/tests/native/test_native_block_select.py, results/NC.6_layer.md):
  bit-identical to the Python route at the model's shapes (single 8192 / packed n_tensor / 5120), 0 cache growth on
  fresh buffers and new n; host a call gather 122.4 -> 30.0 us, keys 87.1 -> 30.9, strip 62.7 -> 20.8, fold 151.7 ->
  32.2; device equal (gather 246 us, strip 118 us at 256 x 16384).
- Files: block_select.py, kernels/ -> ttnn/cpp/ttnn/operations/{block_gather,block_keys,block_max_bf16}, generic/ (new,
  test only), README.md; ../INDEX.md; ttnn/cpp/ttnn-nanobind/__init__.cpp, ttnn/cpp/ttnn/operations/CMakeLists.txt.

### Brought to model_bringup (2026-11-28, deepseek_v41_flash_d_p MB.NC)
- What: the native C++ ttnn.block_max_bf16 / block_keys / block_gather (NC.6), taken as is from dnijemcevic/model_bringup_batching (the entries above); the pre-port Python route stays
  in `generic/` for the op test only.
- Why: OWNER 2026-11-27, one batch of the batching branch's C++ ports on model_bringup (no packed path here).
- Measured (8x4, 5120 chunk shapes, models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/results/MB.NC_layer.md):
  native == the Python prepared route bit for bit, repeats 0, no program-cache growth on fresh buffers.
- Files: as the entries above; tests models/demos/deepseek_v41_flash_d_p/tests/native/, tests/woa/test_gmm_native_layer.py.
