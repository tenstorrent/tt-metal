# Changelog: g6_to_layout

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). It is a dataflow-only TILE → ROW_MAJOR conversion with two regimes:
  - R1 `staged_reorder`: live face-row prefixes are read in trid-pipelined batches, reordered by 32-B self-aimed NoC copies into a row block, and written as one write per live row segment.
  - R2 `face_direct`: for one tile column with `W ≤ 16`, faces 0/2 are read and each row is written as `2·W` bytes straight from its face row.
  - Work is split over the full grid by a flattened row-major tile range (`split_work_to_cores`, `row_wise=True`).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], output_dtype=[bfloat16], memory_layout=[INTERLEAVED] (DRAM or L1), alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4]
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0.0, mean_abs_err=0.0, rel_rms_err=0.0, max ULP=0, got/true ratio 1.0 (p5–p95 1.0–1.0). All injected specials (-0.0, subnormals, ±inf, NaN payload) are preserved. Measured on 4 shapes (c309, c1993, c286, c2020) via `test_g6_to_layout_precision_baseline.py`.
- **Golden suite at Phase 0**: 58 / 89 registry cells passing (49/73 INPUTS + 9/16 LOOSE_CASES); 31 `xfail_expected`, all queued in Refinements 1–2; 0 supported_fail / xpass_drift / xfail_wrong_mode; 2/2 regression tests pass (`verifier_report.json` in `generated/g6_verify/`).
- **Perf at Phase 0** (device kernel, 110 cores): 308 (perf focus) 3331–3432 µs; 304 3008 µs; 1978 1762 µs; 286 785–794 µs. 308 is bound by its 4.19 M 6-B DRAM writes: writes alone 2167 µs, reads alone 517 µs.
- **Issues encountered**: the implementer's work for this plan was uncommitted in the working tree; it is committed with the verifier pass. Verifier fixes:
  - the R2 slot size is now defined once in the shared kernel header (it was duplicated in the R2 reader and writer);
  - the `pipeline_blocks_per_core = 16` departure from the coarsest-fit block was re-measured and recorded in `l1_ledger.md` (5–6% faster than the L1-cap block on R1).
- **Tests added**: `test_g6_to_layout.py` (acceptance, 31), `test_g6_to_layout_perf.py` (production perf shapes), `test_g6_to_layout_precision_baseline.py` (8).

## Refinement 1 — bfloat8_b TILE input and sharded inputs (HEIGHT / BLOCK)
- **Date**: 2026-11-27
- **What was done**: `ttnn.bfloat8_b` added to `SUPPORTED["dtype"]`; `HEIGHT_SHARDED` and `BLOCK_SHARDED` added to `SUPPORTED["memory_layout"]`. bf16 × sharded is in `EXCLUSIONS` (no production cell; the packer path flushes bf16 -0.0/subnormals). Two regimes went from deferred to built:
  - **R3 `packer_untilize`** (bf8b, interleaved DRAM/L1). It keeps Phase 0's full-grid tile-range split and block walk. The new `g6_to_layout_tile_reader.cpp` reads whole 1088-B bfp8 tiles into `cb_tile_window`, with one barrier per block. The new `g6_to_layout_untilize_compute.cpp` runs `compute_kernel_lib::untilize<block_cols_tiles>`, with init/uninit once per kernel, into `cb_row_block` tile pages. R1's `g6_to_layout_staged_writer.cpp` is reused, in tiled-stage mode. The compute config is hard-coded (HiFi4, 16-bit DEST), because the signature has no `compute_kernel_config`.
  - **R4 `sharded_in_place`** (bf8b, HEIGHT/BLOCK in L1). `cb_tile_window` is backed on the resident shard (`cb_descriptor_from_sharded_tensor`, zero-copy, rebound on every call), so the reader issues no NoC read. The shard is the block, and a ragged last shard moves only its live tile-rows. Core assignment is the caller's shard grid (`ShardPlacement.core_rects`).
  - **NaN rendering.** Unpack → SrcA → pack turns a bfp8 NaN datum into ±inf. The host renders it as 0xFFFF, and no compute routing emits that. So the reader marks each block's NaN datums on a new `cb_nan_marks` (reader → writer, one page per block), and the writer stores 0xFFFF over them before writing. Shared code is in `kernels/g6_to_layout_bfp8_nan_marks.hpp`. -0.0, subnormals, ±inf and zeros of 0xFF groups are already bit-exact through the packer, so no preserve-zero routing was needed.
  - **Scan fast path.** The reader's scan gates the writer's first row. It now tests all 16 exponent words of a tile in one unrolled, non-volatile OR-reduce, and walks groups only when a 0xFF byte exists. Device kernel time before → after: c37 12.7 → 9.3 µs, c48 7.7 → 6.0 µs, c29 5.4 → 4.8 µs.
  - Cache key and `select_plan` carry the shard spec (grid, shard shape, orientation) (BB-07/08).
  - `op_design.md` and `l1_ledger.md` carry the R3/R4 regime rows, the new CBs and the data-movement budget.
- **Accuracy achieved**: bit-exact against the host's bfloat16 rendering of the packed bfloat8_b input. PCC = 1.0, relative RMS = 0, including the injected specials (NaN, ±inf, -0.0, subnormals). Shapes: [1,32,64,256] (c29), [1,128,256,32] (c43), [1,1,1,256] (c20, BLOCK), [1,64,128,256] (c37, HEIGHT), [1,64,128,128] (c48, HEIGHT).
- **Perf (device kernel time, BH)**:

  | Case | Cores | Time | Bound by |
  |------|-------|------|----------|
  | c29 | 110 | 4.8 µs | — |
  | c43 | 110 | 13.4 µs | writer: 298 64-B L1 row writes per core |
  | c20 | 8 | 1.4 µs | — |
  | c37 | 86 | 9.3 µs | writer: about 6.5 µs of 96 × 512-B writes after ~2.7 µs of NaN scan |
  | c48 | 86 | 6.0 µs | — |

  Remaining headroom on R4: marks per tile-row, which would let the writer start on tile-row 0 while the reader scans tile-row 1.
- **Golden test progress**: the 8 target cells pass in a `-k BFLOAT8_B` slice: c20, c29, c37, c43, c48, loose c29 specials (exact), and loose c37/c48 with the L1 guard. bf16 representatives pass with no regression (c304, c294, c2038, c286, c1993, c309 and their loose cases: 11/11). Expected total: 66/89 (58 + 8); the remaining 23 `xfail_expected` are Refinement 2's ROW_MAJOR-input cells.
- **Issues encountered**: the packer renders bfp8 NaN as ±inf (the earlier attempt's 775 bit mismatches on loose c29), fixed by the reader → writer NaN marks. The NaN scan initially dominated the sharded critical path (reader 6.2 µs on c37); fixed by the fast path. The host lost `/opt/openmpi-v5.0.7-ulfm`, so tests run with `LD_PRELOAD=generated/mpix_stub/libmpix_stub.so`.
- **Tests added**:
  - `test_g6_to_layout_bf8b_sharded.py` (11 tests): the five production bf8b cases, bf8b specials bit-exact on an interleaved and a ragged-shard case, regime selection, and bf16 × sharded rejection.
  - `test_g6_to_layout_perf.py::test_perf_bf8b` (5 tests).
