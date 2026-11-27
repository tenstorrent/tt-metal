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

## Refinement 2 — ROW_MAJOR input → TILE output (bfloat16 and bfloat8_b)
- **Date**: 2026-11-27
- **What was done**: `ttnn.ROW_MAJOR_LAYOUT` added to `SUPPORTED["layout"]` and `ttnn.bfloat8_b` to `SUPPORTED["output_dtype"]`. The design's deferred `row_major_to_tile` regime is built as two regimes. Both are in-kernel, on the full grid, with Phase 0's tile-range split and block walk applied to the output tile space. There is no wrapper.
  - **R5 `row_major_to_tile`** (bf16 → TILE bf16) is dataflow-only, so it is bit-exact by construction.
    - The new `g6_to_layout_row_reader.cpp` reads each live row segment of a tile-row into a `cb_row_stage` slot. These loads are trid-pipelined, `row_window_depth = 2` tile-rows in flight.
    - 32-B self-aimed copies then move the face rows into the block's tile slots of `cb_tile_block`. This is the inverse of R1's reorder.
    - The new `g6_to_layout_tile_writer.cpp` writes whole tiles.
    - The copies are split by face half (`split_reorder`, a host knob). The reader copies the top faces on NoC0, pushes each landed slot, and the writer copies the bottom faces on NoC1 (shared code in `g6_to_layout_face_copies.hpp`).
  - **R6 `packer_tilize`** (bf16 → TILE bf8b): the same reader lands the block's rows in `cb_row_stage` as 32-row tile-rows. The new `g6_to_layout_tilize_compute.cpp` makes one `compute_kernel_lib::tilize` call per core, which packs bfp8 into `cb_tile_block`. The writer is R5's.
  - **bfp8 conditioning** (reader, before the push). Two steps:
    1. Every bf16 datum with an all-ones mantissa (0x7F) is lowered to the next value toward zero. When such a datum holds its group's exponent, the BH packer rounds it into the next binade and requantizes the whole group one exponent higher. The host saturates the mantissa and keeps the exponent instead.
    2. When `W % 16 ≠ 0`, the padding that shares a 16-value group with a live value is zeroed.
  - Shared code:
    - `first_out_page_of` is renamed to `first_row_page_of`, because the RM page is the input in this direction.
    - The row-segment length is now `row_segment_bytes()` in the walk header, used by the R1/R3/R4 writer and the new reader.
  - The plan computes the L1 output share in tile pages for a TILE output. `select_plan` reports the new regimes, plus `row_window_depth` / `split_reorder`.
  - `op_design.md` and `l1_ledger.md` carry the R5/R6 regime rows, CBs (indices 4 and 5), caps, schedule and data-movement budget.
- **Accuracy achieved**:
  - bf16 output: bit-exact (PCC = 1.0, relative RMS = 0). This includes the injected specials (-0.0, subnormals, ±inf, NaN payload) on [16,1,1536] (loose c1989), [1,1,1,185] and [3,48,100].
  - bf8b output, within one host quantization step everywhere:
    - [1,1,1,256] (c16): PCC = 0.999973, relative RMS = 0.0091
    - [1,1,32,256] (Q3, group-scaled): PCC = 0.999971, relative RMS = 0.0093
    - [1,2,70,208]: PCC = 0.999972, relative RMS = 0.0093
  - The host's own bfp8 rendering has relative RMS 0.0073. 76–79% of device datums are bit-equal to it; the rest are one step away, which is the packer's rounding.
- **Perf (device kernel time, BH, 110 cores)**, before and after the face-half split:

  | Case | Shape | Before | After | Notes |
  |------|-------|--------|-------|-------|
  | c301 | [1,128,256,256] DRAM | 161.5 µs | 129.3 µs | |
  | c38 | L1 → L1 | 86.5 µs | 67.2 µs | |
  | c49 | [1,128,256,128] L1 → L1 | 49.8 µs | 39.2 µs | |
  | c277 | [1,64,128,256] | 42.8 µs | 38.7 µs | |
  | c296 | [1,128,256,32] | 62.8 µs | 64.4 µs | flat: W = 32 |
  | c1989 | [16,1,1536] | 6.7 µs | 6.7 µs | flat: H = 1, so there are no bottom faces to share |
  | c16 | bf8b | 1.7 µs | 1.7 µs | |
  | Q3 | bf8b | 9.5 µs | 5.7–6.1 µs | conditioning loop unrolled 8 wide: 6.94 → 3.65 µs |

  Findings:
  - Before the split, the reader was bound by 64 copies per tile (c38: about 0.93 µs per tile) while the writer waited 95% of the time.
  - After the split both RISCs are balanced on c38 (49 vs 48 µs). Copy throughput rose only 1.45× (per-copy cost 20 → 27 cycles), so the two halves contend for a shared resource.
  - `row_window_depth = 4` was measured and is mixed (c38 +13%, c49 −7%, the rest flat), so it was kept at 2.
  - c296 has 64-B rows. Its reader issue time is 20 µs mean and 52 µs max across cores: skew, not in-flight depth. `bank_stagger`, or batching two tile-rows per load, is the next lever.
  - Q3's remaining conditioning cost is L1-cache-line fill latency on scalar loads (~7 cycles/word). An SFPU pass in compute would remove it, at the cost of a raw-SFPU stage and an extra CB.
- **Golden test progress**:
  - The slice of all 23 `ROW_MAJOR` target cells passes: 18 bf16 INPUTS, c16, loose c1989 specials (exact), loose Q3 (`block_step`), and loose c38/c49 with the 1347776-B L1 guard intact.
  - Prior-path representatives also pass (37/37 with R1 c2038/c294, R2 c286, and every R3/R4 bf8b cell).
  - Expected total: 89/89.
- **Issues encountered**:
  - The bfp8 packer's renormalization on mantissa round-up overflow broke the `block_step` gate on 4 of 8192 Q3 values (316 of 524288 on [1,1024,512]), fixed by the conditioning pass. BH exposes no packer saturation control.
  - The pre-commit black hook crashed on a corrupt user cache ("Ran out of input"); commits run with `BLACK_CACHE_DIR=generated/black_cache`.
- **Tests added**:
  - `test_g6_to_layout_row_major_to_tile.py` (33 tests):
    - the 18 bf16 production cells plus 3 off-production walk corners (H % 32 ≠ 0 across images, a 131072-wide row, 128 tile-rows);
    - bf16 specials;
    - bf8b c16 / Q3 / W=200 / [1,1024,512] (block_step);
    - L1 guard (c38/c49);
    - regime selection.
  - `test_g6_to_layout_perf.py::test_perf_row_major_to_tile` (8 shapes).
