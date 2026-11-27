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

## Refinement 3 — Speed up the perf-flagged profile: DRAM write transactions on case 308
- **Date**: 2026-11-27
- **What was done**: the design's deferred `same_bank_slot_writes` regime is built as **R7**. It is selected by shape: one tile column, `W ≤ 16`, and `2·W < out_slot_bytes` (the output's aligned page; 64 B in DRAM). That covers every DRAM-output R2 shape, so cases 308 and 286 move off `face_direct`. `face_direct` stays for L1 outputs whose `2·W` is a multiple of 16.
  - **Write side.** An interleaved bank holds pages `p, p+B, p+2B…` in consecutive 64-B slots. So each block's rows are gathered into a run-major slot image, and every bank run is written in one NoC write. On 308 that is 105 600 writes of 2560 B instead of 4 194 304 writes of 6 B. The slot's alignment gap is written too: it is the output buffer's own memory.
  - **Image build.** The build is split by rows (`reader_slot_eighths = 5`):
    - The reader moves its share with RISC word loads/stores. It issues block `k+1`'s face reads before building its rows of block `k`, so those reads stay in flight during the build.
    - The writer moves the rest with self-aimed NoC reads, then writes the runs. It pops the previous image after one `noc_async_writes_flushed`.
  - **Depths and block size.** `cb_left_faces` and `cb_slot_image` are 3 deep, so the two halves of consecutive blocks overlap. R7 has its own `slot_pipeline_blocks_per_core = 128` (308: 10-tile blocks).
  - **Reused**: R2's face loads, the block walk, `cb_left_faces`, `_choose_blocking` (now taking the block count), the local-copy helpers (`set_read_state` / `read_with_state`).
  - **Added**:
    - shared `kernels/g6_to_layout_slot_image.hpp` (run layout, row cursor, both copy engines);
    - the slot mode of `g6_to_layout_face_reader.cpp` / `g6_to_layout_face_writer.cpp`;
    - `cb_slot_image` (index 6);
    - knobs `SLOT_FACES_DEPTH`, `SLOT_IMAGE_DEPTH`, `READER_SLOT_EIGHTHS`, `WRITER_SLOT_NOC_COPIES`, `SLOT_PIPELINE_BLOCKS_PER_CORE`.
  - `select_plan` reports the regime, `out_slot_bytes` and the R7 knobs. `test_regime_selection` now pins 286/308 to `same_bank_slot_writes`. `op_design.md` and `l1_ledger.md` carry the R7 row, its CBs and the new data-movement budget (308 output: 24 → 256 MiB).
  - R7's zones are two run totals per RISC (`TT_METAL_PROFILER_SUM`). At ~120 blocks per core, per-block zones overflow the ~250-marker buffer after ~31 blocks, and the overflow is silent.
- **Perf (device kernel time, BH, 110 cores; same session, before → after)**:

  | Case | Before | After | Speedup |
  |------|--------|-------|---------|
  | 308 `[1,2048,2048,3]` | 3394.7 µs | 1119.8 µs | 3.03× |
  | 286 `[1,1024,1024,3]` | 797.1 µs | 273.0 µs | 2.92× |

  Every other perf-file shape (R1 c304/c1978/c2031/c294/c1993/c2038, R3/R4 c29/c43/c20/c37/c48, R5/R6 c301/c277/c296/c1989/c38/c49/c16/Q3) is unchanged within noise. Their kernels and plans did not change.
- **Steps measured on 308** (µs):
  1. Writer-only build with 32-B NoC copies: 1429. The build cost 637 µs, ≈22.5 cycles per copy.
  2. RISC word copies instead: 1315. The build cost 403 µs, ≈14.3 cycles per row, and it is store-bound: stores alone take 320 µs, ≈5.7 cycles per store. Turning the data cache off changed nothing.
  3. RISC copies split across both RISCs: flat (1305, then 1309 after the dependency fix). Two causes:
     - The two halves first serialized on the 2-deep CBs.
     - Once overlapped, the RISCs' L1 accesses contend: per-row cost went from 14 to 22–26 cycles.
  4. Reader RISC copies plus writer NoC copies: 1265.
  5. Plus 128 blocks per core: 1113.
- **What binds now** (ablations at the final configuration, 308):
  - reads only: 520 µs;
  - slot-run writes only: 866 µs (≈310 GB/s);
  - both, no copies: 1034 µs;
  - full: 1113 µs.

  So the op is bound by the DRAM traffic of the scheme. The copies add ≈80 µs. The kernel time is set by the core rows whose read issue blocks on NoC0 back-pressure (reader load 66–690 µs, graded by core row y).
- **Measured nulls (not kept)**:
  - rotating each core's run write order: 1108–1113 µs, noise;
  - writes on NoC0: 1497 µs writes-only, worse;
  - writes alternating between NoCs: 854 vs 866 µs, flat;
  - depth 4/4: 1125 µs, flat;
  - 3/8, 4/8 and 6/8 reader shares: 1190, 1148 and 1197 µs.
- **Accuracy achieved**: bit-exact (PCC = 1.0, relative RMS = 0). This covers loose 308 specials (-0.0, subnormals, ±inf, NaN payload) and 308 poisoned padding (1000.0 and NaN). It also covers the new ragged shapes: `[3,50,5]`, `[7,5,3]`, `[1,1,2]`, `[2,1000,16]`, and L1 outputs with 16-B and 32-B slots, each with NaN-poisoned padding.
- **Golden test progress**: a targeted slice passes 20/20, with no regression on prior paths:
  - every 308 cell (INPUTS plus the 4 loose: perf, specials, poison 1000, poison NaN) and 286 plus its loose case;
  - representatives c304, c294 (with its loose cases), c2038, c29, c37, c301, c16.

  Expected total: 89/89 (no SUPPORTED change).
- **Issues encountered**:
  - In the profile build, two zone macros in one scope collide (`conflicting declaration 'hash'`).
  - Tracy's capture tool crashed once (capture side); the re-run was clean.
  - `/opt` MPI is fine on this slot, so no `LD_PRELOAD` stub was needed.
- **Tests added**:
  - `test_g6_to_layout_slot_writes.py` (14 tests): regime selection and bit-exact output on ragged images, `H < 32`, a single row, `W = 16` with ragged `H`, L1 outputs with 16-B and 32-B slots, and `face_direct` on an L1 full slot, each with randn and NaN-poisoned padding.
  - `test_g6_to_layout.py::test_regime_selection` pins updated.

## Refinement 4 — Speed up the perf-flagged profile: read overlap and block co-tune on case 308
- **Date**: 2026-11-27
- **What was done**: every named lever is built into the R7 reader as a host knob, measured on 308 and left at its no-op default. None moved 308.
  - **Trid read ring across blocks** (`SLOT_READ_DEPTH`). Block `n`'s face reads carry trid `n % depth + 1`, and only the oldest block's trid is waited on, so reads stay in flight across every wait. Block `n` lands in page `n % SLOT_FACES_DEPTH`, addressed from the ring base, because the CB write pointer lags the blocks issued ahead of it. `SLOT_FACES_DEPTH` is now derived as `SLOT_READ_DEPTH_MAX + 2`. Depth 1 is Refinement 3's schedule. `face_direct` runs the same reader at depth 1.
  - **Adaptive read-ahead** (`SLOT_READ_DEPTH_MAX`). After its build, the reader issues more blocks only while the oldest has not landed (a non-blocking trid poll), so only starved cores deepen their queue.
  - **Read `bank_stagger`** (`SLOT_READ_STAGGER`). Each block's first read goes to input bank `(x + y) % banks` of the core.
  - **Block co-tune**: `SLOT_PIPELINE_BLOCKS_PER_CORE` × read depth.
  - R2/R7 blocks are capped at 127 tiles. A block's two reads per tile share one trid, whose outstanding counter holds 255.
  - `select_plan` reports `slot_read_depth`, `slot_read_depth_max` and `slot_read_stagger`. `op_design.md` and `l1_ledger.md` carry the knobs, the derived faces depth and the findings.
  - **Reused**: R7's reader loop, `load_left_faces`, the kernel_lib trid helpers (`set_read_trid` / `async_read_barrier_with_trid`), `_max_block_tiles`.
  - **Added**: the trid ring, the adaptive issue and the stagger offset in `g6_to_layout_face_reader.cpp`, plus three knobs.
- **Perf (device kernel time, BH, 110 cores, 308 unless noted)**:

  | Variant | 308 (µs) |
  |---------|---------:|
  | Refinement 3 baseline (4 runs) | 1095–1130 |
  | Read depth 2 / 3 / 4 | 1144 / 1152 / 1151 |
  | Adaptive max 2 / 4 | 1147 / 1173 |
  | Stagger on / off | 1110 / 1109 |
  | 64 blocks/core, depth 1 / 2 | 1119 / 1128 |
  | 256 blocks/core, depth 1 / 2 | 1109 / 1125 |
  | Final (defaults) | 1112.7 (286: 273.0) |

  The guard paths match Refinement 3 within noise: c304 2933, c1978 1788, c2031 153, c294 19.3, c2038 3.6, c29 4.9, c37 9.2, c301 130, c38 67.7, Q3 6.2 µs.
- **What binds 308**:
  - **Ablations** (at Refinement 3's code):
    - reads only: 530 µs;
    - writes only: 866 µs;
    - both without the copies: 1041 µs;
    - sync only: 9.7 µs.
  - **Per-core gradient.** With equal work per core, kernel time runs from 400 µs (core row y = 11) to 1120 µs (y = 2). NoC0 traffic grades by row:
    - reads on NoC0: 155 → 528 µs;
    - writes on NoC0: 80 → 1500 µs.

    NoC1 traffic grades by column instead:
    - reads on NoC1: x = 1 at 222 µs, x = 15 at 590 µs;
    - writes on NoC1: 74 → 862 µs.
  - **Zone split.** The top rows' readers wait up to 725 µs for read responses, and their read issue costs 59–293 µs, graded by column. The writer's flush wait is 1.6 µs in total; it spends up to 777 µs waiting for the reader's input. So the reader is the critical path, and no writer twin would move 308.
  - **Queue depth does not help.** Deeper rings, uniform or adaptive, raise the favoured cores' share and slow the top rows further.
  - **Volume barely helps.** Removing 30% of the reads outright gives only 977 µs.
  - **Distribution barely helps.** A probe-only split weighted by measured row speed gives 1075 µs. The top rows still finish at ~1066 µs with 40% less work.
  - **Conclusion.** The gradient is NoC0 arbitration order, not idle capacity. 308 moves 402.6 MB at ≈368 GB/s; c304 (363 GB/s) and c1978 (375 GB/s) reach the same combined read + write rate through different kernels. The op sits at this board's DRAM ceiling for mixed traffic.
- **What next**:
  - Only fewer DRAM bytes would move 308. The reads (128 MiB) are the face-row minimum, and the 64-B page slot fixes the bytes of a run write.
  - A dynamic block queue (NoC atomic fetch-and-add, which BH supports) is bounded at ≈2–5% by the weighted-split probe, so it was not built.
- **Accuracy achieved**: bit-exact (PCC = 1.0, relative RMS = 0). This holds on 308 and 286 at every knob setting measured, and on the new knob tests with NaN-poisoned padding.
- **Golden test progress**:
  - The targeted slice passes 11/11: every 308 cell (INPUTS plus 4 loose: perf, specials, poison 1000, poison NaN), 286 and its loose case, c304 and c2038.
  - Expected total: 89/89 (no SUPPORTED change).
- **Issues encountered**:
  - The NoC estimator CLI is not built on this checkout (it needs `--build-tests`), so the DRAM ceiling comes from the measured mixed-traffic rate of three ops.
  - `tt_metal/third_party/tt_ops_code_gen` shows as modified in the working tree. It is not part of this change.
- **Tests added**: `test_g6_to_layout_slot_writes.py::test_read_ring_knobs` (6 tests). It checks bit-exact output with NaN-poisoned padding at read depth 3, adaptive max 4, and depth 2 + adaptive 4 + stagger, on a ragged 12-block-per-core shape and on c286. These cases pin the ring's landing-page arithmetic.
  - The probe `probes/test_probe_r4_variant.py` swaps the kernel directory, knobs, NoCs and a row-weighted split for measurement.
