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

## Perf 1 — loss groups G1–G4 of generated/meta/g6_to_layout/rounds/r01_findings.txt
- **Date**: 2026-11-27
- **Goal**: beat stock `ttnn.to_layout` on every production scenario; bit-exact; SUPPORTED unchanged. The flagged case 308 already wins (9.4×), so it is a guard here. Device kernel ns, Blackhole p150b, chip 2. Every "stock" number below comes from the same process as the op's (bench `BENCH_IMPL=both`, median of launches 2–12).

### Measured breakdown (before)
Cumulative ablation (payload stubbed, synchronization kept; `perf1/ablate.sh` probe) and zones:

| Case | Full | Ablation steps | Ranked bottleneck |
|---|---:|---|---|
| c2036 R1 `[1,1024,2560]` | 46.1 µs | no reorder copies 26.3 · reads only 14.4 · writes only 18.9 · copies only 26.7 · sync only 2.5 | the 64 self-aimed 32-B NoC copies per tile on NCRISC (load_reorder_block 45.3k of 47.6k cyc), then the read/write overlap (1–2 blocks per core) |
| c49 R5 `[1,128,256,128]` L1 | 39.2 µs | no copies 17.7 · reads only 15.9 · copies only 28.6 · sync only 5.5 | the face-half copies on both DM RISCs; after them the row-read issue (reads alone ≈ stock's whole op) |
| c277 R5 `[1,64,128,256]` DRAM | 38.4 µs | no copies 30.1 · reads only 23.0 · writes only 15.1 · copies only 14.3 · sync 2.8 | row reads, then copies |
| small cells | — | c2025 (1 tile per core): read ~400 cyc + reorder 1960 + 32 row writes 1560 (~49 cyc each) + barrier 207 | serial chain: reorder copies, then per-row NoC issue (~48–54 cyc per row write: c2006, c29, c37 alike) |
| c37 R4 bf8b HEIGHT | 9.28 µs | writer waits 1456 + whole-shard NaN marks 2060 cyc, then 96 writes (5016 cyc) | the writer starts only after the whole shard's NaN scan |
| c29 R3 bf8b L1 | 4.84 µs | median core 4060 cyc, max 6005 (2-block cores) | writes of the cores that straddle two tile-rows |

The data-movement rooflines did not gate any of these: the reorder stage is RISC-issue-bound, not at a NoC or DRAM ceiling. My measurements agree with the operator's findings, with one correction to candidate 1's blocker (see below).

### Portfolio (one `perf-part-optimizer` each, `perf_experiments/<slug>/`)
| Idea | Verdict | Focus before → after (ns) | Domain / exceptions |
|---|---|---|---|
| `tensix_untilize_bf16`: R1's reorder on the Tensix | **WIN** | c2036 46,130 → 26,910 (stock 29,020) | Bit-exact only with **UInt16 CBs**. Float16_b plus a preserve-pinned Src zero flag alters 509/65536 patterns: -0.0 and subnormals become +0 and NaN becomes ±inf in unpack/pack, whatever the flag. Measured regressions: c297 +14% (later fixed by the row-issue idea) and the tiny H=1 cells +3–6% |
| `tensix_tilize_bf16`: R5's reorder on the Tensix | **WIN** | c49 39,179 → 14,960 (stock 16,304) | UInt16 CBs (Float16_b alters 522/65536). No exceptions. c277/c301/c272/c30 stay read-bound |
| `row_noc_issue`: cheap per-row NoC issue (reader + writer twin) | **WIN** | c2025 3,729 → 3,086 (stock 2,464) | `state_bulk`: bank cursor, stateful cmd buf, counters bumped once; 48 → 28 cyc per row. Nulls: per-core bank rotation; run-major coalescing (c297 0.85×, c273 0.93×). Hazard: full-grid fast L1 reads collapse |
| `bf8b_tile_row_pipeline`: R3/R4 marks per tile-row, write-then-check, NoC checkerboard | **WIN** | c37 9,347 → 5,066 (stock 6,904) | No measured exceptions. The cheap-scan variant was slower; live-face untilize was not built (c20 is writer-bound) |
| `small_tensor_split`: work split for few tiles per core | **WIN on the old DM path, superseded** | c2025 3,800 → 2,241 (16-row bands) | R1 sub-tile bands need the DM reorder that the Tensix untilize deleted. The R3 no-straddle split targeted c29, which now beats stock without it. Nulls: whole tile-rows per core (c2006 103 µs), fewer cores (c2025 +31%) |

### What graduated (commits `6cb5fb3e6f8`, `11cfdb9743a`, `14d737c29ef`, `c916a6e95a1`, `4227f032a38`)
- **R5 on the Tensix tilize.** It runs R6's reader, the tilize compute and the tile writer, with both CBs UInt16. Domain: every bf16 RM→TILE cell, no carve-out. Deleted: `load_and_reorder`, `g6_to_layout_face_copies.hpp`, the writer's `split_reorder` path, `ROW_WINDOW_DEPTH` and `SPLIT_REORDER`.
- **R1 on the Tensix untilize.** It runs R3's reader (live face-prefix reads kept: whole tiles made c294 27.3 vs 18.8 µs), the untilize compute and the row writer, with both CBs UInt16. Block columns are capped at 8 when H < 32 (c294 20.7 → 18.8 µs). Domain: every bf16 TILE→RM cell with more than one tile column. Deleted: `g6_to_layout_staged_reader.cpp`, `READ_BATCH_TILES` and `TILE_WINDOW_DEPTH`, and the writer's `tiled_stage` argument. Added `test_g6_to_layout_all_bf16_patterns.py`: all 65536 patterns, both directions, DRAM and L1, aligned, ragged and H=1 (20/20).
- **Row NoC issue.** `kernels/g6_to_layout_row_issue.hpp` serves the row writer (R1/R3/R4) and the row reader (R5/R6). A sharded accessor, a segment over 16 KB, or a single row takes the plain path inside the helper. Row reads from an L1 source are token-bucket paced (`L1_ROW_READ_GRID_BYTES_PER_CYCLE = 780`, one tile-row of burst). Unpaced, c38 went 28.6 → 37.3 µs. A plain gap without the burst cost c21/c45 +11%/+23%. The reader twin is about flat (DRAM −1..−3%, L1 ±1%); the writer twin is the win.
- **bf8b tile-row pipeline.** There is one cb_nan_marks page per tile-row (16-B summary header plus records). The writer consumes per tile-row, writes first, and patches and rewrites only a tile-row that holds a NaN. The per-tile-row wait also serves R1.
- **NoC checkerboard (host-side).** Cores with odd logical x + y get reader NoC1 / writer NoC0 through a second pair of kernel descriptors; there is no in-kernel NoC switch. It applies to R3, R4, R5 and R6, and to R1 at ≤ 4 tiles per core. **Carve-outs, each earned by a measured regression:**
  - R1 above 4 tiles per core: c283 362 → 412, c295 11.7 → 12.4, c304 2691 → 2789, c2036 26.4 → 27.3, c1998 7.54 → 7.82 µs. c2017, at 4 tiles per core, is flat.
  - R2/R7: Refinement 3 measured NoC0 run writes at 1497 vs 866 µs.
  - Measured slower and not adopted: x-only parity, y-only parity, a writer-only swap and an all-cores swap.

### Whole op, all 80 scenarios
Op after vs stock, same process. "Op before" is the operator's A/B at b01ab79f10d (my own baselines agree within 2%: c2036 46.1, c2025 3.78, c49 40.6 with zone sums, c2006 5.66, c277 39.3, c29 4.84, c2038 3.63, c37 9.28, c20 1.39 µs).
- **Before:** 55 slower than stock.
- **After:** 51 faster by > 3%, 20 within ±3% with the op not slower, 9 slower.
  - Within noise: c1981 1.000, c1978 1.003, c1987 1.005, c295 1.008, c276 1.016, c304/c306 1.020.
  - **Material losses: c49 1.066 and c277 1.031.**

| group | case | shape | op before (A/B, ns) | op after (ns) | stock (same run, ns) | after/stock |
|---|---|---|---:|---:|---:|---:|
| G1a | c1998 | 2x128x2560 | 14,000 | 7,539 | 8,746 | 0.862 |
| G1a | c2025 | 1x32x512 | 3,760 | 2,141 | 2,459 | 0.871 |
| G1a | c2009 | 1x256x2560 | 13,900 | 7,648 | 8,752 | 0.874 |
| G1a | c287 | 1x256x128 | 3,720 | 2,219 | 2,531 | 0.877 |
| G1a | c2003 | 1x32x2560 | 3,640 | 2,404 | 2,736 | 0.879 |
| G1a | c2023 | 1x32x1536 | 3,560 | 2,304 | 2,595 | 0.888 |
| G1a | c1992 | 1x64x512 | 3,760 | 2,220 | 2,499 | 0.888 |
| G1a | c1990 | 1x64x1536 | 3,690 | 2,493 | 2,791 | 0.893 |
| G1a | c2040 | 1x128x2560 | 8,010 | 4,436 | 4,747 | 0.934 |
| G1a | c273 | 1x64x128x32 | 6,770 | 5,399 | 5,727 | 0.943 |
| G1a | c2012 | 1x96x2560 | 6,970 | 3,791 | 3,990 | 0.950 |
| G1a | c292 | 32x5120 | 5,640 | 3,165 | 3,321 | 0.953 |
| G1a | c2017 | 1x160x2560 | 9,490 | 5,485 | 5,578 | 0.983 |
| G1a | c271 | 1x2048x128 | 7,050 | 4,013 | 4,061 | 0.988 |
| G1a | c2006 | 1x64x2560 | 5,670 | 3,145 | 3,159 | 0.996 |
| G1b | c283 | 1x512x1024x64 | 517,890 | 360,980 | 399,764 | 0.903 |
| G1b | c2028 | 8x128x2560 | 46,160 | 26,260 | 28,593 | 0.918 |
| G1b | c305 | 1x1024x2048x64 | 2,022,950 | 1,540,575 | 1,663,419 | 0.926 |
| G1b | c2036 | 1x1024x2560 | 45,990 | 26,464 | 28,379 | 0.933 |
| G1b | c1995 | 16x128x2560 | 85,240 | 53,549 | 56,466 | 0.948 |
| G1b | c1984 | 1x3872x2560 | 147,040 | 99,959 | 102,803 | 0.972 |
| G1b | c2031 | 1x4096x2560 | 153,150 | 104,841 | 106,608 | 0.983 |
| G1b | c280 | 1x256x512x256 | 391,540 | 301,535 | 306,311 | 0.984 |
| G1b | c282 | 1x512x1024x128 | 748,790 | 650,762 | 660,748 | 0.985 |
| G1b | c284 | 1x512x1024x128 | 748,570 | 650,762 | 660,748 | 0.985 |
| G1b | c281 | 1x256x512x128 | 194,020 | 160,005 | 162,014 | 0.988 |
| G1b | c300 | 1x256x256x256 | 202,500 | 152,412 | 154,138 | 0.989 |
| G1b | c274 | 1x64x128x256 | 34,840 | 21,906 | 22,075 | 0.992 |
| G1b | c275 | 1x64x128x256 | 34,910 | 21,906 | 22,075 | 0.992 |
| G1b | c278 | 1x64x128x256 | 34,740 | 21,906 | 22,075 | 0.992 |
| G1b | c303 | 1x512x1024x256 | 1,532,060 | 1,222,096 | 1,229,415 | 0.994 |
| G1b | c279 | 1x128x256x256 | 107,910 | 78,566 | 78,933 | 0.995 |
| G1b | c298 | 1x128x256x256 | 108,160 | 78,566 | 78,933 | 0.995 |
| G1b | c299 | 1x128x256x256 | 108,120 | 78,566 | 78,933 | 0.995 |
| G1b | c302 | 1x128x256x256 | 108,170 | 78,566 | 78,933 | 0.995 |
| G1b | c2020 | 8x1024x2560 | 271,790 | 206,584 | 206,980 | 0.998 |
| G1b | c1981 | 8x4096x2560 | 968,750 | 804,776 | 804,570 | 1.000 |
| G1b | c1978 | 16x4096x2560 | 1,782,900 | 1,602,773 | 1,598,166 | 1.003 |
| G1b | c1987 | 16x1024x2560 | 503,780 | 411,621 | 409,767 | 1.005 |
| G1b | c295 | 1x8192x128 | 17,790 | 11,731 | 11,633 | 1.008 |
| G1b | c276 | 1x128x128x256 | 60,060 | 41,003 | 40,362 | 1.016 |
| G1b | c304 | 1x1024x2048x128 | 2,960,290 | 2,681,241 | 2,627,776 | 1.020 |
| G1b | c306 | 1x1024x2048x128 | 2,933,450 | 2,681,241 | 2,627,776 | 1.020 |
| G2 | c45 | 1x32x128x256 | 10,700 | 5,313 | 9,681 | 0.549 |
| G2 | c21 | 1x32x64x256 | 7,080 | 4,430 | 5,544 | 0.799 |
| G2 | c38 | 1x128x256x256 | 65,450 | 29,063 | 32,527 | 0.894 |
| G2 | c30 | 1x64x128x256 | 21,370 | 9,127 | 9,624 | 0.948 |
| G2 | c49 | 1x128x256x128 | 39,160 | 17,307 | 16,233 | 1.066 |
| G3 | c272 | 1x64x128x32 | 16,720 | 10,110 | 15,641 | 0.646 |
| G3 | c301 | 1x128x256x256 | 129,230 | 90,544 | 97,690 | 0.927 |
| G3 | c277 | 1x64x128x256 | 38,760 | 26,895 | 26,085 | 1.031 |
| G4 | c37 | 1x64x128x256 | 9,290 | 4,727 | 6,911 | 0.684 |
| G4 | c2038 | 1x32x2560 | 3,520 | 2,118 | 2,559 | 0.828 |
| G4 | c29 | 1x32x64x256 | 4,870 | 3,228 | 3,544 | 0.911 |
| G4 | c20 | 1x1x1x256 | 1,420 | 1,286 | 1,296 | 0.992 |
| guard | c270 | 1x1x1x512 | 1,560 | 1,498 | 20,201 | 0.074 |
| guard | c2024 | 8x1x512 | 2,170 | 2,190 | 21,899 | 0.100 |
| guard | c286 | 1x1024x1024x3 | 271,460 | 272,650 | 2,585,121 | 0.105 |
| guard | c308 | 1x2048x2048x3 | 1,101,820 | 1,118,687 | 10,334,393 | 0.108 |
| guard | c1991 | 16x1x512 | 3,190 | 3,079 | 22,817 | 0.135 |
| guard | c289 | 1x1x1x256 | 1,420 | 1,458 | 10,450 | 0.140 |
| guard | c291 | 1x1x1x196 | 1,440 | 1,467 | 9,266 | 0.158 |
| guard | c312 | 1x1x1x207 | 1,440 | 1,469 | 9,250 | 0.159 |
| guard | c16 | 1x1x1x256 | 1,650 | 1,667 | 10,344 | 0.161 |
| guard | c310 | 1x1x1x185 | 1,460 | 1,438 | 8,023 | 0.179 |
| guard | c269 | 1x512 | 1,480 | 1,547 | 7,599 | 0.204 |
| guard | c1989 | 16x1x1536 | 6,610 | 6,499 | 27,382 | 0.237 |
| guard | c2022 | 8x1x1536 | 4,370 | 5,190 | 14,454 | 0.359 |
| guard | c288 | 1x256 | 1,430 | 1,536 | 4,059 | 0.378 |
| guard | c311 | 1x207 | 1,450 | 1,545 | 3,642 | 0.424 |
| guard | c290 | 1x196 | 1,450 | 1,547 | 3,642 | 0.425 |
| guard | c309 | 1x185 | 1,420 | 1,496 | 3,236 | 0.462 |
| guard | c48 | 1x64x128x128 | 5,970 | 3,168 | 6,741 | 0.470 |
| guard | c296 | 1x128x256x32 | 63,940 | 38,045 | 63,952 | 0.595 |
| guard | c43 | 1x128x256x32 | 13,450 | 9,113 | 15,252 | 0.597 |
| guard | c293 | 1x1x131072 | 19,220 | 18,798 | 25,337 | 0.742 |
| guard | c294 | 1x131072 | 19,050 | 18,769 | 25,275 | 0.743 |
| guard | c2026 | 8x1x2560 | 4,030 | 4,066 | 5,426 | 0.749 |
| guard | c1993 | 16x1x2560 | 7,010 | 7,201 | 9,527 | 0.756 |
| guard | c297 | 1x128x256x32 | 16,700 | 16,713 | 17,301 | 0.966 |


### Guard set and regressions against the commit before this pass
- Golden 91/91 after every graduation.
- Unit tests pass: 31 + 8 + 11 + 33 + 20 + 20, plus the new all-patterns test 20/20.
- c308 is 1,118,687 ns (before 1,101,820, +1.5%, noise) and c286 272,650 (271,460); R7 is untouched.
- c296 leads stock by 40% (was 1%); c297 leads by 3.4% (was 3%).
- **Measured regressions accepted, with reasons:**
  - **Tiny H=1 TILE→RM cells, +4 to +7% (70–100 ns).** c269 1,480 → 1,547; c288 1,430 → 1,536; c290 1,450 → 1,547; c309 1,420 → 1,496; c311 1,450 → 1,545. The cause is the compute kernel's fixed cost on 6–16-tile cells. They still beat stock by 2.1–4.9×. A carve-out would bring back the deleted DM reorder as a second R1 path, and that dual path is not worth 70–100 ns.
  - **c2022 `[8,1,1536]` RM→TILE, 4,370 → 5,190 ns (stock 14,454).** This comes from the NoC checkerboard: 5,153 with it vs 4,236 without. Its siblings gain from it (c1989 6.63 → 6.36, c272 16.2 → 10.0), and no predicate separates them other than fencing this one shape.
  - c1993 7,010 → 7,201 (+2.7%, noise edge).

### Remaining headroom (measured, not closed)
- **c49 (L1 RM→TILE, 256-B rows), 1.066×.** It is 15,457 ns without the checkerboard and 17,307 with it, while c30 is 9,127 with it and 10,782 without (stock 9,624). One of the two loses to stock either way, so the unified rule (checkerboard on) stands.
- **c277 (DRAM RM→TILE), 1.031×.** It is bound by row reads: 512-B pages, ~74 per core, and reads alone took 23 µs before this pass.
- **The large DRAM-bound G1b cells (c304/c306/c276/c1978/c1987/c1981/c295)** sit at 1.00–1.02× of stock, all at 1.1–1.2× of the DRAM roofline. Bank-run write coalescing would need a run-major stage, and the packer cannot produce one.

### Operator candidates
- **operator candidate 1: graduated.** Both reorders moved to the Tensix (`tensix_untilize_bf16`, `tensix_tilize_bf16`). The preserve-zero flag did not close the blocker: the Float16_b unpack/pack path itself turns -0.0 and subnormals into +0 and NaN into ±inf, which is 509/522 patterns either way. UInt16 CB views of the bf16 bytes did close it (0/65536). Before → after: c2036 46,192 → 26,464 ns, c49 39,199 → 17,307 (15,457 before the checkerboard), c277 38,615 → 26,895.
- **operator candidate 2: not graduated.** With the reorder on the packer, more column blocks per core are slower on c2036. Capping block columns at 12/8/6/4 gave 26.6 / 27.2–27.6 / 28.9 / 30.0 µs vs 26.0–26.9 at 1×24, so the writer's wait is no longer the limit. The H < 32 cap of 8 did graduate: c294 20.7 → 18.8 µs. Before → after: c2036 46,192 → 26,464 ns, from candidate 1.
- **operator candidate 3: graduated** (`bf8b_tile_row_pipeline`, plus the NoC checkerboard from the same experiment). c37 9,266 → 4,727 ns: 6,499 from the per-tile-row marks alone, the rest from the checkerboard.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `noc_async_write` + `TensorAccessor::get_noc_addr` (per row) | capability | No dataflow API rewrites only the target coordinate of a stateful write command, so each row to the next interleaved bank pays a full 6-register command, a divide by the bank count and two counter updates in L1 (~48 cyc per row). The command register is written raw instead, and the issued/acked counters are bumped once per call. | c2025 3,729 (accessor) | c2025 3,086 (whole op, row-issue experiment) | `kernels/g6_to_layout_row_issue.hpp:119` (counters `:125`) |
| `noc_async_read` + `TensorAccessor::get_noc_addr` (per row) | capability | The same gap for reads: there is no coordinate-only update of a stateful read command, and no way to issue N reads with one counter bump (`noc_async_read_inc_num_issued` exists but does not pair with a per-row coordinate change). | c49 issue 47.6 cyc per read | 30.0 cyc per read | `kernels/g6_to_layout_row_issue.hpp:165` (counter `:171`) |

The Tensix untilize and tilize paths use `compute_kernel_lib::untilize` / `tilize` unchanged: the bit-exactness comes from the host's UInt16 CB format, not from raw LLK.

- **Issues encountered:**
  - The shell exported `TT_METAL_HOME` / `PYTHONPATH` pointing at `/localdev/pmilojevic/tt-metal`; every run here exported the slot root.
  - Tracy's capture tool crashed after about a third of the passing bench runs; the device CSV survives, and `perf1/dev_median.py` reads kernel spans from it.
  - The repo hook forbids a module-level `import torch` under `ttnn/ttnn`, and the experiment tests now use `pytest.importorskip("torch")`.
