# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-01
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One native `ttnn.generic_op` dispatch: TensorAccessor reader → compute (eltwise_chain DEST-accumulated Σx², reduce collapse/combine + rsqrt finalize, fused `x·inv_rms·gamma` chain; tilize/untilize for ROW_MAJOR) → writer. Row-groups over the whole Tensix grid (R1); W split over k ≤ 8 Tensix cores with a symmetric unicast all-gather of per-row partials (R2) when tile-rows ≤ grid/8 or W does not fit L1.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"], gamma_layout=[TILE, ROW_MAJOR, "none"], memory_layout=[INTERLEAVED]; EXCLUSIONS={float32 + fp32_dest_acc_en=False}
- **Accuracy achieved** (4 shapes × {bf16, fp32} × {TILE, ROW_MAJOR}, `test_rms_norm_precision_baseline.py`): bf16 PCC ≥ 0.999997, max_abs_err ≤ 0.094, rel_rms_err ≤ 0.0038 (≤ 2.5 bf16 ULP); fp32 PCC ≥ 0.9999997, max_abs_err ≤ 0.036, rel_rms_err ≤ 0.0020 (tf32-limited)
- **Golden suite at Phase 0**: 737 supported_pass / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode; 6180 xfail_expected, 33900 invalid_skipped (per `verifier_report.json`, BH p150b)
- **Issues encountered** (verifier fixes):
  - L1 budget overshoot: CBs were sized against `get_max_worker_l1_unreserved_size()` (kernel-config base), ~69 KB above the allocator's real CB region on BH → (1,1,544,11008) bf16 threw "circular buffers grow to 1594496 B ... beyond max L1 size". Now `cb_l1_budget(device)` (allocator free region below the lowest L1 buffer; also accounts for L1-resident tensors). Regression test added.
  - DRY: CB capacities restated in `footprint()` and the CB declarations → single `cb_plan()`; `out_unit_tiles` from one `_out_unit()`; writer RT-arg layout base passed as a CT arg instead of a duplicated literal.
- **Tests added**: test_rms_norm.py (acceptance, planner), test_rms_norm_debug.py (implementer; + `test_rms_norm_debug_l1_budget_boundary` by verifier), test_rms_norm_precision_baseline.py, perf_scoreboard_rms_norm.py (verifier perf tool, not collected by directory runs)

## Refinement 1 — Numerical configurability: bfloat8_b + fp32_dest_acc_en=False
- **Date**: 2026-11-01
- **What was done**:
  - `SUPPORTED`: `dtype` += `bfloat8_b`, `gamma_dtype` += `bfloat8_b`, `fp32_dest_acc_en` += `False`. `EXCLUSIONS` += the bfloat8_b cells mirrored from feature_spec INVALID (`bfloat8_b × ROW_MAJOR`, `bfloat8_b × {w,h}_non_aligned`, gamma `bfloat8_b × ROW_MAJOR`) so translated tests get a native refusal; `{float32, False}` stays excluded.
  - Intermediate CB formats follow the DEST width: `acc_dtype(fp32_dest_acc_en)` (Float32 / Float16_b) drives `cb_sumsq_tiles`, `cb_partial_sumsq`, `cb_gathered_sumsq`, `cb_inv_rms` and `cb_plan()`'s page size (`T32` → `T_acc`). Under 16-bit DEST these CBs, the L1 footprint solve and the k > 1 partial all-gather NoC payload (writer reads `get_tile_size(cb_partial_sumsq)`) are all halved. The blocking, grid fill and batched reader/writer are unchanged: the `False` path runs the same Phase 0 schedule.
  - bfloat8_b TILE gamma: the reader's row-0 extraction is now format-aware. New reader CT arg `gamma_exp_bytes` (64 B for bfloat8_b, 0 otherwise; derived on the host as `tile_size − 1024·datum_bytes`). The first 64 B-aligned read now also covers the exponent section. `expand_gamma_exponents()` then rewrites it so every face row carries row 0's shared exponent (16 word stores per tile), and the datum-row doubling and face 0/1 → 2/3 copies run on the area after the exponent section.
  - Host: `_datum_bytes()` replaces `Tensor.element_size()`, which throws for block-float dtypes.
  - Compute kernel: unchanged (helpers handle data-format reconfig).
  - Reused: whole Phase 0 path (blocking, CB plan, all three kernels). Added: one CT arg, one 16-word exponent re-pattern, `acc_dtype()` / `_datum_bytes()` / `_bfp_exp_section_bytes()` helpers.
- **Accuracy achieved** (randn x and gamma; reference computed from the device-quantized inputs; shapes (1,1,64,256), (1,1,32,7168), (1,1,640,6144), (1,1,8192,1024)):
  - bf16, fp32_dest=False, HiFi2 (the perf-profile config): PCC ≥ 0.999983, max_abs ≤ 0.19, rel_rms ≤ 0.0064.
  - bf16, fp32_dest=False, HiFi4: PCC ≥ 0.999993, rel_rms ≤ 0.010.
  - bfloat8_b, fp32_dest=True: PCC ≥ 0.999953, rel_rms ≤ 0.0098.
  - bfloat8_b, fp32_dest=False: PCC ≥ 0.999950, rel_rms ≤ 0.013.
  - No wide-W bf16-DEST precision near-miss, so no Refinement 1b is needed.
- **Golden test progress** (targeted slice, 968 cells: every INTERLEAVED `test_op` cell of 1x1x32x64 / 1x1x32x8192 (W-split) / 2x1x100x47 (non-aligned), plus all 104 INTERLEAVED `test_op_loose` cells with fp32_dest_acc_en=False, i.e. the LLM perf profiles and resilience cases): 236 passed, 36 xfailed (exactly the `{float32, False}` EXCLUSION cells), 696 skipped (INVALID), 0 failed.
- **Issues encountered**: `Tensor.element_size()` raises for bfloat8_b. A `static_assert` inside an `if constexpr` of a non-template function fired on non-bfp gamma; it is hoisted and made conditional.
- **Tests added**: `test_rms_norm_numeric_config.py` (41 cases): 16-bit DEST × {bf16, bf8b} × {no gamma, bf16 gamma, bf8b TILE gamma} across the R1 / R2-occupancy / R2-fit regimes; bf8b with fp32 DEST and mixed gamma dtypes; the perf-profile config (bf16, False, HiFi2, TILE gamma, PCC ≥ 0.9995) incl. the multi-block depth-2 shape; and the bf8b non-aligned refusal. Phase 0 acceptance + debug + precision-baseline: 145/145 green.

## Refinement 2 — WIDTH_SHARDED + BLOCK_SHARDED: shard-resident blocks + a cross-core combine that scales with the shard grid
- **Date**: 2026-11-01
- **What was done**:
  - `SUPPORTED["memory_layout"]` += `WIDTH_SHARDED`, `BLOCK_SHARDED` (input and output share the shard spec; TILE and ROW_MAJOR; all alignments). `validate()` accepts a sharded output only if it equals the input's memory config, and refuses DRAM-sharded inputs. `EXCLUSIONS` += ROW_MAJOR × {WIDTH, BLOCK} × TILE gamma, mirroring feature_spec INVALID: an RM shard's W slice starts at a 16 B granule, not a tile edge.
  - **Work split from the shard spec** (`select_sharded_blocking()`): one group per shard row (BLOCK) or one group (WIDTH). k = shard columns, up to 128 on the BH 13×10 grid. Member → core comes from the shard grid in its orientation, so WIDTH groups span grid rows. Ragged last shard row/column (`Split.PAD`) owns only its valid tile-rows / W tiles. Block = the resident shard, sub-blocked along rows only when the CB plan does not fit the L1 left by the two shards. The b-solve is shared with the interleaved path (`_solve_rows()`).
  - **Zero-copy TILE**: `cb_x` and `cb_out` are backed on the input and output shards (`ttnn.cb_descriptor_from_sharded_tensor`). The reader only publishes the shard's tile-rows. Compute packs straight into the output shard, and the writer only waits/pops per block. No accessor touches a local shard (empty `TensorAccessorArgs` on the sharded path).
  - **W tail without mutating the caller's shard**: the tail member's reader emits a 0/1 row mask (`prepare_reduce_mask`, `cb_tail_mask`). Compute collapses x² of the full W tiles into the reduce `Accumulate` CB (`cb_sumsq_acc`), then folds the tail tile's x² in with `ReducePartialScaler::only_partial()` and finalizes on that last chunk. Covers the poisoned-padding cases.
  - **ROW_MAJOR stick shards** (16 B-granular widths, any height): per tile-row, the valid stick pieces are copied locally (own-L1 NoC reads) into the existing `cb_x_rm` tilize staging and zero-filled to the tile edge. The writer copies the untilized pieces back into the local output shard. Nothing is gathered across Tensix cores.
  - **Scalable combine (design R3)**: a per-row owner topology. Tile-row r goes to member r % k, which gathers the k partials, combines and finalizes them (`cb_inv_owned`), and returns inv_rms into every member's `cb_inv_rms` (second semaphore). Every member pushes `ceil(rows/k)·k` gathered pages per block (lockstep slots). L1 per member ≈ b + k pages instead of the symmetric 2·k·b. Owned inv tiles are returned one at a time from the CB front, because members own different counts per block. Partial and inv payloads ship faces 0/2 only (`PARTIAL_FACES_ONLY`, interleaved R2 too). The symmetric all-gather needs a 2-block `cb_gathered` only when a core runs more than one block.
  - **Topology knob, measured**: `K_SYM_MAX_INTERLEAVED = 8` (catalog; unchanged), `K_SYM_MAX_SHARDED = 1` (owner topology for every sharded k > 1). BH p150b, bf16, HiFi2, 16-bit DEST:
    - BLOCK (7168,1024) [736,128] 8×10: 70 820 → 32 258 ns
    - BLOCK (8192,1024) [1024,128] 8×8: 96 694 → 42 749 ns
    - BLOCK (640,7168): 30 004 → 29 035 ns
    - 1-tile-row WIDTH cases: flat (≤ 1 %)
  - RM gamma slice load generalized to an element offset that is not 64 B-aligned (aligned-down read + lead skip).
  - Reused: the whole compute pipeline (Σx² chain, collapse/combine reduce, fused normalize chain, tilize/untilize), the RM chunking, the gamma loader, the interleaved b-solve and the R2 writer exchange. Added: the sharded blocking, 3 CBs (`cb_inv_owned`, `cb_sumsq_acc`, `cb_tail_mask`), the owner exchange in the writer, the local shard copy (RM) and publish (TILE) branches in the reader/writer, and the masked-tail collapse.
- **Accuracy achieved**: every case meets its gate. bf16 PCC ≥ 0.999 (HiFi4) and ≥ 0.9995 (perf-profile config: HiFi2, 16-bit DEST) on the 9 pinned perf geometries. fp32 PCC ≥ 0.999. bfloat8_b PCC ≥ 0.99 against the device-quantized input. The input shard is bit-identical after every call (asserted).
- **Device time (sharded perf profiles, BH p150b, ns, vs feature_spec `achievable_ns` at 1350 MHz)**:
  - WIDTH (32,1024) 8 Tensix cores: 5 733 vs 4 110
  - WIDTH (32,2304) 9 Tensix cores: 7 533 vs 4 617
  - WIDTH (32,5120) 32 Tensix cores: 8 663 vs 5 267
  - WIDTH (32,7168) 28 Tensix cores: 9 621 vs 5 481
  - BLOCK (640,6144) 90 Tensix cores: 22 031 vs 19 039
  - BLOCK (640,7168) 70 Tensix cores: 29 035 vs 21 669
  - BLOCK (7168,1024) 80 Tensix cores: 32 258 vs 18 871
  - BLOCK (8192,1024) 64 Tensix cores: 42 749 vs 25 640
  - BLOCK (7168,2048) 11×10: produced no profiler row in this run, so not measured.
  - Finding: on tall shards (few W tiles, many tile-rows) per-tile-row SFPU work dominates — the within-tile collapse and a full-tile rsqrt finalize. The next levers are the design's "Finalize SFPU scope" lamp (rsqrt on column 0 only, catalog `sfpu_tile_scope`), then the catalog `tensix_all_reduce` per-row-vector payload (row partial + multicast return for k ≥ 12) on the decode WIDTH cases.
- **Golden test progress** (targeted slices, BH p150b):
  - Every WIDTH/BLOCK `test_op_loose` cell (pinned perf geometries, resilience, padding-poison): 192 passed, 3 skipped `INFEASIBLE_L1` (harness-side: the input or output shard does not fit L1; uncharged).
  - WIDTH/BLOCK `test_op` cells of 7 shapes (1x1x32x8192, 2x1x100x47, 4x8x47x256, 1x1x2048x256, 32x17, 4x1x512x512, 1x1x50x128) together with the loose cells: 642 passed, 126 xfailed (EXCLUSION cells), 0 failed.
  - INTERLEAVED representatives (shared code changed): 94 passed, 24 xfailed, 0 failed.
- **Issues encountered**:
  - In the owner topology, members own different row counts per block, so `cb_inv_owned`'s read pointer is not block-aligned. A multi-tile linear walk from `get_read_ptr` returned wrong inv_rms on multi-block shards ((7136,736) WIDTH, (1,1,4064,160) RM WIDTH). Fixed by returning owned tiles one at a time from the CB front.
  - `ReduceDim` needs `ckernel::` in a dataflow kernel.
- **Tests added**: `test_rms_norm_sharded.py` (54 cases):
  - WIDTH TILE at k = 1 / 2 / 64 / 128 (owner, groups spanning grid rows) and BLOCK TILE
  - non-aligned W/H, and poisoned padding on both schemes
  - ROW_MAJOR stick shards, fp32 / no gamma, bfloat8_b with RM gamma
  - the 8 pinned perf geometries at the perf-profile config
  - owner-combine multi-block regressions
  - every case asserts the caller's input shard is unchanged

  Prior-phase acceptance + debug + numeric-config + precision-baseline: 186/186 green.

## Refinement 3 — Speed up the large interleaved cells: fill the grid at mid tile-row counts, overlap the one-block prefill
- **Date**: 2026-11-01
- **What was done** (perf; no SUPPORTED change; BH p150b 13×10, perf config = bf16, `fp32_dest_acc_en=False`, HiFi2, TILE x, TILE gamma; device-kernel ns, median of 3–5):
  - **(a) Occupancy split, `_choose_k`.** The `8·tensor_row_tiles ≤ G` predicate is replaced by a per-core cost model: the k ≥ k_fit that minimizes busiest-group rows × (slice tiles + `COMBINE_TILE_COST` = 3). Same floors as before: ≥ 4 tiles per member, RM pieces ≥ 512 B. For tile-rows ≤ 10 the choice is unchanged. It also fixes the 11–16 tile-row band, where k = 8 left only 10 groups for up to 16 rows (2 rows per core).
    - (640,6144) k 1 → 6, 20 → 120 Tensix cores
    - (640,7168) k 1 → 6
    - (1024,1024) k 1 → 3, 32 → 96 Tensix cores
    - k alone: 99 683 → 56 847, 116 090 → 64 564, 22 710 → 16 007
    - ROW_MAJOR (640,6144): 87 433 → 47 846
  - **Ablation of (b) found that the gamma load, not overlap, was binding.** On (8192,7168), stubbing x read, compute and write still left 131 µs. The reader loaded gamma *before its first x read*: every Tensix core issued 2 × 64 B DRAM reads per gamma tile (224 tiles × 130 Tensix cores, transaction-bound), then did 9 local NoC copies per tile for the row expansion. Compute waited for gamma at kernel start.
    - Bank-staggering that walk: no change (131 → 129 µs).
    - One 576 B read per tile: reads 94 → 59 µs, but then 16.8 MB of redundant DRAM bytes.
  - **Gamma: moved to the writer and multicast once per W slice** (`GAMMA_MCAST`, kernel-lib `ttnn.Mcast` / `McastArgs` helper, one-shot, NoC1):
    - The writer (BRISC) is idle until the first output tiles, so every reader starts its x stream at once. Compute waits for `cb_gamma` before its first pass 2.
    - `_gamma_mcast_groups`: k = 1 is one group of all Tensix cores (one DRAM copy of gamma). k > 1 is one group per grid column, since column x holds only slice x % k; this needs the row-groups to fill whole grid rows.
    - Otherwise (k > 1 with a partial last grid row) it falls back to a per-core load, still on the writer.
    - A member-major core layout (k groups) was measured and rejected: its longer all-gather cost as much as the multicast saved (1024×1024: 18.2 µs without the multicast vs 17.4 µs for the adjacent layout).
    - Sharded paths load their gamma slice per core on the writer.
  - **Overlap knob `OVERLAP_BLOCKS`** (design lamp "Overlap (one-block assignment)"). `_solve_rows` splits a fitting assignment into ceil(rows/N)-row depth-2 blocks. Measured at N = 2:
    - before the gamma fix: flat (771 → 772, 90.8 → 93.0, 235.8 → 230.3 µs)
    - after it: worse (630 → 673, 76 → 81, 192 → 204 µs)
    - The knob stays live, parked at 1. These shapes are DRAM-write-bound (see below), and mixing block-1 reads with block-0 writes does not help.
  - **Writer unit size (`OUT_UNIT_CAP`) and depth (`OUT_DEPTH`)**: 8/16/32 tiles × depth 2/4 measured on write-only ablations — flat (7168,1024 57.9–58.9 µs; 8192,2304 142–143 µs). Left at 16 × 2.
  - **Also measured and not kept:**
    - per-core bank rotation of the x read / out write walk: read-only 313 → 301 µs, full op flat
    - swapping the reader / writer NoCs: writes on NoC0 much slower (full 629 → 820 µs)
  - Reused: the whole compute pipeline, the R2 all-gather, the CB plan and the gamma loader. The gamma loader moved verbatim from the reader to the writer, and the reader now owns only cb_x. Added: `_choose_k` / `_occ_cost`, `_gamma_mcast_groups` + one `ttnn.Mcast` attach, the overlap branch in `_solve_rows`, and the compute gamma wait moved into `apply_back`.
- **Device time, before → after (ns)**:
  - (a) (640,6144): 99 683 → 45 884 (2.17×)
  - (a) (640,7168): 116 090 → 52 039 (2.23×)
  - (a) (1024,1024): 22 710 → 15 226 (1.49×)
  - (b) (8192,7168): 771 298 → 628 756 (1.23×)
  - (b) (7168,1024): 90 816 → 77 611 (1.17×)
  - (b) (8192,2304): 235 847 → 194 893 (1.21×)
  - (b) (8192,1024): 103 004 → 90 299 (1.14×)
  - Guard set, interleaved, HEAD → now:
    - TILE (2048,256): 11 273 → 9 363
    - TILE (4,1,512,512): 18 576 → 16 721
    - TILE (32,1024): 6 755 → 5 687
    - TILE (32,7168): 17 260 → 14 501
    - TILE (32,16384): 32 916 → 27 616
    - TILE (352,4096): 27 023 → 25 605
    - RM (2048,256): 13 299 → 10 270
    - RM (512,512): 20 103 → 16 250
    - RM (32,1024): 10 025 → 7 704
    - RM (32,16384): 36 944 → 30 321
    - RM (352,4096): 30 059 → 26 116
  - Sharded pinned geometries, R2 → now (ns):
    - WIDTH (32,1024): 5 733 → 5 189
    - WIDTH (32,2304): 7 533 → 6 865
    - WIDTH (32,5120): 8 663 → 8 133
    - WIDTH (32,7168): 9 621 → 9 456
    - BLOCK (640,6144): 22 031 → 19 916
    - BLOCK (640,7168): 29 035 → 26 352
    - BLOCK (7168,1024): 32 258 → 30 061
    - BLOCK (8192,1024): 42 749 → 40 727
  - No measured regression.
- **Bottleneck now (finding)**:
  - **k = 1 prefill (class b): DRAM-write-bound.** At (8192,7168), stubbing compute leaves the time unchanged (635 vs 629 µs).
    - read-only: 313 µs (374 GB/s)
    - write-only: 438 µs (268 GB/s, insensitive to unit size / depth / bank order)
    - full: 629 µs (373 GB/s combined, vs ~559 µs at 420 GB/s)
    - Next lever: raise write throughput by splitting output writes across both NoCs. The reader is idle once its block is read, but that needs a second output CB (cb_out has one consumer) and a compute pack split. It is a scheme change beyond this refinement.
  - **Mid tile-row (class a):** one tile-row per Tensix core, so read → pass 1 → combine → pass 2 + write serialize inside each core. 45.9 µs vs ≈ 37 µs DRAM ceiling.
    - Next lever: stream pass 1 per W chunk as tiles land: per-chunk chain calls plus a (rows × chunks) collapse. The chain's Strided addressing needs caller-managed waits.
- **Accuracy achieved**: unchanged numerics (same FP order). Perf config: PCC ≥ 0.999983 on all 13 measured shapes. New test: PCC ≥ 0.9995 at the perf config (bf16 gamma), ≥ 0.999 with fp32 / bf8b gamma.
- **Golden test progress** (targeted slice: every INTERLEAVED `test_op_loose` cell — the perf profiles and resilience cases — plus INTERLEAVED gamma `test_op` cells of 4x1x512x512, 2x1x100x47, 1x1x32x8192, 1x1x2048x256): 257 passed, 40 xfailed (EXCLUSION cells), 386 skipped (INVALID), 0 failed.
- **Issues encountered**:
  - `ttnn.Mcast` cuts receivers into equal consecutive groups, which drove the column grouping and the per-core fallback.
  - A W slice held by a single Tensix core has no receivers: the plain `receiver()` fails a static_assert. Fixed with `optional_sender` / `optional_receiver`.
- **Tests added**: `test_rms_norm_gamma_share.py` (19 cases). It covers every `_gamma_mcast_groups` regime: k = 1 with a partial grid row, k > 1 column groups, k > 1 per-core fallback, and single-core slices. Each runs across TILE / ROW_MAJOR input, TILE / RM gamma, bf16 / fp32 / bf8b gamma, and a non-aligned W / H tail. It asserts the regime chosen on the 13×10 grid. `perf_r3_rms_norm.py` is the profiling probe (no `test_` prefix, not collected). Unit directory: 259/259 green.

## Refinement 4 — HEIGHT_SHARDED: consume the resident row shard in place
- Date: 2026-11-01
- What was done:
  - **HEIGHT as the k = 1 case of Refinement 2's shard-resident path.** `select_sharded_blocking()` maps shard row g to the g-th Tensix core of the shard grid in its orientation. That covers multi-range grids (full grid rows + a partial last row from `auto_shard_config`) and the ragged last shard. k = 1, so there are no combine CBs or semaphores.
    - TILE: `cb_x` / `cb_out` are backed on the input / output shards (zero-copy, never re-read through a TensorAccessor). The W tail is masked in compute, so the caller's shard is never written.
    - RM: resident sticks are copied locally into the tilize staging; the untilized pieces are copied back into the local output shard.
    - `is_sharded_wb()` became `is_sharded()`. `SUPPORTED["memory_layout"]` gains `HEIGHT_SHARDED`.
    - Reused: shard-backed CBs, reader/writer/compute sharded branches (unchanged), owner-combine-free k = 1 compute path. Added: the HEIGHT core walk (≈ 12 lines).
  - **Gamma multicast for sharded k = 1** (`GAMMA_MCAST_SHARDED_K1`, default on). Every Tensix core of a HEIGHT program needs the whole gamma, so Refinement 3's one-group multicast applies: one DRAM read per launch instead of one per Tensix core. Measured BH p150b, perf config (bf16, `fp32_dest_acc_en=False`, HiFi2), device kernel ns, per-core load → multicast:
    - TILE (4096,4096): 92 633 → 54 770
    - TILE (8192,1024): 28 956 → 19 993
    - TILE (7136,736): 20 847 → 14 793
    - RM (4096,4096): 106 812 → 79 687
    - RM (8192,1024): 32 941 → 27 225
    - RM (7136,736): 24 182 → 19 679
    - (99991,64) TILE / RM: flat (gamma is 2 tiles)
  - **Wide-W L1 fallbacks (sharded only, preference order in `_sharded_plan_candidates`).** HEIGHT holds whole rows beside two resident shards, so a W split cannot relieve L1. A no-dispatch probe over the 843 golden HEIGHT configs (shape × layout × dtype × gamma) found 54 whose one-tile-row plan did not fit.
    - (a) Default: resident gamma, RM chunk cap 32. Every shape that fit before keeps this path.
    - (b) TILE: gamma streamed through a 2-chunk ring (`cb_gamma` = 2·G, the largest G ≤ 16 that fits). The writer re-streams it per block; compute normalizes in G-tile windows (`normalize_window` gained a gamma-offset argument).
    - (c) RM: halve the chunk cap C (the staging / `cb_out` / untilize CBs scale with C). At each C, try resident gamma first, then gamma streamed with G = C. The writer prefetches `GAMMA_STREAM_DEPTH` chunks and refills chunk c + depth after draining chunk c's sticks.
    - Result: 54 → 0 configs that do not fit.
    - 18 configs remain unbuildable by construction: fp32 TILE at W = 8192 needs a 1 MB input shard plus a 1 MB output shard of the same spec, which exceeds L1. The golden harness skips them.
    - Knobs: `GAMMA_STREAM_CHUNK_CAP`, `GAMMA_STREAM_DEPTH`, `GAMMA_STREAM_FORCE_CHUNK` (test hook), `PlanOpts.gamma_chunk`, `PlanOpts.rm_chunk_cap`.
- Accuracy achieved: PCC ≥ 0.999 (rtol / atol not gated; PCC-only tests) on every HEIGHT unit case:
  - TILE / RM; aligned, W- and H-non-aligned, poisoned padding
  - fp32 / bf16 / bf8b; with and without gamma
  - shapes (1,1,256,512), (2,4,128,512), (1,1,8192,1024), (7136,736), (99991,64), (100,5120), (1,1,32,8192), (1,1,160,11008)
  - Perf config: PCC 0.999983–0.999985.
- Golden test progress (targeted slices):
  - every HEIGHT `test_op_loose` cell: 92/94 on the first pass. The 2 failures were the RM (·,11008) L1 class; they are green after fallback (c).
  - HEIGHT cartesian + loose for the widest shapes (1x1x32x8192, 1x1x160x11008, 1x224x11008, *32x4096, 128x8192): 187 passed, 45 xfailed (EXCLUSIONS), 0 failed.
- Issues encountered:
  - The resident gamma / RM tilized row did not fit beside two resident wide shards (fixed by the fallbacks above).
  - A deadlock in the RM streamed-gamma design was avoided by construction: pushing a whole block's gamma chunks before draining `cb_out_rm` blocks both kernels once a block has ≥ 3 tile-rows and > 2 chunks. It is pinned by a test with 8 tile-rows × 4 chunks per shard.
- Tests added (`test_rms_norm_sharded.py`):
  - HEIGHT added to every WIDTH/BLOCK parametrization (non-aligned, poisoned padding, RM, fp32 / no gamma, bf8b + RM gamma).
  - `test_height` (7 shapes × TILE/RM, perf config), `test_height_rm_tile_gamma`.
  - `test_height_streamed_gamma` (wide W, TILE/RM).
  - `test_forced_streamed_gamma` (TILE: multi-block, W tail, owner combine, ragged column gap tiles).
  - `test_forced_streamed_gamma_rm` (RM: G = C, 8 tile-rows × 4 chunks, H tail, k > 1).
  - `perf_r4_rms_norm.py`: profiling probe (no `test_` prefix).
  - Unit directory: 307 passed, 3 skipped.

## Refinement 5 — Speed up the few-row wide-W interleaved cells: widen the W split past 8 Tensix cores
- Date: 2026-11-01
- What was done:
  - **W split past 8 Tensix cores (knob-turn).** Interleaved `K_CAP` is now `min(K_CAP_MAX, grid cores)` instead of `min(8, grid_x)`.
    - `K_CAP_MAX = (341 − 13) / 2 = 164`: the writer carries one (x, y) NoC pair per member as runtime args, and a kernel gets at most 341 runtime args.
    - k ≤ 8 keeps the symmetric all-gather. k > 8 uses Refinement 2's owner combine, which is already built for sharded.
    - k ≤ grid_x keeps the one-grid-row group placement, so the gamma multicast column groups are unchanged. k > grid_x packs groups row-major across grid rows (`_max_groups`, `core_of`), and those groups load gamma per Tensix core (`_gamma_mcast_groups` → None).
    - Cost model: `_occ_cost` adds `ceil(rows/k)·k·OWNER_MEMBER_COST` for k > 8. The constant is 0.4 x-tile-equivalents, fitted on a forced-k sweep: the owner pays ≈ 0.15 µs per member versus ≈ 0.41 µs per slice tile.
    - `_choose_k` now rejects a candidate k whose owner gathered CB does not fit L1.
  - **Gamma load finished after the exchange** (`GAMMA_AFTER_SEND`, gated to blocks of ≤ `GAMMA_AFTER_SEND_MAX_ROWS = 8` tile-rows, per-core gamma load only).
    - Ablation, (160,11008) at k = 16–25 without gamma: 21.4 µs flat. With gamma: 29.4 µs. The writer loaded gamma before it shipped its partial, and those DRAM reads queue behind the readers' x stream, so the whole combine waited on the slowest gamma load.
    - The writer now issues the gamma DRAM reads at kernel start (`GammaSliceLoad::begin`). It finishes them (barrier + expand, `end`) only after block 0's partials are out and its owner duties (combine wait, inv_rms return) are done.
    - The exchange is split into `exchange_send` / `exchange_recv_partials` / `exchange_recv_inv`.
    - Not applied to multicast gamma, which measured worse: a late sender stalls its whole slice group (k = 13 (160,11008) 25.7 → 34.8 µs).
    - Not applied to long blocks, which measured worse: tall BLOCK (7168,1024), 23 tile-rows, 29.9 → 30.8 µs, because the early load was already hidden behind pass 1.
  - **Signal fence** (`SIGNAL_AFTER_FLUSH`, catalog `tensix_all_reduce`, design perf lamp "Signal fence").
    - Each semaphore increment that publishes partial or inv_rms writes now follows `noc_async_writes_flushed()` instead of a write barrier. The writes and `noc_semaphore_inc` share NoC, VC (`NOC_UNICAST_WRITE_VC`) and destination, so delivery is in order. The kernel-exit barrier retires the acks.
    - Result: 2–4% on every k > 1 shape.
  - Reused: owner combine (compute + writer), Mcast plumbing, CB plan, all compute code (unchanged).
  - Added: `_max_groups`, a row-spanning `core_of`, `OWNER_MEMBER_COST`, `GammaSliceLoad` (begin/end split of the gamma loader), the exchange split, `fence_before_signal`, and 2 writer CT args (20 `gamma_after_send`, 21 `signal_after_flush`).
- Device kernel time (BH p150b 13×10, perf config: bf16, `fp32_dest_acc_en=False`, HiFi2, TILE gamma), HEAD → now (ns):
  - TILE:
    - (1,1,32,32768): 50 598 → 20 876 (2.42×, k 8 → 49)
    - (1,1,32,16384): 27 710 → 14 585 (1.90×, k 32)
    - (1,1,32,7168): 14 497 → 9 962 (1.46×, k 25)
    - (1,1,32,5120): 11 531 → 9 083 (1.27×, k 20)
    - (1,1,96,6144): 16 491 → 13 308 (1.24×, k 24, 72 Tensix cores)
    - (1,1,160,11008): 28 982 → 24 270 (1.19×, k 25, 125 Tensix cores)
  - ROW_MAJOR:
    - (32,32768): 54 048 → 23 513 (2.30×)
    - (32,16384): 30 144 → 17 039 (1.77×)
    - (32,7168): 16 010 → 12 570 (1.27×)
    - (96,6144): 18 370 → 14 884 (1.23×)
    - (160,11008): 30 066 → 25 216 (1.19×)
    - (32,5120): 12 905 → 11 640 (1.11×)
  - Guard set, unchanged k:
    - TILE (640,6144): 45 884 → 45 550
    - TILE (1024,1024): 15 226 → 15 101
    - TILE (352,4096) (k 6, per-core gamma): 25 918 → 23 543
    - TILE (32,1024): 5 816 → 5 536
    - RM (640,6144): 47 597 → 48 219 (+1.3%, noise band)
    - RM (1024,1024): 16 004 → 16 036
    - RM (352,4096): 26 160 → 24 790
  - Sharded pinned geometries (same writer path):
    - WIDTH (32,1024) 5 176 → 5 124 · (32,2304) 6 865 → 6 741 · (32,5120) 8 127 → 7 764 · (32,7168) 9 253 → 8 342
    - BLOCK (640,6144) 19 815 → 19 059 · (640,7168) 26 344 → 24 604 · (7168,1024) 30 068 → 29 924 · (8192,1024) 40 727 → 40 517
  - The one slower cell is RM (1,1,32,1024), a 32-tile cell below the >32-tile perf bar: 7 720 → ~7 900. The late gamma costs ~0.3 µs there (k = 4, per-core gamma, pass 1 includes tilize). Knob off: 7 584.
- Bottleneck now (finding):
  - (32,32768) at k = 49: 19.0 µs without gamma vs 20.9 µs with it. A forced-k sweep is flat from k = 32 to k = 64 (21.4–23.0 µs). Time ≈ 3.5 + 0.41·slice + 0.15·k µs, so at the optimum about a third is the owner's per-member combine: k gathered tile-adds plus 2k face writes of the inv_rms return, serial on one Tensix core.
  - Next levers, not attempted (each is a combine-topology rewrite, not this knob-turn):
    - multicast return of inv_rms. It needs rectangular groups (k a multiple of grid_x) or idle-core handling inside the multicast rectangle.
    - row-vector partials (32 datums per member, ⌈k/32⌉ tiles to column-reduce on the owner), catalog `tensix_all_reduce`.
    - gamma without the per-core expand: a Row-broadcast gamma multiply in compute. It would remove the remaining 1.6–3.4 µs gamma cost on these shapes.
- Accuracy achieved: PCC ≥ 0.999984 on every measured perf-config shape (rtol / atol not gated; PCC-only checks). New unit tests: PCC ≥ 0.9995 (perf config), ≥ 0.999 (fp32 / bf8b gamma), ≥ 0.9999 (fp32 input + fp32 DEST).
- Golden test progress (targeted slices): every INTERLEAVED `test_op_loose` cell 107/107 passed (perf profiles + resilience, incl. 32×{5120, 7168, 16384, 32768}, 96×6144, 160×11008, 1×224×11008); INTERLEAVED bf16 cartesian of 1x1x32x8192 and 2x1x64x4096: 48 passed, 144 skipped (INVALID), 0 failed.
- Issues encountered:
  - Late gamma before the symmetric gather push delayed the local combine (RM 32×1024 9.1 µs). Fixed by finishing gamma after the gathered push on both topologies.
  - Tracy's `-k` passthrough breaks on parentheses (the probe selects cells by env var instead).
- Tests added:
  - `test_rms_norm_wide_split.py` (18 cases):
    - cost-model choice k > 8 (k > grid_x on 13×10), TILE / RM
    - forced k = 12 / 20 / 40 on W- and H-non-aligned, TILE / RM
    - gamma formats bf16-RM / fp32 / bf8b; no gamma; fp32 input + fp32 DEST
    - multi-block owner exchange (`OVERLAP_BLOCKS = 2`)
    - both new knobs off
  - `test_rms_norm_gamma_share.py`: (96,4096) re-pinned to the new spanning per-core regime, and (96,1024) pins the column regime.
  - `perf_r5_rms_norm.py`: profiling probe with forced-k / no-gamma / sharded-guard env hooks (no `test_` prefix).
  - Unit directory: 329 passed, 3 skipped.

## Perf 1 — perf tournament round 1 (fan-out; nothing added to SUPPORTED)
- Date: 2026-11-01. Board: Blackhole p150b, 13×10 = 130 Tensix cores, AICLK 1350 MHz. Timer: the grader's
  real-time profiler (`eval/eval_test_runner.sh`, `device_kernel_ns`). Breakdown runs use Tracy zones. Before → after
  pairs come from one session: the round-start op (worktree of 8216b2fa1e6) vs the final op, built by
  `tests/ttnn/unit_tests/operations/rms_norm/perf_p1_ab_rms_norm.py`. Precision contract untouched (no change to
  `fp32_dest_acc_en` / `math_fidelity` / `math_approx_mode` / dtypes).

### Instrumentation (permanent)
- `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope`; the header named in the perf
  methodology was missing from this checkout, so it was added).
- Block-level zones on every stage, with each wait split from its work:
  - reader: `rd_reserve` / `rd_read` / `rd_block`
  - compute: `cmp_wait_x` / `cmp_pass1` / `cmp_collapse` / `cmp_wait_gathered` / `cmp_combine` / `cmp_wait_inv` /
    `cmp_wait_gamma` / `cmp_pass2` / `cmp_tilize`
  - writer: `wr_gamma_issue` / `wr_gamma_finish` / `wr_wait_partial` / `wr_xchg_send` / `wr_wait_peers` /
    `wr_inv_return` / `wr_wait_inv` / `wr_gamma_stream` / `wr_store`
- Ablation defines (`PERF_ABLATE`: `RMS_ABLATE_EMPTY`, `RMS_ABLATE_DM`, `CKL_ELTWISE_CHAIN_SKIP_COMPUTE`). Probes only;
  the op never sets them.
- Probes:
  - `perf_p1_rms_norm.py`: scoreboard cells built exactly like the golden harness.
  - `perf_p1_ab_rms_norm.py`: round start vs head.
  - `perf_experiments/tools/zones.py`: per-zone max / p50 summary over Tensix cores.

### Measured breakdown (round start)
- Fixed-cost floor (all three kernels return at once, same Tensix cores): 0.18–0.39 µs.
- Cumulative ablation, µs. "chains stubbed" = `CKL_ELTWISE_CHAIN_SKIP_COMPUTE`; "no DM" = x/out NoC payload removed.

  | cell | full | no gamma | + chains stubbed | no DM | DM + chains stubbed |
  |---|---|---|---|---|---|
  | hgt_tile_g | 79.4 | 36.3 | 10.4 | — | — |
  | blk_perf_7168x1024 | 30.3 | 21.4 | 15.9 | — | — |
  | wid_perf_32x7168 | 8.9 | 6.4 | 5.5 | — | — |
  | wid_tile_2048x256 | 54.9 | 46.7 | 39.4 | — | — |
  | int_prefill_7168 | 650 | 633 | — | 40.7 | 18.3 |
  | int_mid_7168 | 52.2 | 50.2 | — | 6.7 | — |

- Ranked bottlenecks:
  1. **Gamma handling.**
     - Expand, stream and load cost 15–55 % of HEIGHT / BLOCK / WIDTH / ROW_MAJOR cells.
     - HEIGHT wide W: `wr_gamma_stream` 75.6 of 79 µs; pass 2 starved.
  2. **Per-tile-row fixed cost on tall k > 1 shards.**
     - SFPU collapse ≈ 220 ns per tile-row.
     - Full-tile finalize (rsqrt ≈ 750 ns per tile, catalog `sfpu_tile_scope`).
     - Exchange and owner combine not overlapped with compute.
  3. **Interleaved DRAM-bound cells.** Equal work per Tensix core, but Tensix-core imbalance max/p50 = 1.47 (rd_read
     15..545 µs; NoC arbitration by grid position).
  4. **ROW_MAJOR narrow shards.** 16 B stick pieces, one local NoC transaction + one zero fill each.
- Close to ceiling (no ideas spent): blk_perf_640x6144, 19.2 µs vs achievable 19.0 µs.

### Earlier-snapshot check
- The R5 golden snapshot showed DRAM-bound interleaved cells 7.5 % (median) to 28 % faster than R3 / R4 and today,
  e.g. (8192,7168): R3 643 µs, R4 638 µs, R5 478 µs.
  - Not reproducible: the same golden node measured 640 µs at round start, with the code R5 left.
  - Recorded as a snapshot environment artifact, not a regression to chase. 16 of the 19 cells still above their R5
    time are faster than R4 now; the rest are within 2–4 % of R4 (see "Guard set").
- test_regression (2,1,128,256) fp32: 5.5 µs in R2–R4, 8.9 µs in the R5 snapshot. At round start it measured 5.5 µs,
  so the R5 value is the same artifact. It is 4.48 µs now.

### Classes targeted and why
1. HEIGHT_SHARDED wide W (largest measured ÷ ceiling: 2.2–3.5×, gamma-bound).
2. Tall WIDTH / BLOCK k > 1: the BLOCK perf profiles sat 1.5–1.6× over `achievable_ns`, WIDTH 2048×256 about 6× over
   the per-Tensix-core ceiling.
3. WIDTH decode: 1.5–1.6× over `achievable_ns`.
4. ROW_MAJOR sharded narrow: about 5–8× over.
5. Interleaved DRAM-bound: 1.14–1.22× over the ~430 GB/s read+write ceiling. These are the largest cells.

### Portfolio and verdicts
Eight `perf-part-optimizer` subagents ran in parallel; two later ports were rebased onto the partly graduated op.
Artifacts: `perf_experiments/<slug>/`; raw profiler dumps over 400 KB are gitignored.

| idea | verdict | before → after, ns | domain / notes |
|---|---|---|---|
| `gamma_rowbcast`: gamma as a ROW-broadcast dest-reuse operand, no expand | **WIN, graduated** | hgt_tile_grm 119 794 → 70 276 · hgt_rm_grm 183 468 → 86 137 · blk_perf_640x6144 19 878 → 15 953 · wid_perf_32x7168 9 518 → 7 964 · int_rm_128x4096_gf32 20 237 → 15 033 | everywhere, bit-identical. Helper-only reorder (A1) changed output by ≤ 1 bf16 ulp and was +7–10 % on pass-2-bound cells (rejected). |
| `gamma_stream_l1`: compact resident gamma row, local chunk assembly, NCRISC split | WIN, superseded | hgt_tile_g 79 293 → 69 790; `gamma_rowbcast` reaches 70 1xx on the same cells (floor 69.7) | its **bug fix** graduated: streamed gamma on a shard-backed output packed multi-row blocks window-major (PCC 0.11 on (1,1,8320,4608) HEIGHT fp32 gamma, 0.50 forced). |
| `fused_rowsum_finalize`: FPU `reduce_tile` row collapse + column-0 sfpi finalize + raw batched pass 1 | **WIN, graduated** | wid_tile_2048x256 55 165 → 43 953 · blk_perf_7168x1024 30 241 → 24 696 · blk_perf_8192x1024 40 973 → 33 406 · blk_tail_224x1000 6 930 → 5 243 | everywhere. Fusing pass 1 with the SFPU collapse alone was only 8–12 %; the FPU collapse is the lever. |
| `owner_raw_partials`: ship uncollapsed Σx², owner collapses once | WIN behind k·T_acc ≤ 32 KB, superseded | wid_tile 54 804 → 50 467 · blk 7168 30 236 → 27 279; k·T_acc > 32 KB +1.5–4 % | saves the per-row SFPU collapse that the FPU collapse already cut to ~55 ns (break-even ~70 ns/row). Not graduated. |
| `combine_pipeline`: sharded owner combine over row sub-blocks | **WIN, graduated (port)** | blk_perf_7168x1024 24 930 → 22 401 · 8192x1024 33 510 → 29 272 · 7168x2048 34 987 → 30 572 · wid_tile_2048x256 43 875 → 31 447 | rule below; found a mask-tail cb_sumsq ring-wrap bug that `fused_rowsum_finalize` removes. |
| `inv_mcast_return`: owner returns inv_rms by one multicast | **WIN for single-owner blocks, graduated** | wid_perf_32x7168 8 940 → 7 381 · wid_perf_32x5120 8 239 → 6 487 · int_decode_32768 1.02–1.05× | multi-owner blocks: measured carve-out. Graduated the kernel_lib `Mcast` version; raw was only 2.5–3.2 % faster (noise band). |
| `dram_noc_balance`: reversed NoC pairing on the 3 DRAM-adjacent worker columns | **WIN, graduated (port), no carve-out** | int_prefill_7168 639 246 → 603 219 · int_prefill_1024 89 168 → 82 828 · int_mid_7168 50 609 → 45 344 · int_decode_32768 20 016 → 18 228 · (1024,4096) bf8b gamma 46 578 → 40 774 | see "Carve-outs and measured trades". |
| `rm_narrow_staging`: ROW_MAJOR sharded staging | **WIN, graduated** | wid_rm_2048x256 246 401 → 153 021 · blk_rm_2048x256 25 359 → 20 848 · wid_rm_1504x544 197 687 → 130 290 | everywhere, bit-identical. The assigned RISC-V copy LOST at every width ≥ 16 B. The win is: zero each staging slot's tail once, one transfer per tile-row when pitches match, stateful NoC per stick otherwise. |

### What graduated (commits), domain, deleted code
- **aaa60521c87, row-broadcast gamma.** The single path everywhere.
  - Deleted: `expand_gamma_rows` / `expand_gamma_exponents`.
  - `cb_gamma` pages are faces 0/1 only.
  - Also contains the streamed-window strided-pack correctness fix, with a regression param.
- **e4ca4c0003c, FPU collapse + column-0 finalize + raw batched pass 1.** The single path everywhere.
  - Deleted: the two-call `Accumulate` / `only_partial` W-tail path and `CB_SUMSQ_ACC`.
  - Added: `cb_ones` (the 1.0 scaler).
- **583d00cdad1, ROW_MAJOR sharded staging.** The single path. Each sub-lever falls back to the old per-stick
  behaviour only where its condition cannot hold (pitches differ, more than one chunk).
- **cb6097a66f7, single-owner inv_rms multicast.** Applies to every owner-combine block of one tile-row whose groups
  `ttnn.Mcast` can express (≤ 3 member-only rectangles, member 0 first). Otherwise the unicast return.
- **cc8f0cc2920, combine pipeline.** TILE sharded, owner combine, resident gamma, whole shard one block.
  - n = the most sub-blocks in [2, n_max] such that, with s = ceil(crt/n): s ≥ 2, s·(w+1) ≥ 8, k ≤ 2·s·(w+1).
  - n_max = 3, or max(3, ceil(crt / max(8, k))) for one-tile-wide shards.
  - `RMS_INV_MCAST` stays off on this path: a rotated one-row sub-block is not owned by member 0.
  - New test `test_combine_pipeline` pins it on.
- **aeecdcad80d, reversed NoC pairing.** Every interleaved cell. Sharded is untouched by construction. Every multicast
  sender keeps the usual pairing.
- **9b355328ef7, RM-gamma fix found by the guard-set run.** `gamma_rowbcast` read an RM gamma with one 64 B DRAM read
  per tile on every loader. An RM gamma is ONE DRAM page, so per-core loads (sharded k > 1, interleaved groups without
  multicast) and streamed re-reads hammered one bank.
  - Golden-suite context, without → with the fix: RM HEIGHT fp32 (128,8192) 409 → 113 µs, RM HEIGHT (1,224,11008)
    340 → 122 µs, RM BLOCK (96,6144) 27.3 → 12.2 µs.
  - Isolated runs did not take that path, so the regression was invisible to the subagent's bench.
  - Direct reads now run only on the multicast sender (a slice's sole reader).

### Carve-outs and measured trades
- **inv_rms multicast — multi-owner blocks keep unicast.** Concurrent owner multicasts into the same members
  serialize: BLOCK (7168,1024) 30.3 → 52.8–64.7 µs, WIDTH (2048,256) 55.2 → 91–95 µs. Written as a narrow runtime
  branch (`rows == 1` → multicast), not an allow-list.
- **Combine pipeline gates** (each a measured regression):
  - s = 1: blk_perf_640x6144 +6.7 %.
  - k > 2·s·(w+1): WIDTH (256,2048) k = 64 +18 %.
  - s·(w+1) < 8: (2048,256) BLOCK s = 3 0.96× vs s = 4 0.90×.
- **Reversed NoC pairing — no carve-out.** No portable predicate separates the losers from winning neighbours with the
  same Tensix-core count, k, tiles or bytes per Tensix core, or total bytes. A W sweep at 8192 rows wins 1.06–1.11× at
  every Wt from 8 to 48. Measured costs kept:
  - small k = 2 cells at 16–20 Tensix cores: +4..9 % (0.2–0.6 µs);
  - (128,8192) bf8b k = 26: about +5–7 % against the swap-off variant, still 1.18× faster than round start;
  - (3,3104,544): 3–4 % in paired runs; vs round start 0.958–0.989× over 3 runs, at the noise edge;
  - against about 36 µs won on (8192,7168) alone.
- **RM gamma direct reads restricted to the multicast sender.** Measured regression on the per-core / streamed loads,
  see 9b355328ef7.

### Scoreboard — round start → final, same session (ns, speedup)

| class (bottleneck) | cell | before | after | × |
|---|---|---|---|---|
| interleaved TILE prefill (DRAM) | (8192,7168) perf cfg | 637 197 | 598 522 | 1.07 |
| interleaved TILE prefill (DRAM) | (8192,1024) | 88 141 | 82 554 | 1.07 |
| interleaved TILE mid (DRAM + combine) | (640,7168) k = 6 | 52 313 | 45 569 | 1.15 |
| interleaved decode (owner k = 49) | (32,32768) HiFi4 fp32 DEST | 22 693 | 18 391 | 1.23 |
| interleaved fp32 | (128,8192) fp32 + fp32 gamma | 27 511 | 24 066 | 1.14 |
| interleaved bf8b | (128,8192) bf8b | 13 889 | 11 745 | 1.18 |
| interleaved RM | (1,1,128,4096) fp32 RM gamma | 19 000 | 15 704 | 1.21 |
| interleaved RM owner | (1,1,160,11008) RM | 29 742 | 24 016 | 1.24 |
| interleaved RM tall | (99991,64) RM | 273 900 | 235 384 | 1.16 |
| interleaved non-aligned | (13,777,1023) | 115 927 | 109 737 | 1.06 |
| BLOCK perf (combine + pass 2) | (7168,1024) [736,128] 8×10 | 30 254 | 21 552 | 1.40 |
| BLOCK perf | (8192,1024) [1024,128] 8×8 | 40 934 | 28 342 | 1.44 |
| BLOCK perf | (7168,2048) [736,192] 11×10 | 39 198 | 29 783 | 1.32 |
| BLOCK perf (close to ceiling at start) | (640,6144) [64,704] 9×10 | 19 456 | 14 387 | 1.35 |
| WIDTH decode (combine) | (32,7168) [32,256] 7×4 | 8 745 | 4 725 | 1.85 |
| WIDTH decode | (32,5120) [32,160] 8×4 | 8 172 | 4 469 | 1.83 |
| WIDTH tall k = 8 | (1,1,2048,256) | 54 829 | 29 902 | 1.83 |
| HEIGHT wide W (gamma) | (128,8192) TILE, bf16 TILE gamma | 79 237 | 68 803 | 1.15 |
| HEIGHT wide W | (128,8192) RM gamma / bf8b gamma | 119 651 / 126 074 | 68 831 / 68 909 | 1.74 / 1.83 |
| HEIGHT | (1,1,128,4096) | 57 346 | 35 200 | 1.63 |
| HEIGHT RM | (128,8192) RM + RM gamma | 183 682 | 84 189 | 2.18 |
| HEIGHT RM | (1,224,11008) / fp32 (128,8192) | 165 932 / 155 009 | 122 111 / 113 337 | 1.36 / 1.37 |
| WIDTH RM narrow | (1,1,2048,256) 16 B pieces | 246 566 | 140 602 | 1.75 |
| BLOCK RM | (1,1,2048,256) / (96,6144) / (128,4096) | 24 978 / 16 042 / 12 627 | 17 870 / 12 070 / 9 069 | 1.40 / 1.33 / 1.39 |
| small (≤ 64 tiles) | (2,1,128,256) fp32, (256,256) fp32 | 5 551 / 6 078 | 4 484 / 5 034 | 1.24 / 1.21 |

- vs `achievable_ns`:
  - WIDTH perf profiles now beat their references: (32,7168) 4.7 vs 5.5 µs, (32,5120) 4.5 vs 5.3 µs.
  - BLOCK (640,6144) beats its reference: 14.4 vs 19.0 µs.
  - Remaining over their references:
    - BLOCK (7168,1024): 1.14× (21.6 vs 18.9 µs)
    - BLOCK (8192,1024): 1.11× (28.3 vs 25.6 µs)
    - BLOCK (7168,2048): 1.15× (29.8 vs 25.9 µs)
  - Interleaved perf profiles were already below their references.
- DRAM ceiling (~430 GB/s read + write): (8192,7168) 1.10×, (8192,1024) 1.06×, (640,7168) 1.07×.

### Guard set / golden (all green)
- Golden `test_golden` + `test_regression`: 5444 passed, 0 failed, 0 hangs. Same counts as R5; run in 4 placement
  chunks plus `test_regression`.
- `test_translated`: 104 passed, 2 xfailed.
- Unit directory: 336 passed, 3 skipped. New tests: streamed-window regression param, single-row / mixed owner
  blocks, `test_combine_pipeline`.
- Per axis value vs the R5 snapshot (3227 cells > 32 tiles): median 0.724×.
  - HEIGHT 0.611, WIDTH 0.616, BLOCK 0.712, INTERLEAVED 0.814.
  - Every sharded class: max ≤ 0.987×.
  - The 19 cells above their R5 time are all DRAM-bound INTERLEAVED (the R5 artifact). 16 of them are faster than R4;
    3 are within 2–4 % of R4, at the noise edge — (2047,2047) TILE, (7136,736) RM, (3,3104,544) TILE.
- Direct round-start A/B over 33 cells: every cell faster except (3,3104,544) 0.96–0.99× and (7136,736) RM 0.98×
  (noise band), plus (2047,2047) 0.96–1.03× (noise).

### Next bottlenecks (for Perf 2)
- **BLOCK perf cells: pass 2.** About 14.5 of 21.6 µs on (7168,1024): two FPU ops + pack per tile at ~157 ns/tile.
- **HEIGHT wide W: compute.** Pass 2 is 52 µs on 256 tiles at HiFi4, fp32 DEST.
- **WIDTH (2048,256): the writer exchange.** `wr_xchg_send` 14 µs.
- **Interleaved prefill: DRAM.** 1.10× over the ceiling.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::DestReuseBinary` (+ `mul_reuse_dest_init` / `mul_reuse_dest_tiles`) | capability | No `BroadcastDim` on the dest-reuse multiply: init/exec hard-code `BroadcastType::NONE`. The LLK supports a ROW-broadcast CB operand (`llk_unpack_A<ROW, …, DEST_TO_SRCA>`, `llk_math_eltwise_binary<ELWMUL, ROW, …, DEST_TO_SRCA>`). Without it gamma must be expanded to all 32 rows in L1. Suggested API: a `BroadcastDim` template param on `DestReuseBinary` and the `*_reuse_dest_*` APIs. | 119 794 (hgt_tile_grm, helper with expanded gamma) | 70 276 | `kernels/rms_norm_compute.cpp:234` (`MulGammaRowBcast`) |
| SFPU compute API `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile` (in `reduce()`'s `post_reduce_op`) | capability | Work can be scoped by face (`VectorMode`) or `ITERATIONS`, never to one column parity. Column 0 lives in the 8 even-parity vectors of faces 0/2, so the three calls run three full-tile passes (32 vectors each, three DEST round trips). Suggested API: a column-parity vector mode / fused scalar-mul-add-rsqrt entry. | 30 241 (blk_perf_7168x1024, finalize only) | 28 881 | `kernels/rms_norm_compute.cpp:97` (`finalize_col0_body`) |
| `eltwise_chain` `BinaryFpu<…, DestAccumulation::PerRow>` (pass 1) | capability | Cannot fold the masked W-tail tile into the PerRow DEST accumulation (`x·mask` bcast-row, then DEST-reuse `·x`). It also packs one tile-row per DEST acquire (no multi-row batching). On cores without a tail the chain was equal (≤ 1 %); the raw form keeps one pass-1 path for both. | 6 699 (blk_tail_224x1000, chain + legacy tail) / 44 507 (wid_tile, no tail) | 5 243 / 43 953 | `kernels/rms_norm_compute.cpp:129` (`sum_squares_block`) |
| `dataflow_kernel_lib` local copy (`local_copy_helpers_dataflow.hpp`: `set_read_state` / `read_with_state`) | capability | The stateful read programs source + size ONCE and varies only the destination. The ROW_MAJOR shard staging steps BOTH a source pitch (shard stick) and a destination pitch (staging row) per stick. There is no stateful local-WRITE helper for the write-back twin, so raw `noc_async_{read,write}_one_packet_set_state` / `_with_state` are used. | 163 563 (wid_rm_2048x256, non-stateful per-stick NoC) | 157 073 | `kernels/rms_norm_reader.cpp:159`, `kernels/rms_norm_writer.cpp:717` |

The inv_rms multicast uses the kernel_lib `Mcast` helper (no bypass). The raw multicast variant was within the noise
band of it.

## Perf 2 — perf tournament round 2 (fan-out; nothing added to SUPPORTED)
- Date: 2026-11-02. Board: Blackhole p150b, 13×10 = 130 Tensix cores, AICLK 1350 MHz. Instrument: device profiler
  `DEVICE KERNEL DURATION [ns]` (same device timer as the grader's `device_kernel_ns`). Before → after pairs come from
  one session: the round-start op (worktree of d521ece3aed = ebe9c9d1ebc + probe-only ablation hooks) vs the final op,
  built by `tests/ttnn/unit_tests/operations/rms_norm/perf_p2_ab_rms_norm.py`; cells are built exactly like the golden
  harness (`perf_p2_rms_norm.py`). Precision contract untouched (no change to `fp32_dest_acc_en` / `math_fidelity` /
  `math_approx_mode` / dtypes). No `attention:`-flagged case exists in `LOOSE_CASES`.

### Instrumentation (permanent)
- Every Perf 1 zone kept. `RMS_ABLATE_DM` now also drops the ROW_MAJOR sharded stick staging / write-back payload
  (probe only).
- New kernels carry their own zones: `rms_norm_flat_*` (rd_block, cmp_wait_x, cmp_pass1, cmp_collapse, cmp_wait_gamma,
  cmp_pass2, wr_gamma_issue / wr_gamma_finish, wr_store) and `rms_norm_narrow_flat_*` (reader stages incl.
  rd_wait_peers / rd_inv_return / rd_lead_mcast, cmp_pass1 / cmp_combine / cmp_pass2, writer stages).

### Earlier-snapshot check
- Only INTERLEAVED had cells > 5 % slower than an earlier snapshot: 18 DRAM-bound cells vs `golden_refinement_5`
  ((8192,7168) 593.6 vs 477.7 µs, (8192,5120) 419.5 vs 333.8 µs). Targeted anyway (idea `prefill_dram`).
- Resolved as a measurement artifact, not a code regression: in the R5 snapshot every DRAM-bound interleaved cell is
  0.735–0.79× of its R4 time while every L1-resident (compute-bound) cell is 1.00×. That is the signature of ticks
  counted at a throttled AICLK and converted at 1350 MHz (DRAM time is fixed in real time, compute time is fixed in
  cycles). The `prefill_dram` subagent caught the mirror effect in one of its own runs (every cell 1.2–1.6× slower for
  ~10 minutes, normal again with no code change).

### Measured breakdown (round start)
- Empty-program floor (all kernels return at once, same Tensix cores): 0.29–0.48 µs.
- Cumulative ablation, µs. "chains" = `CKL_ELTWISE_CHAIN_SKIP_COMPUTE`; "no DM" = x/out payload (interleaved TILE) or
  stick staging + write-back payload (RM sharded) removed.

  | cell | full | chains stubbed | no DM | no DM + chains | main zones |
  |---|---|---|---|---|---|
  | int_prefill_7168 | 598.0 | 571.6 | 82.6 | 28.8 | rd_read max 517 / p50 203 (2.5× Tensix-core imbalance) |
  | int_prefill_5120 | 421.5 | 397.7 | 59.4 | 20.6 | |
  | wid_tile_128x4096 (K = 128, b = 4) | 26.6 | 25.7 | — | — | wr_wait_inv 23.2, wr_xchg_send 12.1, owners' cmp_wait_gathered 12.4 |
  | wid_tile_224x3072 (K = 96, b = 7) | 21.5 | 19.4 | — | — | exchange |
  | wid_tile_13x777x1023 (K = 32, b = 325) | 386.8 | 322.2 | — | — | exchange + per-row collapse |
  | wid_rm_5x3x928x544 ([13920, 8] × 68) | 850.2 | 765.0 | 469.9 | 384.6 | rd_block 817 (stick staging), wr_store 337 |
  | wid_rm_13x777x1023 | 677.8 | 616.0 | 398.2 | 336.5 | |
  | wid_rm_1024x1024 | 117.2 | 111.2 | 89.1 | 82.5 | exchange (K = 128) |
  | wid_rm_f32_4x1x512x512 | 227.1 | 215.3 | 177.3 | 161.0 | |
  | hgt_rm_f32_128x4096 (1 stick / core) | 93.8 | 86.9 | — | — | wr_gamma_stream 48.5; 32×-padded tilize / pass / untilize |
  | hgt_rm_224x11008 (2 sticks / core) | 121.9 | 107.8 | — | — | pass 2 + untilize 87, wr_gamma_stream 22 |
  | hgt_tile_res_128x8192 (fp32 RM gamma) | 124.1 | — | — | — | wr_gamma_stream 120.4 (resident-gamma twin: 68.7) |
  | blk_rm_99991x64 ([10000, 8] × 80) | 513.2 | 452.2 | 236.6 | 174.1 | staging + write-back |

- Ranked headroom (measured ÷ ceiling; L1-resident cells use the fastest large cell's ns per tile per Tensix core at
  the same dtype and `Layout`):
  1. WIDTH / BLOCK ROW_MAJOR with 16 B shards (the harness's ROW_MAJOR granule): 15–35× — 16 B stick staging,
     75 %-padding tiles, per-tile-row K-way exchange.
  2. HEIGHT ROW_MAJOR with 1–2 sticks per Tensix core: 13–25× — 32× tile padding, streamed RM gamma.
  3. WIDTH TILE with large K (96–128): 20–49× vs that rule; 4.9× vs the catalog's measured K = 128, b = 4 all-reduce
     (5.4 µs) — the owner combine.
  4. Streamed gamma (HEIGHT wide W): ~1.8× (cmp_pass2 starved on wr_gamma_stream).
  5. Interleaved prefill: 1.04–1.06× of the op's own copy-only run (≈ 405–415 GB/s, the catalog's full-grid copy
     ceiling) → **close to ceiling**: (8192,7168) 600 µs vs copy-only 567–579 µs.

### Classes targeted and why
1. RM WIDTH / BLOCK narrow shards — the largest absolute times of the op (up to 850 µs) and the largest ratios.
2. RM HEIGHT few sticks per core — 52–122 µs cells doing 1/32 useful work.
3. TILE WIDTH large-K owner combine — exchange is the whole kernel time.
4. Streamed gamma — a different mechanism (L1 fit), hits HEIGHT TILE cells too.
5. Interleaved prefill — the earlier-snapshot priority rule. It turned out to be close to ceiling (see above).

### Portfolio and verdicts
Five `perf-part-optimizer` subagents ran in parallel. Artifacts: `perf_experiments/<slug>/` (raw profiler dumps are
gitignored). Brief: `perf_experiments/BRIEF_perf2.md`.

| idea | verdict | before → after, ns | domain / notes |
|---|---|---|---|
| `narrow_rm_flat`: read 16 B-wide RM shards as flat tiles; per-stick sums by 0/1 matmuls (S_j) into compressed unit tiles; combine over Q sub-blocks spread across members; the unit's leader multicasts inv; pass 2 = (inv @ E_j) · x with gamma folded into E_j | **WIN, graduated** | wid_rm_5x3x928x544 850 233 → 38 862 · wid_rm_13x777x1023 679 556 → 35 703 · wid_rm_1024x1024 117 210 → 15 947 · wid_rm_f32_4x1x512x512 227 556 → 14 191 · blk_rm_99991x64 513 351 → 28 597 | ROW_MAJOR WIDTH / BLOCK shards whose width divides a 16-datum face row (bf16 8/16, fp32 4/8/16 datums) and whose pitch equals the width. Menu: broadcast-form partials (no compression) 2–3× slower than compressed; one multicast per sub-unit 3–8× slower (collisions) → one leader multicast per unit. 1–2 bf16 ulp from the old output; PCC ≥ 0.99991. |
| `row_partial_mcast`: row-form partial (ones-row matmul with transposed sumsq), K × 128 B into ⌈K/32⌉ gather tiles, REDUCE_COL + DEST transpose on the owner, root multicast return, owner spread | **WIN, graduated (with one carve-out)** | wid_tile_128x4096 26 659 → 6 955 · wid_tile_224x3072 20 973 → 7 795 · wid_tile_160x11008 23 093 → 9 522 · wid_tile_13x777x1023 386 850 → 229 669 · wid_rm_1024x1024 117 406 → 61 770 (before narrow-flat) | every owner-combine block. K sweep at b = 4: K = 8 1.05×, 16 1.27×, 32 1.78×, 64 2.58×, 128 3.83×. Carve-out below. ≤ 2 bf16 ulp. |
| `height_rm_flat`: a HEIGHT RM stick = ⌈W/1024⌉ flat tiles; DEST-accumulated x², one REDUCE_SCALAR, one-vector finalize; resident RM gamma page (one read + multicast); cb_x / cb_out alias the shards | **WIN, graduated (cost-model gate)** | hgt_rm_f32_128x4096 93 223 → 3 279 · hgt_rm_224x11008 121 755 → 6 936 · (128,8192) 85 758 → 4 592 · (96,6144) 62 441 → 3 388 · (100,5120) 52 420 → 3 119 · (3104,4064) 24 sticks 43 239 → 29 499 | gate below. Error vs torch at or below the old path's on every measured cell but one (bf16, fp32 DEST, no gamma: rms 0.00145 vs 0.00122, PCC 0.999999 both). |
| `compact_gamma`: cb_gamma page stride 192 B (row 0 of faces 0/1 at +0 / +face_bytes, host-checked non-overlap, tail pages) | **WIN, graduated, no carve-out** | hgt_tile_res_128x8192 124 128 → 68 270 · hgt_rm_f32_128x4096 93 178 → 53 756 · hgt_rm_f32_128x8192 178 298 → 106 285 · hgt_rm_224x11008 121 577 → 111 863 | everywhere, bit-identical (NaN-fill proof on unwritten bytes). Its menu option 2 (compact RM row + local chunk assembly) not built: option 1 already removes streaming on every target. |
| `prefill_dram`: bank stagger (read / write / both), OVERLAP_BLOCKS 2, 32-tile output units, write split onto the read NoC, NoC-swap on top rows, static rebalance | **NULL** | (8192,7168): base 597–620k; stag_r 590–609k; ovl2 599–607k; write split 589–609k (ws2 regression); row swap 602–679k (regression); rebalance 633–675k (regression) | the op runs at its own copy ceiling. Starved Tensix cores (NoC arbitration by grid position) do not speed up with less work. Untested: dynamic tile-row claiming (rebalance data says it would not help). |

### What graduated (commits), domain, deleted code
- **9936df5e0dc, compact cb_gamma.** The single cb_gamma layout everywhere. Streaming stays only where even the compact
  slice does not fit (bf16 TILE HEIGHT W = 11008 on 5–7 Tensix cores: flat there). RM sharded plans now prefer resident
  gamma at any chunk cap before any streamed gamma (the old interleaved order was deleted).
- **562855eea1f, row partial + root return + owner spread.** Every owner-combine block (sharded k > 1, interleaved
  k > 8), TILE and RM. Deleted: the faces-0/2 owner partial path (`static_assert`s pin it), the k-tile owner add.
  `cb_gathered` is ⌈k/32⌉ pages per owned row (k = 128: 16 KB instead of 512 KB).
- **807efb3acbb, HEIGHT RM flat regime.** New kernels `rms_norm_flat_{reader,compute,writer}.cpp`; router in
  `create_program_descriptor`. Also runs when the tilized plan does not fit L1 (new: fp32 [32,4096] shards).
- **4952a4ae10b, narrow-flat regime.** New `rms_norm_narrow_flat_program_descriptor.py` + kernels
  `rms_norm_narrow_flat_*`. For its domain it replaces the stick staging → tilize → per-tile-row owner combine →
  untilize → stick write-back path.

### Carve-outs and measured trades
- **Narrow-flat — inexpressible elsewhere (feasibility).** A shard width that does not divide a 16-datum face row
  (24, 40, 320 datums …) spans two logical rows of a flat tile, so one right-multiplication cannot reduce a stick; the
  module returns None. Those cells measured unchanged (blk_rm_128x4096 13.4 → 13.0 µs, wid_rm_3x1x736x5119
  198 → 197 µs in the subagent session; it still gains 1.41× from the row-partial exchange).
- **HEIGHT RM flat — measured crossover.** Written as a cost model in portable units (S = sticks per Tensix core,
  W): flat ≈ 1.2 µs + S·(150 + 230·⌈W/1024⌉ + 80·[W % 1024 ≠ 0]) ns vs tilized ≈ ⌈S/32⌉·(2300 + 310·⌈W/32⌉) ns.
  Forcing flat above the crossover measured 0.54–0.58× ((2048,256), (3104,416)); the model picked the winner on every
  measured cell. Also inexpressible: TILE-layout gamma (golden-INVALID with RM HEIGHT input) and bf8b.
- **Root return — interleaved multi-row blocks keep the unicast return.** Root hop measured +2.2–3.7 % on DRAM-bound
  interleaved k = 16, b = 2–8 (3 paired runs: (1024,16384) bf16 167.5 → 171.2 µs, fp32 167.3 → 173.6 µs,
  (512,32768) 172.0 → 175.8 µs). Written as `root = sharded` in `_inv_mcast_order`.
- **Combine pipeline K gate kept.** Lifting it with the root return measured 0.76–0.78× ((224,3072), (256,2048)).

### Scoreboard — round start → final, same session (ns, speedup)

| class (bottleneck) | cell | before | after | × |
|---|---|---|---|---|
| RM WIDTH 16 B shards (staging + exchange) | wid_rm_5x3x928x544 | 851 173 | 38 897 | 21.88 |
| RM WIDTH 16 B, W tail (7 of 8 cols) | wid_rm_13x777x1023 | 678 635 | 35 352 | 19.20 |
| RM WIDTH 16 B, fp32 DEST, fp32 RM gamma | wid_rm_1024x1024 | 118 068 | 15 901 | 7.43 |
| RM WIDTH fp32 x, 16 B (4 datums) | wid_rm_f32_4x1x512x512 | 227 963 | 14 213 | 16.04 |
| RM WIDTH (2048,256) 16 B | wid_rm_2048x256 | 141 762 | 12 822 | 11.06 |
| RM WIDTH 48 B shards (not narrow; exchange) | wid_rm_224x3072 | 36 999 | 18 202 | 2.03 |
| RM WIDTH 80 B shards (not narrow; exchange) | wid_rm_3x1x736x5119 | 197 919 | 140 322 | 1.41 |
| RM BLOCK 16 B shards | blk_rm_99991x64 | 513 668 | 28 644 | 17.93 |
| RM BLOCK 48 B | blk_rm_2048x256 | 18 186 | 17 000 | 1.07 |
| RM BLOCK wide | blk_rm_128x4096 / blk_rm_96x6144 | 13 212 / 12 122 | 13 364 / 12 250 | 0.99 / 0.99 (noise; median of 3: 0.986) |
| RM HEIGHT 1 stick / core, fp32 | hgt_rm_f32_128x4096 | 93 352 | 3 265 | 28.59 |
| RM HEIGHT 1 stick / core, fp32 W = 8192 | hgt_rm_f32_128x8192 | 113 039 | 3 982 | 28.39 |
| RM HEIGHT 2 sticks / core, W % 1024 ≠ 0 | hgt_rm_224x11008 | 121 784 | 6 944 | 17.54 |
| RM HEIGHT 24 sticks / core | hgt_rm_3104x4064 | 43 224 | 29 543 | 1.46 |
| RM HEIGHT many sticks (tilized by the gate) | hgt_rm_2048x256 | 5 046 | 5 010 | 1.01 |
| TILE WIDTH K = 128 (owner combine) | wid_tile_128x4096 | 27 164 | 7 031 | 3.86 |
| TILE WIDTH bf8b, no gamma | wid_tile_bf8_128x4096_nog | 26 051 | 6 430 | 4.05 |
| TILE WIDTH K = 96 | wid_tile_224x3072 | 21 126 | 7 806 | 2.71 |
| TILE WIDTH K = 115 | wid_tile_160x11008 | 23 182 | 9 672 | 2.40 |
| TILE WIDTH K = 32, b = 325 (now pass-2 bound) | wid_tile_13x777x1023 | 386 804 | 229 841 | 1.68 |
| TILE WIDTH K = 8, pipelined | wid_tile_2048x256 | 31 521 | 30 147 | 1.05 |
| TILE WIDTH perf decode (achievable 5 481 / 5 267) | wid_perf_32x7168 / 32x5120 | 4 988 / 4 641 | 4 724 / 4 321 | 1.06 / 1.07 |
| TILE HEIGHT, fp32 RM gamma (was streamed) | hgt_tile_res_128x8192 | 124 024 | 68 286 | 1.82 |
| TILE HEIGHT compute-bound (4 Tensix cores) | hgt_tile_128x8192 / hgt_tile_4096_g | 68 710 / 35 089 | 68 747 / 35 031 | 1.00 / 1.00 |
| TILE HEIGHT W = 11008 (still streams) | hgt_tile_160x11008 | 90 781 | 90 867 | 1.00 |
| TILE HEIGHT bf8b | hgt_tile_bf8_2048x256 | 3 776 | 3 730 | 1.01 |
| TILE BLOCK perf (achievable 18 871 / 25 947 / 25 640 / 19 039) | blk_perf_7168x1024 / 7168x2048 / 8192x1024 / 640x6144 | 22 423 / 30 730 / 29 275 / 14 714 | 22 188 / 30 339 / 29 201 / 14 852 | 1.01 / 1.01 / 1.00 / 0.99 |
| TILE BLOCK | blk_tile_128x4096 / blk_tile_13x777x1023 | 10 072 / 33 079 | 10 090 / 33 004 | 1.00 / 1.00 |
| interleaved prefill (DRAM, close to ceiling) | int_prefill_7168 / 5120 / 1024 | 603 890 / 419 431 / 83 116 | 606 974 / 419 355 / 82 818 | 0.99 / 1.00 / 1.00 (median of 3 on 7168: 0.991) |
| interleaved perf | int_perf_7168x2048 / int_mid_7168 / int_perf_32x7168 | 144 470 / 45 628 / 7 570 | 145 938 / 45 063 / 7 337 | 0.99 / 1.01 / 1.03 |
| interleaved decode, owner k = 49 | int_decode_32768 | 18 292 | 18 238 | 1.00 |
| interleaved fp32 / bf8b | int_tile_f32_128x8192 / int_tile_bf8_128x8192 | 25 278 / 11 944 | 23 696 / 12 276 | 1.07 / 0.97 (median of 3: 0.994) |
| interleaved RM | int_rm_99991x64 / int_rm_128x4096_gf32 / int_rm_160x11008 | 236 267 / 15 516 / 24 968 | 237 492 / 15 574 / 25 141 | 0.99 / 1.00 / 0.99 (medians of 3: 1.001 / — / 1.007) |
| interleaved non-aligned | int_tile_13x777x1023 | 110 334 | 109 034 | 1.01 |

- Every axis value appears above: dtype f32 / bf16 / bf8b; fp32 DEST on / off; `Layout::TILE` / ROW_MAJOR; alignment
  tile / w_non (13x777x1023, 3x1x736x5119) / h_non (99991x64); rank 2 / 3 / 4; gamma none / bf16 / fp32 / bf8b,
  TILE / RM; every `TensorMemoryLayout`.
- Close to ceiling: interleaved prefill (8192,7168) 600 µs vs its copy-only 567–579 µs at ≈ 410 GB/s.

### Guard set / golden (all green)
- Golden `test_golden` (12 chunks: 4 placements × {bf8b, fp32, other}) + `test_regression` + `test_translated` after
  each of the four graduations: every chunk passed (same pass / skip / xfail counts as round start; 0 failed, 0 hangs).
- Unit dir `tests/ttnn/unit_tests/operations/rms_norm/`: 336 passed, 3 skipped after each graduation.
- Same-session A/B over 48 cells (above): no cell slower beyond noise. The four cells that read 0.97–0.99× in the
  single A/B were re-run 3× alternating: medians 0.986–1.007.

### Next bottlenecks (for a Perf 3)
- **TILE WIDTH K = 32, many rows** (13,777,1023): 230 µs, now pass 2 (≈ 180 µs on the pack thread).
- **RM WIDTH shards wider than 16 datums but not narrow** (40-datum: 140 µs): staging + tilize; the narrow-flat
  mechanism needs a two-sided (P @ x² @ B) reduction to reach them.
- **TILE HEIGHT W = 11008** still streams gamma on 5–7 Tensix cores (flat: pass 2 runs at the resident rate).
- **TILE HEIGHT compute** on 4 Tensix cores (harness geometry): 268 ns per tile at HiFi4 fp32 DEST.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::reduce` (owner combine + member collapse) | capability | No row-form output: its REDUCE_ROW leaves the sums in column 0, and `post_reduce_op` cannot call `transpose_dest` (it reprograms the FPU MOP / addressing the next `reduce_tile` needs). No "REDUCE_COL over gather tiles → DEST transpose → finalize" combine. Raw `matmul_init(…, transpose in1)` / `matmul_tiles` (member: ones-row × sumsq^T) and raw `reduce_init<REDUCE_COL>` / `reduce_tile` + `transpose_dest` (owner). | 26 659 (wid_tile_128x4096, helper combine of k face-0/2 tiles) | 6 955 | `kernels/rms_norm_compute.cpp:213` (`collapse_rows_to_row0` :226, `combine_rows_block` :255) |
| no helper (dataflow: partial-row scatter) | capability | No kernel_lib helper writes partial tile rows (two 16-datum face rows into row m % 32 of a remote gather tile) or zero-fills gather rows from the NoC zeros region; raw `noc_async_write` / `noc_async_read(MEM_ZEROS_BASE)`. | n/a (no helper path) | — | `kernels/rms_norm_writer.cpp:499` (face rows), `:410` (`zero_unused_gather_rows`) |
| `compute_kernel_lib::matmul_block` (narrow-flat pass 1 / pass 2) | capability | Built around K-blocked activations × weights with its own CB lifecycle: cannot index a per-tile resident constant (S_j / E_j selected by flat-tile index), reuse ONE inv tile as in0 across R outputs, or leave the product in DEST for a following DEST-reuse multiply before packing. Raw `matmul_init` / `matmul_tiles` / `matmul_block_init` / `matmul_block`. | n/a (no helper path; old op path 850 233 on wid_rm_5x3x928x544) | 38 862 | `kernels/rms_norm_narrow_flat_compute.cpp:22` (calls :145, :240) |
| `eltwise_chain` `DestReuseBinary` (narrow-flat pass 2) | capability | A chain cannot start from a matmul result already in DEST; raw `mul_reuse_dest_tiles<DEST_TO_SRCA>` with the srcB reconfig done by hand. | n/a | — | `kernels/rms_norm_narrow_flat_compute.cpp:246` |
| SFPU compute API (`binop_with_scalar` + `rsqrt_tile`) in narrow-flat combine | capability | No entry point sums the Q sub-blocks of one tile and then runs rsqrt on sub-block 0 only (`FoldFinalize`); stock calls scope only by face. | n/a | — | `kernels/rms_norm_narrow_flat_compute.cpp:105` |
| kernel_lib `Mcast` (narrow-flat inv return) | capability | One fixed (or fixed-order rotating) sender per group; here the sender is the leader of each unit (owner of its first sub-unit) while every member is simultaneously a sender of other units. Raw multicast + per-unit flag. | n/a | — | `kernels/rms_norm_narrow_flat_reader.cpp:22` |
| no helper (dataflow: 0/1 selection matrices) | capability | Nothing in kernel_lib builds constant 0/1 (or gamma-carrying) matrix tiles; the narrow-flat reader writes S_j / E_j with RISC-V stores from a table. | n/a | — | `kernels/rms_norm_narrow_flat_reader.cpp:22` |
| `eltwise_chain` (HEIGHT RM flat pass 1 / pass 2) | capability | Inputs are addressed by tile index with a unit column step (`Strided{base, row_stride}`); flat tile j of stick s sits at page s·stick_pages + j·tile_pages of the zero-copy cb_x (page = gcd(stick pitch, tile bytes)), and a stick's last flat tile may come from a second CB (tail slot). Raw `mul_tiles`, `mul_tiles_bcast_scalar`, `mul_reuse_dest_tiles`, out-of-order `pack_tile`. | n/a (no helper path; old op path 93 223 on hgt_rm_f32_128x4096) | 3 279 | `kernels/rms_norm_flat_compute.cpp:16` |
| SFPU compute API `rsqrt_tile` (HEIGHT RM flat finalize) | capability | Scope only by face (`VectorMode`), never to the ONE vector holding element (0, 0) of a REDUCE_SCALAR result; raw `_llk_math_eltwise_unary_sfpu_params_(…, VectorMode::RC_custom)`. Bit-identical. | 3 270 (8-vector column-0 finalize, per stick +~210 ns at 32 sticks) | 3 279 (1 stick: equal) | `kernels/rms_norm_flat_compute.cpp:72` |
| `dataflow_kernel_lib` local copy (`set_read_state` / `read_with_state`) — flat tail staging | capability | The stateful copy fixes the source address and varies only the destination; the flat tail copy steps the source per stick (and the write-back steps the destination). Raw `noc_async_read` / `noc_async_write`. | n/a | — | `kernels/rms_norm_flat_reader.cpp:61`, `kernels/rms_norm_flat_writer.cpp:100` |

Compact cb_gamma added no bypass (the existing `MulGammaRowBcast` row in Perf 1 covers the unpack it relies on). The
root inv_rms return uses the kernel_lib `Mcast` helper (no bypass).
