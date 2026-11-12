# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-12
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime R1 row-resident: tile-rows split over the full compute grid, whole rows resident per core (x read from DRAM once), Σx² by packer L1 accumulation with masked partial-tile fold, rstd by reduce post-op, W-chunked normalize / gamma scale / untilize; one `ttnn.generic_op` dispatch; native TILE and ROW_MAJOR.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"], gamma_layout=[ROW_MAJOR, "none"], memory_layout=[INTERLEAVED]; EXCLUSION `{float32, fp32_dest_acc_en=False}`
- **Accuracy achieved** (with gamma, HiFi4, fp32 DEST; 4 shapes × 2 layouts via test_rms_norm_precision_baseline.py): bf16 PCC ≥ 0.999998, max_abs_err ≤ 0.089, rel_rms_err ≤ 0.0022, ULP p99 = 1; fp32 PCC ≥ 0.9999997, max_abs_err ≤ 0.041, rel_rms_err ≤ 0.0019 (systematic −0.14% tf32 operand-truncation bias, within tolerance)
- **Golden suite at Phase 0**: 436 supported_pass / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode; 6481 xfail_expected (per `verifier_report.json`)
- **Issues encountered**: verifier fixes — host `n_sq` now derived from the `core_w_tiles` knob (`_n_sq`) instead of the tensor width (DRY with the compute kernel); `l1_ledger.md` stale mask symbol and worked examples replaced with measured block knobs. No drift fixes needed.
- **Tests added**: test_rms_norm.py (acceptance), test_rms_norm_debug.py, test_rms_norm_perf_shapes.py, test_rms_norm_precision_baseline.py

## Refinement 1 — TILE-layout gamma + HEIGHT_SHARDED placement (dataflow-boundary refinement)
- Date: 2026-11-12
- What was done:
  - **TILE gamma** (reader only): `gamma_tiled` CT flag; per W chunk the reader reads the chunk's valid gamma tile pages straight into `cb_gamma` (one barrier per chunk) and skips the RM row-0 relayout. Works for TILE and RM x, fp32/bf16 gamma (incl. mixed with x). Padding columns of the last gamma tile only multiply output padding (poisoned-padding test).
  - **HEIGHT_SHARDED (R6)**: work split = the shard grid (core i owns shard i, ragged last shard, orientation-aware). Placement CT arg shared by reader/writer:
    - `RESIDENT` — TILE x: `cb_x` / `cb_out` ARE the input / output shards (`cb_descriptor_from_sharded_tensor`, tile-size pages); RM x with `W % 32 == 0` and 32-stick-multiple shards: `cb_x_rm` / `cb_out_rm` are the shards (in-place tilize / untilize). Reader just hands the resident pages to compute; writer just drains credit. Zero NoC traffic for x / out.
    - `LOCAL_STICKS` — RM shards that tilize cannot consume in place (stride `align16(W·elem) ≠ Wc·32·elem`, or stick counts that are not whole tile-rows — `auto_shard_config` typically gives 1–18 sticks/core): resident sticks restrided into `cb_x_rm` with self-aimed L1→L1 reads, and the valid output sticks copied into the output shard. No DRAM.
    - Block selection for resident shards: `K = 1`, largest `r` (whole shard when it fits); wide rows fall back to `r = 1`, `c | Wc` (pack order must equal shard row order). RM in-place needs `K = 1`, else falls back to LOCAL_STICKS.
  - `validate()`: output `memory_config` must match the input placement (interleaved→interleaved; HEIGHT_SHARDED→same shard spec); WIDTH/BLOCK still rejected via SUPPORTED. Default output for a HEIGHT_SHARDED input = the input's memory config.
  - **Bug fix (shared helper)**: `reduce<…, AccumulateViaAdd, Collapse>` armed `sfpu_reduce_init` once before its per-output loop, but an SFPU `post_reduce_op` (ours: binop_with_scalar + rsqrt inits) clobbers that macro, so every output after the first in one call collapsed wrongly. Re-armed after the post-op in `reduce_helpers_compute.inl` (gated: Collapse + non-NoOp post-op + more outputs pending). This was a latent Phase 0 bug — any block with `block_rows ≥ 2` (e.g. interleaved 7680×512 fp32: max err 0.48) — never hit by Phase 0 tests because every tested shape had ≤ 1 tile-row per core.
- Accuracy achieved: unchanged on Phase 0 shapes (precision baseline green); new paths within helpers.TOLERANCES on all tested shapes (bf16 rel-RMS < 0.01, fp32 multi-row interleaved 7680×512 now max err ~0.01).
- Golden test progress (targeted slices, fp32_dest_acc_en=True): 975 passed (all 4 targeted loose cases incl. `(1,1,32,16384)`, `(1,1,64,12288)`, HEIGHT_SHARDED `(1,1,256,512)`; 820 valid TILE-gamma / HEIGHT_SHARDED cartesian cells; 28 prior-passing RM-gamma interleaved reps), 16 infeasible-L1 skips (output shard does not fit next to the input shard), 1 expected failure `(1,1,32,32768)` (needs R3 — Refinement 4).
- Issues encountered: the reduce-helper post-op init clobber above; RM shard widths are page-granule padded (shard_w ≥ W, not == W).
- Tests added: `test_rms_norm_sharded_tile_gamma.py` (TILE gamma interleaved incl. poisoned padding + mixed dtype; HEIGHT_SHARDED auto shards × layout × dtype × gamma; RM resident pinned shards; default output inherits the shard spec; multi-tile-row interleaved blocks pinning the helper fix); `test_rms_norm_debug.py::test_debug_two_tile_rows_one_block`.

## Refinement 1b — TILE-layout gamma + HEIGHT_SHARDED placement (debug: fix gate violations)
- Date: 2026-11-12
- What was done: The gate failure `test_height_sharded_auto[nog-fp32-rm-(128, 8192)]` (NotImplementedError R3) passed in isolation but failed when the module ran in order. Cause: earlier `oom_guard`-skipped cases (fp32 (128,8192) TILE HEIGHT_SHARDED, where input + output shards do not fit L1) kept their sharded input tensor alive in the pytest traceback. The leaked low L1 buffer shrank later cases' CB space, from ~1.3 MB to ~0.8 MB. Fixes: (a) the test's `_run` deallocates its device tensors in `finally`; (b) the program descriptor's CB budget (`_cb_budget`) is now `lowest allocated L1 address (relative to allocator base) - margin`. The old bound was the largest free block, which overestimates whenever a free hole sits above a low buffer, and that caused a TT_THROW "static CBs clash with L1 buffers" instead of a clean selection. Reused: block selection, kernels unchanged. Added: `_cb_budget` helper, test teardown.
- Accuracy achieved: unchanged (kernels untouched); all HEIGHT_SHARDED / TILE-gamma unit cases within helpers.TOLERANCES.
- Golden test progress: full suite 1297 passed / 1 failed / 5621 xfailed / 33917 skipped, 0 hangs. The single failure is the `(1,1,32,32768)` interleaved loose case excepted in Refinement 1's Done-when (needs W-split R3, Refinement 4).
- Issues encountered: order-dependent L1 leak from skipped tests (above).
- Tests added: none (existing `test_rms_norm_sharded_tile_gamma.py` hardened with tensor teardown). Unit dir: 229 passed, 19 skipped, 0 failed.

## Refinement 2 — Numerical configurability: `fp32_dest_acc_en=False`, bfloat8_b input and gamma
- Date: 2026-11-12
- What was done:
  - `SUPPORTED`: `fp32_dest_acc_en += False`, `dtype += bfloat8_b`, `gamma_dtype += bfloat8_b`. bf8b+RM / bf8b+non-aligned / bf8b RM gamma stay INVALID in `feature_spec.py`. `{float32, fp32_dest_acc_en=False}` is still the only EXCLUSION. No new exclusions needed.
  - Program descriptor: one intermediate format `_Geometry.interm_fmt`, set from the DEST width (Float32 for fp32 DEST, Float16_b for 16-bit DEST). `cb_rstd` and `cb_normed` use it, and so do their sizes in the block-selector footprint. With 16-bit DEST `cb_normed` is half the size, so the selector can pick a larger `r`/`c`. `cb_sumsq` (and `cb_reduce_mask`, which has to match its srcB format) stays Float32 in both modes. Reason: it is a packer L1 accumulator, so the Σx² adds happen in L1 at the page format, not in DEST. Measured with bf16 pages at W=7168: relative RMS 0.017 (bf8b: 0.023), a ~1–2% rstd scale bias. With Float32 pages: 0.006. Cost is ≤ 2 tiles per row.
  - bf8b: element sizes are now computed only for ROW_MAJOR tensors (`element_size()` throws for block-float). `cb_x` / `cb_gamma` / `cb_out` already took their format from the tensor dtype.
  - **No kernel change.** The verifier expected the Σx² packer-L1 accumulation to need replacing for 16-bit DEST. Measurement says otherwise: packer L1 accumulation works correctly with 16-bit DEST, both into Float32 pages (shipped) and into bf16 pages (tested, less accurate). The chunk-batched L1-acc `sumsq_block` stays as the only Σx² mechanism for both DEST widths. Refinement 3 should profile this same path.
- Accuracy achieved (bf16 output, gamma): 16-bit DEST HiFi4 PCC ≥ 0.99999, relative RMS ≈ 0.005–0.006 on (1,1,32,32), (2,3,64,96), (1,1,50,47) TILE/RM, (1,1,256,7168). HiFi2 perf corner: (1,1,8192,7168) PCC 0.99992 / rRMS 0.0055 and (1,1,8192,1024) PCC 0.99994, both above the 0.9995 soft gate. bf8b input: PCC ≥ 0.99989, rRMS 0.010–0.014. bf8b gamma with fp32 DEST: PCC 0.999999, rRMS 0.002.
- Golden test progress (targeted slices): `test_op_loose` with fp32_dest_acc_en=False, INTERLEAVED + HEIGHT_SHARDED: 196/196 passed, including all 12 interleaved `_PERF_BASE` cases. Cartesian `test_op` slice (shapes 1x1x32x64, 128x8192, 4x8x32x256, 2x1x100x47 × {bf8b or fp32-DEST-off} × {INTERLEAVED, HEIGHT_SHARDED}): 179 passed, 84 xfailed (all `{float32, False}`), 1465 skipped (INVALID), 0 failed, 0 hangs.
- Issues encountered: `test_rms_norm_debug.py` constructs `_Geometry` directly, so the new `fp32_dest_acc_en` parameter defaults to True (the prior behavior).
- Tests added: `test_rms_norm_numeric_config.py` (62 cases): bf16 with 16-bit DEST × TILE/RM × aligned/non-aligned × gamma; bf8b input × {fp32, bf16} DEST × gamma {bf8b, bf16, fp32, none}; bf8b gamma with bf16/fp32 input; HiFi2 8192-row perf corner at PCC 0.9995.

## Refinement 3 — Speed up the interleaved prefill perf profile
- Date: 2026-11-12
- What was done:
  - **Diagnosis**: profiled the six interleaved `_PERF_BASE` prefill cases at their exact config (new `test_rms_norm_prefill_perf.py`). For an empirical DRAM roofline, profiled `ttnn.clone` on the same tensors (new `test_rms_norm_dram_roofline.py`), since `noc_estimate` is not built in this tree. The op was 15–24% above clone. Root cause: the TILE-layout gamma `(1,1,1,W)` is padded to 32 rows per tile, and the reader fetched whole 2 KiB pages once per block. Only row 0 is used (row-broadcast multiply), so each block paid one extra x tile-row of DRAM reads (+33–50% reads at 2–3 rows per core).
  - **Lever 1 (shipped, reader + host)**: TILE bf16/fp32 gamma now reads only the 64 B head of face 0 and face 1 per tile. These are the broadcast row, rounded up to the DRAM read alignment. Two new CT args, `g_row0_bytes` and `g_face1_offset`, are derived once in `_Geometry`. bf8b gamma (shared exponents ahead of the data) keeps whole-page reads (`g_row0_bytes = 0`).
  - **Lever 2 (measured, parked as a live knob)**: `x_block_depth` (op_design perf lamp). When `X_BLOCK_DEPTH > 1`, cb_x holds that many blocks and `block_rows` is capped at `ceil(rows_cap / BLOCKS_PER_CORE)`, so the reader prefetches block b+1 while block b is normalized and written. Footprint and selection carry the depth. No kernel change was needed: compute addresses cb_x relative to the front and pops per block. Measured depth 2 with r = 1: **10–12% slower** (8192×1024: 89.3 → 98.2 µs; NCRISC 75 → 90 µs). At 2–3 tile-rows per core, the serial read-all / write-all phases use DRAM better than interleaved read+write. Parked at `X_BLOCK_DEPTH = BLOCKS_PER_CORE = 1`, which gives the same selection as before.
  - Reused: reader gamma chunk loop, `_select_blocks`, `_footprint`. Added: 2 reader CT args, a `x_depth` footprint term, and 2 host knobs.
- Device kernel ns (Blackhole p150b @ 1350 MHz = reference clock), before → after (achievable_ns / clone roofline):
  - 8192×1024: 103 094 → 90 227 (96 744 / 83 233)
  - 8192×2304: 217 930 → 188 401 (211 345 / 182 801)
  - 8192×5120: 476 980 → 418 990 (738 307 / 413 404)
  - 8192×7168: 681 422 → 569 744 (1 032 281 / 556 699)
  - 7168×1024: 86 767 → 74 669 (89 392 / 71 836)
  - 7168×2048: 172 677 → 145 999 (172 892 / 146 827)

  Every case now meets its `achievable_ns`. Ablation: without gamma the op equals the clone roofline (8192×1024: 83.0 µs; 8192×7168: 557.4 µs), so the op is DRAM-bound. The remaining gap is the gamma multiply pass plus the 3-row tail cores on 256-tile-row shapes.
- Accuracy achieved: soft PCC gate 0.9995 holds on all six prefill cases (HiFi2, bf16 DEST). The unit precision tests are unchanged.
- Golden test progress: slice of `test_op_loose` (INTERLEAVED + HEIGHT_SHARDED) plus the cartesian `test_op` for 2x1x100x47 / 4x8x32x256 (non-W/B-sharded): 349 passed, 42 xfailed, 1 failed. The failure is the excepted `(1,1,32,32768)` case, which needs R3 (Refinement 4). Unit dir: 303 passed, 19 skipped, 0 failed.
- Issues encountered: none. The NoC-estimator CLI is not built in this tree, so `ttnn.clone` served as the measured DRAM roofline.
- Tests added: `test_rms_norm_prefill_perf.py` (exact-config prefill cases, PCC 0.9995) and `test_rms_norm_dram_roofline.py` (clone roofline, for `--profile`).

## Refinement 3 — Speed up the interleaved prefill perf profile
- Date: 2026-11-12
- What was done:
  - **Diagnosis**: profiled the six interleaved `_PERF_BASE` prefill cases at their exact config (new `test_rms_norm_prefill_perf.py`). For an empirical DRAM roofline, profiled `ttnn.clone` on the same tensors (new `test_rms_norm_dram_roofline.py`), since `noc_estimate` is not built in this tree. The op was 15–24% above clone. Root cause: the TILE-layout gamma `(1,1,1,W)` is padded to 32 rows per tile, and the reader fetched whole 2 KiB pages once per block. Only row 0 is used (row-broadcast multiply), so each block paid one extra x tile-row of DRAM reads (+33–50% reads at 2–3 rows per core).
  - **Lever 1 (shipped, reader + host)**: TILE bf16/fp32 gamma now reads only the 64 B head of face 0 and face 1 per tile. These are the broadcast row, rounded up to the DRAM read alignment. Two new CT args, `g_row0_bytes` and `g_face1_offset`, are derived once in `_Geometry`. bf8b gamma (shared exponents ahead of the data) keeps whole-page reads (`g_row0_bytes = 0`).
  - **Lever 2 (measured, parked as a live knob)**: `x_block_depth` (op_design perf lamp). When `X_BLOCK_DEPTH > 1`, cb_x holds that many blocks and `block_rows` is capped at `ceil(rows_cap / BLOCKS_PER_CORE)`, so the reader prefetches block b+1 while block b is normalized and written. Footprint and selection carry the depth. No kernel change was needed: compute addresses cb_x relative to the front and pops per block. Measured depth 2 with r = 1: 10–12% slower (8192×1024: 89.3 → 98.2 µs; NCRISC 75 → 90 µs). At 2–3 tile-rows per core, the serial read-all / write-all phases use DRAM better than interleaved read+write. Parked at `X_BLOCK_DEPTH = BLOCKS_PER_CORE = 1`, which gives the same selection as before.
  - Reused: reader gamma chunk loop, `_select_blocks`, `_footprint`. Added: 2 reader CT args, a `x_depth` footprint term, and 2 host knobs.
- Device kernel ns (Blackhole p150b @ 1350 MHz = reference clock), before → after (achievable_ns / clone roofline):
  - 8192×1024: 103 094 → 90 227 (96 744 / 83 233)
  - 8192×2304: 217 930 → 188 401 (211 345 / 182 801)
  - 8192×5120: 476 980 → 418 990 (738 307 / 413 404)
  - 8192×7168: 681 422 → 569 744 (1 032 281 / 556 699)
  - 7168×1024: 86 767 → 74 669 (89 392 / 71 836)
  - 7168×2048: 172 677 → 145 999 (172 892 / 146 827)

  Every case now meets its `achievable_ns`. Ablation: without gamma the op equals the clone roofline (8192×1024: 83.0 µs; 8192×7168: 557.4 µs), so the op is DRAM-bound. The remaining gap is the gamma multiply pass plus the 3-row tail cores on 256-tile-row shapes.
- Accuracy achieved: soft PCC gate 0.9995 holds on all six prefill cases (HiFi2, bf16 DEST). The unit precision tests are unchanged.
- Golden test progress: slice of `test_op_loose` (INTERLEAVED + HEIGHT_SHARDED) plus the cartesian `test_op` for 2x1x100x47 / 4x8x32x256 (non-W/B-sharded): 349 passed, 42 xfailed, 1 failed. The failure is the excepted `(1,1,32,32768)` case, which needs R3 (Refinement 4). Unit dir: 303 passed, 19 skipped, 0 failed.
- Issues encountered: none. The NoC-estimator CLI is not built in this tree, so `ttnn.clone` served as the measured DRAM roofline.
- Tests added: `test_rms_norm_prefill_perf.py` (exact-config prefill cases, PCC 0.9995) and `test_rms_norm_dram_roofline.py` (clone roofline, for `--profile`).

## Refinement 4 — Cross-core W-split (regime R3): logical W-split on interleaved + WIDTH_SHARDED + BLOCK_SHARDED
- Date: 2026-11-12
- What was done:
  - **SUPPORTED**: `memory_layout` += `WIDTH_SHARDED`, `BLOCK_SHARDED`; sharded output follows the input's shard spec (any sharded layout).
  - **Host**: work planning now yields one record per core: a tile-row run plus a W slice `[w_elem_start, w_elem_end)`, its row-group, slot and root. Regimes:
    - **R1** (interleaved, `_r1_fits`): unchanged split.
    - **R6** (sharded, one W slice): HEIGHT as before. WIDTH/BLOCK grids with a single slice use the same path.
    - **R3** (WIDTH/BLOCK with ≥ 2 slices): shard (i, j) → row-group i, slot j, root = j = 0. BLOCK requires ROW_MAJOR orientation.
    - **R3, interleaved logical split**: taken only when R1 cannot hold one whole row (`F(1, K_floor) > B`). `G = ceil(Wc / ceil(Wc / grid.x))` cores per grid row; tile-rows are split evenly over `min(grid.y, Rt)` row-groups.
    - The interleaved predicate is pinned (no grid-size or row-count term). `FORCE_W_SPLIT` is a test hook.
  - `core_w_tiles` and the slice's partial-tile width stay compile-time. There is one reader/writer/compute triple per distinct `(core_w_tiles, w_partial)`, e.g. the ragged last shard column or RM granule slices.
  - **Compute**: `rstd_block` splits into two steps:
    - `partial_block`: `reduce<SUM, ROW, AccumulateViaAdd, Collapse>`, no post-op → `cb_partial`.
    - On the root, `combine_block`: `reduce<…, AccumulateViaAdd, ReduceWithinTile::Skip>` over `(rows, G)` with the existing rsqrt post-op → `cb_rstd_root`.
  - **Writer**: per block, every core unicasts its `rows` partial tiles into the root's `cb_gather` at slot `row·G + slot`. Contributors then increment the root's gather semaphore. The root waits for a cumulative count of `(G−1)·(block+1)` and pushes the gather.
  - The root multicasts `cb_rstd_root` into every group core's `cb_rstd` with the multicast helper: host `ttnn.Mcast` (NOC_1, fixed sender = slot 0, one channel over all groups) + `attach()`; kernel `McastArgs::optional_sender().send()` / `optional_receiver().receive()`. Loopback fills the root's own `cb_rstd`, so `cb_rstd` has one producer (the writer) on every core.
  - **Reader/writer**: RM chunks are addressed by slice element range. Local-shard offsets are slice-relative. Chunks lying wholly past a narrow slice are no-ops. The RM gamma segment read is aligned down to the 64 B DRAM read alignment, and the relayout skips the slack, because RM shard slices start on any 16 B granule.
  - Resident drain/push counts use the row stride (= shard tile width).
  - **CBs**: `cb_partial`, `cb_gather` (G·r pages, on every core so its address is uniform) and `cb_rstd_root` are new. Pages follow the DEST width; see Issues for why. Ledger updated (`l1_ledger.md` R3 section).
  - Reused: R1/R6 kernels, block selector, sumsq/normalize/scale/untilize phases, resident-shard CBs, LOCAL_STICKS restride path. Added: the R3 combine (compute + writer), work planner, kernel groups.
- Accuracy achieved: all focused cases pass `helpers.TOLERANCES`:
  - bf16 WIDTH/BLOCK, TILE/RM, poisoned padding, pinned perf geometries.
  - 32×32768 interleaved, fp32 DEST.
  - Forced interleaved R3 runs: partial W, RM, RM gamma.
- Golden test progress (targeted slices; the harness full run is the record):
  - WIDTH/BLOCK + `32x32768` loose cases: 193/196 pass, 3 skipped (infeasible shard geometry), 0 fail.
  - Cartesian WIDTH/BLOCK, fp32 input + fp32 gamma: 210/210 pass.
  - Cartesian, bf8b + bf16/DEST-off with bf16 gamma: 290/290 pass.
  - Cartesian, bf16/DEST-on: 601 pass, 0 fail before the slice hit the session time cap.
  - Unit dir: 322 passed, 0 failed.
- Perf (device ns, Blackhole p150b, not a target of this refinement):
  - Interleaved 32×32768 (R3, 12 cores, HiFi4 / fp32 DEST): 33.6 µs.
  - WIDTH 32×7168 `[32,256]×(7,4)`: 6.18 µs (HiFi4; `achievable_ns` 5.48 µs at HiFi2).
  - BLOCK 640×6144 `[64,704]×(9,10)`: 14.3 µs (`achievable_ns` 19.0 µs).
- Issues encountered:
  1. A kernel group made only of receivers (the last slice column) hit the mcast `sender()` static assert → switched to the role-polymorphic `optional_sender` / `optional_receiver`.
  2. `13×777×1023` WIDTH (325 tile-rows × 32 one-tile slices) did not fit: the input + output shards leave a 96 KiB CB budget, and a Float32 `cb_gather` alone was 128 KiB. Partial/gather pages now follow the DEST width: a partial leaves DEST once, so Float32 pages under 16-bit DEST held no extra bits. That halves the gather and the cell passes.
  3. Stall shadow (x·γ during the rstd wait) is **not built**. It needs a block-wide x·γ buffer, which does not fit next to resident shards that already fill L1. It also changes the rounding order. Left for Refinements 5/6, together with tuning the combine (`tensix_all_reduce` reduce-scatter / tree for large G).
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_w_split.py`. 19 cases:
  - WIDTH/BLOCK auto + pinned perf geometries + pad-poison.
  - Regime-pinned predicate test (32768 → R3, 16384 → R1).
  - 32×32768 interleaved.
  - `FORCE_W_SPLIT` interleaved cases (TILE/RM, partial W, RM gamma, 3 row-groups).

## Refinement 5 — Speed up the decode / few-row profile on the W-split scheme
- Date: 2026-11-12
- What was done:
  - **Diagnosis** (new `test_rms_norm_decode_perf.py`: exact-config `_PERF_BASE` decode / few-row cases).
    - Interleaved decode ran R1 on 1 core (32×7168: 39.1 µs vs 14.9 µs target), and 640-row cases on 20 cores.
    - Device zones on the R3 combine (32×7168) showed the root's rstd post-op taking **1.16 µs** (·1/W, +eps,
      rsqrt over a full tile), against 0.23 µs for the 28 tile adds.
    - The gather was **incast-serialized**: 27 × 2 KiB partials landed ~45 ns apart on the root.
  - **Lever 1 — interleaved predicate + group shape (host).**
    - `_take_r3_interleaved`: R3 when R1 cannot hold a row, **or** when R3 occupies ≥ `R3_OCCUPANCY_RATIO`
      (1.2) × R1's cores. Pinned from regime-forced measurements: R3 wins at 1.2× (3200×2048 73.5 → 70.7 µs;
      3200×7168 252 → 226 µs) and above, and loses at 0.64× (1504×544 11.6 vs 19.3 µs). Prefill never flips.
    - `_r3_group_rect`: a row-group is a `gx × gy` core rectangle (several grid rows when there are few
      row-groups). G ≤ Wt / `R3_MIN_SLICE_TILES`, and the W split is even.
    - `R3_MIN_SLICE_TILES = 5` was measured best over {2, 4, 5, 6, 8, 12}. `R3_MAX_GROUP_CORES = 0` (no cap):
      32×32768 drops 33.6 → 19.7 µs at G = 120; caps of 64 and 32 were slower.
  - **Lever 2 — column-scoped rstd post-op (compute).** `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile`
    take `VectorMode::C` (faces 0 and 2; column 0 is all that is read). Root post-op 1.16 → 0.66 µs. This is
    shared by R1/R6, one source.
  - **Lever 3 — half-tile gather (writer).** Contributors send only faces 0 and 2 of each partial, which
    halves the root incast. WS 32×7168 5253 → 4915 ns; WS 32×5120 5159 → 4496 ns.
  - **Lever 4 — stall shadow (compute + host).** x·γ is computed for the whole block into `cb_normed`
    during the gather / combine / multicast (on the root, after its combine), and `cb_x` is then released.
    Only ·rstd remains after rstd arrives. `R3_GAMMA_SHADOW` gates it to R3 + TILE + gamma, with a
    fallback when the block-wide `cb_normed` does not fit.
    - It is disabled for resident multi-row shards (BLOCK grids). There the gamma read takes 8 µs: 64
      row-0 reads per core, re-read by every row of the grid column. The shadow put that read on the
      critical path (640×7168 BLOCK 12.7 → 13.8 µs). Scaling after rstd hides it.
  - **Parked live knob**: `R3_STREAM_CHUNK_TILES` (0 = whole slice per chunk). Finer chunks measured slower
    (32×7168: c = 8 8.17 µs, c = 4 8.34, c = 2 8.87).
  - **Measured and dropped**: rstd multicast handshake off, and a 3/4-tile multicast payload. Both measured
    the same ~0.5 µs latency, so the latency is not protocol or payload.
  - Reused: R4's interleaved R3 planner, combine, multicast and kernel groups; MulGammaRow / MulRstdCol
    chain shapes. Added: the occupancy predicate, `_r3_group_rect`, the shadow element types,
    `shadow_block`, and one compute CT arg (`gamma_shadow`).
- Device kernel ns (Blackhole p150b @ 1350 MHz), before → after (`achievable_ns`):
  - Interleaved:
    - 32×1024: 8 903 → 4 105 (9 149)
    - 32×2304: 16 885 → 4 698 (17 003)
    - 32×5120: 30 004 → 6 023 (75 825)
    - 32×7168: 39 093 → 6 796 (14 894)
    - 640×6144: 55 201 → 45 218 (101 968)
    - 640×7168: 60 796 → 48 871 (119 307)
  - WIDTH_SHARDED:
    - 32×1024 `[32,128]×(8,1)`: 3 864 → 3 214 (4 110)
    - 32×2304: 4 596 → 3 802 (4 617)
    - 32×5120 `[32,160]×(8,4)`: 5 386 → 4 164 (5 267)
    - 32×7168 `[32,256]×(7,4)`: 5 543 → 4 374 (5 481)
  - BLOCK_SHARDED:
    - 640×6144: 11 057 → 9 944 (19 039)
    - 640×7168: 13 786 → 12 645 (21 669)
  - Guard set:
    - Prefill R1 8192×1024 90.2 → 88.1 µs; 8192×7168 569.7 → 562.3; 7168×2048 146.0 → 146.8 (noise).
    - 32×4096 RM: 8.9 µs, now on R3.
- Accuracy achieved: the soft PCC gate 0.9995 holds on all 12 decode / few-row cases (HiFi2, bf16 DEST).
  `helpers.TOLERANCES` holds on the unit dir and golden slices.
- Golden test progress (targeted slices):
  - `test_op` INTERLEAVED for 1×1×32×4096 + 2×512×1024 (both flipped to R3, every dtype / layout / gamma
    axis): 96 passed, 24 xfailed, 0 failed.
  - `test_op_loose` INTERLEAVED: 107/107.
  - `test_op_loose` WIDTH/BLOCK_SHARDED: 192 passed, 3 skipped (infeasible shard).
  - Unit dir: 354 passed, 19 skipped, 0 failed.
- Issues encountered:
  - The mcast `noc` must match the writer kernel's NoC, so an NOC_0 multicast is not expressible from the
    NOC_1 writer.
  - Remaining headroom is recorded on the refinement's Outcome line.
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_decode_perf.py`:
  - Exact-config decode / few-row cases (for `--profile`).
  - The `test_regime_guard` R1-vs-R3 pinned pairs that set the predicate.
  - The `test_r3_occupancy_predicate` regime pins.

## Refinement 6 — Speed up the block-sharded prefill profile
- Date: 2026-11-12
- What was done:
  - **Diagnosis** (new `test_rms_norm_block_prefill_perf.py`, exact `_PERF_BASE` BLOCK configs; per-stage device
    zones on 7168×1024 `[736,128]×(8,10)`, 23 tile-rows × 4 tiles per core, G = 8):
    - Compute-bound, not DM: NCRISC (gamma) ~1.9 µs of a 36 µs kernel.
    - The root's combine (23 rows × 8 partials + rsqrt post-op) took **17 µs** while the other 70 cores
      sat **18.7 µs** in the rstd wait. Per-core work before it: sumsq 5.7 µs, partial collapse 3.6 µs.
  - **Lever 1 — row-wise reduce-scatter combine (scheme change, writer + compute + host).**
    - Slot `o` combines rows `[o·q, (o+1)·q)`, `q = ceil(block_rows / G)`, then relays its rstd rows to the
      root's `cb_rstd_gather`. The root keeps the single helper multicast.
    - `cb_gather` is slot-major `G·q` pages (was `G·block_rows`) and reduced by
      `reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>` over `(G, q)`. Owners always push / consume the full
      quantum, so remotely addressed CBs stay at their base.
    - One owner (decode: `rows = 1`) is exactly the R5 path. Knob: `R3_SCATTER_COMBINE`.
    - 7168×1024: 36.0 → 24.8 µs.
  - **Lever 2 — stall shadow for tall resident blocks (host gate).** R5 disabled x·γ-during-the-wait for
    every resident multi-row shard because 640×7168 `[64,1024]` has an 8 µs gamma read. Gated now by
    `R3_SHADOW_RESIDENT_MIN_ROWS = 4` rows per block: the prefill shards (23–32 rows) take it; the 640-row
    shards (2 rows) don't. 24.8 → 23.0 µs.
  - **Lever 3 — pipelined half-tile gather (writer).** The gather was NoC-bandwidth-bound: a row all-to-all
    on one NoC ring costs bytes × hops.
    - Measured and rejected: one coalesced full-tile write per owner (fewer issues, 2× bytes) made the
      gather wait worse (3.4 → 4.6 µs). Staggering the owner order did nothing.
    - Kept: faces 0/2 only, and each owner's run is sent as soon as those partials are packed, overlapping
      the rest of the partial reduce. Gather wait 3.4 → 1.3 µs; 23.0 → 20.8 µs.
  - **Lever 4 — partial on the FPU reduce datapath (compute + reader).**
    - The per-core partial is ONE Σx² tile per row. There `reduce<…, ReduceTile>` (scaler 1.0, new
      `cb_reduce_scaler`, made by the reader with block 0's reads) beats accumulate + SFPU collapse
      (catalog `reduce_block`). Partial 3.6 → 1.2 µs.
    - Slices with a partial tail keep the masked AccumulateViaAdd path. Knob: `R3_PARTIAL_REDUCE_TILE`.
  - **Lever 5 — partial interleaved into sumsq (compute).** One ReduceTile call per owner run inside the sumsq
    row loop. Owner 0's partials leave after 3 rows, so the all-to-all overlaps sumsq: gather wait 2.3 → 0.6 µs.
    Knob: `R3_PARTIAL_INTERLEAVE`.
  - Reused: R4/R5's R3 planner, gather semaphore, mcast channel, shadow element types, and the reduce helper
    (three datapath configs). Added: `_combine_rows`, `CB_RSTD_GATHER`, `CB_REDUCE_SCALER`,
    `SEM_RSTD_GATHER`, group NoC coordinates as writer RT args, and four knobs.
- Device kernel ns (Blackhole p150b @ 1350 MHz = reference clock), before → after (`achievable_ns`):
  - BLOCK prefill, gamma (`_PERF_BASE` config):
    - 7168×1024 `[736,128]×(8,10)`: 36 039 → 18 784 (18 871)
    - 7168×2048 `[736,192]×(11,10)`: 45 507 → 25 318 (25 947)
    - 8192×1024 `[1024,128]×(8,8)`: 50 403 → 25 112 (25 640)
  - No gamma, same shards: 31.9 → 16.5, 38.4 → 21.2, 43.7 → 21.9 µs.
  - Guard set (the R3 code is shared):
    - BLOCK 640×6144 9 944 → 9 454; 640×7168 12 645 → 12 178.
    - WS decode 32×{1024, 2304, 5120, 7168}: 3 214 / 3 802 / 4 164 / 4 374 → 3 039 / 3 647 / 4 039 / 4 227.
    - Interleaved decode 32×{1024, 2304, 5120, 7168}: 4 105 / 4 698 / 6 023 / 6 796 → 3 953 / 4 537 / 5 773 / 6 506.
    - Interleaved 640×6144 45 218 → 44 119; 640×7168 48 871 → 48 376.
    - R1 prefill 8192×1024 88.1 → 88.3 µs; 7168×2048 146.8 → 147.4 µs (noise, untouched path).
- Accuracy achieved: soft PCC gate 0.9995 + `helpers.TOLERANCES[bf16]` RMS hold on all 10 BLOCK prefill
  cases (gamma / no gamma) and the 12 decode cases (HiFi2, bf16 DEST).
- Golden test progress (targeted slices):
  - `test_op_loose` WIDTH/BLOCK_SHARDED: 192 passed, 3 skipped (same as R5).
  - `test_op` 1×1×32×4096 + 2×512×1024 × {INTERLEAVED, WIDTH, BLOCK} (fp32 / bf16 / bf8b, both DEST
    widths): 252 passed, 60 xfailed, 0 failed.
  - Unit: w_split + decode_perf + block_prefill_perf + sharded_tile_gamma: 155 passed, 19 skipped.
- Issues encountered:
  - The first version wrote full tiles per owner and was slower (see Lever 3).
  - The reader first built the scaler before block 0's x reads, which made interleaved decode +4%. Moving
    it after the reads fixed that.
  - A core with no blocks skips the scaler pop.
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_block_prefill_perf.py`. The 3 BLOCK
  prefill + 2 BLOCK 640-row `_PERF_BASE` geometries × {gamma, no gamma}, PCC-gated, for `--profile`.
