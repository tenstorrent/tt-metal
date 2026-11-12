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
