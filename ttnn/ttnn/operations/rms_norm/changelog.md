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
