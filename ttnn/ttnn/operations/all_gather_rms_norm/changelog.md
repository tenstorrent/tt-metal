# Changelog: all_gather_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-19
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). R1
  mirror_resident_gather end to end: fabric plan (probe placement, balanced ring / line schedules, ports_union
  excluded from the compute grid of every chip), port reader / sender (ready fence, relay, one fused arrival increment
  per (wave, shard) entry, landing notifications), member reader / compute / writer (base-op local norm with symmetric
  partial exchange), LN pivot + Chan combine, dense assembly for `w_non_aligned`. One `ttnn.generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16, bfloat8_b], fp32_dest_acc_en=[True, False], layout=[TILE],
  alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode both, gamma_dtype=[fp32, bf16,
  none], gamma_layout=[TILE, ROW_MAJOR, none], norm_type=[rms_norm, layer_norm], beta_mode both,
  memory_layout=[INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED], cluster_axis=[0, 1], topology=[Linear, Ring],
  num_links=[1, 2] (== TARGET). EXCLUSIONS: {float32, fp32_dest_acc_en=False}, {bfloat8_b, w_non_aligned}.
- **Accuracy achieved** (fp32 DEST, 1×4 ring, worst device, 4 shapes via
  `test_all_gather_rms_norm_precision_baseline.py`): bf16 rms PCC ≥ 0.999998, max_abs ≤ 0.045, rel_rms ≤ 0.0019;
  bf16 layer_norm PCC ≥ 0.999998, max_abs ≤ 0.045, rel_rms ≤ 0.0019; fp32 layer_norm PCC 1.000000, max_abs ≤ 0.011,
  rel_rms ≤ 0.0006; bf8b rms PCC ≥ 0.99995, rel_rms ≤ 0.0097. Known gap: bf16 layer_norm with fp32_dest_acc_en=False
  rel_rms 0.005–0.014 here, 0.015–0.025 on half of the golden family (Refinement 2).
- **Golden suite at Phase 0** (hash-sampled matrix + all ring-mock / regression / loose tests, `verifier_report.json`):
  2357 supported_pass, 174 supported_fail (143 numerical-precision, 31 OOM — both queued), 644 xfail_expected,
  0 xpass_drift, 0 xfail_wrong_mode.
- **Perf** (eval ccl_perf protocol, 5 experiments, fastest chip): focus 640×384 rms 21.29 µs (roofline 7.6); indexer
  640×32 layer_norm 16.33 µs (roofline 0.6). Implementer's rounds: focus 22.23 → 21.25 µs (direct gamma reads after the
  exchange, out unit 4); indexer 22.68 → 16.16 µs (LN FPU fast path, k = 1 combine); the verifier's precision fix
  below then cost +0.05 / +0.17 µs.
- **Issues encountered / verifier fixes**:
  - FPU normalize chain truncated its fp32 intermediate to the input's 16-bit format at every dest-reuse step
    (DEST → srcA in cb_x's format): −0.6 % per-row scale bias on LN bf16, −0.3 % on rms bf16, and one borderline
    golden failure (bf16 LN 1×384 rel_rms 0.0100). Fixed in `DestReuseBcast` (srcA switched to the DEST width once per
    chain, restored once). LN bf16 rel_rms 0.0072 → 0.0019.
  - `{bfloat8_b, h_non_aligned}` exclusion dropped (measured unnecessary: 241/241 sampled cells pass);
    `{bfloat8_b, w_non_aligned}` kept (measured necessary: 238/238 inf).
  - Split-fit refusal reworded to carry `OOM` (classified as OOM, not `other`).
  - L1: `cb_moment_a` no longer allocated on the LN FPU path (dead), `cb_moment_b` = rows there, `cb_out` capped at
    the member block; the writer's cb_out ring size reads `cb_plan()` (was a restated `OUT_DEPTH·unit`).
  - DRY: one `MulScalar` reduce post-op (was two identical structs).
  - Environment: with `--dev` (watcher), re-opening a fabric mesh after any fabric traffic times out in the Ethernet
    teardown — reproduced with the `fabric_all_gather` example, so not this op. `perf_measure.py` fails with
    `ENAMETOOLONG` on its result file name; `perf_agrms.py` runs the same protocol.
  - Deferred: owner combine for k > 8 (Refinement 1), compact gamma pages (Refinement 1), zero-copy own shard,
    credited landing.
- **Tests added**: `test_all_gather_rms_norm.py` (acceptance, 19), `test_all_gather_rms_norm_precision_baseline.py`
  (20), `perf_agrms.py` (perf harness wrapper).

## Refinement 1 — L1 fit for wide rows and L1-sharded I/O: port the base op's owner combine (k > 8) and compact gamma pages
- Date: 2026-11-19
- What was done:
  - **Owner combine for k > 8** (base op R3 / Perf 2 row partial, unicast return): `_k_candidates` now runs up to
    `K_CAP_MAX = 128` (writer RT-arg bound) with the slice-aligned rule, and `select_split` prices k > 8 with the base
    op's `OWNER_MEMBER_COST · ceil(rows/k) · k`. Row r is owned by member r % k. Partials travel in ROW form (two 16-datum
    face rows into row m % 32 of gather tile m / 32 of the owner: `ceil(k/32)` gather pages per owned row instead of k);
    the owner zero-fills the never-written rows, combines, and returns column-form statistics (faces 0/2) into slot r of
    every member's `cb_rowstat` (+ `cb_shift`, `cb_kg`), one `SEM_INV` increment per row. rms: member collapse = ones-row
    × sumsqᵀ matmul, owner = `reduce_tile` COL + `transpose_dest` + the column-0 finalize (base op code). layer_norm (not
    in the base op): member finalize transposes (mean_m, M2_m) in DEST; the owner's Chan combine is centered on a pivot
    `K = bf16(mean of member 0)` (SFPU, fp32), weights n_m from a column-0 constant (`cb_wcol`), sums over members by a
    ones-row matmul of centered terms only, then `transpose_dest` → (inv_std, shift, K_g). Every perf-flagged shape
    keeps its regime (focus k = 4, indexer k = 1).
  - **Compact gamma / beta pages** (base op Perf 2): `cb_gamma` / `cb_beta` page stride = `gamma_page_layout()` (192 B
    for bf16 / fp32, + tail pages for the last page's unpacker read) instead of a whole gamma tile; loaders unchanged
    except the stride.
  - **Port stage batch fitted** to L1 (`port_batch_candidates`: 4 → 2 → 1 packets per batch); the cap stays the default.
  - **LN L1**: live-disjoint scratch CBs share one region (one CBDescriptor, several buffer indices: `cb_alias_groups`):
    moments scratch (`cb_moment_a`/`cb_moment_b` or `cb_cent`) with the owner's `cb_dev`, and `cb_mstat` with `cb_mu`.
    With fp32 DEST the LN FPU path falls back to the SFPU moments path (no block-sized `cb_cent`) when no split fits.
  - **Global semaphores no longer pin L1**: the cached (arrival, ready) pairs were created under the first call's L1
    shards and, once those were freed, set the CB budget of every later call (measured: budgets 625 KB → 101 KB → 68 KB
    as configs accumulated; the isolated cells all fit). `_relocate_pinning_sems` re-creates a pair once free L1 has
    opened above it (allocator is top-down) and keeps the higher one; a no-op in steady state.
  - Program hash now covers the fitted knobs (out unit, port batch, LN path). `FORCE_K` / `LAST_INFO` test hooks.
- Accuracy achieved (1×4 ring, owner path k = 24 / 16, worst device and slice): bf16 rms (1,1,32,2048) PCC 0.999998,
  rel RMS 0.0018; bf16 LN PCC 0.999998, rel RMS 0.0021; bf16 LN fp32_dest_acc_en=False PCC 0.99997, rel RMS 0.0095;
  fp32 LN offset 50 (1,1,32,1024) PCC 1.000000, rel RMS 0.0015.
- Golden test progress: the 31 Phase-0 OOM cells 31/31; family slices `1x1x32x2048` 840 pass / 0 fail;
  `1x1x32x1280, 1x1x32x1792, 1x1x8x1792, 1x1x640x384, 1x1x64x1000, 1x1x32x50, 4x128x47, 32x17` 5009 pass / 256 fail (all
  bf16 LN acc16 precision, Refinement 2's family; none new); ring-mock + regression 160/160.
- Perf (eval protocol, 5 experiments, fastest chip): case0 21.29 µs, case1 16.32 µs — unchanged (baseline 21.29 / 16.33).
- Issues encountered: the "OOM" budgets in the batch were mostly L1 fragmentation by the op's own persistent global
  semaphores, not the cells' footprint (each remaining cell passed alone); fixed at the source. Not done (not needed by
  any named cell): zero-copy own shard / output shard (item 3), owner-spread / root multicast return (base op perf).
- Tests added: `test_all_gather_rms_norm_owner.py` (13: forced k = 12 / 16 / 20 / 40 incl. zero-filled gather rows, two
  gather tiles, multi-row owners, rms / LN FPU / LN SFPU, 16-bit DEST, non-aligned dense assembly);
  `test_all_gather_rms_norm_debug_l1.py` (debug, kept: L1 block view across calls, owner metrics).

## Refinement 1b — L1 fit for wide rows and L1-sharded I/O (debug: fix gate violations)
- Date: 2026-11-20
- What was done: the gate's regression (92 prior-passing cells OOM, 3 `L1ProfilingError`) was order dependent: each
  cell passed alone. Root cause, measured with the harness plugins on the real 700-cell order: L1 tensors of FAILED
  cells stay alive in reference cycles (pytest's exception traceback → the test frames → input / output shards) until a
  full (gen-2) GC, which a 200k-item pytest process runs rarely. Dead shards accumulated to ~1.3 MB per bank and
  fragmented L1, so `cb_l1_budget` (free L1 below the lowest allocated buffer) dropped to 133–363 KB. Refinement 1 made
  more sharded cells run to the precision check (leaking input + output shards) instead of raising OOM early, which is
  why Phase 0 did not hit it. When such a GC finally fires inside a later DRAM cell's graph capture, the profiler's
  `size_t` L1 base wraps (`peak_total = 2^64 − 61184`) — the 3 profiler errors.
  Fix (host side, op file): after each dispatch (this call's CBs already placed) `_collect_dead_l1` runs one
  `gc.collect()` when L1 holds blocks no call argument / cached semaphore owns AND that were not already seen to survive
  a collection (live foreign tensors therefore do not re-trigger it); on the OOM path it collects with `force` and
  retries the split once. The OOM message now lists the allocated L1 blocks.
- Accuracy achieved: unchanged (no kernel change).
- Golden test progress (FULL suite, eval_test_runner, same order/plugins as the gate): 29118 passed / 1284 failed /
  0 hangs / 0 OOM / 0 `L1ProfilingError` (previous attempt 27115 / 1867 incl. 799 OOM; Phase 0 23269 / 5289).
  Regressions vs Phase 0: 0. All 1284 failures are `numerical-precision`, bf16 layer_norm with fp32_dest_acc_en=False
  (Refinement 2's family). 40 of them (1x1x17x1280 / 1x1x640x1792 HEIGHT_SHARDED, failed in Phase 0) had passed in the
  previous attempt only because leaked L1 forced a wider member split (k = 16 instead of 8 → narrower members →
  fewer tiles accumulated in 16-bit DEST): evidence for Refinement 2 that narrower members lower the 16-bit error.
- Perf (eval protocol, 5 experiments, fastest chip): case0 21.31 µs, case1 16.32 µs (unchanged).
- Issues encountered: none beyond the above; unit dir 58/58.
- Tests added: none (repro = the golden order with `EVAL_CAPTURE_L1=1 -p eval.metrics_plugin`).
