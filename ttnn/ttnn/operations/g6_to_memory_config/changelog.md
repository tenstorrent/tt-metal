# Changelog: g6_to_memory_config

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Owner-move design: `pull_shard`, `pull_bank`, `push_shard`, `push_bank` and `bounce` regimes. Each page is moved by a core that owns one side of it (one NoC transfer per page), or by `bounce` for DRAM→DRAM (two transfers, the minimum). One `generic_op` program per call, cached per placement pair, with the base addresses patched on a cache hit.
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16], layout = [TILE], rank = [2, 3, 4], alignment = [tile_aligned, h_non_aligned];
  - in_scheme / out_scheme = all four schemes; in_buffer / out_buffer = [dram, l1];
  - EXCLUSIONS = [{out_scheme: width_sharded, out_buffer: dram}].
- **Accuracy achieved**: bit-exact. PCC = 1.000000, max_abs_err = 0, rel_rms_err = 0, max ULP = 0, measured on 4 shapes (one per regime: Qwen 2058, 2105, 2071, Qwen 2077 shape) via `test_g6_to_memory_config_precision_baseline.py`. The 19 bf8b/bf4b cells are also bit-exact with the gate lifted in-process.
- **Golden suite at Phase 0**: 91 / 117 registry cells passing; 26 xfail_expected (bf8b, bf4b, DRAM width-sharded target); 0 supported_fail / xpass_drift / xfail_wrong_mode. Plus 88 / 88 regression tests, for 179 / 205 total (per `verifier_report.json`).
- **Device perf baseline**: PERF FOCUS Qwen 2183 (bf4b, gate lifted) runs in 307–318 µs. That beats stock's 348.36 µs but misses the lead's ≤ 276.48 µs; the target is Refinement 2. Qwen 2078 `pull_bank` runs in 53.5 µs (1.18× DRAM roofline). Full table in `verification_report.md`.
- **Issues encountered**: no drift. Verifier fixes:
  - DRY: one source for the per-kernel `movers_per_owner` / `mover_index` CT args;
  - removed the unreferenced compute-kernel stub;
  - corrected the stale bounce-CB maximum in `l1_ledger.md` (429696 B).
- **Tests added**: `test_g6_to_memory_config.py` (acceptance), `test_g6_to_memory_config_perf_knobs.py`, `test_g6_to_memory_config_precision_baseline.py`.

## Refinement 1 — Block formats and DRAM width-sharded targets
- **Date**: 2026-11-27
- **What was done**: Widened `SUPPORTED["dtype"]` to `[bfloat16, bfloat8_b, bfloat4_b]` and emptied `EXCLUSIONS` (the DRAM width-sharded target is gone), so `SUPPORTED == TARGET` on every axis. Zero kernel or descriptor changes: `page_size` from `buffer_aligned_page_size()` is the only dtype-dependent quantity and `cb_bounce_pages` already takes the input dtype as its data format. DRAM width-sharded targets run on the unchanged `bounce` regime through the target `TensorAccessor`. Reused: all five regimes, both kernels, the bounce CB and the BB-05 clamp. Added: nothing on the device side.
- **Accuracy achieved**: bit-exact (`torch.equal`; `helpers.TOLERANCES` is `"exact"` for every dtype). PCC = 1.0, relative RMS = 0, on Panoptic 11 bf8b (`push_shard`), Qwen 2178 bf4b (`bounce`, DRAM width-sharded on 8 banks), all 19 block-format golden cells and all 12 loose cases.
- **Golden test progress**: targeted slice 31 / 31 (12 bf8b + 7 bf4b `test_op` cells, 12 `test_op_loose` cases incl. the 7 formerly xfail). With EXCLUSIONS empty and SUPPORTED equal to TARGET, every cell is now a supported cell: expected 117 / 117 registry cells + 88 / 88 regression = 205 / 205 (the harness's full run is the confirmation of record).
- **BB-05**: every new cell is under the 1347776 B budget; max `l1_op_bytes` = 429696 B (Qwen 2183, `bounce_block_pages` 373). bf8b cells land on `push_shard` / `pull_bank`, bf4b cells on `bounce` with 110 movers (`probes/probe_016.py`).
- **Device perf**: Qwen 2183 (PERF FOCUS, bf4b `bounce`) 311.2 / 317.7 / 313.7 µs device kernel duration (median 313.7 µs) vs the Phase 0 band of 307–318 µs: no regression. Still 1.53× its 204.8 µs DRAM roofline, above the lead's 276.48 µs bound; the coalesced-write lever (`bounce_dram_shard`) is Refinement 2.
- **Issues encountered**: None. A previous run was interrupted after the code and test edits; this run re-verified them on device and closed the refinement out.
- **Tests added**: `test_g6_to_memory_config.py::test_block_format_case` (Panoptic 11 bf8b `push_shard`, Qwen 2178 bf4b `bounce`, regime and page size pinned); `test_refused_cells` re-targeted to float32 → `UnsupportedAxisValue` (no `ExcludedCell` remains); `test_g6_to_memory_config_precision_baseline.py` gains one bf8b and one bf4b exact case. Acceptance + precision baseline: 32 / 32.

## Refinement 2 — Speed up the perf-flagged profile: coalesce writes into the DRAM width-sharded target
- Date: 2026-11-27
- What was done: Knob-turn on the `bounce` regime; still one program per call. Each knob has one source of truth in the descriptor module and is part of the cache key.
  - Shipped, default on: `BOUNCE_WRITE_RUN_PAGES = 0`. The bounce writer issues one NoC write per run of pages contiguous in the target (a 64-page, 36 KB shard row on 2183) instead of one per page. Run lengths come from `num_contiguous_pages` on the target accessor, so the ragged last shard of 2184 is handled by the accessor. Runs never cross a CB block; adjacent runs merge by address. Interleaved targets still write per page.
  - Parked, default off (measured a loss): `BOUNCE_TARGET_ORDER`. Movers take contiguous ranges of a sharded DRAM target's storage order, `[storage_slot, core_pages]`, mapped on the host by `storage_slot_of` and walked on the device through `shard_pages` (`g6::for_each_storage_page`). Each mover then writes one bank, and movers are interleaved over the shards.
  - Parked, default off (noise): `BOUNCE_BANK_STAGGER`. Mover i's page-order walk is rotated so it starts on target shard i mod S (`g6::for_each_page_segment`).
  - `ABLATE_DEFINES` (default empty) compiles out the read or write payload for perf ablation.
  - Reused: the `bounce` regime, its CB, its block/depth knobs and the BB-05 clamp. Added: the `BounceFill` / `BounceDrain` block cursors shared by both walks, CT arg 8 `bounce_order`, a third bounce RT arg (`rotation`), and `plan_of` field `bounce_order`.
  - Removed the env-var debug hooks an interrupted earlier attempt had left in the descriptor.
- Device perf (Tracy, device kernel ns, median of 3, same session; per-page writes → coalesced runs):
  - 2183: 325.5 → 294.8 µs (Refinement 1 recorded 313.7 µs). That is 1.44× the 204.8 µs roofline, above the 276.48 µs goal and below stock's 348.36 µs.
  - 2184: 261.6 → 233.0; 2179: 178.6 → 157.7; 2182: 67.0 → 58.8; 2178: 26.3 → 25.2.
  - bf16 DRAM→DRAM interleaved: 30.5 → 31.2 (noise; no runs exist on an interleaved target).
  - Guard set, one per regime, HEAD vs new (ns): pull_shard 2933/3021, pull_bank 2881/2859, push_shard 3047/3144, push_bank 1658/1541. All within noise; the non-bounce kernel code is unchanged.
- Levers measured and not kept on:
  - storage-order assignment: 555 µs with row-clustered movers, 375 µs with movers interleaved over shards (vs 295–302 µs). Writes alone 211 vs 184 µs, reads alone 180 vs 165 µs. Per-core maps show the movers feeding one bank starving each other on NoC1.
  - bank stagger: 306.8 vs 302.2 µs. A host-side stagger of the read start banks also measured as noise (300.7 µs).
  - `bounce_depth` 3/4/8: 316/296/306 µs.
  - write caps of 8/28 pages: 295.6/296.6 µs. Past ~4.6 KB per write, size no longer matters.
- Bottleneck now (2183 ablation): reads alone 165 µs, writes alone 184 µs, sync floor about 1 µs, full 295–302 µs. Per-core read time ranges from 30 to 290 µs by grid row, but the aggregate read rate is flat at about 165 GB/s for the whole call. The op is DRAM-efficiency-bound on 576 B reads mixed with coalesced writes, not on per-core balance; bf16 with 2 KB pages reaches about 515 GB/s on the same path. The remaining lever is read coalescing from the interleaved source, which is Refinement 3's scope.
- Accuracy achieved: bit-exact (`torch.equal`, gate `"exact"`): PCC = 1.0, relative RMS = 0. Covered: every bf4b golden cell, all 12 loose cases, and the new knob matrix (2184 ragged width-sharded, 2178, bf16 interleaved, ragged DRAM height-sharded bf16, ragged DRAM block-sharded bf8b × 5 knob variants).
- Golden test progress: targeted slice `-k "BFLOAT4_B or test_op_loose"` 19 / 19. Unit directory 72 / 72 (acceptance + perf knobs + precision 47, bounce knob matrix 25).
- Issues encountered:
  - A first version of the drain computed a sharded `get_noc_addr` per page and regressed 2183 to about 323 µs. Feeding the drain one address per run restored the baseline.
  - Tracy capture crashed intermittently (exit 3, no CSV); those runs were repeated.
  - The NoC-estimator CLI is not built in this checkout, so the ceiling reasoning uses measured ablations.
- Tests added: `test_g6_to_memory_config_bounce_write_runs.py::test_bounce_knobs_exact` (25 cases). Probes: `probes/bench_bounce.py` (perf-only bench with knob and ablation switches), `probes/zones.py` and `probes/zone_map.py` (device-zone summaries and per-core maps).
