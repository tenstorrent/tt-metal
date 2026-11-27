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
