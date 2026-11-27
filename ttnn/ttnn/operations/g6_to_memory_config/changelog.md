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
