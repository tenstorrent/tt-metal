# Changelog: to_memory_config

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier),
  for `quasar` on ttsim. One reader + one writer kernel per engaged cluster, one staging ring `dfb_pages`
  with implicit sync on both ends, regimes R-in / R-out / R-bal as core tables over the same kernels,
  `block_pages` derived once in `ring_geometry` (the whole per-core assignment when it fits the ring
  budget and the 255-entries-per-txn-id cap, else `BUFFER_DEPTH = 2` blocks). Thread plan `(1, 0, 1)`:
  no compute kernel (the op computes nothing; R1 forbids unpacking bytes). The design's risk "DM→DM ring
  with implicit sync on both ends is unexercised" did not bite.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR], memory_config=[DRAM, L1],
  sharding=[interleaved, height], to=[DRAM, L1, L1/height], alignment=[tile_aligned, w_non_aligned,
  h_non_aligned], rank=[2, 3, 4] (= spec PHASE0)
- **Accuracy achieved**: PCC=1.000, max_abs_err=0.000, rms_err=0.000, 0 ULP, got/true ratio exactly 1
  (measured on 4 shapes via test_to_memory_config_precision_baseline.py) — bit-exact
- **Golden suite at Phase 0**: 558 / 558 supported cells passing (558 float32 cells xfail as expected,
  372 INVALID skipped, X1 / X2 rejections pass; 0 loud categories, per `verifier_report.json`)
- **Issues encountered**: no code fix was needed in verification. Findings queued: the height-shard
  owner side is not zero-copy (Refinement 1); single-txn-id rings serialize reads and writes on some
  R-bal cores (Refinement 3); R-bank is reachable after all (Refinement 4). L1 ledger given the
  verifier's disposition of its two deferrals.
- **Tests added**: test_to_memory_config.py (harness acceptance, 55), test_to_memory_config_regimes.py
  (47: R-bal / R-in / R-out pinned, reshard 2→1 and 1→2 both layouts, COL_MAJOR + partial shard,
  depth-2 multi-block, `(2, 0, 2)` plan, one device session with `ttnn.to_layout` after),
  test_to_memory_config_precision_baseline.py (4)
- **Structural counts at Phase 0** (no timing claimed on this backend): DRAM crossings 1 per tensor side
  in DRAM; mesh hops per page 1 (R-in / R-out), 2 when an L1-interleaved side is in R-bal; one
  core-local ring copy per page on the shard side; ring txn-id slices 1 on 6 R-bal configurations of
  INPUTS × layout.
