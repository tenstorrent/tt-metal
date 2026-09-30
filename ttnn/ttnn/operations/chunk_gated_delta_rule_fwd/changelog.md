# Changelog: chunk_gated_delta_rule_fwd

## Phase 0 — Core Implementation
- **Date**: 2026-09-30
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  One `ttnn.generic_op` dispatch, regime R1. Stage P does item-parallel prep over `(bh, chunk)`. Stage S
  is a V-split sequential state scan with the state resident in L1. Stage E does item-parallel output
  assembly. The stages are sequenced by segmented semaphore handoffs.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], layout=[TILE], state_mode=[no_h0, with_h0],
  chunk_size=[32, 64], seq_alignment=[chunk_aligned, chunk_ragged], head_dims=[square, wide_v]
  (= TARGET). EXCLUSIONS: none. INVALID: none.
- **Accuracy achieved** (worst over 4 shapes × 2 states × 6 outputs, measured by
  test_chunk_gated_delta_rule_fwd_precision_baseline.py):
  - fp32: PCC ≥ 0.9999980, rel-RMS ≤ 4.4e-3, max_abs_err ≤ 4.7e-2 (on `g_cumsum`, whose |values| reach ~100). tf32-limited, slight low bias.
  - bf16: PCC ≥ 0.9999946, rel-RMS ≤ 3.3e-3, ULP p99 ≤ 7.
- **Golden suite at Phase 0**: 90 / 90 tests passing. verifier_report.json: 76 supported_pass,
  0 supported_fail / xpass_drift / xfail_wrong_mode, 0 xfail_expected, 14 no_axes_found (the
  non-registry numerics regression tests, passing).
- **Issues encountered** (verifier fixes):
  - Reader constant-tile build: per-lane non-inlined stores replaced by face-pattern stores + NoC copies. This took 47 µs off every core's start, e.g. 76 → 56 µs on the minimal shape and 1.07–1.12× on mid-size shapes.
  - Gate gather now uses one read barrier per item instead of one per token tile (perf-neutral).
  - Compile-time guards on `GATHER_STAGE_TOKENS` / `GATHER_DEPTH`.
  - `validate()` now refuses sharded inputs and sharded `memory_config` (documented mechanism cap).
  - `l1_ledger.md`: closed form corrected to the as-built quanta; the missing disjoint-lifetime justifications are recorded; a verifier audit and measurement section was added.
  - Environment: `TT_MESH_GRAPH_DESC_PATH` must be unset on this p300a box.
- **Perf baseline (post-fix, device kernel µs)**:
  - LOOSE (1,4096,16,128,128) c64: 4425 (bf16 / with_h0), 5341 (fp32 / no_h0).
  - (1,256,32,128,128) c64 fp32: 813.
  - (4,128,16,64,64) c32 fp32: 339.
  - (1,256,4,128,256) c64 bf16: 344.
  - (1,2048,2,128,128) c64 fp32: 472.
  - (1,1000,4,128,128) c64 bf16: 331.
  - (1,32,1,32,32) c32: 56.
- **Tests added**:
  - test_chunk_gated_delta_rule_fwd.py (acceptance, planner).
  - test_chunk_gated_delta_rule_fwd_precision_baseline.py.
  - test_chunk_gated_delta_rule_fwd_extended.py (sharded-output refusal, fp32 + 16-bit DEST refusal, bf16 16-bit DEST path).
  - test_chunk_gated_delta_rule_fwd_perf.py (device-perf harness: perf target + guard set, run under `--profile`).
