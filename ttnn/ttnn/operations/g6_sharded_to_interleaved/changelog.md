# Changelog: g6_sharded_to_interleaved

## Phase 0 — Core Implementation
- **Date**: 2026-11-28
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  - Regime `owner_writes`: one writer kernel (BRISC/NoC1) per shard-grid core walks its resident shard rectangle and issues one `noc_async_write_page` per tile to `first_page + r*C + c`, then one `noc_async_write_barrier`. The shard is read through the zero-copy `cb_input_shard` (`cb_descriptor_from_sharded_tensor`). The slot stride is the full `shard_col_tiles`.
  - Shard ↔ core order comes from `ttnn.corerange_to_cores(grid, row_wise=…)`. Grid cores with no shard exit immediately.
  - Knobs (`Knobs`, single source of truth): `blocks_per_shard = 1`, `writer_lanes = 1`.
  - Program cache keyed on shape / dtype / scheme / shard spec / buffer / knobs, with `custom_program_hash`. `last_plan()` and `roles()` cover BB-08 and BB-21; zones `S2I_STORE_BLOCK` / `S2I_DRAIN`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], in_scheme=[width_sharded], buffer=[dram]; EXCLUSIONS none
- **Accuracy achieved**: bit-exact. PCC=1.000000, max_abs_err=0, rel_rms_err=0, max ULP=0, 0 bit mismatches. Measured on 4 cases (p2 and p1, each on its production and other shard spec; p1 pre-claim with SUPPORTED widened) via `test_g6_sharded_to_interleaved_precision_baseline.py`.
- **Golden suite at Phase 0**: 13 / 17 cells passing (13 supported_pass, 4 xfail_expected, 0 supported_fail / xpass_drift / xfail_wrong_mode), per `verifier_report.json`
- **Perf baseline** (Blackhole, kernel ns median of calls 3–12; replay ns/op over 50 × 16-call trace):

  | Case | Kernel ns | Replay ns/op |
  |------|-----------|--------------|
  | p2 → DRAM, 1 lane | 1429.5 | ≈ 2820 |
  | p2 → DRAM, 2 lanes | 1319 | ≈ 2948 |
  | p1 → L1, 1 lane | 1939 | ≈ 5957 |
  | p1 → L1, 2 lanes | 1564.5 | ≈ 6168 |

- **Issues encountered**: no drift and no kernel defect. Verifier fixes:
  - the tile-edge literal in `validate()` was routed through the descriptor's `TILE`;
  - three stale test files left by an earlier interrupted run were deleted (they imported non-existent APIs and broke collection);
  - the ledger got a note on the knobs' 0 B L1 cost.
- **Tests added**: test_g6_sharded_to_interleaved.py (planner), test_g6_sharded_to_interleaved_trace_replay.py, test_g6_sharded_to_interleaved_precision_baseline.py, test_g6_sharded_to_interleaved_perf.py
