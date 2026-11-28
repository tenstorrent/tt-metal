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

## Refinement 1 — Claim the Panoptic production contract: bfloat8_b, height_sharded, L1 output
- Date: 2026-11-28
- What was done: Widened `SUPPORTED` to equal TARGET: dtype += bfloat8_b, in_scheme += height_sharded, buffer += l1. EXCLUSIONS stays `[]`. No kernel or descriptor change: the owner-writes path already moves raw page bytes (`buffer_page_size()` = 1088 B for bf8b, shared exponents copied verbatim), computes the shard origin on the host for either scheme, and takes the output buffer type from `TensorAccessorArgs(output)`. The input is still consumed through the zero-copy `cb_input_shard`.
  - Reused: the whole Phase 0 owner_writes path. Added: three SUPPORTED values.
  - Tests: removed the `monkeypatch` SUPPORTED widening (and the now-unused `op_module` / `sys` imports) from `test_..._precision_baseline.py` and `test_..._perf.py`. The acceptance helper `_assert_bits_equal` hard-coded `torch.bfloat16`, but bf8b's `to_torch` decodes to `torch.float32`, so it now bit-compares at the returned width (int16 or int32 view).
- Accuracy achieved: bit-exact. PCC=1.000000, relative RMS=0, max ULP=0, 0 bit mismatches on p2_production and p2_other_spec ((1,5120) bf16 width → DRAM) and on p1_production and p1_other_spec ((1,128,256,32) bf8b height → L1).
- Golden test progress: 17/17 passing (was 13 supported_pass + 4 xfail_expected).
- Issues encountered: the acceptance test's dtype assertion only accepted bf16 (a test-side bug, not an op defect).
- Tests added: none new. Unit acceptance file: 7 passed, 0 xfailed (the file has 7 cases; the "8 passed" in Done-when assumed one more). Precision baseline: 4/4 bit-exact. Perf eager: 4/4 exact.

## Refinement 2 — Speed up the perf-flagged profile: kernel-side write issue
- Date: 2026-11-28
- What was done: Three new `Knobs`, single source of truth, all in the cache key and `last_plan()`. Each is trivial by default (the Phase 0 plan):
  - `bank_stagger` (catalog `bank_stagger`): each lane's slot walk starts at the host-derived `start_slot = (first_page // num_output_banks) % lane_slots` (new RT arg 6) and wraps. On p1, pages 10k..10k+9 meant the 103 cores hit only 11 of the 110 L1 banks at each step; with the stagger they hit 103 distinct banks.
  - `dual_noc`: the lane's kernel runs in `DM_DYNAMIC_NOC` and alternates its page writes between its own NoC and the other one, then drains both. One RISC uses both NoCs' links without a second kernel instance.
  - `lane_nocs` (catalog `noc_placement`): explicit NoC per lane through `DataMovementConfigDescriptor`; the default `(1, 0)` matches the old Writer/Reader configs.
  - Host predicate `select_knobs(buffer_type, num_pages)`: an L1 output with ≥ `L1_OUTPUT_MIN_PAGES` (512) pages uses `L1_OUTPUT_KNOBS = Knobs(bank_stagger=True, dual_noc=True)`, everything else uses `DEFAULT_KNOBS`. `_knobs=None` (the default) means auto.
  - Reused: the single owner-writes kernel and descriptor. Added: a rotated walk (one division per lane), a dual-NoC CT switch, one RT arg, the predicate.
- Perf (Blackhole, kernel ns median of calls 3–12; replay ns/op over 50 × 16-call trace):

  | Case | Plan | Kernel ns | Replay ns/op |
  |------|------|-----------|--------------|
  | p1 bf8b → L1 (flagged) | before (Phase 0) | 1939–1960 | 5952–5960 |
  | p1 | **auto = dual_noc + stagger** | **1297–1307** | **5883–5887** |
  | p1 | lanes2 / lanes2+stagger / dual / stagger / NoC0 | 1570 / 1390 / 1710 / 2067 / 2397 | — |
  | p2 bf16 → DRAM (guard) | auto = default | 1442–1487 (noise vs 1429.5) | 2819–2822 |
  | p2 → L1 (guard) | auto = default | 992–1002 | 2782–2787 |

  The bottleneck was NoC congestion, not RISC issue cost. With one lane the issue zone took 1198 cycles p50 (2215 max) against 292 uncontended. Under the new plan the per-core end spread is tight (max 1821 vs 2812 cycles) and the remaining time is evenly spread NoC/bank throughput (~98 cycles/write) plus ~230 cycles of launch skew. On DRAM outputs, NoC0 / dual hurt (NoC0 doubles p2), and dual on p2 → L1 is a flat kernel and +1.5% replay; both stay on the Phase 0 plan.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0) on p1 production / other spec and p2 → DRAM / → L1, under every knob variant in the perf harness.
- Golden test progress: 17/17 passing.
- Issues encountered: the Tracy capture tool crashes intermittently (exit 3, no .tracy); re-run. `probes/zone_timeline.py` had stale zone names (fixed).
- Tests added: `test_g6_sharded_to_interleaved_plan_selection` (3 cases) in the acceptance file; perf harness gains an `auto` plan, the lever variants and a p2 → L1 guard case.
