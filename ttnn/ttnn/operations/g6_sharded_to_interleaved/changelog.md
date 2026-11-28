# Changelog: g6_sharded_to_interleaved

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  Regime `owner_writes`: one writer kernel per shard core (BRISC / NoC1) writes its resident shard
  rectangle straight to the interleaved pages. Each tile is one non-posted NoC write, there are no reads,
  and each kernel ends with one barrier. `cb_input_shard` is zero-copy on the input shard. No reader or
  compute kernel. Knobs live in `Knobs` (`num_write_lanes=1`, `block_tile_rows/cols=None` = the whole lane
  rectangle). The memoized descriptor is patched per call (output address, shard CB binding).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], in_scheme=[width_sharded], buffer=[dram]
- **Accuracy achieved**: bit-exact. PCC=1.000000, max_abs_err=0, rel_rms_err=0, max ULP=0, 0 bit
  mismatches, measured on 4 cases (p2 and p1, each on its production spec and its other spec) via
  `test_g6_sharded_to_interleaved_precision_baseline.py`. The p1 cases go through `_dispatch` until
  Refinement 1 claims them.
- **Golden suite at Phase 0**: 13 / 17 cells passing (13 supported_pass, 4 xfail_expected, 0
  supported_fail / xpass_drift / xfail_wrong_mode) per `verifier_report.json`
- **Issues encountered**: no drift. The verifier found no code defect to fix. One design defect (perf):
  the `bank_stagger` analysis missed that p1's L1 destination is hit by only 11 of 110 banks per write
  step. Filed as Refinement 2. Superseded debug/perf test files were removed.
- **Tests added**: test_g6_sharded_to_interleaved.py, test_g6_sharded_to_interleaved_impl_debug.py,
  test_g6_sharded_to_interleaved_trace_replay.py, test_g6_sharded_to_interleaved_precision_baseline.py,
  test_g6_sharded_to_interleaved_perf.py

### Phase 0 measurements (Blackhole p150b, 11 x 10 grid)

Implementer ablations (device kernel duration, single fresh runs):

| Case | default (1 lane) | `num_write_lanes=2` | all writes stubbed | address math only |
|------|------------------|---------------------|--------------------|-------------------|
| p2 (1,5120) bf16 width -> DRAM, 32 cores | 1500, 1419 ns | 1561, 1524 ns | 590 ns | 641 ns |
| p1 (1,128,256,32) bf8b height -> L1, 103 cores | 2118, 2118 ns | 1879, 1964, 1917 ns | 807 ns | 919 ns |

A per-core start-row stagger that took a modulo on every row measured slower on p1 (2282, 2360 ns), so it
was not kept.

Verifier baseline (median of calls 3–12 of 12; trace replay = host clock over 50 x 16 replays):

| Case | lanes | kernel ns | trace replay ns/op |
|------|-------|-----------|--------------------|
| p1 | 1 | 2110 | 5950–5953 |
| p1 | 2 | 1875 | 6224–6233 |
| p2 | 1 | 1494.5 | 2843–2847 |
| p2 | 2 | 1563.5 | 2888–2893 |

Two lanes win kernel time on p1 but lose trace-replay time (the second kernel's dispatch cost), so the
default stays at 1 lane. p1 is above the 2.05 us stock incumbent, so the flagged goal is open
(Refinements 2 and 3).
