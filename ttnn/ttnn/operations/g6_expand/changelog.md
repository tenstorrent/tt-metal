# Changelog: g6_expand

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier):
  regime R1 — column-major `split_work_to_cores` over output tiles, per-core row-0 fetch of x into the
  broadcast tile slot, in-L1 replicate (RISC-V rows 0..7 + local NoC doubling), whole-tile writes split
  between writer (NoC1) and reader (NoC0). One `generic_op` dispatch, single CB, no compute kernel.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned, w_non_aligned]
  (= TARGET).
- **Accuracy achieved**: bit-exact — PCC=1.0, max_abs_err=0, mean_abs_err=0, rel_rms_err=0, bit diffs 0
  (measured on W ∈ {185, 196, 256, 512} via test_g6_expand_precision_baseline.py); special values and
  poisoned padding bit-exact in golden + acceptance.
- **Device time at Phase 0** (BH, DEVICE KERNEL DURATION): W=512 ≈ 2.38–2.51 µs (110 cores; goal
  ≤ 1.54 µs); W=256 ≈ 1.70 µs; W=196/207 ≈ 1.52 µs; W=185 ≈ 1.49 µs.
- **Golden suite at Phase 0**: 20 / 20 passing; verifier 14 supported_pass, 0 supported_fail /
  xpass_drift / xfail_wrong_mode (per `verifier_report.json`).
- **Issues encountered**: verifier fix — reader `cb_reserve_back` wrapped in its own profiler zone
  `G6E_RESERVE` (BB-21). No drift.
- **Tests added**: test_g6_expand.py (acceptance), test_g6_expand_precision_baseline.py,
  test_g6_expand_perf_bench.py (device-ns bench for the perf refinements).

## Refinement 1 — Co-tune the cheap knobs on the perf-flagged W=512 profile
- Date: 2026-11-27
- What was done: exposed the grid-size knob (`MAX_CORES`, `MIN_TILES_PER_CORE` → `_split_core_count`,
  single source; the split now runs over `num_cores_to_corerangeset(split_cores, grid, row_wise=True)`;
  `split_cores`/`max_cores`/`min_tiles_per_core` reported in `last_plan`). Co-tuned with
  `READER_WRITE_EIGHTHS`, `CORE_ORDER_STRIDE`, `REPLICATE_CPU_ROWS` via an opt-in device sweep
  (`test_g6_expand_knob_sweep.py`, `G6E_SWEEP=1`, 9-rep medians — run-to-run noise is ~±5 %).
  Chosen: `MAX_CORES=80`, other knobs unchanged (4 / 1 / 8). Device kernel time W=512: 2.49 → 2.31 µs
  median (−7 %). Guards W=256/196/207/185 run the identical program (cap binds only above 80 tiles):
  1.73 / 1.57 / 1.57 / 1.52 µs, unchanged. Findings: bank stagger cannot help (Wt=16 ⇒ a whole column
  sits on one of 8 banks; aggregate bank load already balanced); stride 3/7 and cpu_rows 2/4/16/32 were
  neutral-to-worse; 16–48 cores plateau ≈ 2.45–2.5 µs → remaining cost is the per-core fetch→replicate
  →write critical path (Refinement 2).
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0) on W ∈ {512, 256, 196, 207, 185} (every sweep
  case asserts torch.equal).
- Golden test progress: 20/20 passing
- Issues encountered: None
- Tests added: tests/ttnn/unit_tests/operations/g6_expand/test_g6_expand_knob_sweep.py (opt-in perf
  sweep tooling; skipped unless G6E_SWEEP=1)
