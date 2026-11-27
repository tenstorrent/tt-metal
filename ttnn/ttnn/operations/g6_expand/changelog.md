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

## Refinement 2 — Shorten the fetch → replicate → write critical path on W=512
- Date: 2026-11-27
- What was done: reader/writer mechanism change, everything behind single-source host knobs in
  `g6_expand_program_descriptor.py` (reported in `last_plan`):
  - `HALF_TILE=1` — build only faces 0/1 in a 1 KiB slot (CB page = slot) and write every output page
    as two 1 KiB writes from it (reader and writer write shares both); drops the faces 0/1 → 2/3 copy.
  - `REPLICATE_SPLIT=1` — the writer (BRISC) fetches row 0 of face 1 on NoC1 and builds face 1 (and
    face 3 when not half) by RISC-V stores, in parallel with the reader building face 0. It hands
    off through a local counter semaphore `SEM_FACE1_BUILT` that the reader polls before its write
    share. The host gates it to the pure-RISC-V build and one block per core (no slot reuse).
  - `REPLICATE_CPU_ROWS=16` — pure RISC-V build (no local NoC copies or barriers), each column
    published as soon as it is built (finer publish).
  - Measured and parked at the old byte-identical defaults: `ROW_FETCH_REQUESTS` (1 request of
    face+row vs 2 × 64 B: flat) and `REPLICATE_NOC_FANOUT` (prefix fan-out under one barrier instead
    of doubling: worse for prefixes < 8 rows, flat at 8).
  - Device kernel ns (BH, paired A/B medians): W=512 ≈2.33 → ≈2.25 µs; W=256 1.73 → 1.52;
    W=207 1.58 → 1.42; W=196 1.57 → 1.41; W=185 1.53 → 1.37 µs.
  - Ablations (W=512): stubbing the replicate alone ≈ −0.25 µs. Stubbing fetch + replicate together
    (sync kept) gives a write-only floor of ≈1.79 µs: 512 KiB of writes take ≈1900 cyc (≈270 B/cyc),
    which is above the 1.54 µs goal. The remaining headroom is in write throughput and fetch latency
    (≈500 cyc), not on-core replicate cost.
- Accuracy achieved: bit-exact — PCC=1.0, relative RMS=0 on W ∈ {512, 256, 207, 196, 185} plus the
  multi-column/multi-block knob-path matrix (int16 view equality, poisoned x padding).
- Golden test progress: 20/20 passing
- Issues encountered: tracy-capture crashed on a few `--profile` sweep runs for W=196 (capture
  side; the test passed). W=196 was measured through the standard perf bench instead.
- Tests added: tests/ttnn/unit_tests/operations/g6_expand/test_g6_expand_replicate_paths.py
  (5 build paths × 4 geometries, including a single core with 16 columns and one with 66 columns,
  i.e. two blocks with the split gated off). test_g6_expand_knob_sweep.py was extended with
  half_tile / fetch_reqs / fanout / split fields.
