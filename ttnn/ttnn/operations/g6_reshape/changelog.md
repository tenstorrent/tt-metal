# Changelog: g6_reshape

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier). Output-stationary face-row gather: one generic_op, one data-movement kernel source run as two movers per core (RISCV_1/NoC0, RISCV_0/NoC1), gather → await → permute → store per block, regimes `tile_face_row` and `tile_element_run`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE_LAYOUT], DRAM interleaved, rank 1..4 on both sides, any last two dims.
- **Accuracy achieved**: bit-exact. PCC=1.000000, max_abs_err=0, mean_abs_err=0, rel_rms_err=0, max ULP=0, got/true ratio exactly 1.0. Measured on 4 shapes with `test_g6_reshape_precision_baseline.py` (3 `tile_face_row`, 1 `tile_element_run`).
- **Golden suite at Phase 0**: with `EVAL_CAPTURE_L1=0`, 21 / 21 supported cells pass (59 total: 32 `xfail_expected`, 6 regression tests without axes, all passing). In the graded default run, 20 / 21 pass; the trace cell and 4 validation regression tests fail for harness reasons H1/H2 (`verification_report.md`). 0 xpass_drift, 0 xfail_wrong_mode in both.
- **Issues encountered**: none in the kernel. Verifier fixes:
  - DRY for L1 sizing: one helper feeds the BB-05 cap, the CB sizes and the plan report.
  - DRY for mover placement: one `MOVERS` table feeds the descriptor and the plan report.
  - New `MAX_BLOCK_ATOMS` knob.
  - Knobs added to the descriptor-cache key (BB-07).
  - Tile-table `--dev` ASSERT.
  - `max_blocks_per_mover` in `describe_plan`.
  - Dead constant removed.
  - Ledger capacity expressions brought current (`align_up` rounding, knobs, measured peaks).
- **Tests added**: `test_g6_reshape.py` (acceptance, incl. the new `test_block_knobs`, which covers the multi-block, depth-2, one-mover and one-core paths), `test_g6_reshape_debug.py`, `test_g6_reshape_precision_baseline.py`.
- **Perf reference (profiler build, device kernel ns)**: `(16,1,5120)→(1,1,16,5120)` 8781, `(1,1,16,4608)→(16,1,4608)` 7015; small cells ≈ 2.0 µs on 8–48 cores. The mover is RISC-V-cycle-bound, about 250 ns per output face-row (zone profile in `verification_report.md`).
