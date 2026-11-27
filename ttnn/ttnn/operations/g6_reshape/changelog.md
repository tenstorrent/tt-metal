# Changelog: g6_reshape

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier, two verifier passes). Output-stationary face-row gather: one `generic_op`, one data-movement kernel source run as two movers per core (RISCV_1/NoC0, RISCV_0/NoC1), gather → await → permute → store per block. Regimes: `tile_face_row` and `tile_element_run`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE_LAYOUT], DRAM interleaved, rank 1..4 on both sides, any last two dims. The second verifier pass added float32: the tile kernel already moved 4-byte elements bit-exactly, and TILE is float32's production layout.
- **Accuracy achieved**: bit-exact. PCC=1.000000, max_abs_err=0, mean_abs_err=0, rel_rms_err=0, max ULP=0, got/true ratio exactly 1.0. Measured on 5 shapes with `test_g6_reshape_precision_baseline.py`: 3 bf16 `tile_face_row`, 1 bf16 `tile_element_run`, 1 fp32 `tile_element_run`.
- **Golden suite at Phase 0**: with `EVAL_CAPTURE_L1=0`, 28 / 28 supported cells pass, out of 59 total (25 `xfail_expected`, all int32/uint32 ROW_MAJOR; 6 regression tests without axes, all passing). In the graded default run, 27 / 28 pass; the trace cell and 4 validation regression tests fail for harness reasons H1/H2 (`verification_report.md`). Both runs show 0 xpass_drift and 0 xfail_wrong_mode.
- **Issues encountered**: none in the kernel. Verifier fixes, first pass:
  - one sizing helper feeds the BB-05 cap, the CB sizes and the plan report;
  - one `MOVERS` placement table;
  - new `MAX_BLOCK_ATOMS` knob;
  - knobs added to the descriptor-cache key (BB-07);
  - tile-table `--dev` ASSERT;
  - `max_blocks_per_mover` in `describe_plan`;
  - dead constant removed.

  Second pass:
  - float32 added to SUPPORTED on probe + golden evidence (7 cells xfail → pass);
  - plan-CB tile table sized by `block_atoms` instead of `block_atoms·atom_rows`. The ledger had claimed the wrong bound; bf16 L1 peaks drop, e.g. 12672 → 12416 B on `(1,1,16,4608)→(16,1,4608)`;
  - ledger and design CB table corrected;
  - acceptance and precision tests made dtype-generic, with fp32 FLUX, fp32 special-value and fp32 multi-block knob coverage.
- **Tests added**:
  - `test_g6_reshape.py`: acceptance, including `test_block_knobs` (multi-block, depth-2, one-mover and one-core paths, bf16 and fp32) and `test_float32_*`;
  - `test_g6_reshape_debug.py`;
  - `test_g6_reshape_precision_baseline.py`.
- **Perf reference (profiler build, device kernel ns)**:
  - bf16 `(16,1,5120)→(1,1,16,5120)` 8783, `(1,1,16,4608)→(16,1,4608)` 6974.
  - fp32 FLUX cells 1.8–2.2 µs on 110 cores.
  - Small bf16 cells ≈ 2.0 µs on 8–48 cores.

  The mover is RISC-V-cycle-bound, about 250 ns per output face-row (zone profile in `verification_report.md`).
