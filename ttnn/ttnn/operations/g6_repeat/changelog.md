# Changelog: g6_repeat

## Phase 0 — Core Implementation
- **Date**: 2026-11-26
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime 1, `column_split_row_broadcast`: one data-movement kernel per core.
  - Each kernel loads the row-0 faces of its x tiles into the out-tile slots, replicates row 0 in
    place by NoC loopback, and writes each out page once.
  - Out tile columns are split over `min(Wt, grid)` cores.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[h_non_aligned]. Contract: x `(1,1,1,W)`, `W % 32 == 0`, reps `[1,1,R,1]`, `1 <= R < 32`, DRAM interleaved.
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0.0, mean_abs_err=0.0, rel_rms=0.0, max ULP=0, got/true ratio exactly 1.0. Measured on 4 shapes (W/R = 256/16, 64/17, 96/31, 9600/5) via `test_g6_repeat_precision_baseline.py`.
- **Golden suite at Phase 0**: 5 / 6 registry cells passing (`verifier_report.json`), plus 3 / 3 regression cases.
  - The one `supported_fail`, `poisoned_padding_nan`, is a golden-harness precondition failure:
    `ttnn.fill_implicit_tile_padding(x, nan)` lands +inf (`0x7F80`), and the case fails before the
    op runs.
  - The op is bit-exact with NaN-payload padding in the acceptance test.
- **Device time (incidental, golden rows)**: p1 ~1.76 µs on 8 cores, one program, 2048 B L1 per core.
- **Issues encountered**:
  - Verifier fixes:
    - The kernel's face-row stride now reuses CT `row0_faces_bytes` instead of re-deriving it (DRY).
    - The final write barrier is zoned (BB-21).
    - The host L1 NoC alignment comes from `ttnn.get_l1_alignment()`.
    - The acceptance NaN-padding cases are poisoned through a same-buffer `(1,1,32,W)` view instead
      of `fill_implicit_tile_padding`.
    - The ledger's page-size source is corrected.
  - Host environment: `/opt/openmpi-v5.0.7-ulfm` (the `libtt_metal.so` RUNPATH) is missing, so
    `import ttnn` fails on `MPIX_Comm_shrink`. Device runs used an uncommitted LD_PRELOAD stub
    (`generated/mpix_stub/`).
  - The refinement queue is empty. TARGET == SUPPORTED, and the only perf-flagged case (8 tiles)
    is below the perf-rules 32-tile floor.
- **Tests added**: `test_g6_repeat.py` (acceptance, 34 cases), `test_g6_repeat_debug.py`, `test_g6_repeat_precision_baseline.py`.
