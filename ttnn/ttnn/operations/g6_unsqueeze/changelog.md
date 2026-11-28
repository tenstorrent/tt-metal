# Changelog: g6_unsqueeze

## Phase 0 — Core Implementation
- **Date**: 2026-11-28
- **What was done**: initial implementation via the incremental pipeline (planner → implementer →
  verifier).
  - Face-row move: 64 B DRAM reads of input row 0 per tile half, an in-place RISC scatter of row → column
    0, and 1024 B face writes.
  - `w_tile` split across cores; tile halves split across NCRISC/BRISC; no compute kernel.
  - `resident` + `streaming` regimes.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE] (= TARGET); no shape-derived or op-specific
  registry axes. Validation accepts only `dim ∈ {r, -1}`, rank 1–3, one logical row, DRAM interleaved.
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0, rms_err=0, ULP=0, got/true ratio = 1.0 at
  p5/p50/p95. Measured on 4 shapes ((1,32), (1,256), (1,512), (1,1,4100)) with special values and a NaN
  payload, via `test_g6_unsqueeze_precision_baseline.py`.
- **Golden suite at Phase 0**: 20 / 20 passing with `EVAL_CAPTURE_L1=0` (`verifier_report.json`:
  supported_pass 9, no loud categories). With the default L1 capture, 10 / 20: harness artifacts only (9
  validation refusals have a zero allocation peak; the trace case runs with the program cache disabled).
  See verification_report.md.
- **Device time**: p1 ≈ 1.35–1.45 µs (16 cores), p2 ≈ 1.35–1.40 µs (8 cores); stock ttnn is 3.32 µs.
- **Issues encountered / verifier fixes**:
  - BB-07: added a host-side ProgramDescriptor cache (`get_program_descriptor`). It is keyed on shape,
    dtype, layout, placement, grid and all knobs; a hit patches only the buffer addresses.
  - DRY: the 64 B read alignment is now a single host constant, passed as CT arg 8.
  - Removed the unused `cb_face_stage` template parameter of `FaceMover`.
  - `l1_ledger.md`: the footprint and cap expressions now carry `halves_per_risc`; the unmeasured
    cheapest-traffic deferral is folded into Refinement 2.
- **Tests added**:
  - `test_g6_unsqueeze_precision_baseline.py`.
  - `test_descriptor_cache_patches_addresses` in `test_g6_unsqueeze_knobs.py`.
  - Existing: `test_g6_unsqueeze.py`, `test_g6_unsqueeze_knobs.py`.
