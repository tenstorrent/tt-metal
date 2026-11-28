# Changelog: g6_zeros

## Phase 0 — Core Implementation
- **Date**: 2026-11-28
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier). Regime R1, bank-affine coalesced zero runs:
  - one writer kernel (BRISC, NoC1);
  - 104 cores = 8 banks × `cores_per_bank` 13;
  - each core writes one contiguous run of bank-rows of one DRAM bank, from a ≤16 KiB core-local zero source, ending with one write barrier.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE]. No op-specific axes.
- **Accuracy achieved**: exact. PCC=1.0, max_abs_err=0.0, rms_err=0.0, max ULP 0, no -0.0, measured on 2 supported shapes via test_g6_zeros_precision_baseline.py. bf16 p4 and bf8b p1 are also bit-exact past `validate()` (probe_031).
- **Golden suite at Phase 0**: 13 / 19 cells passing (per `verifier_report.json`). The other 6 are `xfail_expected` (bf16 / bf8b, which go to Refinement 1). 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode.
- **Device kernel time** (default plan): p1 1,092,860 ns (target ≤ 837,190 ns, Refinement 2), p2 47,114 ns, p3 24,271 ns, p4 1,807 ns.
- **Issues encountered (verifier fixes)**:
  - `g6_zeros_plan()` no longer allocates a device tensor. A shared host derivation, `_plan_inputs`, now feeds both the plan report and dispatch, and is cross-checked against the buffer on a cache miss.
  - The descriptor-cache key is now the plan input `(dtype, N, aps, grid, B)` and no longer uses `id(device)`.
  - The stale perf harness was rewritten as a `cores_per_bank` sweep.
- **Tests added**: test_g6_zeros.py (acceptance, planner), test_g6_zeros_precision_baseline.py, test_g6_zeros_perf.py (knob sweep harness).
