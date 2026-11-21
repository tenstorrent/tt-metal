# Changelog: softmax

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Metal 2.0 `ProgramSpec`, one `ttnn.generic_op` per call, regime `column_resident`, thread plan `(R, C, W) = (2, 4, 2)` (`THREAD_PLAN`, one host constant).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[DRAM, L1], sharding=[interleaved], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[4], dim=[-2] (= PHASE0)
- **Accuracy achieved**: PCC ≥ 0.999997, max_abs_err ≤ 2.44e-3, rel_rms_err ≤ 0.0024, every element within 1 bf16 ULP of the correctly rounded result. Measured on 6 cases (5 shapes, DRAM/L1, scale 1 and 10) via `test_softmax_precision_baseline.py`.
- **Golden suite at Phase 0**: 72 supported cells pass, 0 supported_fail, 1468 xfail_expected, 592 invalid_skipped, 0 xpass_drift, 0 xfail_wrong_mode (per `pipeline_logs/verifier_report.json`; 2,191 tests, 0 failed, 0 hangs)
- **Issues encountered**:
  - Verifier fix: rejection X1 (in-range `dim` canonicalising to −3) raised a plain `ValueError`. The 38 translated tt-metal cells calling `dim=-3` / `dim=0` landed in `xfail_wrong_mode`. It now raises the typed `UnsupportedAxisValue` (a `NotImplementedError`, allowed by X1). An out-of-range `dim` stays a `ValueError`.
  - Verifier fix: the resident-fit refusals raise `SupportRefusal`. The dead constant `NEOS` was removed.
  - Verifier fix: the L1 ledger's `cb_output_tiles` live set now states the real per-push size (1–2 tiles at fp32 half-sync DEST) and why the capacity is larger.
  - Implementer deviations (advisory parts of the design only):
    - `cb_h_mask` is always allocated (8 KB per cluster).
    - The session test uses a device-side `ttnn.to_layout` tilize as the preceding tt-metal program; `ttnn.add` is Metal 1 and does not run on Quasar.
- **Structural counts** (ttsim has no clock; no timing is claimed):
  - input DRAM crossings 1×, output 1× (+ at most C − 1 recomputed columns per op);
  - `[1,4,16,8400]`: 32 of 32 clusters, 4 NEOs each, 8–9 columns per NEO, one block per NEO;
  - SFPU inits per input tile: 0 (once per block operation);
  - per `Ht = 1` output tile: 9 `copy_tile` unpacks, 3 packs, 2 exponentials (baseline for Refinement 3).
- **Tests added**:
  - `test_softmax.py` (acceptance, generated);
  - `test_softmax_regimes.py`: regimes, knobs, thread plans, device session, R1/R2/R5/R8/R10, X1 typed refusal, out-of-range `dim`; 26 tests;
  - `test_softmax_precision_baseline.py`: 6 cases.
