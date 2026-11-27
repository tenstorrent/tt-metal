# Changelog: g6_indexed_fill

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One data-movement kernel per core (RISCV_0 / NOC_0), no compute kernel. It moves x pages ++ value pages over the full grid, reads `batch_ids` on the device in every call in the shadow of the unconditional page loads, and writes each result page exactly once. Regimes `ids_resident` / `ids_streamed`.
- **SUPPORTED at Phase 0**: dtype=[uint32], layout=[ROW_MAJOR_LAYOUT], alignment=[h_non_aligned, w_non_aligned, tile_aligned]. The verifier widened alignment from [h_non_aligned] because the prompt's MUST rule forbids a host refusal of a valid shape; no kernel change was needed.
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0, rms_err=0, max ULP=0 on 4 shapes (production `[7,1,1,32]`, rule-9 `[5,2,1,64]`, `[5,1,1,17]`, `[3,1,32,32]`) via `test_g6_indexed_fill_precision_baseline.py`.
- **Golden suite at Phase 0**: 11 / 11 with `EVAL_CAPTURE_L1=0` (verifier: 9 supported_pass, 0 supported_fail / xpass_drift / xfail_wrong_mode, 2 untagged regression tests passing); 9 / 11 under the default runner. The 2 failures are L1-capture harness defects that stock `ttnn.add` reproduces (`verification_report.md`).
- **Issues encountered**: `SUPPORTED["alignment"]` refused `w_non_aligned` / `tile_aligned` shapes on the host (MUST violation), fixed in place. Escalated to the eval owners: L1 capture disables the program cache, which breaks the `new_index` trace capture, and it raises on the refusal-only case (`peak_total=0`).
- **Tests added**: `test_g6_indexed_fill.py` (planner, 34 cases), `test_g6_indexed_fill_precision_baseline.py` (verifier, 4 shapes covering all three alignment tags).
- **Refinement queue**: empty. TARGET − SUPPORTED = ∅, and `perf_rules.txt` bars measuring cells of 32 tiles or fewer; every golden cell is ≤ 10 tiles.
