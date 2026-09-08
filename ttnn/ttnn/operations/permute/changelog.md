# Changelog: permute

## Phase 0 — Core Implementation (`whole_tile_relocation`)
- **Date**: 2026-09-08
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). One native `ttnn.generic_op` dispatch: reader (NoC0) → `cb_tiles` (depth 2 × `BLOCK_TILES` = 8 tile pages) → writer (NoC1), multi-core over a linear output-tile range per core (`split_work_to_cores(..., row_wise=True)`). No compute kernel — every 32×32 tile is relocated intact.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE], alignment=[tile_aligned], rank=[4], swap_hw=[False], mem=[dram_interleaved]; validate-only gate: `dims` must preserve the innermost pair.
- **Accuracy achieved**: bit-exact — PCC=1.0, max_abs_err=0.0, mean_abs_err=0.0, rel_rms_err=0.0, got/true ratio 1.000000 (p5=p95=1.0) on 4 shapes via `test_permute_precision_baseline.py`.
- **Golden suite at Phase 0**: 7 supported_pass / 532 total (xfail_expected=433, invalid_skipped=88, supported_fail=0, xpass_drift=0, xfail_wrong_mode=0) per `verifier_report.json`.
- **Perf**: not yet measured on device (no perf refinement landed). Baseline framing from `eval/prompts/permute.txt`: stock permute on `[2,4,512,512]` dims=(1,0,2,3) is 47.5 us, DRAM-bandwidth-bound, ~0% NoC congestion.
- **Issues encountered**: one drift fix — `SUPPORTED` declared a validate-only `inner_pair` axis the harness never generates, which made `unsupported_reason` classify **every** cell as unsupported (supported_pass=0, xpass_drift=7). Moved the gate to `SUPPORTED_INNER_PAIR` + an explicit `validate()` check; refusal semantics unchanged. Re-ran golden: all loud categories 0. No other code-review or ledger finding required a change.
- **Tests added**: `test_permute.py` (acceptance, 10/10), `test_permute_precision_baseline.py` (4/4).
