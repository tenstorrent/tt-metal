# Changelog: subtract

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). Regime R0 `tile_stream_same_shape`: one Metal 2.0 `ttnn.ProgramSpec` dispatch on Quasar. The flat output-tile range is split over the full 8×4 grid (row-wise, contiguous), with one `compute_kernel_lib::sub` call per NEO over its share, and implicit-sync tagged reader/writer that hand the tag back. Thread plan (1,1,1): the op is bound by data movement.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[DRAM, L1], sharding=[interleaved], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], broadcast=[none], output_dtype=[None, bfloat16]. EXCLUSIONS = spec rules R1, R2.
- **Accuracy achieved**: PCC ≥ 0.999998, max_abs_err = 0.0156, relative RMS ≤ 0.00186, max 1 bf16 ULP (10 of 1,048,576 elements; 0 elsewhere). Measured on 4 shapes × {DRAM, L1} via `test_subtract_precision_baseline.py`. Golden `ulp_p99` = 0 on every graded cell.
- **Golden suite at Phase 0**: 78 / 78 supported cells passing (72 cells + 6 numeric regression), 1,512 xfail_expected, X1 rejection passing. 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`).
- **Issues encountered**: Verifier code-review fix: `split_tiles` had two duplicated branches computing the same contiguous row-major assignment. Collapsed to one path, with a host partition test that also covers `MIN_TILES_PER_CORE` > 1. No drift fixes were needed.
- **Tests added**: `test_subtract.py` (harness acceptance), `test_subtract_plan.py` (thread plans × quanta, multi-cluster shapes, one-session test, L1 placement, refusals, split partition), `test_subtract_precision_baseline.py`.
