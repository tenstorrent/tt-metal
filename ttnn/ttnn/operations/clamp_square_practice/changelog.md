# Changelog: clamp_square_practice

## Phase 0 — Core Implementation
- **Date**: 2026-10-28
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). It is
  one `ttnn.generic_op` dispatch:
  - The flattened tile axis is split across the full grid (`split_work_to_cores(row_wise=True)`).
  - The reader and writer are block-batched (one barrier or flush per block).
  - Compute is one fused `eltwise_chain`: Copy → Clamp(fp32 RT bounds) → Square → Pack.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_layout=[INTERLEAVED], buffer_type=[DRAM],
  rank=[rank_ge_2], alignment=[tile_aligned]
- **Accuracy achieved**:
  - bf16-exact bounds: PCC=0.99999943, max_abs_err=1.94e-3, rel_rms_err=4.85e-4, **0 ULP** (bit-exact vs the
    bf16-rounded fp32 reference).
  - Non-bf16-exact bounds (-0.3, 0.7): PCC=0.9999983, max_abs_err=1.72e-3, rel_rms=3.6e-3, ≤ 1 ULP.
  - Measured on 4 shapes × 3 bound sets via `test_clamp_square_practice_precision_baseline.py`.
- **Golden suite at Phase 0**: n/a. `eval/golden_tests/clamp_square_practice/` does not exist, so `verify_supported`
  was not run. Acceptance is 48/48 on Blackhole.
- **Perf baseline (BH, 120 cores)**:
  - `(2,4,256,512)`: 13.75 µs (305 GB/s)
  - `(8,1,1024,1024)`: 87.2 µs (385 GB/s)
  - `(1,1,4096,4096)`: 182.5 µs (368 GB/s)
- **Issues encountered**: verifier fix — finite `min_val`/`max_val` beyond the fp32 range escaped as `OverflowError`
  from the host fp32 packer. `validate()` now rejects them with `ValueError`, consistent with `torch.clamp`.
- **Tests added**: `test_clamp_square_practice.py` (planner), `test_clamp_square_practice_precision_baseline.py`,
  `test_clamp_square_practice_extended.py` (verifier).
