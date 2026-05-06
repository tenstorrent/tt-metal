# Changelog: linear

## Phase 0 — Core Implementation
- **Date**: 2026-05-06
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Single-core 2D matmul + optional row-broadcast bias add, using the `compute_kernel_lib::matmul_block` and `compute_kernel_lib::add_bias_bcast_rows` helpers end-to-end. `MathFidelity.HiFi4 + fp32_dest_acc_en=True` for K-accumulation precision. Subblock geometry `1×1` with `num_k_blocks=1`.
- **Accuracy achieved** (from `test_linear_precision_baseline.py`):

  | Shape (M,K,N) | Bias | PCC | Max abs | Mean abs | Rel RMS |
  |---------------|:----:|------|---------|----------|---------|
  | (32, 32, 32) | no  | 0.99999930 | 0.125 | 0.001255 | 0.001192 |
  | (128, 128, 128) | no | 0.99999947 | 0.250 | 0.002594 | 0.001034 |
  | (128, 256, 128) | no | 0.99999946 | 0.250 | 0.003880 | 0.001043 |
  | (256, 256, 256) | no | 0.99999945 | 0.250 | 0.003790 | 0.001050 |
  | (32, 32, 32) | yes | 0.99999775 | 0.1875 | 0.003949 | 0.002206 |
  | (128, 128, 128) | yes | 0.99999868 | 0.250 | 0.005333 | 0.001657 |
  | (128, 256, 128) | yes | 0.99999845 | 0.250 | 0.008554 | 0.001814 |
  | (256, 256, 256) | yes | 0.99999826 | 0.500 | 0.009020 | 0.001925 |

  All shapes clear PCC ≥ 0.99999 (asserted threshold: 0.999). Errors consistent with bf16 K-accumulation rounded into fp32 DEST plus a final bf16 pack; bias path adds one extra bf16 round-trip through `cb_partials`, raising max abs from 0.25 to 0.50 at the largest shape.
- **Issues encountered**: None. Acceptance suite (`test_linear.py`) passed on the first verifier run; one minor consistency fix applied in `linear_compute.cpp` (switch raw `cb_wait_front`/`cb_pop_front` to `bias_buf.wait_front`/`bias_buf.pop_front` to match the canonical `bmm_large_block_zm_fused_bias_activation.cpp` pattern).
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/linear/test_linear.py` — acceptance suite (pre-existing, 19 cases)
  - `tests/ttnn/unit_tests/operations/linear/test_linear_extended.py` — output contract, memory-config kwarg, zero-bias equivalence, bias-row-1..31-ignored, bias-height validation, back-to-back bias/no-bias state-leak (8 cases)
  - `tests/ttnn/unit_tests/operations/linear/test_linear_precision_baseline.py` — PCC + abs/RMS error baseline across 4 shapes × {no bias, bias} (8 cases)
- **Total**: 35/35 tests passing.
