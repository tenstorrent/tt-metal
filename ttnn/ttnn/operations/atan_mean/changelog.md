# Changelog: atan_mean

## Phase 0 — Core Implementation

- **Date**: 2026-05-13
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). Fused `torch.atan(x).mean(dim=-1)` into a single TTNN program. Compute kernel uses `compute_kernel_lib::sfpu_atan` to apply atan in-place per tile, then `compute_kernel_lib::reduce<AVG, REDUCE_ROW>` to collapse each row into a single output tile. Reader emits a one-shot bf16 `1/W` scaler at program startup via `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler` and streams `Wt` fp32 input tiles per row-tile from DRAM. Writer drains one output tile per row-tile. Work is distributed via `ttnn.split_work_to_cores` over the full Tensix grid; one row-tile = one `(n, c, ht)` triple.
- **Accuracy achieved** (measured by `test_atan_mean_precision_baseline.py` on N(0,1) inputs):
  - Shape (1,1,32,32): PCC=0.99999996, max_abs=2.45e-4, mean_abs=6.72e-5, rms_rel=7.16e-4
  - Shape (1,1,64,64): PCC=0.99999995, max_abs=1.09e-4, mean_abs=4.08e-5, rms_rel=6.47e-4
  - Shape (1,1,256,128): PCC=0.99999997, max_abs=1.07e-4, mean_abs=3.00e-5, rms_rel=6.32e-4
  - Shape (1,8,128,128): PCC=0.99999997, max_abs=1.28e-4, mean_abs=2.98e-5, rms_rel=6.18e-4
  - All shapes well inside the Phase-0 acceptance contract (PCC ≥ 0.9995, max-abs ≤ 1e-2).
- **Issues encountered during verification**:
  - **Redundant compile-time arg**: The reader kernel took both `W` and `Wt` as compile-time args, but `Wt = W / 32`. Verifier removed the `Wt` CT arg from the reader and derives it inline (`constexpr uint32_t Wt = W / 32;`). Updated the program descriptor to match. All 20 acceptance tests continue to pass.
  - No correctness or design-conformance bugs found.
- **Tests added by verifier**:
  - `tests/ttnn/unit_tests/operations/atan_mean/test_atan_mean_precision_baseline.py` — 4 shapes, measures PCC + max/mean abs err + relative RMS err.
  - `tests/ttnn/unit_tests/operations/atan_mean/test_atan_mean_extended.py` — 6 tests: 2 mid-size shapes (Wt=3, NC=4), zero input invariant, sign antisymmetry, saturation asymptote at large |x|, positive-only domain (covers SFPU atan range-reduction branch).
- **Final test count**: 30/30 passing (20 acceptance + 6 extended + 4 precision baseline).

## Refinement 2 — Support bfloat16 (and bfloat8_b) input dtype

- **Date**: 2026-05-13
- **What was done**:
  - Relaxed `_validate_input` in `atan_mean.py` to accept `{float32, bfloat16, bfloat8_b}` via a `_SUPPORTED_DTYPES` tuple. Other dtypes (e.g. uint32) continue to raise `ValueError`.
  - No kernel changes needed: `cb_input_tiles`, `cb_atan_tiles`, and `cb_output_tiles` already key their page sizes off `input_tensor.buffer_page_size()` / `input_tensor.dtype`, so the program descriptor adapts automatically. `cb_scaler` remains bf16 (matmul col-0 convention) regardless of input dtype, matching the verifier's note that the scaler tile format is fixed.
  - Updated the docstring on `atan_mean(...)` and the matching capabilities entry to reflect the new dtype contract.
  - Flipped the Phase-0 negative test `test_atan_mean_rejects_bf16_dtype` (which directly contradicts the new contract) into a positive test `test_atan_mean_accepts_bf16_dtype` asserting that bf16 round-trips through the op with the correct output dtype and shape.
- **Accuracy achieved** (measured by `test_atan_mean_dtype.py` on N(0,1) inputs across shapes `[(1,1,32,32), (1,1,64,64), (1,1,128,64), (1,1,32,96), (1,8,64,64)]`):
  - float32:   PCC ≥ 0.99995, max-abs ≤ 1e-2  (per-test asserts; baseline shapes hit 0.99999995 / ~1.3e-4 as before)
  - bfloat16:  PCC ≥ 0.999,   max-abs ≤ 3e-2
  - bfloat8_b: PCC ≥ 0.99,    max-abs ≤ 1e-1
  - Zero-input and π/2 saturation both hold for every supported dtype.
  - Per the verifier note: with `fp32_dest_acc_en=True`, the matmul ingress unpacks bf16/bf8 → fp32 on SrcA, so accumulation precision is unchanged from fp32 input; degradation is dominated by the input quantisation itself, not the reduce path.
- **Golden test progress**: N/A — `eval/golden_tests/atan_mean/` does not exist for this op.
- **Issues encountered**: None. Implementation was a one-line validator change plus tests; the verifier's preflight analysis (CBs already dtype-keyed, scaler format fixed) held up exactly as described.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/atan_mean/test_atan_mean_dtype.py` (NEW): 20 tests covering {fp32, bf16, bf8} × 5 shapes for correctness, plus zero-input and saturation tests for bf16/bf8, plus an unsupported-dtype negative test.
  - `tests/ttnn/unit_tests/operations/atan_mean/test_atan_mean.py::test_atan_mean_accepts_bf16_dtype`: replaces the Phase-0 rejection test, asserting bf16 acceptance and output-dtype contract.
- **Final test count**: 50/50 passing (20 acceptance + 6 extended + 4 precision baseline + 20 dtype).
