# Verification Report: atan_mean

## Code Review

### Design conformance

The implementation matches `op_design.md` on every binding dimension:

- **Algorithm**: SFPU atan per tile → matmul-mode REDUCE_ROW AVG with `1/W` col-0 scaler. The intermediate `atan(x)` is buffered in an L1 CB of exactly `Wt` pages — never materialised to DRAM. ✓
- **Pipeline topology**: Reader (NCRISC) streams `Wt` input tiles per row-tile + one-shot scaler write; compute (TRISCs) runs `sfpu_atan` then `reduce<AVG, REDUCE_ROW>`; writer (BRISC) drains 1 output tile per row-tile. ✓
- **Parallelization**: `ttnn.split_work_to_cores(grid, total_row_tiles)` two-group split over the full Tensix grid. ✓
- **Inter-core communication**: None — each output row-tile is computed end-to-end on a single core, as specified. ✓

### Helper usage

- `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles)` — required prologue, present and correctly anchored on the three CBs with relevant data formats. ✓
- `compute_kernel_lib::sfpu_atan<cb_input_tiles>(cb_atan_tiles, Wt)` — used with defaults (`SfpuBatching::Auto`, `WaitAndPopPerTile`, `PerTile`, `INPUT_AND_OUTPUT`). With Wt ≤ 4 and DEST capacity = 4 tiles, the auto-batching handles the row in one batch. ✓
- `compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_atan_tiles, cb_scaler, cb_output_tiles, row(Wt))` — used with defaults, correctly dispatching to the matmul-mode reduce path (which AVG+REDUCE_ROW always selects per `reduce_helpers_common.hpp`). ✓
- `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, AVG, REDUCE_ROW, W>()` — pool-type-aware overload, correctly placing the `1/W` value in matmul col-0 layout in bf16. ✓
- Reader/writer use raw `TensorAccessor` + `noc_async_read_tile` / `noc_async_write_tile` — no helper exists for "stream a tiled tensor's tiles through a CB" and the standard idiom is canonical (matches `multigammaln_lanczos` and `toy_variance`). ✓

No raw compute API call is used where a helper exists. The two-helper composition (atan, then reduce) is the smallest expression of the operation in the kernel library — there is no `pre_reduce_op` hook in `reduce<>` and the `post_reduce_op` runs after reduction, too late to apply atan.

### Fixes applied

| File | Fix | Reason |
|------|-----|--------|
| `kernels/atan_mean_reader.cpp` | Removed `Wt` from compile-time args; derive `constexpr uint32_t Wt = W / 32;` inside the kernel. | `Wt = W / 32` was passed redundantly alongside `W`. Eliminating one CT arg simplifies the contract and removes the possibility of a `W`/`Wt` mismatch ever being introduced. |
| `atan_mean_program_descriptor.py` | Reader CT args reduced from `[CB_INPUT_TILES, CB_SCALER, W, Wt, ...]` to `[CB_INPUT_TILES, CB_SCALER, W, ...]`. TensorAccessorArgs starts at index 3 instead of 4. | Matches the kernel change. |

Verified: all 20 original acceptance tests still pass after the CT-arg change.

### Code quality observations (no fix needed)

- CB indices use `0, 8, 16, 24` (multiples of 8). This is harmless — index assignment in 0–31 is arbitrary. Not changed.
- Per-tile `noc_async_read_barrier()` in the reader is the standard streaming idiom (matches `multigammaln_lanczos` and `toy_variance`). For Wt ≥ 4 the batched-issue / single-barrier optimisation could reduce per-tile NoC stall (noted in `data_transfer.md` §10), but Phase 0 shapes cap at Wt=4 so this is not an active concern. Deferred to refinements only if profiling shows it matters.
- Scaler is computed locally on every core (each core executes `calculate_and_prepare_reduce_scaler`). The alternative would be a multicast from one sender core — but the cost of one local `1/W` constant computation per core is negligible. Not changed.
- Output write amplification is intrinsically 32× over valid output bytes (col-0 fill REDUCE_ROW layout). Avoiding this would require a row-major output and an untilize path; out of scope for Phase 0.

### Architectural items not fixed (scope of refinements)

| Item | Rationale for deferring |
|------|-------------------------|
| Expose `compute_kernel_config` | Phase 0 spec explicitly hard-codes `HiFi4` + `fp32_dest_acc_en=True`. Lifting this is a Phase-1 refinement. |
| Support BFLOAT16 / BFLOAT8_B inputs | Entry-point validation rejects them. Phase-1 refinement. |
| Support non-tile-aligned H/W | Phase 0 hard-rejects. The partial-scaler infrastructure in `kernel_lib` (`calculate_and_prepare_partial_reduce_scalers`, `ReducePartialScaler::last_tile_at`) exists but is unwired. Phase-1 refinement. |
| Support ROW_MAJOR input | Would require an in-kernel tilize stage. Phase-1 refinement. |
| Support rank ≠ 4 | Entry point hard-rejects. Could be lifted with a flatten-to-rank-4 wrapper. Phase-1 refinement. |
| `cb_atan_tiles` L1 ceiling for large W | Sized to `Wt` fp32 pages — caps W somewhere in the low thousands before L1 overflow. Would need a tile-block-streaming reduce variant. Last refinement. |

## Precision Baseline

Measured by `tests/ttnn/unit_tests/operations/atan_mean/test_atan_mean_precision_baseline.py` against fp32 PyTorch reference on N(0, 1) inputs.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| (1,1,32,32) | 0.99999996 | 2.45e-4 | 6.72e-5 | 7.16e-4 |
| (1,1,64,64) | 0.99999995 | 1.09e-4 | 4.08e-5 | 6.47e-4 |
| (1,1,256,128) | 0.99999997 | 1.07e-4 | 3.00e-5 | 6.32e-4 |
| (1,8,128,128) | 0.99999997 | 1.28e-4 | 2.98e-5 | 6.18e-4 |

**Assessment**: Errors are consistent with a fp32-dest-accumulated REDUCE_ROW path bounded by the SFPU `atan_tile` polynomial approximation. The bf16 `1/W` scaler is bit-exact for the tested powers-of-two W, so it contributes no error here. PCC > 0.99999 on every shape; max-abs is ~1e-4, three orders of magnitude tighter than the Phase-0 acceptance tolerance (1e-2).

**Recommended tolerances** (starting points for refinements):
- PCC ≥ 0.9999
- atol ≈ 5e-4
- rtol ≈ 1e-3

## Test Results

| Test file | Passing |
|-----------|---------|
| `test_atan_mean.py` (acceptance, immutable) | 20/20 |
| `test_atan_mean_extended.py` (verifier-added) | 6/6 |
| `test_atan_mean_precision_baseline.py` (verifier-added) | 4/4 |
| **Total** | **30/30** |

Extended tests added:
- Two extra shapes (Wt=3, NC=4) to broaden coverage.
- `zero_input`: atan(0)=0 invariant.
- `sign_antisymmetry`: `atan_mean(-x) == -atan_mean(x)` (atan is odd).
- `saturation_large_positive`: large-magnitude inputs converge to atan asymptote π/2.
- `positive_domain`: inputs in [1, 5] — covers the range-reduction branch (`|x| > 1` → `π/2 − atan(1/x)` inside the SFPU).

## Recommendations

Synthesized from the numerical stability and data transfer analyses (see `numerical_stability.md`, `data_transfer.md`):

1. **Most valuable refinement = `compute_kernel_config` exposure**. The numerical stability analysis flags `fp32_dest_acc_en` and `math_fidelity` as the primary precision/throughput levers. Together they let callers switch from "Phase 0 maximum precision" (HiFi4 + fp32 dest) to "throughput-leaning" variants (HiFi3 + fp16b dest) without changing the algorithm. The implementation already routes everything through kernel-lib helpers, so the plumbing is straightforward.

2. **Latent precision risk if reduce path ever changes**: `cb_atan_tiles` has no `UnpackToDestMode::UnpackToDestFp32` configured. This is fine today because matmul-mode REDUCE_ROW uses SrcA/SrcB ingress, not `copy_tile`. If a future refinement switches to the non-matmul reduce path (e.g., to support partial scalers without going through matmul), `UnpackToDestFp32` would need to be set on `cb_atan_tiles` to preserve precision on accumulator reloads. Refinement agents touching the reduce path should consult `numerical_stability.md` § "Tile-Boundary Precision".

3. **Wormhole-B0 HW issue #38306**: Phase 0 uses the exact `HiFi4 + fp32_dest_acc_en=True` combination flagged in the LLK reference docs as potentially incorrect on Wormhole. The acceptance tests pass at 20/20 today, but if precision regressions appear on a new LLK or silicon, this combination is the prime suspect. Recommendation: refinement that exposes `compute_kernel_config` should also document this as a tunable workaround.

4. **NoC balance vs Wt**: Read:write ratio is `Wt : 1`. For Wt ≥ 4, NoC1 is mostly idle. The batched-read / single-barrier optimisation in the reader (`data_transfer.md` §10 item 1) would not shift the bottleneck (compute-bound on SFPU atan) but is a clean code change if anyone is touching the reader anyway.

5. **L1 ceiling on `cb_atan_tiles`**: This is the practical W-axis ceiling and should be the last refinement (memory pressure). Lifting it requires a different fusion (streaming the SFPU output into a partial reduce one tile-block at a time) rather than a parameter tweak.
