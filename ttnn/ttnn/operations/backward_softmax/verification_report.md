# Verification Report: backward_softmax

## Summary

The Phase-0 implementation of `backward_softmax` was **two real bugs short of
correctness** when handed to the verifier; both fixed in this pass. With the
fixes the operation is mathematically correct (PCC ≥ 0.9999 across the
verified shape set). The remaining spec-test failures (`atol=0.01` exceeded
on multi-tile reductions) are a hardware precision-floor issue documented in
the precision baseline below, not a logic bug.

## Code Review

### Bugs fixed

1. **`BroadcastDim::SCALAR` was used in pass-2 sub** (`backward_softmax_compute.cpp:90-94`)
   - `accumulate_reduce_block<SUM, REDUCE_ROW>` produces an output tile with
     **per-row sums in column 0** (one value per within-tile row), not a
     single scalar. Likewise `REDUCE_COL` produces per-column sums in row 0.
     `BroadcastDim::SCALAR` reads element [0,0] only, yielding sum-for-row-0
     for every row — silently wrong for any tile larger than one within-tile
     row.
   - **Empirical confirmation before fix**: shape `(1,1,32,32)` test failed
     with row 0 matching exactly and rows 1-31 systematically wrong.
   - **Fix**: use `BroadcastDim::COL` for `dim=-1` (REDUCE_ROW path) and
     `BroadcastDim::ROW` for `dim=-2` (REDUCE_COL path). The standard
     `binary_op_helpers.hpp` doc table maps these correctly.
   - The `op_design.md` "Reduce Direction Verification" table also
     incorrectly listed SCALAR. The kernel was implemented faithfully to a
     wrong design.

2. **`cb_output` sized 2 pages caused deadlock for `BLOCK_SIZE > 2`** (`backward_softmax_program_descriptor.py:97-101`)
   - In pass 2, `sub` consumes only `cb_grad_output` (BLOCK_SIZE tiles per
     block) while `cb_output` (`y`) fills with the lockstep reader pushes.
     With `cb_output = 2`, the reader's third lockstep iteration blocks on
     `cb_reserve_back(cb_output, 1)`, which prevents `cb_grad_output` from
     receiving more dy tiles, so sub stalls.
   - **Empirical confirmation**: shape `(1,1,32,256)` (BLOCK_SIZE=8) hung in
     `cb_wait_front` on the compute side with the reader stuck in scaler
     init (initial probe), then in `cb_reserve_back(cb_output)` (after
     forward progress).
   - **Fix**: size `cb_output` to `2 × BLOCK_SIZE` pages — enough headroom
     for one full block of `y` to sit while sub drains `dy`. L1 footprint
     remains comfortably under budget (216 KiB at `BLOCK_SIZE=8`).

### Items reviewed and left as-is

- Helper usage is correct: `compute_kernel_lib::mul`,
  `accumulate_reduce_block`, `sub<COL/ROW, WaitAndPopPerTile,
  WaitUpfrontNoPop>`, and the explicit `cb_pop_front(cb_sum, 1)` at end of
  lane all follow the toy_variance precedent.
- Reader uses pool-type-aware `calculate_and_prepare_reduce_scaler` (correct
  fill pattern for both `REDUCE_ROW` and `REDUCE_COL`).
- Compute kernel uses `void kernel_main()`, modern include path
  (`api/dataflow/dataflow_api.h` etc.), and `TensorAccessor` (not
  deprecated `InterleavedAddrGen`).
- Compute config matches the design: `HiFi4 + fp32_dest_acc_en=True`.
- CB push/wait counts balance per CB (verified by inspection against
  op_design.md's CB sync table).
- The reader's lockstep dy/y push is correct for pass 1 (mul consumes both
  per-tile) and is now also safe for pass 2 (cb_output has enough capacity).

### Items deferred to refinements

- **Single-core only** — multi-core lane distribution is Refinement 1.
- **`compute_kernel_config` not exposed** — Refinement 2.
- **Precision floor on `dy − s`** — the hardware fp32-dest accumulation in
  the matmul reduce produces ~0.1-level absolute error on row sums for
  random-N(0,1) inputs at reduction depths ≥ 64 elements. After multiplying
  by `y`, the per-element absolute error reaches ~0.3 at the worst position,
  failing the spec test's `atol=0.01`. This is documented under
  Refinement 3 — it requires either an algorithm reformulation (e.g.
  transpose + REDUCE_COL non-matmul path) or a Kahan-compensation pass.

### Items I tried as quick precision fixes that did not help

- **`UnpackToDestMode::UnpackToDestFp32`** on the fp32 CBs: produced `inf`
  outputs immediately. The matmul-based REDUCE_ROW SUM is incompatible with
  this mode (it requires the SrcA path).
- **`MathFidelity::HiFi3`**: 1.4× *worse* max_abs (0.36 vs 0.26 on the
  failing shape). HiFi4 is the better setting.
- **`dst_full_sync_en=True`**: no measurable change.

## Precision Baseline

Measured on `tests/.../test_backward_softmax_precision_baseline.py` with
seed 42 and random N(0,1) inputs:

| Shape | dim | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-----|-------------|--------------|------------------|
| (1,1,32,32) | -1 | 0.9999996 | 0.083 | 0.009 | 0.0024 |
| (1,1,32,256) | -1 | 0.9999996 | 0.255 | 0.026 | 0.0026 |
| (1,1,64,128) | -1 | 0.9999997 | 0.256 | 0.018 | 0.0026 |
| (2,4,64,128) | -1 | 0.9999997 | 0.306 | 0.019 | 0.0026 |
| (1,1,32,32) | -2 | 0.9999994 | 0.062 | 0.009 | 0.0025 |
| (1,1,32,256) | -2 | 0.9999995 | 0.125 | 0.009 | 0.0029 |
| (1,1,64,128) | -2 | 0.9999995 | 0.221 | 0.013 | 0.0029 |
| (2,4,64,128) | -2 | 0.9999996 | 0.266 | 0.015 | 0.0029 |

**Assessment**: PCC ≥ 0.9999 confirms the operation is mathematically
correct. Mean absolute error is ~0.01-0.03 in tensors with mean abs values
~10. Relative RMS is consistently ~0.0024-0.003 (~0.25% RMS), which is
*much* tighter than what the absolute peak suggests — the peak is
catastrophic-cancellation-driven, concentrated at positions where `dy_i ≈ s`
(small expected outputs). This is a property of the algorithm + hardware, not
of the implementation.

**Recommended tolerances** for refinement and downstream tests:

- `pcc >= 0.999` — passes today across the baseline.
- `rel_rms <= 0.005` — passes today (worst is 0.0029) with margin.
- `atol`-based comparison: not advisable below `0.5` for shapes with
  reduce-axis tile count ≥ 2 until Refinement 3 lands. The spec test's
  `atol=0.01` is the tightest workable value only at `(1,1,32,32)`.

## Test Results

- **Acceptance** (`test_backward_softmax.py`, --run-all): **17 / 26 passing**.
  - The 9 failures are all `test_backward_softmax_correctness` parametrizations
    where `torch.allclose(..., atol=0.01)` fails due to the precision floor
    described above. PCC ≥ 0.9999 in every failing case.
  - Single-tile (32×32) cases pass for both dims; the negative-validation
    tests, default-dim test, and softmax-derived test all pass.
- **Precision baseline** (`test_backward_softmax_precision_baseline.py`): **8 / 8 passing** (PCC ≥ 0.999 floor).
- **Extended** (`test_backward_softmax_extended.py`): **9 / 9 passing**.

## Recommendations

Synthesizing from `numerical_stability.md`, `data_transfer.md`, and the test
results:

### Highest priority

1. **Address the precision floor (Refinement 3 in op_requirements.md).** The
   spec test cannot pass until this is solved or until the test is relaxed.
   Options ranked by likely effectiveness vs effort:
   - *Cheap*: switch to non-matmul reduce path for `dim=-1` by transposing
     to `dim=-2` (cost: extra transpose, but `reduce_tile`-based REDUCE_COL
     might avoid the SrcA TF32 truncation).
   - *Medium*: Kahan-compensated streaming sum via `transform_in_place` on
     `cb_sum` between blocks.
   - *Expensive*: rewrite the reduce step manually (skip the helper) so we
     control fidelity per phase.

### Standard precision/perf refinements

2. **Multi-core (Refinement 1)** — pure parallelism, no new data movement
   per `data_transfer.md`. Should drop wall time near-linearly with grid
   size until DRAM saturates.
3. **Compute config exposure (Refinement 2)** — gives callers an escape
   hatch for the WH B0 HiFi4 + fp32_dest bug (#38306) and lets perf-
   sensitive users trade fidelity for throughput.
4. **Tradeoff to flag**: per `data_transfer.md`, the two-pass algorithm
   re-reads both inputs in pass 2 (5× footprint DRAM bandwidth). A future
   single-pass variant could cache inputs in L1 between passes for shapes
   whose lane fits, dropping DRAM traffic by ~40%. This conflicts with the
   precision Refinement 3 if that one chooses to re-shape via transpose
   (transpose + 2 passes is even more L1 traffic). Pick one optimization
   target before coding either.

### Lower priority

5. **Float32-only is the major coverage gap** — `bfloat16` and `bfloat8_b`
   support (Refinement 4) would expand usability significantly.
6. **Non-tile-aligned shapes (Refinement 5)** — straightforward via
   `prepare_partial_reduce_scalers` (toy_variance is the reference).
7. **Rank flexibility (Refinement 7)** — pure entry-point reshape; cheap.

## Files Produced

| File | Purpose |
|------|---------|
| `verification_report.md` | this file |
| `capabilities.md` | living document of current op capabilities |
| `data_transfer.md` | DRAM bandwidth, NoC channel balance, L1 footprint |
| `numerical_stability.md` | (pre-existing) error sources and accumulation strategy |
| `op_requirements.md` | phased refinement roadmap |
| `changelog.md` | initialized with Phase 0 + precision baseline |
| `tests/.../test_backward_softmax_precision_baseline.py` | PCC + abs/RMS across 4 shapes × 2 dims |
| `tests/.../test_backward_softmax_extended.py` | additional shape, softmax-derived input, determinism, memory_config tests |
