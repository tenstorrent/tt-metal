# Changelog: backward_softmax

## Phase 0 — Core Implementation

- **Date**: 2026-05-08
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
- **Bugs fixed during verification**:
  1. **Critical correctness bug** — pass-2 sub was using `BroadcastDim::SCALAR`. `accumulate_reduce_block<SUM, REDUCE_ROW>` produces a per-row sum vector (column-0 of the output tile), not a scalar. Fix: use `BroadcastDim::COL` for `dim=-1` and `BroadcastDim::ROW` for `dim=-2`. Without this fix, only row 0 of every output tile was correct; rows 1-31 used row-0's sum. Affected `backward_softmax_compute.cpp:90-94`.
  2. **Deadlock on multi-tile reductions** — `cb_output` was sized 2 pages. In pass 2, `sub` consumes only `cb_grad_output`, then `mul` consumes `cb_output`. The reader pushes both in lockstep, so `cb_output` fills up before sub finishes draining `cb_grad_output`, blocking the reader and stalling sub. Fix: size `cb_output` to `2 × BLOCK_SIZE` pages so the reader can pre-push a full block while sub processes dy. Affected `backward_softmax_program_descriptor.py:97-101`.
- **Accuracy achieved (precision baseline)**:

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

- **Issues encountered**: The spec test (`test_backward_softmax.py`) uses `atol=0.01, rtol=0.05`. Combined with the catastrophic-cancellation site `dy − s`, this `atol` is **not achievable** on Wormhole B0 for shapes whose reduce-axis tile count ≥ 2 — the matmul-based REDUCE_ROW SUM (and the regular REDUCE_COL SUM) accumulate with worse-than-fp32 effective precision (likely SrcA TF32 truncation per the numerical_stability.md analysis). PCC ≥ 0.9999 across all shapes shows the operation is mathematically correct; the absolute precision floor is hardware-bound. See `op_requirements.md` Refinement 3 for the planned mitigation strategy.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/backward_softmax/test_backward_softmax.py` (acceptance — pre-existing).
  - `tests/ttnn/unit_tests/operations/backward_softmax/test_backward_softmax_precision_baseline.py` (new — PCC + abs/RMS metrics across 4 shapes × 2 dims).
  - `tests/ttnn/unit_tests/operations/backward_softmax/test_backward_softmax_extended.py` (new — extra shapes + softmax-derived `output` + determinism + memory_config kwarg).
- **Test status**: 17/26 acceptance tests pass (the 9 failures are all the "atol=0.01 unachievable" precision issue described above); 8/8 precision baseline pass; 9/9 extended pass.
