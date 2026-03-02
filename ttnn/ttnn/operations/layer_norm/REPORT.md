# Layer Norm Operation - Build Report

## Summary

**Operation**: `layer_norm` - PyTorch-compatible Layer Normalization
**Math**: `y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta`
**Result**: ALL STAGES PASSED - Operation fully functional

## Pipeline Execution

| Phase | Agent | Output | Result |
|-------|-------|--------|--------|
| 0: Discovery | Manual | Reference selection | moreh_group_norm + generic_op template |
| 1: Analysis | ttnn-operation-analyzer | moreh_group_norm_analysis.md | Complete |
| 2: Design | ttnn-operation-architect | op_design.md + .tdd_state.json | 4 TDD stages registered |
| 3: Build | ttnn-generic-op-builder | Python infra + stub kernels | 7/7 infra tests pass |
| 4: TDD Kernels | ttnn-kernel-writer (x4) | Implemented kernels | 4/4 stages pass (20/20 tests) |
| 5: Report | Manual | REPORT.md | This file |

## TDD Pipeline Results

| Stage | Name | Tests | Attempts | Result |
|-------|------|-------|----------|--------|
| 0 | data_passthrough | 5/5 | 1 | PASS |
| 1 | mean_reduction | 5/5 | 1 | PASS |
| 2 | variance_normalize | 5/5 | 1 | PASS |
| 3 | affine_transform | 5/5 | 1 | PASS |

**Integration tests**: 8/8 PASSED (numerical accuracy verified against PyTorch)

## Architecture

- **Execution**: Single-core (0,0), DRAM interleaved
- **Layout**: TILE_LAYOUT, bfloat16
- **Algorithm**: "Small" variant - all Wt width tiles per row fit in L1

### CB Layout (13 circular buffers)

| CB | ID | Pages | Purpose |
|----|-----|-------|---------|
| cb_input | 0 | Wt | Input tiles (persistent, 3 reads per row) |
| cb_scaler | 1 | 1 | Reduce scaler (1/W for AVG) |
| cb_eps | 2 | 1 | Epsilon tile |
| cb_gamma | 3 | Wt | Gamma tiles (if present) |
| cb_beta | 4 | Wt | Beta tiles (if present) |
| cb_out | 16 | Wt | Output tiles |
| cb_mean | 24 | 1 | Per-row mean |
| cb_centered | 25 | Wt | x - mean (persistent, 2 reads) |
| cb_squared | 26 | Wt | (x - mean)^2 |
| cb_var | 27 | 1 | Per-row variance |
| cb_rstd | 28 | 1 | 1/sqrt(var + eps) |
| cb_normalized | 29 | Wt | Normalized output (before affine) |
| cb_gamma_out | 30 | Wt | After gamma multiply |

### Compute Pipeline (per row, 6-8 phases)

1. **REDUCE_ROW** (AVG): input -> mean
2. **SUB** (COL broadcast): input - mean -> centered
3. **MUL** (self): centered^2 -> squared
4. **REDUCE_ROW** (AVG): squared -> variance
5. **ADD** (SCALAR) + **RSQRT**: var + eps -> rstd
6. **MUL** (COL broadcast): centered * rstd -> normalized
7. **MUL** (ROW broadcast): normalized * gamma (if present)
8. **ADD** (ROW broadcast): + beta (if present)

## Files Produced

### Operation (`ttnn/ttnn/operations/layer_norm/`)
- `__init__.py` - Package exports
- `layer_norm.py` - Entry point with validation
- `layer_norm_program_descriptor.py` - CB config, kernel setup, runtime args
- `kernels/reader_layer_norm.cpp` - DRAM reader with scaler/eps fill
- `kernels/compute_layer_norm.cpp` - 6-8 phase normalization pipeline
- `kernels/writer_layer_norm.cpp` - DRAM writer
- `op_design.md` - Architecture and kernel design document
- `.tdd_state.json` - TDD pipeline state (all stages passed)

### Tests (`tests/ttnn/unit_tests/operations/layer_norm/`)
- `test_layer_norm.py` - Integration tests (8 tests, numerical accuracy)
- `test_stage_data_passthrough.py` - TDD stage 1
- `test_stage_mean_reduction.py` - TDD stage 2
- `test_stage_variance_normalize.py` - TDD stage 3
- `test_stage_affine_transform.py` - TDD stage 4

## Git History

```
e825527 Fix reader kernel CT args for optional gamma/beta tensors
33edc8e [ttnn-kernel-writer] advance TDD state: affine_transform -> COMPLETE
ca96fec [ttnn-kernel-writer] Stage 4 affine_transform: gamma/beta ROW broadcast
f6983e6 [ttnn-kernel-writer] advance TDD state: variance_normalize -> affine_transform
e9094f6 [ttnn-kernel-writer] Stage 3 variance_normalize: full normalization pipeline
dbd5223 [ttnn-kernel-writer] advance TDD state: mean_reduction -> variance_normalize
8187d5f [ttnn-kernel-writer] Stage 2 mean_reduction: reduce + bcast subtract
9e1ffb3 [ttnn-kernel-writer] advance TDD state: data_passthrough -> mean_reduction
c082286 [ttnn-kernel-writer] Stage 1 data_passthrough: implement reader/compute/writer kernels
9c042c9 [ttnn-generic-op-builder] stubs: layer_norm
```

## Key Decisions and Deviations

1. **Raw compute APIs vs helpers**: Used raw `reduce_tile`, `sub_tiles_bcast`, `mul_tiles_bcast` instead of `compute_kernel_lib::` helpers, as the helpers were not available/applicable for the generic_op kernel context.

2. **Reduce scaler = 1/(Wt*32)**: The hardware reduce sums 32 columns per tile internally, so for mean across W elements, the scaler must be `1/W = 1/(Wt*32)`.

3. **CT args padding for optional tensors**: Always include dummy TensorAccessorArgs in compile-time args for gamma/beta even when absent, to avoid template instantiation failures in non-template `if constexpr` context.

4. **Persistent CB pattern**: cb_input read 3 times and cb_centered read 2 times per row using NoPop policies with explicit manual `cb_pop_front` at row end.

5. **binary_op_init_common required**: Must call this before reduce/bcast operations to properly initialize hardware packer state.

## Usage

```python
from ttnn.operations.layer_norm import layer_norm

# Without affine
output = layer_norm(input_tensor, epsilon=1e-5)

# With affine (gamma and beta)
output = layer_norm(input_tensor, weight=gamma, bias=beta, epsilon=1e-5)
```
