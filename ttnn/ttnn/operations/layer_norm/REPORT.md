# Layer Norm Operation — Pipeline Report

## Summary

**Operation**: `layer_norm`
**Formula**: `y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta`
**Result**: ALL STAGES PASSED — fully functional

Implements PyTorch-compatible `torch.nn.LayerNorm` for 2D row-major tensors on a single Tensix core. Normalizes over the last dimension (W) with optional learnable weight (gamma) and bias (beta).

## Pipeline Execution

| Phase | Agent | Duration | Output |
|-------|-------|----------|--------|
| 0: Discovery | (manual) | — | 3 reference operations identified |
| 1: Analysis | ttnn-operation-analyzer ×3 | ~2.5 min (parallel) | tilize_analysis.md, untilize_analysis.md, softmax_analysis.md |
| 2: Design | ttnn-operation-architect | ~4 min | op_design.md, .tdd_state.json (4 stages) |
| 3: Build | ttnn-generic-op-builder | ~17 min | Python infra + stub kernels + tests (10/10 pass) |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~29 min | All 4 stages implemented and passing |
| 5: Report | (manual) | — | This file |

## Reference Operations

| Role | Operation | Path |
|------|-----------|------|
| input_stage | tilize | `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_single_core_program_factory.cpp` |
| output_stage | untilize | `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_single_core_program_factory.cpp` |
| compute_core | softmax | `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general.cpp` |

## TDD Stage Results

| Stage | Name | Result | Hard/Free Attempts | Description |
|-------|------|--------|--------------------|-------------|
| 0 | data_pipeline | PASS | 1/0 | Identity passthrough: tilize → untilize |
| 1 | subtract_mean | PASS | 0/2 | Mean reduction + broadcast subtraction |
| 2 | full_normalize | PASS | 0/0 | Variance, rsqrt, normalization |
| 3 | affine_transform | PASS | 0/1 | Weight (gamma) multiply + bias (beta) add |

## CB Layout (13 Circular Buffers)

| CB | Index | Purpose |
|----|-------|---------|
| cb_in | 0 | RM input pages (tilize input) |
| cb_eps | 1 | Epsilon tile |
| cb_scaler | 2 | Reduce scaler 1/W |
| cb_weight | 3 | Gamma tiles |
| cb_bias | 4 | Beta tiles |
| cb_tilize_out | 16 | Tilized input |
| cb_out | 17 | Normalized output |
| cb_untilize_out | 18 | RM output for writer |
| cb_mean | 24 | Row mean |
| cb_x_minus_mean | 25 | x - mean |
| cb_sq | 26 | (x-mean)^2 |
| cb_var | 27 | Variance |
| cb_inv_std | 28 | 1/sqrt(var+eps) |
| cb_norm | 29 | Pre-affine normalized |

## Compute Pipeline (8 Phases)

1. **Tilize**: RM sticks → tile format (cb_in → cb_tilize_out)
2. **Mean reduce**: SUM reduce with 1/W scaler over row → cb_mean
3. **Subtract mean**: COL broadcast subtraction → cb_x_minus_mean
4. **Square**: Element-wise square → cb_sq
5. **Variance reduce**: SUM reduce with 1/W scaler → cb_var
6. **Rsqrt**: Add eps + rsqrt → cb_inv_std
7. **Normalize**: COL broadcast multiply → cb_norm (or cb_out if no affine)
8. **Affine** (optional): gamma multiply + beta add → cb_out
9. **Untilize**: Tile → RM sticks (cb_out → cb_untilize_out)

## Files Produced

### Operation (`ttnn/ttnn/operations/layer_norm/`)
- `__init__.py` — Package init, exports `layer_norm`
- `layer_norm.py` — Entry point with validation
- `layer_norm_program_descriptor.py` — CB config, work distribution, kernel setup
- `kernels/reader.cpp` — Reads RM sticks, generates scaler/eps, reads weight/bias
- `kernels/compute.cpp` — Full compute pipeline (tilize → normalize → untilize)
- `kernels/writer.cpp` — Writes RM sticks to DRAM
- `op_design.md` — Architecture + kernel implementation design
- `.tdd_state.json` — TDD pipeline state

### Tests (`tests/ttnn/unit_tests/operations/layer_norm/`)
- `test_layer_norm.py` — Integration tests (6 tests)
- `test_stage_data_pipeline.py` — TDD stage 0
- `test_stage_subtract_mean.py` — TDD stage 1
- `test_stage_full_normalize.py` — TDD stage 2
- `test_stage_affine_transform.py` — TDD stage 3

## Git History

```
bb635d2 [ttnn-kernel-writer-tdd] stage affine_transform: passed
9a9b780 [ttnn-kernel-writer-tdd] stage full_normalize: passed
e299e24 [ttnn-kernel-writer-tdd] stage subtract_mean: passed
693b9c4 [ttnn-kernel-writer-tdd] stage data_pipeline: passed
350a091 [ttnn-generic-op-builder] stubs: layer_norm
e4ed04c [ttnn-operation-architect] design: layer_norm
454f966 [ttnn-operation-analyzer] analysis: softmax
f47564f [ttnn-operation-analyzer] analysis: untilize
26aad13 [ttnn-operation-analyzer] analysis: tilize
```

## Design Deviations

1. **Manual scaler generation**: Used manual bfloat16 tile fill instead of `dataflow_kernel_lib::prepare_reduce_scaler` due to compile-time `get_dataformat` visibility issue
2. **Weight/bias broadcast**: Used `BroadcastDim::NONE` instead of `BroadcastDim::ROW` — weight/bias are expanded to (32, W) on host before tilize, filling all tile rows identically
3. **compute_kernel_hw_startup**: Fixed from 2-arg to 3-arg form for correct CB initialization

## Limitations & Future Work

- **Single-core only**: No multi-core work distribution yet
- **W-small only**: All tiles for one row must fit in L1 (no streaming/re-read variant)
- **Tile-aligned only**: Input width must be a multiple of 32
- **2D input only**: No support for higher-rank tensors or multi-dimensional normalized_shape
