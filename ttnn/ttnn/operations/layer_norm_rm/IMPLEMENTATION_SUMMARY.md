# Layer Norm RM - Generic Op Implementation Summary

## Operation Name
`layer_norm_rm`

## Implementation Status
**Phase**: Python orchestration complete, kernels are stubs
**Date**: 2026-02-10
**Agent**: generic_op_builder

## Files Created

### 1. Python Orchestration Files

#### `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/__init__.py`
- **Purpose**: Package initialization, re-exports main function
- **Lines**: 7
- **Status**: Complete

#### `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/layer_norm_rm.py`
- **Purpose**: Entry point with input validation, output allocation, and generic_op call
- **Lines**: 155
- **Key Features**:
  - Full input validation (layout, dtype, shape constraints, device)
  - Gamma/beta validation (optional parameters)
  - Output tensor allocation with correct shape/dtype/layout
  - Proper io_tensors list construction (output last)
- **Status**: Complete

#### `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/layer_norm_rm_program_descriptor.py`
- **Purpose**: ProgramDescriptor creation with CB config, kernel setup, and runtime args
- **Lines**: 502
- **Key Features**:
  - 16 circular buffers configured according to spec
  - Single-core execution (core 0,0)
  - Reduce scaler and epsilon scalar packing functions
  - TensorAccessor integration for all tensors
  - Proper data format handling (bfloat16 vs float32)
  - Intermediate precision handling (float32 for better accuracy)
  - CB 6 (reduce scaler) always configured as bfloat16
- **Status**: Complete

### 2. Test Files

#### `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/test_layer_norm_rm.py`
- **Purpose**: Pytest-based tests with PyTorch reference comparison structure
- **Lines**: 194
- **Test Coverage**:
  - 8 shapes: [1,1,32,32], [1,1,128,128], [1,1,32,1024], [1,1,1024,32], [1,1,4096,32], [1,1,512,512], [2,3,64,128], [1,64,128]
  - 2 dtypes: bfloat16, float32
  - 2 gamma configurations: has_gamma=True/False
  - 2 beta configurations: has_beta=True/False
  - Total: 160 test combinations
  - Validation tests for dtype mismatch
- **Features**:
  - Uses `device` fixture (never opens device manually)
  - All parameters in pytest parametrizations
  - torch imports inside functions (not global)
  - PCC-based correctness checks (commented out for stubs)
  - Shape, layout, dtype verification
- **Status**: Complete (numerical checks commented out until kernels implemented)

### 3. Kernel Files (Stubs)

#### `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/kernels/layer_norm_rm_reader.cpp`
- **Purpose**: Read input, gamma, beta from DRAM; generate scaler tiles
- **Compile-time args**: stick_size, gamma_beta_stick_size, TensorAccessor args (input, gamma, beta)
- **Runtime args**: src_addr, gamma_addr, beta_addr, num_sticks, num_tile_rows, Wt, reduce_scaler, eps_scalar
- **Status**: Stub (compiles, but no implementation)

#### `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/kernels/layer_norm_rm_compute.cpp`
- **Purpose**: Tilize, layer norm computation, untilize
- **Compile-time args**: Wt, num_tile_rows, has_gamma, has_beta
- **Runtime args**: None
- **Status**: Stub (compiles, but no implementation)

#### `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/kernels/layer_norm_rm_writer.cpp`
- **Purpose**: Write output sticks to DRAM
- **Compile-time args**: output_stick_size, tile_height, num_tile_rows, Wt, TensorAccessor args (output)
- **Runtime args**: dst_addr
- **Status**: Stub (compiles, but no implementation)

## Circular Buffer Configuration

Total: 16 CBs configured

| CB ID | Name | Size | Purpose | Data Format |
|-------|------|------|---------|-------------|
| 0 | cb_input_rm | Wt tiles | Input row-major sticks | input dtype |
| 1 | cb_tilized_input | Wt tiles | Tilized input tiles | input dtype |
| 2 | cb_gamma_rm | Wt tiles | Gamma row-major sticks | input dtype |
| 3 | cb_gamma_tilized | Wt tiles | Gamma tilized (persistent) | input dtype |
| 4 | cb_beta_rm | Wt tiles | Beta row-major sticks | input dtype |
| 5 | cb_beta_tilized | Wt tiles | Beta tilized (persistent) | input dtype |
| 6 | cb_reduce_scaler | 1 tile | Reduce scaler (1/W) | **bfloat16 always** |
| 7 | cb_eps_scalar | 1 tile | Epsilon scalar | input dtype |
| 8 | cb_output_tiles | Wt tiles | Final output tiles | input dtype |
| 16 | cb_output_rm | Wt tiles | Output row-major sticks | input dtype |
| 24 | cb_mean | 1 tile | Row-wise mean | float32 |
| 25 | cb_centered | Wt tiles | x - mean | float32 |
| 26 | cb_centered_sq | Wt tiles | (x - mean)^2 | float32 |
| 27 | cb_var | 1 tile | Row-wise variance | float32 |
| 28 | cb_rstd | 1 tile | 1/sqrt(var+eps) | float32 |
| 29 | cb_normed | Wt tiles | x_centered * rstd | float32 |
| 30 | cb_gamma_applied | Wt tiles | gamma * normed | float32 |

**Note**: Intermediate CBs (24-30) use float32 format for better precision, even when input is bfloat16.

## Work Distribution

- **Parallelization**: Single-core (core 0,0)
- **Work unit**: One tile-row (32 rows, Wt tiles)
- **Total work units**: num_tile_rows = num_sticks / 32
- **Loop structure**: Sequential processing of tile-rows

## API Usage

### Entry Point
```python
from ttnn.operations.layer_norm_rm import layer_norm_rm

output = layer_norm_rm(
    input_tensor,      # ROW_MAJOR, bfloat16 or float32, on device
    gamma=gamma,       # Optional, same dtype as input
    beta=beta,         # Optional, same dtype as input
    epsilon=1e-5,      # Default 1e-5
    device=device,     # Optional, defaults to input's device
    memory_config=ttnn.DRAM_MEMORY_CONFIG  # Optional
)
```

### Input Requirements
- Layout: ROW_MAJOR
- Dtype: bfloat16 or float32
- Device: Must be on device
- Memory: DRAM, interleaved
- Shape: Rank >= 2, H and W must be multiples of 32

### Output Specification
- Shape: Same as input
- Dtype: Same as input
- Layout: ROW_MAJOR
- Memory: DRAM, interleaved

## Test Execution

Run tests using the dev-test script:
```bash
.claude/scripts/dev-test.sh /localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/test_layer_norm_rm.py
```

**Note**: Tests will pass shape/layout/dtype checks but numerical validation is commented out until kernels are implemented.

## Next Steps

### Immediate (Kernel Implementation)
1. **Reader kernel**:
   - Implement `generate_reduce_scaler()` call for CB 6
   - Implement `generate_bcast_scalar()` call for CB 7
   - Implement gamma/beta stick reading with 32-row replication for tilize
   - Implement main loop with TensorAccessor-based stick reading

2. **Compute kernel**:
   - Implement gamma/beta tilize (one-time, at program start)
   - Implement per-tile-row loop:
     - Tilize input (CB 0 -> CB 1)
     - Reduce for mean (CB 1, CB 6 -> CB 24)
     - Subtract mean (CB 1, CB 24 -> CB 25)
     - Square (CB 25 -> CB 26)
     - Reduce for variance (CB 26, CB 6 -> CB 27)
     - Add epsilon (CB 27, CB 7 -> CB 27)
     - Rsqrt (CB 27 -> CB 28)
     - Multiply by rstd (CB 25, CB 28 -> CB 29)
     - Multiply by gamma (CB 29, CB 3 -> CB 30) if has_gamma
     - Add beta (CB 30, CB 5 -> CB 8) if has_beta
     - Untilize (CB 8 -> CB 16)

3. **Writer kernel**:
   - Implement main loop with TensorAccessor-based stick writing
   - Write 32 sticks per tile-row from CB 16 to DRAM

### Follow-up (Optimization)
- Multi-core work distribution for large tensors
- L1 capacity validation
- WLarge pattern for very large W
- Optional fp32_dest_acc_en configuration

## Critical Implementation Notes

1. **Reduce scaler CB 6 must ALWAYS be bfloat16**, regardless of input dtype
2. **Output tensor must be LAST in io_tensors list** for generic_op
3. **allocate_tensor_on_device uses positional args**, not keyword args
4. **Intermediate CBs (24-30) use float32** for better precision
5. **Gamma/beta are tilized once** and persist for all tile-rows (CBs 3 and 5)
6. **Compute kernel uses ComputeConfigDescriptor**, not ComputeConfig
7. **Reader/writer includes use full path**: `api/dataflow/dataflow_api.h`

## References

- Spec: `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/layer_norm_rm_spec.md`
- Template: `.claude/references/generic_op_template/`
- API Reference: `.claude/skills/ttnn-generic-op/SKILL.md`
- Working Examples:
  - `tests/ttnn/unit_tests/operations/debug/test_generic_op.py`
  - `models/demos/deepseek_v3_b1/micro_ops/rmsnorm/op.py`

## Execution Log

Breadcrumbs: `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/agent_logs/generic_op_builder_breadcrumbs.jsonl`
