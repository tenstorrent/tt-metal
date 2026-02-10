# Generic Op Builder Summary - layer_norm_rm

**Operation**: layer_norm_rm
**Date**: 2026-02-10
**Agent**: generic_op_builder
**Status**: ✅ COMPLETE

## Deliverables

All required files have been created and tested:

### Python Infrastructure

1. **`__init__.py`**
   - Re-exports main `layer_norm_rm` function
   - Allows import as `from ttnn.operations.layer_norm_rm import layer_norm_rm`

2. **`layer_norm_rm.py`** (Entry Point)
   - Main user-facing function
   - Input validation (layout, dtype, dimensions, gamma/beta shapes)
   - Output tensor allocation using `ttnn.allocate_tensor_on_device()` with positional args
   - Calls `ttnn.generic_op([input, gamma, beta, output], program_descriptor)`
   - Output tensor is correctly placed LAST in io_tensors list

3. **`layer_norm_rm_program_descriptor.py`** (Program Descriptor)
   - Extracts tensor metadata (shape, dtype, element_size)
   - Calculates work distribution (single-core, processes tile-rows)
   - Configures 17 circular buffers:
     - c_0: Input RM sticks (Wt tiles)
     - c_1: Reduce scaler (1/W, 1 tile)
     - c_2: Input tilized (Wt tiles)
     - c_3: Gamma RM sticks (Wt tiles)
     - c_4: Beta RM sticks (Wt tiles)
     - c_5: Gamma tilized (Wt tiles, persistent)
     - c_6: Beta tilized (Wt tiles, persistent)
     - c_7: Epsilon scalar (1 tile)
     - c_16: Output RM sticks (Wt tiles)
     - c_24: Mean (1 tile)
     - c_25: Centered (Wt tiles)
     - c_26: Squared (Wt tiles)
     - c_27: Variance (1 tile)
     - c_28: Rstd (1 tile)
     - c_29: Normalized (Wt tiles)
     - c_30: Gamma applied (Wt tiles)
     - c_31: Output tilized (Wt tiles)
   - Packs reduce scaler (1/W) and epsilon scalar in correct format:
     - bfloat16: `(bf16 << 16 | bf16)` format
     - float32: raw float32 bits
   - Creates kernel descriptors for reader, writer, compute
   - Sets compile-time and runtime args correctly
   - Uses `ttnn.ComputeConfigDescriptor()` for compute kernel config

### Stub Kernels

4. **`kernels/layer_norm_rm_reader.cpp`**
   - Stub reader kernel with correct includes (`api/dataflow/dataflow_api.h`)
   - Will be implemented by kernel-writer agent

5. **`kernels/layer_norm_rm_compute.cpp`**
   - Stub compute kernel with correct includes (`compute_kernel_api/common.h`)
   - Will be implemented by kernel-writer agent

6. **`kernels/layer_norm_rm_writer.cpp`**
   - Stub writer kernel with correct includes (`api/dataflow/dataflow_api.h`)
   - Will be implemented by kernel-writer agent

### Tests

7. **`test_layer_norm_rm.py`**
   - Comprehensive test suite with PyTorch reference comparison
   - Test shapes: [1,1,32,32], [1,1,128,128], [1,1,32,1024], [1,1,1024,32], [1,1,4096,32], [1,1,512,512], [2,3,64,128], [1,64,128]
   - Both bfloat16 and float32 dtypes tested
   - PCC > 0.99 threshold for correctness (per spec)
   - Validation tests for layout and dimension checks
   - Uses `device` fixture (never opens device manually)
   - Torch imports inside functions (not global)

## Test Results

### ✅ Infrastructure Tests (PASS)

**Test**: `test_layer_norm_rm_runs[minimal]`
- **Status**: PASS
- **Result**: Operation runs without errors, shape preserved
- **Significance**: Proves that:
  - Program descriptor is created correctly
  - All CBs are allocated without errors
  - Stub kernels compile successfully
  - `ttnn.generic_op()` executes without hanging
  - Output tensor allocation works correctly

### ⚠️ Correctness Tests (EXPECTED FAIL)

**Test**: `test_layer_norm_rm_correctness` (all 14 variants)
- **Status**: EXPECTED FAIL
- **Reason**: Stub kernels don't implement computation logic
- **Observed**: PCC values are low/NaN, large output differences
- **Expected**: Will pass after kernel-writer agent implements the kernels

### 🐛 Issues Found and Fixed

1. **Struct Import Issue** (Fixed)
   - Problem: `import struct` was inside if-block, causing UnboundLocalError in else-block
   - Solution: Moved import to module level (already present, removed duplicate)
   - Status: ✅ Fixed

## Key Design Decisions

1. **Single-Core Execution**: All work on core (0,0), processes tile-rows sequentially
2. **WSmall Pattern**: Entire tile-row (Wt tiles) loaded at once, no streaming
3. **Persistent Gamma/Beta**: Tilized once, reused for all rows (not popped)
4. **Scalar Format**: Correct packing for both bfloat16 and float32
5. **CB Allocation**: 17 CBs with correct sizes, formats, and indices

## API Critical Notes Applied

- ✅ `allocate_tensor_on_device()` uses positional args, not keyword args
- ✅ Compute kernel uses `ttnn.ComputeConfigDescriptor()`, not `ttnn.ComputeConfig()`
- ✅ Dataflow kernels use full path: `api/dataflow/dataflow_api.h`
- ✅ Output tensor is LAST in `generic_op([..., output_tensor], ...)`
- ✅ Data formats use `ttnn.DataType` values directly (no `ttnn.DataFormat` enum)
- ✅ Runtime args set for all cores in grid (single core, but follows pattern)

## File Paths Created

All files at: `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/`

1. `__init__.py`
2. `layer_norm_rm.py`
3. `layer_norm_rm_program_descriptor.py`
4. `test_layer_norm_rm.py`
5. `kernels/layer_norm_rm_reader.cpp`
6. `kernels/layer_norm_rm_compute.cpp`
7. `kernels/layer_norm_rm_writer.cpp`
8. `agent_logs/generic_op_builder_breadcrumbs.jsonl`
9. `agent_logs/generic_op_builder_summary.md` (this file)

## Next Steps

The operation is ready for the **kernel-writer** agent to implement the actual kernel logic:

1. **Reader kernel**: Generate scalers, read gamma/beta, read input sticks
2. **Compute kernel**: Tilize, layer norm computation (mean, center, variance, rsqrt, normalize, gamma/beta), untilize
3. **Writer kernel**: Write output sticks to DRAM

After kernel implementation, the correctness tests should pass with PCC > 0.99.

## Validation Checklist

- [x] All required files created
- [x] Entry point validates inputs correctly
- [x] Program descriptor creates all CBs
- [x] Scalar values packed correctly for both dtypes
- [x] Kernel descriptors configured correctly
- [x] Stub kernels compile successfully
- [x] Tests run without hanging
- [x] Shape preservation verified
- [x] Struct import issue fixed
- [x] Test uses device fixture (not manual open)
- [x] Torch imports inside functions

## Logging

Execution breadcrumbs logged to: `agent_logs/generic_op_builder_breadcrumbs.jsonl`

Events logged:
- start
- read_spec
- read_template
- create_files
- fix_struct_import
- test_execution (runs, correctness)
- complete
