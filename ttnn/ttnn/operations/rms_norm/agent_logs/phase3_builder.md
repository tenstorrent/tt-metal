# ttnn-generic-op-builder Execution Log: rms_norm

## 1. Input Interpretation

- **Operation**: rms_norm
- **Math**: RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma
- **Supported dtypes**: bfloat16, float32
- **Layouts**: TILE_LAYOUT and ROW_MAJOR_LAYOUT (in-kernel tilize/untilize)
- **Gamma**: Always ROW_MAJOR_LAYOUT shape (1,1,1,W), tilized in-kernel

### Upstream Issues (architect's stage tests)
- Stage test files had `Markup('"TILE_LAYOUT"')` instead of `"TILE_LAYOUT"` (Python syntax error)
- `pytorch_reference()` functions had undefined defaults (`layout=layout` at module scope)
- Relative imports `from .rms_norm import rms_norm` wouldn't work from test directory
- Gamma test referenced `ttnn_gamma` before definition in reference call
- All stage tests passed `layout=layout` to rms_norm which isn't a parameter

All issues fixed in the stage test rewrites.

## 2. Execution Timeline

### Attempt 1: Initial implementation
1. Read op_design.md and .tdd_state.json
2. Read template files in .claude/references/generic_op_template/
3. Created all 9 files (entry point, program descriptor, __init__, 3 kernels, integration test, test __init__)
4. Fixed all 4 stage test files
5. Ran tests -> FAILED: TensorAccessor include path wrong

### Attempt 2: Fix include path
- Changed `#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` to `#include "api/tensor/tensor_accessor.h"` in reader and writer stubs
- Ran tests -> 8/8 PASSED

## 2a. Circular Buffer Configuration

| CB ID | Page Size | Num Pages | Data Format | Purpose |
|-------|-----------|-----------|-------------|---------|
| 0 (c_0) | tile_size | Wt (RM) / 1 (TILE) | input_dtype | RM sticks for tilize |
| 1 (c_1) | tile_size | Wt | input_dtype | Tilized input (persistent) |
| 2 (c_2) | bf16_tile_size | 1 | bfloat16 | Reduce scaler (1/W) |
| 3 (c_3) | tile_size | 2 | input_dtype | Squared tiles (double-buf) |
| 4 (c_4) | tile_size | 2 | input_dtype | Reduce output (double-buf) |
| 5 (c_5) | tile_size | 1 | input_dtype | Epsilon constant |
| 6 (c_6) | tile_size | 1 | input_dtype | rsqrt(mean+eps) |
| 7 (c_7) | tile_size | Wt (gamma) / 1 | input_dtype | Gamma weights |
| 16 (c_16) | tile_size | Wt | input_dtype | Output tiles |
| 17 (c_17) | tile_size | Wt (RM) / 1 (TILE) | input_dtype | Untilized output |

### Work Distribution

| Parameter | Value | Source |
|-----------|-------|--------|
| Core grid | compute_with_storage_grid_size | Device |
| Total work units | Ht_total = product(shape[:-1]) / 32 | Design doc |
| Work per core | split_work_to_cores() | ttnn API |

### Files Created

| File | Type | Purpose |
|------|------|---------|
| ttnn/ttnn/operations/rms_norm/__init__.py | Package init | Re-exports rms_norm |
| ttnn/ttnn/operations/rms_norm/rms_norm.py | Entry point | Validation, allocation, generic_op call |
| ttnn/ttnn/operations/rms_norm/rms_norm_program_descriptor.py | Program descriptor | 10 CBs, 3 kernels, runtime args |
| ttnn/ttnn/operations/rms_norm/kernels/rms_norm_reader.cpp | Kernel stub | Empty, includes TensorAccessor + reduce_helpers |
| ttnn/ttnn/operations/rms_norm/kernels/rms_norm_compute.cpp | Kernel stub | Empty, includes tilize/untilize/reduce/binary/rsqrt |
| ttnn/ttnn/operations/rms_norm/kernels/rms_norm_writer.cpp | Kernel stub | Empty, includes TensorAccessor |
| tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py | Integration test | 8 tests: shapes, gamma, RM, validation |
| tests/ttnn/unit_tests/operations/rms_norm/__init__.py | Package init | Required for pytest |

### Test Results

| Test | Result | Notes |
|------|--------|-------|
| Stub compiles | PASS | All 3 kernels compile at runtime |
| generic_op executes | PASS | No hang with empty stubs |
| Output shape correct | PASS | All shapes verified |
| Gamma path | PASS | gamma tensor passed correctly |
| RM layout path | PASS | ROW_MAJOR_LAYOUT works |
| Validation: rank | PASS | Test exists (trivial) |
| Validation: gamma mismatch | PASS | RuntimeError raised correctly |

## 3. Recovery Summary

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Kernel compile failure | Wrong TensorAccessor include path for device kernels | Changed to `api/tensor/tensor_accessor.h` |

## 6. Handoff Notes for Kernel Writer

- Kernel stubs are completely empty (just includes + empty kernel_main)
- The TDD stages are: data_pipeline -> square_reduce_rsqrt -> normalize -> gamma
- Stage test files have been fixed and are ready to use
- CB indices match design doc exactly
- Compile-time args layout documented in program descriptor comments
- Runtime args: [buffer_addr, start_id, num_rows, (gamma_addr for reader)]
- Compute runtime args: [rows_per_core] (actual Ht for that core)
- FP32 accumulation enabled when input dtype is float32

## 8. Git Commit History

| SHA | Message |
|-----|---------|
| 78d5c28c6f | [ttnn-generic-op-builder] stubs: rms_norm |
