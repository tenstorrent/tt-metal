# Generic Op Builder Execution Log - row_standardize

**Agent**: generic_op_builder
**Operation**: row_standardize
**Start Time**: 2026-02-09
**Workflow**: Generic Op (Python-based)

---

## Phase 1: Specification Analysis

### Input Documents
- `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_standardize/row_standardize_spec.md`

### Key Requirements Extracted
- **Operation type**: Row-wise standardization (LayerNorm without gamma/beta)
- **Math**: `output = (x - mean_row) * rsqrt(var_row + epsilon)`
- **Layout**: ROW_MAJOR input and output
- **Pipeline**: Reader -> Tilize -> Compute (6 phases) -> Untilize -> Writer
- **Work distribution**: Single-core prototype (multi-core deferred)
- **Dtypes**: bfloat16 and float32 support
- **Epsilon**: Runtime parameter (default 1e-5)

### Circular Buffer Strategy
Total CBs: 11
- c_0: cb_rm_in (RM sticks from reader)
- c_1: cb_scaler (reduce scaler 1/W)
- c_2: cb_eps (epsilon scalar)
- c_3: cb_tilized (tilized input)
- c_4: cb_tilized_out (normalized output tiles)
- c_16: cb_rm_out (RM sticks to writer)
- c_24: cb_mean (row means)
- c_25: cb_xmm (x - mean)
- c_26: cb_xmm_sq ((x-mean)^2)
- c_27: cb_var (row variance)
- c_28: cb_invstd (rsqrt(var + eps))

### Work Unit Definition
- One work unit = one tile-row (Wt tiles spanning full W dimension)
- Total work units = `nblocks = (batch_dims * H) / 32`

---

## Phase 2: File Creation

### Files to Create
1. `__init__.py` - Package initialization, re-export row_standardize
2. `row_standardize.py` - Entry point, validation, tensor allocation
3. `row_standardize_program_descriptor.py` - ProgramDescriptor creation
4. `test_row_standardize.py` - Pytest tests with PyTorch reference
5. `kernels/row_standardize_reader.cpp` - Stub reader kernel
6. `kernels/row_standardize_compute.cpp` - Stub compute kernel
7. `kernels/row_standardize_writer.cpp` - Stub writer kernel

---

## Implementation Progress

### Step 1: Create package structure
COMPLETED - Created `ttnn/ttnn/operations/row_standardize/` directory structure

### Step 2: Implement __init__.py
COMPLETED - Created package initialization with row_standardize export

### Step 3: Implement row_standardize.py
COMPLETED - Entry point implementation
- Input validation (rank, layout, dtype, dimension alignment)
- Output tensor allocation using positional args
- Call to create_program_descriptor
- Execution via ttnn.generic_op with output tensor last

### Step 4: Implement row_standardize_program_descriptor.py
COMPLETED - ProgramDescriptor creation
- Extracted tensor metadata (dtype, element_size, tile_size)
- Computed work distribution (Wt, nblocks)
- Packed scaler and epsilon values (bf16 or f32 format)
- Created 11 CB descriptors with correct formats and sizes
- Created reader, compute, writer kernel descriptors
- Set compile-time and runtime args per spec

### Step 5: Implement stub kernels
COMPLETED - Created 3 stub kernel files
- `kernels/row_standardize_reader.cpp` - Stub with full header comments
- `kernels/row_standardize_compute.cpp` - Stub with pipeline description
- `kernels/row_standardize_writer.cpp` - Stub with responsibilities

### Step 6: Implement test_row_standardize.py
COMPLETED - Pytest test suite
- PyTorch reference implementation
- PCC computation helper
- Parameterized tests for all spec shapes and dtypes
- Validation tests (rank, layout, dtype, alignment)
- Edge case test (constant rows / zero variance)
- All torch imports inside functions (not global)

---

## Decisions and Deviations

### Decision 1: CB Page Size for ROW_MAJOR CBs
**Context**: Spec says "Page size = tile_size for the dtype" for cb_rm_in and cb_rm_out, even though they hold RM sticks.
**Decision**: Used `tile_size = tensor.tile.get_tile_size(dtype)` for all CB page sizes, including RM CBs.
**Rationale**: Matches spec guidance. The CB page size represents the granularity for producer-consumer synchronization, not necessarily the physical stick size.

### Decision 2: Intermediate Format for cb_eps
**Context**: Spec says cb_eps uses `intermed_fmt` because epsilon is added to intermediate variance.
**Decision**: Set cb_eps data_format to `intermed_fmt` (Float32 if fp32_dest_acc_en, else Float16_b).
**Rationale**: Follows spec exactly. Variance is in intermediate format, so epsilon must match.

### Decision 3: Bfloat16 Conversion
**Context**: Need to pack scaler and epsilon as bfloat16 for bf16 inputs.
**Decision**: Implemented `_float_to_bfloat16()` helper that truncates lower 16 bits of f32 mantissa.
**Rationale**: Standard bfloat16 conversion. Matches the format expected by `generate_reduce_scaler()`.

### Decision 4: Runtime Args for Compute Kernel
**Context**: Compute kernel needs Wt and nblocks.
**Decision**: Passed as compile-time args (not runtime args).
**Rationale**: These values are constant for a given tensor shape and don't vary per core. Making them compile-time enables better optimization and matches common patterns.

### Decision 5: Test Validation for Non-Aligned Dimensions
**Context**: Spec requires H and W multiples of 32, but ttnn.from_torch may auto-pad.
**Decision**: Implemented validation in row_standardize.py, tested with aligned shapes in tests.
**Rationale**: Our validation catches misaligned inputs. Testing unaligned shapes would require manual tensor construction, which is complex and not critical for stub verification.

### Decision 6: Compute Kernel Runtime Args
**Context**: Compute kernel descriptor expects runtime_args parameter.
**Decision**: Passed empty RuntimeArgs() object.
**Rationale**: Single-core with all work determined by compile-time args. No per-core runtime configuration needed.

---

## Pain Points Encountered

### Pain Point 1: Logging Script JSON Parsing
**Issue**: First attempt to use append_breadcrumb.sh failed due to incorrect JSON format.
**Resolution**: Switched to proper JSON object notation: `'{"event":"phase","name":"..."}'`

### Pain Point 2: ttnn.Shape Slicing
**Context**: Needed to extract output shape from input tensor.
**Issue**: ttnn.Shape objects cannot be sliced with `shape[:]`.
**Resolution**: Used list comprehension: `[shape[i] for i in range(len(shape))]`

### Pain Point 3: Kernel Path Resolution
**Context**: KernelDescriptor.kernel_source expects path relative to tt-metal root.
**Issue**: Initially unclear whether to use absolute or relative paths.
**Resolution**: Used `str(KERNEL_DIR / "filename.cpp")` which resolves to absolute path. This works because Python's Path.resolve() is called internally.

---

## File Summary

### Created Files

| File | Path | Purpose | Status |
|------|------|---------|--------|
| Package init | `ttnn/ttnn/operations/row_standardize/__init__.py` | Export row_standardize | COMPLETE |
| Entry point | `ttnn/ttnn/operations/row_standardize/row_standardize.py` | Validation, allocation, generic_op call | COMPLETE |
| Program descriptor | `ttnn/ttnn/operations/row_standardize/row_standardize_program_descriptor.py` | CB config, kernel setup, runtime args | COMPLETE |
| Tests | `ttnn/ttnn/operations/row_standardize/test_row_standardize.py` | Pytest suite with PyTorch reference | COMPLETE |
| Reader kernel | `ttnn/ttnn/operations/row_standardize/kernels/row_standardize_reader.cpp` | Stub kernel | COMPLETE |
| Compute kernel | `ttnn/ttnn/operations/row_standardize/kernels/row_standardize_compute.cpp` | Stub kernel | COMPLETE |
| Writer kernel | `ttnn/ttnn/operations/row_standardize/kernels/row_standardize_writer.cpp` | Stub kernel | COMPLETE |

### Lines of Code
- Python: ~550 lines (entry point + descriptor + tests)
- C++ stubs: ~80 lines (3 stub kernels with headers)
- Total: ~630 lines

---

## Next Steps (for ttnn-kernel-writer)

The stub kernels need full implementations:

### Reader Kernel Implementation
1. Extract compile-time and runtime args
2. Call `generate_reduce_scaler(cb_scaler, scaler_packed)` once at start
3. Call `generate_bcast_scalar()` for epsilon once at start
4. Loop over blocks:
   - Use TensorAccessor to compute stick addresses
   - Read 32 sticks via `noc_async_read`
   - Push to cb_rm_in

### Compute Kernel Implementation
1. Loop over nblocks:
   - Tilize phase: `compute_kernel_lib::tilize<c_0, c_3>()`
   - Mean reduce: `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>(cb_tilized, cb_scaler, cb_mean, {1, Wt, 1})`
   - Subtract: `sub<COL, NoWaitNoPop>(cb_tilized, cb_mean, cb_xmm, Wt)` + manual pops
   - Square: `square<NoWaitNoPop>(cb_xmm, cb_xmm_sq, Wt)`
   - Var reduce: `reduce<SUM, REDUCE_ROW, BulkWaitBulkPop>(cb_xmm_sq, cb_scaler, cb_var, {1, Wt, 1})`
   - Add+rsqrt: `add_tiles_bcast_scalar` + `rsqrt_tile`
   - Normalize: `mul<COL, NoWaitNoPop>(cb_xmm, cb_invstd, cb_tilized_out, Wt)` + manual pops
   - Untilize phase: `compute_kernel_lib::untilize<Wt, c_4, c_16>()`

### Writer Kernel Implementation
1. Loop over blocks:
   - Wait for Wt pages in cb_rm_out
   - Loop over 32 rows:
     - Compute stick address via TensorAccessor
     - Write stick via `noc_async_write`
   - Barrier and pop cb_rm_out

---

## Completion Status

**FULLY AUTOMATED MODE - NO USER INTERACTION REQUIRED**

All Phase 1-2 deliverables COMPLETED:
- [x] Package structure created
- [x] Entry point implemented
- [x] Program descriptor implemented
- [x] Stub kernels created
- [x] Test suite implemented
- [x] Execution log maintained

**Agent Status**: READY FOR HANDOFF to ttnn-kernel-writer
