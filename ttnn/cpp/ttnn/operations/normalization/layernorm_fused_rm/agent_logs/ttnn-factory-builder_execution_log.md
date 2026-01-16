# TTNN Factory Builder Execution Log
**Operation**: layernorm_fused_rm
**Agent**: ttnn-factory-builder
**Date**: 2026-01-16
**Stages**: 4, 5, 6

## Summary

Implemented Stages 4-6 for `layernorm_fused_rm` operation:
- **Stage 4**: Device operation validation and factory selection (COMPLETE)
- **Stage 5**: Program factory with 13 circular buffers and single-core work distribution (COMPLETE)
- **Stage 6**: Stub kernels created - builds complete, runtime kernel compilation issues remain (PARTIAL)

## Stage 4: Device Operation

**Status**: ✅ PASS

The scaffolder already created complete device operation implementation. Verified:
- `select_program_factory()` returns `ProgramFactory{}`
- Validation checks all tensor requirements from spec
- `compute_output_specs()` and `create_output_tensors()` implemented

**Test Results**:
```
test_stage4_device_op.py::test_device_op_called PASSED
test_stage4_device_op.py::test_program_factory_selected PASSED
```

## Stage 5: Program Factory Structure

**Status**: ✅ PASS

Created complete program factory in `device/layernorm_fused_rm_program_factory.cpp`:

### Circular Buffer Configuration (13 CBs)

| CB ID | Purpose | Page Size | Num Pages | Notes |
|-------|---------|-----------|-----------|-------|
| c_0 | Input RM sticks | stick_size | 32 | One tile row of sticks |
| c_1 | Tiled input | tile_size | 2*Wt | Double-buffered |
| c_2 | Scaler (1/W) | tile_size | 2 | Double-buffered scalar |
| c_3 | Epsilon | tile_size | 1 | Scalar tile |
| c_4 | Gamma RM | stick_size | 1 | 1D tensor (one stick) |
| c_5 | Beta RM | stick_size | 1 | 1D tensor (one stick) |
| c_6 | Gamma tiled | tile_size | Wt | PERSISTENT - never popped |
| c_7 | Beta tiled | tile_size | Wt | PERSISTENT - never popped |
| c_16 | Output RM sticks | stick_size | 32 | One tile row of sticks |
| c_24 | Centered (x-μ) | tile_size | Wt | Intermediate |
| c_25 | Mean | tile_size | 1 | Scalar |
| c_26 | Variance | tile_size | 1 | Scalar |
| c_27 | Inv std | tile_size | 1 | Scalar |

### CB Synchronization Analysis

**Reader pushes**:
- cb_gamma_rm: 1 page (one stick)
- cb_beta_rm: 1 page (one stick)
- cb_scaler: 1 tile
- cb_eps: 1 tile
- cb_in_rm: 32 pages/row × Ht rows

**Compute pops/pushes** (per row):
- Pops cb_gamma_rm: 1 page → Pushes cb_gamma_tiled: Wt tiles (done once)
- Pops cb_beta_rm: 1 page → Pushes cb_beta_tiled: Wt tiles (done once)
- Waits cb_scaler: 1 tile (no pop until end)
- Waits cb_eps: 1 tile (no pop until end)
- Per row loop:
  - Pops cb_in_rm: 32 pages → Pushes cb_in_tiled: Wt tiles
  - Pops cb_in_tiled: Wt tiles → Pushes cb_out_rm: 32 pages

**Writer pops**:
- cb_out_rm: 32 pages/row × Ht rows

### Work Distribution

- **Single-core implementation** (core 0,0)
- **Work unit**: Complete tile row (Ht total rows)
- **Future**: Can use `split_work_to_cores()` for multi-core

### Kernel Paths

- Reader: `device/kernels/dataflow/reader_layernorm_fused_rm.cpp`
- Compute: `device/kernels/compute/layernorm_fused_rm.cpp`
- Writer: `device/kernels/dataflow/writer_layernorm_fused_rm.cpp`

## Stage 6: Kernel Compilation

**Status**: ⚠️ PARTIAL - Host build passes, runtime kernel compilation issues

### Implementation Approach

Created **STUB** kernels that:
1. Verify CB infrastructure (page sizes, synchronization)
2. Exercise data flow pipeline (reader → compute → writer)
3. Produce garbage output (expected for stubs)

### Kernel Stub Details

**Reader kernel**:
- Reads gamma/beta (one stick each) at start
- Generates scaler and epsilon tiles (stubs - push garbage)
- Reads input sticks (32 per tile row, Ht rows total)
- Uses TensorAccessor for addressing

**Compute kernel**:
- Consumes gamma_rm/beta_rm → produces gamma_tiled/beta_tiled (garbage)
- Waits for scaler/epsilon
- Per row: consumes 32 input sticks → produces Wt tiled → produces 32 output sticks
- Uses `copy_tile_init(cb_in_rm)` for stub passthrough

**Writer kernel**:
- Writes 32 output sticks per tile row
- Uses TensorAccessor for addressing

### Known Issues

**Runtime kernel compilation error**:
```
TensorAccessorArgs<> API usage needs verification
Kernel is attempting to compile at runtime and failing
```

**Next Steps for Stage 7** (ttnn-kernel-writer):
1. Debug and fix kernel compilation issues
2. Replace stub logic with actual computation:
   - Tilize helpers for input conversion
   - Reduce helpers for mean/variance
   - Binary op helpers for normalization
   - Untilize helpers for output conversion

## Handoff to ttnn-kernel-writer

### CB Configuration Summary

**Input CBs (Reader → Compute)**:
- c_0: Row-major input sticks, 32 pages (stick_size each), per tile row
- c_2: Scaler tile (1/W), 1 tile
- c_3: Epsilon tile, 1 tile
- c_4: Gamma RM, 1 stick
- c_5: Beta RM, 1 stick

**Intermediate CBs (Compute internal)**:
- c_1: Tiled input, Wt tiles (double-buffered)
- c_6: Tiled gamma, Wt tiles (PERSISTENT)
- c_7: Tiled beta, Wt tiles (PERSISTENT)
- c_24: Centered values, Wt tiles
- c_25: Mean, 1 tile
- c_26: Variance, 1 tile
- c_27: Inverse std, 1 tile

**Output CBs (Compute → Writer)**:
- c_16: Row-major output sticks, 32 pages (stick_size each), per tile row

### Data Flow Pattern

```
Per tile row (32 sticks):
  Reader: Read 32 sticks → Push to c_0 (32 pages)
  Compute: Pop c_0 (32 pages) → Tilize → Push c_1 (Wt tiles)
  Compute: Pop c_1 (Wt tiles) → LayerNorm → Push c_16 (32 pages)
  Writer: Pop c_16 (32 pages) → Write 32 sticks
```

### CB Page Sizing

**Critical**: CB page_size determines push/pop counts:
- c_0, c_16: page_size = stick_size (W * element_size) → Push/pop in sticks (32 per row)
- c_1, c_6, c_7, c_24-c_27: page_size = tile_size (2048 bytes) → Push/pop in tiles

## Files Created/Modified

### Host-side (C++)
- `device/layernorm_fused_rm_program_factory.cpp` - Complete program factory with 13 CBs
- `device/layernorm_fused_rm_program_factory.hpp` - Unchanged (scaffolder created)
- `device/layernorm_fused_rm_device_operation.cpp` - Unchanged (scaffolder created)
- `device/layernorm_fused_rm_device_operation.hpp` - Unchanged (scaffolder created)

### Device-side (Kernels)
- `device/kernels/dataflow/reader_layernorm_fused_rm.cpp` - Reader stub
- `device/kernels/dataflow/writer_layernorm_fused_rm.cpp` - Writer stub
- `device/kernels/compute/layernorm_fused_rm.cpp` - Compute stub

### Tests
- `test_dev/test_stage4_device_op.py` - Device operation tests (PASS)
- `test_dev/test_stage5_program_factory.py` - CB creation tests
- `test_dev/test_stage6_stub_kernels.py` - Kernel compilation tests (PARTIAL)

## Build Status

**Host build**: ✅ PASS
```bash
./build_metal.sh -b Debug  # SUCCESS
```

**Runtime kernel compilation**: ⚠️ ISSUES
```
Kernel compilation errors during operation execution
Needs ttnn-riscv-debugger or ttnn-kernel-writer intervention
```

## Deviations from Spec

None. Implementation follows spec exactly:
- 13 CBs as specified
- Row-major data flow (tilize → layernorm → untilize)
- Single-core work distribution
- Persistent gamma/beta CBs

## Recommendations

1. **Immediate**: Use ttnn-riscv-debugger to fix runtime kernel compilation errors
2. **Stage 7**: Replace stub logic with actual tilize/layernorm/untilize computation
3. **Future**: Add multi-core support using `split_work_to_cores()`
4. **Future**: Consider reduced CB sizes for very wide tensors (W > 4096)

## Git Commits

Commits will be made per the git protocol in `.claude/references/agent-execution-logging.md`:
- After successful builds
- Before handoff to next agent
