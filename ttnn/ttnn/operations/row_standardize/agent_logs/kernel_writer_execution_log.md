# Agent Execution Log: ttnn-kernel-writer

## Metadata
| Field | Value |
|-------|-------|
| Operation | `row_standardize` |
| Agent | `ttnn-kernel-writer` |
| Stages | Kernel implementation, testing |
| Input | `kernel_design.md`, `row_standardize_spec.md`, `row_standardize_program_descriptor.py` |
| Predecessor | `ttnn-kernel-designer`, `ttnn-generic-op-builder` |
| Final Status | SUCCESS |
| Total Attempts | 4 (1 build error fix, 1 data format fix, 1 scaler format fix, 1 test fix) |

---

## 1. Input Interpretation

### Spec/Input Fields Extracted

| Field | Value | Confidence | Notes |
|-------|-------|------------|-------|
| CB indices | c_0=RM_in, c_1=scaler, c_2=eps, c_3=tilized, c_4=tilized_out, c_16=RM_out, c_24-c_28=intermediates | HIGH | Explicitly in design |
| Phase 1 (Tilize) | USE HELPER: compute_kernel_lib::tilize | HIGH | Explicit |
| Phase 2 (Mean) | USE HELPER: reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop> | HIGH | Explicit |
| Phase 3 (Sub) | USE HELPER: sub<COL, WaitUpfrontPopAtEnd, WaitAndPopPerTile, Bulk> | HIGH | Explicit |
| Phase 4 (Square) | USE HELPER: square<WaitUpfrontNoPop, Bulk> | HIGH | Explicit |
| Phase 5 (Var) | USE HELPER: reduce<SUM, REDUCE_ROW, BulkWaitBulkPop> | HIGH | Explicit |
| Phase 6 (Add+Rsqrt) | NO HELPER: raw add_tiles_bcast_scalar + rsqrt_tile | HIGH | Explicit |
| Phase 7 (Normalize) | USE HELPER: mul<COL, WaitUpfrontPopAtEnd, WaitAndPopPerTile, Bulk> | HIGH | Explicit |
| Phase 8 (Untilize) | USE HELPER: compute_kernel_lib::untilize | HIGH | Explicit |
| Scaler CB format | Not specified -- design says "dtype" | MEDIUM | Required inference |
| Epsilon CB format | intermed_fmt | HIGH | Explicit |

### Interpretation Issues

1. **Scaler CB data format**: The design document specifies the scaler CB (c_1) should use `dtype` format. However, `generate_reduce_scaler` always writes packed bf16 values (assertion enforces `(scaler >> 16) == (scaler & 0xFFFF)`). The softmax reference implementation always uses `Float16_b` for the scaler CB. The design doc is wrong here -- the scaler CB MUST be bfloat16 regardless of input dtype.

2. **Program descriptor used `ttnn.DataFormat.Float16_b`** which does not exist in the Python API. `CBFormatDescriptor` accepts `DataType` enum values like `ttnn.bfloat16` and `ttnn.float32`.

### Upstream Feedback

| Target Agent | Issue | Suggestion | Severity |
|--------------|-------|------------|----------|
| ttnn-kernel-designer | Scaler CB format specified as "dtype" but must always be bfloat16 | Explicitly state: "cb_scaler always uses bfloat16 format regardless of input dtype, matching softmax reference" | HIGH |
| ttnn-generic-op-builder | Used `ttnn.DataFormat.Float16_b` which doesn't exist | Use `ttnn.bfloat16` / `ttnn.float32` (DataType enum) for CBFormatDescriptor | HIGH |
| ttnn-generic-op-builder | Scaler packed as float32 reinterpreted bits for f32 input | Scaler must always be packed as bf16 (bf16 << 16 | bf16), matching generate_reduce_scaler's assertion | HIGH |

---

## 2. Execution Timeline

### Phase: Kernel Implementation

#### Attempt 1: Initial implementation of all 3 kernels
| Field | Value |
|-------|-------|
| Action | Wrote reader, compute, writer kernels following design doc |
| Expected | Kernels compile and produce correct output |
| Actual | Build error: `compute_kernel_api/eltwise_unary/sfpu_split_work.h: No such file or directory` |
| Result | FAIL |

- **Error Type**: build_error
- **Error Summary**: Non-existent header included in compute kernel
- **Root Cause Hypothesis**: H0: sfpu_split_work.h does not exist; only rsqrt.h is needed
- **Evidence**: File not found in include path
- **Recovery Action**: Removed the spurious include

#### Attempt 2: After removing bad include
| Field | Value |
|-------|-------|
| Action | Ran minimal tests (bf16 + f32) |
| Expected | Basic correctness |
| Actual | Program descriptor crash: `ttnn.DataFormat` does not exist |
| Result | FAIL |

- **Error Type**: test_fail
- **Error Summary**: `AttributeError: module 'ttnn' has no attribute 'DataFormat'`
- **Root Cause Hypothesis**: H1: CBFormatDescriptor uses DataType enum, not DataFormat
- **Evidence**: `ttnn.bfloat16` is `DataType.BFLOAT16`, not `DataFormat`
- **Recovery Action**: Changed `ttnn.DataFormat.Float32` -> `ttnn.float32`, `ttnn.DataFormat.Float16_b` -> `ttnn.bfloat16`

#### Attempt 3: After fixing DataFormat
| Field | Value |
|-------|-------|
| Action | Ran minimal tests (bf16 + f32) |
| Expected | Basic correctness |
| Actual | Both minimal tests PASS; Full correctness test: 11 bf16 PASS, f32 FAIL with PCC=0.69 |
| Result | PARTIAL |

- **Error Type**: wrong_output
- **Error Summary**: Float32 output has PCC=0.69 and max_diff=951 vs reference
- **Root Cause Hypothesis**: H2: `generate_reduce_scaler` always writes packed bf16 values. When cb_scaler is configured as float32, hardware interprets the packed bf16 bits as float32, producing garbage.
- **Evidence**: Softmax reference always uses `Float16_b` for scaler CB; `generate_reduce_scaler` has assertion `(scaler >> 16) == (scaler & 0xFFFF)` enforcing bf16 packing
- **Recovery Action**: Changed cb_scaler to always use bfloat16 format + bf16 tile size. Changed scaler packing to always use bf16 format.

#### Attempt 4: After fixing scaler format
| Field | Value |
|-------|-------|
| Action | Ran full test suite |
| Expected | All 29 tests pass |
| Actual | 28 pass, 1 fail (validation_dtype test -- bfloat8_b typecast fails) |
| Result | PARTIAL |

- **Error Type**: test_fail
- **Error Summary**: `ValueError: datum for bfp2, bfp4, bfp8 is invalid` when trying to typecast to bfloat8_b
- **Root Cause Hypothesis**: Pre-existing test issue -- typecast to bfloat8_b requires TILE_LAYOUT
- **Recovery Action**: Fixed test to use TILE_LAYOUT for the typecast step

#### Attempt 5: Final run
| Field | Value |
|-------|-------|
| Action | Ran full test suite |
| Expected | All 29 tests pass |
| Actual | ALL 29 TESTS PASS |
| Result | PASS |

---

### 2a. Design Document Compliance

#### Helper Usage Compliance

| Phase | Design Directive | Implementation | Compliant? |
|-------|------------------|----------------|------------|
| 1 Tilize | USE HELPER: tilize | `compute_kernel_lib::tilize<c_0, c_3>(Wt, 1)` | YES |
| 2 Mean | USE HELPER: reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop> | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop, INPUT_AND_OUTPUT>(...)` | YES |
| 3 Sub | USE HELPER: sub<COL, WaitUpfrontPopAtEnd, WaitAndPopPerTile, Bulk> | `compute_kernel_lib::sub<COL, WaitUpfrontPopAtEnd, WaitAndPopPerTile, Bulk, INPUT_AND_OUTPUT>(...)` | YES |
| 4 Square | USE HELPER: square<WaitUpfrontNoPop, Bulk> | `compute_kernel_lib::square<WaitUpfrontNoPop, Bulk, INPUT_AND_OUTPUT>(...)` | YES |
| 5 Var | USE HELPER: reduce<SUM, REDUCE_ROW, BulkWaitBulkPop> | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, BulkWaitBulkPop, INPUT_AND_OUTPUT>(...)` | YES |
| 6 Add+Rsqrt | NO HELPER: raw calls | Raw `add_tiles_bcast_scalar` + `rsqrt_tile` with manual DST/CB mgmt | YES |
| 7 Normalize | USE HELPER: mul<COL, WaitUpfrontPopAtEnd, WaitAndPopPerTile, Bulk> | `compute_kernel_lib::mul<COL, WaitUpfrontPopAtEnd, WaitAndPopPerTile, Bulk, INPUT_AND_OUTPUT>(...)` | YES |
| 8 Untilize | USE HELPER: untilize | `compute_kernel_lib::untilize<Wt, c_4, c_16>(1)` | YES |

#### Redundant CB Operation Check

| Helper Used | Wrapper CB Ops Present? | Status |
|-------------|-------------------------|--------|
| compute_kernel_lib::tilize() | NO | CLEAN |
| compute_kernel_lib::reduce<...>() | NO | CLEAN |
| compute_kernel_lib::sub<...>() | NO | CLEAN |
| compute_kernel_lib::square<...>() | NO | CLEAN |
| compute_kernel_lib::mul<...>() | NO | CLEAN |
| compute_kernel_lib::untilize<...>() | NO | CLEAN |

### Correctness Test Results

| Test Case | Input Shape | Dtype | PCC Threshold | Result |
|-----------|-------------|-------|---------------|--------|
| test_row_standardize | (32, 32) | bf16 | > 0.99 | PASS |
| test_row_standardize | (32, 64) | bf16 | > 0.99 | PASS |
| test_row_standardize | (64, 128) | bf16 | > 0.99 | PASS |
| test_row_standardize | (128, 128) | bf16 | > 0.99 | PASS |
| test_row_standardize | (32, 1024) | bf16 | > 0.99 | PASS |
| test_row_standardize | (128, 1024) | bf16 | > 0.99 | PASS |
| test_row_standardize | (1024, 32) | bf16 | > 0.99 | PASS |
| test_row_standardize | (1024, 1024) | bf16 | > 0.99 | PASS |
| test_row_standardize | (2, 32, 64) | bf16 | > 0.99 | PASS |
| test_row_standardize | (4, 64, 128) | bf16 | > 0.99 | PASS |
| test_row_standardize | (2, 4, 32, 64) | bf16 | > 0.99 | PASS |
| test_row_standardize | (32, 32) | f32 | > 0.999 | PASS |
| test_row_standardize | (32, 64) | f32 | > 0.999 | PASS |
| test_row_standardize | (64, 128) | f32 | > 0.999 | PASS |
| test_row_standardize | (128, 128) | f32 | > 0.999 | PASS |
| test_row_standardize | (32, 1024) | f32 | > 0.999 | PASS |
| test_row_standardize | (128, 1024) | f32 | > 0.999 | PASS |
| test_row_standardize | (1024, 32) | f32 | > 0.999 | PASS |
| test_row_standardize | (1024, 1024) | f32 | > 0.999 | PASS |
| test_row_standardize | (2, 32, 64) | f32 | > 0.999 | PASS |
| test_row_standardize | (4, 64, 128) | f32 | > 0.999 | PASS |
| test_row_standardize | (2, 4, 32, 64) | f32 | > 0.999 | PASS |

---

## 3. Recovery Summary

### Error Recovery Table

| # | Stage | Error Type | Root Cause | Recovery Action | Resolved? |
|---|-------|------------|------------|-----------------|-----------|
| 1 | Build | build_error | Spurious include of non-existent sfpu_split_work.h | Removed include | YES |
| 2 | Test | test_fail | `ttnn.DataFormat` does not exist; CBFormatDescriptor uses DataType | Changed to `ttnn.float32` / `ttnn.bfloat16` | YES |
| 3 | Test | wrong_output | Scaler CB must always be bf16 format; generate_reduce_scaler writes packed bf16 | Fixed cb_scaler to always bfloat16; fixed scaler packing | YES |
| 4 | Test | test_fail | bfloat8_b typecast requires TILE_LAYOUT not ROW_MAJOR | Fixed test to use TILE_LAYOUT for typecast step | YES |

### Attempts Per Stage

| Stage | Attempts | Final Result |
|-------|----------|--------------|
| Kernel implementation | 1 | PASS |
| First test run | 3 | PASS (after fixing build error, DataFormat, scaler format) |
| Full test suite | 2 | PASS (after fixing dtype validation test) |

### Unresolved Issues

All issues were resolved.

---

## 4. Deviations from Instructions

| What | Why | Impact |
|------|-----|--------|
| Fixed program descriptor (DataFormat and scaler format) | Program descriptor had bugs preventing correct execution | Required for any tests to pass; downstream benefit |
| Fixed dtype validation test | Test used ROW_MAJOR for bfloat8_b typecast which is invalid | Test now works correctly |
| Changed cb_scaler from `dtype` to always `bfloat16` | Design doc incorrectly specifies dtype for scaler CB; generate_reduce_scaler always writes packed bf16 | Critical for float32 correctness |

---

## 5. Artifacts

### Files Created

| Path | Purpose |
|------|---------|
| `agent_logs/kernel_writer_breadcrumbs.jsonl` | Execution breadcrumbs |
| `agent_logs/kernel_writer_execution_log.md` | This execution log |

### Files Modified

| Path | Changes |
|------|---------|
| `kernels/row_standardize_reader.cpp` | Full implementation: TensorAccessor reads, scaler/epsilon generation |
| `kernels/row_standardize_compute.cpp` | Full implementation: 8-phase pipeline with helpers and raw Phase 6 |
| `kernels/row_standardize_writer.cpp` | Full implementation: TensorAccessor writes per block |
| `row_standardize_program_descriptor.py` | Fixed DataFormat->DataType, cb_scaler always bf16, scaler always packed bf16 |
| `test_row_standardize.py` | Fixed dtype validation test for bfloat8_b |

---

## 6. Handoff Notes

### For Next Agent: N/A

N/A - This is the final stage. Operation is complete.

**Key Configuration**:
- cb_scaler (c_1) MUST always be bfloat16 format regardless of input dtype
- generate_reduce_scaler always expects packed bf16 format (bf16 << 16 | bf16)
- Phase 6 (add+rsqrt) is the only raw-call phase; all others use kernel lib helpers
- compute_kernel_hw_startup uses c_0 (input A), c_1 (input B/scaler), c_16 (output)

**Known Limitations**:
- Single-core only (design spec limitation)
- Requires H and W to be multiples of 32 (tile alignment)
- No sharded memory support

---

## 7. Instruction Improvement Recommendations

### Recommendation 1: Scaler CB format guidance
- **Observed**: Design doc said scaler CB uses `dtype` but it must always be bf16
- **Frequency**: Once, but would affect every float32 operation using reduce scalers
- **Current Instruction**: Design doc says "cb_scaler: Uses dtype format"
- **Suggested Change**: Add to kernel-writer instructions: "CRITICAL: Reduce scaler CBs are ALWAYS bfloat16 format. The generate_reduce_scaler helper writes packed bf16 regardless of input dtype. This matches the softmax reference implementation."
- **Rationale**: Prevents a common float32 correctness bug
- **Confidence**: HIGH

### Recommendation 2: Include validation for sfpu headers
- **Observed**: Included non-existent header sfpu_split_work.h
- **Frequency**: Once
- **Current Instruction**: No guidance on which sfpu headers exist
- **Suggested Change**: List available sfpu headers in the kernel-writer instructions or design doc
- **Rationale**: Saves one build cycle
- **Confidence**: MEDIUM

---

## 8. Raw Logs

<details>
<summary>Final Test Output (29/29 PASSED)</summary>

```
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-32x32]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-32x64]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-64x128]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-128x128]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-32x1024]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-128x1024]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-1024x32]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-1024x1024]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-2x32x64]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-4x64x128]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[bf16-2x4x32x64]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-32x32]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-32x64]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-64x128]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-128x128]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-32x1024]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-128x1024]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-1024x32]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-1024x1024]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-2x32x64]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-4x64x128]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize[f32-2x4x32x64]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize_minimal[bf16]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize_minimal[f32]
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize_validation_rank
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize_validation_layout
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize_validation_dtype
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize_validation_width_alignment
PASSED ttnn/ttnn/operations/row_standardize/test_row_standardize.py::test_row_standardize_constant_row
============================== 29 passed in 3.07s ==============================
```

</details>
