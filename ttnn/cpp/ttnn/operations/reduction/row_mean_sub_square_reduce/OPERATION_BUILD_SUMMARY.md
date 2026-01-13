# TTNN Operation Build Summary: row_mean_sub_square_reduce

**Operation**: `ttnn.row_mean_sub_square_reduce`
**Date**: 2026-01-13
**Mode**: Fully Automated (No User Confirmations)
**Breadcrumbs Logging**: ENABLED

---

## Executive Summary

Successfully built a complete TTNN operation that computes variance along the width dimension:
```
Var(x) = mean((x - mean(x))^2)
```

**Input**: Row-major interleaved tensor [N, C, H, W] (BFLOAT16)
**Output**: Row-major interleaved tensor [N, C, H, 32] (variance per row, width reduced to 1, padded to TILE_WIDTH=32)

**Final Status**: ✅ ALL 12 TESTS PASSING

---

## Phase Summary

| Phase | Agent | Status | Duration |
|-------|-------|--------|----------|
| 0. Discovery | Orchestrator | ✅ Complete | ~2 min |
| 1. Analysis | ttnn-operation-analyzer (x3) | ✅ Complete | ~15 min |
| 2. Planning | ttnn-operation-planner | ✅ Complete | ~5 min |
| 3. Scaffolding (Stages 1-3) | ttnn-operation-scaffolder | ✅ Complete | ~8 min |
| 4. Factory Building (Stages 4-6) | ttnn-factory-builder | ✅ Complete | ~10 min |
| 5. Kernel Design | ttnn-kernel-designer | ✅ Complete | ~5 min |
| 6. Kernel Implementation (Stage 7) | ttnn-kernel-writer | ✅ Complete | ~10 min |

**Total Time**: ~55 minutes

---

## Autonomous Decisions Made

### Design Decisions (Phase 2)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data Flow | Single-pass with intermediate CB storage | Avoids re-reading input from DRAM, saves memory bandwidth |
| Work Distribution | Parallelize over tile-rows (Ht) | Each tile row is independent for variance computation |
| CB Count | 7 CBs | c_0 input, c_1 tilized, c_2 scaler, c_3 mean, c_4 intermediate, c_5 variance, c_16 output |
| Scaler Strategy | Single scaler = 1/W | Same operation (divide by W) applies to both mean computations |
| Mean Broadcast | Column broadcast (not scalar) | Mean tile has 32 different means (one per row) |

### Implementation Decisions (Phases 3-6)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Registration Pattern | Modern device operation (launch<> API) | Current TTNN standard |
| Validation Location | validate_on_program_cache_miss() | Per spec requirements |
| CB Buffering | Single-buffered (capacity = block size) | Simpler, sufficient for this op |
| Kernel Helper Usage | 4 phases with helpers, 1 phase raw | Helpers available for tilize, reduce, untilize; not for sub+square |
| PERSISTENT Mode | Yes for Phase 2 reduce | Keeps tilized input for reuse in Phase 3 |

### Bug Fix Decisions (Phase 6)

| Issue | Original | Fixed To | Rationale |
|-------|----------|----------|-----------|
| Broadcast Type | sub_tiles_bcast_scalar | sub_tiles_bcast_cols | Scalar broadcasts only [0,0]; need column 0 broadcast for per-row means |
| TensorAccessor API | accessor.get_noc_addr() | get_noc_addr(id, accessor) | Correct function signature |

---

## Component Sources (Hybrid Mode)

| Component | Reference Operation | Analysis File |
|-----------|---------------------|---------------|
| Reader (stick reading) | tilize_multi_core_interleaved | tilize_multi_core_interleaved_analysis.md |
| Scaler generation | reduce_multi_core_w | reduce_multi_core_w_analysis.md |
| Tilize phase | tilize_multi_core_interleaved | tilize_multi_core_interleaved_analysis.md |
| Reduce (mean & variance) | reduce_multi_core_w | reduce_multi_core_w_analysis.md |
| Sub + Square | NEW (custom) | N/A |
| Untilize phase | untilize_multi_core | untilize_multi_core_analysis.md |
| Writer (stick writing) | untilize_multi_core | untilize_multi_core_analysis.md |

---

## Files Created

### Implementation Files (9)
```
ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/
├── row_mean_sub_square_reduce.hpp
├── row_mean_sub_square_reduce.cpp
├── row_mean_sub_square_reduce_nanobind.hpp
├── row_mean_sub_square_reduce_nanobind.cpp
├── device/
│   ├── row_mean_sub_square_reduce_device_operation.hpp
│   ├── row_mean_sub_square_reduce_device_operation.cpp
│   ├── row_mean_sub_square_reduce_device_operation_types.hpp
│   ├── row_mean_sub_square_reduce_program_factory.hpp
│   └── row_mean_sub_square_reduce_program_factory.cpp
```

### Kernel Files (3)
```
├── device/kernels/
│   ├── compute/row_mean_sub_square_reduce_compute.cpp
│   └── dataflow/
│       ├── reader_row_mean_sub_square_reduce.cpp
│       └── writer_row_mean_sub_square_reduce.cpp
```

### Test Files (4)
```
├── test_dev/
│   ├── test_stage1_api_exists.py
│   ├── test_stage2_validation.py
│   ├── test_stage3_registration.py
│   └── test_stage7_kernel_correctness.py
```

### Documentation & Logs (7)
```
├── row_mean_sub_square_reduce_spec.md
├── row_mean_sub_square_reduce_planner_execution_log.md
├── kernel_design.md
├── SCAFFOLDING_VERIFICATION.md
├── OPERATION_BUILD_SUMMARY.md (this file)
└── agent_logs/
    ├── ttnn-factory-builder_breadcrumbs.jsonl
    ├── ttnn-factory-builder_execution_log.md
    ├── ttnn-kernel-designer_execution_log.md
    ├── ttnn-kernel-writer_breadcrumbs.jsonl
    └── ttnn-kernel-writer_execution_log.md
```

---

## Pain Points Encountered

### Agent: ttnn-operation-analyzer

| Pain Point | Severity | Resolution |
|------------|----------|------------|
| Reduce W analysis incomplete | Medium | Agent analyzed available files; some internal reduce details were inferred from DeepWiki |

### Agent: ttnn-operation-scaffolder

| Pain Point | Severity | Resolution |
|------------|----------|------------|
| Python test environment issue (EnablePersistentKernelCache missing) | Low | Verified via C++ symbol check instead |
| launch_on_device vs launch API | Low | Auto-corrected during build verification |

### Agent: ttnn-factory-builder

| Pain Point | Severity | Resolution |
|------------|----------|------------|
| Stage 6 stub kernels cause hang | Expected | CB sync mismatch with stub kernels; real implementation fixed it |
| clang-tidy warnings (auto* vs auto) | Low | Fixed manually after agent completed |

### Agent: ttnn-kernel-designer

| Pain Point | Severity | Resolution |
|------------|----------|------------|
| Incorrect broadcast type recommendation | Medium | Design doc said sub_tiles_bcast_scalar; kernel-writer correctly identified need for sub_tiles_bcast_cols |

### Agent: ttnn-kernel-writer

| Pain Point | Severity | Resolution |
|------------|----------|------------|
| TensorAccessor API usage | Low | get_noc_addr(id, accessor) not accessor.get_noc_addr(id) |
| Design doc broadcast mismatch | Medium | Autonomously corrected from scalar to column broadcast |

---

## Git Commits

| Commit | Agent | Description |
|--------|-------|-------------|
| 0b5478c536 | ttnn-operation-scaffolder | stage 1-3: scaffold row_mean_sub_square_reduce |
| 413dea9260 | ttnn-factory-builder | stages 4-6: CB config, stub kernels (partial) |
| baf4edfe21 | ttnn-factory-builder | Add execution log |
| 8eb2ece727 | ttnn-factory-builder | auto-commit before handoff |
| 6328a46e89 | ttnn-kernel-designer | auto-commit before handoff |
| 83ceec7590 | ttnn-kernel-writer | auto-commit before handoff (API fixes) |

---

## Test Results

### Stage 7 Kernel Correctness Tests: 12/12 PASSED

| Test | Description | Status |
|------|-------------|--------|
| test_basic_correctness_single_tile | Single tile width | ✅ PASSED |
| test_basic_correctness_multi_tile_width | Multiple tiles in width | ✅ PASSED |
| test_basic_correctness_multi_tile_row | Multiple tile rows | ✅ PASSED |
| test_uniform_values_zero_variance | All same values → variance=0 | ✅ PASSED |
| test_known_variance | Precomputed variance check | ✅ PASSED |
| test_batched_input | N > 1 | ✅ PASSED |
| test_multi_channel_input | C > 1 | ✅ PASSED |
| test_full_batch_channel | N > 1, C > 1 | ✅ PASSED |
| test_large_width | W = 256 (8 tiles) | ✅ PASSED |
| test_large_height | H = 128 (4 tile rows) | ✅ PASSED |
| test_numerical_stability_large_values | Large input values | ✅ PASSED |
| test_numerical_stability_small_values | Small input values | ✅ PASSED |

---

## Usage Example

```python
import torch
import ttnn

# Open device
device = ttnn.open_device(device_id=0)

# Create input tensor
input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
input_tt = ttnn.from_torch(input_torch, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

# Compute variance along width
output_tt = ttnn.row_mean_sub_square_reduce(input_tt)

# Convert back to torch
output_torch = ttnn.to_torch(output_tt)

# Output shape: [1, 1, 32, 32] (width reduced to 1, padded to 32)
# Only column 0 contains valid variance values
variance = output_torch[..., :1]

ttnn.close_device(device)
```

---

## Lessons Learned

1. **Broadcast Types Matter**: When reducing to per-row values and then broadcasting, column broadcast (not scalar) is required.

2. **Helper Library Coverage**: The kernel helper library covers tilize, reduce, and untilize well, but compound operations (subtract + square) require raw calls.

3. **PERSISTENT Mode**: The reduce helper's PERSISTENT mode is valuable for operations that need to reuse input data after reduction.

4. **TensorAccessor API**: The function signature is `get_noc_addr(page_id, accessor)`, not `accessor.get_noc_addr(page_id)`.

5. **Stub Kernel Limitations**: Stub kernels that bypass the real data flow will cause CB deadlocks; this is expected behavior.

---

## Breadcrumb Log Locations

All agents with breadcrumb support produced logs:

```
ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/agent_logs/
├── ttnn-factory-builder_breadcrumbs.jsonl
├── ttnn-factory-builder_execution_log.md
├── ttnn-kernel-designer_execution_log.md
├── ttnn-kernel-writer_breadcrumbs.jsonl
└── ttnn-kernel-writer_execution_log.md
```

The planner's execution log is at:
```
ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/row_mean_sub_square_reduce_planner_execution_log.md
```
