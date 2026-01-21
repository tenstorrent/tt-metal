# TTNN Operation Creation Report: reduce_mean_w_rm

## Executive Summary

The `reduce_mean_w_rm` operation was successfully created using the fully automated TTNN operation creation pipeline. The operation takes a row-major interleaved input tensor (at least 2D), computes the mean across the width dimension, and outputs a tensor with logical shape `[..., 1]` and padded shape `[..., 32]`.

**Final Status**: SUCCESS - All 6 correctness tests pass

**Total Agents Involved**: 8 (3 analyzers + 5 pipeline agents)

**Approach**: Hybrid mode combining patterns from tilize, reduce_w, and untilize reference operations

---

## Operation Overview

| Property | Value |
|----------|-------|
| **Name** | `reduce_mean_w_rm` |
| **Python API** | `ttnn.reduce_mean_w_rm(input_tensor, memory_config=None)` |
| **Category** | reduction |
| **Input Requirements** | Rank ≥ 2, ROW_MAJOR layout, INTERLEAVED memory, BFLOAT16/FLOAT32 dtype |
| **Output Shape** | Logical: `[..., 1]`, Padded: `[..., 32]` |
| **Implementation** | Single-core with tilize → reduce → untilize pipeline |

---

## Agent Pipeline Execution

### Phase 1: Reference Analysis (3 Parallel Analyzers)

#### 1.1 Tilize Analyzer (`ttnn-operation-analyzer`)

| Field | Value |
|-------|-------|
| **Reference** | `tilize_multi_core_interleaved_program_factory.cpp` |
| **Status** | SUCCESS |
| **Analysis Output** | `references/tilize_analysis.md` |
| **Execution Log** | `agent_logs/analyzer_tilize_execution_log.md` |

**Key Findings**:
- Block-based processing: 32 rows (sticks) per tile height
- CB c_0 for input (row-major sticks), CB c_16 for output (tiles)
- Uses `tilize_helpers.hpp` unified helper library
- Single-buffered CB configuration (capacity = block size)
- Reader pre-computes 32 NoC addresses before reading

**Decisions**:
- Work unit: Block (tile rows) - reader processes 32 rows at a time
- CB buffering: Single-buffered
- Core distribution: 1D using `split_blocks_for_tilize()`

---

#### 1.2 Reduce W Analyzer (`ttnn-operation-analyzer`)

| Field | Value |
|-------|-------|
| **Reference** | `reduce_op_multi_core_w_program_factory.cpp` |
| **Status** | SUCCESS |
| **Analysis Output** | `references/reduce_w_analysis.md` |
| **Execution Log** | `agent_logs/analyzer_reduce_w_execution_log.md` |

**Key Findings**:
- Width reduction uses SUM with 1/W scaler for MEAN
- CB c_0 (input), c_2 (scaler, persistent), c_3 (output)
- `REDUCE_ROW_SUM_VIA_MM` optimization for W-dimension SUM
- Uses `reduce_helpers.hpp` unified helper library
- Double-buffered CBs (2 tiles capacity)

**Decisions**:
- Mean computation: SUM with 1/W scaler (hardware accelerated)
- Scaler format: Packed bfloat16 in tile format
- Core distribution: `split_work_to_cores()` with core_group_1/core_group_2 split

**Pain Points Identified**:
- Scaler hardcoded to bfloat16 (potential precision issues with float32)
- No batched reading (single tile reads with barriers)

---

#### 1.3 Untilize Analyzer (`ttnn-operation-analyzer`)

| Field | Value |
|-------|-------|
| **Reference** | `untilize_multi_core_program_factory.cpp` |
| **Status** | SUCCESS |
| **Analysis Output** | `references/untilize_analysis.md` |
| **Execution Log** | `agent_logs/analyzer_untilize_execution_log.md` |

**Key Findings**:
- Two compute kernel variants: pack_untilize (fast) and standard untilize (fallback)
- CB c_0 (input tiles), CB c_16 (output row-major)
- Uses `untilize_helpers.hpp` unified helper library
- Auto-dispatches based on width and datatype

**Decisions**:
- Kernel selection: Automatic based on DEST limit and dtype
- Sharded input optimization: Reader kernel is trivial (just `cb_push_back`)

---

### Phase 2: Operation Planning (`ttnn-operation-planner`)

| Field | Value |
|-------|-------|
| **Mode** | Hybrid (3 references) |
| **Status** | SUCCESS |
| **Spec Output** | `reduce_mean_w_rm_spec.md` |
| **Execution Log** | `agent_logs/planner_execution_log.md` |
| **Breadcrumbs** | `agent_logs/ttnn-operation-planner_breadcrumbs.jsonl` |

**Key Design Decisions**:

| Decision | Options | Choice | Rationale |
|----------|---------|--------|-----------|
| Compute kernel structure | (1) Three separate kernels, (2) Single unified | Single unified with 3 phases | Avoids launch overhead, simplifies CB coordination |
| CB configuration | (1) Reuse CBs, (2) Dedicated per phase | 5 dedicated CBs | Clear data flow; phases run sequentially |
| Core distribution | (1) Single-core, (2) Multi-core | Single-core | User requirement for simplicity |
| Mean computation | (1) SUM then divide, (2) SUM with 1/W scaler | SUM with 1/W scaler | Standard approach, hardware accelerated |

**Component Mapping**:

| Component | Source Reference | Modifications |
|-----------|-----------------|---------------|
| Reader | tilize_analysis.md | Add scaler generation |
| Compute (tilize) | tilize_analysis.md | Output to CB_1 instead of CB_16 |
| Compute (reduce) | reduce_w_analysis.md | MEAN scaler (1/W) |
| Compute (untilize) | untilize_analysis.md | Input from CB_3, output width = 1 tile |
| Writer | untilize_analysis.md | Reduced output width (32 elements) |

**Deviations**: None

---

### Phase 3: Scaffolding (`ttnn-operation-scaffolder`)

| Field | Value |
|-------|-------|
| **Stages** | 1-3 (API existence, validation, registration) |
| **Status** | SUCCESS |
| **Files Created** | 12 (9 implementation + 3 tests) |
| **Execution Log** | `agent_logs/scaffolder_execution_log.md` |

**Files Generated**:
- API wrapper: `reduce_mean_w_rm.hpp`, `reduce_mean_w_rm.cpp`
- Python bindings: `reduce_mean_w_rm_nanobind.hpp`, `reduce_mean_w_rm_nanobind.cpp`
- Device operation: `device/reduce_mean_w_rm_device_operation.hpp`, `device/reduce_mean_w_rm_device_operation.cpp`
- Types: `device/reduce_mean_w_rm_device_operation_types.hpp`
- Program factory stubs: `device/reduce_mean_w_rm_program_factory.hpp`, `device/reduce_mean_w_rm_program_factory.cpp`

**Code Fixes Applied**:

| Fix | Issue | Resolution |
|-----|-------|------------|
| Launch API | Used `detail::launch<>()` | Changed to `launch<>()` |
| Shape constructor | Invalid two-argument constructor | Use separate Shape objects for TensorLayout |
| Unused parameter warnings | Compiler warnings | Added `(void)parameter;` casts |
| Return type | Missing class qualifier | Added `ReduceMeanWRmDeviceOperation::` prefix |

**Tests Passed**: 7/7 (Stage 1: 2, Stage 2: 3, Stage 3: 2)

---

### Phase 4: Factory Building (`ttnn-factory-builder`)

| Field | Value |
|-------|-------|
| **Stages** | 4-6 (device operation, program factory, stub kernels) |
| **Status** | SUCCESS |
| **Execution Log** | `agent_logs/factory_builder_execution_log.md` |
| **Breadcrumbs** | `agent_logs/ttnn-factory-builder_breadcrumbs.jsonl` |

**CB Configuration**:

| CB ID | Index | Page Size | Num Pages | Purpose |
|-------|-------|-----------|-----------|---------|
| cb_in_rm | c_0 | input_stick_size_aligned | 64 | Input RM sticks (double-buffered) |
| cb_in_tiled | c_1 | tile_size | 2*Wt | Tiled input (double-buffered) |
| cb_scaler | c_2 | tile_size | 1 | Scaler tile (1/W, persistent) |
| cb_reduced_tiled | c_3 | tile_size | 1 | Reduced tiled output |
| cb_out_rm | c_16 | output_stick_size_aligned | 1 | Output RM sticks |

**Kernel Stubs Created**:
- `device/kernels/dataflow/reader_reduce_mean_w_rm.cpp`
- `device/kernels/compute/reduce_mean_w_rm_compute.cpp`
- `device/kernels/dataflow/writer_reduce_mean_w_rm.cpp`

**Decisions**:
- Single-core implementation on core (0,0)
- All Ht tile-rows processed by single core

**Instruction Recommendation**:
> "Empty stub clarification: The instructions say 'empty stub kernels' but these cause device hangs. Recommend clarifying that minimal CB sync is required."

---

### Phase 5: Kernel Design (`ttnn-kernel-designer`)

| Field | Value |
|-------|-------|
| **Status** | SUCCESS |
| **Design Output** | `kernel_design.md` |
| **Execution Log** | `agent_logs/kernel_designer_execution_log.md` |
| **Breadcrumbs** | `agent_logs/ttnn-kernel-designer_breadcrumbs.jsonl` |

**Helper Function Mapping**:

| Phase | Helper | Rationale |
|-------|--------|-----------|
| Tilize | `compute_kernel_lib::tilize()` | Handles CB ops, tilize_block, init/uninit internally |
| Reduce | `compute_kernel_lib::reduce<SUM, REDUCE_ROW>()` | Handles DST management, reduce_tile, CB ops |
| Untilize | `compute_kernel_lib::untilize<1, c_3, c_16>()` | Auto-dispatches pack_untilize or standard |

**Raw Call Phases**:

| Kernel | Phase | Reason |
|--------|-------|--------|
| Reader | Scaler generation | Dataflow kernels don't use compute helpers |
| Reader | Stick reads | Use `noc_async_read` with TensorAccessor |
| Writer | Stick writes | Use `noc_async_write` with TensorAccessor |

**Key Decisions**:
- Mean via SUM + Scaler (not AVG pool type)
- Valid regions documented: After REDUCE_ROW, only Column 0 of each tile is valid

---

### Phase 6: Kernel Implementation (`ttnn-kernel-writer`)

| Field | Value |
|-------|-------|
| **Status** | SUCCESS |
| **Execution Log** | `agent_logs/kernel_writer_execution_log.md` |
| **Breadcrumbs** | `agent_logs/ttnn-kernel-writer_breadcrumbs.jsonl` |

**Implementation Summary**:

| Kernel | Implementation |
|--------|----------------|
| Reader | `generate_reduce_scaler()` + TensorAccessor input reads |
| Compute | `tilize()` → `reduce<SUM, REDUCE_ROW>()` → `untilize<1>()` |
| Writer | TensorAccessor output writes with 32 sticks per tile |

**CB Configuration Fix Required**:

| CB | Original Page Size | Fixed Page Size | Reason |
|----|-------------------|-----------------|--------|
| c_0 | stick_size | tile_size | Tilize helper expects tile-sized pages for `cb_wait_front(cb, Wt)` |
| c_16 | stick_size | tile_size | Untilize helper pushes tiles, not sticks |

**Test Results**: 6/6 PASSED

| Test Case | Input Shape | Result |
|-----------|-------------|--------|
| test_basic_correctness_32x64 | [1,1,32,64] | PASS |
| test_multi_tile_height_64x64 | [1,1,64,64] | PASS |
| test_larger_width_32x128 | [1,1,32,128] | PASS |
| test_square_64x64 | [1,1,64,64] | PASS |
| test_uniform_values | [1,1,32,64] | PASS |
| test_zeros | [1,1,32,64] | PASS |

**Instruction Recommendation**:
> "Add explicit note in kernel_design.md template: 'CB page_size MUST be tile_size for tilize/untilize CBs to match helper's wait/pop counts'"

---

## Summary of All Decisions

### High-Impact Design Decisions

| Decision | Agent | Rationale |
|----------|-------|-----------|
| Single unified compute kernel | Planner | Avoids kernel launch overhead, simplifies CB coordination |
| 5 dedicated CBs | Planner | Clear data flow between phases |
| Mean via SUM + 1/W scaler | Planner | Hardware accelerated, standard pattern |
| Use kernel helper libraries | Designer | Reduces implementation complexity |

### Implementation Fixes

| Fix | Agent | Reason |
|-----|-------|--------|
| Launch API namespace | Scaffolder | Verification script requires modern pattern |
| Shape constructor | Scaffolder | Invalid two-argument constructor |
| CB page_size (c_0, c_16) | Kernel Writer | Tilize/untilize helpers expect tile-sized pages |

### Pain Points Identified

| Pain Point | Agent | Recommendation |
|------------|-------|----------------|
| Empty stub kernels cause device hangs | Factory Builder | Clarify in instructions that minimal CB sync is required |
| CB page_size mismatch | Kernel Writer | Add explicit note about tile_size requirement |
| Scaler hardcoded to bfloat16 | Analyzer (reduce_w) | May cause precision issues with float32 |

---

## Files and Artifacts

### Agent Logs

| Agent | Execution Log | Breadcrumbs |
|-------|---------------|-------------|
| Analyzer (tilize) | `agent_logs/analyzer_tilize_execution_log.md` | N/A |
| Analyzer (reduce_w) | `agent_logs/analyzer_reduce_w_execution_log.md` | N/A |
| Analyzer (untilize) | `agent_logs/analyzer_untilize_execution_log.md` | N/A |
| Planner | `agent_logs/planner_execution_log.md` | `agent_logs/ttnn-operation-planner_breadcrumbs.jsonl` |
| Scaffolder | `agent_logs/scaffolder_execution_log.md` | N/A |
| Factory Builder | `agent_logs/factory_builder_execution_log.md` | `agent_logs/ttnn-factory-builder_breadcrumbs.jsonl` |
| Kernel Designer | `agent_logs/kernel_designer_execution_log.md` | `agent_logs/ttnn-kernel-designer_breadcrumbs.jsonl` |
| Kernel Writer | `agent_logs/kernel_writer_execution_log.md` | `agent_logs/ttnn-kernel-writer_breadcrumbs.jsonl` |

### Reference Analyses

| Reference | Path |
|-----------|------|
| Tilize | `references/tilize_analysis.md` |
| Reduce W | `references/reduce_w_analysis.md` |
| Untilize | `references/untilize_analysis.md` |

### Specification and Design

| Document | Path |
|----------|------|
| Functional Specification | `reduce_mean_w_rm_spec.md` |
| Kernel Design Document | `kernel_design.md` |

### Implementation Files

| Category | Files |
|----------|-------|
| API Wrapper | `reduce_mean_w_rm.hpp`, `reduce_mean_w_rm.cpp` |
| Python Bindings | `reduce_mean_w_rm_nanobind.hpp`, `reduce_mean_w_rm_nanobind.cpp` |
| Device Operation | `device/reduce_mean_w_rm_device_operation.hpp`, `device/reduce_mean_w_rm_device_operation.cpp` |
| Types | `device/reduce_mean_w_rm_device_operation_types.hpp` |
| Program Factory | `device/reduce_mean_w_rm_program_factory.hpp`, `device/reduce_mean_w_rm_program_factory.cpp` |
| Reader Kernel | `device/kernels/dataflow/reader_reduce_mean_w_rm.cpp` |
| Compute Kernel | `device/kernels/compute/reduce_mean_w_rm_compute.cpp` |
| Writer Kernel | `device/kernels/dataflow/writer_reduce_mean_w_rm.cpp` |

### Test Files

| Stage | Path |
|-------|------|
| Stage 1 | `test_dev/test_stage1_api_exists.py` |
| Stage 2 | `test_dev/test_stage2_validation.py` |
| Stage 3 | `test_dev/test_stage3_registration.py` |
| Stage 4 | `test_dev/test_stage4_device_op.py` |
| Stage 5 | `test_dev/test_stage5_program_factory.py` |
| Stage 6 | `test_dev/test_stage6_kernel_compilation.py` |
| Stage 7 | `test_dev/test_stage7_kernel_correctness.py` |

---

## Git Commits

| SHA | Message | Agent |
|-----|---------|-------|
| c04c5d1a55 | [ttnn-operation-planner] spec: reduce_mean_w_rm | Planner |
| 466861236e | [ttnn-operation-scaffolder] stage 1-3: scaffold reduce_mean_w_rm | Scaffolder |
| d47350add9 | [ttnn-factory-builder] stage 4: device operation validation | Factory Builder |
| d866062035 | [ttnn-factory-builder] stage 5: CB configuration and work distribution | Factory Builder |
| 8ce1847659 | [ttnn-factory-builder] stage 6: kernel infrastructure | Factory Builder |
| 147408ed05 | [ttnn-kernel-designer] design: reduce_mean_w_rm | Kernel Designer |
| 06e5b233d3 | [ttnn-kernel-writer] stage 7: implement reduce_mean_w_rm kernels | Kernel Writer |

---

## Conclusion

The `reduce_mean_w_rm` operation was successfully created using the fully automated TTNN operation creation pipeline. The operation combines patterns from three reference operations (tilize, reduce_w, untilize) in a Hybrid mode approach, resulting in a single-core implementation that correctly computes the mean across the width dimension.

**Known Limitations**:
- Single-core only (multi-core would be future enhancement)
- Requires tile-aligned dimensions (H and W multiples of 32)
- bfloat16/float32 data types only

**Usage**:
```python
import ttnn

# Input: row-major tensor with shape [..., W]
output = ttnn.reduce_mean_w_rm(input_tensor)
# Output: row-major tensor with shape [..., 1] (logical), [..., 32] (padded)
```
