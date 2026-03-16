# Operation Design: softmax

## Overview
- **Operation Name**: softmax
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operation(s)**: reduce_helpers_compute, binary_op_helpers, copy_tile_helpers, reduce_helpers_dataflow, reader/writer_unary_interleaved_start_id

## Mathematical Definition
```
softmax(x, dim)_i = exp(x_i - max(x, dim)) / sum(exp(x_j - max(x, dim)), dim)
```
When `numeric_stable=False`, the max subtraction is skipped: `softmax(x)_i = exp(x_i) / sum(exp(x_j))`.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | ttnn.Tensor | Yes | 4D, bf16, TILE_LAYOUT, H%32==0, W%32==0 | - | Input tensor |
| dim | int | No | -1, -2 | -1 | Reduction dimension |
| numeric_stable | bool | No | True/False | True | Subtract max before exp |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Dtype | bfloat16 | "Input must be bfloat16" |
| Layout | TILE_LAYOUT | "Input must be TILE_LAYOUT" |
| Rank | 4 | "Input must be 4D [N,C,H,W]" |
| H,W | divisible by 32 | "H and W must be divisible by 32" |
| dim | -1 or -2 only | "Only dim=-1 (W) and dim=-2 (H) supported" |

### Output Tensor Specification
- **Shape**: same as input [N, C, H, W]
- **Dtype**: bfloat16
- **Layout**: TILE_LAYOUT
- **Memory**: interleaved DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile (32x32) | R=1 tile, reduce is trivial (all 1.0 output) |
| numeric_stable=False | Skip max reduction and subtraction phases |
| dim=-2 | Reader/writer use strided access; compute unchanged |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | reader_unary_interleaved_start_id | input_stage | Add strided reads for dim=-2; read R tiles per work unit |
| Compute (max) | reduce_helpers_compute | compute_core | reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop> |
| Compute (sub+exp) | binary_op_helpers | compute_core | sub<COL, NoWaitNoPop, WaitAndPopPerTile> with exp post-op |
| Compute (sum+recip) | reduce_helpers_compute | compute_core | reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop> with recip post-op |
| Compute (mul) | binary_op_helpers | compute_core | mul<COL, NoWaitNoPop, WaitAndPopPerTile> |
| Compute (exp copy) | copy_tile_helpers | compute_core | copy_tiles with exp post-op (unstable mode) |
| Writer | writer_unary_interleaved_start_id | output_stage | Add strided writes for dim=-2 |

### Unified Compute Design

Both dim=-1 and dim=-2 use REDUCE_ROW in the compute kernel. The difference is only in dataflow:

| Property | dim=-1 (W) | dim=-2 (H) |
|----------|-----------|-----------|
| R (tiles per work unit) | Wt = W/32 | Ht = H/32 |
| Total work units | NC * Ht | NC * Wt |
| Reader access | contiguous tiles | strided (stride = Wt) |
| Writer access | contiguous tiles | strided (stride = Wt) |

### Work Distribution
- **Work unit**: One "virtual row" of R tiles (a row for dim=-1, a column for dim=-2)
- **Grid**: dynamic, via `split_work_to_cores`
- **Work per core**: `ceil(total_work_units / num_cores)` work units
- **Remainder**: last core handles fewer work units

### Data Flow
Reader loads R tiles per work unit into cb_input. Compute processes all R tiles through max/sub-exp/sum-recip/mul phases. Writer drains R output tiles per work unit from cb_out.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| 0 | cb_input | Input tiles | Reader | Compute | R | Per work unit (WaitUpfront by reduce, then NoWait by sub) |
| 8 | cb_scaler | Reduce scaler (1.0f) | Reader | Compute | 1 | Persistent (entire kernel) |
| 16 | cb_out | Final output | Compute (mul) | Writer | 2 | Double-buffered, PerTile |
| 24 | cb_max | Max reduction result | Compute (reduce MAX) | Compute (sub) | 1 | Per work unit |
| 25 | cb_exp | exp(x - max) intermediate | Compute (sub+exp) | Compute (sum, mul) | R | Per work unit |
| 26 | cb_recip_sum | 1/sum(exp) result | Compute (reduce SUM) | Compute (mul) | 1 | Per work unit |

### Kernel Arguments

**Compile-time** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | cb_input | uint32_t | Input CB ID (0) |
| Reader | 1 | cb_scaler | uint32_t | Scaler CB ID (8) |
| Reader | 2 | R | uint32_t | Tiles per work unit |
| Reader | 3 | is_dim_h | uint32_t | 1 if dim=-2, 0 if dim=-1 |
| Reader | 4 | Wt | uint32_t | Width in tiles |
| Reader | 5+ | TensorAccessorArgs | - | Input tensor accessor |
| Compute | 0 | cb_input | uint32_t | 0 |
| Compute | 1 | cb_scaler | uint32_t | 8 |
| Compute | 2 | cb_out | uint32_t | 16 |
| Compute | 3 | cb_max | uint32_t | 24 |
| Compute | 4 | cb_exp | uint32_t | 25 |
| Compute | 5 | cb_recip_sum | uint32_t | 26 |
| Compute | 6 | R | uint32_t | Tiles per work unit |
| Compute | 7 | numeric_stable | uint32_t | 1 or 0 |
| Writer | 0 | cb_out | uint32_t | Output CB ID (16) |
| Writer | 1 | is_dim_h | uint32_t | 1 if dim=-2, 0 if dim=-1 |
| Writer | 2 | Wt | uint32_t | Width in tiles |
| Writer | 3+ | TensorAccessorArgs | - | Output tensor accessor |

**Runtime** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input tensor base address |
| Reader | 1 | num_work_units | uint32_t | Work units for this core |
| Reader | 2 | start_work_unit | uint32_t | First work unit index |
| Writer | 0 | dst_addr | uint32_t | Output tensor base address |
| Writer | 1 | num_work_units | uint32_t | Work units for this core |
| Writer | 2 | start_work_unit | uint32_t | First work unit index |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB is bfloat16
- [x] DEST register holds max 8 tiles (bf16) / 4 tiles (f32) -- helpers manage DEST internally
- [x] All CBs count pages in tiles (TILE_LAYOUT throughout)
- [x] cb_input needs R pages for WaitUpfrontNoPop (max R limited by L1)

### Test Criteria
- Output shape matches input shape exactly
- Numerical accuracy vs `torch.softmax(input, dim=dim)`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile per work unit | `(1, 1, 32, 32)` |
| Multi-tile | Multiple tiles per row | `(1, 1, 64, 128)` |
| Non-square | W != H | `(1, 1, 32, 256)` |
| Multi-batch | Batch dimension | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| 0 (cb_input) | R | TILE | All | Per work unit; waited upfront by reduce MAX, then reused by sub |
| 8 (cb_scaler) | 1 | TILE (bf16) | Row0 | Persistent; prepare_reduce_scaler fills it once |
| 16 (cb_out) | 2 | TILE | All | Double-buffered streaming output |
| 24 (cb_max) | 1 | TILE | Col0 | REDUCE_ROW output; broadcast via COL |
| 25 (cb_exp) | R | TILE | All | exp(x-max) tiles; reused by sum and mul |
| 26 (cb_recip_sum) | 1 | TILE | Col0 | 1/sum; broadcast via COL |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| Sub+Exp | SUB | All (cb_input, R tiles) | Col0 (cb_max, REDUCE_ROW output) | COL |
| Mul | MUL | All (cb_exp, R tiles) | Col0 (cb_recip_sum, REDUCE_ROW output) | COL |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | passthrough | Reader+writer data pipeline, compute copies input to output | `input_tensor` | Same | - |
| 2 | exp_only | Compute applies exp to each tile | `torch.exp(input_tensor)` | Same | - |
| 3 | softmax_w_stable | Full softmax dim=-1, numeric_stable=True | `torch.softmax(input, dim=-1)` | Same | - |
| 4 | softmax_w_unstable | Softmax dim=-1, numeric_stable=False | `torch.softmax(input, dim=-1)` | Same | - |
| 5 | softmax_h_stable | Full softmax dim=-2, numeric_stable=True | `torch.softmax(input, dim=-2)` | Same | - |
| 6 | softmax_h_unstable | Softmax dim=-2, numeric_stable=False | `torch.softmax(input, dim=-2)` | Same | - |

### Stage 1: passthrough
- **Scope**: reader, writer, compute (copy_tiles only)
- **Reference**: `return input_tensor`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute just copies cb_input to cb_out via copy_tiles

### Stage 2: exp_only
- **Scope**: compute kernel adds exp post-op to copy_tiles
- **Reference**: `return torch.exp(input_tensor)`
- **Shapes**: same as stage 1
- **Tolerances**: rtol=0.01, atol=0.05
- **Delta from previous**: copy_tiles gets exp_tile post-op lambda

### Stage 3: softmax_w_stable
- **Scope**: Full compute pipeline for dim=-1 with numeric_stable=True
- **Reference**: `return torch.softmax(input_tensor, dim=-1)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Replace copy+exp with full 4-phase softmax compute

### Stage 4: softmax_w_unstable
- **Scope**: Unstable mode for dim=-1 (skip max subtraction)
- **Reference**: `return torch.softmax(input_tensor, dim=-1)`
- **Shapes**: same as stage 3
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Test numeric_stable=False path (copy+exp, then sum+recip, then mul)

### Stage 5: softmax_h_stable
- **Scope**: Reader/writer switch to strided access for dim=-2
- **Reference**: `return torch.softmax(input_tensor, dim=-2)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Reader/writer use strided page IDs, R=Ht

### Stage 6: softmax_h_unstable
- **Scope**: Unstable mode for dim=-2
- **Reference**: `return torch.softmax(input_tensor, dim=-2)`
- **Shapes**: same as stage 5
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Test numeric_stable=False with dim=-2

### Reader Kernel

Per work unit, computes page IDs based on dim:
- **dim=-1**: `page_id = start_work_unit * R + tile_idx` (contiguous)
- **dim=-2**: `page_id = (nc * Ht * Wt) + (ht * Wt) + col_wt` where `ht` iterates 0..Ht-1 (strided)

Reads R tiles into cb_input per work unit. On first work unit, also calls `prepare_reduce_scaler<cb_scaler>(1.0f)`.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_input, cb_scaler, cb_out)`

Loops over `num_work_units`, each iteration processes one work unit of R tiles.

#### Stable Mode (numeric_stable=True):

#### Phase 1: Find max across R tiles
```cpp
compute_kernel_lib::reduce<MAX, REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop,
    ReduceDataFormatReconfigMode::NONE>(
    cb_input, cb_scaler, cb_max,
    ReduceInputBlockShape::row(R));
```
- A: cb_input [R tiles, FRESHLY PUSHED by reader, WaitUpfrontNoPop -- tiles persist]
- Out: cb_max [1 tile, pushed by helper]

**CB state after Phase 1:**
| CB | Tiles | State |
|----|-------|-------|
| cb_input | R | waited, not popped -- persists for Phase 2 |
| cb_max | 1 | freshly pushed |

#### Phase 2: Subtract max and apply exp (fused)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_input, cb_max, cb_exp,
    BinaryInputBlockShape::of(1, R),
    exp_post_op);
cb_pop_front(cb_input, R);  // manual pop -- NoWaitNoPop on A
```
- A: cb_input [R tiles, ALREADY WAITED from Phase 1, NoWaitNoPop]
- B: cb_max [1 tile, WaitAndPopPerTile -- waited and popped per tile (1 tile total)]
- Out: cb_exp [R tiles, PerTile push]
- Post-op: `[](uint32_t dst_idx) { exp_tile_init(); exp_tile(dst_idx); }`

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| cb_input | 0 | freed (manually popped) |
| cb_max | 0 | freed (popped by WaitAndPopPerTile) |
| cb_exp | R | freshly pushed |

#### Phase 3: Sum exp and apply reciprocal (fused)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    cb_exp, cb_scaler, cb_recip_sum,
    ReduceInputBlockShape::row(R),
    ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    recip_post_op);
```
- A: cb_exp [R tiles, FRESHLY PUSHED by Phase 2, WaitUpfrontNoPop -- tiles persist for Phase 4]
- Out: cb_recip_sum [1 tile, pushed by helper]
- Post-op: `[](uint32_t dst_idx) { recip_tile_init(); recip_tile(dst_idx); }`

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| cb_exp | R | waited, not popped -- persists for Phase 4 |
| cb_recip_sum | 1 | freshly pushed |

#### Phase 4: Multiply exp by 1/sum
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_exp, cb_recip_sum, cb_out,
    BinaryInputBlockShape::of(1, R));
cb_pop_front(cb_exp, R);  // manual pop -- NoWaitNoPop on A
```
- A: cb_exp [R tiles, ALREADY WAITED from Phase 3, NoWaitNoPop]
- B: cb_recip_sum [1 tile, WaitAndPopPerTile]
- Out: cb_out [R tiles total, PerTile push -- writer drains concurrently]

**CB state after Phase 4:**
| CB | Tiles | State |
|----|-------|-------|
| cb_exp | 0 | freed (manually popped) |
| cb_recip_sum | 0 | freed (popped by WaitAndPopPerTile) |
| cb_out | up to R | being drained by writer |

#### Unstable Mode (numeric_stable=False):

#### Phase 1: Copy with exp
```cpp
compute_kernel_lib::copy_tiles<CopyInputPolicy::WaitAndPop,
    CopyDataFormatReconfig::NONE>(
    cb_input, cb_exp, R, exp_post_op);
```
- Then Phase 3 and Phase 4 same as stable mode.

### Writer Kernel

Per work unit, computes page IDs (same logic as reader) and writes R tiles from cb_out:
- **dim=-1**: contiguous page IDs
- **dim=-2**: strided page IDs

### Critical Notes
- **BinaryInputBlockShape::of(1, R)**: For COL broadcast with 1 tile in B, shape is `(1, R)` meaning 1 row of R tiles. The single B tile broadcasts across all R tiles.
- **WaitAndPopPerTile on B in sub/mul**: cb_max and cb_recip_sum each hold exactly 1 tile. WaitAndPopPerTile will wait for 1 tile, use it R times (broadcast), then pop it. This is correct because COL broadcast reuses the same B tile for all columns.
- **Manual pops required**: Both sub (Phase 2) and mul (Phase 4) use NoWaitNoPop for input A. The caller MUST call `cb_pop_front` after each.
- **Scaler is 1.0f**: Both MAX and SUM reduce with scaler=1.0f. `prepare_reduce_scaler<cb_scaler>(1.0f)` handles correct bf16 packing.
- **L1 constraint**: R tiles in cb_input + R tiles in cb_exp = 2R tiles of intermediate storage. For large R (e.g., Wt=256 tiles = 8192 width), this may exceed L1. The program factory should validate L1 capacity.

### Implementation Checklist
- [ ] Reader: TensorAccessor-based tile reading, contiguous or strided; prepare_reduce_scaler once
- [ ] Compute: 4 phases (stable) or 3 phases (unstable) using helpers: reduce<MAX>, sub<COL>+exp, reduce<SUM>+recip, mul<COL>
- [ ] Writer: TensorAccessor-based tile writing, contiguous or strided
- [ ] CB push/pop balance verified: reduce uses WaitUpfrontNoPop (manual pop after sub/mul); binary ops use NoWaitNoPop on A (manual pop)
