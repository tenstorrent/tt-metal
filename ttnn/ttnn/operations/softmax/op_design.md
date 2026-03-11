# Operation Design: softmax

## Overview
- **Operation Name**: softmax
- **Category**: eltwise (multi-phase: reduction + elementwise)
- **Planning Mode**: Hybrid
- **Reference Operations**: reduce_w (REDUCE_ROW for dim=-1), reduce_h (REDUCE_COL for dim=-2)

## Mathematical Definition
```
softmax(x, dim)[i] = exp(x[i] - max(x, dim)) / sum(exp(x[j] - max(x, dim)), dim)
```
When `numeric_stable=False`, the max subtraction is skipped: `exp(x[i]) / sum(exp(x[j]), dim)`.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | 4D, bf16, TILE | - | Input tensor |
| dim | int | No | {-1, -2} | -1 | Reduction dimension |
| numeric_stable | bool | No | {True, False} | True | Subtract max before exp |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| dtype | bfloat16 | "softmax requires bfloat16 input" |
| layout | TILE_LAYOUT | "softmax requires TILE_LAYOUT" |
| rank | 4 | "softmax requires 4D input [N,C,H,W]" |
| H, W | divisible by 32 | "H and W must be tile-aligned (divisible by 32)" |
| dim | -1 or -2 | "softmax only supports dim=-1 or dim=-2" |

### Output Tensor Specification
- **Shape**: same as input `[N, C, H, W]`
- **Dtype**: bfloat16
- **Layout**: TILE_LAYOUT
- **Memory**: DRAM interleaved

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W=32 (single tile width) | dim=-1: max/sum reduce single tile, output = all 1.0 / tile_sum |
| H=32 (single tile height) | dim=-2: max/sum reduce single tile |
| numeric_stable=False | Skip max reduction phase; exp applied directly |
| All elements equal | Output = 1/N along reduction dim (uniform distribution) |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader (dim=-1) | reduce_w | input_stage | Sequential tile reads, Wt tiles per row |
| Reader (dim=-2) | reduce_h | input_stage | Chunked column reads, Ht*chunk tiles per chunk |
| Compute (dim=-1) | New (uses helpers) | compute_core | 4-phase: max, sub+exp, sum+recip, mul |
| Compute (dim=-2) | New (uses helpers) | compute_core | 4-phase: max, sub+exp, sum+recip, mul (columnar) |
| Writer | reduce_w | output_stage | Same-shape output, sequential tile writes |

### Work Distribution
- **Work unit**: tile-row (dim=-1) or tile-column-chunk (dim=-2)
- **Grid**: 1D, up to `min(total_work_units, max_cores)`
- **Work per core**: `ceil(total_units / num_cores)` or `floor(total_units / num_cores)` (two-group split)
- **Remainder**: core_group_1 gets +1 unit if `total_units % num_cores != 0`

For dim=-1: total_units = NC * Ht (one unit per tile-row of Wt tiles)
For dim=-2: total_units = NC * Wt (one unit per tile-column of Ht tiles)

### Data Flow

**dim=-1 (row-buffered, single DRAM pass per row):**
Reader loads Wt tiles for one row into c_0. Compute processes all 4 phases in-place, producing Wt output tiles. Writer drains c_out. Repeat for each row.

**dim=-2 (chunk-buffered, single DRAM pass per chunk):**
Reader loads Ht * chunk_size tiles for one chunk into c_0 (chunked column order). Compute processes all 4 phases, producing Ht * chunk_size output tiles. Writer drains c_out. Repeat for each chunk.

### Circular Buffer Requirements

**dim=-1 (per tile-row of Wt tiles):**

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input | Input row tiles | Reader | Compute (P1,P2) | Wt | Row (freed in phase 2) |
| c_2 | cb_scaler | Reduce scaler (1.0) | Reader | Compute | 1 | Program (persistent) |
| c_16 | cb_out | Final output tiles | Compute (P4) | Writer | 2 | Per-tile (double-buffered) |
| c_24 | cb_max | Max per row (stable only) | Compute (P1) | Compute (P2) | 1 | Row (freed in phase 2) |
| c_25 | cb_exp | exp(x - max) values | Compute (P2) | Compute (P3,P4) | Wt | Row (freed in phase 4) |
| c_26 | cb_recip | 1/sum(exp) per row | Compute (P3) | Compute (P4) | 1 | Row (freed in phase 4) |

**dim=-2 (per chunk of chunk_size columns, Ht rows):**

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input | Input chunk tiles | Reader | Compute (P1,P2) | Ht * chunk_size | Chunk (freed in phase 2) |
| c_2 | cb_scaler | Reduce scaler (1.0) | Reader | Compute | 1 | Program (persistent) |
| c_16 | cb_out | Final output tiles | Compute (P4) | Writer | 2 | Per-tile (double-buffered) |
| c_24 | cb_max | Max per column | Compute (P1) | Compute (P2) | chunk_size | Chunk (freed in phase 2) |
| c_25 | cb_exp | exp(x - max) values | Compute (P2) | Compute (P3,P4) | Ht * chunk_size | Chunk (freed in phase 4) |
| c_26 | cb_recip | 1/sum(exp) per column | Compute (P3) | Compute (P4) | chunk_size | Chunk (freed in phase 4) |

**chunk_size** = min(DEST_AUTO_LIMIT, Wt). Program factory computes based on available L1 and Ht.

### Kernel Arguments

**Compile-time** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | scaler_bits | uint32_t | Bit-cast 1.0f for reduce scaler |
| Reader | 1+ | TensorAccessorArgs | uint32_t[] | Input tensor accessor |
| Compute | 0 | Ht | uint32_t | Tile-rows (dim=-1: 1, dim=-2: actual Ht) |
| Compute | 1 | Wt | uint32_t | Tile-cols (dim=-1: actual Wt, dim=-2: chunk_size for this core) |
| Compute | 2 | NC | uint32_t | Always 1 (batch folded into work distribution) |
| Compute | 3 | numeric_stable | uint32_t | 1 if stable mode, 0 if unstable |
| Writer | 0 | output_cb_index | uint32_t | CB index for output (c_16) |
| Writer | 1+ | TensorAccessorArgs | uint32_t[] | Output tensor accessor |

**Runtime** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input buffer DRAM address |
| Reader | 1 | num_tiles | uint32_t | Total input tiles for this core |
| Reader | 2 | start_id | uint32_t | First tile index |
| Writer | 0 | dst_addr | uint32_t | Output buffer DRAM address |
| Writer | 1 | num_pages | uint32_t | Total output tiles for this core |
| Writer | 2 | start_id | uint32_t | First output tile index |

dim=-2 reader additionally needs:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 3 | col_start_tile_id | uint32_t | Starting tile in flat array |
| 4 | curr_col_in_batch | uint32_t | Starting column within batch |
| 5 | num_cols | uint32_t | Columns assigned to this core |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (verified per phase)
- [x] Reduce scaler CB is bfloat16 (Float16_b format)
- [x] DEST register holds max 8 tiles (bf16 SyncHalf) / 4 tiles (f32)
- [x] All CBs use tile-sized pages (TILE_LAYOUT operation)
- [x] dim=-2 chunk_size bounded by DEST_AUTO_LIMIT

### Test Criteria
- Output shape matches input shape exactly
- Numerical accuracy vs `torch.nn.functional.softmax(input, dim=dim)`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile per dim | `(1, 1, 32, 32)` |
| Multi-tile W | Tile iteration along W | `(1, 1, 32, 128)` |
| Multi-tile H | Tile iteration along H | `(1, 1, 128, 32)` |
| Non-square | W!=H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

**dim=-1:**

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 | Wt | TILE | All | Per-row |
| c_2 | 1 | TILE | Row0 (scaler) | Program |
| c_16 | 2 | TILE | All | Per-tile |
| c_24 | 1 | TILE | Col0 (REDUCE_ROW output) | Per-row |
| c_25 | Wt | TILE | All | Per-row |
| c_26 | 1 | TILE | Col0 (REDUCE_ROW output) | Per-row |

**dim=-2:**

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 | Ht * chunk_size | TILE | All | Per-chunk |
| c_2 | 1 | TILE | Row0 (scaler) | Program |
| c_16 | 2 | TILE | All | Per-tile |
| c_24 | chunk_size | TILE | Row0 (REDUCE_COL output) | Per-chunk |
| c_25 | Ht * chunk_size | TILE | All | Per-chunk |
| c_26 | chunk_size | TILE | Row0 (REDUCE_COL output) | Per-chunk |

### Binary Op Broadcast Verification

**dim=-1 (per-row, Ht=1, Wt=Wt):**

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| P2 | SUB | All (input row) | Col0 (REDUCE_ROW max) | SCALAR (Ht=1) |
| P4 | MUL | All (exp row) | Col0 (REDUCE_ROW recip) | SCALAR (Ht=1) |

Since we process one row at a time (Ht=1), REDUCE_ROW output is a single tile. Broadcast is SCALAR (1x1 tile applied to 1xWt).

**dim=-2 (per-chunk, Ht=Ht, Wt=chunk_size):**

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| P2 | SUB | All (input chunk) | Row0 (REDUCE_COL max) | ROW (1xchunk broadcast to HtxChunk) |
| P4 | MUL | All (exp chunk) | Row0 (REDUCE_COL recip) | ROW (1xchunk broadcast to HtxChunk) |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline_w | Reader+Writer for dim=-1 (identity passthrough) | input (passthrough) | same | - |
| 2 | exp_w | Compute exp on each tile (dim=-1 unstable partial) | exp(input) | same | - |
| 3 | softmax_unstable_w | Full unstable softmax dim=-1 (exp+sum+recip+mul) | softmax(input, dim=-1) | same | - |
| 4 | softmax_stable_w | Full stable softmax dim=-1 (max+sub+exp+sum+recip+mul) | softmax(input, dim=-1) | same | - |
| 5 | softmax_stable_h | Full stable softmax dim=-2 | softmax(input, dim=-2) | same | - |

### Stage 1: data_pipeline_w
- **Scope**: reader_w.cpp (reads tiles from DRAM to c_0), compute_w.cpp (copy c_0 to c_16 via pack), writer.cpp (drains c_16 to DRAM)
- **Reference**: `input` (identity passthrough)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute copies tiles from c_0 to c_16 using tilize/untilize or copy_tile. All softmax phases skipped.

### Stage 2: exp_w
- **Scope**: compute_w.cpp adds exp_tile to each tile (no reduce, no broadcast)
- **Reference**: `torch.exp(input)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.05
- **Delta from previous**: Compute now applies exp_tile_init + exp_tile to each tile in DST before packing to output.

### Stage 3: softmax_unstable_w
- **Scope**: compute_w.cpp implements exp + reduce_sum(REDUCE_ROW) + recip + mul per row
- **Reference**: `torch.nn.functional.softmax(input, dim=-1)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Adds phases 3 (reduce_sum with recip post-op) and 4 (mul broadcast). Changes phase 2 to write exp to c_exp instead of c_out.
- **extra_args**: `, numeric_stable=False`

### Stage 4: softmax_stable_w
- **Scope**: compute_w.cpp adds phase 1 (reduce_max) and phase 2 (sub max with exp post-op)
- **Reference**: `torch.nn.functional.softmax(input, dim=-1)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Adds max reduction phase. Enables numeric_stable=True (default).

### Stage 5: softmax_stable_h
- **Scope**: New compute_h.cpp kernel, new reader_h.cpp (chunked column reads). Program factory dispatches based on dim.
- **Reference**: `torch.nn.functional.softmax(input, dim=-2)`
- **Shapes**: `(1,1,32,32)`, `(1,1,128,64)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Entirely new kernel path for dim=-2. Reader uses chunked column order. Compute uses REDUCE_COL and ROW broadcast.
- **extra_args**: `, dim=-2`

### Reader Kernel (dim=-1: reader_w.cpp)
Sequential tile reads. For each core's assigned rows, reads Wt tiles per row to c_0. Prepares reduce scaler in c_2 using `dataflow_kernel_lib::prepare_reduce_scaler<c_2>(1.0f)`. Uses TensorAccessor for DRAM reads.

### Reader Kernel (dim=-2: reader_h.cpp)
Chunked column reads. For each chunk of columns, reads Ht * chunk_size tiles in column-interleaved order (all columns in chunk for row 0, then row 1, etc.). Uses the same stride formula as reduce_h: advancing by Wt moves to same column in next row. Prepares reduce scaler in c_2.

### Compute Kernel (dim=-1: compute_w.cpp)

**Startup**: `compute_kernel_hw_startup(c_0, c_2, c_16)`

Processing is per-row (loop NC*Ht rows, each Wt wide). For each row:

#### Phase 1: Max Reduction (stable mode only)
```cpp
cb_wait_front(c_0, Wt);  // explicit wait for NoWaitNoPop
compute_kernel_lib::reduce<MAX, REDUCE_ROW,
    ReduceInputPolicy::NoWaitNoPop,
    ReduceDataFormatReconfigMode::NONE>(
    c_0, c_2, c_24, ReduceInputBlockShape::of(1, Wt, 1));
```
- A: c_0 [Wt tiles, EXPLICITLY WAITED, no pop]
- Out: c_24 [1 tile, pushed by helper]

Note: reduce with NoWaitNoPop reserves output upfront (1 tile) and pushes at end.

**CB state after Phase 1:**
| CB | Tiles | State |
|----|-------|-------|
| c_0 | Wt | waited, not popped -- reused in Phase 2 |
| c_24 | 1 | freshly pushed (max tile) |

#### Phase 2: Subtract Max + Exp (stable) or Exp only (unstable)

Stable mode:
```cpp
compute_kernel_lib::sub<BroadcastDim::SCALAR,
    BinaryInputPolicy::NoWaitPopAtEnd,    // A: c_0, already waited, pop Wt at end
    BinaryInputPolicy::WaitAndPopPerTile, // B: c_24, wait+pop 1 tile (SCALAR)
    BinaryOutputPolicy::Bulk,             // reserve Wt upfront, push Wt at end
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_0, c_24, c_25, BinaryInputBlockShape::of(1, Wt),
    [](uint32_t dst_idx) {
        exp_tile_init();
        exp_tile(dst_idx);
    });
```

Unstable mode: same but skip sub, apply exp directly to c_0 tiles, write to c_25. Implemented as a copy+exp from c_0 to c_25.

- A: c_0 [Wt tiles, ALREADY WAITED from Phase 1, popped at end]
- B: c_24 [1 tile, waited+popped by helper (SCALAR broadcast)]
- Out: c_25 [Wt tiles, bulk reserve/push]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_0 | 0 | freed (popped at end by sub helper) |
| c_24 | 0 | freed (popped by sub helper's SCALAR B handling) |
| c_25 | Wt | freshly pushed (exp tiles) |

#### Phase 3: Sum Reduction + Reciprocal
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_25, c_2, c_26, ReduceInputBlockShape::of(1, Wt, 1),
    ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    });
```
- A: c_25 [Wt tiles, waited upfront by helper, NOT popped -- persists for Phase 4]
- Out: c_26 [1 tile, pushed by helper via bulk reserve/push for non-pop modes]

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_25 | Wt | waited, not popped -- reused in Phase 4 |
| c_26 | 1 | freshly pushed (recip_sum tile) |

#### Phase 4: Multiply by Reciprocal Sum
```cpp
compute_kernel_lib::mul<BroadcastDim::SCALAR,
    BinaryInputPolicy::NoWaitPopAtEnd,    // A: c_25, already waited, pop Wt at end
    BinaryInputPolicy::WaitAndPopPerTile, // B: c_26, wait+pop 1 tile (SCALAR)
    BinaryOutputPolicy::PerTile,          // per-tile output for writer overlap
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_25, c_26, c_16, BinaryInputBlockShape::of(1, Wt));
```
- A: c_25 [Wt tiles, ALREADY WAITED from Phase 3, popped at end]
- B: c_26 [1 tile, waited+popped by helper]
- Out: c_16 [per-tile reserve/push, double-buffered with writer]

**CB state after Phase 4:**
| CB | Tiles | State |
|----|-------|-------|
| c_25 | 0 | freed |
| c_26 | 0 | freed |
| c_16 | streaming | writer drains per tile |

### Compute Kernel (dim=-2: compute_h.cpp)

**Startup**: `compute_kernel_hw_startup(c_0, c_2, c_16)`

Processing is per-chunk (loop over NC * ceil(Wt/chunk_size) chunks). For each chunk:

#### Phase 1: Max Reduction (REDUCE_COL)
```cpp
cb_wait_front(c_0, Ht * current_chunk);
compute_kernel_lib::reduce<MAX, REDUCE_COL,
    ReduceInputPolicy::NoWaitNoPop,
    ReduceDataFormatReconfigMode::NONE>(
    c_0, c_2, c_24, ReduceInputBlockShape::of(Ht, current_chunk, 1),
    ReduceInputMemoryLayout::with_row_stride(current_chunk));
```
- Input tiles are in chunked column order: stride = current_chunk
- Out: c_24 [current_chunk tiles, max per column]
- c_0 tiles persist (NoWaitNoPop)

#### Phase 2: Subtract Max + Exp (ROW broadcast)
```cpp
compute_kernel_lib::sub<BroadcastDim::ROW,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitUpfrontPopAtEnd,
    BinaryOutputPolicy::Bulk,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_0, c_24, c_25, BinaryInputBlockShape::of(Ht, current_chunk),
    [](uint32_t dst_idx) {
        exp_tile_init();
        exp_tile(dst_idx);
    });
```
- A: c_0 [Ht*chunk tiles, indexed access], B: c_24 [chunk tiles, ROW broadcast]
- Out: c_25 [Ht*chunk tiles, bulk]

#### Phase 3: Sum Reduction + Reciprocal (REDUCE_COL)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_COL,
    ReduceInputPolicy::WaitUpfrontNoPop,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_25, c_2, c_26, ReduceInputBlockShape::of(Ht, current_chunk, 1),
    ReduceInputMemoryLayout::with_row_stride(current_chunk),
    NoAccumulation{},
    [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    });
```

#### Phase 4: Multiply by Reciprocal Sum (ROW broadcast)
```cpp
compute_kernel_lib::mul<BroadcastDim::ROW,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_25, c_26, c_16, BinaryInputBlockShape::of(Ht, current_chunk));
```

### Writer Kernel (shared: writer.cpp)
Generic tile writer. Loops num_pages times: `cb_wait_front(c_16, 1)`, `noc_async_write_page`, `cb_pop_front(c_16, 1)`. Uses TensorAccessor for DRAM writes. Same kernel for both dim=-1 and dim=-2.

Note for dim=-2: output tiles are produced in chunked column order (within each chunk: row0 cols, row1 cols, ...). The writer maps back to correct output tile positions. The output tensor has the same shape as input, so tile IDs match input tile IDs -- the writer can use the same sequential start_id-based indexing as the reader.

### Critical Notes
1. **reduce with NoWaitNoPop requires explicit cb_wait_front before call.** The helper will NOT wait for tiles.
2. **reduce with WaitUpfrontNoPop does NOT pop tiles.** Caller must pop manually or use a subsequent op with PopAtEnd policy.
3. **reduce scaler CB (c_2) is persistent.** Pushed once by reader, waited once by first reduce call, never popped. All subsequent reduce calls see it already waited.
4. **SCALAR broadcast for dim=-1** (not COL): Since we process Ht=1 per row, the REDUCE_ROW output is a single 1x1 tile, so SCALAR broadcast is correct.
5. **ROW broadcast for dim=-2**: REDUCE_COL output has Row0 valid (column reduction produces row-shaped result), broadcast replicates across Ht rows.
6. **exp_tile_init must be called before exp_tile** in every post-op lambda. The init reconfigures SFPU.
7. **recip_tile_init must be called before recip_tile** in the post-reduce lambda.
8. **reduce_uninit is called internally by the reduce helper** at the end. The sub/mul helpers that follow need INPUT_AND_OUTPUT reconfig to restore data format.
9. **dim=-2 output tile ordering**: Output tiles within a chunk emerge in the same order as input (chunked column). Since softmax preserves shape, output tile IDs = input tile IDs. The writer uses the same start_id mapping.

### Implementation Checklist
- [ ] Reader (dim=-1): Sequential read of Wt tiles per row, prepare_reduce_scaler
- [ ] Reader (dim=-2): Chunked column reads (Ht * chunk tiles per chunk), prepare_reduce_scaler
- [ ] Compute (dim=-1): 4 phases using reduce<MAX,REDUCE_ROW>, sub<SCALAR>+exp, reduce<SUM,REDUCE_ROW>+recip, mul<SCALAR>
- [ ] Compute (dim=-2): 4 phases using reduce<MAX,REDUCE_COL>, sub<ROW>+exp, reduce<SUM,REDUCE_COL>+recip, mul<ROW>
- [ ] Writer: Generic tile writer with TensorAccessor
- [ ] Program factory: dim-based kernel dispatch, CB sizing (dynamic chunk_size for dim=-2)
- [ ] CB push/pop balance verified per phase
