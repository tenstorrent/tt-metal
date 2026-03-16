# Operation Design: softmax

## Overview
- **Operation Name**: softmax
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: softmax (tt-train), reduce_w, reduce_h

## Mathematical Definition
```
softmax(x, dim)[i] = exp(x[i] - max(x, dim)) / sum(exp(x - max(x, dim)), dim)
```
Numerically stable softmax along dim=-1 (width) or dim=-2 (height). When `numeric_stable=False`, skip max subtraction: `exp(x[i]) / sum(exp(x), dim)`.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| dim | int | No | {-1, -2} | -1 | Dimension to reduce |
| numeric_stable | bool | No | {True, False} | True | Subtract max before exp |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| dtype | bfloat16 | "Input must be bfloat16" |
| layout | TILE_LAYOUT | "Input must be TILE_LAYOUT" |
| rank | >= 2 | "Input rank must be >= 2" |
| H, W | divisible by 32 | Tile-aligned (enforced by TILE_LAYOUT) |

### Output Tensor Specification
- **Shape**: same as input [N, C, H, W]
- **Dtype**: bfloat16
- **Layout**: TILE_LAYOUT
- **Memory**: DRAM interleaved

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| dim not in {-1, -2} | ValueError |
| Wrong dtype | ValueError |
| Wrong layout | ValueError |
| Rank < 2 | ValueError/RuntimeError |
| Single-tile tensor | Standard algorithm, Wt=1 or Ht=1 reduce is trivial |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Algorithm structure | tt-train softmax | compute_core | Simplify to streaming-only (no L1-fit path); add dim=-2 |
| dim=-1 reduce | reduce_w analysis | reduce_w_pattern | Use reduce helper with WaitAndPopPerTile |
| dim=-2 reduce | reduce_h analysis | reduce_h_pattern | Use reduce helper with WaitAndPopPerTile |
| Binary ops | binary_op_helpers | compute_core | Use sub/mul with COL or ROW broadcast |
| Exp+copy | copy_tile_helpers | compute_core | Use copy_tiles with exp post_op |

### Work Distribution
- **Work unit**: tile-row (dim=-1) or tile-column (dim=-2)
- **Grid**: 1D linearized, up to `compute_grid.x * compute_grid.y` cores
- **dim=-1**: total_units = NC * Ht rows; per_core = ceil/floor split via split_work_to_cores
- **dim=-2**: total_units = NC * Wt columns; per_core = ceil/floor split
- **Remainder**: Two-group strategy (group_1 gets ceil, group_2 gets floor)

### Data Flow

**Streaming 3-pass approach** (no L1-fit path -- simplifies implementation, works for all sizes):

**dim=-1 (width softmax)**: Reader reads each tile-row 3 times from DRAM.
1. Pass 1 (max): reader streams Wt tiles -> compute finds max -> cb_max holds 1 tile (column vector)
2. Pass 2 (exp+sum): reader streams Wt tiles -> compute does sub_max, exp, accumulates sum -> cb_recip holds 1 tile
3. Pass 3 (normalize): reader streams Wt tiles -> compute does sub_max, exp, mul_recip -> writer writes Wt output tiles

**dim=-2 (height softmax)**: Reader reads each tile-column Ht tiles, 3 times.
1. Pass 1 (max): reader streams Ht tiles per column -> compute finds max -> cb_max holds 1 tile (row vector)
2. Pass 2 (exp+sum): reader streams Ht tiles per column -> compute does sub_max, exp, accumulates sum -> cb_recip holds 1 tile
3. Pass 3 (normalize): reader streams Ht tiles per column -> compute does sub_max, exp, mul_recip -> writer writes Ht output tiles

**numeric_stable=False**: Skip passes 1; passes 2 and 3 merge into: exp, accumulate sum; then exp, mul_recip.

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 (0) | cb_input | Input tile staging | Reader | Compute | 2 | Per-tile (double-buffered) |
| c_1 (1) | cb_scaler | Reduce scaler tile (all 1.0) | Reader | Compute | 1 | Program |
| c_16 (16) | cb_output | Output tile staging | Compute | Writer | 2 | Per-tile (double-buffered) |
| c_24 (24) | cb_max | Reduced max tile | Compute | Compute | 1 | Per-row/col |
| c_25 (25) | cb_exp | Exp intermediate (pass 2 accumulation) | Compute | Compute | 2 | Per-tile |
| c_26 (26) | cb_recip | 1/sum(exp) tile | Compute | Compute | 1 | Per-row/col |

CB c_1 format: bfloat16 (scaler CB must be bfloat16).
CB c_25 format: bfloat16 (exp intermediate).
All other CBs: bfloat16.

### Kernel Arguments

**Compile-time (Reader)**:
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | Ht | uint32_t | Height in tiles |
| Reader | 1 | Wt | uint32_t | Width in tiles |
| Reader | 2 | HtWt | uint32_t | Ht * Wt (tiles per batch) |
| Reader | 3 | dim | uint32_t | 0 = width (dim=-1), 1 = height (dim=-2) |
| Reader | 4 | numeric_stable | uint32_t | 1 = stable, 0 = unstable |
| Reader | 5+ | TensorAccessorArgs | uint32_t[] | Input tensor accessor |

**Compile-time (Writer)**:
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | Ht | uint32_t | Height in tiles |
| Writer | 1 | Wt | uint32_t | Width in tiles |
| Writer | 2 | HtWt | uint32_t | Ht * Wt (tiles per batch) |
| Writer | 3 | dim | uint32_t | 0 = width, 1 = height |
| Writer | 4+ | TensorAccessorArgs | uint32_t[] | Output tensor accessor |

**Compile-time (Compute)**:
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Compute | 0 | Ht | uint32_t | Height in tiles |
| Compute | 1 | Wt | uint32_t | Width in tiles |
| Compute | 2 | num_rows_or_cols | uint32_t | Per-core work units (rows for dim=-1, cols for dim=-2) |
| Compute | 3 | dim | uint32_t | 0 = width, 1 = height |
| Compute | 4 | numeric_stable | uint32_t | 1 = stable, 0 = unstable |

**Runtime (Reader)**:
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input buffer DRAM address |
| Reader | 1 | num_work_units | uint32_t | Number of tile-rows (dim=-1) or tile-cols (dim=-2) for this core |
| Reader | 2 | start_work_unit | uint32_t | Starting tile-row or tile-col index |

**Runtime (Writer)**:
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | dst_addr | uint32_t | Output buffer DRAM address |
| Writer | 1 | num_work_units | uint32_t | Number of work units for this core |
| Writer | 2 | start_work_unit | uint32_t | Starting work unit index |

**Runtime (Compute)**: None. All parameters are compile-time.

### Compile-Time Defines

| Define | Condition | Value |
|--------|-----------|-------|
| REDUCE_OP | Always | PoolType::MAX (used by reduce helper default template) |
| REDUCE_DIM | dim=-1 | ReduceDim::REDUCE_ROW |
| REDUCE_DIM | dim=-2 | ReduceDim::REDUCE_COL |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (1 tile per wait for streaming)
- [x] Reduce scaler CB (c_1) is bfloat16
- [x] DEST register holds max 4 tiles (bf16 with fp32_dest_acc_en=true, half-sync)
- [x] All CBs use tile pages

### Test Criteria
- Output shape matches input shape exactly
- Numerical accuracy vs `torch.nn.functional.softmax(x, dim=dim)`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile W | Tile iteration along W | `(1, 1, 32, 128)` |
| Multi-tile H | Tile iteration along H | `(1, 1, 128, 32)` |
| Non-square | W!=H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (input) | 2 | TILE bf16 | All | Per-tile (double-buffered) |
| c_1 (scaler) | 1 | TILE bf16 | Row0 | Program (persistent) |
| c_16 (output) | 2 | TILE bf16 | All | Per-tile (double-buffered) |
| c_24 (max) | 1 | TILE bf16 | Col0 (dim=-1) / Row0 (dim=-2) | Per-row/col |
| c_25 (exp) | 2 | TILE bf16 | All | Per-tile (double-buffered) |
| c_26 (recip) | 1 | TILE bf16 | Col0 (dim=-1) / Row0 (dim=-2) | Per-row/col |

### Binary Op Broadcast Verification

**dim=-1 (width softmax)**:
| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|----|-------------------|-------------------|-----------|
| sub_max | SUB | All | Col0 (REDUCE_ROW output) | COL |
| mul_recip | MUL | All | Col0 (REDUCE_ROW output) | COL |

**dim=-2 (height softmax)**:
| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|----|-------------------|-------------------|-----------|
| sub_max | SUB | All | Row0 (REDUCE_COL output) | ROW |
| mul_recip | MUL | All | Row0 (REDUCE_COL output) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline | Reader/writer passthrough, compute identity copy | Input passthrough | Same as input | N/A |
| 2 | exp_passthrough | Compute applies exp to input | exp(input) | Same as input | N/A |
| 3 | softmax_dim_w | Full softmax dim=-1 (stable) | torch softmax dim=-1 | Same as input | N/A |
| 4 | softmax_dim_h | Full softmax dim=-2 (stable) | torch softmax dim=-2 | Same as input | N/A |

### Stage 1: data_pipeline
- **Scope**: reader, writer, compute (copy_tiles identity)
- **Reference**: `input` (passthrough)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **What to build**: Reader generates scaler in c_1, reads input tiles into c_0. Compute uses `copy_tiles(c_0, c_16, num_tiles)` to pass through. Writer reads c_16 and writes to DRAM. For dim=-1: reader streams Wt tiles per row, 1 pass. For dim=-2: reader streams Ht tiles per column, 1 pass. Both dim modes must work since dim is a CT arg. Test with dim=-1 only since passthrough is dim-agnostic.
- **CB bypass**: No compute phases beyond identity copy.

### Stage 2: exp_passthrough
- **Scope**: compute kernel adds exp post_op to copy_tiles
- **Reference**: `torch.exp(input)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.05
- **Delta from previous**: Compute uses `copy_tiles(c_0, c_16, num_tiles, [](uint32_t dst) { exp_tile_init(); exp_tile(dst); })` instead of plain copy. Reader does 1 pass. Tests dim=-1.
- **CB bypass**: Still single-pass, no reduce or broadcast ops.

### Stage 3: softmax_dim_w
- **Scope**: Full compute kernel for dim=-1 with all 3 passes (max, exp+sum, normalize)
- **Reference**: `torch.nn.functional.softmax(input, dim=-1)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Compute now implements full 3-pass softmax for dim=-1. Reader does 3 passes per row (stable) or 2 passes (unstable). This is a significant change but the phases are well-defined and use helper library calls.

### Stage 4: softmax_dim_h
- **Scope**: Compute kernel dim=-2 code path
- **Reference**: `torch.nn.functional.softmax(input, dim=-2)`
- **Shapes**: `(1,1,32,32)`, `(1,1,128,32)`, `(1,1,256,32)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Adds dim=-2 path with REDUCE_COL, ROW broadcast. Reader changes to column-major tile delivery for reduce. Tests use dim=-2.

### Reader Kernel

**dim=-1**: For each assigned tile-row, reader streams Wt tiles from DRAM into c_0, one at a time. In stable mode, this is done 3 times per row (passes for max, exp+sum, normalize). In unstable mode, 2 times (exp+sum, normalize). Between passes, compute synchronizes via CB handshake. Reader also generates scaler tile in c_1 at startup using `dataflow_kernel_lib::prepare_reduce_scaler<c_1>(1.0f)`.

**dim=-2**: For each assigned tile-column, reader streams Ht tiles per column from DRAM into c_0. In stable mode, 3 passes per column. Uses strided access: for column `w` in batch `nc`, tiles are at indices `nc * HtWt + ht * Wt + w` for `ht` in `[0, Ht)`. Reader also generates scaler tile in c_1 at startup.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_0, c_1, c_16)` then `binary_op_init_common(c_0, c_24, c_16)`

**dim=-1 (width softmax), stable mode, per row:**

#### Phase 1: Find max along row
```cpp
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_0, c_1, c_24,
    compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));
```
- A: c_0 [1 tile at a time, FRESHLY PUSHED by reader, pop per tile]
- B: c_1 [1 tile, persistent scaler]
- Out: c_24 [1 tile, max column vector]

#### Phase 2: Subtract max and exp, accumulate sum
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_0, c_24, c_25,
    BinaryInputBlockShape::of(1, Wt));
```
Then feed c_25 (exp of subtracted values) through reduce for sum:
Actually, this phase is more complex -- we need to compute exp(x-max) and simultaneously accumulate sum. The approach:

For each tile in the row (Wt tiles):
1. Reader pushes 1 tile to c_0
2. Compute: sub(c_0, c_24) -> exp -> pack to c_25
3. Feed c_25 tiles through reduce<SUM, REDUCE_ROW> into c_26

Implementation: Use `sub<COL>` with exp post_op to produce exp(x-max) tiles in c_25, then `reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>` on c_25 with recip_tile post_reduce_op to get 1/sum in c_26.

```cpp
// Step 2a: Compute exp(x - max) for all Wt tiles, output to c_25
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_0, c_24, c_25,
    BinaryInputBlockShape::of(1, Wt),
    [](uint32_t dst) { exp_tile_init(); exp_tile(dst); });
// c_24 (max) NOT popped -- NoWaitNoPop for B. Caller pops after phase 3.

// Step 2b: Sum exp tiles and compute reciprocal
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_25, c_1, c_26,
    compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [](uint32_t dst) { recip_tile_init(); recip_tile(dst); });
```

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_24 (max) | 1 | waited, not popped -- persists for Phase 3 |
| c_25 (exp) | 0 | all consumed by reduce |
| c_26 (recip) | 1 | freshly pushed -- 1/sum(exp) |

#### Phase 3: Normalize (sub max, exp, multiply by recip)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_0, c_24, c_25,
    BinaryInputBlockShape::of(1, Wt),
    [](uint32_t dst) { exp_tile_init(); exp_tile(dst); });
// c_24 still not popped

compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_25, c_26, c_16,
    BinaryInputBlockShape::of(1, Wt));

// Manual cleanup: pop persistent CBs
cb_pop_front(c_24, 1);  // max
cb_pop_front(c_26, 1);  // recip
```

- A: c_25 [exp tiles, freshly produced by sub, popped per tile by mul]
- B: c_26 [1 tile, 1/sum, persistent during mul, popped manually after]
- Out: c_16 [Wt output tiles, writer consumes]

**dim=-2 (height softmax), stable mode, per column:**

Same 3-phase algorithm but with REDUCE_COL and ROW broadcast:

#### Phase 1: Find max along column
```cpp
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_0, c_1, c_24,
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, 1, 1));
```

#### Phase 2: exp(x - max) + sum + recip
```cpp
compute_kernel_lib::sub<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_0, c_24, c_25,
    BinaryInputBlockShape::of(Ht, 1),
    [](uint32_t dst) { exp_tile_init(); exp_tile(dst); });

compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_25, c_1, c_26,
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, 1, 1),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [](uint32_t dst) { recip_tile_init(); recip_tile(dst); });
```

#### Phase 3: Normalize
```cpp
compute_kernel_lib::sub<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_0, c_24, c_25,
    BinaryInputBlockShape::of(Ht, 1),
    [](uint32_t dst) { exp_tile_init(); exp_tile(dst); });

compute_kernel_lib::mul<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_25, c_26, c_16,
    BinaryInputBlockShape::of(Ht, 1));

cb_pop_front(c_24, 1);
cb_pop_front(c_26, 1);
```

### Writer Kernel

**dim=-1**: For each assigned row, waits for Wt tiles in c_16, writes them sequentially to DRAM. Uses TensorAccessor for page addressing.

**dim=-2**: For each assigned column, waits for Ht tiles in c_16, writes them with strided addressing to DRAM (tile at `nc * HtWt + ht * Wt + w`).

### Critical Notes

1. **Reader-compute synchronization for multi-pass**: The reader must not send pass 2 tiles until compute has finished consuming pass 1. Since c_0 is double-buffered (2 pages) and compute pops each tile, the natural CB backpressure handles this -- the reader blocks on `cb_reserve_back(c_0, 1)` if compute hasn't popped yet. Between passes, the CB is naturally empty because compute has popped all tiles.

2. **c_24 (max) and c_26 (recip) persistence**: These CBs use NoWaitNoPop on the B input of binary ops, so they persist across the sub and mul phases. The caller MUST manually `cb_pop_front` after the last use in each row/column.

3. **REDUCE_DIM define**: The compute kernel needs `REDUCE_OP` and `REDUCE_DIM` defines. Since dim=-1 uses REDUCE_ROW and dim=-2 uses REDUCE_COL, these should be set based on dim. The MAX reduce in phase 1 uses the define, and the SUM reduce in phase 2 explicitly specifies PoolType::SUM. Use `REDUCE_DIM` define that matches the dim parameter, and set `REDUCE_OP` to `PoolType::MAX` (for the first reduce). The SUM reduce passes explicit template args.

4. **sub helper with exp post_op**: The `sub` binary op helper supports a post_op lambda. The exp is applied to each tile in DST after the subtraction, before packing to output CB. This avoids needing a separate copy+exp phase.

5. **dim=-2 column addressing**: For REDUCE_COL with WaitAndPopPerTile, tiles arrive one at a time. The reader must deliver tiles for a single column (Ht tiles at stride Wt) sequentially. The reduce helper processes them in order.

6. **fp32_dest_acc_en**: Set to true for better precision in sum accumulation. This limits DEST to 4 tiles (half-sync) but softmax only needs 1-2 DST registers at a time.

### Implementation Checklist
- [ ] Reader: TensorAccessor input reads, `prepare_reduce_scaler<c_1>(1.0f)`, 3-pass row/col streaming
- [ ] Compute: 3 phases using helpers: `reduce<MAX>`, `sub<COL/ROW>` + exp post_op + `reduce<SUM>` + recip post_op, `sub<COL/ROW>` + exp post_op + `mul<COL/ROW>`
- [ ] Writer: TensorAccessor output writes, per-tile streaming from c_16
- [ ] CB push/pop balance verified (manual pop for c_24, c_26 after each row/col)
