# Operation Design: softmax

## Overview
- **Operation Name**: softmax
- **Category**: eltwise (numerically-stable softmax with reduction)
- **Planning Mode**: Hybrid
- **Reference Operations**: tt-train softmax (compute_core), reduce_w (dim=-1 reduction), reduce_h (dim=-2 reduction)

## Mathematical Definition
```
output[n,c,h,w] = exp(x[n,c,h,w] - max_d(x)) / sum_d(exp(x - max_d(x)))
```
where `d` is the dimension specified by `dim` (-1 for width, -2 for height), and `max_d`/`sum_d` denote reduction along that dimension. When `numeric_stable=False`, the max subtraction is skipped.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| dim | int | Yes | {-1, -2} | -1 | Reduction dimension |
| numeric_stable | bool | No | {True, False} | True | Subtract max before exp |

### Input Tensor Requirements
| Property | Requirement |
|----------|-------------|
| Rank | 4 (N, C, H, W) |
| Dtype | bfloat16 |
| Layout | TILE_LAYOUT |
| Memory | DRAM interleaved |
| H, W | Divisible by 32 |

### Output Tensor Specification
- **Shape**: Same as input [N, C, H, W]
- **Dtype**: bfloat16
- **Layout**: TILE_LAYOUT
- **Memory**: DRAM interleaved

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile (32x32) | Reduce within one tile |
| H=32 with dim=-2 | Ht=1, reduce is trivial (single tile per column) |
| W=32 with dim=-1 | Wt=1, reduce is trivial (single tile per row) |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | reduce_w / reduce_h | input_stage | Multi-pass: reads same tiles 3x per row (dim=-1) or 3x per column (dim=-2). Generates scaler + mm_scaler tiles. |
| Compute | tt-train softmax + helpers | compute_core | 6-phase pipeline using reduce + binary_op helpers. Branching on DIM compile-time define. |
| Writer | reduce_w | output_stage | Standard tile writer, same shape as input. |

### Work Distribution
- **Work unit**: Row of tiles (dim=-1) or column of tiles (dim=-2)
- **Grid**: Single core (initial version)
- **Work per core**: All NCHt rows (dim=-1) or all NCWt columns (dim=-2)
- **Remainder**: N/A (single core)

### Data Flow

**dim=-1 (REDUCE_ROW)**: For each row of Wt tiles, the reader sends tiles 3 times (pass1: find max, pass2: exp-sub + sum, pass3: final multiply). The compute kernel reduces along W, producing column-vector intermediates that broadcast back across W tiles for subtraction and multiplication.

**dim=-2 (REDUCE_COL)**: For each column of Ht tiles, the reader sends tiles 3 times in column-chunked order. The compute kernel reduces along H, producing row-vector intermediates that broadcast back across H tiles.

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input | Input tiles from DRAM | Reader | Compute | 2 | Block (per tile) |
| c_1 | cb_scaler | Reduce scaler (all 1.0) | Reader | Compute | 1 | Program |
| c_2 | cb_mm_scaler | Matmul row-reduce scaler | Reader | Compute | 1 | Program |
| c_3 | cb_max | Max reduction output (col or row vector) | Compute | Compute | 2 | Row/Col (persists across passes) |
| c_4 | cb_exp_sum | Sum of exp output (col or row vector) | Compute | Compute | 2 | Row/Col |
| c_5 | cb_recip_sum | 1/sum (col or row vector) | Compute | Compute | 2 | Row/Col |
| c_16 | cb_output | Output tiles to DRAM | Compute | Writer | 2 | Block (per tile) |

**CB sizing rationale**:
- c_0: 2 tiles for double-buffering (reader/compute overlap, streaming 1-tile-at-a-time)
- c_1, c_2: 1 tile each, persistent constants
- c_3, c_4, c_5: 2 tiles for double-buffering compute self-production/consumption
- c_16: 2 tiles for double-buffering (compute/writer overlap)

### Kernel Arguments

**Compile-time** (all kernels share):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | Wt | uint32_t | Width in tiles |
| Reader | 1 | Ht | uint32_t | Height in tiles |
| Reader | 2+ | TensorAccessor | uint32_t[] | Input tensor accessor args |
| Compute | 0 | num_rows_or_cols | uint32_t | Total rows (dim=-1) or total cols (dim=-2) to process |
| Compute | 1 | inner_dim | uint32_t | Wt (dim=-1) or Ht (dim=-2): tiles along reduce dimension |
| Writer | 0 | num_output_tiles | uint32_t | Total output tiles |
| Writer | 1+ | TensorAccessor | uint32_t[] | Output tensor accessor args |

**Runtime** (per kernel):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | input_addr | uint32_t | Input buffer DRAM address |
| Writer | 0 | output_addr | uint32_t | Output buffer DRAM address |

**Preprocessor Defines** (passed to compute):
| Define | Value | Description |
|--------|-------|-------------|
| REDUCE_OP | PoolType::MAX | Required by reduce LLK |
| REDUCE_DIM | ReduceDim::REDUCE_ROW or REDUCE_COL | Set based on dim param |
| NUMERIC_STABLE | 1 or 0 | Enables max subtraction |
| DIM_W | 1 (if dim=-1) | Selects width reduction path |
| DIM_H | 1 (if dim=-2) | Selects height reduction path |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (1 tile for streaming)
- [x] Reduce scaler CB (c_1) is bfloat16 (Float16_b)
- [x] DEST register holds max 4 tiles (fp32_dest_acc_en=true, half-sync) -- all phases use <= 4 DST regs
- [x] All CBs count pages in tiles (TILE_LAYOUT throughout)
- [x] fp32_dest_acc_en=true for precision in sum accumulation

### Test Criteria
- Output shape matches input shape exactly
- Numerical accuracy vs `torch.nn.functional.softmax(x, dim=dim)`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile W | Tile iteration along W | `(1, 1, 32, 128)` |
| Multi-tile H | Tile iteration along H | `(1, 1, 128, 32)` |
| Non-square | W!=H | `(1, 1, 64, 256)` |
| Multi-batch | Batch handling | `(2, 3, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (input) | 2 | TILE bf16 | All | Block |
| c_1 (scaler) | 1 | TILE bf16 | Row0 of each face | Program |
| c_2 (mm_scaler) | 1 | TILE bf16 | Col0 of left faces | Program |
| c_3 (max) | 2 | TILE bf16 | Col0 (dim=-1) / Row0 (dim=-2) | Row/Col |
| c_4 (exp_sum) | 2 | TILE bf16 | Col0 (dim=-1) / Row0 (dim=-2) | Row/Col |
| c_5 (recip_sum) | 2 | TILE bf16 | Col0 (dim=-1) / Row0 (dim=-2) | Row/Col |
| c_16 (output) | 2 | TILE bf16 | All | Block |

**Note on precision**: The tt-train reference uses fp32 intermediate CBs (c_8, c_9) for sum accumulation. We use bf16 CBs but enable `fp32_dest_acc_en=true` so DST registers accumulate in fp32. The reduce helper + matmul-based sum achieves sufficient precision. If precision issues arise, the kernel writer should upgrade c_4/c_5 to fp32 data format.

### Binary Op Broadcast Verification

**dim=-1 (REDUCE_ROW produces column vector in c_3, c_5)**:
| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| subtract_max | SUB | All (input) | Col0 (max) | COL |
| multiply_recip | MUL | All (exp) | Col0 (recip_sum) | COL |

**dim=-2 (REDUCE_COL produces row vector in c_3, c_5)**:
| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| subtract_max | SUB | All (input) | Row0 (max) | ROW |
| multiply_recip | MUL | All (exp) | Row0 (recip_sum) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline | Reader+writer passthrough, compute copies input to output | `x` (identity) | same as input | N/A |
| 2 | exp_only | Compute applies exp to each tile | `torch.exp(x)` | same as input | N/A |
| 3 | softmax_dim_w | Full softmax for dim=-1 | `F.softmax(x, dim=-1)` | same as input | N/A |
| 4 | softmax_dim_h | Full softmax for dim=-2 | `F.softmax(x, dim=-2)` | same as input | N/A |

### Stage 1: data_pipeline
- **Scope**: reader, writer, compute (copy_tile only)
- **Reference**: `x` (identity passthrough)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(2,3,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute copies tiles from c_0 to c_16 using `copy_tile`. Reader sends all tiles once. All reduction phases skipped.
- **Kernels**: reader reads input tiles into c_0, compute copies c_0->c_16, writer writes c_16 to output.

### Stage 2: exp_only
- **Scope**: compute kernel adds exp_tile after copy
- **Reference**: `torch.exp(x)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(2,3,64,64)`
- **Tolerances**: rtol=0.01, atol=0.05
- **Delta from previous**: Compute now applies `exp_tile` to each tile in DST before packing to output. Tests that SFPU exp works correctly. Reader still sends tiles once.

### Stage 3: softmax_dim_w
- **Scope**: Full softmax for dim=-1 with all 6 phases
- **Reference**: `torch.nn.functional.softmax(x, dim=-1)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(2,3,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Complete rewrite of compute kernel with 3-pass reader, 6-phase compute (max, reduce_max, sub+exp+sum_accum, reduce_sum+recip, final_mul), writer. This is the core implementation.
- **Extra args**: `dim=-1`

### Stage 4: softmax_dim_h
- **Scope**: Full softmax for dim=-2
- **Reference**: `torch.nn.functional.softmax(x, dim=-2)`
- **Shapes**: `(1,1,32,32)`, `(1,1,128,64)`, `(1,1,256,32)`, `(2,3,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Compute kernel handles REDUCE_COL path. Reader sends tiles in column-chunked order for dim=-2. Different broadcast directions (ROW instead of COL).
- **Extra args**: `dim=-2`

### Reader Kernel

**Startup**: Generate two constant tiles:
1. `prepare_reduce_scaler<c_1>(1.0f)` -- all-1.0 scaler for `reduce_tile<MAX>`
2. `generate_mm_scaler(c_2, packed_bf16_1_0)` -- column-vector scaler for matmul-based SUM reduction

**Per-row/col loop** (3 passes):
- **Pass 1**: Read inner_dim tiles into c_0 (for max computation)
- **Pass 2**: Read inner_dim tiles into c_0 (for exp-sub + sum)
- **Pass 3**: Read inner_dim tiles into c_0 (for final multiply)

For dim=-1: tiles are read sequentially along W for each row. For dim=-2: tiles are read in column-major order (stride by Wt) for each column, matching reduce_h reader pattern.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_input, cb_scaler, cb_output)`

The compute kernel selects between dim=-1 and dim=-2 paths via `#ifdef DIM_W` / `#ifdef DIM_H`.

#### dim=-1 (REDUCE_ROW) Compute Phases

For each row of Wt tiles (total: NC * Ht rows):

**Phase 1: Find max along row**
```cpp
compute_kernel_lib::reduce<
    PoolType::MAX, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
    cb_input, cb_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
- Input: c_0 [Wt tiles, streamed 1-at-a-time from reader pass 1]
- Scaler: c_1 [1 tile, persistent]
- Output: c_3 (cb_max) [1 tile, column vector with row-maxes]

**Phase 2: Subtract max + exp + accumulate sum**

Process tiles one at a time from reader pass 2. For each of Wt tiles:
1. Wait for 1 tile in c_0 and 1 tile in c_3 (max, already available)
2. `sub_tiles_bcast<BroadcastType::COL>(c_0, c_3, ...)` -- subtract max
3. `exp_tile<false>(dst_idx)` -- exact exp in-place
4. Accumulate exp tile into running sum (in DST or via matmul_tiles for precision)
5. Pack exp result to c_16 (output) -- this IS the exp(x-max) value
6. Pop input tile

The sum accumulation uses `matmul_tiles(c_0_exp_tile, c_2_mm_scaler, ...)` to accumulate row sums with higher precision, following the tt-train pattern. After all Wt tiles:
7. Apply `recip_tile` to the accumulated sum
8. Pack reciprocal to c_5

**Phase 3: Final multiply**

Process tiles one at a time from reader pass 3. For each of Wt tiles:
1. Wait for 1 tile in c_0
2. Recompute: `sub_tiles_bcast<COL>(c_0, c_3, ...)` then `exp_tile<false>(...)`
3. `mul_tiles_bcast<COL>(exp_result, c_5, ...)` -- multiply by 1/sum
4. Pack to c_16 (output)
5. Pop input tile, pop c_3 and c_5 after last tile of row

**CB state after each row**:
| CB | State |
|----|-------|
| c_0 | empty (all tiles consumed) |
| c_3 | freed after Phase 3 |
| c_5 | freed after Phase 3 |

#### dim=-2 (REDUCE_COL) Compute Phases

For each column of Ht tiles (total: NC * Wt columns):

**Phase 1: Find max along column**
```cpp
compute_kernel_lib::reduce<
    PoolType::MAX, ReduceDim::REDUCE_COL,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
    cb_input, cb_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::col(Ht));
```
- Input: c_0 [Ht tiles, streamed 1-at-a-time from reader pass 1]
- Output: c_3 (cb_max) [1 tile, row vector with column-maxes]

**Phase 2: Subtract max + exp + accumulate sum**

Process Ht tiles from reader pass 2 one at a time:
1. sub_tiles_bcast<ROW>(input, max) -> exp -> accumulate sum via matmul
2. Pack exp results to c_16

After all Ht tiles: recip_tile on sum, pack to c_5.

**Phase 3: Final multiply**

Process Ht tiles from reader pass 3:
1. Recompute sub+exp, multiply by recip_sum broadcast ROW
2. Pack to c_16

### Writer Kernel

Simple sequential tile writer: waits for 1 tile in c_16, writes to DRAM via TensorAccessor, pops. Repeats for all output tiles (same count as input tiles since output shape = input shape).

### Critical Notes

1. **3-pass DRAM reads**: Each row (dim=-1) or column (dim=-2) requires 3 reads from DRAM. This matches the tt-train streaming path. A fits-in-L1 optimization can be added later.

2. **matmul_tiles for SUM precision**: The tt-train analysis explicitly notes that `reduce_tile<SUM>` causes precision loss. The matmul-based approach via `generate_mm_scaler` tile is mandatory for the sum phase.

3. **Phase 2 dual output**: Phase 2 must both (a) output exp tiles for the writer AND (b) accumulate the sum. Since we are in streaming mode (3-pass), exp tiles written in Phase 2 are NOT the final output -- they are discarded. Phase 3 recomputes exp(x-max) and multiplies by 1/sum to produce the final output. The writer only outputs Phase 3 results.

4. **DST register pressure**: With fp32_dest_acc_en=true, only 4 DST registers available (half-sync). The matmul accumulation uses 1 register. The sub+exp+mul pipeline uses 1-2 registers. This fits within limits.

5. **dim=-2 tile ordering**: The reader must produce tiles in column-major order for REDUCE_COL. This means for each column position wt, iterate over all Ht rows, reading tile at (ht * Wt + wt). This stride-Wt access pattern is non-sequential in DRAM.

6. **Scaler data format**: The reduce scaler CB (c_1) must be Float16_b regardless of input format, as required by the reduce_tile LLK. The mm_scaler CB (c_2) is also bf16.

### Implementation Checklist
- [ ] Reader: generate scaler tiles (c_1, c_2), 3-pass tile reads per row/col
- [ ] Compute: 3-phase pipeline (max, sub+exp+sum, mul) using reduce helper for max, raw LLK for sub/exp/mul/matmul, recip_tile for reciprocal
- [ ] Writer: sequential tile write via TensorAccessor
- [ ] CB push/pop balance verified across all 3 passes
- [ ] dim=-1 and dim=-2 paths both implemented and tested
