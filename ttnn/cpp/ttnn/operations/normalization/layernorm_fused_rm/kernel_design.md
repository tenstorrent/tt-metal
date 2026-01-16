# Kernel Design: layernorm_fused_rm

## Design Summary

| Kernel | Phases | Helpers Used | Raw Calls Needed |
|--------|--------|--------------|------------------|
| Reader | 5 | None (dataflow) | NOC reads, CB management, scalar tile generation |
| Compute | 12 | tilize, reduce, sub, mul, add, untilize | mul_tiles (square), add_tiles (eps), rsqrt_tile |
| Writer | 1 | None (dataflow) | NOC writes, CB management |

## Helper Library Analysis

### Available Helpers Reviewed
- [x] tilize_helpers.hpp - **relevant: YES** - for input, gamma, beta tilization
- [x] untilize_helpers.hpp - **relevant: YES** - for output untilization
- [x] reduce_helpers.hpp - **relevant: YES** - for mean and variance computation (reduce row)
- [x] binary_op_helpers.hpp - **relevant: YES** - for sub/mul/add with broadcast
- [x] dest_helpers.hpp - **relevant: YES** - DEST_AUTO_LIMIT for register capacity

### Helper Functions Applicable to This Operation

| Helper | Signature | Use Case in This Op |
|--------|-----------|---------------------|
| `compute_kernel_lib::tilize()` | `tilize(icb, block_w, ocb, num_blocks, subblock_h, old_icb, input_count, total_rows)` | Tilize input (c_0->c_1), gamma (c_4->c_6), beta (c_5->c_7) |
| `compute_kernel_lib::reduce<SUM, REDUCE_ROW>()` | `reduce<PoolType, ReduceDim>(icb, icb_scaler, ocb, TileShape, TileLayout, accum, post_op)` | Mean computation, variance computation |
| `compute_kernel_lib::sub<BroadcastDim::COL>()` | `sub<BroadcastDim::COL>(icb_a, icb_b, ocb, BinaryTileShape)` | Center values (x - mean) |
| `compute_kernel_lib::mul<BroadcastDim::COL>()` | `mul<BroadcastDim::COL>(icb_a, icb_b, ocb, BinaryTileShape)` | Normalize (centered * inv_std) |
| `compute_kernel_lib::mul<BroadcastDim::ROW>()` | `mul<BroadcastDim::ROW>(icb_a, icb_b, ocb, BinaryTileShape)` | Apply gamma (normalized * gamma) |
| `compute_kernel_lib::add<BroadcastDim::ROW>()` | `add<BroadcastDim::ROW>(icb_a, icb_b, ocb, BinaryTileShape)` | Apply beta (scaled + beta) |
| `compute_kernel_lib::untilize<Wt, icb, ocb>()` | `untilize<tile_width, icb_id, ocb_id>(num_rows)` | Untilize output (tiled->c_16 RM) |

## Reader Kernel Design

### Prerequisites
- Requires `#include "api/dataflow/dataflow_api.h"`
- Uses TensorAccessor for DRAM access

### Phase 1: Read Scaler Tile
- **Description**: Generate 1/W scaler tile for reduce operations
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - `cb_reserve_back(c_2, 2)` - reserve space for scaler tile (2 tiles for reduce helper)
    - Generate scaler value using `bfloat16(1.0f / W).to_uint16()`
    - Fill tile with scalar using `fill_cb_with_value()` or manual loop
    - `cb_push_back(c_2, 2)`
- **CB Flow**: c_2 (scaler) - produced once at program start, never popped

### Phase 2: Read Epsilon Tile
- **Description**: Generate epsilon scalar tile for variance stabilization
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel)
  - **RAW CALLS**:
    - `cb_reserve_back(c_3, 1)`
    - Generate column-broadcast epsilon tile (fill column 0 with epsilon value)
    - `cb_push_back(c_3, 1)`
- **CB Flow**: c_3 (epsilon) - produced once, persists for program duration

### Phase 3: Read Gamma (Once)
- **Description**: Read gamma tensor from DRAM in RM format
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel)
  - **RAW CALLS**:
    - `cb_reserve_back(c_4, Wt)` - reserve for full row of gamma sticks
    - TensorAccessor to get NOC addresses for gamma sticks
    - `noc_async_read()` for each stick
    - `noc_async_read_barrier()`
    - `cb_push_back(c_4, Wt)`
- **CB Flow**: c_4 (gamma RM) - produced once, consumed by compute tilize phase

### Phase 4: Read Beta (Once)
- **Description**: Read beta tensor from DRAM in RM format
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel)
  - **RAW CALLS**:
    - `cb_reserve_back(c_5, Wt)` - reserve for full row of beta sticks
    - TensorAccessor to get NOC addresses for beta sticks
    - `noc_async_read()` for each stick
    - `noc_async_read_barrier()`
    - `cb_push_back(c_5, Wt)`
- **CB Flow**: c_5 (beta RM) - produced once, consumed by compute tilize phase

### Phase 5: Read Input Rows (Per-Row Loop)
- **Description**: Read input tensor row by row (32 sticks per tile row)
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel)
  - **RAW CALLS**:
    - For each tile row (Ht iterations):
      - `cb_reserve_back(c_0, Wt)` - reserve for Wt tiles worth of sticks
      - For each of 32 sticks in tile row:
        - TensorAccessor.get_noc_addr(stick_id)
        - `noc_async_read(noc_addr, l1_addr, stick_size)`
        - Advance l1_addr by stick_size
      - `noc_async_read_barrier()`
      - `cb_push_back(c_0, Wt)`
- **CB Flow**: c_0 (input RM) - produced per row, consumed by compute tilize phase

## Compute Kernel Design

### Prerequisites
- [x] Requires `compute_kernel_hw_startup()`: YES - must be called at kernel start
- [x] Template parameters for reduce helper:
  - `PoolType`: SUM
  - `ReduceDim`: REDUCE_ROW
  - `ReduceInputMode`: STREAMING (default) - processes tiles one at a time
  - `ReduceDataFormatReconfig`: BOTH (default)

### Initialization Phase
- **Description**: Initialize compute kernel hardware
- **Implementation Approach**:
  - `compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out_rm)` with appropriate CBs

### Phase 1: Tilize Gamma (ONCE)
- **Description**: Convert gamma from RM (c_4) to tiled format (c_6)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::tilize()`
  - **Parameters**: `tilize(c_4, Wt, c_6, 1)` - tilize 1 block of Wt tiles
  - **CB Management**: Helper handles cb_wait_front(c_4), cb_pop_front(c_4), cb_reserve_back(c_6), cb_push_back(c_6) internally
- **CB Flow**: c_4 -> c_6 (gamma persists in c_6, NOT popped after tilize)

### Phase 2: Tilize Beta (ONCE)
- **Description**: Convert beta from RM (c_5) to tiled format (c_7)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::tilize()`
  - **Parameters**: `tilize(c_5, Wt, c_7, 1)` - tilize 1 block of Wt tiles
  - **CB Management**: Helper handles internally
- **CB Flow**: c_5 -> c_7 (beta persists in c_7, NOT popped after tilize)

### Phase 3: Tilize Input (Per-Row)
- **Description**: Convert input row from RM (c_0) to tiled format (c_1)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::tilize()`
  - **Parameters**: `tilize(c_0, Wt, c_1, 1)` per row iteration
  - **CB Management**: Helper handles cb_wait_front(c_0, Wt), cb_reserve_back(c_1, Wt), tilize_block, cb_pop_front(c_0, Wt), cb_push_back(c_1, Wt)
- **CB Flow**: c_0 -> c_1 (per row, c_1 is double-buffered)

### Phase 4: Compute Mean (Per-Row)
- **Description**: Reduce row to compute mean: mean = reduce_row(c_1) * scaler
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>()`
  - **Parameters**: `reduce<SUM, REDUCE_ROW>(c_1, c_2, c_25, TileShape::row(Wt))`
    - Uses STREAMING mode (default): waits/pops tiles one-at-a-time
    - Scaler tile in c_2 applies 1/W automatically
  - **CB Management**: Helper handles tile_regs_acquire/commit/wait/release, cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back, pack_tile internally
  - **IMPORTANT**: Input tiles from c_1 are consumed (popped) by reduce. Need to re-read or use PERSISTENT mode.
- **CB Flow**: c_1, c_2 -> c_25 (mean tile, 1 tile output)

**DESIGN NOTE - Mean/Variance Data Reuse Issue:**
The spec shows c_1 tiles being used for:
1. Mean computation (Phase 4)
2. Centering subtraction (Phase 5)
3. Variance computation (via squares, Phase 7)

Since reduce in STREAMING mode pops input tiles, we need to either:
- Option A: Re-tilize input for each use (wasteful)
- Option B: Use PERSISTENT mode for reduce (tiles persist for reuse)
- Option C: Use double-buffered intermediate CBs

**Recommended**: Use PERSISTENT mode for mean reduce, then use same c_1 tiles for centering.

### Phase 5: Subtract Mean (Center Values) (Per-Row)
- **Description**: Compute centered values: x - mean (broadcast mean across columns)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::sub<BroadcastDim::COL>()`
  - **Parameters**: `sub<BroadcastDim::COL>(c_1, c_25, c_24, BinaryTileShape::row(Wt))`
    - c_1: input tiles (Wt tiles)
    - c_25: mean tile (1 tile, broadcast across all columns)
    - c_24: centered output (Wt tiles)
  - **CB Management**: Helper handles internally
- **CB Flow**: c_1, c_25 -> c_24 (Wt tiles of centered values)

### Phase 6: Square Centered Values (Per-Row)
- **Description**: Compute squared deviations: (x - mean)^2
- **Implementation Approach**:
  - **USE HELPER**: No - squaring requires element-wise mul of same tensor
  - **RAW CALLS**:
    - `mul_tiles_init(c_24, c_24)`
    - For each tile in c_24:
      - `tile_regs_acquire()`
      - `cb_wait_front(c_24, 1)` or use indexed access
      - `mul_tiles(c_24, c_24, tile_idx, tile_idx, 0)` - square tile
      - `tile_regs_commit()`, `tile_regs_wait()`
      - Pack to intermediate CB or use in-place
      - `tile_regs_release()`
  - **Alternative**: Use `compute_kernel_lib::mul<BroadcastDim::NONE>()` with same CB for both inputs
- **CB Flow**: c_24 -> intermediate (squared values, Wt tiles)

**DESIGN NOTE**: The binary helper `mul<NONE>` could work if we read from same CB twice, but this requires careful CB indexing. Raw mul_tiles may be cleaner.

### Phase 7: Compute Variance (Per-Row)
- **Description**: Reduce squared values to variance: var = reduce_row(squares) * scaler
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>()`
  - **Parameters**: `reduce<SUM, REDUCE_ROW>(cb_squares, c_2, c_26, TileShape::row(Wt))`
  - **CB Management**: Helper handles internally
- **CB Flow**: cb_squares, c_2 -> c_26 (variance tile, 1 tile)

### Phase 8: Add Epsilon (Per-Row)
- **Description**: Add epsilon to variance for numerical stability: var + eps
- **Implementation Approach**:
  - **USE HELPER**: No - epsilon is scalar broadcast from column 0
  - **RAW CALLS**:
    - `add_bcast_cols_init_short(c_26, c_3)`
    - `tile_regs_acquire()`
    - `cb_wait_front(c_26, 1)`
    - `cb_wait_front(c_3, 1)` - epsilon tile
    - `add_tiles_bcast_cols(c_26, c_3, 0, 0, 0)`
    - `tile_regs_commit()`, `tile_regs_wait()`
    - Pack result (var + eps) - can overwrite c_26 or use temp
    - `tile_regs_release()`
  - **Alternative**: Use `add<BroadcastDim::COL>()` helper with c_3 as scalar CB
- **CB Flow**: c_26, c_3 -> intermediate (var + eps, 1 tile)

### Phase 9: Reciprocal Square Root (Per-Row)
- **Description**: Compute inverse std: 1/sqrt(var + eps)
- **Implementation Approach**:
  - **USE HELPER**: No - rsqrt is not covered by helpers
  - **RAW CALLS**:
    - `rsqrt_tile_init()`
    - `tile_regs_acquire()`
    - `cb_wait_front(cb_var_eps, 1)` - (var + eps) from previous phase
    - `rsqrt_tile(0)` - compute rsqrt in DST[0]
    - `cb_reserve_back(c_27, 1)`
    - `tile_regs_commit()`, `tile_regs_wait()`
    - `pack_tile(0, c_27)`
    - `cb_push_back(c_27, 1)`
    - `tile_regs_release()`
- **CB Flow**: (var+eps) -> c_27 (inv_std, 1 tile)

### Phase 10: Normalize (Per-Row)
- **Description**: Multiply centered values by inverse std: (x - mean) * inv_std
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::mul<BroadcastDim::COL>()`
  - **Parameters**: `mul<BroadcastDim::COL>(c_24, c_27, cb_normalized, BinaryTileShape::row(Wt))`
    - c_24: centered values (Wt tiles)
    - c_27: inv_std (1 tile, broadcast across columns)
    - cb_normalized: normalized output (Wt tiles)
  - **CB Management**: Helper handles internally
- **CB Flow**: c_24, c_27 -> cb_normalized (Wt tiles)

### Phase 11: Apply Gamma (Per-Row)
- **Description**: Scale by gamma: normalized * gamma (broadcast gamma across rows)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::mul<BroadcastDim::ROW>()`
  - **Parameters**: `mul<BroadcastDim::ROW>(cb_normalized, c_6, cb_scaled, BinaryTileShape::row(Wt))`
    - cb_normalized: normalized values (Wt tiles)
    - c_6: gamma (Wt tiles, PERSISTENT - same for all rows)
    - cb_scaled: scaled output (Wt tiles)
  - **CB Management**: Helper handles internally
  - **NOTE**: c_6 (gamma) must NOT be popped - use PERSISTENT input mode or ensure helper doesn't pop B in ROW broadcast
- **CB Flow**: cb_normalized, c_6 -> cb_scaled (Wt tiles)

### Phase 12: Apply Beta (Per-Row)
- **Description**: Add beta bias: scaled + beta (broadcast beta across rows)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::add<BroadcastDim::ROW>()`
  - **Parameters**: `add<BroadcastDim::ROW>(cb_scaled, c_7, cb_final, BinaryTileShape::row(Wt))`
    - cb_scaled: scaled values (Wt tiles)
    - c_7: beta (Wt tiles, PERSISTENT)
    - cb_final: final result (Wt tiles)
  - **CB Management**: Helper handles internally
  - **NOTE**: c_7 (beta) must NOT be popped
- **CB Flow**: cb_scaled, c_7 -> cb_final (Wt tiles)

### Phase 13: Untilize Output (Per-Row)
- **Description**: Convert final result from tiled to RM format
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::untilize<Wt, cb_final, c_16>()`
  - **Parameters**: `untilize<Wt, cb_final, c_16>(1)` - untilize 1 row
  - **CB Management**: Helper handles cb_wait_front, cb_reserve_back, untilize_block/pack_untilize_block, cb_pop_front, cb_push_back internally
- **CB Flow**: cb_final -> c_16 (RM output, Wt tiles worth of sticks)

## Writer Kernel Design

### Prerequisites
- Requires `#include "api/dataflow/dataflow_api.h"`
- Uses TensorAccessor for DRAM writes

### Phase 1: Write Output Rows (Per-Row Loop)
- **Description**: Write RM output sticks to DRAM
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - For each tile row (Ht iterations):
      - `cb_wait_front(c_16, Wt)` - wait for Wt tiles worth of RM data
      - `l1_addr = get_read_ptr(c_16)`
      - For each of 32 sticks in tile row:
        - `noc_async_write(l1_addr, noc_addr, stick_size)`
        - Advance l1_addr and stick_id
      - `noc_async_write_barrier()`
      - `cb_pop_front(c_16, Wt)`
- **CB Flow**: c_16 (RM output) consumed per row

## CB Synchronization Summary

| CB | Producer | Consumer | Pages per Block | Sync Point |
|----|----------|----------|-----------------|------------|
| c_0 | Reader | Compute (tilize) | Wt | Per tile-row: Reader pushes Wt, Compute pops Wt |
| c_1 | Compute (tilize) | Compute (reduce+sub) | 2*Wt (double buffer) | Per tile-row: tilize pushes Wt, reduce/sub consume |
| c_2 | Reader | Compute (reduce) | 2 | Once: Reader pushes 2, persists for program |
| c_3 | Reader | Compute (add eps) | 1 | Once: Reader pushes 1, persists for program |
| c_4 | Reader | Compute (tilize) | Wt | Once: Reader pushes Wt, tilize pops Wt |
| c_5 | Reader | Compute (tilize) | Wt | Once: Reader pushes Wt, tilize pops Wt |
| c_6 | Compute (tilize) | Compute (mul gamma) | Wt | Once: tilize pushes Wt, PERSISTENT - no pop |
| c_7 | Compute (tilize) | Compute (add beta) | Wt | Once: tilize pushes Wt, PERSISTENT - no pop |
| c_16 | Compute (untilize) | Writer | Wt | Per tile-row: untilize pushes Wt, Writer pops Wt |
| c_24 | Compute (sub) | Compute (square+normalize) | Wt | Per tile-row |
| c_25 | Compute (reduce mean) | Compute (sub) | 1 | Per tile-row: reduce pushes 1, sub uses 1 |
| c_26 | Compute (reduce var) | Compute (add eps) | 1 | Per tile-row |
| c_27 | Compute (rsqrt) | Compute (normalize mul) | 1 | Per tile-row: rsqrt pushes 1, normalize uses 1 |

## Helper Encapsulation Acknowledgment

For phases marked "USE HELPER", the following is encapsulated BY THE HELPER:
- CB wait/pop/reserve/push operations
- DST register management (tile_regs_acquire/commit/wait/release)
- Init/uninit sequences (tilize_init, reduce_init, add_bcast_rows_init, etc.)

**CRITICAL**: The kernel writer MUST NOT add redundant CB or DST operations around helper calls. The helper functions are self-contained.

### Specifically:

| Helper | CB Ops Encapsulated | DST Ops Encapsulated |
|--------|---------------------|----------------------|
| `tilize()` | cb_wait_front, cb_reserve_back, cb_pop_front, cb_push_back | tilize_block handles DST |
| `untilize()` | cb_wait_front, cb_reserve_back, cb_pop_front, cb_push_back | pack_untilize handles DST |
| `reduce()` | cb_wait_front (scaler + input), cb_pop_front, cb_reserve_back, cb_push_back | tile_regs_acquire/commit/wait/release, reduce_tile, pack_tile |
| `sub<COL>()` | cb_wait_front (both inputs), cb_pop_front, cb_reserve_back, cb_push_back | tile_regs_acquire/commit/wait/release, sub_tiles_bcast_cols, pack_tile |
| `mul<COL>()` | Same as sub | Same as sub |
| `mul<ROW>()` | Same pattern, B tiles may persist depending on mode | Same pattern |
| `add<ROW>()` | Same pattern | Same pattern |

## Intermediate CB Requirements

Based on the compute flow, additional intermediate CBs may be needed:

| Logical Purpose | Suggested CB | Capacity | Notes |
|-----------------|--------------|----------|-------|
| Squared values | c_28 or reuse c_24 | Wt tiles | After squaring, c_24 can be reused |
| Normalized values | c_29 or reuse c_1 | Wt tiles | After reduce consumes c_1 |
| Scaled values | c_30 or reuse | Wt tiles | Before beta add |
| Final tiled result | c_31 or c_1 | Wt tiles | Before untilize |

**Optimization**: Many intermediate results can share CBs if processed sequentially and not needed simultaneously.

## Implementation Checklist for Kernel Writer

- [ ] Reader:
  - [ ] Generate scaler tile (1/W) into c_2 (2 tiles for reduce helper compatibility)
  - [ ] Generate epsilon tile into c_3 (column broadcast format)
  - [ ] Read gamma RM sticks into c_4, push Wt pages
  - [ ] Read beta RM sticks into c_5, push Wt pages
  - [ ] Per-row loop: read 32 input sticks into c_0, push Wt pages

- [ ] Compute:
  - [ ] Call `compute_kernel_hw_startup()`
  - [ ] ONE-TIME: `tilize(c_4, Wt, c_6, 1)` for gamma
  - [ ] ONE-TIME: `tilize(c_5, Wt, c_7, 1)` for beta
  - [ ] Per-row loop:
    - [ ] `tilize(c_0, Wt, c_1, 1)` for input
    - [ ] `reduce<SUM, REDUCE_ROW, PERSISTENT>(c_1, c_2, c_25, ...)` for mean (PERSISTENT keeps c_1)
    - [ ] `sub<BroadcastDim::COL>(c_1, c_25, c_24, ...)` for centering
    - [ ] Raw `mul_tiles()` for squaring c_24 -> intermediate
    - [ ] `reduce<SUM, REDUCE_ROW>(squares, c_2, c_26, ...)` for variance
    - [ ] `add<BroadcastDim::COL>(c_26, c_3, ...)` or raw add_tiles_bcast_cols for var+eps
    - [ ] Raw `rsqrt_tile()` for inv_std -> c_27
    - [ ] `mul<BroadcastDim::COL>(c_24, c_27, ...)` for normalization
    - [ ] `mul<BroadcastDim::ROW>(normalized, c_6, ...)` for gamma (c_6 PERSISTENT)
    - [ ] `add<BroadcastDim::ROW>(scaled, c_7, ...)` for beta (c_7 PERSISTENT)
    - [ ] `untilize<Wt, cb_final, c_16>(1)` for output

- [ ] Writer:
  - [ ] Per-row loop: wait for c_16 Wt pages, write 32 sticks, pop Wt pages

- [ ] Verify: CB push/pop counts match across kernels for each CB

## Design Decisions and Rationale

### Decision 1: PERSISTENT Mode for Mean Reduce
Using PERSISTENT mode for the mean reduce phase allows c_1 tiles to remain available for the subsequent centering subtraction without re-reading from DRAM.

### Decision 2: Helper for Most Phases
All broadcast operations (sub, mul, add with COL/ROW) use helpers because:
- Helpers correctly manage DST registers
- Helpers handle CB synchronization
- Helpers are tested and maintained

### Decision 3: Raw Calls for Specific Operations
Raw calls are required for:
- **Square**: Element-wise self-multiplication is not a standard helper pattern
- **Add Epsilon**: Could use helper, but single-tile operation is simple enough raw
- **rsqrt_tile**: No helper exists for this SFPU operation

### Decision 4: CB Reuse Strategy
To minimize L1 usage, intermediate CBs should be reused:
- After tilize consumes c_4/c_5, those CBs are free
- After reduce consumes c_1, that CB can be reused for later outputs
- c_24 can be reused after squaring if squared values go elsewhere
