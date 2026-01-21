# Kernel Design: variance_w_rm

## Spec Validation Issues

### Issue 1: CB_4 Does Not Need Persistence After Square

- **Spec says**: "CB_4 (centralized tiles) uses the same PERSISTENT pattern as CB_1 in centralize_w_rm"
- **Problem**: CB_4 is only consumed by the square phase (Phase 4). After squaring, the centralized tiles are no longer needed. The squared values in CB_5 are what gets reduced.
- **Resolution**: CB_4 can use standard streaming policy in the square phase - wait upfront for Wt tiles, process them, pop at end. No persistence needed.

### Issue 2: Square Phase Input Policy Clarification

- **Spec says**: "Use element-wise mul (A*A) for squaring" with some ambiguity about CB handling
- **Problem**: The binary_op_helpers.hpp has a dedicated `square()` function that takes the same input CB for both A and B operands internally. The spec's suggested policies need clarification.
- **Resolution**: Use the `square()` convenience function from binary_op_helpers.hpp. Input policy should be `WaitUpfront, PopAtEnd` since centralized tiles are all present after Phase 3.

No other spec validation issues found.

## Data Semantics Model (MANDATORY)

### Buffer Content Analysis

| CB | Layout | Logical Shape | Tile Shape | Valid Region | Lifetime |
|----|--------|---------------|------------|--------------|----------|
| c_0 | ROW_MAJOR | [32, W] (32 sticks per block) | N/A | All | Block |
| c_1 | TILE | [32, W] | 1 x Wt | All | Block (persistent through reduce) |
| c_2 | TILE | [1] (scaler) | 1 x 1 | All (filled with 1/W) | Program |
| c_3 | TILE | [32, 1] (mean per row) | 1 x 1 | **Col0** (REDUCE_ROW output) | Block |
| c_4 | TILE | [32, W] (centralized) | 1 x Wt | All | Block |
| c_5 | TILE | [32, W] (squared) | 1 x Wt | All | Block |
| c_6 | TILE | [32, 1] (variance per row) | 1 x 1 | **Col0** (REDUCE_ROW output) | Block |
| c_16 | ROW_MAJOR | [32, 32] (output sticks) | N/A | **Col0** (first column only, reduced width) | Block |

**Key observations**:
1. **c_0**: Row-major sticks from DRAM, 32 sticks form one tile-row, all elements valid
2. **c_1**: After tilize, full [32, W] data in tiled format. **PERSISTS** through reduce phase for bcast_sub.
3. **c_2**: Scaler tile with value 1/W for both mean and variance calculations (persistent for entire program)
4. **c_3**: After REDUCE_ROW (mean), output is [32, 1] logically - only Column 0 of the tile is valid
5. **c_4**: After bcast_sub, full [32, W] centralized data in tiled format. Consumed immediately by square.
6. **c_5**: After square, full [32, W] squared data in tiled format. Consumed immediately by reduce.
7. **c_6**: After REDUCE_ROW (variance), output is [32, 1] logically - only Column 0 is valid
8. **c_16**: After untilize of 1 variance tile, 32 sticks of width 32 (tile width). Only first element of each stick is the actual variance value.

### Binary Op Broadcast Verification (MANDATORY)

**For EVERY binary operation, verify broadcast dimension matches valid regions:**

| Phase | Op | CB_A | CB_A Valid | CB_B | CB_B Valid | Broadcast Required |
|-------|-----|------|------------|------|------------|-------------------|
| Phase 3 (bcast_sub) | sub | c_1 (original) | All (full [32,W]) | c_3 (mean) | **Col0** (REDUCE_ROW output) | **COL** |
| Phase 4 (square) | mul | c_4 (centralized) | All (full [32,W]) | c_4 (same) | All | **NONE** |

**Verification using broadcast selection rules:**

| CB_A Valid | CB_B Valid | Required Broadcast |
|------------|------------|-------------------|
| All | Col0 | **COL** (replicate B's col 0 right across all columns) |
| All | All | **NONE** (element-wise) |

Phase 3 correctly uses `BroadcastDim::COL`. REDUCE_ROW reduces across the width dimension, producing a column-shaped result where only column 0 contains valid data (the row means). To subtract this from the full-width original data, we must broadcast the column values across all width positions.

Phase 4 uses `BroadcastDim::NONE` because both operands are the same CB (self-multiply for square).

**This verification confirms the design is correct.**

### Dataflow Graph

```
DRAM (RM)    Reader       c_0         c_1              c_3           c_4          c_5          c_6         c_16       Writer      DRAM (RM)
+-------+   +------+   +-------+   +--------+      +-------+     +--------+   +--------+   +-------+   +--------+   +------+   +-------+
| sticks|-->| NOC0 |-->|RM data|-->| tiled  |----->|  mean |     |central.|-->| squared|-->|variance|-->| RM out |-->| NOC1 |-->| sticks|
+-------+   +------+   | 32xW  |   | 1 x Wt |      | 1 x 1 |     | 1 x Wt |   | 1 x Wt |   | 1 x 1 |   | 32 x 32|   +------+   +-------+
                       | Valid:|   | Valid: |      | Valid:|     | Valid: |   | Valid: |   | Valid:|   | Valid: |
                       |  All  |   |  All   |      |  Col0 |     |  All   |   |  All   |   |  Col0 |   | Col0** |
                       +-------+   +---+----+      +---+---+     +---+----+   +---+----+   +-------+   +--------+
                           |           |               |             |            |            |
                           |           |               |             |            |            |
                        tilize     +---+---------------+             |            |            |
                       (Phase 1)   |       |           |             |            |            |
                                   |    reduce      bcast_sub      square     reduce        untilize
                                   |   (Phase 2)    (Phase 3)     (Phase 4)  (Phase 5)     (Phase 6)
                                   |  PERSISTENT                  STREAMING  STREAMING
                                   |       |           |             |            |
                                   |       v           |             |            |
                                   |   +---+---+       |             |            |
                                   |   | c_2   |-------+-------------+            |
                                   |   | scaler|                                  |
                                   |   | (1/W) |                                  |
                                   |   +-------+                                  |
                                   |                                              |
                                   +----------------------------------------------+
                                     (c_1 persists only for bcast_sub)
```

**Note**: c_16 "Col0" means the first element of each 32-element stick contains the variance value. The rest is tile padding.

**Data transformations**:
1. **Reader**: Reads 32 sticks from DRAM into c_0 (row-major), generates scaler in c_2 (once)
2. **Phase 1 - Tilize**: Converts c_0 (RM) to c_1 (TILE), preserving all data
3. **Phase 2 - Reduce (Mean)**: Reduces c_1 [1 x Wt] to c_3 [1 x 1], output valid region is Col0. **c_1 NOT popped** (PERSISTENT mode).
4. **Phase 3 - BcastSub**: Subtracts c_3 (broadcast COL) from c_1, output to c_4 [1 x Wt]. Releases c_1 and c_3.
5. **Phase 4 - Square**: Squares c_4 element-wise, output to c_5 [1 x Wt]. Releases c_4.
6. **Phase 5 - Reduce (Variance)**: Reduces c_5 [1 x Wt] to c_6 [1 x 1], output valid region is Col0. STREAMING mode pops c_5.
7. **Phase 6 - Untilize**: Converts c_6 (1 TILE) to c_16 (32 RM sticks of width 32)
8. **Writer**: Writes 32 sticks (width 32, not W!) from c_16 to DRAM

### Persistence Analysis

| CB | Read Count | Last Reader | Can Release After | Persist? |
|----|------------|-------------|-------------------|----------|
| c_0 | 1 | tilize (Phase 1) | tilize completes | No |
| c_1 | 2 | reduce (Phase 2), bcast_sub (Phase 3) | bcast_sub completes | **Yes (Block)** |
| c_2 | 2*Ht | reduce mean (Phase 2), reduce variance (Phase 5) | program end | **Yes (Program)** |
| c_3 | 1 | bcast_sub (Phase 3) | bcast_sub completes | No |
| c_4 | 1 | square (Phase 4) | square completes | No |
| c_5 | 1 | reduce variance (Phase 5) | reduce completes | No |
| c_6 | 1 | untilize (Phase 6) | untilize completes | No |
| c_16 | 1 | writer | writer completes | No |

**Critical**: c_1 has read count of 2 (once by reduce, once by bcast_sub). It MUST persist through reduce (Phase 2) and only be released after bcast_sub (Phase 3) completes.

## Design Summary

| Kernel | Phases | Helpers Used | Raw Calls Needed |
|--------|--------|--------------|------------------|
| Reader | 2 | None (dataflow) | noc_async_read, cb_reserve_back, cb_push_back, generate_reduce_scaler |
| Compute | 6 | tilize(), reduce<SUM, REDUCE_ROW, PERSISTENT>(), sub<COL>(), square(), reduce<SUM, REDUCE_ROW, STREAMING>(), untilize() | None (helpers handle all) |
| Writer | 1 | None (dataflow) | noc_async_write, cb_wait_front, cb_pop_front |

## Helper Library Analysis

### Available Helpers Reviewed
- [x] tilize_helpers.hpp - **yes** (tilize RM to TILE)
- [x] untilize_helpers.hpp - **yes** (untilize TILE to RM)
- [x] reduce_helpers.hpp - **yes** (reduce along width dimension with PERSISTENT and STREAMING modes)
- [x] binary_op_helpers.hpp - **yes** (broadcast subtract with COL, square for A*A)
- [x] dest_helpers.hpp - **yes** (DEST_AUTO_LIMIT for untilize dispatch)

### Helper Functions Applicable to This Operation

| Helper | Signature | Use Case in This Op |
|--------|-----------|---------------------|
| `compute_kernel_lib::tilize()` | `tilize(icb, block_w, ocb, num_blocks)` | Phase 1: Convert c_0 (RM) to c_1 (TILE) |
| `compute_kernel_lib::reduce<>()` | `reduce<SUM, REDUCE_ROW, PERSISTENT>(icb, icb_scaler, ocb, TileShape)` | Phase 2: Reduce c_1 to c_3 (mean) |
| `compute_kernel_lib::sub<>()` | `sub<COL, PolicyA, PolicyB>(icb_a, icb_b, ocb, shape)` | Phase 3: c_1 - c_3 -> c_4 |
| `compute_kernel_lib::square()` | `square<PolicyA>(icb, ocb, shape)` | Phase 4: c_4 * c_4 -> c_5 |
| `compute_kernel_lib::reduce<>()` | `reduce<SUM, REDUCE_ROW, STREAMING>(icb, icb_scaler, ocb, TileShape)` | Phase 5: Reduce c_5 to c_6 (variance) |
| `compute_kernel_lib::untilize<>()` | `untilize<1, c_6, c_16>(num_rows)` | Phase 6: Convert c_6 (TILE) to c_16 (RM) |

## Reader Kernel Design

### Prerequisites
- Uses TensorAccessor for DRAM read addressing
- Generates scaler tile in c_2 with value 1/W (reused for both reduce operations)

### Phase 1: Generate Scaler Tile (once at start)
- **Description**: Creates a tile filled with 1/W value for mean and variance calculations
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel - no compute helpers)
  - **RAW CALLS**: Use `generate_reduce_scaler()` pattern from centralize_w_rm reference
    - `cb_reserve_back(c_2, 1)`
    - Fill tile with packed scaler value (1/W as 2x bfloat16)
    - `cb_push_back(c_2, 1)`
- **CB Flow**: Reserve 1 tile in c_2, fill with scaler, push to make available for compute

### Phase 2: Read Input Sticks (per tile-row)
- **Description**: Reads 32 row-major sticks from DRAM per tile-row
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel)
  - **RAW CALLS**:
    - `cb_reserve_back(c_0, Wt)` - Reserve space for one tile-row worth of data
    - For each of 32 sticks:
      - `noc_async_read(noc_addr, l1_write_addr, input_stick_size)`
      - Increment addresses
    - `noc_async_read_barrier()`
    - `cb_push_back(c_0, Wt)` - Signal data ready for compute
- **CB Flow**: Reserve Wt pages, read 32 sticks, push Wt pages
- **Loop**: Repeat for each tile-row (Ht iterations total)

**Important CB Semantics**:
- c_0 is configured with page semantics that align with tilize helper expectations
- Push Wt pages per tile-row to match tilize helper's wait

## Compute Kernel Design

### Prerequisites
- [x] Requires `compute_kernel_hw_startup()`: **yes** - must initialize before using helpers
- [x] Template parameters for reduce helper (Phase 2 - mean):
  - `PoolType`: **SUM** (mean = sum * scaler where scaler = 1/W)
  - `ReduceDim`: **REDUCE_ROW** (reduces W dimension, outputs column-shaped result)
  - `ReduceInputMode`: **PERSISTENT** (tiles persist for reuse by bcast_sub)
  - `ReduceDataFormatReconfig`: **BOTH** (default)

- [x] Template parameters for reduce helper (Phase 5 - variance):
  - `PoolType`: **SUM** (variance = sum(squared) * scaler where scaler = 1/W)
  - `ReduceDim`: **REDUCE_ROW** (reduces W dimension, outputs column-shaped result)
  - `ReduceInputMode`: **STREAMING** (can pop tiles immediately, not needed after)
  - `ReduceDataFormatReconfig`: **BOTH** (default)

### Phase 1: Tilize
- **Description**: Convert row-major data in c_0 to tiled format in c_1
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::tilize()`
  - **Parameters**: `(c_0, Wt, c_1, 1)` - process one tile-row at a time
    - `icb = c_0` (row-major input)
    - `block_w = Wt` (width in tiles)
    - `ocb = c_1` (tiled output)
    - `num_blocks = 1` (one block per iteration)
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow**: Helper waits Wt on c_0, reserves Wt on c_1, processes, pushes Wt, pops Wt

### Phase 2: Reduce Mean (with PERSISTENT mode)
- **Description**: Reduce tiled data in c_1 along width to c_3 (mean calculation)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputMode::PERSISTENT>()`
  - **Parameters**: `(c_1, c_2, c_3, TileShape::row(Wt))`
    - `icb = c_1` (tiled input)
    - `icb_scaler = c_2` (scaler tile with 1/W)
    - `ocb = c_3` (reduced output - mean)
    - `shape = TileShape::row(Wt)` - single row of Wt tiles
  - **CB Management**: Helper handles internally
  - **CRITICAL**: PERSISTENT mode waits for all Wt tiles upfront but does NOT pop them
- **CB Flow**: Helper waits for Wt tiles from c_1, reduces to 1 tile in c_3, **c_1 tiles persist**

**Why PERSISTENT mode**: After reduce, we need c_1 tiles for the bcast_sub operation. STREAMING mode would pop tiles, destroying them. PERSISTENT mode leaves c_1 tiles in place.

### Phase 3: Broadcast Subtract (Centralize)
- **Description**: Subtract mean (c_3) from original tiled data (c_1), producing centralized data (c_4)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::sub<BroadcastDim::COL, InputAPolicy, InputBPolicy>()`
  - **Parameters**: `(c_1, c_3, c_4, BinaryTileShape::row(Wt))`
    - `icb_a = c_1` (original tiled data, already in CB from tilize, persistent from reduce)
    - `icb_b = c_3` (mean tile, Col0 valid)
    - `ocb = c_4` (centralized output)
    - `shape = BinaryTileShape::row(Wt)` - single row of Wt tiles
  - **CB Management**: Helper handles output CB ops
  - **Input Policies**:
    - **c_1 (Input A)**: `InputPolicy<WaitCallerManaged, PopAtEnd>` - tiles already present (PERSISTENT from Phase 2), pop all Wt at end
    - **c_3 (Input B)**: `InputPolicy<WaitUpfront, PopAtEnd>` - wait for 1 mean tile upfront, pop after all A tiles processed
- **CB Flow**:
  - Uses c_1 tiles in place (already waited via PERSISTENT reduce)
  - Waits for 1 tile from c_3, broadcasts it across all Wt output tiles
  - Reserves/pushes Wt tiles to c_4
  - Pops Wt from c_1 (finally releasing the persistent tiles)
  - Pops 1 from c_3

**Why BroadcastDim::COL**: REDUCE_ROW produces a column-shaped output where only column 0 is valid. To apply this to the full [Ht, Wt] original data, we broadcast the column values across all width positions.

### Phase 4: Square (Element-wise Multiply)
- **Description**: Square the centralized tiles (c_4 * c_4) to produce squared deviations (c_5)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::square<InputAPolicy>()`
  - **Parameters**: `(c_4, c_5, BinaryTileShape::row(Wt))`
    - `icb = c_4` (centralized data - used as both A and B internally)
    - `ocb = c_5` (squared output)
    - `shape = BinaryTileShape::row(Wt)` - single row of Wt tiles
  - **CB Management**: Helper handles output CB ops
  - **Input Policy**:
    - **c_4**: `InputPolicy<WaitUpfront, PopAtEnd>` - wait for all Wt tiles upfront (they were just pushed by Phase 3), pop all after processing
- **CB Flow**:
  - Waits for Wt tiles from c_4
  - Computes c_4[i] * c_4[i] for each tile
  - Reserves/pushes Wt tiles to c_5
  - Pops Wt from c_4

**Note**: The `square()` function in binary_op_helpers.hpp is a convenience wrapper around `binary_op<BinaryOpType::SQUARE>` that automatically uses the same CB for both operands.

### Phase 5: Reduce Variance (with STREAMING mode)
- **Description**: Reduce squared deviations in c_5 along width to c_6 (variance calculation)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputMode::STREAMING>()`
  - **Parameters**: `(c_5, c_2, c_6, TileShape::row(Wt))`
    - `icb = c_5` (squared tiles)
    - `icb_scaler = c_2` (same scaler tile with 1/W - reused!)
    - `ocb = c_6` (reduced output - variance)
    - `shape = TileShape::row(Wt)` - single row of Wt tiles
  - **CB Management**: Helper handles internally
- **CB Flow**: Helper waits/pops tiles from c_5 one at a time (STREAMING), reserves/pushes 1 tile to c_6

**Why STREAMING mode**: After variance reduce, we don't need c_5 tiles anymore. STREAMING mode processes and pops tiles one at a time, which is simpler and allows for lower CB capacity requirements.

**Why reuse c_2**: Both mean and variance are averages (sum divided by W). The scaler 1/W is identical for both operations. Reusing c_2 saves memory and is correct.

### Phase 6: Untilize
- **Description**: Convert variance tile in c_6 back to row-major in c_16
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::untilize<1, c_6, c_16>()`
  - **Parameters**: `(1)` - process 1 tile-row per iteration
    - `tile_width = 1` (output is 1 tile wide - reduced!)
    - `icb_id = c_6` (tiled input - variance)
    - `ocb_id = c_16` (row-major output)
    - `num_rows = 1` (one tile-row per iteration)
  - **CB Management**: Helper handles internally
- **CB Flow**: Helper waits 1 on c_6, reserves 1 on c_16, untilizes, pushes 1, pops 1

**Output Note**: After untilizing 1 variance tile, c_16 contains 32 sticks of width 32 elements each. Only the first element of each stick is meaningful (the actual variance value). The rest is tile padding.

### Compute Kernel Structure (Outer Loop)

```cpp
compute_kernel_hw_startup(c_0, c_2, c_16);  // Initialize hardware

for (block = 0; block < Ht; block++) {
    // Phase 1: Tilize one tile-row (c_0 -> c_1)
    compute_kernel_lib::tilize(c_0, Wt, c_1, 1);

    // Phase 2: Reduce for mean (c_1 -> c_3), c_1 persists
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
                               ReduceInputMode::PERSISTENT>(
        c_1, c_2, c_3, TileShape::row(Wt));

    // Phase 3: Broadcast subtract (c_1 - c_3 -> c_4), releases c_1 and c_3
    using PreloadedPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitCallerManaged, cb_policies::PopAtEnd>;
    using WaitUpfrontPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitUpfront, cb_policies::PopAtEnd>;
    compute_kernel_lib::sub<BroadcastDim::COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(
        c_1, c_3, c_4, BinaryTileShape::row(Wt));

    // Phase 4: Square (c_4 * c_4 -> c_5), releases c_4
    compute_kernel_lib::square<WaitUpfrontPopAtEnd>(
        c_4, c_5, BinaryTileShape::row(Wt));

    // Phase 5: Reduce for variance (c_5 -> c_6), STREAMING pops c_5
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
                               ReduceInputMode::STREAMING>(
        c_5, c_2, c_6, TileShape::row(Wt));

    // Phase 6: Untilize the result (c_6 -> c_16)
    compute_kernel_lib::untilize<1, c_6, c_16>(1);
}
```

## Writer Kernel Design

### Phase 1: Write Output Sticks (per tile-row)
- **Description**: Writes 32 row-major sticks from c_16 to DRAM (reduced width = 32, not W!)
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel)
  - **RAW CALLS**:
    - `cb_wait_front(c_16, 1)` - Wait for 1 untilized tile (contains 32 sticks of width 32)
    - For each of 32 sticks:
      - `noc_async_write(l1_read_addr, noc_addr, output_stick_size)`
      - Increment addresses
    - `noc_async_write_barrier()`
    - `cb_pop_front(c_16, 1)` - Release CB space
- **CB Flow**: Wait 1 tile, write 32 sticks (width 32), pop 1 tile
- **Loop**: Repeat for each tile-row (Ht iterations total)

**Critical Output Difference from centralize_w_rm**:
- centralize_w_rm: Output stick width = W (same as input)
- variance_w_rm: Output stick width = **32** (tile width, padded from logical width 1)

The output tensor has shape `[..., 32]` padded (logical `[..., 1]`), so each output stick is only 32 elements wide.

## CB Synchronization Summary

| CB | Producer | Consumer | Pages per Block | Sync Point |
|----|----------|----------|-----------------|------------|
| c_0 | Reader | Compute (tilize) | Wt | Reader pushes after reading 32 sticks |
| c_1 | Compute (tilize) | Compute (reduce, bcast_sub) | Wt | Tilize pushes Wt, reduce waits Wt (PERSISTENT), bcast_sub pops Wt |
| c_2 | Reader | Compute (reduce x2) | 1 | Generated once, read 2*Ht times (persistent, no pop) |
| c_3 | Compute (reduce) | Compute (bcast_sub) | 1 | Reduce pushes 1, bcast_sub waits 1, pops 1 |
| c_4 | Compute (bcast_sub) | Compute (square) | Wt | BcastSub pushes Wt, square waits Wt, pops Wt |
| c_5 | Compute (square) | Compute (reduce) | Wt | Square pushes Wt, reduce (STREAMING) waits/pops 1 at a time |
| c_6 | Compute (reduce) | Compute (untilize) | 1 | Reduce pushes 1, untilize waits 1, pops 1 |
| c_16 | Compute (untilize) | Writer | 1 | Untilize pushes 1, writer waits 1, pops 1 |

**Critical c_1 Synchronization**:
- Tilize: pushes Wt tiles
- Reduce (PERSISTENT): waits for Wt tiles, processes them, does NOT pop
- BcastSub (PreloadedPopAtEnd): uses Wt tiles via indexed access, pops Wt at end
- Net effect: c_1 tiles persist through reduce for reuse in bcast_sub

**Critical c_2 Synchronization**:
- Reader: pushes 1 (once at start)
- Reduce mean (Phase 2): waits 1 (Ht times, no pop - reduce_helpers waits once per call but doesn't pop scaler)
- Reduce variance (Phase 5): waits 1 (Ht times, no pop)
- Net effect: c_2 scaler persists for entire program

## Helper Encapsulation Acknowledgment

For phases marked "USE HELPER", the following is encapsulated BY THE HELPER:
- CB wait/pop/reserve/push operations (per helper's internal policy)
- DST register management (acquire/commit/wait/release)
- Init/uninit sequences (tilize_init, reduce_init, binary_init, untilize_init, etc.)

**CRITICAL**: The kernel writer MUST NOT add redundant CB or DST operations around helper calls. The helper functions are self-contained.

### Tilize Helper Encapsulates:
- `tilize_init()` / `tilize_uninit()`
- `cb_wait_front(icb, block_w)`
- `cb_reserve_back(ocb, block_w)`
- `tilize_block(icb, block_w, ocb)`
- `cb_push_back(ocb, block_w)`
- `cb_pop_front(icb, block_w)`

### Reduce Helper (PERSISTENT mode) Encapsulates:
- `reduce_init()` / `reduce_uninit()`
- `cb_wait_front(icb_scaler, 1)` (once at start)
- `cb_wait_front(icb, total_tiles)` (all tiles upfront - PERSISTENT mode)
- `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()`
- `reduce_tile()` for each input tile (indexed access)
- `pack_tile()` for output
- `cb_reserve_back()` / `cb_push_back()` for output
- **NO cb_pop_front()** for input (PERSISTENT mode - tiles persist)

### Reduce Helper (STREAMING mode) Encapsulates:
- `reduce_init()` / `reduce_uninit()`
- `cb_wait_front(icb_scaler, 1)` (once at start)
- For each tile: `cb_wait_front(icb, 1)`, process, `cb_pop_front(icb, 1)`
- `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()`
- `reduce_tile()` for each input tile
- `pack_tile()` for output
- `cb_reserve_back()` / `cb_push_back()` for output

### Binary Op Helper (sub with BroadcastDim::COL, custom policies) Encapsulates:
- `binary_init()` (eltwise binary initialization)
- `cb_wait_front()` for inputs (based on policies - WaitCallerManaged skips for A, WaitUpfront for B)
- `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()`
- `binary_exec()` for each tile pair
- `pack_tile()` for output
- `cb_reserve_back()` / `cb_push_back()` for output
- `cb_pop_front()` for inputs (PopAtEnd for both A and B)

### Square Helper (binary_op<SQUARE> internally) Encapsulates:
- `binary_init()` (eltwise binary initialization)
- `cb_wait_front()` for input (based on policy - WaitUpfront)
- `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()`
- `binary_exec()` with same CB for A and B
- `pack_tile()` for output
- `cb_reserve_back()` / `cb_push_back()` for output
- `cb_pop_front()` for input (PopAtEnd)

### Untilize Helper Encapsulates:
- `pack_untilize_init()` / `pack_untilize_uninit()` (or standard variants based on width/type)
- `cb_wait_front(icb, tiles)`
- `cb_reserve_back(ocb, tiles)`
- `pack_untilize_block()` or `untilize_block()`
- `cb_push_back(ocb, tiles)`
- `cb_pop_front(icb, tiles)`

## Implementation Checklist for Kernel Writer

### Reader Kernel
- [ ] Get compile-time args: input_stick_size, packed_scaler_value, Ht, Wt, TensorAccessorArgs
- [ ] Get runtime args: src_addr
- [ ] Generate scaler tile in c_2 using `generate_reduce_scaler()` pattern (value = 1/W)
- [ ] Loop Ht times: reserve c_0 (Wt), read 32 sticks via NOC, push c_0 (Wt)

### Compute Kernel
- [ ] Get compile-time args: Ht, Wt (loop bounds)
- [ ] Call `compute_kernel_hw_startup(c_0, c_2, c_16)`
- [ ] Loop Ht times:
  - [ ] Phase 1: Call `compute_kernel_lib::tilize(c_0, Wt, c_1, 1)`
  - [ ] Phase 2: Call `compute_kernel_lib::reduce<SUM, REDUCE_ROW, PERSISTENT>(c_1, c_2, c_3, TileShape::row(Wt))`
  - [ ] Phase 3: Call `compute_kernel_lib::sub<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(c_1, c_3, c_4, BinaryTileShape::row(Wt))`
  - [ ] Phase 4: Call `compute_kernel_lib::square<WaitUpfrontPopAtEnd>(c_4, c_5, BinaryTileShape::row(Wt))`
  - [ ] Phase 5: Call `compute_kernel_lib::reduce<SUM, REDUCE_ROW, STREAMING>(c_5, c_2, c_6, TileShape::row(Wt))`
  - [ ] Phase 6: Call `compute_kernel_lib::untilize<1, c_6, c_16>(1)`

### Writer Kernel
- [ ] Get compile-time args: output_stick_size (= 32 * elem_size), Ht, TensorAccessorArgs
- [ ] Get runtime args: dst_addr
- [ ] Loop Ht times: wait c_16 (1), write 32 sticks (width 32!) via NOC, pop c_16 (1)

### Verify
- [ ] CB push/pop counts match across kernels:
  - c_0: Reader pushes Wt per block, tilize pops Wt per block
  - c_1: Tilize pushes Wt per block, reduce does NOT pop (PERSISTENT), bcast_sub pops Wt per block
  - c_2: Reader pushes 1 (once), reduce waits 1 (2*Ht times total, no pop - persistent scaler)
  - c_3: Reduce (mean) pushes 1 per block, bcast_sub pops 1 per block
  - c_4: BcastSub pushes Wt per block, square pops Wt per block
  - c_5: Square pushes Wt per block, reduce (STREAMING) pops Wt per block (1 at a time internally)
  - c_6: Reduce (variance) pushes 1 per block, untilize pops 1 per block
  - c_16: Untilize pushes 1 per block, writer pops 1 per block

### Key Differences from centralize_w_rm
- [ ] Output is reduced: c_16 holds 1 tile (not Wt tiles)
- [ ] Writer uses output_stick_size = 32 * elem_size (not W * elem_size)
- [ ] 2 additional phases: square (Phase 4) and variance reduce (Phase 5)
- [ ] 2 additional CBs: c_5 (squared) and c_6 (variance)
- [ ] Phase 5 uses STREAMING mode (not PERSISTENT) since c_5 not reused
