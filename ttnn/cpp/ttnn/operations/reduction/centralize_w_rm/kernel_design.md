# Kernel Design: centralize_w_rm

## Spec Validation Issues

### Issue 1: Reduce Helper CB Persistence Behavior

- **Spec says**: CB_1 must NOT be popped after reduce (needed for step 3)
- **Problem**: The reduce helper with STREAMING mode (default) pops tiles after processing them. This would consume CB_1 tiles before bcast_sub can use them.
- **Resolution**: Use `ReduceInputMode::PERSISTENT` for the reduce helper call. This mode waits for all tiles upfront and does NOT pop them, leaving CB_1 tiles available for the subsequent bcast_sub operation.

### Issue 2: Binary Op Input Policy for CB_1 Persistence

- **Spec says**: Uses CB_1 (original tiled, Wt tiles) for bcast_sub
- **Problem**: If we use the default Streaming policy for binary_op input A, it would wait and pop tiles one at a time. CB_1 tiles are already present (from tilize) and persistent (from reduce).
- **Resolution**: Use `cb_policies::Preloaded` for input A in sub() call. This tells the binary_op helper that tiles are already in the CB and it should use indexed access without additional wait/pop operations.

## Data Semantics Model (MANDATORY)

### Buffer Content Analysis

| CB | Layout | Logical Shape | Tile Shape | Valid Region | Lifetime |
|----|--------|---------------|------------|--------------|----------|
| c_0 | ROW_MAJOR | [32, W] (32 sticks per block) | N/A | All | Block |
| c_1 | TILE | [32, W] | 1 x Wt | All | Block (persistent through reduce) |
| c_2 | TILE | [1] (scaler) | 1 x 1 | All (filled with 1/W) | Program |
| c_3 | TILE | [32, 1] (mean per row) | 1 x 1 | **Col0** (REDUCE_ROW output) | Block |
| c_4 | TILE | [32, W] (centralized) | 1 x Wt | All | Block |
| c_16 | ROW_MAJOR | [32, W] (output sticks) | N/A | All | Block |

**Key observations**:
1. **c_0**: Row-major sticks from DRAM, 32 sticks form one tile-row, all elements valid
2. **c_1**: After tilize, full [32, W] data in tiled format, all elements valid. **PERSISTS** through reduce phase for bcast_sub.
3. **c_2**: Scaler tile with value 1/W for mean calculation (persistent for entire program)
4. **c_3**: After REDUCE_ROW, output is [32, 1] logically - only Column 0 of the tile is valid (contains row means)
5. **c_4**: After bcast_sub, full [32, W] centralized data in tiled format, all elements valid
6. **c_16**: After untilize, row-major output with full width W

### Binary Op Broadcast Verification (MANDATORY)

**For the broadcast subtract operation:**

| Phase | Op | CB_A | CB_A Valid | CB_B | CB_B Valid | Broadcast Required |
|-------|-----|------|------------|------|------------|-------------------|
| bcast_sub | sub | c_1 (original) | All (full [32,W]) | c_3 (mean) | **Col0** (REDUCE_ROW output) | **COL** |

**Verification using broadcast selection rules:**

| CB_A Valid | CB_B Valid | Required Broadcast |
|------------|------------|-------------------|
| All | Col0 | **COL** (replicate B's col 0 right across all columns) |

The spec correctly specifies `BroadcastDim::COL`. REDUCE_ROW reduces across the width dimension, producing a column-shaped result where only column 0 contains valid data (the row means). To subtract this from the full-width original data, we must broadcast the column values across all width positions.

**This is correct and verified.**

### Dataflow Graph

```
DRAM (RM)    Reader       c_0         c_1              c_3           c_4          c_16       Writer      DRAM (RM)
+-------+   +------+   +-------+   +--------+      +-------+     +--------+   +--------+   +------+   +-------+
| sticks|-->| NOC0 |-->|RM data|-->| tiled  |----->|  mean |     |central.|-->| RM out |-->| NOC1 |-->| sticks|
+-------+   +------+   | 32xW  |   | 1 x Wt |      | 1 x 1 |     | 1 x Wt |   | 32 x W |   +------+   +-------+
                       | Valid:|   | Valid: |      | Valid:|     | Valid: |   | Valid: |
                       |  All  |   |  All   |      |  Col0 |     |  All   |   |  All   |
                       +-------+   +---+----+      +---+---+     +---+----+   +--------+
                           |           |               |             ^
                           |           |               |             |
                        tilize     +---+---------------+-------------+
                       (compute)   |       |           |
                                   |    reduce      bcast_sub      untilize
                                   |   (compute)    (compute)     (compute)
                                   |       |           |             |
                                   |       v           |             |
                                   |   +---+---+       |             |
                                   |   | c_2   |       |             |
                                   |   | scaler|-------+             |
                                   |   | (1/W) |                     |
                                   |   +-------+                     |
                                   |                                 |
                                   +---------------------------------+
                                     (c_1 persists for bcast_sub)
```

**Data transformations**:
1. **Reader**: Reads 32 sticks from DRAM into c_0 (row-major), generates scaler in c_2
2. **Tilize**: Converts c_0 (RM) to c_1 (TILE), preserving all data
3. **Reduce**: Reduces c_1 [1 x Wt] to c_3 [1 x 1], output valid region is Col0. **c_1 NOT popped** (PERSISTENT mode).
4. **BcastSub**: Subtracts c_3 (broadcast COL) from c_1, output to c_4 [1 x Wt]
5. **Untilize**: Converts c_4 (TILE) to c_16 (RM)
6. **Writer**: Writes 32 sticks from c_16 to DRAM

### Persistence Analysis

| CB | Read Count | Last Reader | Can Release After | Persist? |
|----|------------|-------------|-------------------|----------|
| c_0 | 1 | tilize | tilize completes | No |
| c_1 | 2 | reduce (read), bcast_sub (read) | bcast_sub completes | **Yes (Block)** |
| c_2 | Ht | reduce (each row) | program end | **Yes (Program)** |
| c_3 | 1 | bcast_sub | bcast_sub completes | No |
| c_4 | 1 | untilize | untilize completes | No |
| c_16 | 1 | writer | writer completes | No |

**Critical**: c_1 has read count of 2 (once by reduce, once by bcast_sub). It MUST persist through reduce and only be released after bcast_sub completes.

## Design Summary

| Kernel | Phases | Helpers Used | Raw Calls Needed |
|--------|--------|--------------|------------------|
| Reader | 2 | None (dataflow) | noc_async_read, cb_reserve_back, cb_push_back, generate_reduce_scaler |
| Compute | 4 | tilize(), reduce<SUM, REDUCE_ROW, PERSISTENT>(), sub<COL>(), untilize<Wt>() | None (helpers handle all) |
| Writer | 1 | None (dataflow) | noc_async_write, cb_wait_front, cb_pop_front |

## Helper Library Analysis

### Available Helpers Reviewed
- [x] tilize_helpers.hpp - **yes** (tilize RM to TILE)
- [x] untilize_helpers.hpp - **yes** (untilize TILE to RM)
- [x] reduce_helpers.hpp - **yes** (reduce along width dimension with PERSISTENT mode)
- [x] binary_op_helpers.hpp - **yes** (broadcast subtract)
- [x] dest_helpers.hpp - **yes** (DEST_AUTO_LIMIT for untilize dispatch)

### Helper Functions Applicable to This Operation

| Helper | Signature | Use Case in This Op |
|--------|-----------|---------------------|
| `compute_kernel_lib::tilize()` | `tilize(icb, block_w, ocb, num_blocks)` | Phase 1: Convert c_0 (RM) to c_1 (TILE) |
| `compute_kernel_lib::reduce<>()` | `reduce<SUM, REDUCE_ROW, PERSISTENT>(icb, icb_scaler, ocb, TileShape)` | Phase 2: Reduce c_1 to c_3 (mean) |
| `compute_kernel_lib::sub<>()` | `sub<COL, Preloaded, Streaming>(icb_a, icb_b, ocb, shape)` | Phase 3: c_1 - c_3 -> c_4 |
| `compute_kernel_lib::untilize<>()` | `untilize<Wt, c_4, c_16>(num_rows)` | Phase 4: Convert c_4 (TILE) to c_16 (RM) |

## Reader Kernel Design

### Prerequisites
- Uses TensorAccessor for DRAM read addressing
- Generates scaler tile in c_2 with value 1/W

### Phase 1: Generate Scaler Tile (once at start)
- **Description**: Creates a tile filled with 1/W value for mean calculation
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel - no compute helpers)
  - **RAW CALLS**: Use `generate_reduce_scaler()` pattern from reduce_w reference
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
      - `noc_async_read(noc_addr, l1_write_addr, stick_size)`
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
- [x] Template parameters for reduce helper:
  - `PoolType`: **SUM** (mean = sum * scaler where scaler = 1/W)
  - `ReduceDim`: **REDUCE_ROW** (reduces W dimension, outputs column-shaped result)
  - `ReduceInputMode`: **PERSISTENT** (tiles persist for reuse by bcast_sub)
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

### Phase 2: Reduce (with PERSISTENT mode)
- **Description**: Reduce tiled data in c_1 along width to c_3 (mean calculation)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputMode::PERSISTENT>()`
  - **Parameters**: `(c_1, c_2, c_3, TileShape::row(Wt))`
    - `icb = c_1` (tiled input)
    - `icb_scaler = c_2` (scaler tile with 1/W)
    - `ocb = c_3` (reduced output)
    - `shape = TileShape::row(Wt)` - single row of Wt tiles
  - **CB Management**: Helper handles internally
  - **CRITICAL**: PERSISTENT mode waits for all Wt tiles upfront but does NOT pop them
- **CB Flow**: Helper waits for Wt tiles from c_1, reduces to 1 tile in c_3, **c_1 tiles persist**

**Why PERSISTENT mode**: After reduce, we need c_1 tiles for the bcast_sub operation. STREAMING mode would pop tiles, destroying them. PERSISTENT mode leaves c_1 tiles in place.

### Phase 3: Broadcast Subtract
- **Description**: Subtract mean (c_3) from original tiled data (c_1), producing centralized data (c_4)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::sub<BroadcastDim::COL, cb_policies::Preloaded, cb_policies::Streaming>()`
  - **Parameters**: `(c_1, c_3, c_4, BinaryTileShape::row(Wt))`
    - `icb_a = c_1` (original tiled data, already in CB from tilize, persistent from reduce)
    - `icb_b = c_3` (mean tile, Col0 valid)
    - `ocb = c_4` (centralized output)
    - `shape = BinaryTileShape::row(Wt)` - single row of Wt tiles
  - **CB Management**: Helper handles output CB ops, uses Preloaded policy for c_1
  - **Input Policies**:
    - **c_1 (Input A)**: `Preloaded` - tiles already present, use indexed access, pop at end
    - **c_3 (Input B)**: `Streaming` - default behavior for COL broadcast (wait upfront, pop after)
- **CB Flow**:
  - Uses c_1 tiles in place (Preloaded policy with indexed access)
  - Waits for 1 tile from c_3, broadcasts it across all Wt output tiles
  - Reserves/pushes Wt tiles to c_4
  - Pops Wt from c_1 (finally releasing the persistent tiles)
  - Pops 1 from c_3

**Why BroadcastDim::COL**: REDUCE_ROW produces a column-shaped output [Ht, 1] where only column 0 is valid. To apply this to the full [Ht, Wt] original data, we broadcast the column values across all width positions.

### Phase 4: Untilize
- **Description**: Convert centralized tiles in c_4 back to row-major in c_16
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::untilize<Wt, c_4, c_16>()`
  - **Parameters**: `(1)` - process 1 tile-row per iteration
    - `tile_width = Wt` (output is Wt tiles wide)
    - `icb_id = c_4` (tiled input)
    - `ocb_id = c_16` (row-major output)
    - `num_rows = 1` (one tile-row per iteration)
  - **CB Management**: Helper handles internally
- **CB Flow**: Helper waits Wt on c_4, reserves Wt on c_16, untilizes, pushes Wt, pops Wt

### Compute Kernel Structure (Outer Loop)
```
compute_kernel_hw_startup(c_0, c_2, c_16);  // Initialize hardware

for (block = 0; block < Ht; block++) {
    // Phase 1: Tilize one tile-row (c_0 -> c_1)
    compute_kernel_lib::tilize(c_0, Wt, c_1, 1);

    // Phase 2: Reduce the tile-row (c_1 -> c_3), c_1 persists
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
                               ReduceInputMode::PERSISTENT>(
        c_1, c_2, c_3, TileShape::row(Wt));

    // Phase 3: Broadcast subtract (c_1 - c_3 -> c_4), releases c_1 and c_3
    compute_kernel_lib::sub<BroadcastDim::COL,
                           cb_policies::Preloaded, cb_policies::Streaming>(
        c_1, c_3, c_4, BinaryTileShape::row(Wt));

    // Phase 4: Untilize the result (c_4 -> c_16)
    compute_kernel_lib::untilize<Wt, c_4, c_16>(1);
}
```

## Writer Kernel Design

### Phase 1: Write Output Sticks (per tile-row)
- **Description**: Writes 32 row-major sticks from c_16 to DRAM
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel)
  - **RAW CALLS**:
    - `cb_wait_front(c_16, Wt)` - Wait for untilized data
    - For each of 32 sticks:
      - `noc_async_write(l1_read_addr, noc_addr, output_stick_size)`
      - Increment addresses
    - `noc_async_write_barrier()`
    - `cb_pop_front(c_16, Wt)` - Release CB space
- **CB Flow**: Wait Wt tiles, write 32 sticks (extracting from tiles), pop Wt tiles
- **Loop**: Repeat for each tile-row (Ht iterations total)

**Output Stick Details**:
- Each output stick has width W elements (same as input)
- All elements contain valid centralized data
- Output is full width (not reduced like reduce_mean_w_rm)

## CB Synchronization Summary

| CB | Producer | Consumer | Pages per Block | Sync Point |
|----|----------|----------|-----------------|------------|
| c_0 | Reader | Compute (tilize) | Wt | Reader pushes after reading 32 sticks |
| c_1 | Compute (tilize) | Compute (reduce, bcast_sub) | Wt | Tilize pushes Wt, reduce waits Wt (PERSISTENT), bcast_sub uses indexed, pops Wt |
| c_2 | Reader | Compute (reduce) | 1 | Generated once, read Ht times (persistent, no pop) |
| c_3 | Compute (reduce) | Compute (bcast_sub) | 1 | Reduce pushes 1, bcast_sub waits 1, pops 1 |
| c_4 | Compute (bcast_sub) | Compute (untilize) | Wt | BcastSub pushes Wt, untilize waits Wt, pops Wt |
| c_16 | Compute (untilize) | Writer | Wt | Untilize pushes Wt, writer waits Wt, pops Wt |

**Critical c_1 Synchronization**:
- Tilize: pushes Wt tiles
- Reduce (PERSISTENT): waits for Wt tiles, processes them, does NOT pop
- BcastSub (Preloaded): uses Wt tiles via indexed access, pops Wt at end
- Net effect: c_1 tiles persist through reduce for reuse in bcast_sub

## Helper Encapsulation Acknowledgment

For phases marked "USE HELPER", the following is encapsulated BY THE HELPER:
- CB wait/pop/reserve/push operations
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

### Binary Op Helper (sub with BroadcastDim::COL) Encapsulates:
- `binary_init()` (eltwise binary initialization)
- `cb_wait_front()` for inputs (based on policies)
- `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()`
- `binary_exec()` for each tile pair
- `pack_tile()` for output
- `cb_reserve_back()` / `cb_push_back()` for output
- `cb_pop_front()` for inputs (based on policies)

### Untilize Helper Encapsulates:
- `pack_untilize_init()` / `pack_untilize_uninit()` (or standard variants)
- `cb_wait_front(icb, tiles)`
- `cb_reserve_back(ocb, tiles)`
- `pack_untilize_block()` or `untilize_block()`
- `cb_push_back(ocb, tiles)`
- `cb_pop_front(icb, tiles)`

## Implementation Checklist for Kernel Writer

### Reader Kernel
- [ ] Get compile-time args: stick_size, packed_scaler_value, Ht, Wt, TensorAccessorArgs
- [ ] Get runtime args: src_addr
- [ ] Generate scaler tile in c_2 using `generate_reduce_scaler()` pattern
- [ ] Loop Ht times: reserve c_0 (Wt), read 32 sticks via NOC, push c_0 (Wt)

### Compute Kernel
- [ ] Get compile-time args: Ht, Wt (loop bounds)
- [ ] Call `compute_kernel_hw_startup(c_0, c_2, c_16)`
- [ ] Loop Ht times:
  - [ ] Phase 1: Call `compute_kernel_lib::tilize(c_0, Wt, c_1, 1)`
  - [ ] Phase 2: Call `compute_kernel_lib::reduce<SUM, REDUCE_ROW, PERSISTENT>(c_1, c_2, c_3, TileShape::row(Wt))`
  - [ ] Phase 3: Call `compute_kernel_lib::sub<COL, Preloaded, Streaming>(c_1, c_3, c_4, BinaryTileShape::row(Wt))`
  - [ ] Phase 4: Call `compute_kernel_lib::untilize<Wt, c_4, c_16>(1)`

### Writer Kernel
- [ ] Get compile-time args: output_stick_size, Ht, TensorAccessorArgs
- [ ] Get runtime args: dst_addr
- [ ] Loop Ht times: wait c_16 (Wt), write 32 sticks via NOC, pop c_16 (Wt)

### Verify
- [ ] CB push/pop counts match across kernels:
  - c_0: Reader pushes Wt per block, tilize pops Wt per block
  - c_1: Tilize pushes Wt per block, reduce does NOT pop (PERSISTENT), bcast_sub pops Wt per block
  - c_2: Reader pushes 1 (once), reduce waits 1 (once per block, no pop - persistent)
  - c_3: Reduce pushes 1 per block, bcast_sub pops 1 per block
  - c_4: BcastSub pushes Wt per block, untilize pops Wt per block
  - c_16: Untilize pushes Wt per block, writer pops Wt per block
