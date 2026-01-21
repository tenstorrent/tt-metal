# Kernel Design: reduce_mean_w_rm

## Spec Validation Issues

No issues found. The spec correctly:
- Defines separate CBs for row-major and tiled formats (c_0 for RM input, c_1 for tiled)
- Allocates dedicated CBs for each intermediate result (c_1, c_2, c_3)
- Uses REDUCE_ROW to reduce along width dimension (correct naming)

## Data Semantics Model (MANDATORY)

### Buffer Content Analysis

| CB | Layout | Logical Shape | Tile Shape | Valid Region | Lifetime |
|----|--------|---------------|------------|--------------|----------|
| c_0 | ROW_MAJOR | [32, W] (32 sticks per block) | N/A | All | Block |
| c_1 | TILE | [32, W] | 1 x Wt | All | Block |
| c_2 | TILE | [1] (scaler) | 1 x 1 | All (filled with 1/W) | Program |
| c_3 | TILE | [32, 1] | 1 x 1 | **Col0** (REDUCE_ROW output) | Block |
| c_16 | ROW_MAJOR | [32, 32] (padded) | N/A | **Col0** (only first column valid logically) | Block |

**Key observations**:
1. **c_0**: Row-major sticks from DRAM, 32 sticks form one tile-row, all elements valid
2. **c_1**: After tilize, full [32, W] data in tiled format, all elements valid
3. **c_2**: Scaler tile with value 1/W for mean calculation
4. **c_3**: After REDUCE_ROW, output is [32, 1] logically - only Column 0 of each tile is valid
5. **c_16**: After untilize, row-major output with width=32 (padded from logical width=1)

### Binary Op Broadcast Verification (MANDATORY)

This operation does not use binary operations (add/sub/mul). The only "multiplication" is embedded in the reduce operation via the scaler CB.

| Phase | Op | CB_A Valid | CB_B Valid | Broadcast Required |
|-------|-----|------------|------------|-------------------|
| N/A | N/A | N/A | N/A | N/A |

The reduce helper handles the scaler multiplication internally - no explicit binary_op helper is needed.

### Dataflow Graph

```
                                    +-------------+
                                    | cb_scaler   |
                                    | c_2 (1/W)   |
                                    | Valid: All  |
                                    +------+------+
                                           |
                                           | (persistent, read by reduce)
                                           v
DRAM (RM)      Reader        c_0          c_1           c_3           c_16        Writer       DRAM (RM)
+-------+     +------+    +-------+    +-------+    +-------+    +-------+     +------+      +-------+
| sticks|---->| NOC0 |--->|RM data|--->| tiled |--->|reduced|--->|RM out |---->| NOC1 |----->| sticks|
+-------+     +------+    | 32x W |    | 1xWt  |    | 1x1   |    | 32x32 |     +------+      +-------+
                          | Valid:|    | Valid:|    | Valid:|    | Valid:|
                          |  All  |    |  All  |    | Col0  |    | Col0  |
                          +-------+    +-------+    +-------+    +-------+
                              |            |            |            |
                              +-----+------+-----+------+-----+------+
                                    |            |            |
                                 tilize       reduce       untilize
                                (compute)   (compute)    (compute)
```

**Data transformations**:
1. **Reader**: Reads 32 sticks from DRAM into c_0 (row-major)
2. **Tilize**: Converts c_0 (RM) to c_1 (TILE), preserving all data
3. **Reduce**: Reduces c_1 [1 x Wt] to c_3 [1 x 1], output valid region is Col0
4. **Untilize**: Converts c_3 (TILE) to c_16 (RM), output is 32 elements wide (padded)
5. **Writer**: Writes 32 sticks from c_16 to DRAM

### Persistence Analysis

| CB | Read Count | Last Reader | Can Release After | Persist? |
|----|------------|-------------|-------------------|----------|
| c_0 | 1 | tilize | tilize completes | No |
| c_1 | 1 | reduce | reduce completes | No |
| c_2 | Ht | reduce (each row) | program end | **Yes (Program)** |
| c_3 | 1 | untilize | untilize completes | No |
| c_16 | 1 | writer | writer completes | No |

**Note**: c_2 (scaler) is the only persistent CB - it persists for the entire program as it's reused for each tile-row reduction.

## Design Summary

| Kernel | Phases | Helpers Used | Raw Calls Needed |
|--------|--------|--------------|------------------|
| Reader | 2 | None (dataflow) | noc_async_read, cb_reserve_back, cb_push_back, generate_reduce_scaler |
| Compute | 3 | tilize(), reduce<SUM, REDUCE_ROW>(), untilize<1>() | None (helpers handle all) |
| Writer | 1 | None (dataflow) | noc_async_write, cb_wait_front, cb_pop_front |

## Helper Library Analysis

### Available Helpers Reviewed
- [x] tilize_helpers.hpp - **yes** (tilize RM to TILE)
- [x] untilize_helpers.hpp - **yes** (untilize TILE to RM)
- [x] reduce_helpers.hpp - **yes** (reduce along width dimension)
- [x] binary_op_helpers.hpp - **no** (no binary ops needed)
- [x] dest_helpers.hpp - **yes** (DEST_AUTO_LIMIT for untilize dispatch)

### Helper Functions Applicable to This Operation

| Helper | Signature | Use Case in This Op |
|--------|-----------|---------------------|
| `compute_kernel_lib::tilize()` | `tilize(icb, block_w, ocb, num_blocks, subblock_h=1)` | Phase 1: Convert c_0 (RM) to c_1 (TILE) |
| `compute_kernel_lib::reduce<>()` | `reduce<PoolType, ReduceDim, InputMode, Reconfig>(icb, icb_scaler, ocb, TileShape)` | Phase 2: Reduce c_1 to c_3 using SUM with scaler |
| `compute_kernel_lib::untilize<>()` | `untilize<tile_width, icb_id, ocb_id>(num_rows)` | Phase 3: Convert c_3 (TILE) to c_16 (RM) |

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
- c_0 is configured with `page_size = input_stick_size_aligned`
- `num_pages = 2 * TILE_HEIGHT = 64` (double-buffered)
- Push/pop count is in terms of CB pages (each page = 1 stick)
- For tilize sync: Push 32 sticks, compute waits for Wt tiles worth of data

**Critical Sync Point**: The reader pushes 32 sticks, but the tilize helper expects Wt tiles. The CB is sized so that 32 sticks occupy Wt tiles worth of CB space (since 32 sticks at width W = Wt tiles of 32x32).

## Compute Kernel Design

### Prerequisites
- [x] Requires `compute_kernel_hw_startup()`: **yes** - must initialize before using helpers
- [ ] Template parameters for reduce helper:
  - `PoolType`: **SUM** (mean = sum * scaler where scaler = 1/W)
  - `ReduceDim`: **REDUCE_ROW** (reduces W dimension)
  - `ReduceInputMode`: **STREAMING** (default, one-at-a-time for safety)
  - `ReduceDataFormatReconfig`: **BOTH** (default, since tilize changes formats)

### Phase 1: Tilize
- **Description**: Convert row-major data in c_0 to tiled format in c_1
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::tilize()`
  - **Parameters**: `(c_0, Wt, c_1, 1)` - process one tile-row at a time
    - `icb = c_0` (row-major input)
    - `block_w = Wt` (width in tiles)
    - `ocb = c_1` (tiled output)
    - `num_blocks = 1` (one block per iteration, called Ht times in outer loop)
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow**: Helper waits Wt on c_0, reserves Wt on c_1, processes, pushes Wt, pops Wt

### Phase 2: Reduce
- **Description**: Reduce tiled data in c_1 along width to c_3
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputMode::STREAMING>()`
  - **Parameters**: `(c_1, c_2, c_3, TileShape::row(Wt))`
    - `icb = c_1` (tiled input)
    - `icb_scaler = c_2` (scaler tile with 1/W)
    - `ocb = c_3` (reduced output)
    - `shape = TileShape::row(Wt)` - single row of Wt tiles
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
  - **Note**: Using SUM not AVG because the scaler already contains 1/W
- **CB Flow**: Helper waits for each tile from c_1, reduces, produces 1 tile to c_3

**Why SUM not AVG**: The reduce helper's AVG mode would expect raw values and apply its own averaging. Since we pre-compute 1/W in the scaler, we use SUM which just multiplies by the scaler. This is the standard pattern from reduce_w reference.

### Phase 3: Untilize
- **Description**: Convert reduced tile in c_3 back to row-major in c_16
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::untilize<1, c_3, c_16>()`
  - **Parameters**: `(1)` - process 1 row (one reduced tile)
    - `tile_width = 1` (output is 1 tile wide after reduction)
    - `icb_id = c_3` (tiled input)
    - `ocb_id = c_16` (row-major output)
    - `num_rows = 1` (one tile-row per iteration)
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow**: Helper waits 1 on c_3, reserves 1 on c_16, untilizes, pushes 1, pops 1

### Compute Kernel Structure (Outer Loop)
```
compute_kernel_hw_startup(c_0, c_2, c_16);  // Initialize hardware

for (block = 0; block < Ht; block++) {
    // Phase 1: Tilize one tile-row
    compute_kernel_lib::tilize(c_0, Wt, c_1, 1);

    // Phase 2: Reduce the tile-row
    compute_kernel_lib::reduce<SUM, REDUCE_ROW>(c_1, c_2, c_3, TileShape::row(Wt));

    // Phase 3: Untilize the reduced result
    compute_kernel_lib::untilize<1, c_3, c_16>(1);
}
```

## Writer Kernel Design

### Phase 1: Write Output Sticks (per tile-row)
- **Description**: Writes 32 row-major sticks from c_16 to DRAM
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernel)
  - **RAW CALLS**:
    - `cb_wait_front(c_16, 1)` - Wait for untilized data
    - For each of 32 sticks:
      - `noc_async_write(l1_read_addr, noc_addr, output_stick_size)`
      - Increment addresses
    - `noc_async_write_barrier()`
    - `cb_pop_front(c_16, 1)` - Release CB space
- **CB Flow**: Wait 1 tile, write 32 sticks (extracting from tile), pop 1 tile
- **Loop**: Repeat for each tile-row (Ht iterations total)

**Output Stick Details**:
- Each output stick is 32 elements (padded from logical 1 element)
- Only the first element of each stick contains the mean value
- Remaining 31 elements are padding (typically zeros from tile storage)

## CB Synchronization Summary

| CB | Producer | Consumer | Pages per Block | Sync Point |
|----|----------|----------|-----------------|------------|
| c_0 | Reader | Compute (tilize) | Wt tiles worth (32 sticks) | Reader pushes after reading 32 sticks |
| c_1 | Compute (tilize) | Compute (reduce) | Wt tiles | Tilize pushes Wt, reduce waits Wt |
| c_2 | Reader | Compute (reduce) | 1 tile | Generated once, read Ht times (persistent) |
| c_3 | Compute (reduce) | Compute (untilize) | 1 tile | Reduce pushes 1, untilize waits 1 |
| c_16 | Compute (untilize) | Writer | 1 tile | Untilize pushes 1, writer waits 1 |

**Critical Note on c_0 sync**:
- c_0 is configured with page_size = stick_size (row-major)
- Reader pushes 32 sticks worth of data per iteration
- Tilize helper expects Wt tiles worth of input
- The memory equivalence: 32 sticks * W bytes = Wt tiles * tile_bytes
- This works because tilize_block reads row-major data in tile-height chunks

## Helper Encapsulation Acknowledgment

For phases marked "USE HELPER", the following is encapsulated BY THE HELPER:
- CB wait/pop/reserve/push operations
- DST register management (acquire/commit/wait/release)
- Init/uninit sequences (tilize_init, reduce_init, untilize_init, etc.)

**CRITICAL**: The kernel writer MUST NOT add redundant CB or DST operations around helper calls. The helper functions are self-contained.

### Tilize Helper Encapsulates:
- `tilize_init()` / `tilize_uninit()`
- `cb_wait_front(icb, block_w)`
- `cb_reserve_back(ocb, block_w)`
- `tilize_block(icb, block_w, ocb)`
- `cb_push_back(ocb, block_w)`
- `cb_pop_front(icb, block_w)`

### Reduce Helper Encapsulates:
- `reduce_init()` / `reduce_uninit()`
- `cb_wait_front(icb_scaler, 1)` (once at start)
- `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()`
- `reduce_tile()` for each input tile
- `pack_tile()` for output
- `cb_wait_front()` / `cb_pop_front()` for input tiles
- `cb_reserve_back()` / `cb_push_back()` for output

### Untilize Helper Encapsulates:
- `pack_untilize_init()` / `pack_untilize_uninit()` (or standard untilize variants)
- `cb_wait_front(icb, tiles)`
- `cb_reserve_back(ocb, tiles)`
- `pack_untilize_block()` or `untilize_block()`
- `cb_push_back(ocb, tiles)`
- `cb_pop_front(icb, tiles)`

## Implementation Checklist for Kernel Writer

### Reader Kernel
- [ ] Get compile-time args: stick_size, packed_scaler_value, TensorAccessorArgs
- [ ] Get runtime args: src_addr, num_sticks, start_stick_id (Ht, Wt for loop control)
- [ ] Generate scaler tile in c_2 using `generate_reduce_scaler()` pattern
- [ ] Loop Ht times: reserve c_0, read 32 sticks via NOC, push c_0

### Compute Kernel
- [ ] Get compile-time args: Ht, Wt (loop bounds)
- [ ] Call `compute_kernel_hw_startup(c_0, c_2, c_16)`
- [ ] Loop Ht times:
  - [ ] Call `compute_kernel_lib::tilize(c_0, Wt, c_1, 1)`
  - [ ] Call `compute_kernel_lib::reduce<SUM, REDUCE_ROW>(c_1, c_2, c_3, TileShape::row(Wt))`
  - [ ] Call `compute_kernel_lib::untilize<1, c_3, c_16>(1)`

### Writer Kernel
- [ ] Get compile-time args: output_stick_size, TensorAccessorArgs
- [ ] Get runtime args: dst_addr, num_sticks, start_stick_id
- [ ] Loop Ht times: wait c_16, write 32 sticks via NOC, pop c_16

### Verify
- [ ] CB push/pop counts match across kernels:
  - c_0: Reader pushes Wt per block, tilize pops Wt per block
  - c_1: Tilize pushes Wt per block, reduce pops Wt per block
  - c_2: Reader pushes 1 (once), reduce waits 1 (once per block, no pop - persistent)
  - c_3: Reduce pushes 1 per block, untilize pops 1 per block
  - c_16: Untilize pushes 1 per block, writer pops 1 per block
