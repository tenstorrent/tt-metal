# Kernel Design: row_mean_sub_square_reduce

## Design Summary

| Kernel | Phases | Helpers Used | Raw Calls Needed |
|--------|--------|--------------|------------------|
| Reader | 2 | None (dataflow) | NOC reads, generate_reduce_scaler, CB push |
| Compute | 5 | tilize(), reduce() x2, untilize() | sub_tiles_bcast_scalar, mul_tiles |
| Writer | 1 | None (dataflow) | NOC writes, CB pop |

## Helper Library Analysis

### Available Helpers Reviewed
- [x] tilize_helpers.hpp - **relevant: yes** (Phase 1: row-major to tiled conversion)
- [x] untilize_helpers.hpp - **relevant: yes** (Phase 5: tiled to row-major conversion)
- [x] reduce_helpers.hpp - **relevant: yes** (Phase 2 and 4: REDUCE_ROW for mean computations)
- [x] dest_helpers.hpp - **relevant: yes** (DEST register limits for untilize path selection)

### Helper Functions Applicable to This Operation

| Helper | Signature | Use Case in This Op |
|--------|-----------|---------------------|
| `compute_kernel_lib::tilize()` | `tilize(icb, block_w, ocb, num_blocks, subblock_h, ...)` | Phase 1: Convert row-major input sticks to tiles |
| `compute_kernel_lib::reduce<SUM, REDUCE_ROW>()` | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(icb, icb_scaler, ocb, TileShape::row(Wt))` | Phase 2: Compute mean along W; Phase 4: Compute variance |
| `compute_kernel_lib::untilize<1, icb, ocb>()` | `untilize<1, cb_in, cb_out>(num_rows)` | Phase 5: Convert variance tile back to row-major |

## Reader Kernel Design

### Phase 1: Generate Scaler (once at start)
- **Description**: Generate the 1/W scaler tile for mean computation
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - `generate_reduce_scaler(cb_scaler, packed_scaler_value)` - fills scaler CB with 1/W value
    - Scaler value = 1/W packed as two bfloat16 in uint32
- **CB Flow**:
  - CB c_2 (scaler): reserve 1 tile, fill with scaler, push 1 tile
  - Persists for entire program duration

### Phase 2: Read Row-Major Sticks (per tile-row)
- **Description**: Read 32 sticks (one tile-row worth) from DRAM, stage for tilize
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - `TensorAccessor` for DRAM addressing
    - `noc_async_read()` for each stick
    - `noc_async_read_barrier()` after 32 sticks
    - `cb_reserve_back(cb_rm_in, Wt)` before reading
    - `cb_push_back(cb_rm_in, Wt)` after 32 sticks read
- **CB Flow**:
  - Reserve Wt pages in c_0 (each page = tile_size bytes)
  - Read 32 sticks into contiguous memory (32 * row_width_bytes = Wt * tile_size bytes)
  - Push Wt pages to signal compute

### Reader Loop Structure
```
// Once at start
generate_reduce_scaler(cb_scaler, packed_scaler_value)

// Per tile-row assigned to this core
for each tile_row:
    cb_reserve_back(cb_rm_in, Wt)
    l1_addr = get_write_ptr(cb_rm_in)
    for j in 0..31:
        stick_id = tile_row * 32 + j
        noc_addr = get_noc_addr(stick_id, tensor_accessor)
        noc_async_read(noc_addr, l1_addr, row_width_bytes)
        l1_addr += row_width_bytes
    noc_async_read_barrier()
    cb_push_back(cb_rm_in, Wt)
```

## Compute Kernel Design

### Prerequisites
- [x] Requires `compute_kernel_hw_startup()`: **yes** - must call before any helper
- [ ] Template parameters for reduce helper:
  - `PoolType`: **SUM** (mean = sum * scaler)
  - `ReduceDim`: **REDUCE_ROW** (reduce along W dimension)
  - `ReduceInputMode`: **STREAMING** (default, one-at-a-time tile processing)
  - `ReduceDataFormatReconfig`: **BOTH** (default, reconfig input and output formats)

**Note**: `REDUCE_OP` and `REDUCE_DIM` macros are **deprecated**. Always specify template parameters explicitly.

### Phase 1: Tilize (cb_rm_in -> cb_tilized)
- **Description**: Convert 32 row-major sticks to Wt tiles
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::tilize()`
  - **Parameters**: `(cb_rm_in, Wt, cb_tilized, 1)` - block_w=Wt, num_blocks=1 per tile-row
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow** (handled by helper):
  - Waits for Wt pages from cb_rm_in (c_0)
  - Reserves Wt pages in cb_tilized (c_1)
  - Tilizes data
  - Pushes Wt pages to cb_tilized, pops Wt from cb_rm_in

### Phase 2: Reduce Mean (cb_tilized + cb_scaler -> cb_mean)
- **Description**: Reduce Wt tiles along W to produce mean tile
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>()`
  - **Parameters**: `(cb_tilized, cb_scaler, cb_mean, TileShape::row(Wt))`
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow** (handled by helper):
  - Waits for scaler in cb_scaler (c_2) - once, persists
  - Processes Wt tiles from cb_tilized streaming (wait 1, reduce, pop 1)
  - Reserves/pushes 1 tile to cb_mean (c_3)

### Phase 3: Subtract Mean + Square (cb_tilized_reread - cb_mean -> cb_intermediate)
- **Description**: For each input tile: subtract mean (broadcast), square the result
- **Implementation Approach**:
  - **USE HELPER**: No
  - **Reason**: No combined sub_bcast_scalar + square helper exists in kernel_lib
  - **RAW CALLS**:
    - `sub_tiles_bcast_scalar_init_short()` - init subtraction
    - `cb_wait_front(cb_mean, 1)` - wait for mean tile (once per tile-row)
    - Loop over Wt tiles:
      - `cb_wait_front(cb_tilized, 1)` - wait for input tile
      - `tile_regs_acquire()`
      - `sub_tiles_bcast_scalar(cb_tilized, cb_mean, 0, 0, 0)` - diff = x - mean
      - `mul_tiles_init()` then `mul_tiles(0, 0, 0)` OR `square_tile_init()` then `square_tile(0)` - square
      - `cb_reserve_back(cb_intermediate, 1)`
      - `tile_regs_commit()`
      - `tile_regs_wait()`
      - `pack_tile(0, cb_intermediate)`
      - `tile_regs_release()`
      - `cb_push_back(cb_intermediate, 1)`
      - `cb_pop_front(cb_tilized, 1)`
    - `cb_pop_front(cb_mean, 1)` - release mean tile after all Wt tiles processed
- **CB Flow**:
  - Input: cb_tilized (c_1) - need to re-read Wt tiles (see note below)
  - Input: cb_mean (c_3) - 1 tile, persist during loop
  - Output: cb_intermediate (c_4) - push 1 tile per input tile (Wt total)

**CRITICAL NOTE**: The spec mentions "Re-tilize input (or double-buffer to keep tilized)". Since cb_tilized is consumed by Phase 2 (reduce helper pops all tiles), we need one of:
1. **Re-tilize**: Call tilize() again (requires reader to send data twice, or double-buffer in reader)
2. **Double-buffer cb_tilized**: Size cb_tilized for 2*Wt tiles, have tilize output to first half, reduce reads from first half, sub/square reads from second half
3. **Interleave**: Process one tile-row at a time with careful CB management

**Recommended approach**: Double-buffer by re-tilizing. The reader sends 32 sticks once, compute tilizes twice (or tilize once to a double-sized CB that isn't fully consumed by reduce).

**Alternative design**: Use `ReduceInputMode::PERSISTENT` for Phase 2 reduce, which does NOT pop input tiles. Then Phase 3 can read from the same CB.

### Phase 4: Reduce Variance (cb_intermediate + cb_scaler -> cb_out_tiled)
- **Description**: Reduce Wt squared difference tiles to variance tile
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>()`
  - **Parameters**: `(cb_intermediate, cb_scaler, cb_out_tiled, TileShape::row(Wt))`
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow** (handled by helper):
  - Uses same scaler CB (c_2) as Phase 2
  - Processes Wt tiles from cb_intermediate streaming
  - Outputs 1 tile to cb_out_tiled (c_5)

### Phase 5: Untilize (cb_out_tiled -> cb_rm_out)
- **Description**: Convert variance tile to row-major format
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::untilize<1, cb_out_tiled, cb_rm_out>()`
  - **Parameters**: `(1)` - num_rows=1 (processing one tile-row at a time)
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow** (handled by helper):
  - Waits for 1 tile from cb_out_tiled (c_5)
  - Reserves/pushes 1 tile to cb_rm_out (c_16)
  - Pops 1 tile from cb_out_tiled

### Compute Loop Structure (per tile-row)
```cpp
void MAIN {
    // Hardware init (required before any helper)
    compute_kernel_hw_startup(cb_rm_in, cb_tilized, cb_scaler, cb_mean,
                              cb_intermediate, cb_out_tiled, cb_rm_out);

    constexpr uint32_t cb_rm_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_tilized = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_mean = tt::CBIndex::c_3;
    constexpr uint32_t cb_intermediate = tt::CBIndex::c_4;
    constexpr uint32_t cb_out_tiled = tt::CBIndex::c_5;
    constexpr uint32_t cb_rm_out = tt::CBIndex::c_16;

    for (uint32_t row = 0; row < num_rows_this_core; ++row) {
        // Phase 1: Tilize
        compute_kernel_lib::tilize(cb_rm_in, Wt, cb_tilized, 1);

        // Phase 2: Reduce mean (PERSISTENT mode to keep tiles for Phase 3)
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
                                   ReduceInputMode::PERSISTENT>(
            cb_tilized, cb_scaler, cb_mean, TileShape::row(Wt));

        // Phase 3: Sub + Square (raw calls - no helper)
        // ... see detailed implementation above ...

        // Phase 4: Reduce variance
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_intermediate, cb_scaler, cb_out_tiled, TileShape::row(Wt));

        // Phase 5: Untilize
        compute_kernel_lib::untilize<1, cb_out_tiled, cb_rm_out>(1);
    }
}
```

## Writer Kernel Design

### Phase 1: Write Row-Major Output Sticks
- **Description**: Write 32 output sticks (one per tile row) to DRAM
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - `TensorAccessor` for DRAM addressing
    - `cb_wait_front(cb_rm_out, 1)` - wait for untilized tile
    - Loop over 32 sticks:
      - `noc_async_write()` for each stick
    - `noc_async_write_barrier()`
    - `cb_pop_front(cb_rm_out, 1)`
- **CB Flow**:
  - Wait for 1 tile from cb_rm_out (c_16)
  - Extract 32 sticks, write each to DRAM
  - Pop 1 tile

### Writer Loop Structure
```
for each output_tile (num_rows_this_core):
    cb_wait_front(cb_rm_out, 1)
    base_l1_addr = get_read_ptr(cb_rm_out)

    for row in 0..31:
        output_stick_id = tile_row_index * 32 + row
        l1_addr = base_l1_addr + row * output_stick_size
        noc_addr = get_noc_addr(output_stick_id, tensor_accessor)
        noc_async_write(l1_addr, noc_addr, output_stick_size)

    noc_async_write_barrier()
    cb_pop_front(cb_rm_out, 1)
```

## CB Synchronization Summary

| CB | Index | Producer | Consumer | Pages per Block | Sync Point |
|----|-------|----------|----------|-----------------|------------|
| cb_rm_in | c_0 | Reader | Compute (tilize) | Wt | Reader pushes Wt pages per tile-row |
| cb_tilized | c_1 | Compute (tilize) | Compute (reduce, sub) | Wt | Tilize pushes Wt, PERSISTENT reduce keeps them |
| cb_scaler | c_2 | Reader (once) | Compute (reduce x2) | 1 | Generated once, persists entire program |
| cb_mean | c_3 | Compute (reduce) | Compute (sub) | 1 | Reduce pushes 1, sub reads and pops 1 |
| cb_intermediate | c_4 | Compute (sub+sq) | Compute (reduce) | Wt | Sub pushes Wt tiles one at a time |
| cb_out_tiled | c_5 | Compute (reduce) | Compute (untilize) | 1 | Reduce pushes 1, untilize pops 1 |
| cb_rm_out | c_16 | Compute (untilize) | Writer | 1 | Untilize pushes 1, writer pops 1 |

### CB Balance Verification (per tile-row)

| CB | Push Count | Pop Count | Balanced |
|----|------------|-----------|----------|
| c_0 | Reader: Wt | Compute (tilize helper): Wt | YES |
| c_1 | Compute (tilize): Wt | Compute (PERSISTENT reduce): 0 + Sub loop: Wt | YES |
| c_2 | Reader: 1 (once) | Compute: 0 (scaler persists, not popped) | SPECIAL |
| c_3 | Compute (reduce): 1 | Compute (sub loop): 1 | YES |
| c_4 | Compute (sub loop): Wt | Compute (reduce): Wt | YES |
| c_5 | Compute (reduce): 1 | Compute (untilize): 1 | YES |
| c_16 | Compute (untilize): 1 | Writer: 1 | YES |

**Note on c_2 (scaler)**: The scaler CB is populated once and read multiple times without popping. The reduce helper calls `cb_wait_front(icb_scaler, 1)` but does NOT pop it. This is intentional - the scaler persists.

## Helper Encapsulation Acknowledgment

For phases marked "USE HELPER", the following is encapsulated BY THE HELPER:
- CB wait/pop/reserve/push operations
- DST register management (acquire/commit/wait/release)
- Init/uninit sequences (tilize_init, reduce_init, untilize_init, etc.)

**CRITICAL**: The kernel writer MUST NOT add redundant CB or DST operations around helper calls. The helper functions are self-contained.

### Helper Operation Summary

| Helper | CB Ops Handled | DST Ops Handled | Init/Uninit Handled |
|--------|---------------|-----------------|---------------------|
| `tilize()` | wait, reserve, push, pop | N/A (tilize uses HW unit) | tilize_init, tilize_uninit |
| `reduce<...>()` | wait, reserve, push, pop (per mode) | acquire, commit, wait, release | reduce_init, reduce_uninit |
| `untilize<...>()` | wait, reserve, push, pop | N/A (pack_untilize uses HW) | untilize_init, untilize_uninit |

## Design Decisions

### Decision 1: PERSISTENT vs Re-tilize for Input Reuse
- **Issue**: Input tiles needed for both mean reduction (Phase 2) and subtract (Phase 3)
- **Options**:
  1. Re-tilize: Reader sends data twice, tilize twice
  2. PERSISTENT mode: Reduce helper keeps tiles in CB for Phase 3
- **Choice**: Use `ReduceInputMode::PERSISTENT` for Phase 2
- **Rationale**: Avoids doubling DRAM bandwidth and compute work. PERSISTENT mode waits for tiles but does NOT pop them, allowing Phase 3 to consume them.

### Decision 2: Sub + Square Implementation
- **Issue**: No combined helper for subtract-broadcast + square
- **Choice**: Raw calls with explicit DST management
- **Rationale**: Unavoidable - this is the gap in helper coverage. The kernel writer must implement this phase manually.

### Decision 3: Untilize Tile Width
- **Issue**: Output is 1 tile wide (variance per row)
- **Choice**: Use `untilize<1, cb_in, cb_out>()` with tile_width=1
- **Rationale**: Single tile fits in DEST, so pack_untilize (hardware-accelerated) path is selected automatically.

## Implementation Checklist for Kernel Writer

### Reader Kernel
- [ ] Include `"api/dataflow/dataflow_api.h"` (correct path)
- [ ] Generate scaler once at start with `generate_reduce_scaler()`
- [ ] Use TensorAccessor for input addressing
- [ ] Reserve Wt pages, read 32 sticks, push Wt pages per tile-row
- [ ] Ensure row_width_bytes * 32 = Wt * tile_size

### Compute Kernel
- [ ] Include all helper headers:
  - `"ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"`
  - `"ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"`
  - `"ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"`
- [ ] Call `compute_kernel_hw_startup()` before any helper
- [ ] Phase 1: Call `tilize()` helper (no manual CB ops)
- [ ] Phase 2: Call `reduce<SUM, REDUCE_ROW, PERSISTENT>()` (no manual CB ops)
- [ ] Phase 3: Implement sub+square with raw calls (manual CB and DST ops)
- [ ] Phase 4: Call `reduce<SUM, REDUCE_ROW>()` (no manual CB ops)
- [ ] Phase 5: Call `untilize<1, ...>()` helper (no manual CB ops)
- [ ] DO NOT add cb_wait/pop/reserve/push around helper calls

### Writer Kernel
- [ ] Include `"api/dataflow/dataflow_api.h"` (correct path)
- [ ] Use TensorAccessor for output addressing
- [ ] Wait for 1 tile, write 32 sticks, pop 1 tile per tile-row
- [ ] Output stick size = TILE_WIDTH * element_size = 32 * 2 = 64 bytes (bfloat16)

### CB Configuration Verification
- [ ] c_0: page_size = tile_size, num_pages = Wt
- [ ] c_1: page_size = tile_size, num_pages = Wt
- [ ] c_2: page_size = tile_size, num_pages = 1
- [ ] c_3: page_size = tile_size, num_pages = 1 (or 2 for double-buffer)
- [ ] c_4: page_size = tile_size, num_pages = Wt (or 2 for streaming)
- [ ] c_5: page_size = tile_size, num_pages = 1
- [ ] c_16: page_size = tile_size, num_pages = 2 (double-buffer for overlap)

## Raw Call Implementation Detail: Phase 3 (Sub + Square)

Since Phase 3 has no helper, here is the detailed raw call implementation:

```cpp
// Phase 3: Subtract mean, square result
// Input: cb_tilized has Wt tiles (from PERSISTENT reduce in Phase 2)
// Input: cb_mean has 1 tile
// Output: cb_intermediate gets Wt tiles

// Init subtraction operation
sub_tiles_bcast_scalar_init_short();

// Wait for mean tile (already there from Phase 2)
cb_wait_front(cb_mean, 1);

// Process each input tile
for (uint32_t wt = 0; wt < Wt; ++wt) {
    // Input tile already in cb_tilized (PERSISTENT mode kept it)
    // Access via index since we didn't pop

    tile_regs_acquire();

    // Subtract: dst[0] = tilized[wt] - mean[0] (broadcast scalar)
    sub_tiles_bcast_scalar(cb_tilized, cb_mean, wt, 0, 0);

    // Square: dst[0] = dst[0]^2
    // Option A: mul_tiles (self-multiply)
    mul_tiles_init();
    mul_tiles(0, 0, 0);  // dst[0] = dst[0] * dst[0]
    // Option B: square_tile if available
    // square_tile_init(); square_tile(0);

    // Pack result to intermediate CB
    cb_reserve_back(cb_intermediate, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_intermediate);
    tile_regs_release();
    cb_push_back(cb_intermediate, 1);
}

// Pop the tiles from cb_tilized (Wt tiles) and cb_mean (1 tile)
cb_pop_front(cb_tilized, Wt);
cb_pop_front(cb_mean, 1);
```

## Open Design Questions

1. **CB c_1 (tilized) sizing**: If using PERSISTENT mode for reduce, cb_tilized needs to hold Wt tiles for the duration of both Phase 2 and Phase 3. Verify L1 has sufficient space.

2. **FP32 accumulation**: The spec recommends `fp32_dest_acc_en = true` for better precision. The reduce helper auto-detects this via `get_fp32_dest_acc_enabled()`. Ensure the program factory sets the appropriate ComputeConfig.

3. **Sub-broadcast precision**: Verify `sub_tiles_bcast_scalar` broadcasts correctly from tile position [0,0] of cb_mean. The mean value after REDUCE_ROW is in the scalar position of the output tile.
