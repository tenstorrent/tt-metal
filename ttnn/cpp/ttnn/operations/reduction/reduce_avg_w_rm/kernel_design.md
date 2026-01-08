# Kernel Design: reduce_avg_w_rm

## Design Summary

| Kernel | Phases | Helpers Used | Raw Calls Needed |
|--------|--------|--------------|------------------|
| Reader | 2 | None | `generate_reduce_scaler()`, `noc_async_read()`, CB API |
| Compute | 3 | `compute_kernel_lib::tilize()`, `compute_kernel_lib::reduce()`, `compute_kernel_lib::untilize()` | None |
| Writer | 1 | None | `noc_async_write()`, CB API |

## Helper Library Analysis

### Available Helpers Reviewed
- [x] tilize_helpers.hpp - **relevant: yes** - Used for Phase 1 (ROW_MAJOR -> TILE_LAYOUT)
- [x] untilize_helpers.hpp - **relevant: yes** - Used for Phase 3 (TILE_LAYOUT -> ROW_MAJOR)
- [x] reduce_helpers.hpp - **relevant: yes** - Used for Phase 2 (W-dimension reduction)
- [x] dest_helpers.hpp - **relevant: yes** - Auto-detects DEST register limits for reduce/untilize

### Helper Functions Applicable to This Operation

| Helper | Signature | Use Case in This Op |
|--------|-----------|---------------------|
| `compute_kernel_lib::tilize()` | `tilize<init, uninit, use_fast, use_dt, skip_wait>(icb, block_w, ocb, num_blocks, subblock_h, old_icb, input_count, total_rows)` | Phase 1: Convert row-major sticks to tiles |
| `compute_kernel_lib::reduce()` | `reduce<PoolType, ReduceDim, ReduceInputMode, ReduceDataFormatReconfig, init, uninit>(icb, icb_scaler, ocb, TileShape, TileLayout, post_reduce_op)` | Phase 2: Sum tiles across width dimension with 1/W scaler |
| `compute_kernel_lib::untilize()` | `untilize<tile_width, icb_id, ocb_id, init, uninit, wait_upfront>(num_rows, block_rt_dim, total_tiles)` | Phase 3: Convert reduced tiles to row-major |

## Reader Kernel Design

### Phase 1: Scaler Tile Generation
- **Description**: Generate a single tile filled with the scaler value (1/W) for averaging. This tile is used by the reduce operation to scale the sum to produce an average.
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**: Use `generate_reduce_scaler()` from `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
    - `cb_reserve_back(cb_scaler, 1)` - handled internally by generate_reduce_scaler
    - Fill tile with zeros using `noc_async_read()` from MEM_ZEROS_BASE
    - Write scaler value pattern to first 8 positions of each face
    - `cb_push_back(cb_scaler, 1)` - handled internally by generate_reduce_scaler
- **CB Flow**:
  - Output: `cb_scaler` (c_2) receives 1 tile (once per kernel lifetime)

### Phase 2: Row-Major Stick Reading
- **Description**: Read 32 row-major sticks (one tile-row) from DRAM and push to input CB. Repeat for each block assigned to this core.
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - `cb_reserve_back(cb_in, Wt)` - reserve space for Wt tiles worth of sticks (32 sticks total stick_size)
    - For each of 32 sticks:
      - `accessor.get_noc_addr(stick_id)` - compute NOC address via TensorAccessor
      - `noc_async_read()` - read stick from DRAM to L1
    - `noc_async_read_barrier()` - wait for all reads to complete
    - `cb_push_back(cb_in, Wt)` - signal Wt pages ready
- **CB Flow**:
  - Output: `cb_in` (c_0) receives Wt pages per block
  - Note: CB page_size = tile_size, but physical data is 32 sticks of stick_size bytes
  - Memory equivalence: `32 * stick_size = Wt * tile_size` (32 sticks = Wt tiles worth of data)

### Runtime Args
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer DRAM address |
| 1 | num_sticks | uint32_t | Total input sticks to read (num_blocks * 32) |
| 2 | start_stick_id | uint32_t | Starting stick index for this core |

### Compile-Time Args
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scaler_value | uint32_t | Scaler (1/W) as float bits |
| 1 | stick_size | uint32_t | Input stick size in bytes (W * 2) |
| 2+ | TensorAccessorArgs | varies | Input tensor accessor configuration |

## Compute Kernel Design

### Prerequisites
- [x] Requires `compute_kernel_hw_startup()`: **yes** - Must call before using any helper
- [ ] Template parameters for reduce helper (applicable):
  - `PoolType`: **SUM** (we scale by 1/W to get average)
  - `ReduceDim`: **REDUCE_ROW** (reduce along W dimension)
  - `ReduceInputMode`: **STREAMING** (tiles arrive one-at-a-time from tilize)
  - `ReduceDataFormatReconfig`: **BOTH** (default, reconfigure unpacker and packer)

### Phase 1: Tilize
- **Description**: Convert row-major sticks from `cb_in` to tile format in `cb_tilized`. Process Wt tiles per block.
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::tilize()`
  - **Parameters**:
    ```cpp
    compute_kernel_lib::tilize(
        cb_in,           // icb: c_0
        Wt,              // block_w: tiles per row
        cb_tilized,      // ocb: c_1
        num_blocks       // num_blocks: tile rows to process
    );
    ```
  - **Template Parameters**: Use defaults: `<true, true, false, false, false>` (init, uninit, not fast, no DT, no skip_wait)
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow** (handled by helper):
  - Input: `cb_in` (c_0) - waits for Wt pages, pops Wt pages per block
  - Output: `cb_tilized` (c_1) - reserves Wt tiles, pushes Wt tiles per block

### Phase 2: Reduce
- **Description**: Reduce each row of Wt tiles to a single output tile using SUM with 1/W scaler (producing average). Process Ht rows.
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>()`
  - **Parameters**:
    ```cpp
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
                               compute_kernel_lib::ReduceInputMode::STREAMING>(
        cb_tilized,      // icb: c_1
        cb_scaler,       // icb_scaler: c_2
        cb_reduced,      // ocb: c_3
        compute_kernel_lib::TileShape::grid(num_blocks, Wt, 1)  // shape: num_blocks rows x Wt cols x 1 batch
    );
    ```
  - **Template Parameters**:
    - `PoolType::SUM` - use SUM with scaler for averaging
    - `ReduceDim::REDUCE_ROW` - reduce across width dimension
    - `ReduceInputMode::STREAMING` - tiles arrive one-at-a-time
    - `ReduceDataFormatReconfig::BOTH` - default, reconfig both unpacker and packer
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow** (handled by helper):
  - Input: `cb_tilized` (c_1) - helper waits/pops 1 tile at a time in STREAMING mode
  - Input: `cb_scaler` (c_2) - helper waits for scaler once (does not pop)
  - Output: `cb_reduced` (c_3) - helper reserves/pushes 1 tile per row

### Phase 3: Untilize
- **Description**: Convert reduced tiles from `cb_reduced` to row-major format in `cb_out`. Each tile becomes 32 sticks of width 32.
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::untilize<tile_width, icb_id, ocb_id>()`
  - **Parameters**:
    ```cpp
    compute_kernel_lib::untilize<1, cb_reduced, cb_out>(num_blocks);
    ```
  - **Template Parameters**:
    - `tile_width = 1` - each row has 1 tile (output width = 32 elements = 1 tile)
    - `icb_id = cb_reduced` (c_3)
    - `ocb_id = cb_out` (c_16)
    - `init = true` (default)
    - `uninit = true` (default)
    - `wait_upfront = false` (default)
  - **CB Management**: Helper handles internally - DO NOT add cb_wait/pop/reserve/push
- **CB Flow** (handled by helper):
  - Input: `cb_reduced` (c_3) - helper waits for 1 tile, pops 1 tile per row
  - Output: `cb_out` (c_16) - helper reserves/pushes 1 tile per row

### Compile-Time Args
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_blocks_per_core | uint32_t | Number of tile rows for this core |
| 1 | Wt | uint32_t | Width in tiles (tiles per row) |

### Runtime Args
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_blocks | uint32_t | Number of blocks (tile rows) to process |

## Writer Kernel Design

### Phase 1: Row-Major Stick Writing
- **Description**: Wait for output data in `cb_out`, extract 32 sticks per tile, and write to DRAM.
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - `cb_wait_front(cb_out, 1)` - wait for 1 page (1 tile's worth of sticks)
    - `get_read_ptr(cb_out)` - get L1 read address
    - For each of 32 sticks:
      - Compute L1 offset for stick within tile
      - `accessor.get_noc_addr(stick_id)` - compute NOC address via TensorAccessor
      - `noc_async_write()` - write stick from L1 to DRAM
    - `noc_async_write_barrier()` - wait for all writes to complete
    - `cb_pop_front(cb_out, 1)` - release the page
- **CB Flow**:
  - Input: `cb_out` (c_16) - waits for 1 page, pops 1 page per block
  - Output stick width: 32 elements (one tile width)

### Runtime Args
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer DRAM address |
| 1 | num_sticks | uint32_t | Total output sticks to write (num_blocks * 32) |
| 2 | start_stick_id | uint32_t | Starting output stick index for this core |

### Compile-Time Args
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output CB ID (c_16) |
| 1 | output_stick_size | uint32_t | Output stick size in bytes (32 * 2 = 64) |
| 2+ | TensorAccessorArgs | varies | Output tensor accessor configuration |

## CB Synchronization Summary

| CB | Producer | Consumer | Pages per Block | Sync Point |
|----|----------|----------|-----------------|------------|
| c_0 (cb_in) | Reader | Compute (tilize) | Wt | Reader pushes Wt pages of sticks, tilize helper waits for Wt pages |
| c_1 (cb_tilized) | Compute (tilize) | Compute (reduce) | Wt | Tilize pushes Wt tiles, reduce streams 1 at a time |
| c_2 (cb_scaler) | Reader | Compute (reduce) | 1 | Reader pushes 1 tile once, reduce waits for 1 (no pop) |
| c_3 (cb_reduced) | Compute (reduce) | Compute (untilize) | 1 | Reduce pushes 1 tile per row, untilize waits/pops 1 |
| c_16 (cb_out) | Compute (untilize) | Writer | 1 | Untilize pushes 1 tile per row, writer waits/pops 1 |

### CB Sizing (from program factory)
| CB | Size | Formula |
|----|------|---------|
| c_0 (cb_in) | 32 * stick_size | 32 sticks = Wt tiles worth of row-major data |
| c_1 (cb_tilized) | Wt * tile_size | Holds one row of tilized tiles |
| c_2 (cb_scaler) | 1 * tile_size | Single scaler tile |
| c_3 (cb_reduced) | 1 * tile_size | One reduced tile per row |
| c_16 (cb_out) | 32 * 64 | 32 sticks of width 32 (output_stick_size = 64 bytes) |

## Helper Encapsulation Acknowledgment

For phases marked "USE HELPER", the following is encapsulated BY THE HELPER:
- CB wait/pop/reserve/push operations
- DST register management (acquire/commit/wait/release)
- Init/uninit sequences (tilize_init, reduce_init, untilize_init, etc.)

**CRITICAL**: The kernel writer MUST NOT add redundant CB or DST operations around helper calls. The helper functions are self-contained.

### tilize() encapsulates:
- `tilize_init()` / `tilize_uninit()`
- `cb_wait_front(icb, ...)` / `cb_pop_front(icb, ...)`
- `cb_reserve_back(ocb, ...)` / `cb_push_back(ocb, ...)`
- `tilize_block()` calls

### reduce() encapsulates:
- `reduce_init()` / `reduce_uninit()`
- `reconfig_data_format()` / `pack_reconfig_data_format()`
- `cb_wait_front(icb, ...)` / `cb_pop_front(icb, ...)`
- `cb_wait_front(icb_scaler, 1)` (wait for scaler, no pop)
- `cb_reserve_back(ocb, ...)` / `cb_push_back(ocb, ...)`
- `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()`
- `reduce_tile()` calls
- `pack_tile()` calls

### untilize() encapsulates:
- `pack_untilize_init()` or `untilize_init()` (auto-selected)
- `cb_wait_front(icb, ...)` / `cb_pop_front(icb, ...)`
- `cb_reserve_back(ocb, ...)` / `cb_push_back(ocb, ...)`
- `pack_untilize_block()` or `untilize_block()` (auto-selected based on width/datatype)
- `pack_untilize_uninit()` or `untilize_uninit()`

## Implementation Checklist for Kernel Writer

### Reader Kernel
- [ ] Include `api/dataflow/dataflow_api.h`
- [ ] Include `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
- [ ] Call `generate_reduce_scaler(cb_scaler, packed_scaler_value)` once at start
- [ ] Create TensorAccessor from compile-time args
- [ ] Loop over num_sticks / 32 blocks:
  - [ ] `cb_reserve_back(cb_in, Wt)`
  - [ ] Read 32 sticks via `noc_async_read()`
  - [ ] `noc_async_read_barrier()`
  - [ ] `cb_push_back(cb_in, Wt)`

### Compute Kernel
- [ ] Include `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
- [ ] Include `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
- [ ] Include `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
- [ ] Define CB indices as constexpr (c_0, c_1, c_2, c_3, c_16)
- [ ] Call `compute_kernel_hw_startup(cb_in, cb_out)` FIRST
- [ ] Call `compute_kernel_lib::tilize(cb_in, Wt, cb_tilized, num_blocks)`
- [ ] Call `compute_kernel_lib::reduce<SUM, REDUCE_ROW, STREAMING>(cb_tilized, cb_scaler, cb_reduced, TileShape::grid(num_blocks, Wt, 1))`
- [ ] Call `compute_kernel_lib::untilize<1, cb_reduced, cb_out>(num_blocks)`

### Writer Kernel
- [ ] Include `api/dataflow/dataflow_api.h`
- [ ] Create TensorAccessor from compile-time args
- [ ] Loop over num_sticks / 32 blocks:
  - [ ] `cb_wait_front(cb_out, 1)`
  - [ ] Get L1 read address via `get_read_ptr(cb_out)`
  - [ ] Write 32 sticks via `noc_async_write()` with output_stick_size = 64
  - [ ] `noc_async_write_barrier()`
  - [ ] `cb_pop_front(cb_out, 1)`

### Verification
- [ ] CB push/pop counts match across kernels
- [ ] Reader pushes Wt pages to c_0, compute waits for Wt
- [ ] Reader pushes 1 page to c_2 (scaler), compute waits for 1 (no pop)
- [ ] Compute (tilize) pushes Wt to c_1, compute (reduce) pops 1 at a time (Wt total)
- [ ] Compute (reduce) pushes 1 to c_3, compute (untilize) pops 1
- [ ] Compute (untilize) pushes 1 to c_16, writer pops 1

## Important Implementation Notes

### Scaler Value Handling
The program factory computes the scaler as:
```cpp
float scaler = 1.0f / static_cast<float>(W);
uint32_t packed_scaler_value = *reinterpret_cast<uint32_t*>(&scaler);
```

This is passed to the reader kernel, which uses `generate_reduce_scaler()` to create a tile where the first 8 values of each face contain the scaler. This matches the format expected by `reduce_tile()`.

### CB Page Size Alignment
The program factory configures `cb_in` with:
- `page_size = cb_in_size = 32 * stick_size` (entire block)
- `num_pages = 1` (implicit from total_size / page_size)

However, the push/pop operations use `Wt` as the page count because:
- CB API semantics: `cb_push_back(cb, N)` signals N logical pages
- Memory equivalence: `32 * stick_size = Wt * tile_size`
- Both reader and compute must agree on the page count (Wt)

### Reduce Helper Mode Selection
Using `ReduceInputMode::STREAMING` because:
1. Tiles come from tilize one block at a time
2. Each row of Wt tiles is processed sequentially
3. No need to pre-load all tiles (memory efficient)
4. Compatible with single-buffered CBs

Alternative: `STREAMING_BATCHED` would wait for all Wt tiles before processing, but offers no benefit here since tilize already outputs Wt tiles per block.

### Untilize Width = 1
The output has width 1 tile (32 elements) because:
- Reduce across W produces 1 tile per row
- `untilize<1, cb_reduced, cb_out>()` tells the helper the row width is 1 tile
- This enables the fast `pack_untilize` path (width <= DEST limit)
