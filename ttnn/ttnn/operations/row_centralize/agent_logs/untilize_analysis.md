# Untilize Multi-Core Implementation Analysis

## Overview

The untilize operation converts tensor data from **tiled layout** (32x32 tiles with face-based internal ordering) to **row-major layout** (contiguous rows). This is the inverse of the tilize operation. The multi-core program factory distributes tile rows across multiple Tensix cores for parallel processing.

This analysis covers the `UntilizeMultiCoreProgramFactory` which handles:
- **Interleaved input to interleaved output** (the primary path relevant as an output_stage reference)
- **Sharded input to interleaved output** (secondary path, also supported)

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Context**: This analysis serves the **output_stage** role for a hybrid-mode row_centralize operation that performs row-wise standardization on RM interleaved tensors. The key pattern to extract is how tiled compute output is converted to row-major format and written to interleaved DRAM.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = one row of tiles spanning the width) |
| **Unit size** | `num_tiles_per_input_block` tiles (number of tiles across one row of the tensor) |
| **Total units** | `num_tiles_per_col` blocks (total tile-rows in the tensor) |
| **Loop structure** | Outer: iterate over blocks assigned to this core. Inner: process one tile-row at a time (reader reads tiles, compute untilizes, writer extracts row-major sticks and writes them) |

A "block" in this operation corresponds to one tile-row: all tiles in a single row of tiles spanning the tensor width. For a tensor of shape `[H, W]` in tiles, there are `H/tile_height` tile-rows, each containing `W/tile_width` tiles. Each core processes multiple consecutive tile-rows.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D (flattened to 2D: height x width) |
| **Dimension convention** | Last dim = width, all outer dims collapsed to height |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with face-based internal order) |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block sharded) |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT (contiguous rows) |
| **Memory layout** | INTERLEAVED (primary), or WIDTH_SHARDED / BLOCK_SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (BFLOAT16, FLOAT32, INT32, UINT32, UINT16) |

### Layout Transformations

The core transformation is **tiled-to-row-major** conversion:
- **Input**: Tiles in face-ordered format (face0, face1, face2, face3 each 16x16, stored contiguously within each 32x32 tile)
- **Output**: Row-major sticks where each stick is one row of elements spanning the tensor width

The compute kernel performs the actual data reordering (untilize). The writer kernel then rearranges the output from tile-row-sized chunks into individual row-major sticks that are written to the correct pages in DRAM.

**Page structure change**: Input pages are tiles (e.g., 32x32 x element_size bytes). Output pages are row-major sticks (1 row x tensor_width_per_output_block x element_size bytes).

## Data Flow Pattern

### Step-by-step (Interleaved Input Path)

| Stage | Kernel | Reads From | Writes To | CB Operations | Description |
|-------|--------|------------|-----------|---------------|-------------|
| 1 | Reader | DRAM (interleaved) | CB c_0 | `cb_reserve_back(1)` then `cb_push_back(1)` per tile | Reads tiles sequentially by page ID from DRAM into input CB, one tile at a time |
| 2 | Compute | CB c_0 | CB c_16 | `cb_wait_front(block_width)`, `cb_pop_front(block_width)`, `cb_reserve_back(block_width)`, `cb_push_back(block_width)` | Untilizes one tile-row at a time: waits for a full row of tiles, transforms from tile to row-major format, pushes row-major data to output CB |
| 3 | Writer | CB c_16 | DRAM (interleaved) | `cb_wait_front(num_tiles_per_input_block)`, `cb_pop_front(num_tiles_per_input_block)` | Reads row-major data from output CB, iterates over each of the `tile_height` rows in the block, writes each row-stick to the correct DRAM page via TensorAccessor |

### Step-by-step (Sharded Input Path)

| Stage | Kernel | Reads From | Writes To | CB Operations | Description |
|-------|--------|------------|-----------|---------------|-------------|
| 1 | Reader | L1 (shard in-place) | CB c_0 | `cb_push_back(num_tiles)` only (no read needed, CB backed by shard buffer) | Simply marks the entire shard as available in the CB -- the data is already in L1 at the CB's address |
| 2 | Compute | CB c_0 | CB c_16 | Same as interleaved path | Same untilize logic |
| 3 | Writer | CB c_16 | DRAM (interleaved) | Same as interleaved path | Same write logic |

### Key Observation for Output Stage Reference

The writer kernel is the most relevant component for the output_stage pattern. It demonstrates:
1. How to convert tile-row-sized chunks in a CB into individual row-major sticks
2. How to use `TensorAccessor` with `get_noc_addr(page_id, byte_offset)` for sub-page writes
3. How to handle width-sharded output where one tile-row maps to multiple output pages

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|--------------------|-----------|-----------|-----------| ---------|
| c_0 | cb_src0 | Input staging (tiled data) | See note 1 | `num_tiles_per_input_block` (tiles per tile-row) | See note 2 | Reader | Compute | Program |
| c_16 | cb_output | Output staging (row-major data after untilize) | See note 3 | `num_tiles_per_input_block` | See note 4 | Compute | Writer | Program |

### Capacity Notes

**Note 1 - Input CB capacity (c_0)**:
- **Sharded input**: `num_tiles_per_input_block * num_input_blocks_per_full_core` (entire shard, since the CB is backed by the shard buffer)
- **Interleaved, single block per core**: `num_tiles_per_input_block` (single-buffered, only 1 block to process)
- **Interleaved, 2+ blocks per core**: `num_tiles_per_input_block * 2` (double-buffered)

**Note 2 - Input CB buffering**:
- Sharded: effectively single-buffered (entire shard loaded at once)
- Interleaved, 1 block: single-buffered
- Interleaved, 2+ blocks: **double-buffered** -- allows reader to prefetch next tile-row while compute processes current

**Note 3 - Output CB capacity (c_16)**:
- 1 block per core: `num_tiles_per_input_block` (single-buffered)
- 2+ blocks per core: `num_tiles_per_input_block * 2` (double-buffered)

**Note 4 - Output CB buffering**:
- 1 block per core: single-buffered
- 2+ blocks per core: **double-buffered** -- allows compute to produce next tile-row while writer writes current

## Pipeline Pattern Summary

For the common multi-block case (2+ blocks per core on interleaved input):
- **Input CB (c_0)**: Double-buffered (capacity = 2 * block_size). Reader and compute can overlap.
- **Output CB (c_16)**: Double-buffered (capacity = 2 * block_size). Compute and writer can overlap.
- **Pipeline**: Reader -> Compute -> Writer with double-buffering enabling overlap at both boundaries.

For single-block case: All single-buffered, no overlap possible (trivial -- only one block to process).

## Index Calculations

### Reader (Interleaved Path)

The reader uses a simple linear tile ID scheme:
- `start_page_id` = cumulative tile offset for this core (set by host as runtime arg)
- Iterates `page_id` from `start_page_id` to `start_page_id + num_tiles`
- Each `page_id` is a global tile index into the interleaved tile buffer
- `TensorAccessor.get_noc_addr(page_id)` maps the linear page ID to a physical DRAM bank address

### Writer

The writer performs the most complex index calculations to map tile-row output to row-major stick pages:

1. **Block height index** (`height_wise_input_block_start_index`): Global index of the first tile-row this core processes
2. For each tile-row, iterates over `tile_height` (32) individual rows within the tile-row
3. For each row within a tile-row:
   - Computes the global row number: `block_height_index * tile_height + j`
   - Computes the output page ID: `global_row * num_output_blocks_across_width + width_wise_output_block_start_index`
   - Reads row-major data from the CB at offset `j * num_cols_per_input_block * element_size`
   - Writes `num_cols_per_input_block * element_size` bytes per row (or handles split across output pages for sharded output)

### Output Page Mapping (Width-Sharded Output)

When output is width-sharded or block-sharded, a single row of elements may span multiple output pages:
- `num_output_blocks_across_width` = number of output pages per row
- `output_stick_size` = `tensor_width * element_size / num_output_blocks_across_width`
- The writer iterates through columns with a while loop, writing partial rows to consecutive output pages when the input block spans multiple output blocks

## Memory Access Patterns

### Read Pattern (Interleaved Input)

- **Ordering**: Sequential tile IDs (contiguous tile-rows assigned to each core)
- **Granularity**: One tile per read transaction
- **Access type**: NoC async read, tile-sized (e.g., 2048 bytes for BF16 32x32)
- **Barrier**: `noc_async_read_barrier()` after each tile (conservative, no batching)
- **Bank access**: Round-robin across DRAM banks via TensorAccessor

### Read Pattern (Sharded Input)

- **No read**: Data already in L1 (CB backed by shard buffer)
- Reader only calls `cb_push_back` to make tiles visible to compute

### Write Pattern

- **Ordering**: Row-major sticks within each tile-row, then next tile-row
- **Granularity**: One row-stick per write transaction (`output_stick_size` bytes)
- **Access type**: NoC async write with `get_noc_addr(page_id, byte_offset_within_page)`
- **Barrier**: `noc_async_write_barrier()` after each complete tile-row (all `tile_height` rows written)
- **Sub-page writes**: When width-sharded output, partial row writes with byte offset within pages
- **Bank access**: Round-robin across DRAM banks via TensorAccessor

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) for interleaved; shard grid for sharded |
| **Grid dimensions** | Up to `device.compute_with_storage_grid_size()` (e.g., 8x8 = 64 cores on Wormhole) |
| **Total cores** | Determined by `split_blocks_for_tilize(grid_size, num_tiles_per_col)` |
| **Work per core** | `num_rows_per_full_core` tile-rows for full cores; `num_rows_per_cliff_core` for cliff core |
| **Load balancing** | Near-equal distribution with at most 1 cliff core handling remainder |
| **Remainder handling** | Cliff core: the last core may process fewer tile-rows than full cores |

### Interleaved Path Details

`split_blocks_for_tilize` computes:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)` -- each core gets this many tile-rows
- `ncores = ceil(num_tiles_per_col / nblocks_per_core)` -- total cores used
- If `num_tiles_per_col % nblocks_per_core != 0`, the last core is a "cliff core" with fewer blocks
- Full cores form `full_compute_core_range`; cliff core (0 or 1) forms `cliff_compute_core_range`
- Both ranges get the same kernel binaries but different runtime args (different `num_input_blocks_to_process`)

### Sharded Path Details

- Uses the shard spec's core grid directly (no `split_blocks_for_tilize`)
- No cliff cores (each core processes its shard)
- Supports uneven sharding: last shard in row/column may have fewer valid elements
- `num_input_blocks_across_width` > 1 for width-sharded or block-sharded inputs

## Arguments

### Reader Kernel -- Compile-Time Arguments

**Interleaved path** (`reader_unary_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor configuration for source buffer (appended by `TensorAccessorArgs(*src0_buffer).append_to()`) |

**Sharded path** (`reader_unary_sharded.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |

### Reader Kernel -- Runtime Arguments

**Interleaved path**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_tiles | uint32_t | Total tiles to read (= tiles_per_row * blocks_per_core) |
| 2 | start_page_id | uint32_t | Global tile index of first tile for this core |

**Sharded path**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Number of tiles in the shard to make visible |

### Compute Kernel -- Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Number of tiles per tile-row (block width in tiles) |
| 1 | src_cb_id | uint32_t | Input CB ID (c_0) |
| 2 | out_cb_id | uint32_t | Output CB ID (c_16) |

### Compute Kernel -- Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile-rows (blocks) to process on this core |

### Writer Kernel -- Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output CB ID (c_16) |
| 1 | output_stick_size | uint32_t | Bytes per output page (row width in bytes, accounting for output sharding) |
| 2 | tile_height | uint32_t | Tile height (typically 32) |
| 3 | num_tiles_per_input_block | uint32_t | Tiles per tile-row (block width in tiles) |
| 4 | num_output_blocks_across_width | uint32_t | Number of output pages per row (1 for interleaved, >1 for width/block sharded output) |
| 5 | output_element_size | uint32_t | Bytes per element (2 for BF16, 4 for FP32) |
| 6 | num_cols_per_input_block | uint32_t | Columns per input block (= tiles_per_row * tile_width) |
| 7 | num_cols_per_output_block | uint32_t | Columns per output page (= tensor_width / num_output_blocks_across_width) |
| 8+ | TensorAccessorArgs | uint32_t[] | Tensor accessor configuration for destination buffer |

### Writer Kernel -- Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_input_blocks_to_process | uint32_t | Number of tile-rows this core must write |
| 2 | height_wise_input_block_start_index | uint32_t | Global index of first tile-row for this core |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Valid columns (handles uneven sharding; equals num_cols_per_input_block for interleaved) |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page index within a row (0 for interleaved) |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Column offset within first output page (0 for interleaved; non-zero when input block starts mid-page for sharded) |

## Kernel Implementations

### Reader Kernel (Interleaved): `reader_unary_start_id.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM (interleaved tiles) | CB c_0 | Read tiles sequentially by page ID |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

**Key Logic**:
- Creates `TensorAccessor` from compile-time args (starting at index 1)
- Simple loop from `start_page_id` to `start_page_id + num_tiles`
- For each tile: `cb_reserve_back(1)` -> `noc_async_read` -> `noc_async_read_barrier()` -> `cb_push_back(1)`
- One-tile-at-a-time granularity with per-tile barrier (conservative but simple)

### Reader Kernel (Sharded): `reader_unary_sharded.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | N/A | L1 shard (in-place) | CB c_0 | Mark shard tiles as available |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

**Key Logic**:
- Only calls `cb_push_back(cb_id_in0, num_tiles_per_core)` to make the shard visible
- No actual data movement -- the CB is configured with `set_globally_allocated_address` pointing to the shard buffer
- This is a standard pattern for sharded inputs

### Compute Kernel (Fast Path): `pack_untilize_variable_num_blocks.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Compute | RISCV_2 (TRISC) | N/A | CB c_0 (tiled) | CB c_16 (row-major) | Hardware-accelerated pack_untilize |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`

**Key Logic**:
- Delegates to `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt)`
- The unified untilize helper auto-dispatches based on width vs DEST capacity:
  - Width <= DEST limit: Single-pass `pack_untilize_block` (hardware-accelerated, optimal)
  - Width > DEST limit AND integer type: Block-based `pack_untilize_block` in sub-blocks
  - Width > DEST limit AND float type: Falls back to standard untilize (slow path kernel used instead)
- `DST_ACCUM_MODE` define limits `max_bct` (maximum block count in tiles): 4 for 32-bit accum, 8 for 16-bit
- Processes one tile-row per iteration: `cb_wait_front(block_width)` -> untilize -> `cb_pop_front(block_width)` -> `cb_push_back(block_width)`

### Compute Kernel (Slow Path): `untilize_variable_num_blocks.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Compute | RISCV_2 (TRISC) | N/A | CB c_0 (tiled) | CB c_16 (row-major) | Standard untilize (SFPU-based) |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`

**Key Logic**:
- Same API as fast path: `compute_kernel_lib::untilize<...>(per_core_block_cnt)`
- Used when: `!use_pack_untilize`, or UINT16 dtype, or (FLOAT32 AND width >= `MAX_PACK_UNTILIZE_WIDTH`)
- These conditions are checked at program creation time (lines 197-209 of the factory)

### Writer Kernel: `writer_unary_stick_layout_split_rows_multi_core.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Writer | RISCV_1 | NOC1 | CB c_16 (row-major) | DRAM (interleaved sticks) | Write row-major sticks |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Key Logic**:
- Creates `TensorAccessor` from compile-time args (starting at index 8) using destination buffer info
- Outer loop: iterate over assigned tile-rows (`num_input_blocks_to_process`)
- Inner lambda `write_tiles_in_current_block(block_height_index)`:
  1. `cb_wait_front(num_tiles_per_input_block)` -- wait for one tile-row of row-major data
  2. For each of `tile_height` rows within the tile-row:
     - Compute L1 read address: `base_l1_read_addr + j * num_cols_per_input_block * element_size`
     - Compute output page ID from global row number
     - Write the row data to DRAM, handling potential splits across multiple output pages
  3. `noc_async_write_barrier()` -- wait for all writes in this tile-row
  4. `cb_pop_front(num_tiles_per_input_block)` -- free the tile-row in the CB

**Critical detail for output_stage reference**: The writer handles the case where output pages are smaller than the row width (width/block sharded output). It uses a while loop to write partial rows to consecutive output pages, tracking `num_cols_remaining_in_current_output_block` and `output_offset_within_page_in_bytes`.

## Implementation Notes

### Compute Kernel Selection (Factory Lines 196-209)

The factory selects between two compute kernels at program creation time:
- **Slow path** (`untilize_variable_num_blocks.cpp`): Used when `!use_pack_untilize`, or input dtype is UINT16, or (FLOAT32 AND `num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH`)
- **Fast path** (`pack_untilize_variable_num_blocks.cpp`): Used otherwise (the common case for BF16)

Additionally, for FLOAT32 on the slow path, `UnpackToDestMode` is forced to `Default` (not FP32 accumulation) due to hardware limitations (referenced issues #30400, #33795).

### DST_ACCUM_MODE Define

For INT32, UINT32, and FLOAT32 data types, the factory sets `compute_kernel_defines["DST_ACCUM_MODE"] = "1"`. This halves the DEST register capacity (from 16 to 8 tiles in full-sync mode, or 8 to 4 in half-sync mode), which constrains the maximum tile-row width that can be processed in a single pass by `pack_untilize`.

### Sharded Input Handling

When input is sharded:
- The input CB is sized to hold the entire shard (`num_tiles_per_input_block * num_input_blocks_per_full_core`)
- The CB is backed directly by the shard buffer (`set_globally_allocated_address`)
- No cliff cores exist (each core processes exactly its shard)
- Uneven sharding is handled: the last shard in a row/column may have fewer valid elements, controlled via `num_unpadded_cols_per_input_block` and `num_input_blocks_to_process` runtime args

### Width-Sharded/Block-Sharded Output

When output is width- or block-sharded, the writer must handle:
- `num_output_blocks_across_width > 1`: each row spans multiple output pages
- `output_stick_size` is reduced accordingly
- The writer's inner while loop splits row data across output pages, handling byte offsets within pages for the first write

### Program Cache Override

`override_runtime_arguments` updates only buffer addresses when tensors are reallocated:
- For sharded input: updates the CB's globally-allocated address via `UpdateDynamicCircularBufferAddress`
- For interleaved input: updates `runtime_args[0]` (src_addr) for all cores
- Always updates writer `runtime_args[0]` (dst_addr) for all cores

### Relevance as Output Stage Reference

For the row_centralize operation's output stage, the key patterns to replicate are:
1. **CB c_16 as output staging**: Compute kernel writes row-major data, writer reads and writes to DRAM
2. **Double-buffered output CB** when processing multiple blocks for overlap
3. **Writer's row-stick extraction**: Reading from CB at computed L1 offsets, writing to DRAM via TensorAccessor
4. **Per-tile-row barrier**: `noc_async_write_barrier()` after each tile_height worth of row writes
5. **TensorAccessor for output**: `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` on host; `TensorAccessor(dst_args, dst_addr, output_stick_size)` on device

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessorArgs work in tt-metal? How is it used with compile-time args via the append_to pattern?"
   **Reason**: Needed to understand how the factory passes buffer layout information to kernels
   **Key Findings**: `TensorAccessorArgs(*buffer).append_to(compile_time_args)` appends rank, bank count, tensor/shard shapes, and bank coordinates as compile-time args. On the device side, `TensorAccessorArgs<base_idx>()` reconstructs the accessor from compile-time args. The accessor then provides `get_noc_addr(page_id)` and `get_noc_addr(page_id, byte_offset)` for address calculation.

2. **Query**: "What is pack_untilize in TT-Metal? How does it differ from standard untilize?"
   **Reason**: Needed to understand the two compute paths
   **Key Findings**: DeepWiki was unavailable (500 error). Analysis based on source code: `pack_untilize` is a hardware-accelerated path that uses the packer unit to reorder tile data to row-major in a single pass (when width fits in DEST registers). Standard untilize is a software fallback. The unified `compute_kernel_lib::untilize` auto-dispatches between them.

3. **Query**: "How does split_blocks_for_tilize work? What is the cliff core concept?"
   **Reason**: Needed to understand core distribution strategy
   **Key Findings**: DeepWiki was unavailable (500 error). Analysis based on source code in `work_split_tilize.hpp`: `split_blocks_for_tilize(grid_size, nblocks)` divides `nblocks` tile-rows evenly across available cores. If there is a remainder, the last core becomes a "cliff core" processing fewer blocks. Returns a `BlockSplit` with separate `CoreRangeSet` for full cores and cliff core.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tile vs row-major layout, face ordering within tiles, and page definitions
   **Key Information**: Tiles are 32x32 with 16x16 faces in row-major face order. Row-major layout has one page per row. Interleaved distribution is round-robin across banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how kernels address tensor pages in memory
   **Key Information**: TensorAccessor maps page IDs to physical bank addresses. Supports `get_noc_addr(page_id)` for full-page access and `get_noc_addr(page_id, byte_offset)` for sub-page access. Created on device side from compile-time args via `TensorAccessorArgs<base_idx>()`.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize compute library
   **Key Information**: Provides `compute_kernel_lib::untilize<block_width, input_cb, output_cb>(num_blocks)` with automatic dispatch based on DEST capacity and data type. Uses `DEST_AUTO_LIMIT` from `dest_helpers.hpp` (8 or 16 tiles depending on sync mode and accumulation mode). Supports WaitBlock (per-row) and WaitUpfront modes.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity limits that determine compute dispatch
   **Key Information**: DEST capacity: Full-sync 16-bit = 16 tiles, Full-sync 32-bit = 8 tiles, Half-sync 16-bit = 8 tiles, Half-sync 32-bit = 4 tiles. Detected automatically from JIT-generated headers.

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding how tile-rows are distributed across cores
   **Key Information**: `split_blocks_for_tilize(grid_size, nblocks)` returns `BlockSplit{ncores, all_cores, core_range, cliff_core_range, nblocks_per_core, nblocks_per_core_cliff}`. The cliff core handles `nblocks % nblocks_per_core` remaining blocks.

6. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper
   **Key Information**: Wraps `CircularBufferConfig` creation and `CreateCircularBuffer` into a single call. Accepts optional `Buffer*` for sharded input (sets globally allocated address).
