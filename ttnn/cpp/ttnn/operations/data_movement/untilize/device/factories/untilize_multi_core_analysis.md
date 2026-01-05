# Untilize Multi-Core Implementation Analysis

## Overview

The **untilize** operation converts tensor data from **TILE_LAYOUT** (32x32 tiles) back to **ROW_MAJOR_LAYOUT** (linear row-major format). This is the inverse of the tilize operation and is essential when tensor data processed in hardware-native tile format needs to be read in standard row-major format.

**Program Factory Path**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

This analysis focuses on the **multi-core interleaved** variant, which supports both interleaved and sharded input/output combinations.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (row of tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (tiles per row of tensor) |
| **Total units** | `num_tiles_per_col` blocks (one block per tile row in height) |
| **Loop structure** | Outer: blocks (rows of tiles), Inner: tiles within block |

A **work unit** is one horizontal row of tiles (one "block"). For a tensor with dimensions that result in `num_tiles_per_row` tiles horizontally and `num_tiles_per_col` tiles vertically, each block contains `num_tiles_per_row` tiles representing one tile-height (32 rows) of the original tensor.

The untilize operation processes each block by:
1. Reading all tiles in the block from DRAM (or L1 for sharded)
2. Converting each tile from tile format to row-major format via compute kernel
3. Writing the resulting row-major data back to DRAM (or L1 for sharded)

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, H, W, C] or any shape that flattens to 2D |
| **Dimension convention** | Last dimension is width (contiguous in memory) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input |

### Layout Transformation

The core transformation converts tile-ordered data to row-major:
- **Input**: Data stored as 32x32 tiles, with each tile containing 4 faces (16x16 sub-tiles)
- **Output**: Data stored as contiguous rows, where each row is `tensor_width * element_size` bytes

For a tile at position (row_tile, col_tile):
- Input: Tile data is stored contiguously in tile format (face0, face1, face2, face3)
- Output: The 32 rows within this tile are scattered across 32 consecutive output rows, with horizontal offset `col_tile * 32 * element_size`

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) or L1 (sharded) | CB_in (c_0) | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB_in (c_0) | CB_out (c_16) | `cb_wait_front`, `untilize/pack_untilize`, `cb_pop_front`, `cb_push_back` |
| 3 | Writer | CB_out (c_16) | DRAM (interleaved) or L1 (sharded) | `cb_wait_front`, `noc_async_write`, `cb_pop_front` |

### Detailed Flow (Interleaved Input/Output)

1. **Reader Kernel** (`reader_unary_start_id.cpp`):
   - Uses `TensorAccessor` to map logical tile IDs to physical DRAM addresses
   - Reads tiles sequentially from `start_page_id` to `start_page_id + num_tiles - 1`
   - For each tile: reserve CB space, async read from DRAM, wait for completion, push to CB

2. **Compute Kernel** (`pack_untilize_variable_num_blocks.cpp` or `untilize_variable_num_blocks.cpp`):
   - Uses unified `compute_kernel_lib::untilize<>()` helper function
   - Automatically selects between:
     - **pack_untilize**: Hardware-accelerated path when tile width fits in DEST registers
     - **block-based pack_untilize**: For wide integer types exceeding DEST limit
     - **standard untilize**: Fallback for wide non-integer types
   - Processes `per_core_block_cnt` blocks (tile rows)

3. **Writer Kernel** (`writer_unary_stick_layout_split_rows_multi_core.cpp`):
   - Converts tile-based output to stick (row) based writes
   - For each block of `tile_height` (32) rows:
     - Reads row-major data from output CB
     - Writes each row to correct position in output tensor
   - Handles uneven sharding when input/output have different shard widths

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `input_cb_num_tiles` tiles | `num_tiles_per_input_block` tiles | Double (if >1 block) | Reader | Compute | Block |
| c_16 | cb_output | Output row-major staging | `output_cb_num_tiles` tiles | `num_tiles_per_input_block` tiles | Double (if >1 block) | Compute | Writer | Block |

### Capacity Calculation

**Input CB (c_0)**:
- **Sharded input**: `num_tiles_per_input_block * num_input_blocks_per_full_core` (entire shard)
- **Interleaved input, single block**: `num_tiles_per_input_block` (single-buffered)
- **Interleaved input, multiple blocks**: `num_tiles_per_input_block * 2` (double-buffered)

**Output CB (c_16)**:
- **Single block**: `num_tiles_per_input_block` (single-buffered)
- **Multiple blocks**: `num_tiles_per_input_block * 2` (double-buffered)

The double-buffering enables overlap between reader/compute and compute/writer when processing multiple blocks.

## Pipeline Pattern Summary

| Path | Condition | Buffering | Overlap |
|------|-----------|-----------|---------|
| Interleaved, multi-block | `num_input_blocks_per_full_core > 1` | Double | Reader/Compute, Compute/Writer |
| Interleaved, single-block | `num_input_blocks_per_full_core == 1` | Single | None (sequential) |
| Sharded | Always | Single (full shard in CB) | None (all data pre-loaded) |

## Index Calculations

### Tile ID to Physical Address (Reader)

The reader uses `TensorAccessor` for interleaved tensors:
```cpp
const auto s = TensorAccessor(src_args, src_addr, tile_bytes);
uint64_t noc_read_addr = get_noc_addr(page_id, s);
```

`TensorAccessor` internally computes:
1. **Bank ID**: `page_id % num_banks` (round-robin distribution)
2. **Bank offset**: `(page_id / num_banks) * page_size`
3. **NOC address**: Physical address combining bank coordinates and offset

### Output Row Indexing (Writer)

The writer kernel maps from untilized CB data to output tensor rows:

```cpp
// For each block at height_wise_input_block_index:
uint32_t num_rows_already_processed = block_height_index * tile_height + j;
uint32_t output_page_id = num_rows_already_processed * num_output_blocks_across_width + width_wise_output_block_start_index;
```

Key variables:
- `height_wise_input_block_start_index`: Starting block row for this core
- `width_wise_output_block_start_index`: Starting output page column for this core
- `num_cols_already_processed_in_first_output_block`: Byte offset within first output page

## Memory Access Patterns

### Read Pattern (Reader Kernel)

| Attribute | Value |
|-----------|-------|
| **Pattern** | Sequential tile reads |
| **Ordering** | Row-major across tile grid (tile 0, 1, 2, ... per row, then next row) |
| **Granularity** | One tile per read |
| **Memory type** | DRAM (interleaved) or L1 (sharded) |
| **Synchronization** | Barrier after each tile read |

### Write Pattern (Writer Kernel)

| Attribute | Value |
|-----------|-------|
| **Pattern** | Row-by-row writes within each tile block |
| **Ordering** | Process each of 32 rows in a block, then next block |
| **Granularity** | One row (stick) per write, possibly split across multiple output pages |
| **Memory type** | DRAM (interleaved) or L1 (sharded) |
| **Synchronization** | Barrier after all rows in a block written |

The writer handles complex scenarios where input and output shard widths differ, requiring writes to span multiple output pages per input row.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear core assignment) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_input_blocks_per_full_core` blocks (tile rows) |
| **Load balancing** | Nearly equal with optional cliff core |
| **Remainder handling** | Single cliff core processes `num_input_blocks_per_cliff_core` blocks |

### Work Distribution

The `split_blocks_for_tilize()` function distributes tile rows across cores:

```cpp
auto [num_compute_cores, compute_core_range, full_compute_core_range,
      cliff_compute_core_range, num_rows_per_full_core, num_rows_per_cliff_core]
    = ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);
```

- **Full cores**: Each processes `num_rows_per_full_core` blocks
- **Cliff core** (if any): Processes remaining `num_rows_per_cliff_core` blocks
- **Core ordering**: Row-major across the compute grid

### Sharded Input Handling

For sharded inputs, the work distribution follows the shard specification:
- Each core processes its local shard
- `num_input_blocks_across_width` accounts for width-sharded tensors
- Uneven shards are handled via `num_unpadded_cols_per_input_block` and `num_input_blocks_to_process` adjustments

## Arguments

### Compile-Time Arguments

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |
| 1+ | TensorAccessorArgs | varies | Tensor accessor configuration (rank, banks, etc.) |

#### Reader Kernel (Sharded)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer ID (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output row in bytes |
| 2 | tile_height | uint32_t | Height of tile (32) |
| 3 | num_tiles_per_input_block | uint32_t | Tiles per row (width in tiles) |
| 4 | num_output_blocks_across_width | uint32_t | Output pages per tensor row |
| 5 | output_element_size | uint32_t | Bytes per element |
| 6 | num_cols_per_input_block | uint32_t | Elements per input block width |
| 7 | num_cols_per_output_block | uint32_t | Elements per output page width |
| 8+ | TensorAccessorArgs or ShardingArgs | varies | Output addressing configuration |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Tiles per block (width in tiles) |
| 1 | src_cb_id | uint32_t | Input CB ID (c_0) |
| 2 | out_cb_id | uint32_t | Output CB ID (c_16) |

### Runtime Arguments

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_tiles | uint32_t | Total tiles to read |
| 2 | start_page_id | uint32_t | First tile ID for this core |

#### Reader Kernel (Sharded)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_core | uint32_t | Tiles in this core's shard |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_input_blocks_to_process | uint32_t | Blocks (tile rows) for this core |
| 2 | height_wise_input_block_start_index | uint32_t | First block row index |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Valid columns (handles padding) |
| 4 | width_wise_output_block_start_index | uint32_t | First output page column |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Column offset in first output page |
| 6+ | ShardingArgs (if sharded output) | varies | Shard mapping table |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks for this core |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM | CB_in (c_0) | Sequential tile reads |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

**Key Logic**:
- Simple loop reading `num_tiles` tiles starting from `start_page_id`
- Uses `TensorAccessor` for address translation
- One tile at a time with barrier after each read
- For sharded input, uses `reader_unary_sharded.cpp` which simply pushes pre-loaded data

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| pack_untilize_variable_num_blocks | RISCV_2 | N/A | CB_in (c_0) | CB_out (c_16) | Untilize tiles to row-major |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`

**Key Logic**:
- Uses `compute_kernel_lib::untilize<>()` unified helper
- Automatically selects optimal path based on width and data type:
  - **pack_untilize**: Width <= DEST limit (most efficient, hardware-accelerated)
  - **block-based pack_untilize**: Wide integer types (splits into DEST-sized chunks)
  - **standard untilize**: Wide non-integer types (fallback)
- DEST register limit auto-detected via `dest_helpers.hpp` (4-16 tiles based on sync/accum mode)

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_multi_core | RISCV_1 | NOC1 | CB_out (c_16) | DRAM | Row-by-row writes |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Key Logic**:
- Processes one block (32 rows) at a time
- For each row within a block:
  - Calculates output page ID based on row index and width partitioning
  - Handles partial first page writes when input/output shard widths differ
  - May split a single input row across multiple output pages
- Uses `TensorAccessor` or `ShardedAddrGen` for output addressing

## Implementation Notes

### Compute Kernel Selection

The program factory selects between two compute kernels:

1. **pack_untilize_variable_num_blocks.cpp** (default, "fast"):
   - Used when `use_pack_untilize` is true and width is within limits
   - Hardware-accelerated untilization during pack phase
   - Required for UINT32, INT32 data types

2. **untilize_variable_num_blocks.cpp** ("slow"):
   - Used when:
     - `use_pack_untilize` is false
     - Data type is UINT16
     - FLOAT32 with width >= `MAX_PACK_UNTILIZE_WIDTH`
   - Standard untilize via unpack phase

Both kernels use the unified `compute_kernel_lib::untilize<>()` helper which internally dispatches to the appropriate implementation.

### DEST Register Considerations

For 32-bit accumulation mode (`DST_ACCUM_MODE=true`), the DEST register capacity is halved:
- SyncFull + 32-bit: 8 tiles
- SyncHalf + 32-bit: 4 tiles

The untilize helper automatically detects this and adjusts block sizes accordingly.

### Sharded Tensor Handling

When input is sharded:
- Reader kernel simply pushes pre-allocated shard data (no DMA needed)
- CB is backed by the shard's L1 memory via `set_globally_allocated_address()`
- Each core processes its local shard independently
- Uneven shards (last shard in row/column) have adjusted processing counts

### Output Page Complexity

The writer kernel handles the case where input and output shards may have different widths:
- A single input block row may span multiple output pages
- First output page may require a byte offset (`num_cols_already_processed_in_first_output_block`)
- Subsequent pages start at offset 0

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the untilize operation in TTNN and how does it convert tile-layout data back to row-major format?"
   **Reason**: Understanding the fundamental operation being analyzed
   **Key Findings**: Untilize converts 32x32 tile data to row-major format. Two main methods: standard untilize (via unpack) and pack_untilize (hardware-accelerated during pack phase). pack_untilize is preferred when possible.

2. **Query**: "What is the split_blocks_for_tilize function and how does it distribute work across multiple cores?"
   **Reason**: Understanding core distribution strategy
   **Key Findings**: Distributes tile rows (blocks) across cores with load balancing. Returns full core range, cliff core range, and blocks per core. Handles remainder via single cliff core.

3. **Query**: "What is TensorAccessor in tt-metal and how does it map logical tensor indices to physical memory addresses?"
   **Reason**: Understanding memory addressing in reader/writer kernels
   **Key Findings**: TensorAccessor abstracts bank distribution, computing NOC addresses from logical page IDs. TensorAccessorArgs configures host-side parameter passing.

4. **Query**: "What are circular buffers in tt-metal? What is the difference between single-buffering, double-buffering, and multi-buffering?"
   **Reason**: Understanding CB configuration choices
   **Key Findings**: CBs are L1 staging areas. Double-buffering overlaps producer/consumer. Capacity determines buffering depth; double-buffer uses 2x block size.

5. **Query**: "What do the compute kernel API functions untilize_init, untilize_block, pack_untilize_init, and pack_untilize_block do?"
   **Reason**: Understanding compute kernel implementation details
   **Key Findings**: untilize_block processes tiles via unpack path. pack_untilize_block is more efficient, converting directly from DEST to row-major in L1. Template parameters control block and full width.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tile vs row-major layout differences
   **Key Information**: Tiles are 32x32, split into 4 faces (16x16). Row-major has one row per page. Interleaved distributes pages round-robin across banks.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
   **Reason**: Understanding the unified untilize helper implementation
   **Key Information**: Single `untilize<>()` function auto-dispatches based on width and data type. Detects DEST limit from JIT headers. Block-based pack_untilize for wide integers.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity detection
   **Key Information**: DEST capacity varies 4-16 tiles based on DST_SYNC_MODE and DST_ACCUM_MODE. Auto-detected from JIT-generated headers.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding work distribution algorithm
   **Key Information**: `split_blocks_for_tilize()` returns BlockSplit struct with core ranges and blocks per core. Handles cliff cores for remainder blocks.
