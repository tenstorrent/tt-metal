# Untilize Multi-Core Implementation Analysis

## Overview

The **untilize** operation converts tensor data from **TILE_LAYOUT** (32x32 tiles organized in faces) to **ROW_MAJOR_LAYOUT** (consecutive row elements). This is a critical data movement operation for preparing device tensors for host interaction or for operations that require row-major data access patterns.

**Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (row of tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (one full tile row of the tensor) |
| **Total units** | `num_tiles_per_col` (tensor_height / tile_height) |
| **Loop structure** | Outer loop: blocks (tile rows), Inner loop: tiles within block |

A **work unit** in this operation is a single **input block**, which corresponds to one row of tiles (i.e., `tile_height` rows of elements spanning the tensor width). Each core processes multiple input blocks, with the number depending on the core distribution strategy.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Any N-dimensional tensor |
| **Dimension convention** | Flattened to 2D: [tensor_height x tensor_width] |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with 16x16 faces) |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

For **sharded input**:
- **Shard Shape**: `[input_shard_height, input_shard_width]`
- **Core Grid**: From `input_shard_spec.grid`
- **Shard Orientation**: ROW_MAJOR or COL_MAJOR

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input (minus tile padding if applicable) |
| **Dimension convention** | Flattened to 2D: [tensor_height x tensor_width] |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (WIDTH/BLOCK) |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input |

For **sharded output** (WIDTH_SHARDED or BLOCK_SHARDED):
- Output stick size is divided by `output_num_blocks_across_width`
- Writer handles page-level addressing with byte offsets within pages

### Layout Transformation

The core transformation is:

```
TILE_LAYOUT (32x32 tiles with faces):
  Face0 (16x16) | Face1 (16x16)
  Face2 (16x16) | Face3 (16x16)

      |
      v  (untilize)

ROW_MAJOR_LAYOUT:
  Row 0: [element_0, element_1, ..., element_width-1]
  Row 1: [element_0, element_1, ..., element_width-1]
  ...
```

The untilize operation unpacks tile data from face-organized storage into contiguous row storage. This involves reordering elements from the tile's internal face structure to sequential row order.

## Data Flow Pattern

### Step-by-Step Flow

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (tiled) | cb_src0 | reserve_back, push_back (interleaved) OR just push_back (sharded) |
| 2 | Compute | cb_src0 | cb_output | wait_front, pop_front, reserve_back, push_back |
| 3 | Writer | cb_output | DRAM/L1 (row-major) | wait_front, pop_front |

### Detailed Flow

1. **Reader Kernel** (RISCV_0):
   - **Interleaved input**: Reads tiles from DRAM using `TensorAccessor`, one tile at a time with `noc_async_read`
   - **Sharded input**: Simply pushes the entire shard to CB (data already in L1)

2. **Compute Kernel** (RISCV_2):
   - Waits for a row of tiles (`num_tiles_per_input_block`)
   - Executes untilize operation (pack_untilize or standard untilize)
   - Produces row-major output to output CB

3. **Writer Kernel** (RISCV_1):
   - Waits for untilized block
   - Writes row-major data to output, handling:
     - Multiple output blocks per row (for width/block sharding)
     - Byte-level offsets within output pages
     - Uneven shard handling (padding removal)

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tiles (tiled format) | See below | `num_tiles_per_input_block` | Conditional | Reader | Compute | Block |
| c_16 | cb_output | Output data (row-major) | See below | `num_tiles_per_input_block` | Conditional | Compute | Writer | Block |

### CB Capacity Logic

**Input CB (c_0)**:
```cpp
if (input_is_sharded) {
    // Entire shard at once - no pipelining with reader
    input_cb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core;
} else {
    if (num_input_blocks_per_full_core == 1) {
        // Single block - no double buffering needed
        input_cb_num_tiles = num_tiles_per_input_block;
    } else {
        // Multiple blocks - double buffer for reader/compute overlap
        input_cb_num_tiles = num_tiles_per_input_block * 2;
    }
}
```

**Output CB (c_16)**:
```cpp
if (num_input_blocks_per_full_core == 1) {
    // Single block - no double buffering
    output_cb_num_tiles = num_tiles_per_input_block;
} else {
    // Multiple blocks - double buffer for compute/writer overlap
    output_cb_num_tiles = num_tiles_per_input_block * 2;
}
```

## Pipeline Pattern Summary

| Scenario | Input CB | Output CB | Pipeline Behavior |
|----------|----------|-----------|-------------------|
| Sharded input, 1 block | Entire shard (single) | Single | No overlap - sequential |
| Sharded input, N blocks | Entire shard (single) | Double | Compute/Writer overlap |
| Interleaved, 1 block | Single | Single | No overlap - sequential |
| Interleaved, N blocks | Double | Double | Reader/Compute AND Compute/Writer overlap |

**Design Decision**: Sharded input uses a single large CB for the entire shard because:
1. Data is already in L1 - no need for reader pipelining
2. Reduces CB overhead by avoiding unnecessary buffering

## Index Calculations

### Tile Index to Memory Address

The operation uses `TensorAccessor` for mapping logical tile indices to physical DRAM addresses:

```cpp
// Reader kernel (interleaved)
constexpr auto src_args = TensorAccessorArgs<1>();
const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
    uint64_t noc_read_addr = get_noc_addr(page_id, s);
    // ... read tile
}
```

### Block Index Calculations

**Height-wise block start index**:
```cpp
uint32_t height_wise_input_block_start_index =
    (core_index / num_input_blocks_across_width) * num_input_blocks_per_full_core;
```

**Width-wise block index**:
```cpp
uint32_t width_wise_input_block_index = core_index % num_input_blocks_across_width;
```

### Output Page Index Calculation (Writer)

```cpp
// Output page for a given row within a block
uint32_t num_rows_already_processed = block_height_index * tile_height + j;
uint32_t num_pages_already_processed_in_previous_rows =
    num_rows_already_processed * num_output_blocks_across_width;
uint32_t output_page_id =
    num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;
```

## Memory Access Patterns

### Read Pattern (Reader Kernel)

**Interleaved Input**:
- **Pattern**: Sequential tile reads
- **Access**: One tile at a time from DRAM
- **Blocking**: `noc_async_read_barrier()` after each tile (conservative approach)

**Sharded Input**:
- **Pattern**: No actual reads - data already in L1
- **Access**: CB push only (makes shard visible to compute)

### Write Pattern (Writer Kernel)

- **Pattern**: Row-by-row within each block
- **Granularity**: Sub-page writes possible (byte-level addressing)
- **Complexity**: Handles input/output block width mismatches

The writer supports writing partial rows when input and output sharding differ:
```cpp
while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
    uint32_t num_cols_to_write = std::min(
        num_unpadded_cols_per_input_block - num_input_cols_processed,
        num_cols_remaining_in_current_output_block);
    // Write partial row to output page
    noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write);
    // Advance to next output block if needed
}
```

## Core Distribution Strategy

| Attribute | Interleaved Input | Sharded Input |
|-----------|-------------------|---------------|
| **Grid topology** | 1D (linearized) | Matches shard grid |
| **Grid dimensions** | `ncores` x 1 | From `input_shard_spec.grid` |
| **Total cores** | `num_compute_cores` | `input_shard_spec.grid.num_cores()` |
| **Work per core** | `num_input_blocks_per_full_core` blocks | Entire shard |
| **Load balancing** | Full cores + optional cliff | Even (shard-based) |

### Interleaved Work Distribution

Uses `split_blocks_for_tilize()`:
```cpp
auto [num_compute_cores, compute_core_range, full_compute_core_range,
      cliff_compute_core_range, num_rows_per_full_core, num_rows_per_cliff_core]
    = ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);
```

**Cliff Core Handling**:
- A "cliff core" handles the remainder blocks when `num_tiles_per_col % blocks_per_core != 0`
- Separate kernel instance created for cliff core with different block count
- Cliff core always last in the linearized core order

### Sharded Work Distribution

```cpp
if (input_is_sharded) {
    num_compute_cores = input_shard_spec.grid.num_cores();
    compute_core_range = input_shard_spec.grid;
    full_compute_core_range = input_shard_spec.grid;
    cliff_compute_core_range = CoreRangeSet();  // No cliff for sharded

    num_input_blocks_per_full_core = input_shard_height / tile_height;
    num_input_blocks_per_cliff_core = 0;  // Never a cliff core
}
```

## Arguments

### Compile-Time Arguments

**Reader Kernel (Interleaved)**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer index |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor configuration |

**Reader Kernel (Sharded)**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer index |

**Writer Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer index |
| 1 | output_stick_size | uint32_t | Size of one output row in bytes |
| 2 | tile_height | uint32_t | Height of tile (typically 32) |
| 3 | num_tiles_per_input_block | uint32_t | Tiles per row (tensor width / tile width) |
| 4 | num_output_blocks_across_width | uint32_t | Output sharding factor |
| 5 | output_element_size | uint32_t | Bytes per element |
| 6 | num_cols_per_input_block | uint32_t | Elements per input block row |
| 7 | num_cols_per_output_block | uint32_t | Elements per output block row |
| 8+ | TensorAccessorArgs | uint32_t[] | Tensor accessor for output |

**Compute Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Tiles per block (controls untilize width) |
| 1 | src_cb_id | uint32_t | Input CB index |
| 2 | out_cb_id | uint32_t | Output CB index |

### Runtime Arguments

**Reader Kernel (Interleaved)**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer address |
| 1 | num_tiles | uint32_t | Total tiles to read |
| 2 | start_page_id | uint32_t | Starting tile index |

**Reader Kernel (Sharded)**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_core | uint32_t | Tiles in this core's shard |

**Writer Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer address |
| 1 | num_input_blocks_to_process | uint32_t | Blocks this core processes |
| 2 | height_wise_input_block_start_index | uint32_t | Starting block index |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Valid columns (handles padding) |
| 4 | width_wise_output_block_start_index | uint32_t | Output block start |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Offset within first output block |

**Compute Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id.cpp (interleaved) | RISCV_0 | NOC0 | DRAM (tiled) | cb_src0 | Sequential tile reads |
| reader_unary_sharded.cpp (sharded) | RISCV_0 | NOC0 | L1 (shard) | cb_src0 | CB push only |

**File**:
- Interleaved: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- Sharded: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

**Key Logic**:
- Interleaved: Uses `TensorAccessor` for address calculation, reads tiles sequentially
- Sharded: Simply pushes existing shard tiles to CB (`cb_push_back` only)

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| untilize_variable_num_blocks.cpp | RISCV_2 | N/A | cb_src0 | cb_output | Standard untilize |
| pack_untilize_variable_num_blocks.cpp | RISCV_2 | N/A | cb_src0 | cb_output | Pack untilize (faster) |

**File**:
- Standard: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- Pack: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`

**Key Logic** (from `untilize_helpers.hpp`):
- Uses unified `compute_kernel_lib::untilize<>()` function
- Automatically selects optimal path:
  - **Pack untilize** (hardware-accelerated): When width <= DEST limit
  - **Block-based pack untilize**: When width > DEST limit AND integer type
  - **Standard untilize** (fallback): When width > DEST limit AND float type

**Kernel Selection**:
```cpp
if (!use_pack_untilize || a.dtype() == DataType::UINT16 ||
    (a.dtype() == DataType::FLOAT32 && num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH)) {
    // Standard untilize
    compute_kernel = "untilize_variable_num_blocks.cpp";
} else {
    // Fast pack untilize
    compute_kernel = "pack_untilize_variable_num_blocks.cpp";
}
```

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_multi_core.cpp | RISCV_1 | NOC1 | cb_output | DRAM/L1 | Row-by-row writes |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Key Logic**:
- Processes one block at a time
- For each block, iterates through `tile_height` rows
- Handles input/output block width mismatches with partial writes
- Uses `TensorAccessor` for output addressing with byte-level offsets

## Implementation Notes

### Design Decisions

1. **Two Compute Kernel Variants**: Pack untilize is faster but has limitations (doesn't support UINT16, limited FLOAT32 width). The factory automatically selects the appropriate kernel.

2. **Unified Helper Library**: Both compute kernels use `untilize_helpers.hpp` which provides automatic dispatch based on:
   - DEST register capacity (4-16 tiles based on sync/accum mode)
   - Data type (integer vs float)
   - Tile width

3. **Sharded Input Optimization**: When input is sharded, the reader kernel does minimal work (just CB push), allowing compute to start immediately on locally available data.

4. **Flexible Output Sharding Support**: Writer handles complex scenarios where input and output have different sharding patterns, including partial page writes.

5. **Cliff Core Pattern**: For interleaved input, uses the standard cliff core pattern for handling remainders, with a separate kernel instance to avoid runtime conditionals.

### Pain Points and Caveats

1. **CB Sizing Complexity**: The conditional logic for CB capacity (single vs double buffering, sharded vs interleaved) adds complexity. Incorrect sizing can cause deadlocks or inefficiency.

2. **Uneven Shard Handling**: Both width-wise and height-wise uneven sharding requires special handling in runtime args. The code carefully calculates `num_unpadded_cols_per_input_block` and `num_input_blocks_to_process` for edge shards.

3. **FLOAT32 Limitations**: Float32 with wide tiles cannot use pack_untilize, falling back to slower standard untilize. This is noted in code comments as a known limitation (issues #30400, #33795).

4. **TensorAccessor Configuration**: Both reader (interleaved) and writer use TensorAccessorArgs which must be correctly configured. The args are appended to compile-time args, making the arg indices dependent on TensorAccessor state.

5. **Writer Page Complexity**: The writer's handling of input/output block mismatches involves nested loops and multiple address calculations, making it one of the more complex kernels.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the untilize operation in tt-metal? What does it do and what are the different methods of untilizing (standard untilize vs pack_untilize)?"
   **Reason**: Needed to understand the fundamental operation being performed and the two implementation approaches.
   **Key Findings**: Untilize converts TILE_LAYOUT to ROW_MAJOR_LAYOUT. Pack untilize is hardware-accelerated and preferred but has data type limitations (not UINT16, limited FLOAT32 width).

2. **Query**: "What is TensorAccessor in tt-metal and how does it work for mapping tensor indices to physical memory addresses?"
   **Reason**: The reader and writer kernels use TensorAccessor extensively for address calculation.
   **Key Findings**: TensorAccessor abstracts physical memory layout, providing `get_noc_addr(page_id)` for address calculation. TensorAccessorArgs packs configuration into compile-time/runtime args.

3. **Query**: "What is TILE_LAYOUT vs ROW_MAJOR_LAYOUT in tt-metal?"
   **Reason**: Core understanding needed for what the operation transforms.
   **Key Findings**: TILE_LAYOUT uses 32x32 tiles with 16x16 faces stored contiguously. ROW_MAJOR_LAYOUT stores consecutive row elements together.

4. **Query**: "What is the split_work_to_cores function and cliff core concept?"
   **Reason**: The program factory uses `split_blocks_for_tilize` which has cliff core handling.
   **Key Findings**: Cliff cores handle remainder work when blocks don't divide evenly. The split function returns separate core ranges for full cores and cliff cores.

5. **Query**: "What is the create_cb function in tt-metal and what do CB indices like c_0 and c_16 mean?"
   **Reason**: Needed to understand circular buffer creation and indexing.
   **Key Findings**: CB indices identify specific buffers (up to 32 per core). c_0 is commonly used for input, c_16 for output. Indices are arbitrary but conventional.

6. **Query**: "What is the compute_kernel_hw_startup function?"
   **Reason**: Both compute kernels call this function at startup.
   **Key Findings**: Initializes UNPACK, MATH, and PACK hardware units. Must be called once at kernel start with correct CB IDs.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Deep understanding of tile layout, faces, and memory organization.
   **Key Information**: Tiles are 32x32 with 16x16 faces in row-major order (face0->face1->face2->face3).

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor usage patterns.
   **Key Information**: Host creates TensorAccessorArgs from Buffer, device creates TensorAccessor from args + base address + page size.

3. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the BlockSplit structure and split_blocks_for_tilize function.
   **Key Information**: Returns ncores, core_range, cliff_core_range, blocks_per_core, blocks_per_cliff.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
   **Reason**: Understanding the compute kernel's unified untilize dispatch logic.
   **Key Information**: Automatic dispatch based on DEST limit and data type. Three paths: pack untilize, block-based pack untilize, standard untilize.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity detection.
   **Key Information**: DEST capacity is 4-16 tiles depending on sync mode (half/full) and accumulation mode (16-bit/32-bit).
