# Untilize Multi-Core Implementation Analysis

## Overview

The untilize operation converts data from tiled format (32x32 tiles) to row-major (stick) format. This analysis focuses on the multi-core interleaved memory variant, which is the primary pattern for writing computed results back to DRAM in row-major layout.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Key Purpose for Fused LayerNorm**: This analysis serves as the "output_stage" reference for understanding how to:
- Untilize computed results from tiled to row-major format
- Write row-major data to interleaved DRAM
- Distribute untilize work across multiple cores

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (row of tiles) |
| **Unit size** | `num_tiles_per_row` tiles (one tile row) |
| **Total units** | `num_tiles_per_col` (number of tile rows) |
| **Loop structure** | Each core processes `num_input_blocks_per_core` blocks; cliff core may process fewer |

A "block" in the untilize context represents one tile-row: `num_tiles_per_row` tiles horizontally, representing `tile_height` (32) rows of elements. The compute kernel processes one block at a time, converting tiled data to row-major format.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] or flattened |
| **Dimension convention** | Last dimension is width |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16, etc. |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED (focus of this analysis) |
| **Buffer type** | DRAM |
| **Data type** | Same as input |

### Layout Transformations

The core transformation converts tile-layout data to row-major "sticks":
- **Input**: Tiles stored as 32x32 contiguous blocks, tiles arranged in row-major order
- **Output**: Rows (sticks) of elements, each stick is `tensor_width * element_size` bytes
- **Stick size**: `tensor_width / num_output_blocks_across_width * element_size`

For interleaved output, each stick becomes one page distributed across DRAM banks.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved tiles) | CB_in (c_0) | reserve_back(1), push_back(1) per tile |
| 2 | Compute | CB_in (c_0) | CB_out (c_16) | wait_front(block), pop_front(block), reserve_back(block), push_back(block) |
| 3 | Writer | CB_out (c_16) | DRAM (interleaved sticks) | wait_front(block), pop_front(block) |

### Detailed Flow

1. **Reader Kernel** (`reader_unary_start_id.cpp`):
   - Reads tiles sequentially from interleaved DRAM starting at `start_page_id`
   - Uses `TensorAccessor` for address generation
   - Pushes tiles one at a time to CB_in
   - Processes `num_tiles_to_read = num_tiles_per_row * num_blocks_to_process` tiles

2. **Compute Kernel** (`pack_untilize_variable_num_blocks.cpp` or `untilize_variable_num_blocks.cpp`):
   - Waits for one block (row of tiles) in CB_in
   - Converts tiled data to row-major in DEST registers
   - Packs row-major data to CB_out
   - Uses unified `compute_kernel_lib::untilize<>()` helper

3. **Writer Kernel** (`writer_unary_stick_layout_split_rows_multi_core.cpp`):
   - Waits for one block of untilized data in CB_out
   - Extracts `tile_height` rows (sticks) from the block
   - Writes each stick to appropriate DRAM page
   - Handles partial output blocks (when input/output shard widths differ)

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tiles (tiled) | 1-2 blocks | 1 block (`num_tiles_per_row`) | Single/Double | Reader | Compute | Block |
| c_16 | cb_output | Output tiles (row-major) | 1-2 blocks | 1 block (`num_tiles_per_row`) | Single/Double | Compute | Writer | Block |

### Buffering Strategy

```cpp
// Input CB sizing (lines 93-105 of program factory)
if (num_input_blocks_per_full_core == 1) {
    input_cb_num_tiles = num_tiles_per_input_block;      // Single-buffered
} else {
    input_cb_num_tiles = num_tiles_per_input_block * 2;  // Double-buffered
}

// Output CB sizing (lines 116-123)
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;     // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2; // Double-buffered
}
```

**Key Insight**: Double-buffering is enabled only when a core processes 2+ blocks. This allows overlap between:
- Reader filling next block while Compute processes current block
- Compute filling next output block while Writer drains current block

## Pipeline Pattern Summary

| Configuration | Input CB | Output CB | Overlap Potential |
|---------------|----------|-----------|-------------------|
| Single block per core | Single | Single | None (sequential) |
| Multiple blocks per core | Double | Double | Full R-C and C-W overlap |

The double-buffering enables a pipelined execution where:
- Reader can prefetch the next block while Compute untilizes current
- Writer can drain current block while Compute fills next

## Index Calculations

### Reader: Tile Page ID Calculation

The reader uses `TensorAccessor` with compile-time args from `TensorAccessorArgs(*src0_buffer)`:

```cpp
// reader_unary_start_id.cpp
constexpr auto src_args = TensorAccessorArgs<1>();  // Compile-time offset 1
const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
    uint64_t noc_read_addr = get_noc_addr(page_id, s);  // Linear tile ID -> NOC addr
    // ... read tile
}
```

The `start_page_id` is computed per-core in the host factory to partition tile space.

### Writer: Stick Page ID Calculation

The writer computes output page IDs based on:
- `height_wise_input_block_start_index`: Which tile-row this core starts at
- `width_wise_output_block_start_index`: Which output column block to start writing
- `num_cols_already_processed_in_first_output_block`: Offset within first output page

```cpp
// writer_unary_stick_layout_split_rows_multi_core.cpp (lines 49-54)
uint32_t num_rows_already_processed = block_height_index * tile_height + j;
uint32_t num_pages_already_processed_in_previous_rows =
    num_rows_already_processed * num_output_blocks_across_width;
uint32_t output_page_id =
    num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;
```

**Output Page Structure**:
- Each row of elements maps to `num_output_blocks_across_width` pages
- Page ID = `row_index * num_output_blocks_across_width + col_block_index`

## Memory Access Patterns

### Read Pattern (Reader Kernel)

| Attribute | Value |
|-----------|-------|
| **Pattern** | Sequential tile reads |
| **Memory** | DRAM (interleaved) |
| **Page size** | Tile size (typically 2KB for bfloat16) |
| **Stride** | Linear page IDs from `start_page_id` to `end_page_id` |
| **Barrier** | Per-tile (`noc_async_read_barrier()` after each tile) |

### Write Pattern (Writer Kernel)

| Attribute | Value |
|-----------|-------|
| **Pattern** | Strided stick writes |
| **Memory** | DRAM (interleaved) |
| **Page size** | Stick size (`tensor_width * element_size`) |
| **Access** | Row-by-row within each block |
| **Barrier** | Per-block (`noc_async_write_barrier()` after all sticks in block) |

**Write Algorithm**:
```cpp
for each input_block:
    cb_wait_front(cb_out, num_tiles_per_input_block)
    for j in 0..tile_height:
        // Calculate output_page_id for this row
        // Write row (stick) to DRAM
        noc_async_write(l1_read_addr, dst_noc_addr, num_bytes_to_write)
    noc_async_write_barrier()
    cb_pop_front(cb_out, num_tiles_per_input_block)
```

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear core assignment) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `num_compute_cores` (determined by work splitting) |
| **Work per core** | `num_rows_per_full_core` tile-rows |
| **Load balancing** | Nearly equal with optional cliff core |

### Work Splitting via `split_blocks_for_tilize`

The `ttnn::split_blocks_for_tilize()` function (from `work_split_tilize.hpp`) distributes tile-rows across cores:

```cpp
// Program factory lines 60-66
auto [num_compute_cores,
      compute_core_range,
      full_compute_core_range,
      cliff_compute_core_range,
      num_rows_per_full_core,
      num_rows_per_cliff_core] = ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);
```

**Algorithm**:
1. Calculate `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
2. Calculate `ncores = ceil(num_tiles_per_col / nblocks_per_core)`
3. If `num_tiles_per_col % nblocks_per_core != 0`, the last core is a "cliff core" with fewer blocks

**Cliff Core**: Handles remainder blocks when total blocks don't divide evenly. Created with separate kernel instance using same code but potentially different compile-time args.

### Per-Core Work Assignment

```cpp
// For full cores (lines 252-324):
tile_start_index = core_index * num_tiles_per_row * num_rows_per_full_core;
num_tiles_to_read = num_tiles_per_row * num_rows_per_full_core;

// For cliff core (lines 326-376):
tile_start_index = full_cores.size() * num_tiles_per_row * num_rows_per_full_core;
num_tiles_to_read = num_tiles_per_row * num_rows_per_cliff_core;
```

## Arguments

### Compile-Time Arguments

#### Reader Kernel (`reader_unary_start_id.cpp`)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |
| 1+ | TensorAccessorArgs | varies | Tensor accessor configuration for DRAM reads |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Tiles per block (`num_tiles_per_input_block`) |
| 1 | src_cb_id | uint32_t | Input CB ID (c_0) |
| 2 | out_cb_id | uint32_t | Output CB ID (c_16) |

#### Writer Kernel (`writer_unary_stick_layout_split_rows_multi_core.cpp`)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer ID (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output stick in bytes |
| 2 | tile_height | uint32_t | Height of tile (typically 32) |
| 3 | num_tiles_per_input_block | uint32_t | Tiles per block horizontally |
| 4 | num_output_blocks_across_width | uint32_t | Output pages per row |
| 5 | output_element_size | uint32_t | Size of one element in bytes |
| 6 | num_cols_per_input_block | uint32_t | Elements per input block horizontally |
| 7 | num_cols_per_output_block | uint32_t | Elements per output page horizontally |
| 8+ | TensorAccessorArgs | varies | Tensor accessor configuration for DRAM writes |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_tiles | uint32_t | Total tiles to read for this core |
| 2 | start_page_id | uint32_t | First tile page ID for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_input_blocks_to_process | uint32_t | Number of blocks to process |
| 2 | height_wise_input_block_start_index | uint32_t | Starting tile-row index |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Valid columns (excludes padding) |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page column |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Offset within first output page |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM tiles | CB_in (c_0) | Sequential tile reads |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

**Key Logic**:
- Uses `TensorAccessor` for bank-aware address generation
- Single-tile read loop with per-tile barrier
- Simple linear iteration from `start_page_id` to `end_page_id`

```cpp
for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
    cb_reserve_back(cb_id_in0, 1);
    uint64_t noc_read_addr = get_noc_addr(page_id, s);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    noc_async_read(noc_read_addr, l1_write_addr, tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, 1);
}
```

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| pack_untilize_variable_num_blocks | RISCV_2 | N/A | CB_in (c_0) | CB_out (c_16) | Untilize via pack |
| untilize_variable_num_blocks | RISCV_2 | N/A | CB_in (c_0) | CB_out (c_16) | Standard untilize |

**File (pack)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`

**File (standard)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`

**Key Logic**:
- Uses unified `compute_kernel_lib::untilize<>()` from `untilize_helpers.hpp`
- Automatically selects optimal path based on width and data type:
  - **pack_untilize**: Hardware-accelerated, preferred for widths <= DEST limit
  - **block-based pack_untilize**: For wide integer types, splits into DEST-sized blocks
  - **standard untilize**: Fallback for wide non-integer types

```cpp
compute_kernel_hw_startup(src_cb_id, out_cb_id);
compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt);
```

**Dispatch Selection** (in `untilize_helpers.hpp`):
```cpp
constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;  // 4-16 tiles based on sync/accum mode
constexpr bool is_integer = is_integer_format<icb_id>();

if (tile_width > dest_limit && is_integer) {
    // Block-based pack_untilize for wide integers
} else if (tile_width > dest_limit) {
    // Standard untilize for wide floats
} else {
    // Single-pass pack_untilize (optimal)
}
```

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_multi_core | RISCV_1 | NOC1 | CB_out (c_16) | DRAM sticks | Row-by-row writes |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Key Logic**:
- Processes one block at a time
- Extracts `tile_height` rows from untilized block
- Computes output page ID for each row
- Handles partial output pages (for sharded output variants)

```cpp
for (uint32_t i = 0; i < num_input_blocks_to_process; ++i) {
    cb_wait_front(cb_id_out0, num_tiles_per_input_block);
    for (uint32_t j = 0; j < tile_height; ++j) {
        // Calculate output_page_id and write stick
        uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
        noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, num_tiles_per_input_block);
}
```

## Implementation Notes

### Compute Path Selection

The program factory selects between two compute kernels based on conditions (lines 191-204):

```cpp
if (!use_pack_untilize || a.dtype() == DataType::UINT16 ||
    (a.dtype() == DataType::FLOAT32 && num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH)) {
    compute_kernel = "untilize_variable_num_blocks.cpp";  // Standard path
} else {
    compute_kernel = "pack_untilize_variable_num_blocks.cpp";  // Fast pack path
}
```

**Selection Criteria**:
- UINT16 always uses standard path (pack_untilize doesn't support it)
- FLOAT32 with wide rows uses standard path (DEST register limit)
- All other cases prefer pack_untilize for hardware acceleration

### DEST Register Limits

Pack untilize width is limited by DEST register capacity:
- **Half-sync + 16-bit**: 8 tiles max
- **Half-sync + 32-bit**: 4 tiles max
- **Full-sync + 16-bit**: 16 tiles max
- **Full-sync + 32-bit**: 8 tiles max

The unified `untilize_helpers.hpp` automatically handles this by detecting `DEST_AUTO_LIMIT` and dispatching appropriately.

### Output Stick Layout

For interleaved output:
- Each stick (row of elements) is one page
- Page size = `tensor_width * element_size`
- Pages distributed across DRAM banks via interleaved mapping
- Writer uses `TensorAccessor.get_noc_addr(page_id, offset)` for bank-aware addressing

### Key Differences from Tilize

Unlike tilize which reads row-major and writes tiles, untilize:
1. Reads tiles (32x32 blocks)
2. Unpacks to row-major in DEST registers
3. Packs row-major to output CB
4. Writes rows (sticks) to DRAM

The transformation happens in the compute kernel, not in the data movement kernels.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is pack_untilize and how does it differ from standard untilize in tt-metal?"
   **Reason**: Understanding the two untilize implementations and their trade-offs
   **Key Findings**:
   - pack_untilize is hardware-accelerated (80 cycles vs 390 cycles)
   - Limited by DEST register capacity (4-16 tiles based on mode)
   - Supports INT32/UINT32 which standard doesn't
   - Standard untilize used as fallback for wide non-integer types

2. **Query**: "How does TensorAccessor work in tt-metal kernels?"
   **Reason**: Understanding address generation for interleaved memory
   **Key Findings**:
   - Abstracts logical page ID to physical NOC address mapping
   - Handles bank distribution for interleaved tensors
   - TensorAccessorArgs manages compile-time vs runtime arg configuration
   - get_noc_addr() computes bank ID, offset, and NOC coordinates

3. **Query**: "What is split_blocks_for_tilize function in tt-metal?"
   **Reason**: Understanding core work distribution strategy
   **Key Findings**:
   - Distributes blocks evenly across available cores
   - Creates "cliff core" for remainder blocks
   - Returns core ranges for full cores and cliff core separately
   - Enables per-core kernel instance with tailored compile-time args

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host and device APIs
   **Key Information**:
   - TensorAccessorArgs configures which params are compile-time vs runtime
   - get_noc_addr() handles interleaved bank distribution
   - Zero-cost construction when rank is static

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
   **Reason**: Understanding unified untilize dispatch logic
   **Key Information**:
   - Automatic DEST limit detection via dest_helpers.hpp
   - Three paths: single-pass pack, block-based pack, standard
   - Data format detection for integer vs float dispatch

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity calculation
   **Key Information**:
   - Capacity depends on sync mode (Full vs Half) and accum mode (16-bit vs 32-bit)
   - 4/8/16 tiles depending on configuration
   - DEST_AUTO_LIMIT computed at compile time from JIT headers

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding work distribution algorithm
   **Key Information**:
   - BlockSplit struct contains core ranges and work counts
   - Cliff core handles remainder when blocks don't divide evenly
   - Used by both tilize and untilize operations
