# Untilize Operation Implementation Analysis

## Overview

The **untilize** operation converts tensor data from **TILE_LAYOUT** (hardware-native 32x32 tile format) to **ROW_MAJOR_LAYOUT** (sequential row-by-row storage). This is a fundamental data movement operation required when data processed by Tenstorrent hardware needs to be returned to formats expected by host CPUs or other systems.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp`

The untilize operation is implemented through **six distinct program factory functions**, each optimized for different input/output memory configurations:

| Function | Use Case |
|----------|----------|
| `untilize_single_core` | Single-core fallback for small tensors or when multicore is disabled |
| `untilize_multi_core` | Default multicore implementation with flexible sharding support |
| `untilize_multi_core_block` | Optimized 2D block parallelization for wide interleaved tensors |
| `untilize_multi_core_sub_core_grids` | Custom core grid specification for interleaved tensors |
| `untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical` | Optimized path when input/output have identical L1 sharding |
| `untilize_multi_core_parallelize_column` | Column-wise parallelization for single-tile-height wide tensors |

## Work Unit Definition

The fundamental work unit is a **block of tiles**. A block consists of:
- **Width**: `ntiles_per_block` tiles (typically 1-8 tiles, constrained by DEST register capacity)
- **Height**: 1 tile row (32 elements vertically)

Each block represents one iteration through the compute kernel's untilize loop. The compute kernel processes blocks by:
1. Waiting for input tiles in the input CB
2. Untilizing tile data to row-major format
3. Pushing results to the output CB

For the `pack_untilize` path (hardware-accelerated), the block width is constrained by `MAX_PACK_UNTILIZE_WIDTH` (8 tiles for FLOAT32, 16 otherwise) due to DEST register capacity.

## Tensor Format and Layout

### Input Tensor

| Attribute | Value |
|-----------|-------|
| **Logical Shape** | Any shape, typically `[N, C, H, W]` or `[batch, seq_len, hidden_dim]` |
| **Dimension Convention** | Last two dimensions are height and width; height/width must be tile-aligned |
| **Tensor Layout** | `TILE_LAYOUT` - data stored in 32x32 tiles |
| **Memory Layout** | `INTERLEAVED` (round-robin across DRAM banks) or `SHARDED` (HEIGHT/WIDTH/BLOCK) |
| **Buffer Type** | DRAM (interleaved) or L1 (sharded) |
| **Data Type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16, BFLOAT8_B |

**Tile Structure (32x32)**:
- Each tile contains 1024 elements
- Tiles are subdivided into four 16x16 faces stored contiguously
- Face order: [0,0] -> [0,1] -> [1,0] -> [1,1] (row-major within tile)
- Standard tile size: 2048 bytes for BFLOAT16, 4096 bytes for FLOAT32

**Sharding Requirements** (if sharded):
- Shard width must be a multiple of TILE_WIDTH (32)
- Shard height must be a multiple of TILE_HEIGHT (32)
- Shard grid determines which cores hold data

### Output Tensor

| Attribute | Value |
|-----------|-------|
| **Logical Shape** | Same as input (no shape change) |
| **Dimension Convention** | Same as input |
| **Tensor Layout** | `ROW_MAJOR_LAYOUT` - elements stored row-by-row |
| **Memory Layout** | `INTERLEAVED` or `SHARDED` (configurable, can differ from input) |
| **Buffer Type** | DRAM or L1 |
| **Data Type** | Same as input (except BFLOAT8_B converts to BFLOAT16) |

**Row-Major Structure**:
- Each row is stored as a contiguous sequence of elements
- Row size (stick size) = `tensor_width * element_size` bytes
- Pages correspond to complete rows for interleaved storage

### Layout Transformation

The untilize operation performs the following data reorganization:

**Before (Tile Layout)**: Data grouped in 32x32 tiles, tiles stored row-major
```
Tile[0,0]: 32x32 block | Tile[0,1]: 32x32 block | ...
Tile[1,0]: 32x32 block | Tile[1,1]: 32x32 block | ...
```

**After (Row-Major Layout)**: Data stored as sequential rows (sticks)
```
Row 0: elem[0,0], elem[0,1], ..., elem[0,W-1]
Row 1: elem[1,0], elem[1,1], ..., elem[1,W-1]
...
```

## Data Flow Pattern

### General Pipeline (Interleaved Input -> Interleaved Output)

```
DRAM (Tiled)     Reader Kernel       CB_0           Compute Kernel      CB_16          Writer Kernel       DRAM (Row-Major)
    |                 |                |                   |               |                  |                   |
    +--- NoC Read --->+--- cb_push --->+--- cb_wait ------>+               |                  |                   |
                                                           |               |                  |                   |
                                      pack_untilize or untilize_block      |                  |                   |
                                                           |               |                  |                   |
                                                           +--- cb_push -->+--- cb_wait ---->+                   |
                                                                                              |                   |
                                                                           +--- NoC Write ----+------------------>+
```

### Sharded Input -> Sharded Output (Identical Shard Spec)

```
L1 Shard (Tiled)   Reader Kernel      CB_0           Compute Kernel      CB_16          Writer Kernel      L1 Shard (Row-Major)
     |                  |               |                   |               |                  |                   |
     |     (CB backed by input buffer)  |                   |               |                  |                   |
     +---- cb_push ---->+-------------->+--- cb_wait ------>+               |                  |                   |
                                                            |               |                  |                   |
                                       pack_untilize or untilize_block      |                  |                   |
                                                            |               |                  |                   |
                                                            +--- cb_push -->+--- cb_wait ---->+                   |
                                                                            (CB backed by output buffer)          |
                                                                            +------------------------------------>+
```

### Data Flow Steps (Default Multi-Core)

1. **Reader Kernel** (`reader_unary_start_id.cpp` or `reader_unary_sharded.cpp`):
   - Reads tiles from source (DRAM interleaved or L1 shard)
   - Uses `TensorAccessor` for address generation
   - Pushes tiles one-at-a-time to CB_0 (input circular buffer)
   - Tile read order: sequential by tile ID starting from `start_page_id`

2. **Compute Kernel** (`pack_untilize.cpp` or `untilize.cpp`):
   - Waits for input tiles in CB_0
   - Performs tile-to-row-major conversion using hardware intrinsics
   - Output data has rows from tiles interleaved (de-tiled)
   - Pushes row-major data to CB_16 (output circular buffer)

3. **Writer Kernel** (`writer_unary_stick_layout_split_rows_multi_core.cpp`):
   - Waits for untilized data in CB_16
   - Writes row-major data (sticks) to destination
   - Handles complex indexing for sharded outputs
   - Writes tile_height (32) rows at a time per input block

## Circular Buffer Configuration

### CB_0 (Input Circular Buffer, `tt::CBIndex::c_0`)

| Property | Interleaved Input | Sharded Input |
|----------|-------------------|---------------|
| **Index** | 0 | 0 |
| **Purpose** | Stage tiled input data for compute | Same, but backed by shard buffer |
| **Size (tiles)** | `ntiles_per_block * 2` (double-buffered) | `num_tiles_per_shard` (entire shard) |
| **Size (bytes)** | `input_single_tile_size * num_tiles` | `input_single_tile_size * num_tiles_per_shard` |
| **Producer** | Reader kernel | Reader kernel (just pushes, no read) |
| **Consumer** | Compute kernel | Compute kernel |
| **Block Size** | `ntiles_per_block` tiles per iteration | `ntiles_per_block` tiles per iteration |
| **Buffering** | Double-buffered | Single-buffered (entire shard at once) |
| **Backed by Tensor** | No | Yes (input buffer address) |

### CB_16 (Output Circular Buffer, `tt::CBIndex::c_16`)

| Property | Interleaved Output | Sharded Output |
|----------|-------------------|----------------|
| **Index** | 16 | 16 |
| **Purpose** | Stage untilized row-major data for writing | Same, but backed by output buffer |
| **Size (tiles)** | `ntiles_per_block * 2` (double-buffered) | `num_tiles_per_shard` |
| **Size (bytes)** | `output_single_tile_size * num_tiles` | `output_single_tile_size * num_tiles_per_shard` |
| **Producer** | Compute kernel | Compute kernel |
| **Consumer** | Writer kernel | Writer kernel (just waits, no write) |
| **Block Size** | `ntiles_per_block` tiles per iteration | `ntiles_per_block` tiles per iteration |
| **Buffering** | Double-buffered | Single-buffered |
| **Backed by Tensor** | No | Yes (output buffer address) |

### CB Sizing Logic

The CB sizing depends on the parallelization strategy:

```cpp
// For interleaved: double-buffer if processing 2+ blocks
if (num_input_blocks_per_full_core == 1) {
    input_cb_num_tiles = num_tiles_per_input_block;  // Single buffer
} else {
    input_cb_num_tiles = num_tiles_per_input_block * 2;  // Double buffer
}

// For sharded: entire shard fits in one CB
if (input_is_sharded) {
    input_cb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core;
}
```

## Pipeline Pattern Summary

| CB | Typical Capacity | Block Size | Buffering Type | Overlap Possible |
|----|------------------|------------|----------------|------------------|
| CB_0 (interleaved) | 2 * ntiles_per_block | ntiles_per_block | Double-buffered | Yes - reader/compute overlap |
| CB_0 (sharded) | Full shard | ntiles_per_block | Single-buffered | No - entire shard read at once |
| CB_16 (interleaved) | 2 * ntiles_per_block | ntiles_per_block | Double-buffered | Yes - compute/writer overlap |
| CB_16 (sharded) | Full shard | ntiles_per_block | Single-buffered | No - entire shard written at once |

## Index Calculations

### Tile ID Calculation (Reader)

For interleaved tensors:
```cpp
// Linear tile index for reading
uint32_t tile_id = start_page_id + i;  // i = 0 to num_tiles-1

// NoC address calculation using TensorAccessor
uint64_t noc_read_addr = get_noc_addr(tile_id, tensor_accessor);
```

### Row (Stick) ID Calculation (Writer)

For untilized output, the writer maps tiles to output rows:

```cpp
// Each block of ntiles_per_block tiles contains tile_height (32) rows
// Rows are written as sticks with width = ntiles_per_block * tile_width * element_size

// Output page/stick calculation
uint32_t output_page_id = (block_height_index * tile_height + row_within_block)
                          * num_output_blocks_across_width + width_block_index;

// Offset within output page for partial writes
uint32_t output_offset_within_page = num_cols_already_processed * element_size;
```

### Sharded Index Calculation

For sharded tensors, the `ShardedAddrGen` template computes physical addresses:

```cpp
// Compile-time shard info encoded in template parameters
using tensor_shard_info = ShardedInfo<
    memory_layout,      // HEIGHT_SHARDED, WIDTH_SHARDED, or BLOCK_SHARDED
    num_cores,          // Total cores in shard grid
    page_size,          // Size of each page/stick
    pages_per_shard_row,
    contiguous_pages_limit,
    pages_per_shard_x,
    pages_per_shard_y>;

// Run-time address generation
ShardedAddrGen<tensor_shard_info> addr_gen = {
    .bank_base_address = buffer_address,
    .shard_array = mapping_table  // Core coordinates from runtime args
};
```

## Memory Access Patterns

### Read Pattern (Reader Kernel)

**Interleaved Input**:
- Sequential tile reads from DRAM
- Tiles read in linear order: tile 0, tile 1, tile 2, ...
- One tile per NoC transaction
- NoC barrier after each tile (blocking pattern)

**Sharded Input**:
- No actual data movement - CB is backed by L1 shard
- Reader just calls `cb_push_back` to signal data availability

### Write Pattern (Writer Kernel)

**Interleaved Output**:
- Writes organized by tile rows (32 rows at a time)
- Each row written to its destination page
- Complex address calculation handles tile-to-row mapping:
  ```
  Tile row 0: write rows 0-31 to output sticks 0-31
  Tile row 1: write rows 0-31 to output sticks 32-63
  etc.
  ```

**Sharded Output**:
- For identical shard spec: no movement, CB backed by output buffer
- For different shard spec: writes to potentially different cores
- Uses `ShardedAddrGen` for multi-core address generation

### Access Pattern Characteristics

| Pattern | Interleaved | Sharded |
|---------|-------------|---------|
| Read Coalescing | One tile per read | N/A (L1 local) |
| Write Coalescing | Row-width writes | N/A (L1 local) or row-width |
| NoC Traffic | High (all DRAM) | Low (L1 to L1 or none) |
| Latency Hiding | Double-buffering | Minimal (direct CB) |

## Core Distribution Strategy

### Default Multi-Core (`untilize_multi_core`)

Work is parallelized along the **height dimension** (rows of tiles):

```cpp
// Split rows of tiles across available cores
auto [num_compute_cores, compute_core_range, full_compute_core_range,
      cliff_compute_core_range, num_rows_per_full_core, num_rows_per_cliff_core] =
    ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);
```

- **Full cores**: Process `num_rows_per_full_core` rows of tiles each
- **Cliff core**: Processes remaining `num_rows_per_cliff_core` rows (if any)
- Core grid traversal: row-major order

### Block Multi-Core (`untilize_multi_core_block`)

2D parallelization for wide tensors:

```cpp
// Split both width and height dimensions
auto [ncores, all_cores, core_range, cliff_row_core_range, cliff_col_core_range,
      cliff_col_row_core_range, nblocks_per_core, single_block_size, ...] =
    ttnn::split_blocks_for_tilize_wh(grid_size, num_blocks, num_tiles_per_row, num_tiles_per_col);
```

- Divides tensor into 2D grid of blocks
- Each core processes one block
- Handles cliff rows (rightmost blocks) and cliff columns (bottom blocks)
- Corner case: cliff_col_row_core_range for bottom-right block

### Parallelize Column (`untilize_multi_core_parallelize_column`)

For single-tile-height, very wide tensors:

```cpp
// Get largest divisor for column-wise parallelization
ncores_x = get_largest_divisor(ntiles, ncores_x);
ncores_y = get_largest_divisor(ntiles, ncores_y, ncores_x);
```

- Each core processes a subset of columns
- Offset-based writing: each core writes to different column offset

### Sharded Core Distribution

For sharded inputs, cores are determined by the shard specification:

```cpp
ShardSpec shard_spec = input_tensor.shard_spec().value();
CoreRangeSet compute_core_range = shard_spec.grid;
// Each core processes its local shard
```

## Arguments

### Compile-Time Arguments

**Reader Kernel** (`reader_unary_start_id.cpp`):
| Index | Argument | Description |
|-------|----------|-------------|
| 0 | `cb_id_in0` | Input circular buffer index (0) |
| 1+ | TensorAccessorArgs | Buffer type, page size, etc. |

**Compute Kernel** (`pack_untilize.cpp` / `untilize.cpp`):
| Index | Argument | Description |
|-------|----------|-------------|
| 0 | `per_core_block_cnt` | Number of blocks this core processes |
| 1 | `per_core_block_tile_cnt` | Tiles per block |
| 2 | `src_cb_id` | Input CB index (0) |
| 3 | `out_cb_id` | Output CB index (16) |

**Writer Kernel** (`writer_unary_stick_layout_split_rows_multi_core.cpp`):
| Index | Argument | Description |
|-------|----------|-------------|
| 0 | `cb_id_out0` | Output CB index (16) |
| 1 | `output_stick_size` | Row size in bytes |
| 2 | `tile_height` | Tile height (32) |
| 3 | `num_tiles_per_input_block` | Tiles per block width |
| 4 | `num_output_blocks_across_width` | Output sharding width blocks |
| 5 | `output_element_size` | Bytes per element |
| 6 | `num_cols_per_input_block` | Elements per input block width |
| 7 | `num_cols_per_output_block` | Elements per output block width |
| 8+ | TensorAccessorArgs or ShardedInfo | Destination addressing |

### Runtime Arguments

**Reader Kernel**:
| Index | Argument | Description |
|-------|----------|-------------|
| 0 | `src_addr` | Source buffer DRAM/L1 address |
| 1 | `num_tiles` | Total tiles to read |
| 2 | `start_page_id` | First tile index for this core |

**Writer Kernel**:
| Index | Argument | Description |
|-------|----------|-------------|
| 0 | `dst_addr` | Destination buffer address |
| 1 | `num_input_blocks_to_process` | Blocks for this core |
| 2 | `height_wise_input_block_start_index` | Starting row block index |
| 3 | `num_unpadded_cols_per_input_block` | Actual columns (for uneven sharding) |
| 4 | `width_wise_output_block_start_index` | Starting output block column |
| 5 | `num_cols_already_processed_in_first_output_block` | Column offset in first output block |

## Kernel Implementations

### Reader Kernel: `reader_unary_start_id.cpp`

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

**Responsibilities**:
- Read tiled input data from source (DRAM or L1 shard)
- Transfer tiles to input circular buffer (CB_0)
- Handle both interleaved and sharded memory layouts

**Input**: DRAM buffer (interleaved) or L1 buffer (sharded)

**Output**: CB_0 (input circular buffer)

**Key Logic**:
```cpp
for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
    cb_reserve_back(cb_id_in0, 1);
    uint64_t noc_read_addr = get_noc_addr(page_id, tensor_accessor);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    noc_async_read(noc_read_addr, l1_write_addr, tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, 1);
}
```

**Sharded Variant** (`reader_unary_sharded.cpp`):
- Simply pushes tiles to CB without reading (CB backed by shard buffer)
- `cb_push_back(cb_id_in0, num_tiles_per_core);`

### Compute Kernel: `pack_untilize.cpp` (Fast Path)

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp`

**Responsibilities**:
- Convert tile layout to row-major layout using hardware-accelerated `pack_untilize`
- Process blocks of tiles efficiently
- Support INT32/UINT32/FLOAT32 data types

**Input**: CB_0 (tiled data)

**Output**: CB_16 (row-major data)

**Key Logic** (via `untilize_helpers.hpp`):
```cpp
// For narrow widths (fits in DEST register)
pack_untilize_init<tile_width, tile_width>(icb_id, ocb_id);
for (uint32_t r = 0; r < num_rows; ++r) {
    cb_wait_front(icb_id, tiles_per_row);
    cb_reserve_back(ocb_id, tiles_per_row);
    pack_untilize_block<tile_width, tile_width>(icb_id, block_rt_dim, ocb_id, 0);
    cb_pop_front(icb_id, tiles_per_row);
    cb_push_back(ocb_id, tiles_per_row);
}
pack_untilize_uninit(ocb_id);
```

**Optimizations**:
- Uses DEST register for tile-to-row conversion
- Block-based processing for wide integer types
- Automatic dispatch based on data type and width

### Compute Kernel: `untilize.cpp` (Fallback Path)

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp`

**Responsibilities**:
- Fallback for cases where `pack_untilize` is not applicable
- Used for UINT16, or FLOAT32 with wide rows (>= MAX_PACK_UNTILIZE_WIDTH)

**Key Logic**:
```cpp
untilize_init(icb_id);
for (uint32_t r = 0; r < num_rows; ++r) {
    cb_wait_front(icb_id, tile_width);
    cb_reserve_back(ocb_id, tile_width);
    untilize_block(icb_id, tile_width, ocb_id);
    cb_push_back(ocb_id, tile_width);
    cb_pop_front(icb_id, tile_width);
}
untilize_uninit(icb_id);
```

### Writer Kernel: `writer_unary_stick_layout_split_rows_multi_core.cpp`

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Responsibilities**:
- Write row-major (untilized) data to destination
- Handle complex row-to-page mapping
- Support both interleaved and sharded outputs
- Handle uneven sharding (partial columns at shard boundaries)

**Input**: CB_16 (row-major data)

**Output**: DRAM buffer (interleaved) or L1 buffer (sharded)

**Key Logic**:
```cpp
auto write_tiles_in_current_block = [&](uint32_t block_height_index) {
    cb_wait_front(cb_id_out0, num_tiles_per_input_block);
    uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);

    for (uint32_t j = 0; j < tile_height; ++j) {  // 32 rows per tile height
        uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * element_size;

        // Calculate output page and offset
        uint32_t output_page_id = num_pages_already_processed + width_wise_output_block_start_index;

        // Write columns, potentially spanning multiple output blocks
        while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
            uint32_t num_cols_to_write = min(remaining_input_cols, remaining_output_cols);
            uint64_t dst_noc_addr = get_noc_addr(output_page_id, s, output_offset);
            noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write);
            // Update indices...
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, num_tiles_per_input_block);
};
```

**Sharded Output Handling**:
- Uses `ShardedAddrGen` for multi-core address generation
- Writes can span multiple output shards
- Offset calculations account for partial block writes

## Implementation Notes

### Compute Kernel Selection

The program factory selects between `pack_untilize` (fast) and `untilize` (slow) based on:

```cpp
bool use_slow_path = !use_pack_untilize
                   || input_dtype == DataType::UINT16
                   || (input_dtype == DataType::FLOAT32 && ntiles_per_block >= MAX_PACK_UNTILIZE_WIDTH);
```

- `pack_untilize` is hardware-accelerated and preferred
- DEST register capacity limits block width
- FLOAT32 has lower MAX_PACK_UNTILIZE_WIDTH due to larger element size

### Special Case: BFLOAT8_B

BFLOAT8_B input is converted to BFLOAT16 output:
```cpp
DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B
                      ? DataType::BFLOAT16
                      : input_tensor.dtype();
```

### Program Factory Selection Logic

The `create_program` method in `Untilize` selects the appropriate factory:

1. **Single-core**: If `use_multicore == false`
2. **Sub-core grids**: If `sub_core_grids` parameter provided (custom core allocation)
3. **Block multicore**: If `!enough_space_height` and both input/output are interleaved
4. **Identical shard spec**: If input/output are both L1 sharded with same spec
5. **Default multicore**: All other cases

### Uneven Sharding Support

The implementation handles uneven sharding where tensor dimensions are not evenly divisible by shard dimensions:

- Width-wise: Last shard may have fewer valid columns
- Height-wise: Last shard may have fewer valid rows
- Combined: Bottom-right shard may be smaller in both dimensions

The writer kernel uses `num_unpadded_cols_per_input_block` to avoid writing garbage data from padding.

### Override Runtime Arguments

Each program factory returns an `override_runtime_arguments_callback` that allows buffer addresses to be updated without recompilation:

```cpp
auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, ...](
    const void* operation,
    Program& program,
    const std::vector<Tensor>& input_tensors,
    ...) {

    // Update buffer addresses at runtime
    auto& reader_args = GetRuntimeArgs(program, reader_kernel_id, core);
    reader_args[0] = src_buffer->address();

    auto& writer_args = GetRuntimeArgs(program, writer_kernel_id, core);
    writer_args[0] = dst_buffer->address();
};
```

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the untilize operation? How does it convert tile layout to row-major layout? What is the difference between pack_untilize and regular untilize?"
   **Reason**: Understanding the fundamental operation and its hardware implementations
   **Key Findings**:
   - Untilize reverses tilize, converting 32x32 tiled data to row-major
   - `pack_untilize` is hardware-accelerated and supports INT32/UINT32
   - Regular `untilize` is a fallback for unsupported configurations

2. **Query**: "How do circular buffers work in tt-metal? What are the different CB indices and how are producer-consumer relationships established?"
   **Reason**: Understanding inter-kernel communication mechanism
   **Key Findings**:
   - CBs use producer-consumer model with `cb_reserve_back`/`cb_push_back` (producer) and `cb_wait_front`/`cb_pop_front` (consumer)
   - CB indices 0-31 available, typically 0-7 for input, 16-23 for output
   - Hardware metadata synchronization prevents race conditions

3. **Query**: "What is tile layout vs row-major layout? How are tiles organized in memory (32x32 structure)?"
   **Reason**: Understanding data layout transformation
   **Key Findings**:
   - Tiles are 32x32 blocks subdivided into four 16x16 faces
   - Face order is row-major within tile
   - Row-major stores consecutive row elements contiguously

4. **Query**: "What are the different sharding strategies in tt-metal: HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED?"
   **Reason**: Understanding memory distribution across cores
   **Key Findings**:
   - HEIGHT_SHARDED: splits rows across cores
   - WIDTH_SHARDED: splits columns across cores
   - BLOCK_SHARDED: 2D grid splitting both dimensions

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding core distribution strategies
   **Key Information**: `split_blocks_for_tilize` and `split_blocks_for_tilize_wh` functions implement work partitioning with cliff handling

2. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding CB creation helper
   **Key Information**: `create_cb` function wraps CircularBufferConfig creation with proper sizing

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
   **Reason**: Understanding unified untilize dispatch logic
   **Key Information**: Automatic selection between `pack_untilize` and standard `untilize` based on data format and width constraints
