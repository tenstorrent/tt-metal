# Untilize Multi-Core Implementation Analysis

## Overview

The `untilize_multi_core_program_factory.cpp` implements tile-to-row-major data format conversion across multiple Tensix cores. This operation reads tiled data from memory, converts it to row-major (stick) layout using the compute kernel, and writes the row-major result back to memory. The implementation supports both interleaved and sharded memory configurations for input and output tensors.

**Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Key Feature for Hybrid Operations**: This analysis focuses on how the operation produces row-major interleaved output, making it a reference for any hybrid operation that needs to output row-major data to DRAM.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (row of tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (one tile row width) |
| **Total units** | `num_tiles_per_col` blocks (number of tile rows) |
| **Loop structure** | Each core processes `num_input_blocks_per_core` blocks sequentially |

A "block" in this operation corresponds to one row of tiles (height = 1 tile, width = full tensor width in tiles). Each block untilizes into `tile_height` rows of row-major elements.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [*, H, W] (flattened to 2D: tensor_height x tensor_width) |
| **Dimension convention** | Last dim is width, second-to-last is height |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | Any supported (BFLOAT16, FLOAT32, INT32, UINT32, etc.) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT (stick/linear) |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | Same as input |

### Layout Transformation

The compute kernel performs the tile-to-row-major conversion:
- **Input**: Tiles stored as 32x32 blocks with 4 faces (16x16 each)
- **Output**: Row-major sticks where elements are stored sequentially per row
- **Mechanism**: Uses `pack_untilize` (hardware-accelerated) or `untilize_block` (fallback)

## Data Flow Pattern

### Stage 1: Reader Kernel
1. **Interleaved Input**: Reads tiles sequentially from DRAM using `TensorAccessor`
2. **Sharded Input**: Simply pushes pre-loaded L1 data to CB (cb_push_back)
3. Uses CB c_0 for tile staging

### Stage 2: Compute Kernel
1. Waits for `num_tiles_per_input_block` tiles in input CB
2. Reserves space in output CB for the same number of tiles
3. Performs untilize operation (pack_untilize or standard untilize)
4. Pushes row-major data to output CB
5. Pops processed tiles from input CB

### Stage 3: Writer Kernel
1. Waits for untilized data in CB c_16
2. Iterates through each row within the tile height (32 rows typically)
3. For each row, calculates the output page ID and writes to correct DRAM location
4. Handles output sharding boundaries when input and output shard widths differ

```
DRAM/L1 (Tiled) --> Reader --> CB c_0 --> Compute (Untilize) --> CB c_16 --> Writer --> DRAM/L1 (Row-Major)
```

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | 1-2 blocks or full shard | num_tiles_per_input_block | Single/Double | Reader | Compute | Block |
| c_16 | cb_output | Output row-major staging | 1-2 blocks | num_tiles_per_input_block | Single/Double | Compute | Writer | Block |

### CB Sizing Logic

**Input CB (c_0)**:
- **Sharded input**: `num_tiles_per_input_block * num_input_blocks_per_full_core` (entire shard at once)
- **Interleaved, 1 block/core**: `num_tiles_per_input_block` (single-buffered)
- **Interleaved, 2+ blocks/core**: `num_tiles_per_input_block * 2` (double-buffered)

**Output CB (c_16)**:
- **1 block/core**: `num_tiles_per_input_block` (single-buffered)
- **2+ blocks/core**: `num_tiles_per_input_block * 2` (double-buffered)

### Key Insight for Hybrid Operations

The output CB stores data in row-major format after untilization. The CB page size is still calculated using tile size, but the actual data layout within the CB is row-major sticks. The writer kernel reads this row-major data using `get_read_ptr()` and writes it stick-by-stick to DRAM.

## Pipeline Pattern Summary

| Condition | Input CB Buffering | Output CB Buffering | Overlap Potential |
|-----------|-------------------|---------------------|-------------------|
| Sharded input | Single (full shard) | Double | Compute/Write overlap |
| Interleaved, 1 block | Single | Single | No overlap |
| Interleaved, 2+ blocks | Double | Double | Full Read/Compute/Write overlap |

## Index Calculations

### Input Index Mapping

For interleaved input:
```cpp
tile_start_index = core_index * num_tiles_per_input_block * num_input_blocks_per_full_core
page_id = tile_start_index + local_tile_offset
```

### Output Index Mapping (Critical for Row-Major Output)

The writer kernel maps tile-space to row-major stick-space:

```cpp
// For each input block at height index 'block_height_index':
num_rows_already_processed = block_height_index * tile_height + row_within_tile
output_page_id = num_rows_already_processed * num_output_blocks_across_width + width_block_index
```

**Key Calculations**:
1. `num_cols_per_input_block = num_tiles_per_input_block * tile_width` (columns per tile row)
2. `num_cols_per_output_block = tensor_width / num_output_blocks_across_width` (columns per output page)
3. `output_stick_size = tensor_width * element_size / num_output_blocks_across_width` (bytes per output page)

### Stick Address Calculation

Within the writer kernel:
```cpp
// Calculate L1 address for current row
current_l1_read_addr = base_l1_read_addr + row_index * num_cols_per_input_block * element_size

// Calculate output NOC address with optional offset for partial writes
dst_noc_addr = get_noc_addr(output_page_id, tensor_accessor, output_offset_within_page_in_bytes)
```

## Memory Access Patterns

### Read Pattern (Interleaved Input)

- **Access Type**: Sequential tile reads
- **Pattern**: Tiles read in order: tile 0, tile 1, ... tile N
- **Barrier**: `noc_async_read_barrier()` after each tile
- **CB Operations**: `cb_reserve_back(1)`, `cb_push_back(1)` per tile

### Write Pattern (Interleaved Output) - Key for Hybrid Operations

- **Access Type**: Row-major stick writes
- **Pattern**: For each tile row, iterate through all 32 (or tile_height) element rows
- **Granularity**: Writes partial or full sticks (num_bytes_to_write = num_cols_to_write * element_size)
- **Barrier**: `noc_async_write_barrier()` after processing all rows in a tile row
- **CB Operations**: `cb_wait_front(num_tiles_per_input_block)`, `cb_pop_front(num_tiles_per_input_block)` per block

**Write Pattern Pseudocode**:
```cpp
for each input_block:
    cb_wait_front(cb_out, num_tiles_per_input_block)
    base_l1_addr = get_read_ptr(cb_out)

    for row in 0..tile_height:
        l1_addr = base_l1_addr + row * num_cols_per_input_block * element_size
        output_page_id = calculate_output_page(block_index, row)

        while cols_remaining:
            cols_to_write = min(cols_remaining, cols_in_current_output_page)
            noc_addr = get_noc_addr(output_page_id, accessor, byte_offset)
            noc_async_write(l1_addr, noc_addr, cols_to_write * element_size)
            advance pointers

    noc_async_write_barrier()
    cb_pop_front(cb_out, num_tiles_per_input_block)
```

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Determined by `split_blocks_for_tilize` |
| **Total cores** | `num_compute_cores` (full + cliff) |
| **Work per core** | `num_input_blocks_per_full_core` blocks (full) or `num_input_blocks_per_cliff_core` (cliff) |
| **Load balancing** | Near-equal with remainder on cliff core |

### Core Types

1. **Full Cores**: Process `num_input_blocks_per_full_core` blocks each
2. **Cliff Core**: Processes `num_input_blocks_per_cliff_core` blocks (remainder)

### Sharded Input Core Distribution

When input is sharded:
- Core range comes from input shard spec
- No cliff core (shards are pre-distributed)
- `num_input_blocks_across_width` accounts for width-wise sharding

## Arguments

### Compile-Time Arguments

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |
| 1+ | TensorAccessorArgs | varies | Buffer layout info for address calculation |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer ID (c_16) |
| 1 | output_stick_size | uint32_t | Bytes per output row |
| 2 | tile_height | uint32_t | Tile height (typically 32) |
| 3 | num_tiles_per_input_block | uint32_t | Tiles per block (row width in tiles) |
| 4 | num_output_blocks_across_width | uint32_t | Output sharding width factor |
| 5 | output_element_size | uint32_t | Bytes per element |
| 6 | num_cols_per_input_block | uint32_t | Elements per input block row |
| 7 | num_cols_per_output_block | uint32_t | Elements per output page |
| 8+ | TensorAccessorArgs | varies | Buffer layout info (interleaved) or ShardedInfo (sharded) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Tiles per block (num_tiles_per_input_block) |
| 1 | src_cb_id | uint32_t | Input CB ID (c_0) |
| 2 | out_cb_id | uint32_t | Output CB ID (c_16) |

### Runtime Arguments

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer address |
| 1 | num_tiles | uint32_t | Total tiles to read |
| 2 | start_page_id | uint32_t | First tile ID for this core |

#### Reader Kernel (Sharded)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_core | uint32_t | Tiles in this core's shard |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer address |
| 1 | num_input_blocks_to_process | uint32_t | Blocks this core will write |
| 2 | height_wise_input_block_start_index | uint32_t | Starting block row index |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Valid columns (handles padding) |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output width block |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Offset for first write |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |

## Kernel Implementations

### Reader Kernel (Interleaved)

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM (tiled) | CB c_0 | Read tiles via TensorAccessor |

**Key Logic**:
- Creates TensorAccessor from compile-time args
- Loops over tiles: reserve CB, read tile, barrier, push CB
- Sequential tile reading pattern

### Reader Kernel (Sharded)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_sharded | RISCV_0 | N/A | L1 (sharded) | CB c_0 | Push already-present L1 data |

**Key Logic**:
- Data already in L1 via sharding
- Single `cb_push_back(num_tiles_per_core)` call
- No actual data movement needed

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
**Alternate**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| pack_untilize | RISCV_2 (Unpack/Math/Pack) | N/A | CB c_0 | CB c_16 | Tile-to-row-major conversion |

**Key Logic**:
- Calls `compute_kernel_hw_startup()` for initialization
- Uses unified `compute_kernel_lib::untilize<>()` template
- Automatically selects pack_untilize (fast) or standard untilize (fallback) based on:
  - Width vs DEST register limit
  - Data type (integer vs float)
- Processes `per_core_block_cnt` blocks

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_stick_layout | RISCV_1 | NOC1 | CB c_16 | DRAM/L1 (row-major) | Write row-major sticks |

**Key Logic** (Critical for Row-Major Output):
1. Creates TensorAccessor for output buffer
2. For each input block:
   - Waits for untilized data in CB
   - Gets L1 read pointer
   - For each row (0 to tile_height):
     - Calculates output page ID from block height and row
     - Handles potential cross-page writes (when input/output shard widths differ)
     - Writes stick data using `noc_async_write()`
   - Barriers and pops CB

**Row-Major Write Pattern**:
```cpp
// Writer iterates through tile rows, outputting stick-by-stick
for (row in 0..tile_height):
    l1_addr = base_addr + row * cols_per_block * element_size
    page_id = (block_height * tile_height + row) * output_blocks_width + width_block
    noc_async_write(l1_addr, get_noc_addr(page_id, accessor, offset), bytes_to_write)
```

## Implementation Notes

### Pack Untilize vs Standard Untilize Selection

The program factory selects between two compute kernels:

1. **Pack Untilize (Fast Path)**:
   - Hardware-accelerated tile-to-row-major conversion
   - Used when `use_pack_untilize == true` AND width <= DEST limit
   - Limited to 8 tiles width (half-sync, 16-bit) or 4 tiles (32-bit)

2. **Standard Untilize (Fallback)**:
   - Software-based conversion
   - Used for UINT16, FLOAT32 with wide tensors
   - Used when `num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH` for FLOAT32

### FP32 Destination Accumulation

When `fp32_dest_acc_en == true`:
- DEST register capacity is halved
- `unpack_to_dest_mode[src0_cb_index] = UnpackToDestFp32`
- Affects pack_untilize width constraints

### Handling Uneven Shards

The implementation handles width/height shard padding:
- `num_unpadded_cols_per_input_block`: Actual valid columns (excluding padding)
- Cliff cores handle remainder blocks
- Runtime args adjust per-core based on shard boundaries

### Output Page Calculation for Row-Major Interleaved

For DRAM interleaved output, the output is organized as pages where:
- Each page = one row of elements across the tensor width (or output block width)
- Total pages = tensor_height (in rows, not tiles)
- `output_stick_size = tensor_width * element_size`

## Usage as Output Stage Reference

When creating a hybrid operation that outputs row-major interleaved data:

1. **CB Configuration**: Use CB c_16 for output staging, sized for double-buffering
2. **Compute Output Format**: Ensure compute kernel outputs row-major data (or use untilize)
3. **Writer Pattern**: Iterate by element rows, not tiles:
   ```cpp
   for each row_of_elements:
       page_id = row_index * output_blocks_width + width_block
       noc_addr = get_noc_addr(page_id, accessor)
       noc_async_write(l1_addr, noc_addr, stick_size_bytes)
   ```
4. **TensorAccessor Setup**: Use output_stick_size as page size for TensorAccessor
5. **Barrier Placement**: Barrier after processing all rows in a logical block

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize operation work in TT-Metal? What is the difference between pack_untilize and standard untilize?"
   **Reason**: Needed to understand the tile-to-row-major conversion mechanism
   **Key Findings**: pack_untilize operates in PACK thread (hardware-accelerated), standard untilize in UNPACK thread. DEST register limits: 8 tiles (16-bit half-sync), 4 tiles (32-bit half-sync).

2. **Query**: "What is the TensorAccessor API in TT-Metal?"
   **Reason**: Needed to understand memory access abstraction
   **Key Findings**: TensorAccessor provides NoC address calculation for interleaved/sharded tensors. TensorAccessorArgs configures compile-time vs runtime parameters.

3. **Query**: "How does split_blocks_for_tilize work for distributing work across cores?"
   **Reason**: Needed to understand core distribution strategy
   **Key Findings**: Divides blocks among cores, creates full_core_range and cliff_core_range. Cliff core handles remainder work.

4. **Query**: "What is the create_cb function in TT-Metal?"
   **Reason**: Needed to understand CB configuration patterns
   **Key Findings**: create_cb is a helper that wraps CreateCircularBuffer. CB indices c_0, c_1 typically for inputs, c_16 for outputs.

5. **Query**: "How does noc_async_write work for writing row-major data to DRAM interleaved buffers?"
   **Reason**: Critical for understanding row-major output pattern
   **Key Findings**: Pattern involves TensorAccessor creation, CB wait/pop, get_noc_addr per stick, noc_async_write with barrier.

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Core architecture understanding
   **Key Information**: Three-kernel model (reader/compute/writer), CB synchronization, NoC data movement patterns, tile layout fundamentals.

2. **Source**: `.claude/references/table-templates.md`
   **Reason**: Output format templates
   **Key Information**: Standard table formats for analysis documentation.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding create_cb helper
   **Key Information**: Wrapper for CreateCircularBuffer that simplifies CB configuration with single or multiple CB indices.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding work distribution
   **Key Information**: split_blocks_for_tilize implementation, BlockSplit structure with ncores, core_range, cliff_core_range.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
   **Reason**: Understanding untilize compute implementation
   **Key Information**: Unified untilize function with automatic dispatch based on width and data type. DEST_AUTO_LIMIT detection, integer format handling.
