# Untilize Multi-Core Implementation Analysis

## Overview

The untilize multi-core operation converts tensor data from **TILE_LAYOUT** (32x32 tiles) to **ROW_MAJOR** layout using multiple Tensix cores in parallel. This is the reverse of the tilize operation, transforming hardware-optimized tile format back to standard row-major format for CPU consumption or operations that require linear memory access.

**Program Factory Path**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

The implementation supports:
- **Interleaved input** with interleaved or sharded output
- **Sharded input** with interleaved or sharded output
- Two compute paths: **pack_untilize** (hardware-accelerated) and **standard untilize** (fallback)
- Automatic selection between compute paths based on data type and width constraints

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (row of tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (one tile row width) |
| **Total units** | `num_tiles_per_col` (total tile rows across tensor height) |
| **Loop structure** | Outer loop over blocks (tile rows), inner loop processes all tiles in block width |

A **work unit** consists of one "input block" which is one complete row of tiles spanning the tensor width. For interleaved input, each block contains `num_tiles_per_row` tiles. For sharded input, each block contains `input_shard_width / tile_width` tiles.

The compute kernel processes blocks one at a time:
1. Wait for all tiles in a block row to be available in input CB
2. Untilize the entire block row (convert from tile format to row-major)
3. Push the untilized data to output CB

## Tensor Format and Layout

### Input Tensor

| Property | Interleaved Input | Sharded Input |
|----------|-------------------|---------------|
| **Logical shape** | [..., H, W] | [..., H, W] |
| **Dimension convention** | Last dim is width | Last dim is width |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | HEIGHT/WIDTH/BLOCK_SHARDED |
| **Buffer type** | DRAM or L1 | L1 (in shard grid) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 | Same |

For sharded input:
- **Shard Shape**: `[shard_height, shard_width]` in elements
- **Core Grid**: Specified by `input_shard_spec.grid`
- **Shard Orientation**: ROW_MAJOR or COL_MAJOR

### Output Tensor

| Property | Interleaved Output | Sharded Output |
|----------|-------------------|----------------|
| **Logical shape** | [..., H, W] | [..., H, W] |
| **Dimension convention** | Last dim is width | Last dim is width |
| **Tensor layout** | ROW_MAJOR | ROW_MAJOR |
| **Memory layout** | INTERLEAVED | WIDTH/BLOCK_SHARDED |
| **Buffer type** | DRAM or L1 | L1 (in shard grid) |
| **Data type** | Same as input | Same as input |

For sharded output:
- **Shard Shape**: `[shard_height, shard_width]` in elements
- **Core Grid**: Determined by output memory config
- **Shard Orientation**: ROW_MAJOR or COL_MAJOR

### Layout Transformations

The operation performs a single transformation:
- **TILE_LAYOUT -> ROW_MAJOR**: Each 32x32 tile is unpacked into 32 consecutive rows of 32 elements each

No tilize/untilize or reshard operations are performed as intermediate steps - the untilize is the core transformation.

## Data Flow Pattern

### Interleaved Input Path

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) | CB_in (c_0) | reserve_back(1), push_back(1) per tile |
| 2 | Compute | CB_in (c_0) | CB_out (c_16) | wait_front(block), pop_front(block), reserve_back(block), push_back(block) |
| 3 | Writer | CB_out (c_16) | DRAM/L1 (output) | wait_front(block), pop_front(block) |

**Detailed Flow**:
1. **Reader** (`reader_unary_start_id.cpp`): Reads tiles sequentially from DRAM using TensorAccessor. For each tile:
   - `cb_reserve_back(cb_id_in0, 1)` - Reserve space for one tile
   - `noc_async_read()` - DMA tile from DRAM to L1
   - `noc_async_read_barrier()` - Wait for transfer completion
   - `cb_push_back(cb_id_in0, 1)` - Make tile available to compute

2. **Compute** (`untilize_variable_num_blocks.cpp` or `pack_untilize_variable_num_blocks.cpp`): Uses the kernel helper library `compute_kernel_lib::untilize()` which automatically selects the optimal path:
   - Pack untilize for narrow widths (fits in DEST registers)
   - Block-based pack untilize for wide integer types
   - Standard untilize for wide floating-point types

3. **Writer** (`writer_unary_stick_layout_split_rows_multi_core.cpp`): Writes untilized rows to output:
   - Processes one block at a time
   - For each row in the block (32 rows per tile height):
     - Calculates output page ID and byte offset
     - Writes row data using `noc_async_write()`
   - Handles output sharding by computing correct page offsets

### Sharded Input Path

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | L1 (shard buffer) | CB_in (c_0) | push_back(num_tiles) - no actual read |
| 2 | Compute | CB_in (c_0) | CB_out (c_16) | wait_front(block), pop_front(block), reserve_back(block), push_back(block) |
| 3 | Writer | CB_out (c_16) | DRAM/L1 (output) | wait_front(block), pop_front(block) |

**Key Difference**: For sharded input, the input CB is globally allocated at the shard buffer address. The reader kernel (`reader_unary_sharded.cpp`) simply does `cb_push_back()` to signal that data is already available - no actual DMA transfer is needed since data is already in L1.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tiles (tiled format) | See below | num_tiles_per_input_block | Conditional | Reader | Compute | Block |
| c_16 | cb_output | Output data (row-major) | See below | num_tiles_per_input_block | Conditional | Compute | Writer | Block |

### Input CB (c_0) Capacity Logic

**Sharded Input**:
```cpp
input_cb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core
// Entire shard processed at once - single-buffered for the shard
```

**Interleaved Input**:
```cpp
if (num_input_blocks_per_full_core == 1) {
    input_cb_num_tiles = num_tiles_per_input_block;  // Single-buffered
} else {
    input_cb_num_tiles = num_tiles_per_input_block * 2;  // Double-buffered
}
```

### Output CB (c_16) Capacity Logic

```cpp
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;  // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;  // Double-buffered
}
```

### Buffering Strategy Summary

| Condition | Input CB Buffering | Output CB Buffering |
|-----------|-------------------|---------------------|
| Sharded input | Single (entire shard) | Conditional |
| Interleaved, 1 block/core | Single | Single |
| Interleaved, 2+ blocks/core | Double | Double |

## Pipeline Pattern Summary

The pipeline pattern depends on the number of blocks processed per core:

**Single Block Per Core**: No overlap possible - reader fills CB, compute processes, writer drains.

**Multiple Blocks Per Core (Double Buffering)**:
- Reader can fill next block while compute processes current block
- Compute can process while writer drains previous block
- Classic producer-consumer overlap pattern

The helper function `compute_kernel_lib::untilize()` internally handles CB operations with the pattern:
```cpp
for each row:
    cb_wait_front(icb, tile_width)      // Wait for row of tiles
    cb_reserve_back(ocb, tile_width)    // Reserve output space
    // Perform untilize operation
    cb_pop_front(icb, tile_width)       // Release input
    cb_push_back(ocb, tile_width)       // Publish output
```

## Index Calculations

### Tensor Accessor Usage

**Interleaved Input** (`reader_unary_start_id.cpp`):
```cpp
constexpr auto src_args = TensorAccessorArgs<1>();  // Compile-time args start at index 1
const auto s = TensorAccessor(src_args, src_addr, tile_bytes);
uint64_t noc_read_addr = get_noc_addr(page_id, s);
```
- `page_id` = linear tile index (0 to num_tiles-1)
- TensorAccessor handles bank interleaving automatically

**Interleaved Output** (`writer_unary_stick_layout_split_rows_multi_core.cpp`):
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // Compile-time args start at index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
uint64_t dst_noc_addr = get_noc_addr(output_page_id, s, output_offset_within_page_in_bytes);
```
- `output_page_id` = row index within output block structure
- `output_offset_within_page_in_bytes` = byte offset for partial writes when input/output block sizes differ

### Block Index Mapping

The writer kernel uses sophisticated index calculations to handle cases where input and output shards have different widths:

```cpp
// Calculate which output page (row) to write to
uint32_t num_rows_already_processed = block_height_index * tile_height + j;
uint32_t num_pages_already_processed_in_previous_rows =
    num_rows_already_processed * num_output_blocks_across_width;
uint32_t output_page_id =
    num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;
```

This handles the case where input blocks may span multiple output blocks or vice versa.

## Memory Access Patterns

### Read Pattern

**Interleaved Input**:
- **Pattern**: Sequential tile reads
- **Granularity**: One tile at a time
- **Access**: Uses TensorAccessor for bank-interleaved addressing
- **Barrier**: Per-tile barrier (`noc_async_read_barrier()` after each tile)

**Sharded Input**:
- **Pattern**: No actual reads - data already in L1
- **Granularity**: Entire shard available at once
- **Access**: CB points directly to shard buffer

### Write Pattern

**Interleaved Output**:
- **Pattern**: Row-by-row writes with potential partial writes
- **Granularity**: Variable - depends on input/output block alignment
- **Access**: Uses TensorAccessor for bank-interleaved addressing
- **Barrier**: Per-block barrier (`noc_async_write_barrier()` after all rows in block)

**Sharded Output**:
- **Pattern**: Row-by-row writes to shard cores
- **Granularity**: Same as interleaved
- **Access**: Uses ShardedAddrGen for core-local addressing

The writer handles complex scenarios where input shard width differs from output shard width, writing partial rows when needed.

## Core Distribution Strategy

| Attribute | Interleaved Input | Sharded Input |
|-----------|-------------------|---------------|
| **Grid topology** | 1D (linear core assignment) | Matches shard grid |
| **Grid dimensions** | Up to device compute grid | Shard grid dimensions |
| **Total cores** | min(num_tiles_per_col, grid_area) | num_shard_cores |
| **Work per core** | num_rows_per_full_core blocks | num_input_blocks_per_full_core blocks |
| **Load balancing** | Round-robin with cliff core | Equal (shard-based) |
| **Remainder handling** | Last core (cliff) gets fewer | Last shard may be smaller |

### Core Assignment Logic

**Interleaved Input**:
```cpp
auto [num_compute_cores, compute_core_range, full_compute_core_range,
      cliff_compute_core_range, num_rows_per_full_core, num_rows_per_cliff_core] =
    ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);
```
- Uses `split_blocks_for_tilize()` to distribute tile rows across cores
- Full cores process `num_rows_per_full_core` blocks
- Cliff core (if any) processes `num_rows_per_cliff_core` blocks

**Sharded Input**:
```cpp
num_compute_cores = input_shard_spec.grid.num_cores();
compute_core_range = input_shard_spec.grid;
full_compute_core_range = input_shard_spec.grid;
cliff_compute_core_range = CoreRangeSet();  // No cliff for sharded
```
- Each core processes its own shard
- No cliff core needed since sharding is pre-determined

### Uneven Shard Handling

The implementation handles uneven shards in both dimensions:

**Width-wise uneven shards**:
```cpp
if (is_last_input_shard_in_row) {
    num_unpadded_cols_per_input_block =
        num_cols_per_input_block - (round_up(tensor_width, input_shard_width) - tensor_width);
}
```

**Height-wise uneven shards**:
```cpp
if (is_last_input_shard_in_col) {
    num_input_blocks_to_process =
        num_input_blocks_per_full_core -
        (round_up(tensor_height, input_shard_height) - tensor_height) / tile_height;
}
```

## Arguments

### Compile-Time Arguments

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |
| 1+ | TensorAccessorArgs | varies | Bank count, strides, etc. for input tensor |

#### Reader Kernel (Sharded)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer ID (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output row in bytes |
| 2 | tile_height | uint32_t | Height of tile (typically 32) |
| 3 | num_tiles_per_input_block | uint32_t | Tiles per input block width |
| 4 | num_output_blocks_across_width | uint32_t | Output blocks spanning tensor width |
| 5 | output_element_size | uint32_t | Size of single element in bytes |
| 6 | num_cols_per_input_block | uint32_t | Elements per input block width |
| 7 | num_cols_per_output_block | uint32_t | Elements per output block width |
| 8+ | TensorAccessorArgs or ShardedInfo | varies | Output addressing parameters |

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
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_tiles | uint32_t | Total tiles to read |
| 2 | start_page_id | uint32_t | Starting tile index for this core |

#### Reader Kernel (Sharded)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_core | uint32_t | Tiles in this core's shard |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_input_blocks_to_process | uint32_t | Blocks this core will process |
| 2 | height_wise_input_block_start_index | uint32_t | Starting block row index |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Valid columns (handles uneven shards) |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output block column |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Offset into first output block |
| 6+ | Sharding runtime args (if sharded output) | varies | Shard mapping table |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |

## Kernel Implementations

### Reader Kernel (Interleaved Input)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id.cpp | RISCV_0 | NOC0 | DRAM (tiles) | CB_in (c_0) | Read tiles sequentially |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

**Key Logic**:
- Uses TensorAccessor for bank-interleaved DRAM access
- Reads tiles one at a time with barrier per tile
- Supports both interleaved and sharded source via compile-time `#ifdef SHARDED`

### Reader Kernel (Sharded Input)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_sharded.cpp | RISCV_0 | NOC0 | L1 (shard) | CB_in (c_0) | Signal data availability |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

**Key Logic**:
- No actual data movement - CB is globally allocated at shard buffer
- Simply calls `cb_push_back(cb_id_in0, num_tiles_per_core)` to signal availability
- Extremely lightweight kernel

### Compute Kernel (Standard Untilize)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| untilize_variable_num_blocks.cpp | RISCV_2 | N/A | CB_in (c_0) | CB_out (c_16) | Untilize via helper lib |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`

**Key Logic**:
- Uses `compute_kernel_lib::untilize<>()` helper function
- Helper automatically selects optimal path based on width and data type
- Requires `compute_kernel_hw_startup()` before use

### Compute Kernel (Pack Untilize)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| pack_untilize_variable_num_blocks.cpp | RISCV_2 | N/A | CB_in (c_0) | CB_out (c_16) | Untilize via helper lib |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`

**Key Logic**:
- Same as standard kernel - uses unified `compute_kernel_lib::untilize<>()` helper
- Helper detects data format and DEST limit automatically
- Selects pack_untilize (hardware-accelerated) when possible

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_multi_core.cpp | RISCV_1 | NOC1 | CB_out (c_16) | DRAM/L1 | Write rows with complex indexing |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Key Logic**:
- Handles both interleaved and sharded output via `#ifdef SHARDED`
- Processes block by block, row by row
- Complex logic to handle input/output block width mismatches
- Writes partial rows when input block spans multiple output blocks
- Uses barrier per block (not per row) for efficiency

## Implementation Notes

### Compute Path Selection

The program factory selects between two compute kernels:

```cpp
if (!use_pack_untilize || a.dtype() == DataType::UINT16 ||
    (a.dtype() == DataType::FLOAT32 && num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH)) {
    // Use standard untilize (slower but handles all cases)
    compute_kernel = "untilize_variable_num_blocks.cpp";
} else {
    // Use pack untilize (hardware-accelerated)
    compute_kernel = "pack_untilize_variable_num_blocks.cpp";
}
```

However, both kernels now use the unified `compute_kernel_lib::untilize<>()` helper which internally makes the optimal decision. The factory-level selection may be legacy logic.

### DEST Register Limits

The untilize operation is constrained by DEST register capacity:
- **Half-sync mode (16-bit)**: 8 tiles max
- **Half-sync mode (32-bit)**: 4 tiles max
- **Full-sync mode (16-bit)**: 16 tiles max
- **Full-sync mode (32-bit)**: 8 tiles max

The helper library handles this automatically via `DEST_AUTO_LIMIT` detection.

### FP32 Accumulation Mode

For INT32, UINT32, and FLOAT32 data types:
```cpp
if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
    compute_kernel_defines["DST_ACCUM_MODE"] = "1";
}
```
This enables 32-bit destination accumulation mode for proper handling of 32-bit data types.

### Unpack to Dest Mode

For FP32 destination accumulation:
```cpp
if (fp32_dest_acc_en) {
    unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
}
```
This configures the unpacker to handle FP32 data correctly.

### Shard Orientation

The core iteration order respects shard orientation:
```cpp
bool is_row_major = input_is_sharded ?
    a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR : true;
std::vector<CoreCoord> full_cores = corerange_to_cores(full_compute_core_range, std::nullopt, is_row_major);
```

### Runtime Argument Override

The `override_runtime_arguments()` function updates buffer addresses for program caching:
- For sharded input: Updates CB address via `UpdateDynamicCircularBufferAddress()`
- For interleaved input: Updates runtime arg at index 0 (src_addr)
- Always updates writer runtime arg at index 0 (dst_addr)

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the untilize operation and how does it differ from tilize? What are the different variants (pack_untilize vs standard untilize)?"
   **Reason**: Needed to understand the fundamental operation being implemented
   **Key Findings**: Untilize converts tile format (32x32) back to row-major. pack_untilize is hardware-accelerated via PACK stage, while standard untilize uses UNPACK stage. pack_untilize is preferred when width fits in DEST registers.

2. **Query**: "What is TensorAccessor and TensorAccessorArgs used for in tt-metal kernels?"
   **Reason**: The kernels use TensorAccessor for memory addressing
   **Key Findings**: TensorAccessor maps logical page IDs to physical bank-interleaved addresses. TensorAccessorArgs passes configuration (rank, bank count, etc.) to device kernels.

3. **Query**: "What is the cb_utils create_cb function and how does it configure circular buffers?"
   **Reason**: The factory uses create_cb() for CB creation
   **Key Findings**: create_cb() is a helper that creates CircularBufferConfig with specified page size, count, and data format. Can optionally set globally allocated address for sharded buffers.

4. **Query**: "What is ShardedAddrGen and shard_builder used for in tt-metal?"
   **Reason**: Writer kernel uses ShardedAddrGen for sharded output
   **Key Findings**: ShardedAddrGen generates NoC addresses for pages in sharded tensors. shard_builder provides compile-time and runtime args for ShardedAddrGen.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
   **Reason**: Understanding the unified compute kernel helper
   **Key Information**: Provides single `untilize<>()` function that auto-dispatches to pack_untilize (narrow), block-based pack_untilize (wide integer), or standard untilize (wide float).

2. **Source**: `tt_metal/include/compute_kernel_api/untilize.h`
   **Reason**: Understanding standard untilize API
   **Key Information**: `untilize_init()`, `untilize_block()`, `untilize_uninit()` functions. Uses UNPACK thread to read row-major and convert to tiles, then MATH+PACK to output.

3. **Source**: `tt_metal/include/compute_kernel_api/pack_untilize.h`
   **Reason**: Understanding pack_untilize API
   **Key Information**: `pack_untilize_init()`, `pack_untilize_block()`, `pack_untilize_dest()` functions. Directly converts from DEST register to row-major in L1, more efficient for narrow widths.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding core distribution logic
   **Key Information**: `split_blocks_for_tilize()` distributes blocks across cores with cliff handling for remainder. Returns core ranges and blocks per core counts.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding CB creation helper
   **Key Information**: `create_cb()` wraps CircularBufferConfig creation with page size, count, and optional global allocation for sharded buffers.
