# Tilize Operation Implementation Analysis

## Overview

The **tilize** operation converts tensor data from row-major layout to tile layout. This is a fundamental data transformation operation in TTNN that reorganizes data into 32x32 tile blocks to align with Tenstorrent hardware's native processing format. The hardware's matrix engine operates on 16x16 "faces" within these tiles, making tilization essential for efficient computation.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp`

The operation provides **four distinct program variants** to handle different memory configurations and performance requirements:

1. **Single-Core** (`tilize_single_core`): Simple single-core execution for small tensors or low-performance mode
2. **Multi-Core Interleaved** (`tilize_multi_core_interleaved`): Distributes work across cores for interleaved memory tensors
3. **Multi-Core Block** (`tilize_multi_core_block`): 2D block distribution for wide/tall tensors
4. **Multi-Core Sharded** (`tilize_multi_core_sharded`): Optimized for height-sharded input tensors

## Work Unit Definition

The fundamental work unit is a **block of tiles** that corresponds to 32 rows of input data (one tile height). Each block contains `num_tiles_per_row` tiles horizontally.

- **Single-Core/Multi-Core Interleaved**: Work unit = one tile row (32 input rows producing `num_tiles_per_row` output tiles)
- **Multi-Core Block**: Work unit = a 2D block of tiles (square block of size `single_block_size` x `single_block_size` tiles)
- **Multi-Core Sharded**: Work unit = entire shard (all tiles within a core's shard)

## Tensor Format and Layout

### Input Tensor

| Attribute | Value |
|-----------|-------|
| **Logical Shape** | `[N, C, H, W]` (4D) or higher dimensions squeezed to 4D |
| **Dimension Convention** | NCHW (batch, channels, height, width) |
| **Tensor Layout** | `ROW_MAJOR_LAYOUT` (linear row-by-row storage) |
| **Memory Layout** | `INTERLEAVED` (round-robin across banks) or `HEIGHT_SHARDED` |
| **Buffer Type** | DRAM or L1 |
| **Data Type** | BFLOAT16, FLOAT32, UINT32, INT32, or UINT16 |

**Constraints**:
- Physical volume must be divisible by TILE_HW (1024 = 32 x 32)
- Stick size (row width in bytes) must be divisible by 2
- Sharded inputs must use HEIGHT_SHARDED with ROW_MAJOR shard orientation

### Output Tensor

| Attribute | Value |
|-----------|-------|
| **Logical Shape** | Same as input |
| **Padded Shape** | Same as input (pre-padded to tile boundaries) |
| **Tensor Layout** | `TILE_LAYOUT` (32x32 tiles) |
| **Memory Layout** | Same as input (INTERLEAVED or HEIGHT_SHARDED) |
| **Buffer Type** | Same as input |
| **Data Type** | Same as input or specified `output_dtype` |

### Layout Transformation

The tilize operation performs the following transformation:

```
Row-Major Input (example 64x64):
[row0: elem0, elem1, ..., elem63]
[row1: elem0, elem1, ..., elem63]
...
[row63: elem0, elem1, ..., elem63]

Tile Layout Output (2x2 tiles):
Tile(0,0): 32x32 elements from rows 0-31, cols 0-31
Tile(0,1): 32x32 elements from rows 0-31, cols 32-63
Tile(1,0): 32x32 elements from rows 32-63, cols 0-31
Tile(1,1): 32x32 elements from rows 32-63, cols 32-63
```

Each 32x32 tile is further subdivided into four 16x16 "faces" for the hardware matrix engine.

## Data Flow Pattern

### Single-Core / Multi-Core Interleaved Flow

```
DRAM (row-major sticks)
         |
         v
    [Reader Kernel]
    - Reads 32 rows at a time (one tile height)
    - Each row = stick_size bytes
    - Reads block_width_size bytes from each row
    - Stores in tilize-friendly order in CB
         |
         v
    CB_0 (Input CB)
         |
         v
    [Compute Kernel]
    - tilize_init(): Initialize unpacker/packer
    - tilize_block(): Convert row-major to tile format
    - Processes num_tiles_per_block tiles per iteration
         |
         v
    CB_16 (Output CB)
         |
         v
    [Writer Kernel]
    - Reads tiles one at a time
    - Writes to output buffer using tile addressing
         |
         v
DRAM (tile layout)
```

### Multi-Core Block Flow

Similar to interleaved but with 2D block distribution across cores. Each core processes a rectangular region of the input tensor:

```
Input Tensor 2D Grid:
+--------+--------+--------+
| Core 0 | Core 1 | Core 2 | <- cliff_row cores
+--------+--------+--------+
| Core 3 | Core 4 | Core 5 |
+--------+--------+--------+
| Core 6 | Core 7 | Core 8 | <- cliff_col cores
+--------+--------+--------+
                    ^
              cliff_col_row core
```

### Multi-Core Sharded Flow

For sharded tensors, data is already distributed across cores in L1:

```
L1 Shard (row-major)
         |
         v
    [Reader Kernel]
    - Just pushes shard to CB (data already in L1)
    - cb_push_back(num_tiles_per_shard)
         |
         v
    CB_0 (Input CB, backed by shard buffer)
         |
         v
    [Compute Kernel]
    - Same tilize_block() processing
         |
         v
    CB_16 (Output CB, backed by output shard buffer)
         |
         v
    [Writer Kernel]
    - Just waits for output (already in L1)
    - cb_wait_front(num_tiles_per_shard)
         |
         v
L1 Shard (tile layout)
```

## Circular Buffer Configuration

### CB_0 (c_0): Input Circular Buffer

| Variant | Size (tiles) | Size (bytes) | Producer | Consumer | Block Size | Buffering |
|---------|--------------|--------------|----------|----------|------------|-----------|
| Single-Core | num_tiles_per_block | num_tiles_per_block * input_tile_size | Reader | Compute | num_tiles_per_block | Single |
| Multi-Core Interleaved | ntiles_per_block | ntiles_per_block * input_tile_size | Reader | Compute | ntiles_per_block | Single |
| Multi-Core Block | single_block_size (varies by region) | single_block_size * input_tile_size | Reader | Compute | single_block_size | Single |
| Multi-Core Sharded | num_tiles_per_shard | num_tiles_per_shard * input_tile_size | Reader | Compute | num_tiles_per_shard | Single |

**Purpose**: Holds row-major input data staged for tilization. Data is arranged such that 32 consecutive rows (one tile height worth) are stored contiguously, enabling efficient tilize_block() processing.

**Usage Pattern**: Reader->Compute (Reader pushes row-major data, Compute consumes and tilizes)

### CB_16 (c_16): Output Circular Buffer

| Variant | Size (tiles) | Size (bytes) | Producer | Consumer | Block Size | Buffering |
|---------|--------------|--------------|----------|----------|------------|-----------|
| Single-Core | num_tiles_per_block | num_tiles_per_block * output_tile_size | Compute | Writer | 1 tile | Single |
| Multi-Core Interleaved | ntiles_per_block | ntiles_per_block * output_tile_size | Compute | Writer | 1 tile | Single |
| Multi-Core Block | single_block_size (varies by region) | single_block_size * output_tile_size | Compute | Writer | 1 tile | Single |
| Multi-Core Sharded | num_tiles_per_shard | num_tiles_per_shard * output_tile_size | Compute | Writer | num_tiles_per_shard | Single |

**Purpose**: Holds tilized output data ready for writing to destination buffer.

**Usage Pattern**: Compute->Writer (Compute pushes tilized tiles, Writer writes to DRAM/L1)

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering | Overlap Possible? |
|----|----------|------------|-----------|-------------------|
| CB_0 | num_tiles_per_block | num_tiles_per_block | Single-buffered | No - capacity equals block size |
| CB_16 | num_tiles_per_block | 1 (writer) / num_tiles_per_block (compute) | Single-buffered | Partial - writer pops 1 tile at a time |

**Note**: The single-buffered design means compute must wait for reader to complete a full block before processing, and writer must wait for compute to complete before writing. However, within a block, writer can begin writing tiles as soon as compute produces them (tile-by-tile consumption).

## Index Calculations

### Stick/Row Addressing

Row-major input data is addressed using stick (row) indices:

```cpp
// Stick index within input tensor
uint32_t stick_id = start_stick_id + (block_idx * TILE_HEIGHT) + row_within_block;

// NOC address for reading stick from DRAM
uint64_t src_noc_addr = get_noc_addr(stick_id, tensor_accessor);
```

### Tile Index Calculation (Multi-Core Interleaved)

```cpp
// tile_start_id for each core
tile_start_id = core_idx * nblocks_per_core * ntiles_per_block;

// row_start_id for each core
row_start_id = core_idx * nblocks_per_core * TILE_HEIGHT;
```

### 2D Block Addressing (Multi-Core Block)

```cpp
// For core at position (row_idx, col_idx) in core grid:
start_row_id = row_idx * single_block_size_col * TILE_HEIGHT;
start_column_id = col_idx * single_block_size_row * TILE_WIDTH * element_size;

// Tile start ID considers 2D positioning
tile_start_id = row_idx * single_block_size_col * total_tiles_per_row +
                col_idx * single_block_size_row;
```

## Memory Access Patterns

### Read Pattern (Reader Kernel)

**Single-Core/Multi-Core Interleaved** (`reader_unary_stick_layout_split_rows_interleaved.cpp`):
- Reads 32 sticks (rows) per tile block
- For each stick, reads `block_width_size` bytes from current column offset
- Advances stick_id linearly through input tensor
- Uses `noc_async_read` with barrier after each 32-row block

```cpp
for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
    // Get base addresses for 32 consecutive rows
    for (uint32_t j = 0; j < tile_height; j++) {
        base_src_noc_addr[j] = get_noc_addr(stick_id++, s);
    }
    // Read each block of tiles from these 32 rows
    for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
        read_tiles(num_tiles_per_block, block_width_size);
    }
}
```

**Multi-Core Block** (`reader_unary_pad_multicore_both_dims.cpp`):
- Reads rows within assigned 2D region
- Handles padding for partial tiles at boundaries
- Processes column dimension within each tile row

### Write Pattern (Writer Kernel)

**Interleaved** (`writer_unary_interleaved_start_id.cpp`):
- Writes one tile at a time
- Sequential tile indices from `start_id` to `start_id + num_tiles`
- Uses `noc_async_write_tile` with per-tile barrier

**Block** (`writer_unary_interleaved_start_id_wh.cpp`):
- Writes tiles in nested loop: third_dim -> column -> row
- Calculates tile index as: `start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r`

**Sharded** (`writer_unary_sharded.cpp`):
- No actual write needed - just waits for output CB
- Data already positioned in L1 output shard

## Core Distribution Strategy

### Single-Core

Uses core (0,0) or first available core from `sub_core_grids`. All work processed on single core.

### Multi-Core Interleaved

Uses `split_blocks_for_tilize()` to distribute tile blocks:

```cpp
BlockSplit {
    ncores,           // Total cores used
    all_cores,        // CoreRangeSet of all participating cores
    core_range,       // Full cores (process nblocks_per_core blocks)
    core_range_cliff, // Cliff core (processes remainder blocks)
    nblocks_per_core,
    nblocks_per_core_cliff
}
```

**Distribution Algorithm**:
1. Calculate `nblocks_per_core = ceil(nblocks / grid_area)`
2. Calculate `ncores = ceil(nblocks / nblocks_per_core)`
3. Last core may be "cliff" core with fewer blocks (`nblocks_per_core_cliff = nblocks % nblocks_per_core`)

### Multi-Core Block

Uses `split_blocks_for_tilize_wh()` for 2D distribution:

```cpp
BlockSplitWH {
    ncores,
    all_cores,
    core_range,              // Full block cores
    cliff_row_core_range,    // Right edge (partial width)
    cliff_col_core_range,    // Bottom edge (partial height)
    cliff_col_row_core_range,// Corner (partial width and height)
    nblocks_per_core,
    single_block_size,       // Square block dimension
    single_block_size_cliff_row,
    single_block_size_cliff_col,
    has_cliff_row,
    has_cliff_col,
    full_cores_per_row,
    full_cores_per_col
}
```

**Distribution Algorithm**:
1. Find optimal square block size using `closest_square_larger_than_b()`
2. Divide tensor width by block_size -> cores per row (+ cliff if remainder)
3. Divide tensor height by block_size -> cores per column (+ cliff if remainder)
4. Creates 4 distinct core region types for handling edge cases

### Multi-Core Sharded

Uses shard specification's `grid` directly - each core processes its own shard independently.

## Arguments

### Compile-Time Arguments

**Reader Kernel** (`reader_unary_stick_layout_split_rows_interleaved.cpp`):
| Index | Name | Description |
|-------|------|-------------|
| 0 | stick_size | Size of one input row in bytes |
| 1+ | TensorAccessorArgs | Source buffer access parameters |

**Writer Kernel** (`writer_unary_interleaved_start_id.cpp`):
| Index | Name | Description |
|-------|------|-------------|
| 0 | output_cb_index | Output circular buffer ID (16) |
| 1+ | TensorAccessorArgs | Destination buffer access parameters |

**Compute Kernel** (`tilize.cpp`):
| Index | Name | Description |
|-------|------|-------------|
| 0 | per_core_block_cnt | Number of tile blocks to process |
| 1 | per_core_block_tile_cnt | Tiles per block (row width in tiles) |

**Compute Kernel** (`tilize_wh.cpp` for multi-core block):
| Index | Name | Description |
|-------|------|-------------|
| 0 | block_size_col | Column block size in tiles |
| 1 | block_size_row | Row block size in tiles |
| 2 | third_dim | Batch/channel dimension iterations |

### Runtime Arguments

**Reader Kernel**:
| Index | Name | Description |
|-------|------|-------------|
| 0 | src_addr | Source buffer DRAM address |
| 1 | num_sticks | Total number of rows to read |
| 2 | stick_size | Row size in bytes |
| 3 | num_tiles_per_block | Tiles per block |
| 4 | block_width_size | Block width in bytes |
| 5 | num_full_blocks_in_row | Full blocks per row |
| 6 | num_leftover_tiles | Partial block tiles |
| 7 | leftover_width_in_row | Partial block width |
| 8 | start_stick_id | Starting row index |

**Writer Kernel**:
| Index | Name | Description |
|-------|------|-------------|
| 0 | dst_addr | Destination buffer DRAM address |
| 1 | num_tiles | Total tiles to write |
| 2 | start_id | Starting tile index |

## Kernel Implementations

### Reader Kernel: reader_unary_stick_layout_split_rows_interleaved.cpp

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Responsibilities**:
  - Read row-major input data from DRAM
  - Stage data in CB_0 in tilize-friendly format (32 rows per tile height)
- **Input**: DRAM buffer containing row-major tensor data
- **Output**: CB_0 with staged row-major blocks
- **Key Logic**:
  - Iterates over tile blocks (32 rows each)
  - For each block, reads block_width_size bytes from each of 32 rows
  - Uses `read_tiles` lambda to batch reads with NOC barrier
  - Advances through input tensor row by row

### Reader Kernel: reader_unary_pad_multicore_both_dims.cpp

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp`
- **Responsibilities**:
  - Read 2D block region assigned to this core
  - Handle padding for partial tiles at tensor boundaries
- **Input**: DRAM buffer, 2D region specification
- **Output**: CB_0 with block data (includes padding if needed)
- **Key Logic**:
  - Processes rows within assigned column/row range
  - Fills padding rows with pad_value for boundary cases
  - Supports 3D iteration (third_dim batching)

### Reader Kernel: reader_unary_sharded.cpp

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`
- **Responsibilities**: Signal that sharded input data is ready
- **Input**: L1 shard buffer (CB backed by shard)
- **Output**: CB_0 pushed with num_tiles_per_shard
- **Key Logic**: Single `cb_push_back` - data already in L1

### Compute Kernel: tilize.cpp

- **File**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp`
- **Responsibilities**: Convert row-major data blocks to tile format
- **Input**: CB_0 (row-major blocks)
- **Output**: CB_16 (tilized tiles)
- **Key Logic**:
  ```cpp
  compute_kernel_hw_startup(cb_in, cb_out);
  compute_kernel_lib::tilize(cb_in, per_core_block_tile_cnt, cb_out, per_core_block_cnt);
  ```
  - Uses `tilize_helpers.hpp` library for unified tilize implementation
  - Processes `per_core_block_cnt` blocks of `per_core_block_tile_cnt` tiles each

### Compute Kernel: tilize_wh.cpp

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp`
- **Responsibilities**: Tilize with 2D block and batch support
- **Input**: CB_0 (row-major blocks)
- **Output**: CB_16 (tilized tiles)
- **Key Logic**:
  ```cpp
  compute_kernel_hw_startup(cb_in, cb_out);
  compute_kernel_lib::tilize(cb_in, block_size_row, cb_out, block_size_col * third_dim);
  ```
  - Block sizes may differ for row vs column dimension
  - third_dim provides batch iteration

### Writer Kernel: writer_unary_interleaved_start_id.cpp

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Responsibilities**: Write tilized tiles to DRAM output buffer
- **Input**: CB_16 (tilized tiles)
- **Output**: DRAM buffer in tile layout
- **Key Logic**:
  - Iterates from `start_id` to `start_id + num_tiles`
  - For each tile: wait, read from CB, write via NOC, barrier, pop
  - Uses `noc_async_write_tile` for tile-addressed writes

### Writer Kernel: writer_unary_interleaved_start_id_wh.cpp

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp`
- **Responsibilities**: Write tiles with 2D block indexing
- **Input**: CB_16 (tilized tiles)
- **Output**: DRAM buffer in tile layout
- **Key Logic**:
  - Triple nested loop: third_dim, column block, row block
  - Computes tile index from 2D position within tensor

### Writer Kernel: writer_unary_sharded.cpp

- **File**: `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`
- **Responsibilities**: Wait for sharded output completion
- **Input**: CB_16 (backed by output shard buffer)
- **Output**: L1 shard (data already in place)
- **Key Logic**: Single `cb_wait_front` - no actual write needed

## Implementation Notes

### Performance Mode Selection

The operation automatically selects the best implementation variant based on:

1. **Single-Core** conditions:
   - `use_low_perf = true`
   - `use_multicore = false`
   - `sub_core_grids` has fewer than 2 cores

2. **Multi-Core Sharded**: When input is height-sharded

3. **Multi-Core Block** vs **Multi-Core Interleaved**:
   - Block mode chosen when `num_tiles_per_row > 32` AND either:
     - `num_tiles_per_col > 32`, OR
     - `num_tiles_per_row > num_tiles_per_col`
   - Block mode also requires that it would use more cores than interleaved

### Block Size Optimization (Single-Core)

When `use_low_perf = false`, the single-core variant optimizes block size:

```cpp
// Find largest block size that:
// 1. Fits in available L1 (max_tiles based on L1 budget)
// 2. Evenly divides tiles per row
for (uint32_t n_t = max_tiles; n_t > 0; n_t--) {
    if (num_tiles_in_row % n_t == 0) {
        num_tiles_per_block = n_t;
        break;
    }
}
```

### FP32 Accumulation

When input dtype is FLOAT32, the compute kernel enables FP32 destination accumulator:
```cpp
bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;
// Passed to ComputeConfig::fp32_dest_acc_en
```

### Runtime Argument Override

All variants provide an `override_runtime_args_callback` that updates buffer addresses when tensor buffers are reallocated, avoiding program recompilation for address changes.

### N-Dimensional Support

The public API (`ExecuteTilize::invoke`) handles tensors with rank > 4 by:
1. Squeezing to 4D before tilization
2. Unsqueezing back to original shape after tilization

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the tilize operation in TTNN and how does it work? What is the purpose of converting from row-major layout to tile layout, and what are the tile dimensions (TILE_HEIGHT and TILE_WIDTH)?"
   **Reason**: Needed to understand the fundamental purpose of tilization and the standard tile dimensions
   **Key Findings**:
   - Tiles are 32x32 (TILE_HEIGHT = TILE_WIDTH = 32)
   - Each tile subdivided into four 16x16 "faces" for matrix engine
   - Tilization aligns data with hardware's native processing format

2. **Query**: "How do circular buffers work in TT-Metal? What are the key APIs like cb_reserve_back, cb_push_back, cb_wait_front, and cb_pop_front and how do they coordinate between data movement and compute kernels?"
   **Reason**: Needed to understand producer-consumer synchronization between kernels
   **Key Findings**:
   - cb_reserve_back/cb_push_back for producers (reader/compute output)
   - cb_wait_front/cb_pop_front for consumers (compute input/writer)
   - Blocking calls that synchronize kernel execution

3. **Query**: "What is the TensorAccessor API in TT-Metal device kernels? How does TensorAccessorArgs work and what does get_noc_addr do with TensorAccessor?"
   **Reason**: Needed to understand how tensor memory addressing works in kernels
   **Key Findings**:
   - TensorAccessor maps logical page indices to physical NOC addresses
   - TensorAccessorArgs configures compile-time vs runtime parameters
   - get_noc_addr calculates physical address for given page/tile index

4. **Query**: "What is the tilize_block function in compute kernels? How does it transform row-major data into tile format and what are the compute_kernel_api/tilize.h functions like tilize_init and tilize_block?"
   **Reason**: Needed to understand the actual tilization compute operation
   **Key Findings**:
   - tilize_init configures unpacker/math/packer for tilization
   - tilize_block performs actual data reordering via LLK APIs
   - Uses llk_unpack_tilize_block and llk_pack for hardware-optimized transformation

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
   **Reason**: Understanding the compute kernel library used by tilize
   **Key Information**: Unified tilize function with template parameters for init/uninit, fast mode, data type reconfiguration. Handles variable subblock patterns.

2. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding core distribution algorithms
   **Key Information**: `split_blocks_for_tilize()` for 1D distribution, `split_blocks_for_tilize_wh()` for 2D distribution with cliff region handling.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding CB creation helper functions
   **Key Information**: `create_cb()` helper that simplifies circular buffer configuration with optional globally-allocated buffer backing.
