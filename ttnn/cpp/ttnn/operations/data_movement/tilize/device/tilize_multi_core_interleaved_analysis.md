# Tilize Multi-Core Interleaved Implementation Analysis

## Overview

The tilize multi-core interleaved operation converts tensor data from **row-major (stick) layout** to **tile layout** for efficient processing on Tenstorrent hardware. This is a fundamental data transformation operation that prepares data for compute kernels that operate on 32x32 tiles.

**Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Key Use Case**: This operation is essential for hybrid operations that need to:
1. Read row-major interleaved input from DRAM
2. Convert to tile format for compute operations
3. Enable reduction and element-wise operations on tiled data

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (group of tiles forming one row of tiles) |
| **Unit size** | `ntiles_per_block` tiles = `padded_shape[-1] / TILE_WIDTH` |
| **Total units** | `nblocks` = `ceil(total_tiles / ntiles_per_block)` |
| **Loop structure** | Each core processes `nblocks_per_core` blocks, reading 32 sticks (TILE_HEIGHT rows) per block |

A "block" in this context represents one horizontal strip of tiles spanning the full tensor width. Each block consists of:
- **Height**: 32 rows (TILE_HEIGHT)
- **Width**: Full tensor width in tiles (`ntiles_per_block`)

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, H, W, C] or any rank (uses last 2 dimensions) |
| **Dimension convention** | Last dimension is contiguous in memory (row-major) |
| **Tensor layout** | ROW_MAJOR_LAYOUT (stick layout) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16, FLOAT32, or other supported types |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input (padded to tile boundaries) |
| **Dimension convention** | Same logical ordering |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input |

### Layout Transformation Details

**Row-Major (Stick) to Tile Conversion**:
- Input: Rows stored contiguously, each row is a "stick"
- Stick size = `padded_shape[-1] * element_size` bytes
- Output: 32x32 tiles stored contiguously
- Each tile contains 32 consecutive rows, each 32 elements wide
- Tiles are stored in row-major order across the tile grid

**Key Calculation**:
```cpp
block_size_nbytes = padded_shape[-1] * element_size  // One full row in bytes
ntiles_per_block = padded_shape[-1] / TILE_WIDTH     // Tiles per row
```

## Data Flow Pattern

### Stage 1: Reader Kernel (Row-Major Stick Reading)

1. **Pre-compute NOC addresses** for 32 consecutive sticks (one tile height):
   ```cpp
   for (uint32_t j = 0; j < tile_height; j++) {
       base_src_noc_addr[j] = get_noc_addr(stick_id, s);
       stick_id++;
   }
   ```

2. **Reserve CB space** for one block of tiles:
   ```cpp
   cb_reserve_back(cb_id_in0, num_tiles);  // ntiles_per_block
   ```

3. **Read all 32 rows** in parallel using asynchronous NOC reads:
   ```cpp
   for (uint32_t k = 0; k < tile_height; k++) {
       noc_async_read(src_noc_addr, l1_write_addr, width_size);
       l1_write_addr += width_size;  // Next row in L1
   }
   ```

4. **Barrier and push** to signal compute kernel:
   ```cpp
   noc_async_read_barrier();
   cb_push_back(cb_id_in0, num_tiles);
   ```

### Stage 2: Compute Kernel (Tilize)

1. **Initialize** tilize hardware:
   ```cpp
   compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
   ```

2. **Process blocks** using unified tilize helper:
   ```cpp
   tilize(cb_in, ntiles_per_block, cb_out, nblocks_per_core);
   ```

3. **Internal loop** (within tilize helper):
   - Wait for input: `cb_wait_front(icb, block_w)`
   - Reserve output: `cb_reserve_back(ocb, block_w)`
   - Execute hardware tilize: `tilize_block(icb, block_w, ocb)`
   - Push output: `cb_push_back(ocb, block_w)`
   - Pop input: `cb_pop_front(icb, block_w)`

### Stage 3: Writer Kernel (Tile Output)

1. **Single-tile loop** writing tiles sequentially:
   ```cpp
   for (uint32_t i = start_id; i < end_id; ++i) {
       cb_wait_front(cb_id_out, 1);
       noc_async_write_tile(i, s, l1_read_addr);
       noc_async_write_barrier();
       cb_pop_front(cb_id_out, 1);
   }
   ```

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Row-major input staging | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_out | Tiled output staging | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Compute | Writer | Block |

**CB Configuration Details**:
```cpp
// Input CB - holds one row of tiles in row-major format
create_cb(tt::CBIndex::c_0, program, all_cores,
          input_single_tile_size, ntiles_per_block, input_cb_data_format);

// Output CB - holds one row of tiles in tile format
create_cb(tt::CBIndex::c_16, program, all_cores,
          output_single_tile_size, ntiles_per_block, output_cb_data_format);
```

**Important Note**: Both CBs have capacity equal to block size (single-buffered). This means:
- Reader must complete reading a full block before compute can start
- Compute must complete processing before reader can write next block
- No overlap between reader and compute for the same CB slot

## Pipeline Pattern Summary

| Stage | Buffering Type | Overlap Potential |
|-------|---------------|-------------------|
| Read -> Compute | Single-buffered | No overlap within block |
| Compute -> Write | Single-buffered | No overlap within block |

The single-buffered design prioritizes simplicity over throughput. For higher performance, double-buffering could be implemented by increasing CB capacity to `2 * ntiles_per_block`.

## Index Calculations

### Stick ID to NOC Address Mapping

The reader uses `TensorAccessor` to convert logical stick IDs to physical NOC addresses:

```cpp
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

**Stick ID Calculation**:
- `stick_id = start_stick_id + (block_index * TILE_HEIGHT) + row_within_block`
- Each block processes `TILE_HEIGHT` (32) consecutive sticks
- Sticks are contiguous rows in the original tensor

### Tile ID for Writer

The writer uses linear tile indexing:
```cpp
tile_start_id += ntiles_per_block * nblocks_per_core;  // Per core offset
// Writer iterates: start_id, start_id+1, ..., start_id+ntiles-1
```

### Runtime Argument Calculations

**Reader arguments per core**:
```cpp
nblocks_per_core * TILE_HEIGHT    // Total sticks to read
row_start_id                       // Starting stick ID (0-indexed)
```

**Writer arguments per core**:
```cpp
ntiles_per_block * nblocks_per_core  // Total tiles to write
tile_start_id                         // Starting tile ID (0-indexed)
```

## Memory Access Patterns

### Read Pattern (Reader Kernel)

| Aspect | Pattern |
|--------|---------|
| **Access type** | Strided row reads |
| **Stride** | One full row (stick) per NOC read |
| **Burst size** | `block_width_size` bytes per row |
| **NOC operations** | 32 async reads per block |
| **Bank distribution** | Interleaved across DRAM banks |

**Detailed Read Sequence**:
1. Pre-compute 32 NOC addresses for next block
2. Issue 32 parallel async reads (one per row)
3. Each read fetches `block_width_size` bytes
4. Data lands contiguously in L1 CB

### Write Pattern (Writer Kernel)

| Aspect | Pattern |
|--------|---------|
| **Access type** | Sequential tile writes |
| **Granularity** | One tile at a time |
| **NOC operations** | One write per tile with barrier |
| **Bank distribution** | Interleaved across DRAM banks |

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size()` |
| **Total cores** | `ncores` = min(available, needed for work) |
| **Work per core** | `nblocks_per_core` blocks (may have cliff) |
| **Load balancing** | Near-equal with single cliff core |

### Work Split Algorithm

The `split_blocks_for_tilize` function distributes blocks:

```cpp
nblocks_per_core = ceil(nblocks / grid_area);
ncores = ceil(nblocks / nblocks_per_core);
nblocks_per_core_cliff = nblocks % nblocks_per_core;  // Remainder for last core
```

**Core Range Structure**:
- `core_range`: Cores with full `nblocks_per_core` blocks
- `core_range_cliff`: Single core with `nblocks_per_core_cliff` blocks (if remainder exists)

### Block vs 2D Block Program Factory Selection

The factory includes logic to choose between 1D and 2D block distribution:
```cpp
if (num_tiles_per_row > threshold_row_block) {
    // Consider 2D block distribution (TilizeMultiCoreBlockProgramFactory)
    if (ncores < ncores_block) {
        return TilizeMultiCoreBlockProgramFactory::create(...);
    }
}
```

This optimization uses 2D blocking for wide tensors to improve core utilization.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one row in bytes (`block_size_nbytes`) |
| 1+ | TensorAccessorArgs | varies | Buffer addressing parameters (bank info, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output CB ID (c_16) |
| 1+ | TensorAccessorArgs | varies | Buffer addressing parameters |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | nblocks_per_core | uint32_t | Number of blocks to process |
| 1 | ntiles_per_block | uint32_t | Tiles per block (row width in tiles) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total sticks to read (`nblocks * TILE_HEIGHT`) |
| 2 | (unused) | uint32_t | Reserved (was stick_size) |
| 3 | num_tiles_per_block | uint32_t | Tiles per row |
| 4 | block_width_size | uint32_t | Bytes per row |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for this factory |
| 6 | num_leftover_tiles | uint32_t | Always 0 for this factory |
| 7 | leftover_width_in_row | uint32_t | Always 0 for this factory |
| 8 | start_stick_id | uint32_t | First stick ID for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_tiles | uint32_t | Total tiles to write |
| 2 | start_id | uint32_t | First tile ID for this core |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (sticks) | CB c_0 | Read 32 rows, pack into block |

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

**Key Logic**:
- Uses `TensorAccessor` for address computation from stick IDs
- Pre-computes 32 NOC addresses before issuing reads
- Reads entire block width in each NOC operation
- Data arrives in CB in row-major format (32 contiguous rows)

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_2 (Unpack/Math/Pack) | N/A | CB c_0 | CB c_16 | Hardware tilize |

**File**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp`

**Key Logic**:
- Uses `compute_kernel_lib::tilize()` helper from `tilize_helpers.hpp`
- Hardware performs row-major to tile format conversion
- Processes `nblocks_per_core` blocks of `ntiles_per_block` tiles each
- Init/uninit handled by helper function

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_16 | DRAM (tiles) | Write tiles sequentially |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Key Logic**:
- Single-tile granularity writes
- Uses `TensorAccessor` with tile-sized pages
- Sequential tile ID iteration from `start_id`
- Includes `noc_async_write_barrier()` per tile

## Implementation Notes

### Why Single-Buffered CBs

The implementation uses single-buffered CBs (capacity = block size) rather than double-buffered. This design choice:
- **Simplifies synchronization**: No complex overlap management
- **Reduces L1 memory usage**: Important for operations with many CBs
- **Adequate for DRAM-bound operations**: Tilize is often memory-limited

### TensorAccessor Usage for Row-Major Data

For row-major (stick) data, the TensorAccessor is configured with:
- **Page size** = stick size (one full row in bytes)
- **Page ID** = stick ID (row index)

This differs from tile-based accessors where page size equals tile size.

### Compute Kernel Hardware Startup

The tilize operation requires explicit hardware initialization:
```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
```

This configures the unpack/pack hardware for the specific CB pair before any tilize operations.

### FP32 Accumulation Mode

When input dtype is FLOAT32, the compute kernel enables FP32 destination accumulation:
```cpp
bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;
ComputeConfig{.fp32_dest_acc_en = fp32_llk_acc, ...}
```

### Runtime Argument Override

The `override_runtime_arguments` function efficiently updates only buffer addresses:
```cpp
runtime_args[0] = src_buffer->address();  // Reader
runtime_args[0] = dst_buffer->address();  // Writer
```

This enables buffer address updates without program recompilation.

## Usage as Input Stage Reference

For hybrid operations requiring tilize as an input stage:

### Key Parameters to Extract

1. **Stick size**: `padded_shape[-1] * element_size`
2. **Tiles per row**: `padded_shape[-1] / TILE_WIDTH`
3. **CB capacity**: At minimum `ntiles_per_block` tiles

### Integration Pattern

```cpp
// 1. Create input CB for row-major data
create_cb(tt::CBIndex::c_0, program, all_cores,
          input_tile_size, ntiles_per_block, input_data_format);

// 2. Create intermediate CB for tiled data (for compute)
create_cb(tt::CBIndex::c_1, program, all_cores,
          tile_size, capacity_for_compute, data_format);

// 3. Reader reads sticks into c_0
// 4. Compute tilizes from c_0 to c_1, then performs operations
// 5. Writer outputs from final CB
```

### Reader Runtime Args Template

```cpp
const std::array reader_rt_args = {
    src_buffer->address(),           // Buffer address
    num_sticks,                       // Total sticks (rows)
    stick_size_bytes,                 // Bytes per row
    tiles_per_row,                    // ntiles_per_block
    row_width_bytes,                  // Same as stick_size
    1u,                               // Full blocks in row
    0u,                               // Leftover tiles
    0u,                               // Leftover width
    start_stick_id                    // Row offset for this core
};
```

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the TensorAccessor API and how is it used for reading data from DRAM in reader kernels?"
   **Reason**: Needed to understand how stick IDs are converted to NOC addresses
   **Key Findings**: TensorAccessor abstracts bank distribution, `get_noc_addr(page_id, accessor)` computes physical addresses from logical page IDs

2. **Query**: "How does tilize_block work in the compute kernel API?"
   **Reason**: Needed to understand the hardware tilize operation
   **Key Findings**: tilize_block uses LLK functions to reorder row-major data into 32x32 tiles, involves unpack -> datacopy -> pack pipeline

3. **Query**: "What is the difference between stick layout and tile layout in tt-metal?"
   **Reason**: Needed to understand the fundamental data organization difference
   **Key Findings**: Stick = one row stored contiguously; Tile = 32x32 block stored contiguously; conversion groups 32 consecutive rows

4. **Query**: "How do circular buffers work in tt-metal?"
   **Reason**: Needed to understand producer-consumer synchronization
   **Key Findings**: reserve_back/push_back for producer, wait_front/pop_front for consumer; capacity vs block size determines buffering type

### Documentation References

1. **Source**: `/localdev/mstaletovic/tt-metal/METALIUM_GUIDE.md` (lines 50-150)
   **Reason**: Understanding overall Tensix architecture and kernel types
   **Key Information**: Three kernel types (reader, compute, writer), CB-based synchronization, SPMD execution model

2. **Source**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding block distribution algorithm
   **Key Information**: `split_blocks_for_tilize` function, BlockSplit struct, cliff core handling

3. **Source**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
   **Reason**: Understanding compute kernel tilize implementation
   **Key Information**: Unified `tilize()` helper function, template parameters for different modes (fast, DT, skip_wait)

4. **Source**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding CB creation helper
   **Key Information**: `create_cb()` utility simplifies CB configuration with page size and capacity
