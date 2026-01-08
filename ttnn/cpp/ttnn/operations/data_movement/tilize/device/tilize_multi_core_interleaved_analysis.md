# Tilize Multi-Core Interleaved Implementation Analysis

## Overview

The **tilize** operation converts tensor data from **row-major layout** to **tile layout** (32x32 tiles). This transformation is essential because Tenstorrent hardware natively operates on 32x32 element tiles, which optimizes common deep learning operations like matrix multiplication and convolution. Tiled layouts organize data so that elements within a tile are stored contiguously, dramatically reducing memory footprint needed for computation.

**Program Factory Path**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

This specific variant handles:
- **Multi-core execution**: Work is distributed across multiple Tensix cores
- **Interleaved memory layout**: Input and output tensor pages are distributed round-robin across DRAM banks

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (row of tiles) |
| **Unit size** | `ntiles_per_block` tiles (tensor width in tiles) |
| **Total units** | `nblocks = ceil(ntiles / ntiles_per_block)` |
| **Loop structure** | Outer: blocks (rows of tiles), Inner: tile-height sticks per block |

A **block** in this implementation represents one row of tiles (32 rows of data = TILE_HEIGHT). The number of tiles per block equals `padded_shape[-1] / TILE_WIDTH` (i.e., the number of tiles along the width dimension).

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, ..., H, W] (arbitrary batch dimensions) |
| **Dimension convention** | Last two dimensions are H, W |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, ..., H_padded, W_padded] |
| **Dimension convention** | Last two dimensions padded to tile multiples |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input |

### Layout Transformations

The tilize operation performs the following transformation:
1. **Input**: Row-major data where each row is stored contiguously
2. **Output**: Tiled data organized as 32x32 tiles, where each tile contains four 16x16 "faces" (sub-tiles) stored in row-major order within the tile

The reader kernel reads 32 consecutive rows (stick-height = TILE_HEIGHT) of width data, placing them into the input CB. The compute kernel then rearranges this data into tile format using the hardware tilize functionality.

## Data Flow Pattern

```
DRAM (Row-Major) --> Reader Kernel --> CB_0 (Row-Major Sticks)
                                           |
                                           v
                                    Compute Kernel
                                    (Tilize Transform)
                                           |
                                           v
                     CB_16 (Tile Format) --> Writer Kernel --> DRAM (Tile Format)
```

### Step-by-Step Flow

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (row-major sticks) | CB_0 | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB_0 | CB_16 | `cb_wait_front`, `cb_reserve_back`, `tilize_block`, `cb_push_back`, `cb_pop_front` |
| 3 | Writer | CB_16 | DRAM (tiles) | `cb_wait_front`, `noc_async_write_tile`, `cb_pop_front` |

### Reader Logic Details

The reader kernel (`reader_unary_stick_layout_split_rows_interleaved.cpp`):
1. For each block (group of TILE_HEIGHT=32 rows):
   - Precomputes base NoC addresses for all 32 rows using `get_noc_addr(stick_id, s)`
   - Reserves space in CB_0 for `num_tiles_per_block` tiles
   - For each of the 32 rows: issues `noc_async_read` to fetch the full row width
   - Waits for all reads via `noc_async_read_barrier()`
   - Pushes the block to CB_0

### Compute Logic Details

The compute kernel (`tilize.cpp`) uses the helper library:
1. Calls `compute_kernel_hw_startup()` to initialize hardware
2. Calls `compute_kernel_lib::tilize()` which:
   - Initializes tilize hardware via `tilize_init()`
   - For each block: waits for input, reserves output, calls `tilize_block()`, pushes output, pops input
   - Uninitializes via `tilize_uninit()`

### Writer Logic Details

The writer kernel (`writer_unary_interleaved_start_id.cpp`):
1. For each output tile:
   - Waits for one tile in CB_16
   - Gets L1 read pointer
   - Writes tile to DRAM using `noc_async_write_tile()`
   - Waits for write completion
   - Pops the tile from CB_16

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input staging (row-major sticks) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_out | Output staging (tilized data) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Compute | Writer | Block |

**Notes**:
- Both CBs have capacity equal to block size, meaning single-buffered operation
- The `input_single_tile_size` is calculated via `tt::tile_size(input_cb_data_format)`
- The `output_single_tile_size` is calculated similarly for the output data format
- CB total size = `num_pages * page_size` = `ntiles_per_block * tile_size`

## Pipeline Pattern Summary

| Pattern | Description |
|---------|-------------|
| **Buffering Type** | Single-buffered (capacity = block size for both CBs) |
| **Overlap Potential** | Limited - reader must complete full block before compute starts |
| **Bottleneck** | Likely reader (DRAM reads) or compute (tilize transformation) |

With single-buffered CBs, the pipeline operates in lock-step:
1. Reader fills CB_0 with one block
2. Compute processes CB_0, produces to CB_16
3. Writer drains CB_16 tile-by-tile

## Index Calculations

### Reader Index Mapping

The reader uses `TensorAccessor` for address calculation:
```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// For each stick:
uint64_t noc_addr = get_noc_addr(stick_id, s);
```

- `stick_id` is the linear row index starting from `start_stick_id`
- The TensorAccessor maps `stick_id` to the appropriate DRAM bank and offset
- Interleaved layout distributes pages round-robin across banks

### Writer Index Mapping

The writer uses `TensorAccessor` for tile addressing:
```cpp
constexpr auto dst_args = TensorAccessorArgs<1>();
const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);
// For each tile:
noc_async_write_tile(i, s, l1_read_addr);
```

- `i` is the tile index starting from `start_id`
- Tiles are written sequentially to interleaved DRAM

## Memory Access Patterns

### Read Pattern

| Aspect | Description |
|--------|-------------|
| **Ordering** | Sequential within block (32 consecutive rows) |
| **Stride** | Row-to-row (contiguous sticks within a block) |
| **Granularity** | Full row width per read (`block_width_size` bytes) |
| **Bank Access** | Round-robin across DRAM banks (interleaved) |

The reader pre-computes 32 NoC addresses (one per row) and issues all reads before barrier.

### Write Pattern

| Aspect | Description |
|--------|-------------|
| **Ordering** | Sequential tiles within assigned range |
| **Stride** | Tile-by-tile (single tile per iteration) |
| **Granularity** | One tile per write |
| **Bank Access** | Round-robin across DRAM banks (interleaved) |

Writer processes one tile at a time with a barrier after each write.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear assignment from available grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (or `nblocks_per_core_cliff` for last core) |
| **Load balancing** | Near-equal with cliff handling |

### Work Splitting Algorithm

The `split_blocks_for_tilize()` function in `work_split_tilize.hpp`:

1. Calculates `nblocks_per_core = ceil(nblocks / grid_area)`
2. Calculates `ncores = ceil(nblocks / nblocks_per_core)`
3. Calculates `nblocks_per_core_cliff = nblocks % nblocks_per_core`
4. Assigns:
   - `ncores - 1` cores get `nblocks_per_core` blocks each
   - Last core (cliff core) gets `nblocks_per_core_cliff` blocks (if remainder exists)

### Block Variant Selection

The program factory includes logic to potentially switch to a 2D block variant (`TilizeMultiCoreBlockProgramFactory`) when:
- `num_tiles_per_row > 32` (threshold_row_block)
- AND (`num_tiles_per_col > 32` OR `num_tiles_per_row > num_tiles_per_col`)
- AND the block variant would use more cores than the simple 1D variant

This optimization improves parallelism for wide tensors.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one row in bytes (`block_size_nbytes`) |
| 1+ | TensorAccessorArgs | varies | Tensor accessor configuration for source buffer |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer ID (c_16) |
| 1+ | TensorAccessorArgs | varies | Tensor accessor configuration for destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks for this core |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block (`ntiles_per_block`) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total rows to read (`nblocks_per_core * TILE_HEIGHT`) |
| 2 | (unused) | uint32_t | Reserved |
| 3 | num_tiles_per_block | uint32_t | Tiles per block |
| 4 | block_width_size | uint32_t | Width of block in bytes |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for this implementation |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial blocks) |
| 7 | leftover_width | uint32_t | Always 0 |
| 8 | start_stick_id | uint32_t | Starting row index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_tiles | uint32_t | Total tiles to write (`ntiles_per_block * nblocks_per_core`) |
| 2 | start_id | uint32_t | Starting tile index for this core |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (row-major) | CB_0 | Read sticks, assemble blocks |

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

**Key Logic**:
- Uses `TensorAccessor` for efficient address calculation
- Pre-computes all 32 row addresses before issuing reads
- Reads entire row width in one transaction per row
- Uses `noc_async_read_barrier()` to ensure all reads complete before pushing

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_2 (Unpack/Math/Pack) | N/A | CB_0 | CB_16 | tilize_init, tilize_block, tilize_uninit |

**File**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp`

**Key Logic**:
- Calls `compute_kernel_hw_startup()` for hardware initialization
- Uses `compute_kernel_lib::tilize()` helper function
- Helper internally handles:
  - `tilize_init()`: Configures unpacker for row-major input, packer for tile output
  - `tilize_block()`: Transforms `block_w` tiles from row-major to tile format
  - `tilize_uninit()`: Restores default unpacker/packer state

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB_16 | DRAM (tiles) | Write tiles sequentially |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Key Logic**:
- Uses `TensorAccessor` for tile address calculation
- Processes one tile at a time (single-tile ublocks)
- Issues `noc_async_write_tile()` with barrier after each write
- Supports both forward and backward iteration (via `BACKWARDS` define)

## Implementation Notes

### Adaptive Core Distribution

The program factory implements an adaptive algorithm that compares 1D (row-based) and 2D (block-based) distribution:
- For narrow tensors: 1D distribution is sufficient
- For wide tensors (width > 32 tiles): May switch to 2D block distribution via `TilizeMultiCoreBlockProgramFactory` if it enables more parallelism

### FP32 Accumulation Support

When input dtype is FLOAT32, `fp32_dest_acc_en` is set to true in `ComputeConfig`, enabling 32-bit accumulation in the destination registers.

### Cliff Core Handling

The cliff core (last core with remainder work) receives:
- Different compile-time args for compute kernel (`nblocks_per_core_cliff` instead of `nblocks_per_core`)
- Adjusted runtime args for reader and writer

### Program Caching

The factory supports program caching via `override_runtime_arguments()`:
- Only buffer addresses need updating for cached programs
- Core assignments and block counts remain constant

### TensorAccessor Pattern

Both reader and writer use the `TensorAccessor` abstraction:
- Compile-time args include accessor configuration
- Runtime args include base address
- Accessor handles interleaved bank mapping internally

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is tilize operation in tt-metal? How does it convert row-major data to tile layout and what is the purpose of this transformation?"
   **Reason**: Needed to understand the fundamental purpose and mechanics of the tilize operation
   **Key Findings**: Tilize converts row-major to 32x32 tiles with 16x16 faces for hardware efficiency. The matrix engine natively operates on tiles.

2. **Query**: "What is TensorAccessor and TensorAccessorArgs in tt-metal? How are they used in kernels to access tensor data in memory?"
   **Reason**: Reader and writer kernels use TensorAccessor for address calculation
   **Key Findings**: TensorAccessor abstracts memory layout, mapping logical indices to physical addresses. TensorAccessorArgs configures compile-time vs runtime parameters.

3. **Query**: "What do tilize_init, tilize_block, and tilize_uninit functions do in compute kernels?"
   **Reason**: Compute kernel uses these functions via the helper library
   **Key Findings**: These configure unpacker/packer hardware for tilization, perform the actual transformation, and restore default state. The unpacker reads row-major data, and packer outputs tile format.

4. **Query**: "What is the difference between interleaved and sharded memory layout in tt-metal?"
   **Reason**: This operation specifically handles interleaved layout
   **Key Findings**: Interleaved distributes pages round-robin across all DRAM banks. Sharded assigns specific tensor regions to specific banks/cores for locality.

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding overall architecture, three-kernel model, circular buffer semantics
   **Key Information**: Reader/compute/writer kernel pattern, CB synchronization via push/pop/wait/reserve, tile-based computing rationale

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tensor layout formats
   **Key Information**: Row-major vs tiled layout, face structure (4x 16x16 faces per 32x32 tile), page definitions

3. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor usage patterns
   **Key Information**: Host-side setup via TensorAccessorArgs, device-side usage for address calculation, compile-time vs runtime configuration

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding core distribution algorithm
   **Key Information**: `split_blocks_for_tilize()` calculates ncores, blocks per core, and cliff handling

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
   **Reason**: Understanding compute kernel helper implementation
   **Key Information**: Unified tilize function handles multiple patterns (simple, activation, fast, DT reconfiguration) via template parameters

6. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding CB creation utility
   **Key Information**: `create_cb()` helper creates circular buffers with specified page size, num pages, and data format
