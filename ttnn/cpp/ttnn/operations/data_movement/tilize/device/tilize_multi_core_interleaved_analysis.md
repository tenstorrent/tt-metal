# Tilize Multi-Core Interleaved Implementation Analysis

## Overview

The **tilize** operation converts tensor data from **ROW_MAJOR (stick) layout** to **TILE_LAYOUT (32x32 tiles)**. This is a fundamental data transformation operation required before most compute operations on Tenstorrent hardware, as the matrix engine (FPU) natively operates on 32x32 tiles.

**Program Factory Path**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

This analysis focuses on the **multi-core interleaved** variant, which processes row-major data distributed across DRAM banks in an interleaved (round-robin) fashion.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one block = one tile row = 32 rows of input sticks) |
| **Unit size** | `ntiles_per_block` tiles (width of tensor in tiles) |
| **Total units** | `nblocks` = total_tiles / ntiles_per_block = height_in_tiles |
| **Loop structure** | Each core processes `nblocks_per_core` blocks, iterating over 32-row chunks |

**Key Insight**: A "block" in tilize corresponds to a horizontal strip of 32 rows (one tile height). Each block contains `ntiles_per_block` tiles arranged horizontally. The reader reads 32 contiguous sticks (rows), and the compute kernel tilizes them into `ntiles_per_block` output tiles.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] (arbitrary dimensions) | [N, C, H_padded, W_padded] |
| **Dimension convention** | NCHW (row-major) | NCHW (tiled) |
| **Tensor layout** | ROW_MAJOR | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (default) or L1 | DRAM (default) or L1 |
| **Data type** | BFLOAT16, FLOAT32, etc. | Same as input |
| **Page definition** | Stick (one row) | Tile (32x32 elements) |

### Layout Transformation Details

**Input (ROW_MAJOR)**:
- Data stored as contiguous rows ("sticks")
- Each row is one page in interleaved memory
- Consecutive rows may be in different DRAM banks (round-robin distribution)
- Page size = `width * element_size` bytes

**Output (TILE_LAYOUT)**:
- Data organized into 32x32 tiles
- Each tile is one page in interleaved memory
- Tiles stored in row-major order (tile 0, tile 1, ... across the width, then next row of tiles)
- Tile internally organized into 4 faces of 16x16 for FPU compatibility
- Page size = `32 * 32 * element_size` bytes (1024 elements)

### Padding Requirements
- Height must be padded to multiple of 32 (TILE_HEIGHT)
- Width must be padded to multiple of 32 (TILE_WIDTH)
- `num_tiles_per_row = padded_width / 32`
- `num_tiles_per_col = padded_height / 32`

## Data Flow Pattern

```
                         DRAM                    L1 (per core)                    DRAM
                    +-----------+              +---------------+              +-----------+
                    |  ROW_MAJOR |              |               |              | TILE_LAYOUT|
 stick_0 (row 0)    | interleaved| --NoC0-->   |   CB_IN (c_0) |              | interleaved|
 stick_1 (row 1)    |   sticks   |             |  32 rows of   |              |   tiles    |
    ...             |            |             |   stick data  |              |            |
 stick_31 (row 31)  |            |             +-------+-------+              |            |
                    +-----------+                      |                      +-----------+
                                                       v
                                               +-------+-------+
                                               |    COMPUTE    |
                                               |   tilize()    |
                                               | (unpack+math+ |
                                               |     pack)     |
                                               +-------+-------+
                                                       |
                                                       v
                                               +-------+-------+
                                               |  CB_OUT (c_16)|  --NoC1-->  Output tiles
                                               | tilized tiles |              in DRAM
                                               +---------------+
```

### Step-by-Step Flow

1. **Reader Kernel** (RISCV_0 / BRISC):
   - For each block (32 rows):
     - Calculate NoC addresses for 32 consecutive sticks using TensorAccessor
     - Reserve space in CB_IN for `ntiles_per_block` tiles worth of stick data
     - Read 32 sticks from DRAM into CB_IN as a contiguous block
     - The data is arranged so that 32 consecutive sticks form the input for tilization
     - Push block to CB_IN after all 32 rows are read

2. **Compute Kernel** (UNPACK/MATH/PACK RISCs):
   - Wait for input block in CB_IN
   - Reserve space in CB_OUT for `ntiles_per_block` output tiles
   - Call `compute_kernel_lib::tilize()` which:
     - Initializes tilize hardware (`tilize_init`)
     - Unpacks stick data from CB_IN
     - Math engine performs datacopy with tilize enabled (rearranges data into 32x32 tiles)
     - Packs tilized output to CB_OUT
   - Pop input from CB_IN, push output to CB_OUT

3. **Writer Kernel** (RISCV_1 / NCRISC):
   - For each output tile:
     - Wait for tile in CB_OUT
     - Calculate NoC address for output tile using TensorAccessor
     - Write tile to DRAM
     - Pop tile from CB_OUT

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input staging (stick data) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_out0 | Output staging (tilized data) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Compute | Writer | Block |

**Notes**:
- Both CBs are sized to hold one complete horizontal strip (one block)
- Single-buffered design: entire block must be read before compute can start
- CB capacity = `ntiles_per_block * tile_size_bytes`
- Input CB uses `input_cb_data_format` (matches input tensor dtype)
- Output CB uses `output_cb_data_format` (matches output tensor dtype)

### CB Sizing Rationale
The CB must hold `ntiles_per_block` tiles because:
1. Reader reads 32 sticks which, when tilized, produce `width/32` tiles
2. Compute processes all tiles in the block at once (tilize_block operates on full block)
3. Writer writes one tile at a time but must have all tiles available

## Pipeline Pattern Summary

| Stage | Buffering Type | Overlap Potential |
|-------|----------------|-------------------|
| Reader -> Compute | Single-buffered | None (reader must complete block before compute starts) |
| Compute -> Writer | Single-buffered | Partial (writer processes tiles as compute produces them) |

The single-buffered design means no read/compute overlap within a block, but different cores can work on different blocks in parallel.

## Index Calculations

### Input Index Mapping (Sticks)

The reader uses `TensorAccessor` to map stick indices to physical DRAM addresses:

```cpp
// TensorAccessor setup
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);

// For each block, get base addresses for 32 sticks
for (uint32_t j = 0; j < tile_height; j++) {
    base_src_noc_addr[j] = get_noc_addr(stick_id, s);
    stick_id++;
}
```

**Stick ID Calculation**:
- `start_stick_id = row_start_id` (first row this core processes)
- `stick_id` increments sequentially for each row
- Each block consumes 32 stick IDs (TILE_HEIGHT)

### Output Index Mapping (Tiles)

The writer uses `TensorAccessor` with tile indices:

```cpp
// TensorAccessor setup
const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

// Write tiles starting from start_id
for (uint32_t i = start_id; i < end_id; ++i) {
    noc_async_write_tile(i, s, l1_read_addr);
}
```

**Tile ID Calculation**:
- `tile_start_id` = first tile index this core writes
- Tiles are numbered in row-major order across the entire output tensor
- Each core writes `ntiles_per_block * nblocks_per_core` tiles consecutively

## Memory Access Patterns

### Read Pattern (Reader Kernel)

```
Access Type: Sequential with stride
Pattern: For each block of 32 rows:
         - Read stick 0 (row 0) entirely
         - Read stick 1 (row 1) entirely
         - ...
         - Read stick 31 (row 31) entirely

Each stick read: block_width_size bytes (entire row width)
Interleaving: Sticks distributed round-robin across DRAM banks
```

**Key Detail**: The reader reads all data for a tile row in a specific pattern that facilitates tilization. The 32 sticks are read into contiguous L1 memory, arranged so the tilize hardware can rearrange them into tiles.

### Write Pattern (Writer Kernel)

```
Access Type: Sequential tile writes
Pattern: tile_0, tile_1, tile_2, ... (row-major order)

Each write: tile_size bytes (32x32 * element_size)
Interleaving: Output tiles distributed round-robin across DRAM banks
```

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of available cores) |
| **Grid dimensions** | `ncores` (varies based on tensor size and available cores) |
| **Total cores** | min(available_cores, nblocks) |
| **Work per core** | `nblocks_per_core` blocks (or `nblocks_per_core_cliff` for last core) |
| **Load balancing** | Near-equal with cliff handling |

### Work Distribution Algorithm

```cpp
auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
    ttnn::split_blocks_for_tilize(available_grid, nblocks);
```

1. Calculate `nblocks_per_core = ceil(nblocks / available_cores)`
2. Calculate `ncores = ceil(nblocks / nblocks_per_core)`
3. If `nblocks % nblocks_per_core != 0`, the last core is a "cliff core" with fewer blocks
4. `core_range`: cores with full `nblocks_per_core` work
5. `core_range_cliff`: single cliff core (if needed) with `nblocks_per_core_cliff` work

### Per-Core Work Assignment

```cpp
// Full cores (0 to ncores_full-1)
tile_start_id = i * ntiles_per_block * nblocks_per_core
row_start_id = i * TILE_HEIGHT * nblocks_per_core  // Each block is 32 rows

// Cliff core (if present)
tile_start_id = ncores_full * ntiles_per_block * nblocks_per_core
row_start_id = ncores_full * TILE_HEIGHT * nblocks_per_core
```

### Block Program Factory Fallback

For wide tensors (`num_tiles_per_row > 32`), the factory may choose `TilizeMultiCoreBlockProgramFactory` instead, which uses a 2D core distribution strategy for better parallelism. This decision is made based on:
- `num_tiles_per_row > threshold_row_block (32)`
- `num_tiles_per_col > threshold_row_block` OR `num_tiles_per_row > num_tiles_per_col`
- Whether 2D splitting yields more cores than 1D splitting

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one input row in bytes (`width * element_size`) |
| 1+ | TensorAccessorArgs | varies | Bank distribution info for input tensor (appended by `TensorAccessorArgs::append_to`) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | nblocks_per_core | uint32_t | Number of blocks to process (32-row chunks) |
| 1 | ntiles_per_block | uint32_t | Tiles per block (width / 32) |

Note: Cliff cores get separate kernel instantiation with `nblocks_per_core_cliff` instead.

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB ID to read from (c_16) |
| 1+ | TensorAccessorArgs | varies | Bank distribution info for output tensor |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Base address of input buffer in DRAM |
| 1 | num_sticks | uint32_t | Total rows to read (`nblocks_per_core * 32`) |
| 2 | block_size_nbytes | uint32_t | Size of one row in bytes |
| 3 | ntiles_per_block | uint32_t | Tiles per horizontal strip |
| 4 | block_width_size | uint32_t | Same as block_size_nbytes |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for this factory |
| 6 | num_leftover_tiles | uint32_t | Always 0 for this factory |
| 7 | leftover_width | uint32_t | Always 0 for this factory |
| 8 | start_stick_id | uint32_t | First row index this core processes |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output buffer in DRAM |
| 1 | num_tiles | uint32_t | Total tiles to write (`ntiles_per_block * nblocks_per_core`) |
| 2 | start_id | uint32_t | First tile index this core writes |

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (sticks) | CB c_0 | Read 32 rows per block, async reads |

**Key Logic**:
```cpp
// For each block (32 rows)
for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
    // Pre-calculate NoC addresses for 32 consecutive sticks
    for (uint32_t j = 0; j < tile_height; j++) {
        base_src_noc_addr[j] = get_noc_addr(stick_id, s);
        stick_id++;
    }

    // Read tiles using pre-calculated addresses
    for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
        read_tiles(num_tiles_per_block, block_width_size);
    }
}

// read_tiles lambda:
// - cb_reserve_back for output space
// - Loop over 32 rows, async read each from pre-calculated address
// - noc_async_read_barrier
// - cb_push_back
```

### Compute Kernel

**File**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | UNPACK/MATH/PACK | N/A | CB c_0 | CB c_16 | tilize (unpack + datacopy + pack) |

**Key Logic**:
```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
compute_kernel_lib::tilize(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16, per_core_block_cnt);
```

The helper function `compute_kernel_lib::tilize()` handles:
1. `tilize_init()` - Configure unpack, math, and pack hardware for tilization
2. Loop over blocks:
   - `cb_wait_front()` - Wait for input data
   - `cb_reserve_back()` - Reserve output space
   - `tilize_block()` - Unpack row data, math datacopy with tilize, pack to output
   - `cb_push_back()`, `cb_pop_front()` - Update CB pointers
3. `tilize_uninit()` - Cleanup tilize configuration

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM (tiles) | Write tiles sequentially |

**Key Logic**:
```cpp
// Write tiles one at a time
for (uint32_t i = start_id; i < end_id; ++i) {
    cb_wait_front(cb_id_out, onetile);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    noc_async_write_tile(i, s, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out, onetile);
}
```

Note: Writer processes one tile at a time, even though compute produces `ntiles_per_block` tiles at once. This creates a producer-consumer pattern where compute may be blocked waiting for CB space while writer drains tiles.

## Implementation Notes

### FP32 Accumulation Mode
When input dtype is FLOAT32, `fp32_dest_acc_en` is set to true in the compute config, enabling FP32 accumulation in the destination registers.

### Block vs. Block Factory Selection
The factory includes logic to potentially use `TilizeMultiCoreBlockProgramFactory` for wide tensors:
- Threshold: `num_tiles_per_row > 32`
- Uses 2D work distribution (tiles_per_core in both W and H dimensions)
- Selected if 2D splitting yields more active cores than 1D splitting

### TensorAccessor Usage
Both reader and writer use `TensorAccessor` for address calculation:
- Abstracts the round-robin bank distribution
- Handles page-to-bank mapping automatically
- Compile-time args contain bank distribution info
- Runtime args contain base address and page size

### Single-Buffered Pipeline Limitation
The current design uses single-buffered CBs, meaning:
- Reader must complete an entire block before compute can start
- Compute must complete before reader can fill the next block (within a core)
- Parallelism comes from multiple cores, not overlapped execution within a core

For higher performance, double-buffered CBs would allow overlapping read/compute/write within each core.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the tilize operation in TT-Metal? How does it convert row-major data to tile layout?"
   **Reason**: Needed foundational understanding of what tilize does architecturally.
   **Key Findings**: Tilize converts row-major data into 32x32 tiles optimized for the matrix engine. The hardware natively operates on tiles, and tilize is essential for preparing data for compute operations.

2. **Query**: "What does the TensorAccessor class do in TT-Metal kernels? How is it used with get_noc_addr?"
   **Reason**: Needed to understand how address calculations work for interleaved memory.
   **Key Findings**: TensorAccessor abstracts page-to-bank mapping for interleaved tensors. It calculates NoC addresses for pages distributed round-robin across DRAM banks.

3. **Query**: "What do the tilize_block and tilize_init LLK functions do? How does the unpack thread work during tilization?"
   **Reason**: Needed to understand the compute kernel's internal operation.
   **Key Findings**: `tilize_init` configures unpack/math/pack hardware for tilization. `tilize_block` orchestrates the three threads: UNPACK reads from CB, MATH performs datacopy with tilize enabled, PACK writes tilized data to output CB.

4. **Query**: "How does interleaved memory layout work in TT-Metal?"
   **Reason**: Needed to understand input/output memory distribution.
   **Key Findings**: Interleaved layout distributes pages round-robin across DRAM banks. Each new tensor starts allocation at bank 0. Wormhole has 12 DRAM banks.

5. **Query**: "What is the difference between a stick and a tile in TT-Metal?"
   **Reason**: Needed to clarify terminology used in the reader kernel.
   **Key Findings**: A "stick" is a contiguous row of data in row-major layout. A "tile" is a 32x32 block in tiled layout. Sticks are the page unit for row-major tensors.

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Core architecture understanding
   **Key Information**: Tensix core architecture with 5 RISC-V cores, 3-kernel model (reader/compute/writer), circular buffer communication, tile-based computing fundamentals.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tensor and memory layout concepts
   **Key Information**: Row-major vs tiled layout, interleaved vs sharded memory, page definitions for each layout type, tile faces (4x 16x16 sub-tiles per 32x32 tile).

3. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor API for address calculations
   **Key Information**: Host-side setup with `TensorAccessorArgs`, device-side usage with `get_noc_addr()`, compile-time vs runtime argument configuration.

4. **Source**: `tt_metal/include/compute_kernel_api/tilize.h`
   **Reason**: Understanding tilize compute API
   **Key Information**: `tilize_init()`, `tilize_block()`, `tilize_uninit()` functions and their parameters. UNPACK/MATH/PACK thread coordination through LLK calls.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
   **Reason**: Understanding the helper library used by the compute kernel
   **Key Information**: Unified `tilize()` template function that handles initialization, block processing loop, and cleanup. Supports multiple patterns (simple, activation, fast, DT).

6. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding core distribution algorithm
   **Key Information**: `split_blocks_for_tilize()` function calculates ncores, blocks_per_core, and cliff handling for 1D distribution of blocks across available cores.
