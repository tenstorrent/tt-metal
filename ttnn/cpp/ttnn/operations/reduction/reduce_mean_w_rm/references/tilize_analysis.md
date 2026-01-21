# Tilize Multi-Core Interleaved Implementation Analysis

## Overview

The **tilize** operation converts tensor data from **row-major layout** to **tiled layout**. This is a fundamental data transformation required for efficient processing on Tenstorrent hardware, which natively operates on 32x32 element tiles. The multi-core interleaved variant distributes work across multiple Tensix cores with interleaved (round-robin) memory access patterns.

**Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (collection of tile rows) |
| **Unit size** | `ntiles_per_block` tiles (width of tensor in tiles) |
| **Total units** | `nblocks = ceil(ntiles / ntiles_per_block)` |
| **Loop structure** | Outer: blocks, Inner: tile rows (32 sticks per tile height) |

A **block** in this context represents one horizontal strip of tiles spanning the full tensor width. Each block contains `ntiles_per_block` tiles arranged horizontally, and processing one block involves reading 32 consecutive rows ("sticks") of row-major data to form the tile height dimension.

**Key calculation from source** (lines 33-36):
```cpp
int32_t ntiles = a.physical_volume() / TILE_HW;           // Total tiles
uint32_t ntiles_per_block = a.padded_shape()[-1] / TILE_WIDTH;  // Tiles per row
uint32_t nblocks = std::ceil((float)ntiles / ntiles_per_block); // Number of tile rows
uint32_t block_size_nbytes = a.padded_shape()[-1] * a.element_size();  // Row width in bytes
```

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D (flattened to 2D for processing) |
| **Dimension convention** | Last dimension is contiguous |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16, FLOAT32, UINT32, INT32, or UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input (padded to tile boundaries) |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (configurable via output_mem_config) |
| **Data type** | Same as input (configurable via output_dtype) |

### Layout Transformations

The tilize operation performs the following transformation:

1. **Input**: Row-major data where each row is stored contiguously
   - Page = 1 row of width `padded_shape[-1]`
   - Pages distributed round-robin across DRAM banks

2. **Output**: Tiled data where each 32x32 tile is stored contiguously
   - Page = 1 tile (32x32 elements)
   - Tiles stored in row-major tile order (left-to-right, top-to-bottom)
   - Within each tile, data is organized into 4 faces (16x16 sub-tiles)

**Visual representation**:
```
Row-Major Input:           Tiled Output:
Row 0: [a00 a01 a02...]    Tile 0,0: [32x32 elements]
Row 1: [a10 a11 a12...]    Tile 0,1: [32x32 elements]
Row 2: [a20 a21 a22...]    ...
...
Row 31: [...]
Row 32: [...]              Tile 1,0: [32x32 elements]
...                        ...
```

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (row-major sticks) | CB c_0 | reserve_back, push_back |
| 2 | Compute | CB c_0 | CB c_16 | wait_front, pop_front, reserve_back, push_back |
| 3 | Writer | CB c_16 | DRAM (tiles) | wait_front, pop_front |

### Detailed Data Flow

1. **Reader Kernel** (RISCV_0 / BRISC):
   - Pre-computes NoC addresses for 32 consecutive rows (tile height)
   - Reads `block_width_size` bytes from each of the 32 rows
   - Data is written to CB c_0 in a format ready for tilization
   - Pattern: Read 32 rows simultaneously to form one tile row

2. **Compute Kernel** (RISCV_2,3,4 / Unpack, Math, Pack):
   - Waits for input data in CB c_0
   - Calls `tilize_block()` which:
     - Unpacker reads row-major data from CB c_0
     - Math core performs data copy (reordering happens in unpack/pack)
     - Packer writes tile-format data to CB c_16
   - Processes `per_core_block_cnt` blocks, each with `per_core_block_tile_cnt` tiles

3. **Writer Kernel** (RISCV_1 / NCRISC):
   - Waits for tiles in CB c_16
   - Writes one tile at a time to output DRAM buffer
   - Uses `tile_start_id` for offset into output tensor

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | Input CB | Input staging (row-major sticks) | ntiles_per_block tiles | ntiles_per_block tiles | Single | Reader | Compute | Block |
| c_16 | Output CB | Output staging (tiles) | ntiles_per_block tiles | ntiles_per_block tiles | Single | Compute | Writer | Block |

**CB Creation** (lines 47-50):
```cpp
create_cb(tt::CBIndex::c_0, program, all_cores, input_single_tile_size, ntiles_per_block, input_cb_data_format);
create_cb(tt::CBIndex::c_16, program, all_cores, output_single_tile_size, ntiles_per_block, output_cb_data_format);
```

**Note**: Both CBs have capacity equal to block size, which means single-buffered operation. The reader must complete writing a full block before compute can process, and compute must complete before writer can output.

## Pipeline Pattern Summary

| Pattern | Classification | Evidence |
|---------|---------------|----------|
| Reader-Compute | Single-buffered | CB c_0 capacity = ntiles_per_block = block size |
| Compute-Writer | Single-buffered | CB c_16 capacity = ntiles_per_block = block size |

The single-buffered design means no overlap between stages for the same block. However, different cores process different blocks in parallel, providing overall throughput.

## Index Calculations

### Reader Kernel Index Mapping

The reader uses `TensorAccessor` for address generation:

```cpp
// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr auto src_tensor_args = TensorAccessorArgs<1>();

// Runtime: create accessor
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);

// Address calculation for each stick
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

**Mapping logic**:
- `stick_id` = linear index of row in the input tensor
- `TensorAccessor.get_noc_addr(stick_id)` returns 64-bit NoC address containing:
  - Bank ID (derived from stick_id % num_banks for interleaved)
  - Physical address within bank
  - NoC X,Y coordinates of the bank

### Writer Kernel Index Mapping

```cpp
const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

// Write tile i
noc_async_write_page(i, s, l1_read_addr);
```

**Mapping logic**:
- `i` = tile_start_id + tile_offset
- `TensorAccessor.noc_async_write_page(i, ...)` computes:
  - Bank ID from tile index
  - Physical offset within bank
  - Initiates NoC write

## Memory Access Patterns

### Read Pattern

| Aspect | Description |
|--------|-------------|
| **Ordering** | Sequential rows within block, row-by-row |
| **Stride** | `block_size_nbytes` between successive reads within row |
| **Batching** | 32 rows read in parallel to form tile height |
| **Memory type** | DRAM interleaved |

**Reader algorithm** (from kernel, lines 40-51):
```cpp
for (uint32_t i = 0; i < num_sticks / tile_height; i++) {  // For each block
    // Pre-compute addresses for 32 rows
    for (uint32_t j = 0; j < tile_height; j++) {
        base_src_noc_addr[j] = get_noc_addr(stick_id, s);
        stick_id++;
    }
    // Read all tile columns in this block
    for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
        read_tiles(num_tiles_per_block, block_width_size);
    }
}
```

The reader pre-computes 32 NoC addresses, then issues reads for the full row width. This minimizes address computation overhead.

### Write Pattern

| Aspect | Description |
|--------|-------------|
| **Ordering** | Sequential tiles, left-to-right, top-to-bottom |
| **Stride** | Tile-sized pages |
| **Batching** | One tile at a time |
| **Memory type** | DRAM interleaved |

**Writer algorithm** (from kernel, lines 31-39):
```cpp
for (uint32_t i = start_id; i < end_id; ++i) {
    cb_wait_front(cb_id_out, onepage);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    noc_async_write_page(i, s, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out, onepage);
}
```

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized) |
| **Grid dimensions** | Up to device compute grid (e.g., 8x8 on Wormhole) |
| **Total cores** | min(nblocks, grid_area) |
| **Work per core** | nblocks_per_core blocks (cliff core may have fewer) |
| **Load balancing** | Nearly equal with cliff remainder |

### Work Distribution Algorithm

The `split_blocks_for_tilize` function (from work_split_tilize.hpp):

```cpp
auto [ncores, nblocks_per_core] = compute_ncores(grid_area, nblocks);
// ncores = ceil(nblocks / nblocks_per_core)
// nblocks_per_core = ceil(nblocks / grid_area)

const uint32_t nblocks_per_core_cliff = nblocks % nblocks_per_core;
```

**Example**: 100 blocks on 8x8 grid (64 cores)
- `nblocks_per_core = ceil(100/64) = 2`
- `ncores = ceil(100/2) = 50`
- `nblocks_per_core_cliff = 100 % 2 = 0` (no cliff)

**With remainder**: 101 blocks
- `nblocks_per_core = 2`
- `ncores = 51`
- `nblocks_per_core_cliff = 1` (last core processes 1 block)

### Runtime Argument Distribution (lines 110-137)

```cpp
for (uint32_t i = 0; i < ncores_full; ++i) {
    // Reader args: src_addr, num_sticks, block_size_nbytes, ...
    const std::array reader_rt_args = {
        src0_buffer->address(),
        nblocks_per_core * TILE_HEIGHT,  // Total rows for this core
        block_size_nbytes,
        ntiles_per_block,
        block_size_nbytes,
        std::uint32_t{1},   // full blocks in row
        std::uint32_t{0},   // leftover tiles
        std::uint32_t{0},   // leftover width
        row_start_id};      // Starting row index

    // Writer args: dst_addr, num_tiles, start_tile_id
    const std::array writer_rt_args = {
        dst_buffer->address(),
        ntiles_per_block * nblocks_per_core,
        tile_start_id};

    tile_start_id += ntiles_per_block * nblocks_per_core;
    row_start_id += TILE_HEIGHT * nblocks_per_core;
}
```

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one row in bytes (`block_size_nbytes`) |
| 1+ | TensorAccessorArgs | struct | Source buffer accessor parameters |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block (`ntiles_per_block`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output CB ID (c_16) |
| 1+ | TensorAccessorArgs | struct | Destination buffer accessor parameters |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total rows to read (`nblocks_per_core * 32`) |
| 2 | block_size_nbytes | uint32_t | Row width in bytes |
| 3 | num_tiles_per_block | uint32_t | Tiles per row |
| 4 | block_width_size | uint32_t | Width of block in bytes |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for this factory |
| 6 | num_leftover_tiles | uint32_t | Always 0 for this factory |
| 7 | leftover_width | uint32_t | Always 0 for this factory |
| 8 | start_stick_id | uint32_t | Starting row index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_pages | uint32_t | Total tiles to write |
| 2 | start_id | uint32_t | Starting tile index for this core |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (rows) | CB c_0 | Read 32 rows, batch by tile width |

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

**Key Logic**:
- Pre-computes 32 NoC addresses before reading (minimizes latency)
- Uses `noc_async_read` for each row, barrier after all 32
- `read_tiles` lambda handles CB reserve/push and NoC barrier
- Increments base addresses after each read to handle multiple tile columns

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize.cpp | RISCV_2,3,4 | N/A | CB c_0 | CB c_16 | tilize_block per block |

**File**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp`

**Key Logic**:
```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
compute_kernel_lib::tilize(tt::CBIndex::c_0, per_core_block_tile_cnt,
                           tt::CBIndex::c_16, per_core_block_cnt);
```

The `tilize()` function from `tilize_helpers.hpp`:
1. Calls `tilize_init(icb, block_w, ocb)` - configures unpacker/packer
2. For each block:
   - `cb_wait_front(icb, block_w)` - wait for input
   - `cb_reserve_back(ocb, block_w)` - reserve output space
   - `tilize_block(icb, block_w, ocb)` - hardware tilization
   - `cb_push_back(ocb, block_w)` - signal output ready
   - `cb_pop_front(icb, block_w)` - free input
3. Calls `tilize_uninit(icb, ocb)` - cleanup

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_16 | DRAM (tiles) | Write one tile at a time |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Key Logic**:
- Generic writer supporting both tile and row-major layouts
- Page size determined from CB interface at runtime
- One tile per iteration: wait, read pointer, async write, barrier, pop
- Supports BACKWARDS and OUT_SHARDED variants via preprocessor

## Implementation Notes

### Design Decisions

1. **CB Capacity = Block Size**: Single-buffered design simplifies synchronization but limits overlap. This is appropriate for tilize since the operation is memory-bound.

2. **Reused Writer Kernel**: The writer is generic (`writer_unary_interleaved_start_id.cpp`) and shared with eltwise/unary operations. This reduces code duplication.

3. **Pre-computed NoC Addresses**: Reader pre-computes 32 addresses before reading, then reuses/increments them. This reduces per-read overhead.

4. **Cliff Core Handling**: Separate compute kernel instances for regular vs cliff cores with different compile-time args, avoiding runtime conditionals.

5. **TensorAccessor Usage**: Both reader and writer use TensorAccessor for address generation, abstracting interleaved bank distribution.

### Potential Pain Points

1. **No Double Buffering**: Single-buffered CBs mean no overlap between read/compute/write stages within a core. Multi-core parallelism provides throughput instead.

2. **Deprecated Compute Kernel Location**: The compute kernel is in `deprecated/tt_dnn/kernels/compute/`, suggesting planned migration.

3. **Fixed Block Width**: `num_full_blocks_in_row` is hardcoded to 1 and leftover handling is always 0. This limits flexibility for non-tile-aligned widths.

4. **FP32 Accumulator Flag**: `fp32_dest_acc_en` is set based on input dtype being FLOAT32, which may not always be desired behavior.

### Key Formulas

```cpp
// Tile calculations
ntiles = physical_volume / TILE_HW  // TILE_HW = 32*32 = 1024
ntiles_per_block = padded_width / TILE_WIDTH  // TILE_WIDTH = 32
nblocks = ceil(ntiles / ntiles_per_block)

// Work distribution
nblocks_per_core = ceil(nblocks / grid_area)
ncores = ceil(nblocks / nblocks_per_core)
nblocks_per_core_cliff = nblocks % nblocks_per_core

// Reader indexing
num_sticks = nblocks_per_core * TILE_HEIGHT  // 32 rows per block
start_stick_id = core_index * nblocks_per_core * TILE_HEIGHT

// Writer indexing
num_tiles = nblocks_per_core * ntiles_per_block
start_tile_id = core_index * num_tiles
```

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the tilize operation and how does it transform row-major data into tiled format?"
   **Reason**: Needed foundational understanding of the operation's purpose
   **Key Findings**: Transforms ROW_MAJOR to TILE_LAYOUT, 32x32 tiles with 16x16 faces internally, hardware natively processes tiles

2. **Query**: "What does the TensorAccessor do in tt-metal kernels?"
   **Reason**: Understanding address generation mechanism used in reader/writer
   **Key Findings**: Abstracts memory layout, `get_noc_addr(page_id)` maps logical index to physical NoC address with bank interleaving

3. **Query**: "What does tilize_block do in the compute kernel?"
   **Reason**: Understanding hardware-level tilization mechanism
   **Key Findings**: Coordinates unpacker->math->packer, `llk_unpack_tilize_block` reads from CB, `llk_pack` writes tile format

4. **Query**: "What is the relationship between sticks and rows in row-major layout?"
   **Reason**: Clarify terminology used in reader kernel
   **Key Findings**: Stick = one row of data, 32 sticks form tile height, reader fetches sticks into CB

5. **Query**: "How do circular buffers work with tilize?"
   **Reason**: Understanding CB configuration for this operation
   **Key Findings**: Page size = tile size, capacity in tiles, tilize_block processes tile-sized chunks

6. **Query**: "What is a block in the context of tilize operations?"
   **Reason**: Understanding work unit definition
   **Key Findings**: Block = horizontal strip of tiles, split_blocks_for_tilize distributes blocks to cores

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Authoritative documentation on row-major vs tiled layouts
   **Key Information**: Row-major pages are rows, tiled pages are 32x32 tiles, faces are 16x16 sub-tiles

2. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding kernel architecture and CB coordination
   **Key Information**: Reader->CB->Compute->CB->Writer pattern, CB producer-consumer synchronization

3. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding work distribution algorithm
   **Key Information**: BlockSplit struct, cliff core handling, nblocks_per_core calculation
