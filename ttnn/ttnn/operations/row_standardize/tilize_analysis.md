# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The **tilize** operation converts a tensor from **row-major layout** (ROW_MAJOR_LAYOUT) to **tile layout** (TILE_LAYOUT), where data is reorganized from contiguous rows ("sticks") into 32x32 tiles. This is the fundamental format conversion required before any compute operation on Tenstorrent hardware, since Tensix cores operate natively on 32x32 tiles.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

This analysis serves as the **input_stage reference** for the row_standardize operation, demonstrating how to read row-major data from DRAM and convert it into tile format for downstream compute kernels.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a horizontal strip of tiles, 1 tile high x W tiles wide) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / TILE_WIDTH`) |
| **Total units** | `nblocks` = `ntiles / ntiles_per_block` = number of tile-rows across the full tensor |
| **Loop structure** | Outer: iterate over blocks (each = 32 sticks = 1 tile height). Inner: read sticks, tilize, write tiles |

A **block** in this operation represents one horizontal strip of tiles spanning the full width of the tensor and one tile height (32 rows) tall. For a tensor with shape `[1, 1, 128, 256]`:
- `ntiles_per_block` = 256 / 32 = 8 tiles wide
- `nblocks` = (128 * 256 / 1024) / 8 = 4 blocks tall
- Each block contains 32 sticks of width 256 elements, yielding 8 tiles.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [N, ...dims..., H, W] (arbitrary rank, last two dims are spatial) |
| **Dimension convention** | Last dim = W (stick width), second-to-last = H |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 |

In row-major layout, each **page** (stick) is one full row of width W. The stick size in bytes is `padded_shape[-1] * element_size()`. Pages are distributed round-robin across DRAM banks in interleaved layout.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input (padded to tile alignment) |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input (BFLOAT16 or FLOAT32) |

In tile layout, each **page** is a 32x32 tile. The tile size depends on data format (e.g., 2048 bytes for BFLOAT16, 4096 bytes for FLOAT32). Tiles are stored in row-major order (left-to-right, top-to-bottom) and distributed round-robin across DRAM banks.

### Layout Transformations

The core transformation is: **row-major sticks --> 32x32 tiles**

Input data arrives as linear rows (sticks). The compute kernel's `tilize_block` function rearranges 32 consecutive sticks into tiles. Each group of 32 sticks with width W produces `W/32` tiles. Within each tile, data is further organized into 16x16 faces for hardware compatibility with the matrix engine.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (row-major sticks) | CB c_0 (input) | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB c_0 (input) | CB c_16 (output) | `cb_wait_front`, `tilize_block`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back` |
| 3 | Writer | CB c_16 (output) | DRAM (tiles) | `cb_wait_front`, `noc_async_write_page`, `cb_pop_front` |

### Detailed Data Flow

**Step 1 -- Reader reads 32 sticks per block:**
1. For each block (group of 32 sticks = 1 tile height):
   a. Pre-compute NoC addresses for all 32 sticks using `get_noc_addr(stick_id, s)` via TensorAccessor
   b. Reserve space in CB c_0 for `ntiles_per_block` pages
   c. Read all 32 sticks into CB c_0 using `noc_async_read` -- each stick is `block_width_size` bytes
   d. The 32 sticks are laid out contiguously in CB c_0, forming a `32 x W` row-major region
   e. Wait for reads to complete (`noc_async_read_barrier`)
   f. Push `ntiles_per_block` pages to signal compute

**Step 2 -- Compute tilizes the block:**
1. Wait for `ntiles_per_block` pages in CB c_0 (the 32-stick row-major block)
2. Reserve `ntiles_per_block` pages in CB c_16 (output)
3. Call `tilize_block(c_0, ntiles_per_block, c_16)` which uses hardware unpack logic to rearrange row-major data into tile format
4. Push `ntiles_per_block` tiles to CB c_16
5. Pop `ntiles_per_block` from CB c_0

**Step 3 -- Writer writes tiles to DRAM:**
1. For each tile: wait for 1 page in CB c_16
2. Read the tile from CB c_16
3. Write to DRAM using `noc_async_write_page(tile_id, s, l1_read_addr)` via TensorAccessor
4. Pop 1 page from CB c_16

### Key Insight: Reader Reads Sticks, Not Tiles

The reader kernel (`reader_unary_stick_layout_split_rows_interleaved.cpp`) reads data in its **original row-major format** -- it reads sticks (rows) from DRAM. It does NOT read tiles. The CB c_0 page size is set to `input_single_tile_size` and capacity is `ntiles_per_block`, but the reader fills this space with 32 contiguous sticks that happen to occupy the same total bytes as `ntiles_per_block` tiles. The compute kernel's `tilize_block` then reinterprets and rearranges this data.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|-----------|----------|----------|----------|
| c_0 | cb_input | Input staging (row-major sticks) | `ntiles_per_block` | `input_single_tile_size` | Single | Reader | Compute | Block |
| c_16 | cb_output | Output staging (tilized tiles) | `ntiles_per_block` | `output_single_tile_size` | Single | Compute | Writer | Block |

### CB Sizing Rationale

Both CBs have capacity equal to `ntiles_per_block` pages. This means:
- **CB c_0**: Holds exactly one block of row-major data (32 sticks spanning the full tensor width). The reader fills the entire CB before signaling compute.
- **CB c_16**: Holds exactly one block of tilized output (all tiles from one tile-row). Compute fills it, then writer drains it one tile at a time.

### Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Classification |
|----|----------|------------|-------|----------------|
| c_0 | ntiles_per_block | ntiles_per_block | 1:1 | **Single-buffered** |
| c_16 | ntiles_per_block | ntiles_per_block (compute), 1 (writer) | varies | **Single-buffered** (compute perspective) |

Both CBs are single-buffered (capacity = block size). This means:
- Reader must finish writing all 32 sticks before compute can start on a block
- Compute must finish tilizing the entire block before reader can start the next block
- There is **no overlap** between reader and compute for consecutive blocks
- Writer drains tiles one at a time from c_16, which can overlap with reader filling c_0 for the next block

## Index Calculations

### Reader: Stick-level Addressing

The reader uses `TensorAccessor` to map logical stick IDs to physical DRAM addresses:

```cpp
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

- `stick_id` starts at `start_stick_id` (runtime arg, set per core)
- `stick_id` increments by 1 for each stick within a block
- After processing `tile_height` (32) sticks, the next block begins
- Total sticks per core: `nblocks_per_core * TILE_HEIGHT`

### Reader: Intra-block Width Handling

Within each block of 32 sticks, the reader reads the full width in one shot:
```cpp
for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
    read_tiles(num_tiles_per_block, block_width_size);
}
```
In the interleaved factory, `num_full_blocks_in_row` is always 1 (the entire width is read at once). `block_width_size` = `padded_shape[-1] * element_size()`.

### Writer: Tile-level Addressing

The writer uses `TensorAccessor` to map logical tile IDs to physical DRAM addresses:
```cpp
noc_async_write_page(i, s, l1_read_addr);
```
- Tile IDs start at `start_id` (= `tile_start_id` set per core)
- Tiles are written sequentially: `start_id, start_id+1, ..., start_id + ntiles_per_core - 1`

### Mapping Between Stick IDs and Tile IDs

For a tensor with `W_tiles = padded_shape[-1] / 32` tiles per row:
- Core processing blocks `[b_start, b_end)` has:
  - `row_start_id = b_start * TILE_HEIGHT` (stick space)
  - `tile_start_id = b_start * ntiles_per_block` (tile space)
- Each block of 32 sticks produces `ntiles_per_block` tiles

## Memory Access Patterns

### Read Pattern (Reader Kernel)

**Pattern**: Strided reads of contiguous sticks

For each block:
1. Pre-fetch 32 NoC addresses (one per stick, each stick may be in a different DRAM bank due to interleaving)
2. For each of 32 sticks: issue `noc_async_read(src_noc_addr, l1_write_addr, width_size)`
3. L1 write address advances by `width_size` after each stick (contiguous in L1)
4. NoC source address for each stick is independent (computed by TensorAccessor)

The reads are **not sequential in DRAM** because interleaved layout distributes sticks round-robin across DRAM banks. However, all 32 sticks within a block are read into contiguous L1 memory, forming a `32 x W` row-major region.

### Write Pattern (Writer Kernel)

**Pattern**: Sequential single-tile writes

The writer writes tiles one at a time in sequential tile ID order:
```
for (i = start_id; i < end_id; ++i) {
    cb_wait_front(cb_id_out, 1);
    noc_async_write_page(i, s, l1_read_addr);
    noc_async_writes_flushed();
    cb_pop_front(cb_id_out, 1);
}
```
Each tile is written to a potentially different DRAM bank (interleaved distribution).

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from available 2D grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (full cores), `nblocks_per_core_cliff` blocks (cliff core) |
| **Load balancing** | Near-equal with optional cliff core for remainder |

### Work Distribution Details

The function `split_blocks_for_tilize(available_grid, nblocks)` performs the distribution:

1. Compute `nblocks_per_core = ceil(nblocks / grid_area)`
2. Compute `ncores = ceil(nblocks / nblocks_per_core)`
3. Compute `nblocks_per_core_cliff = nblocks % nblocks_per_core`
4. If `nblocks_per_core_cliff > 0`, the last core is a "cliff" core with fewer blocks

**Example**: For a `[1, 1, 256, 512]` tensor on an 8x8 grid:
- `ntiles_per_block` = 512/32 = 16
- `nblocks` = (256/32) * 1 = 8
- `nblocks_per_core` = ceil(8/64) = 1
- `ncores` = 8
- Each of 8 cores processes 1 block (32 sticks, producing 16 tiles)

### Per-Core Runtime Args Setup

For each full core `i`:
- `row_start_id` = `i * nblocks_per_core * TILE_HEIGHT` (which stick to start reading)
- `tile_start_id` = `i * nblocks_per_core * ntiles_per_block` (which tile to start writing)

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size (block_size_nbytes) | uint32_t | Size of one row-major stick in bytes: `padded_shape[-1] * element_size()` |
| 1+ | TensorAccessorArgs | multiple uint32_t | Appended by `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` -- encodes bank count, addressing info for src buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process on this core (`nblocks_per_core` or `nblocks_per_core_cliff`) |
| 1 | per_core_block_tile_cnt | uint32_t | Number of tiles per block (`ntiles_per_block`) |

**Note**: Cliff cores get a separate kernel binary with `nblocks_per_core_cliff` as arg 0.

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out (output_cb_index) | uint32_t | Output CB index (c_16) |
| 1+ | TensorAccessorArgs | multiple uint32_t | Appended by `TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args)` -- encodes bank count, addressing info for dst buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total number of sticks to read: `nblocks_per_core * TILE_HEIGHT` |
| 2 | block_size_nbytes | uint32_t | Same as stick_size: `padded_shape[-1] * element_size()` |
| 3 | num_tiles_per_block | uint32_t | Tiles per block (`ntiles_per_block`) |
| 4 | block_width_size | uint32_t | Width of one stick in bytes (same as block_size_nbytes) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for interleaved (entire width read at once) |
| 6 | num_leftover_tiles | uint32_t | Always 0 for interleaved |
| 7 | leftover_width_in_row | uint32_t | Always 0 for interleaved |
| 8 | start_stick_id | uint32_t | First stick ID for this core: `row_start_id` |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_pages | uint32_t | Total tiles to write: `ntiles_per_block * nblocks_per_core` |
| 2 | start_id | uint32_t | First tile ID for this core: `tile_start_id` |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM (sticks) | CB c_0 | Read 32 sticks per block via `noc_async_read` |
| Compute | RISCV_2 (UNPACK+MATH+PACK) | N/A | CB c_0 | CB c_16 | `tilize_block` -- rearrange sticks to tiles |
| Writer | RISCV_1 | NOC1 | CB c_16 | DRAM (tiles) | Write 1 tile at a time via `noc_async_write_page` |

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

**Key Logic**:
- Creates a `TensorAccessor` from compile-time `TensorAccessorArgs` and runtime `src_addr` + `stick_size`
- Pre-computes all 32 NoC addresses at the start of each block iteration (stored in `base_src_noc_addr[32]` array)
- Uses a lambda `read_tiles` that: reserves CB space, reads 32 sticks into contiguous L1, waits for barrier, pushes to CB
- The `base_src_noc_addr[k] += width_size` advancement within `read_tiles` supports multi-block-per-row scenarios (not used in the interleaved factory where `num_full_blocks_in_row = 1`)

### Compute Kernel

**File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`

**Key Logic**:
- Calls `compute_kernel_hw_startup(c_0, c_16)` to initialize compute hardware
- Delegates to `compute_kernel_lib::tilize<c_0, c_16>(per_core_block_tile_cnt, per_core_block_cnt)` from the kernel library
- The library function (in `tilize_helpers.inl`) uses default template parameters: `InitAndUninit` mode, `WaitBlock` wait mode, `Standard` speed mode
- Inner loop per block: `cb_wait_front(c_0, ntiles)` --> `cb_reserve_back(c_16, ntiles)` --> `tilize_block(c_0, ntiles, c_16)` --> `cb_push_back(c_16, ntiles)` --> `cb_pop_front(c_0, ntiles)`
- The actual data rearrangement happens in `tilize_block` which uses the UNPACK thread's `llk_unpack_tilize_block` to reorder row-major data into face-ordered tile format, then MATH performs datacopy, and PACK writes to output CB

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Key Logic**:
- This is a **generic** writer kernel shared across many operations (not tilize-specific)
- Gets page size from CB interface: `get_local_cb_interface(cb_id_out).fifo_page_size`
- Writes one page at a time: `cb_wait_front` --> `noc_async_write_page` --> `noc_async_writes_flushed` --> `cb_pop_front`
- Uses `noc_async_writes_flushed()` (not full barrier) between tiles for latency hiding
- Final `noc_async_write_barrier()` after all tiles ensures completion

## Implementation Notes

### FP32 Accumulation
When input dtype is FLOAT32, the compute kernel is configured with `fp32_dest_acc_en = true`, enabling 32-bit accumulation in the destination register. This is passed via `ComputeConfig`.

### Compile-Time vs Runtime Separation
The compute kernel uses compile-time args for `per_core_block_cnt` and `per_core_block_tile_cnt` because these determine the loop structure and are fixed for a given tensor shape. This means cliff cores get a **separate kernel binary** with different compile-time args, while full cores share one binary.

### Block Width and Stick Width Equivalence
In the interleaved factory, `block_width_size` equals `block_size_nbytes` (both = `padded_shape[-1] * element_size()`). The reader runtime args pass both at indices 2 and 4, but only index 4 (`block_width_size`) is actually used by the reader kernel. Index 2 is vestigial.

### override_runtime_arguments
The program supports caching via `override_runtime_arguments`, which only updates the buffer addresses (arg 0 for both reader and writer) when the same operation is called with different tensor addresses but the same shapes.

### Sub-Core Grid Support
The factory supports `sub_core_grids` from `TilizeParams`, allowing the caller to restrict which cores are used. This enables composing tilize with other operations that share the same device grid.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does tilize work in tt-metal? What is the tilize operation doing at the hardware level?"
   **Reason**: Needed to understand what `tilize_block` does at the LLK level, since the compute kernel delegates to hardware-accelerated functions.
   **Key Findings**: The tilize operation is executed by the TRISC_UNPACK thread using `llk_unpack_tilize_block`. The UNPACK thread reorders row-major data into tiled format (including face ordering). The MATH thread performs a datacopy operation, and the PACK thread writes to the output CB. On Wormhole, `fast_tilize` provides hardware-accelerated processing.

2. **Query**: "What is TensorAccessor and TensorAccessorArgs in tt-metal?"
   **Reason**: Both the reader and writer kernels use TensorAccessor for address computation. Needed to understand how logical stick/tile IDs map to physical DRAM addresses.
   **Key Findings**: `TensorAccessorArgs` is configured on the host (as compile-time or runtime args) and encodes bank count, tensor shape, and shard info. On the device, `TensorAccessor` is constructed from these args + base address + page size, and provides `get_noc_addr(page_id)` to compute physical NoC addresses for interleaved or sharded buffers.

3. **Query**: "How does the reader kernel read row-major sticks from DRAM in tilize operations?"
   **Reason**: Needed to understand the "stick" concept and how `noc_async_read` works for interleaved buffers.
   **Key Findings**: A "stick" is a single row (or portion of a row) in row-major layout. Its size equals the width in bytes. The TensorAccessor handles the interleaved bank distribution, computing which DRAM bank holds each stick. `noc_async_read` transfers the stick from DRAM to contiguous L1 memory.

4. **Query**: "What is the relationship between blocks, sticks, and tiles in the tilize operation?"
   **Reason**: Needed to clarify the block/stick/tile terminology specific to tilize.
   **Key Findings**: A "block" is a horizontal strip of tiles one tile-height tall. One block = 32 sticks = `ntiles_per_block` tiles. The reader reads 32 sticks into L1, the compute kernel converts them to tiles via `tilize_block`. `nblocks = total_tiles / ntiles_per_block`.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major vs tile layout page structure.
   **Key Information**: In row-major layout, each page = one row. In tile layout, each page = one 32x32 tile. Tiles contain 4 faces (16x16 each) stored in row-major order. Interleaved layout distributes pages round-robin across DRAM banks.

2. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding how blocks are distributed across cores.
   **Key Information**: `split_blocks_for_tilize` computes `nblocks_per_core = ceil(nblocks / grid_area)`, then `ncores = ceil(nblocks / nblocks_per_core)`. Cliff core gets `nblocks % nblocks_per_core` blocks. Returns separate CoreRangeSets for full and cliff cores.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel's tilize library function.
   **Key Information**: The `tilize<>()` template function provides configurable init/uninit, wait mode, speed mode, and non-tile-aligned CB support. Default mode (used by this factory) is Standard speed, WaitBlock, InitAndUninit. Per-block loop: wait input --> reserve output --> tilize_block --> push output --> pop input.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in the program factory.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a CircularBufferConfig with total size = `num_pages * page_size` and registers it with the program. Returns the CB ID and handle.
