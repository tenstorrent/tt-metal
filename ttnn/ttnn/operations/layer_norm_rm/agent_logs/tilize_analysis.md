# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The tilize operation converts a tensor from **row-major (stick) layout** to **tile layout** (32x32 tiles). This is the fundamental format conversion required before any tile-based compute can happen on Tenstorrent hardware. The multi-core interleaved variant distributes rows across available Tensix cores and reads from DRAM-interleaved row-major buffers.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Role focus**: input_stage -- emphasis on reader kernel pattern, input CB sizing, stick-to-tile batching, and core work distribution.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | block (a horizontal strip of tiles spanning the full tensor width, 32 rows tall) |
| **Unit size** | `ntiles_per_block` tiles = `padded_shape[-1] / TILE_WIDTH` tiles across one tile-row |
| **Total units** | `nblocks` = `ceil(total_tiles / ntiles_per_block)` = total number of tile-height rows |
| **Loop structure** | Outer: iterate over blocks (groups of 32 sticks). Inner: iterate across width reading `ntiles_per_block` tiles per block. |

A **block** in this operation corresponds to one tile-height strip (32 rows) spanning the full width of the tensor. The number of blocks equals the number of tile-rows in the tensor: `physical_volume / TILE_HW / ntiles_per_block`.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [N, C, H, W] (arbitrary rank, flattened to 2D internally) |
| **Dimension convention** | Last dimension = W (stick width); all outer dims collapsed |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16 or FLOAT32 |

**Key metric**: `stick_size` = `padded_shape[-1] * element_size()` bytes. This is the size of one row-major page (one stick). In the interleaved layout, each stick is one page distributed round-robin across DRAM banks.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input (BFLOAT16 or FLOAT32) |

### Layout Transformations

The core transformation is ROW_MAJOR to TILE_LAYOUT. The reader reads 32 consecutive sticks (rows) and packs them into the input CB as a contiguous block. The compute kernel then performs the tilize operation (rearranging row-major data into the tile format with faces). No explicit untilize, reshard, or format conversion occurs -- the compute `tilize_block` LLK handles the row-to-tile reorganization.

## Data Flow Pattern (Input Stage Focus)

### Step 1: Reader Reads 32 Sticks from DRAM

For each block (group of 32 rows):

1. **Resolve NOC addresses**: The reader pre-computes NOC addresses for all 32 sticks in the group by calling `get_noc_addr(stick_id, s)` for each of the 32 consecutive stick IDs. This uses the `TensorAccessor` to map logical stick IDs to physical DRAM bank locations (interleaved round-robin).

2. **Reserve CB space**: `cb_reserve_back(cb_id_in0, ntiles_per_block)` -- reserves space for one full row of tiles (the entire width of the tensor).

3. **Read sticks into CB**: For each of the 32 sticks (`k = 0..31`), issues `noc_async_read(src_noc_addr, l1_write_addr, width_size)` to read the full stick width. The L1 write pointer advances by `width_size` after each stick, so the 32 sticks are laid out contiguously in the CB.

4. **Barrier**: `noc_async_read_barrier()` -- waits for all 32 reads to complete.

5. **Push to CB**: `cb_push_back(cb_id_in0, ntiles_per_block)` -- makes the data available to the compute kernel.

### Step 2: Compute Tilizes the Block

The compute kernel calls `compute_kernel_lib::tilize<c_0, c_16>(per_core_block_tile_cnt, per_core_block_cnt)` which:
- Waits for `ntiles_per_block` pages on input CB (c_0) per block
- Reserves `ntiles_per_block` pages on output CB (c_16)
- Calls `tilize_block` (or `fast_tilize_block` when hardware supports it) to convert row-major to tile format
- Pushes tiles to output CB, pops input CB

### Step 3: Writer Writes Tiles to DRAM (de-emphasized)

The writer reads tiles one at a time from CB c_16 and writes them to DRAM using `noc_async_write_page`.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_id_in0 | Input staging (row-major sticks) | ntiles_per_block tiles | ntiles_per_block tiles | Single | Reader | Compute | Block |
| c_16 | cb_id_out0 | Output staging (tilized tiles) | ntiles_per_block tiles | ntiles_per_block tiles | Single | Compute | Writer | Block |

**Critical detail for input CB (c_0)**: The capacity is set to exactly `ntiles_per_block` tiles, where each tile-sized page is `input_single_tile_size` bytes. The total CB allocation is `ntiles_per_block * input_single_tile_size` bytes. Since capacity equals block size, this is **single-buffered** -- the reader must wait for the compute to consume before writing the next block.

**Page size in the input CB**: The page size is `input_single_tile_size` (tile-sized, e.g., 2048 bytes for BFLOAT16 32x32 tiles). Although the reader writes row-major sticks into the CB, the CB pages are tile-sized. The reader writes 32 sticks contiguously, which together fill exactly `ntiles_per_block` tile-sized pages. The compute kernel then treats the same memory as tile-sized pages for the tilize operation. This is the **symmetric page mode** of the tilize helper -- input and output CBs both have tile-sized pages.

### How Stick Data Maps to Tile-Sized CB Pages

Consider a tensor with width W = 128 elements (BFLOAT16).
- `ntiles_per_block` = 128 / 32 = 4 tiles across.
- `block_size_nbytes` = 128 * 2 = 256 bytes per stick.
- 32 sticks * 256 bytes = 8192 bytes total in CB per block.
- 4 tiles * 2048 bytes/tile = 8192 bytes = same total.

The reader fills the CB with raw row-major data (32 consecutive sticks laid out linearly). The compute kernel interprets this same memory region as 4 tiles and rearranges it into tile format in the output CB.

## Pipeline Pattern Summary

Both CBs (c_0 and c_16) have capacity equal to block size (`ntiles_per_block`), making this a **single-buffered** pipeline. The reader cannot start filling the next block until compute has consumed the current one, and compute cannot start the next block until the writer has drained the output CB. There is no overlap between consecutive blocks on the same core.

## Index Calculations

### Stick ID to NOC Address

The reader uses `TensorAccessor` to convert stick IDs to physical DRAM addresses:

```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // compile-time args at index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

For interleaved buffers, `TensorAccessor` maps the logical stick ID to a physical bank via round-robin: `bank_id = stick_id % num_banks`. The NOC address encodes both the bank's physical coordinates on the chip and the offset within that bank.

### Stick ID Assignment Per Core

Each core receives a contiguous range of sticks. The host computes:
- `row_start_id` for each core: the first stick ID this core will read.
- Full cores: `row_start_id += TILE_HEIGHT * nblocks_per_core` (each block = 32 sticks).
- Cliff core: gets the remainder.

The reader increments `stick_id` sequentially from `start_stick_id` through all its assigned blocks.

### Width Traversal Within a Block

The `read_tiles` lambda reads the full width in one call (`num_full_blocks_in_row = 1` in this factory, `block_width_size = stick_size`). The `base_src_noc_addr[k]` array tracks per-stick progress across the width. However, in this interleaved factory, the full width is read in a single pass (one call to `read_tiles` per tile-row), so width splitting is not exercised.

## Memory Access Patterns

### Read Pattern

**Stick-sequential, full-width**: The reader reads 32 consecutive sticks (one tile-height) in sequence. For each stick, it issues one `noc_async_read` for the full stick width. The 32 reads within a block target 32 different stick IDs, which may map to different DRAM banks (interleaved round-robin). After all 32 are issued, a single `noc_async_read_barrier()` waits for completion.

**Access characteristics**:
- Each `noc_async_read` reads `block_size_nbytes` = `padded_shape[-1] * element_size` bytes (the full stick).
- Reads are to contiguous L1 addresses (sticks packed back-to-back in CB).
- Source addresses are non-contiguous in DRAM (different sticks may be in different banks).
- The 32 reads within a block are independent and can be pipelined by the NoC.

### Write Pattern (de-emphasized)

The writer writes one tile at a time, sequentially by tile ID, using `noc_async_write_page`. Tiles are written to interleaved DRAM in order.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of cores from available grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `ncores` = `ceil(nblocks / nblocks_per_core)` |
| **Work per core** | `nblocks_per_core` blocks (each block = 32 sticks wide, `ntiles_per_block` tiles) |
| **Load balancing** | Nearly equal, with optional cliff core |

### Work Splitting Details

The `split_blocks_for_tilize(available_grid, nblocks)` function:

1. Computes `nblocks_per_core = ceil(nblocks / grid_area)` -- blocks assigned to each full core.
2. Computes `nblocks_per_core_cliff = nblocks % nblocks_per_core` -- remainder for the last core (0 means no cliff).
3. `ncores = ceil(nblocks / nblocks_per_core)` -- actual cores used.
4. Returns `core_range` (full cores) and `core_range_cliff` (0 or 1 cliff core).

The full cores each process `nblocks_per_core * TILE_HEIGHT` sticks (i.e., `nblocks_per_core` tile-rows). The cliff core processes `nblocks_per_core_cliff * TILE_HEIGHT` sticks.

Cores are enumerated from the `available_grid` (or the full compute grid if no sub-core-grids specified) using `corerange_to_cores()`, which linearizes the 2D grid in row-major order.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size (block_size_nbytes) | uint32_t | Size of one row-major stick in bytes: `padded_shape[-1] * element_size` |
| 1+ | TensorAccessorArgs | varies | Interleaved buffer access metadata (rank, num_banks, tensor_shape, bank_coords, etc.) appended via `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt (nblocks_per_core) | uint32_t | Number of blocks (tile-height rows) this core processes |
| 1 | per_core_block_tile_cnt (ntiles_per_block) | uint32_t | Number of tiles per block (tiles across the width) |

Note: Cliff cores receive `nblocks_per_core_cliff` for index 0 instead.

#### Writer Kernel (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_16) |
| 1+ | TensorAccessorArgs | varies | Output buffer access metadata |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total sticks to read: `nblocks_per_core * TILE_HEIGHT` |
| 2 | block_size_nbytes | uint32_t | Bytes per stick (same as compile-time arg 0, passed redundantly) |
| 3 | ntiles_per_block | uint32_t | Tiles across the width |
| 4 | block_width_size | uint32_t | Width of the block in bytes (same as block_size_nbytes here) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 in this factory (full width in one pass) |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial width blocks) |
| 7 | leftover_width_in_row | uint32_t | Always 0 (no partial width) |
| 8 | start_stick_id (row_start_id) | uint32_t | First stick ID for this core |

Note: The reader kernel only uses indices 0, 1, 3, 4, 5, and 8 (skips index 2, 6, 7 in this configuration).

#### Writer Kernel (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_tiles | uint32_t | Total tiles to write: `ntiles_per_block * nblocks_per_core` |
| 2 | start_tile_id (tile_start_id) | uint32_t | First tile ID for this core |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (interleaved sticks) | CB c_0 | Read 32 sticks per block, push ntiles_per_block pages |

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

**Key Logic**:

1. **Pre-resolve addresses**: For each group of 32 sticks, pre-computes all 32 NOC addresses into `base_src_noc_addr[32]` array using `get_noc_addr(stick_id, s)`.

2. **Bulk read via lambda**: The `read_tiles` lambda:
   - Calls `cb_reserve_back(cb_id_in0, num_tiles)` to acquire CB space.
   - Gets the L1 write pointer with `get_write_ptr(cb_id_in0)`.
   - Loops over all 32 sticks, issuing `noc_async_read` for each one (reading `width_size` bytes per stick).
   - Increments L1 write address by `width_size` per stick (packing sticks contiguously).
   - Also increments `base_src_noc_addr[k]` by `width_size` to support width splitting (not used here).
   - After all 32 reads are issued, calls `noc_async_read_barrier()`.
   - Pushes `num_tiles` pages to CB.

3. **Outer loop**: `for (i = 0; i < num_sticks / tile_height; i++)` iterates over blocks. Each block processes exactly 32 sticks. Within each block, the inner loop `for (j = 0; j < num_full_blocks_in_row; j++)` iterates across the width (always 1 iteration in this factory).

**Design intent for layer_norm_rm reuse**: This reader pattern (reading 32 contiguous RM sticks from interleaved DRAM, batching them into a single CB push of `ntiles_per_block` tile-sized pages) is directly applicable to reading input data for a row-major layer norm. The key adaptation would be: instead of tilizing the data, the compute kernel would perform normalization. The reader's stick-batching strategy, TensorAccessor usage, and CB management are all reusable.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_2 | N/A | CB c_0 | CB c_16 | tilize_block (row-major to tile conversion) |

**File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`

**Key Logic**: Calls `compute_kernel_lib::tilize<c_0, c_16>(per_core_block_tile_cnt, per_core_block_cnt)` in symmetric mode (no `total_input_pages` argument). This means input and output CBs both have tile-sized pages. The tilize helper loops over `num_blocks`, each time waiting for `block_width_tiles` input pages, then calling `tilize_block` to rearrange data from row-major to tile format with faces.

### Writer Kernel (de-emphasized)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

Standard single-tile-at-a-time writer using `noc_async_write_page`.

## Implementation Notes

### Reader Design Patterns Relevant to layer_norm_rm

1. **Stick-based reading with tile-sized CB pages**: The reader writes raw row-major sticks into a CB configured with tile-sized pages. This works because 32 sticks * stick_width = ntiles_per_block * tile_size. For layer_norm_rm, the same approach can be used: read RM sticks into a CB, then let the compute kernel process them. The CB page size can be set to tile size if the compute kernel will tilize, or to stick size if compute operates on raw rows.

2. **TensorAccessor for interleaved DRAM access**: The pattern of constructing `TensorAccessorArgs` on the host and `TensorAccessor` on the device, then using `get_noc_addr(page_id, accessor)` per stick, is the standard approach for reading interleaved buffers. This is directly reusable.

3. **Pre-resolving 32 NOC addresses**: The `base_src_noc_addr[32]` array pattern pre-computes addresses for a tile-height batch of sticks before issuing reads. This allows all 32 `noc_async_read` calls to be issued without intervening address calculations, maximizing NoC utilization.

4. **Single barrier per block**: All 32 reads within a block are issued before a single `noc_async_read_barrier()`. This allows the NoC to pipeline the reads. This is more efficient than per-stick barriers.

5. **Full-width single-pass reading**: In this factory, `num_full_blocks_in_row = 1` and `block_width_size = stick_size`, so the entire width is read in one pass. The kernel supports width splitting (the lambda advances `base_src_noc_addr[k]`), but the factory does not use it. For layer_norm_rm, reading the full width per stick is natural since normalization operates across the entire last dimension.

### CB Sizing Considerations for layer_norm_rm

- The tilize operation uses `ntiles_per_block` as both CB capacity and block size (single-buffered).
- For layer_norm_rm, if the compute kernel needs to see multiple blocks simultaneously (e.g., to compute mean/variance across the height dimension), the CB sizing strategy would need to differ.
- However, if layer_norm operates per-row (normalizing across the width), the same single-block CB capacity is sufficient per block iteration.

### Compile-Time vs Runtime Argument Strategy

- `stick_size` (block_size_nbytes) is a compile-time argument because it determines the DMA transfer size and does not change between calls with the same tensor shape.
- `src_addr` and `start_stick_id` are runtime arguments because they change with different tensor instances and per-core assignment.
- TensorAccessorArgs are compile-time (default configuration), encoding bank layout information that is fixed for a given buffer allocation.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in ttnn? Specifically, what is the reader kernel pattern for reading row-major sticks from DRAM and converting them to tile layout?"
   **Reason**: Needed to understand the overall tilize data flow pattern and confirm the relationship between reader/compute/writer kernels.
   **Key Findings**: The tilize operation uses LLK APIs (`llk_unpack_tilize`) on the compute side. The reader reads row-major sticks into CBs, and the compute kernel performs the actual row-to-tile rearrangement. Confirmed that the reader is purely a data mover.

2. **Query**: "How does TensorAccessor and TensorAccessorArgs work in tt-metal? What does get_noc_addr with a TensorAccessor do for interleaved buffers?"
   **Reason**: The reader kernel uses `TensorAccessor` to compute DRAM addresses from stick IDs. Needed to understand this mapping.
   **Key Findings**: For interleaved buffers, `get_noc_addr(stick_id, accessor)` maps logical stick IDs to physical DRAM bank addresses via round-robin. The TensorAccessor encapsulates rank, bank count, tensor shape, and bank coordinates. Host-side `TensorAccessorArgs` serializes this as compile-time args.

3. **Query**: "What is the split_blocks_for_tilize function and how does it distribute work across cores?"
   **Reason**: Needed to understand the 1D block distribution strategy and what a "block" means.
   **Key Findings**: A block = one tile-height row of tiles (all tiles across the width at one tile-row). `split_blocks_for_tilize` distributes blocks using ceiling division, with an optional cliff core for the remainder. Returns BlockSplit with core ranges and per-core block counts.

4. **Query**: "How does cb_reserve_back and cb_push_back work for circular buffers? What is the relationship between page_size and num_pages?"
   **Reason**: Needed to confirm CB capacity constraints and blocking behavior.
   **Key Findings**: `cb_reserve_back` blocks until the requested pages are free. `num_pages = total_size / page_size`. You cannot reserve more pages than the CB's total capacity. The CB is single-buffered when capacity = block size.

5. **Query**: "In tt-metal tilize operation, how does the reader kernel read row-major sticks and batch them into tile-height groups of 32?"
   **Reason**: Needed detailed understanding of the stick-to-tile batching mechanism.
   **Key Findings**: Confirmed that 32 sticks are read one at a time via `noc_async_read`, packed contiguously in L1. A single barrier waits for all 32. The total bytes (32 sticks * stick_width) equals `ntiles_per_block * tile_size`.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major vs tile layout, page definitions, and interleaved memory layout.
   **Key Information**: In row-major layout, each row is one page. In interleaved layout, pages are distributed round-robin across DRAM banks. A tile is 32x32 with 4 faces of 16x16 each.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor maps page IDs to NOC addresses.
   **Key Information**: `TensorAccessor` is constructed from `TensorAccessorArgs` + base_address + page_size. `get_noc_addr(page_id)` returns a 64-bit NOC address encoding bank coordinates and offset. For interleaved buffers, bank_id = page_id % num_banks.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute-side tilize implementation and CB interaction pattern.
   **Key Information**: The `tilize<input_cb, output_cb>()` function supports symmetric mode (both CBs tile-sized pages) and asymmetric mode. In symmetric mode (used here), it waits for `block_width_tiles` pages per block, calls `tilize_block`, then pushes/pops. Supports fast_tilize when hardware conditions are met (32x32 tiles, Float32/Float16_b, half-sync dest).

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the block splitting algorithm for core distribution.
   **Key Information**: `split_blocks_for_tilize` uses ceiling division to distribute blocks evenly, with a cliff core for remainders. Returns BlockSplit struct with core ranges, per-core counts, and cliff information.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in the program factory.
   **Key Information**: `create_cb(cb_id, program, core_spec, page_size, num_pages, data_format)` creates a circular buffer with total size = `num_pages * page_size`. Supports optional globally allocated buffers.
