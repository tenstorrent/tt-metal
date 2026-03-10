# Tilize Multi-Core Interleaved Implementation Analysis

**Focus**: Input stage -- reader kernel pattern, input CB sizing/page format, stick-to-tile batching, work distribution.

## Overview

The **tilize** operation converts a row-major (RM) interleaved tensor into tile layout. The input tensor resides in DRAM in row-major format (one stick = one row of the innermost dimension). The output tensor is in TILE_LAYOUT (32x32 tiles) in DRAM interleaved format. The operation distributes work across multiple cores, where each core reads a contiguous range of "tile rows" (groups of 32 sticks), rearranges them into tiles via the compute engine, and writes the tiled output back to DRAM.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | block (= one tile row = one row of tiles spanning the full width) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / TILE_WIDTH`) |
| **Total units** | `nblocks` = `ceil(ntiles / ntiles_per_block)` = total number of tile rows |
| **Loop structure** | Outer: iterate over tile rows (blocks); Inner: for each block, read 32 sticks full-width, then compute tilize |

A "block" is one horizontal row of tiles. For a tensor with width W elements, one block consists of `W / 32` tiles. Processing one block requires reading 32 consecutive sticks (rows) from DRAM, each of width W bytes. After the compute kernel tilizes these 32 sticks, it produces `W / 32` output tiles.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[..., H, W]` (arbitrary batch dims, H rows, W columns) |
| **Dimension convention** | Last dim is width (contiguous in memory) |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) |
| **Data type** | BFLOAT16 or FLOAT32 |

**Page definition for RM interleaved**: One page = one stick = one row of the innermost dimension. Each page is `padded_shape[-1] * element_size` bytes. Pages are distributed round-robin across DRAM banks.

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input (possibly padded to tile alignment) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (follows `output_mem_config`) |
| **Data type** | `output_dtype` (may differ from input) |

### Layout Transformations

The entire purpose of this operation is the RM-to-tile transformation. The reader delivers raw RM sticks into the input CB; the compute kernel performs the tilize (rearranging data into 32x32 tiles with face layout); the writer sends tiles to DRAM.

## Data Flow Pattern (Input Stage Focus)

### Step 1: Reader Reads 32 RM Sticks Into Input CB

For each tile-row block assigned to the core:

1. **Resolve NoC addresses for 32 sticks**: The reader pre-computes `base_src_noc_addr[0..31]` by calling `get_noc_addr(stick_id, s)` for 32 consecutive stick IDs. The `TensorAccessor s` maps logical stick indices to physical DRAM bank locations (round-robin interleaved).

2. **Reserve CB space**: `cb_reserve_back(cb_id_in0, ntiles_per_block)` -- reserves `ntiles_per_block` pages worth of space in the input CB. This call blocks if the CB is full (i.e., compute has not yet consumed previous data).

3. **Read full-width sticks via NoC**: For each of the 32 sticks, issue `noc_async_read(src_noc_addr, l1_write_addr, width_size)` where `width_size = block_width_size = padded_shape[-1] * element_size`. The L1 write address advances by `width_size` after each stick, packing sticks contiguously in L1.

4. **Barrier and push**: `noc_async_read_barrier()` waits for all 32 reads to complete, then `cb_push_back(cb_id_in0, ntiles_per_block)` signals the compute kernel that one full block of data is ready.

### Step 2: Compute Tilizes the Block

The compute kernel calls `compute_kernel_lib::tilize<c_0, c_16>()` which:
- Calls `cb_wait_front(input_cb, ntiles_per_block)` to wait for the reader's push.
- Performs `tilize_block` (or `fast_tilize_block`) converting 32 contiguous RM sticks into `ntiles_per_block` tiles.
- Pushes tiles to the output CB and pops the input CB.

### Step 3: Writer Writes Tiles to DRAM (de-emphasized)

Standard tile writer using TensorAccessor; writes tiles sequentially starting from `tile_start_id`.

### Key Insight: Symmetric CB Pages

In this interleaved program factory, the input CB is configured with **tile-sized pages** (not stick-sized pages). The reader fills `ntiles_per_block` tile-sized slots by writing 32 contiguous sticks across the full width. The compute kernel's `tilize()` is called in "symmetric" mode (no `total_input_pages` argument), meaning both input and output CBs use tile-sized pages. The hardware tilize instruction reads the RM data from the input CB pages and reorganizes it into tile layout in the output CB pages.

## Circular Buffer Configuration (Input Stage Focus)

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|-----------|----------|----------|----------|
| `c_0` | cb_input | Input RM sticks (tile-sized pages) | `ntiles_per_block` | `input_single_tile_size` | Single | Reader | Compute | Block |
| `c_16` | cb_output | Output tiles | `ntiles_per_block` | `output_single_tile_size` | Single | Compute | Writer | Block |

### Input CB (c_0) Sizing Details

- **Page size**: `input_single_tile_size` = `tt::tile_size(input_cb_data_format)`. For BFLOAT16, this is 2048 bytes (32 * 32 * 2 bytes). For FLOAT32, this is 4096 bytes.
- **Number of pages**: `ntiles_per_block` = `padded_shape[-1] / TILE_WIDTH` = width in tiles.
- **Total CB capacity in bytes**: `ntiles_per_block * input_single_tile_size`.
- **Physical meaning**: The CB holds exactly one full tile-row's worth of data. The reader writes 32 sticks * full_width bytes = `32 * padded_shape[-1] * element_size` bytes into it. This equals `ntiles_per_block * tile_size` because `tile_size = 32 * 32 * element_size` and `ntiles_per_block = width / 32`.
- **Buffering**: Single-buffered (capacity == block size). The reader and compute alternate: the reader fills the entire CB for one block, then the compute consumes it before the reader can fill the next block.

### Why Single-Buffered Works Here

The CB capacity equals exactly one block. The reader reserves all `ntiles_per_block` pages, fills them with 32 sticks, then pushes. The compute waits for all pages, processes, then pops. There is no overlap between reader and compute for the same block; the pipeline is: read block N -> compute block N -> read block N+1 -> compute block N+1.

## Reader Kernel: Detailed Analysis

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

### Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `stick_size` | `uint32_t` | Size of one RM stick in bytes (`padded_shape[-1] * element_size`) |
| 1+ | TensorAccessor args | (multiple) | Buffer distribution metadata appended by `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` |

The `TensorAccessorArgs` is constructed with default `ArgConfig::None`, meaning all accessor parameters (rank, num_banks, tensor_shape, shard_shape, bank_coords) are passed as compile-time arguments. This makes stick-to-bank mapping a fully compile-time-resolved operation in the kernel.

### Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | `uint32_t` | Base address of input buffer in DRAM |
| 1 | `num_sticks` | `uint32_t` | Total sticks this core processes (`nblocks_per_core * TILE_HEIGHT`) |
| 2 | `block_size_nbytes` | `uint32_t` | Not used in kernel (arg index 2 is not read) |
| 3 | `num_tiles_per_block` | `uint32_t` | Tiles per block = `padded_shape[-1] / 32` |
| 4 | `block_width_size` | `uint32_t` | Width of one stick in bytes (same as `stick_size` CT arg) |
| 5 | `num_full_blocks_in_row` | `uint32_t` | Always 1 in this factory (entire width in one block) |
| 6 | (unused) | `uint32_t` | `num_leftover_tiles` = 0 |
| 7 | (unused) | `uint32_t` | `leftover_width_in_row` = 0 |
| 8 | `start_stick_id` | `uint32_t` | First stick ID for this core (`row_start_id`) |

**Note on unused arguments**: Arguments at indices 2, 6, and 7 relate to a "split rows" capability where a single row of tiles might be split across multiple blocks. In this interleaved factory, the full width always fits in one block (`num_full_blocks_in_row = 1`, no leftover tiles), so those args are zero.

### Reader Loop Structure

```
stick_id = start_stick_id
for i in range(num_sticks / 32):           // iterate over tile-rows (blocks)
    for j in range(32):                     // pre-resolve 32 stick addresses
        base_src_noc_addr[j] = get_noc_addr(stick_id, s)
        stick_id++

    for j in range(num_full_blocks_in_row): // always 1 iteration here
        read_tiles(num_tiles_per_block, block_width_size)
```

The `read_tiles` lambda:
```
cb_reserve_back(cb_id_in0, num_tiles)       // block until CB has space
l1_write_addr = get_write_ptr(cb_id_in0)    // get L1 address for writing
for k in range(32):                          // read 32 sticks
    noc_async_read(base_src_noc_addr[k], l1_write_addr, width_size)
    l1_write_addr += width_size
    base_src_noc_addr[k] += width_size       // advance for potential next block (unused here)
noc_async_read_barrier()                     // wait for all reads
cb_push_back(cb_id_in0, num_tiles)           // signal data ready
```

### Stick-to-Tile Batching Pattern

Each CB push delivers **exactly one tile row** (32 sticks wide, `ntiles_per_block` tiles). The batching ratio is:
- **32 sticks per push** (always TILE_HEIGHT = 32)
- **ntiles_per_block tiles per push** (= width / 32)
- All 32 sticks are read in a single `cb_reserve_back` / `cb_push_back` pair

This is a "batch-all-sticks-for-one-tile-row" pattern. The reader never pushes partial tile rows.

### NoC Read Pattern

- **32 individual `noc_async_read` calls** per block, one per stick.
- Each read transfers `block_width_size` bytes (full row width).
- Reads are issued back-to-back (non-blocking), then a single `noc_async_read_barrier()` waits for all.
- Sticks land contiguously in L1, packed in row order: stick 0 at offset 0, stick 1 at offset `width_size`, ..., stick 31 at offset `31 * width_size`.
- Source addresses come from different DRAM banks (interleaved round-robin), so the 32 reads may access different banks, enabling some parallelism in the DRAM controller.

## Memory Access Patterns

### Read Pattern (Input, DRAM -> L1)

- **Pattern**: Strided across DRAM banks, sequential within each stick.
- **Granularity**: One full-width stick per NoC read (entire row).
- **Ordering**: Sticks are read in ascending stick_id order within each block of 32.
- **Bank access**: Round-robin interleaved. Consecutive sticks map to consecutive banks, wrapping around. The 32 sticks in one tile-row typically span multiple banks.
- **Batching**: 32 reads are issued without barriers between them, maximizing NoC utilization. A single barrier follows.

### Write Pattern (L1 -> DRAM, de-emphasized)

Standard sequential tile writes via `noc_async_write_tile`. Tiles are written in order starting from `tile_start_id`.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of cores from available grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` tile rows (blocks) |
| **Load balancing** | Near-equal with optional cliff core |

### Distribution Algorithm (`split_blocks_for_tilize`)

1. `nblocks_per_core = ceil(nblocks / grid_area)` -- target blocks per core.
2. `ncores = ceil(nblocks / nblocks_per_core)` -- actual cores needed.
3. `nblocks_per_core_cliff = nblocks % nblocks_per_core` -- remainder for cliff core.
4. If `nblocks_per_core_cliff > 0`, the last core gets fewer blocks (cliff core).
5. If `nblocks_per_core_cliff == 0`, all cores get equal work.

### Per-Core State

Each core gets a contiguous range of sticks and tiles:
- **Reader**: `start_stick_id = row_start_id`, `num_sticks = nblocks_per_core * 32`
- **Writer**: `tile_start_id` and `ntiles = ntiles_per_block * nblocks_per_core`

The ranges are non-overlapping and cover the entire tensor. Core `i` processes sticks `[i * nblocks_per_core * 32, (i+1) * nblocks_per_core * 32)`.

### Example

For a tensor of shape `[1, 1, 256, 128]` with BFLOAT16:
- `ntiles = (256 * 128) / (32 * 32) = 32` tiles
- `ntiles_per_block = 128 / 32 = 4` tiles per block
- `nblocks = 256 / 32 = 8` blocks
- With 8 available cores: `nblocks_per_core = 1`, each core processes 1 block (32 sticks, 4 tiles)
- With 3 available cores: `nblocks_per_core = ceil(8/3) = 3`, `ncores = ceil(8/3) = 3`, cliff core gets `8 % 3 = 2` blocks

## Compute Kernel (Brief)

**File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`

Uses `compute_kernel_lib::tilize<c_0, c_16>()` helper with:
- `WaitMode::WaitBlock` -- compute waits per block for reader to push
- `InitUninitMode::InitAndUninit` -- full init/uninit lifecycle
- Symmetric CB pages (no `total_input_pages` arg)

Compile-time args: `{nblocks_per_core, ntiles_per_block}` (or cliff variants).

The tilize helper automatically selects `fast_tilize_block` when conditions are met (32x32 tiles, Float32/Float16_b format, half-sync dest mode).

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Classification |
|----|----------|------------|-------|----------------|
| c_0 (input) | `ntiles_per_block` pages | `ntiles_per_block` pages | 1:1 | Single-buffered |
| c_16 (output) | `ntiles_per_block` pages | `ntiles_per_block` pages | 1:1 | Single-buffered |

Both CBs are single-buffered. Reader and compute do not overlap on the same block.

## Implementation Notes

### TensorAccessor for RM Sticks

The `TensorAccessor` is the standard mechanism for reading interleaved data. It is constructed from `TensorAccessorArgs` with default config (`ArgConfig::None` = all compile-time). The host calls `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` which appends buffer metadata (rank, num_banks, tensor_shape, bank_coords) to the compile-time argument vector. On the device side, `TensorAccessorArgs<1>()` starts parsing at compile-time arg index 1 (index 0 is `stick_size`). The resulting `TensorAccessor` object maps `get_noc_addr(stick_id, s)` to the correct DRAM bank + offset.

### Relevance to layer_norm_rm

For a new `layer_norm_rm` operation that reads RM interleaved input:
1. **Same reader pattern**: Read 32 sticks at a time using TensorAccessor with `get_noc_addr(stick_id, s)`.
2. **Same CB sizing for input**: Size the input CB to hold one tile-row of sticks (`ntiles_per_block` tile-sized pages or equivalently 32 * width bytes).
3. **Same work distribution**: Use `split_blocks_for_tilize` or a similar 1D block distribution to assign tile-rows to cores.
4. **Key difference**: After reading sticks, layer_norm needs to compute mean/variance along the width dimension rather than tilize. The reader pattern remains identical; the compute kernel changes entirely.
5. **Consider double-buffering**: For layer_norm, if the compute is more expensive (multi-pass reductions), double-buffering the input CB (`num_pages = 2 * ntiles_per_block`) would allow the reader to prefetch the next block while compute processes the current one.

### The "Split Rows" Name

The reader kernel is named `reader_unary_stick_layout_split_rows_interleaved.cpp`. The "split rows" refers to a capability to split one tile-width row into multiple sub-blocks (via `num_full_blocks_in_row` and leftover tiles). In this interleaved factory, the full width always fits in one block, so this split capability is unused (`num_full_blocks_in_row = 1`, leftover = 0).

### fp32_llk_acc

When the input dtype is FLOAT32, the compute kernel enables `fp32_dest_acc_en` for higher-precision accumulation in the destination registers.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor work in tt-metal? How does TensorAccessorArgs pass buffer information as compile-time args, and how does the TensorAccessor constructor work in kernel code?"
   **Reason**: Understanding how the reader kernel resolves stick addresses from logical indices to physical DRAM locations.
   **Key Findings**: `TensorAccessorArgs` serializes buffer metadata (rank, num_banks, shapes, bank_coords) into compile-time args. On device, `TensorAccessorArgs<CTA_OFFSET>()` reconstructs these. `get_noc_addr(stick_id, accessor)` computes the physical NoC address for a given page.

2. **Query**: "How do circular buffers work in tt-metal? Explain cb_reserve_back, cb_push_back, and get_write_ptr."
   **Reason**: Understanding the producer-consumer synchronization between reader and compute kernels.
   **Key Findings**: `cb_reserve_back` is blocking -- it stalls if the CB is full. `get_write_ptr` returns the L1 address for writing. `cb_push_back` increments the pages_received counter, signaling consumers. The capacity/page_size ratio determines how many pages can be in-flight.

3. **Query**: "In tt-metal tilize operations, what does the term 'block' mean?"
   **Reason**: Clarifying the work unit granularity used by `split_blocks_for_tilize`.
   **Key Findings**: A block = one row of tiles spanning the width dimension. `nblocks` = total tile rows. `nblocks_per_core` = blocks assigned to each core. Cliff core handles remainder.

4. **Query**: "How does noc_async_read work? Is it non-blocking? Does noc_async_read_barrier wait for all pending reads?"
   **Reason**: Understanding the read pipeline in the reader kernel where 32 reads are issued before one barrier.
   **Key Findings**: `noc_async_read` is non-blocking, takes (noc_addr, l1_addr, size). Multiple reads can be in-flight. `noc_async_read_barrier` blocks until all pending reads complete. Sticks land contiguously in L1.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM vs tiled page definitions and interleaved memory layout.
   **Key Information**: RM pages = one row (stick). Tiles = 32x32 with 16x16 faces. Interleaved = round-robin page distribution across banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding host-side setup and device-side usage of TensorAccessor.
   **Key Information**: `TensorAccessorArgs(buffer)` with `ArgConfig::None` makes everything compile-time. `append_to(ct_args)` serializes to the compile-time arg vector. On device, `get_noc_addr(page_id)` handles the bank mapping.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding compute kernel behavior to verify CB synchronization pattern.
   **Key Information**: In symmetric mode (no `total_input_pages`), compute uses `cb_wait_front(input_cb, block_width_tiles)` per block, matching the reader's push of `ntiles_per_block` pages. The tilize automatically selects fast vs standard path at compile time.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` work distribution function.
   **Key Information**: 1D distribution: `nblocks_per_core = ceil(total / grid_area)`, cliff core gets remainder. Returns `BlockSplit` struct with core ranges and per-core block counts.
