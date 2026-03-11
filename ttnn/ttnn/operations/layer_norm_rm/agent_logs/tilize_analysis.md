# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The tilize operation converts a row-major (RM) tensor stored in DRAM into a tiled-layout tensor. The input is an interleaved RM tensor where each "page" is one row (stick) of width equal to `padded_shape[-1]`. The output is an interleaved tiled tensor with pages of tile size (32x32 elements). The operation reads groups of 32 consecutive sticks from DRAM, pushes them into an L1 circular buffer, and lets the compute kernel rearrange the data into tile format.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Focus**: This analysis emphasizes the reader kernel pattern, input CB sizing, stick-to-tile batching, and core distribution strategy, as requested for use as an `input_stage` reference for a `layer_norm_rm` operation.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a block = 32 sticks x full row width) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / TILE_WIDTH`) |
| **Total units** | `nblocks` = `physical_volume / TILE_HW / ntiles_per_block` = number of tile-rows |
| **Loop structure** | Outer: iterate over tile-rows (blocks of 32 sticks). Inner: for each tile-row, read all horizontal tile-columns in one pass. |

A "block" corresponds to one tile-row: 32 consecutive sticks spanning the full tensor width. Each block produces `ntiles_per_block` tiles (one tile per 32-element column segment across the width).

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] (flattened outer dims) | [N, C, H, W] (same logical shape) |
| **Dimension convention** | Last dim = W = contiguous in memory | Last dim = W |
| **Tensor layout** | ROW_MAJOR | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (or L1) | DRAM (or L1) |
| **Data type** | BFLOAT16 / FLOAT32 | Same as input (or `output_dtype`) |

### Input Tensor Details

- **Page definition**: One page = one stick = one row of `padded_shape[-1]` elements.
- **Page size in bytes**: `padded_shape[-1] * element_size()` (this is `block_size_nbytes` in the code).
- **Total pages**: `physical_volume / padded_shape[-1]` = total number of sticks.
- **Interleaved distribution**: Pages (sticks) are distributed round-robin across DRAM banks.

### Output Tensor Details

- **Page definition**: One page = one 32x32 tile.
- **Page size**: `tt::tile_size(output_cb_data_format)` bytes.
- **Total pages**: `ntiles` = `physical_volume / TILE_HW`.

### Layout Transformation

The core transformation is: 32 consecutive RM sticks (each of width W) are rearranged into `W/32` tiles, each 32x32. This is the tilize operation performed by the compute kernel's `tilize_block` or `fast_tilize_block` hardware instruction.

## Data Flow Pattern

### Step-by-step Flow

1. **Reader** resolves NoC addresses for 32 consecutive sticks using `TensorAccessor::get_noc_addr(stick_id)`.
2. **Reader** reserves `ntiles_per_block` pages in CB c_0 via `cb_reserve_back`.
3. **Reader** issues 32 `noc_async_read` calls, one per stick, each reading `block_width_size` bytes (full row width) into consecutive positions in the CB. The 32 sticks are laid out contiguously in L1.
4. **Reader** calls `noc_async_read_barrier()` to wait for all 32 reads to complete.
5. **Reader** calls `cb_push_back(c_0, ntiles_per_block)` to signal that `ntiles_per_block` tiles' worth of data is ready.
6. **Compute** calls `cb_wait_front(c_0, ntiles_per_block)` to wait for the data.
7. **Compute** calls `tilize_block` (or `fast_tilize_block`) to convert the 32-row RM block into `ntiles_per_block` tiles, writing results into CB c_16.
8. **Compute** calls `cb_push_back(c_16, ntiles_per_block)` and `cb_pop_front(c_0, ntiles_per_block)`.
9. **Writer** reads tiles one at a time from CB c_16 and writes them to DRAM using the output TensorAccessor.

### Key Insight for layer_norm_rm Reuse

The reader's pattern of "read 32 sticks into a CB, then push `ntiles_per_block` pages" is the canonical RM-to-tile batching pattern. The CB page size is `input_single_tile_size` (tile-sized pages), but the reader fills them with raw RM stick data arranged so that the compute kernel can interpret 32 consecutive stick-widths as tile data. This means:

- **The input CB page size is tile-sized** even though the data written into it is raw RM sticks.
- **The reader writes exactly 32 sticks** (tile height) per push, filling `ntiles_per_block` tile-pages.
- **Each push provides one complete tile-row** to the compute kernel.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input staging (RM sticks batched as tile-pages) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_output | Output staging (tiled data) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Compute | Writer | Block |

### Input CB Sizing Details (Focus Area)

- **Capacity**: `ntiles_per_block * input_single_tile_size` bytes.
- **`ntiles_per_block`** = `padded_shape[-1] / TILE_WIDTH` = number of tiles spanning the width.
- **`input_single_tile_size`** = `tt::tile_size(input_cb_data_format)` = size of one tile in the input data format.
- **Why this size**: The reader writes 32 sticks of `block_width_size` bytes each. Total bytes written = `32 * padded_shape[-1] * element_size()`. This equals `ntiles_per_block * tile_size` because `32 * W * elem_size = (W/32) * (32*32*elem_size)`.
- **Buffering**: Single-buffered (capacity = block size). The reader and compute alternate on the same buffer space. No overlap is possible between reading the next block and computing the current block.

### Why Single-Buffered for This Pattern

The CB capacity equals the block size (`ntiles_per_block` pages). This means the reader must wait for compute to pop the current block before it can reserve space for the next block. This is a deliberate choice to minimize L1 usage since `ntiles_per_block` can be large (e.g., 32 tiles for a 1024-wide tensor).

## Pipeline Pattern Summary

Both CB c_0 and CB c_16 are single-buffered (capacity = block size). This means:
- Reader and compute cannot overlap on c_0.
- Compute and writer cannot overlap on c_16 for a full block, but the writer processes tiles one at a time, so partial overlap within a block is possible.

## Index Calculations

### Stick-to-NoC Address Mapping

The reader uses `TensorAccessor` to convert a linear `stick_id` to a 64-bit NoC address:

```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // compile-time args starting at index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

The `TensorAccessor` encapsulates the interleaved bank mapping: given a stick_id (page index), it computes which DRAM bank holds that page and the byte offset within that bank, then returns the full NoC address. The `stick_size` parameter tells it the page size in bytes (= `block_size_nbytes`, a compile-time arg).

### Block-to-Stick Mapping

For each tile-row block `i`:
- `stick_id` starts at `start_stick_id + i * TILE_HEIGHT` (i.e., `row_start_id + i * 32`)
- The reader iterates `j` from 0 to 31, resolving NoC addresses for sticks `stick_id` through `stick_id + 31`
- These 32 sticks form one complete tile-row

### Horizontal Block Iteration

The inner loop `for (j = 0; j < num_full_blocks_in_row; j++)` handles the case where the row width is wider than what can be read in one DMA burst. In the interleaved program factory, `num_full_blocks_in_row` is always set to 1 and `block_width_size` equals the full row width. This means the entire row width is read in a single pass per stick.

### Row Start ID Tracking Across Cores

Each core receives a `row_start_id` runtime arg that is the global stick index where this core's work begins:
- Core 0: `row_start_id = 0`
- Core i: `row_start_id = TILE_HEIGHT * nblocks_per_core * i`
- Cliff core: `row_start_id = TILE_HEIGHT * nblocks_per_core * ncores_full`

## Memory Access Patterns

### Read Pattern (Focus Area)

The reader performs a **strided gather** of 32 sticks per block:

1. For each block (tile-row), resolve 32 consecutive stick addresses upfront into `base_src_noc_addr[32]`.
2. Issue 32 `noc_async_read` calls, each reading `block_width_size` bytes (one full stick).
3. Sticks are written to consecutive L1 addresses: `l1_write_addr` increments by `width_size` after each stick.
4. All 32 reads are batched before a single `noc_async_read_barrier()`.

**Important**: Each stick may reside in a different DRAM bank (due to interleaved distribution). The 32 reads within one block may target up to 32 different banks. However, since sticks are allocated round-robin, consecutive sticks often target consecutive banks, providing good bandwidth distribution.

**L1 layout after read**: 32 sticks laid out contiguously:
```
[stick_0: W bytes][stick_1: W bytes]...[stick_31: W bytes]
```
This is exactly the RM representation of a 32xW sub-matrix, which the tilize hardware instruction can process.

### Write Pattern (De-emphasized)

The writer reads one tile at a time from CB c_16 and writes it to the output buffer using `noc_async_write_page`. Sequential tile IDs from `tile_start_id`.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of available cores) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `ncores` (computed by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (tile-rows) per full core; `nblocks_per_core_cliff` for cliff core |
| **Load balancing** | Nearly equal: all cores get `nblocks_per_core` except last core gets remainder |

### Work Splitting Algorithm (`split_blocks_for_tilize`)

```
nblocks_per_core = ceil(nblocks / grid_area)
ncores = ceil(nblocks / nblocks_per_core)
nblocks_per_core_cliff = nblocks % nblocks_per_core  (0 means no cliff)
```

- If `nblocks` divides evenly: all cores get equal work, no cliff core.
- Otherwise: `ncores - 1` cores get `nblocks_per_core` blocks, and 1 cliff core gets `nblocks_per_core_cliff` blocks.
- Cores are enumerated linearly from the available grid (`corerange_to_cores`).

### Per-Core Work Assignment

Each core processes a contiguous range of tile-rows (blocks). The runtime args encode:
- `num_sticks`: `nblocks_per_core * TILE_HEIGHT` (total sticks this core reads)
- `row_start_id`: global stick index where this core starts
- `tile_start_id`: global tile index where this core's output starts (for the writer)

## Arguments

### Reader Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one RM stick in bytes (`padded_shape[-1] * element_size()`) |
| 1+ | TensorAccessorArgs | (variable) | Interleaved buffer metadata (bank mapping, etc.) appended by `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` |

### Reader Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total sticks to read (`nblocks_per_core * 32`) |
| 2 | block_size_nbytes | uint32_t | Full row width in bytes (same as stick_size compile-time arg; redundant but kept for interface compat) |
| 3 | ntiles_per_block | uint32_t | Tiles per tile-row (= `padded_shape[-1] / 32`) |
| 4 | block_width_size | uint32_t | Bytes to read per stick per horizontal pass (= full row width here) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for this factory (single horizontal pass) |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial horizontal blocks) |
| 7 | leftover_width_in_row | uint32_t | Always 0 |
| 8 | start_stick_id | uint32_t | Global stick index where this core's work begins |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) for this core |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block (`ntiles_per_block`) |

### Writer Compile-Time Arguments (De-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output CB index (c_16) |
| 1+ | TensorAccessorArgs | (variable) | Output buffer metadata |

### Writer Runtime Arguments (De-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_tiles | uint32_t | Total tiles to write |
| 2 | start_id | uint32_t | Global tile index to start writing from |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per block via noc_async_read |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  1. Pre-computes 32 NoC addresses into `base_src_noc_addr[32]` array before issuing any reads.
  2. Uses a lambda `read_tiles` that: reserves `num_tiles` pages in CB, reads 32 sticks each of `width_size` bytes into consecutive L1 positions, barriers, then pushes.
  3. The `base_src_noc_addr[k] += width_size` increment within `read_tiles` supports the horizontal-block pattern (multiple passes per row), but in this factory it is always a single pass.
  4. Outer loop: `num_sticks / tile_height` iterations = number of blocks assigned to this core.
  5. No CB pop -- that is done by the compute kernel consumer side.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize.cpp | Compute threads | N/A | CB c_0 | CB c_16 | tilize_block / fast_tilize_block |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  1. Calls `compute_kernel_hw_startup(c_0, c_16)` to configure unpack/pack srcA=srcB=c_0.
  2. Invokes `compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(per_core_block_tile_cnt, per_core_block_cnt)`.
  3. The tilize helper's main loop: for each block, waits for `ntiles_per_block` pages in c_0, reserves same count in c_16, calls `tilize_block`/`fast_tilize_block`, pushes to c_16, pops from c_0.
  4. Auto-selects `fast_tilize` when hardware supports it (32x32 tiles, Float32/Float16_b format, half-sync dest mode).

### Writer Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_16 | DRAM (tiles) | Write one tile at a time |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

## Implementation Notes

### Stick-to-Tile Batching Pattern (Critical for layer_norm_rm)

The fundamental pattern this analysis documents is:

1. **CB is sized in tile-page units** (`ntiles_per_block` pages of `input_single_tile_size`).
2. **Reader writes raw RM data** (32 sticks) that occupies exactly `ntiles_per_block` tile-pages worth of bytes.
3. **Compute interprets the same memory** as RM input and produces tiled output via hardware tilize instructions.

This works because `32 sticks * W bytes/stick = (W/32) tiles * tile_size bytes/tile` -- the byte count is identical regardless of interpretation.

For a `layer_norm_rm` operation that needs to:
- Read RM sticks from DRAM
- Tilize in the compute kernel for math operations
- Then untilize and write back to RM

The reader pattern from this tilize operation provides the canonical template: batch 32 sticks into a tile-page-sized CB, let compute tilize, perform math on tiles, untilize, and write back as RM.

### TensorAccessor for Interleaved RM Reads

The `TensorAccessor` is constructed with:
- Compile-time args: buffer metadata (bank layout, interleaving info) appended via `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)`
- Runtime: `src_addr` (buffer base address) and `stick_size` (page size = row width in bytes)

The kernel creates the accessor as:
```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // CT args start at index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
```

Then `get_noc_addr(stick_id, s)` returns the full 64-bit NoC address for that stick's DRAM location.

### Compute Kernel FP32 Accumulation

When input dtype is FLOAT32, the flag `fp32_dest_acc_en = true` is set on the compute kernel config. This enables 32-bit accumulation in the destination register, which is important for numerical precision.

### Program Caching

The `override_runtime_arguments` method enables program caching: when tensors change address but not shape, only the buffer addresses (runtime arg index 0 for both reader and writer) are updated, avoiding full program recompilation.

## External Knowledge Sources

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how `TensorAccessorArgs` and `TensorAccessor` work on both host and device sides.
   **Key Information**: `TensorAccessorArgs(buffer)` extracts bank mapping info and can produce compile-time and/or runtime arguments. On device, `TensorAccessor(args, base_addr, page_size)` maps `page_id -> noc_addr` by computing `(bank_id, bank_offset)` from the interleaved distribution.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM page definition and interleaved bank distribution.
   **Key Information**: In RM layout, one page = one row (stick). In interleaved memory layout, pages are distributed round-robin across banks. In tiled layout, one page = one 32x32 tile stored as 4 faces (16x16 each) in memory.

3. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` work distribution algorithm.
   **Key Information**: Computes `nblocks_per_core = ceil(nblocks / grid_area)`, then `ncores = ceil(nblocks / nblocks_per_core)`, with cliff handling for the remainder.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel's tilize function template and its modes.
   **Key Information**: The tilize helper supports symmetric (tile-page input CB) and asymmetric (row-page input CB) modes. In this factory, symmetric mode is used (no `total_input_pages` provided). The helper auto-selects `fast_tilize` when tile size is 32x32 and data format is Float32 or Float16_b.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper function.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a CircularBuffer with total size = `num_pages * page_size` and sets the page size for the given CB index.

6. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding the reader/compute/writer kernel model and circular buffer synchronization.
   **Key Information**: Reader writes data into CBs and signals availability. Compute waits on input CBs, processes, writes to output CBs. Writer waits on output CBs and writes to DRAM. Synchronization is via `cb_reserve_back`/`cb_push_back` (producer) and `cb_wait_front`/`cb_pop_front` (consumer).
