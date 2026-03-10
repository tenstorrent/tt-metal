# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The **tilize** operation converts row-major (RM) interleaved tensor data into tiled (32x32) format, distributing work across multiple Tensix cores. This analysis focuses on the multi-core interleaved variant, which reads RM sticks from DRAM, converts them to tiles on the compute unit, and writes tiled output back to DRAM.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Intended audience**: Downstream agents building the `layer_norm_rm` operation, which needs an input-stage tilize pattern (RM input from DRAM -> tilize in compute -> further processing).

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (1 block = 1 full row of tiles = `ntiles_per_block` tiles) |
| **Unit size** | `ntiles_per_block` tiles, where `ntiles_per_block = padded_shape[-1] / TILE_WIDTH` |
| **Total units** | `nblocks = total_tiles / ntiles_per_block` (i.e., total tile-rows across all batches/height) |
| **Loop structure** | Reader: outer loop over groups of TILE_HEIGHT (32) sticks, inner loop reads full row width. Compute: outer loop over blocks, each block tilizes `ntiles_per_block` tiles. |

A **block** represents one tile-row: 32 consecutive RM sticks spanning the full padded width of the tensor. This is the minimum unit that produces a complete row of tiles after tilize. Each block produces `ntiles_per_block` output tiles.

---

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | `[N, C, H, W]` (arbitrary rank, flattened to 2D for memory) |
| **Dimension convention** | Last dim = W (width, contiguous in memory as a stick) |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1, interleaved) |
| **Data type** | BFLOAT16 or FLOAT32 |

In row-major interleaved storage, each **stick** (one row of width W) is one page. Pages are distributed round-robin across DRAM banks. The stick size in bytes is `padded_shape[-1] * element_size()`.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input (padded to tile multiples) |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1, interleaved) |
| **Data type** | Same as input (or specified by `output_dtype`) |

### Layout Transformations

The core transformation is ROW_MAJOR -> TILE_LAYOUT. The input tensor's padded shape must be aligned to tile dimensions (height divisible by 32, width divisible by 32). The compute kernel reorders 32 contiguous RM sticks into `ntiles_per_block` tiles of 32x32 elements each.

---

## Data Flow Pattern

### Step-by-Step Flow

1. **Reader kernel** fetches 32 consecutive RM sticks from DRAM into CB c_0 (input CB).
   - For each group of 32 sticks, it resolves NoC addresses for all 32 sticks first (batch address resolution).
   - Then reads the full width of all 32 sticks into the CB in one `cb_reserve_back` / `cb_push_back` cycle covering `ntiles_per_block` tile slots.
   - The data written to CB c_0 is in RM format: 32 sticks laid out sequentially, each stick being `block_width_size` bytes.

2. **Compute kernel** waits for `ntiles_per_block` pages in CB c_0, then calls `tilize_block` to reorder the 32 rows of RM data into 32x32 tiles and writes results to CB c_16 (output CB).
   - Uses `compute_kernel_lib::tilize<c_0, c_16, ...>` helper with `WaitMode::WaitBlock`.
   - Processes `per_core_block_cnt` blocks, each block being `per_core_block_tile_cnt` tiles wide.

3. **Writer kernel** waits for 1 tile at a time in CB c_16, writes each tile to DRAM using `noc_async_write_page`.
   - Uses TensorAccessor for interleaved tile output.
   - Sequential tile-by-tile output starting from `start_id`.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Total Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input RM sticks staged as tile-sized pages | `ntiles_per_block` | `input_single_tile_size` | `ntiles_per_block * input_single_tile_size` | Single | Reader | Compute | Block |
| c_16 | cb_output | Output tiled data | `ntiles_per_block` | `output_single_tile_size` | `ntiles_per_block * output_single_tile_size` | Single | Compute | Writer | Block |

### Critical Detail: CB c_0 Page Size vs Actual Data Written

The CB c_0 is configured with `page_size = input_single_tile_size` (tile-sized pages, e.g., 2048 bytes for bfloat16) and `num_pages = ntiles_per_block`. However, the reader kernel writes **RM stick data** into this CB. The reader writes 32 sticks of `block_width_size` bytes each. The total data written per block is `32 * block_width_size = 32 * padded_shape[-1] * element_size`. This is numerically equal to `ntiles_per_block * tile_size` because:
- `block_width_size = padded_shape[-1] * element_size`
- `ntiles_per_block = padded_shape[-1] / 32`
- `32 * padded_shape[-1] * element_size = (padded_shape[-1] / 32) * (32 * 32 * element_size) = ntiles_per_block * tile_size`

So the total byte capacity matches, even though the data layout in the CB is row-major (not yet tiled). The compute kernel's tilize operation reinterprets and reorders this data into tiles.

### Pipeline Pattern Summary

Both CBs have capacity equal to exactly one block's worth of tiles (`ntiles_per_block`). This means:
- **c_0**: Single-buffered. Reader fills one full block, compute consumes it, then reader can fill the next.
- **c_16**: Single-buffered. Compute fills one full block, writer drains it tile by tile.

There is no double-buffering overlap between consecutive blocks. The reader and compute alternate on a per-block basis for CB c_0.

---

## Index Calculations

### Reader: Stick ID to NoC Address Mapping

The reader uses `TensorAccessor` to map logical stick IDs to physical NoC addresses:

```
stick_id = start_stick_id    // assigned per core, based on which blocks this core handles
for each block (group of 32 sticks):
    for j in 0..31:
        base_src_noc_addr[j] = get_noc_addr(stick_id, s)   // resolve DRAM bank + offset
        stick_id++
    // read full width for all 32 sticks
```

`get_noc_addr(stick_id, s)` uses the TensorAccessor `s` (configured with the buffer's interleaved layout metadata) to compute: `bank_id = stick_id % num_banks`, `offset = (stick_id / num_banks) * stick_size`, then combines with the bank's NoC coordinates to produce a 64-bit NoC address.

### Reader: Width Traversal

For each group of 32 sticks, the reader iterates over `num_full_blocks_in_row` (which is always 1 in this factory). Each iteration reads `block_width_size` bytes (the full stick width) for all 32 sticks. The addresses are advanced by `width_size` after each read to support potential multi-block-per-row scenarios (not used in this interleaved variant).

### Host: Stick Start ID Calculation

```
row_start_id += TILE_HEIGHT * nblocks_per_core   // each block = 32 sticks
tile_start_id += ntiles_per_block * nblocks_per_core
```

Each core starts reading from `row_start_id` (which stick), and writing from `tile_start_id` (which tile).

---

## Memory Access Patterns

### Read Pattern (Reader Kernel)

- **Pattern**: Batched strided reads with grouped address resolution.
- **Granularity**: Per-stick reads (`noc_async_read` of `width_size` bytes per stick).
- **Ordering**: For each block, all 32 stick addresses are pre-resolved, then reads are issued sequentially for the 32 sticks. A `noc_async_read_barrier()` follows the 32 reads.
- **Stride**: Sticks are logically contiguous (stick_id increments by 1), but physically they may be in different DRAM banks (interleaved round-robin). Each NoC read targets a different bank location.
- **Burst size**: Each individual read is `block_width_size` bytes (full stick width).

### Write Pattern (Writer Kernel) (De-emphasized)

- Sequential tile-by-tile writes using `noc_async_write_page`. One tile per CB wait/pop cycle.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (flattened from 2D grid) |
| **Grid dimensions** | Uses full `compute_with_storage_grid_size()` unless `sub_core_grids` overrides |
| **Total cores** | `ncores` (computed by `split_blocks_for_tilize`) |
| **Work per core (full)** | `nblocks_per_core` blocks (each block = `ntiles_per_block` tiles = 32 sticks) |
| **Work per core (cliff)** | `nblocks_per_core_cliff` blocks (remainder) |
| **Load balancing** | Near-equal: `ceil(nblocks / grid_area)` blocks per core, last core gets remainder |

### Work Split Details (`split_blocks_for_tilize`)

The function `ttnn::split_blocks_for_tilize(available_grid, nblocks)` performs 1D distribution:

1. `grid_area = available_grid.num_cores()`
2. `nblocks_per_core = ceil(nblocks / grid_area)`
3. `ncores = ceil(nblocks / nblocks_per_core)` (may be less than grid_area if work is small)
4. `nblocks_per_core_cliff = nblocks % nblocks_per_core` (0 if evenly divisible)
5. If cliff > 0: the last core gets `nblocks_per_core_cliff` blocks, all others get `nblocks_per_core`.

Cores are enumerated from the `CoreRangeSet` in row-major order. The function returns:
- `core_range`: CoreRangeSet of full-work cores
- `core_range_cliff`: CoreRangeSet of the single cliff core (if any)
- `all_cores`: Union of both

### Compile-Time vs Runtime Differentiation for Cliff

The compute kernel uses **separate compile-time args** for full cores vs cliff core:
- Full cores: `{nblocks_per_core, ntiles_per_block}`
- Cliff core: `{nblocks_per_core_cliff, ntiles_per_block}`

This means two separate `CreateKernel` calls with different compile args, but the same kernel source.

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size (block_size_nbytes) | uint32_t | Size of one RM stick in bytes: `padded_shape[-1] * element_size()` |
| 1+ | TensorAccessorArgs | multiple uint32_t | Interleaved buffer metadata (rank, num_banks, etc.) appended by `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks this core processes (nblocks_per_core or nblocks_per_core_cliff) |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block = `ntiles_per_block` = `padded_shape[-1] / 32` |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out (output_cb_index) | uint32_t | Output CB index (c_16) |
| 1+ | TensorAccessorArgs | multiple uint32_t | Output buffer metadata appended by `TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args)` |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total sticks to read = `nblocks_per_core * TILE_HEIGHT` |
| 2 | block_width_size (unused directly but passed) | uint32_t | `block_size_nbytes` (full stick width in bytes) |
| 3 | num_tiles_per_block | uint32_t | `ntiles_per_block` = tiles per row |
| 4 | block_width_size | uint32_t | `block_size_nbytes` (used as `width_size` in read_tiles lambda) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 (full row read in single pass) |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial tile blocks) |
| 7 | leftover_width_in_row | uint32_t | Always 0 (no partial width) |
| 8 | start_stick_id (row_start_id) | uint32_t | First stick ID for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_pages | uint32_t | Total tiles to write = `ntiles_per_block * nblocks_per_core` |
| 2 | start_id (tile_start_id) | uint32_t | First tile ID for this core |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | BRISC (RISCV_0) | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per block, resolve NoC addrs, noc_async_read |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Uses a `base_src_noc_addr[32]` array to pre-resolve all 32 stick addresses before issuing reads.
  - The `read_tiles` lambda performs: `cb_reserve_back(c_0, num_tiles)` -> 32 sequential `noc_async_read` calls (one per stick, each reading `width_size` bytes) -> `noc_async_read_barrier()` -> `cb_push_back(c_0, num_tiles)`.
  - Each read fills one stick's worth of data at the current `l1_write_addr`, advancing by `width_size` per stick.
  - After all 32 sticks are read, the CB holds `ntiles_per_block` tile-sized pages of RM data (not yet tiled).
  - The `num_full_blocks_in_row` is always 1 in this factory, meaning the full tensor width is read in a single pass per block.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | Compute (TRISC) | N/A | CB c_0 | CB c_16 | tilize_block (RM -> tile format conversion) |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` for hardware initialization.
  - Uses `compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(per_core_block_tile_cnt, per_core_block_cnt)`.
  - `WaitBlock` mode: for each block, the helper calls `cb_wait_front(c_0, ntiles_per_block)` then `tilize_block` which unpacks RM data and packs into tiled format.
  - After processing each block: `cb_push_back(c_16, ntiles_per_block)` and `cb_pop_front(c_0, ntiles_per_block)`.
  - `fp32_dest_acc_en` is set when input dtype is FLOAT32.

### Writer Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | NCRISC (RISCV_1) | NOC1 | CB c_16 | DRAM (tiles) | Write tiles sequentially |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Standard single-tile-at-a-time writer. `cb_wait_front(c_16, 1)` -> `noc_async_write_page(i, s, l1_read_addr)` -> `cb_pop_front(c_16, 1)`.

---

## Implementation Notes

### Stick-to-Tile Batching Pattern (Key for layer_norm_rm Reuse)

The central pattern for reading RM data into a tilize-compatible CB is:

1. **CB is sized to hold one tile-row** (`ntiles_per_block` pages of tile size).
2. **Reader writes 32 RM sticks** into the CB. The 32 sticks * full_width bytes = ntiles_per_block * tile_bytes. This is an invariant because `32 * W * elem_size = (W/32) * (32*32*elem_size)`.
3. **Compute interprets** the CB contents as 32 rows of RM data spanning `ntiles_per_block` tile columns, and reorders into tiles.

This pattern means the CB page_size is tile_size, but the reader treats the entire CB space as a linear buffer for RM sticks. The `get_write_ptr(cb_id_in0)` returns the start of the reserved region, and the reader fills it stick-by-stick.

### Address Pre-Resolution Optimization

The reader resolves all 32 NoC addresses before issuing any reads. This avoids interleaving address computation with NoC read latency, allowing the NoC reads to be issued back-to-back for better throughput.

### num_full_blocks_in_row = 1

In this interleaved variant, the full tensor width is always read as a single block. The `num_full_blocks_in_row` and `num_leftover_tiles` mechanism exists in the reader kernel to support partial-width reading (used in other tilize variants), but for the interleaved factory these are hardcoded to `{1, 0, 0}`.

### Cliff Core Handling

The cliff core receives fewer blocks but processes them identically. It gets separate compile-time args for the compute kernel (different `per_core_block_cnt`), and different runtime args for reader/writer (different `num_sticks` / `num_pages`). The compute kernel is compiled twice: once for the full core range, once for the cliff core.

### TensorAccessor Usage

Both reader and writer use `TensorAccessor` for DRAM address resolution. The `TensorAccessorArgs` are appended as compile-time arguments to each kernel, carrying the interleaved buffer metadata (number of banks, bank offsets, etc.). At runtime, only the buffer base address is passed (and can be updated via `override_runtime_arguments` for tensor reuse).

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the TensorAccessor and TensorAccessorArgs work in tt-metal kernels for reading interleaved tensors from DRAM?"
   **Reason**: Needed to understand how `get_noc_addr(stick_id, s)` resolves physical DRAM addresses for interleaved RM sticks.
   **Key Findings**: TensorAccessor abstracts interleaved memory layout. `get_noc_addr` maps logical page/stick ID to bank_id + offset, then combines with NoC coordinates to produce 64-bit address. TensorAccessorArgs configures which metadata is compile-time vs runtime.

2. **Query**: "How does cb_reserve_back and cb_push_back work in the reader kernel?"
   **Reason**: Needed to understand the CB producer protocol and whether `num_pages` in `cb_reserve_back` corresponds to tile-sized pages even when writing RM data.
   **Key Findings**: `cb_reserve_back(cb_id, num_pages)` blocks until `num_pages` free slots exist in the CB. Page size is determined by the CB configuration (`fifo_page_size`). `cb_push_back` signals pages are ready for consumer. The total bytes reserved = `num_pages * page_size`.

3. **Query**: "What is the tilize operation in tt-metal? How does the compute kernel convert row-major sticks into tiles?"
   **Reason**: Needed to understand what `tilize_block` does internally and how it synchronizes with the CB.
   **Key Findings**: `tilize_block` unpacks RM data from input CB, reorders into 32x32 tiles (with 16x16 face structure), packs into output CB. `fast_tilize_block` is used when hardware supports it (32x32 tiles, Float32/Float16_b, half-sync dest). The WaitBlock mode handles per-block CB synchronization automatically.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM page structure and interleaved bank distribution.
   **Key Information**: In RM layout, each row = one page. Interleaved pages are distributed round-robin across banks. This confirms that `stick_id` directly maps to page_id for the TensorAccessor.

2. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` work distribution function.
   **Key Information**: 1D distribution with `ceil(nblocks/grid_area)` blocks per core. Cliff core gets remainder. Returns BlockSplit struct with core ranges and per-core block counts.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding `create_cb` helper used in the program factory.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a CB with total size = `num_pages * page_size`, setting the page_size per CB index. Optionally accepts a Buffer pointer for globally allocated CBs.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the `compute_kernel_lib::tilize` template and its WaitBlock behavior.
   **Key Information**: In WaitBlock mode, each block iteration does `cb_wait_front(input_cb, block_width_tiles)` then `tilize_block`, then `cb_push_back(output_cb, block_width_tiles)` and `cb_pop_front(input_cb, block_width_tiles)`. The helper handles init/uninit lifecycle.
