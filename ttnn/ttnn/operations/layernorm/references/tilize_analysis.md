# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The tilize operation converts row-major (RM) tensor data into tiled (32x32) format. It reads contiguous rows ("sticks") from DRAM, packs groups of 32 consecutive sticks into the input circular buffer, and lets the compute kernel rearrange the data into tile layout. The output is written back to DRAM in interleaved tile format.

**Program factory**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Role focus**: input_stage -- reader kernel pattern, input CB sizing, stick-to-tile batching, work distribution.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a horizontal strip of tiles spanning one tile-row of the tensor) |
| **Unit size** | `ntiles_per_block` tiles = `padded_shape[-1] / 32` tiles wide, 32 rows tall |
| **Total units** | `nblocks = physical_volume / TILE_HW / ntiles_per_block` = total rows / 32 |
| **Loop structure** | Outer loop over tile-rows (groups of 32 sticks), inner loop over horizontal blocks within the row |

A "block" is one full tile-height strip across the entire width of the tensor. Since the inner loop `num_full_blocks_in_row` is always 1 in this interleaved factory (the entire row width is one block), each work unit is effectively one tile-row covering the full tensor width.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary N-D (flattened to 2D internally) |
| **Dimension convention** | Last dim = width (stick length), remaining dims = height |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16, FLOAT32 (any supported type) |

- **Page (stick) size**: `padded_shape[-1] * element_size` bytes -- one full row is one page.
- **Stick**: A single row of the tensor in row-major format. Each stick has `padded_shape[-1]` elements.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input (padded to tile alignment) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input or specified output dtype |

### Layout Transformation

Row-major sticks are read 32 at a time and packed into CB c_0 as a contiguous block of `ntiles_per_block * tile_size` bytes. The compute kernel then rearranges this data into tiled format (face-ordered 32x32 tiles) in CB c_16. No explicit tilize/untilize in the reader -- the reader just copies raw stick bytes.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved RM sticks) | CB c_0 | `cb_reserve_back(c_0, ntiles_per_block)`, `noc_async_read` x 32 sticks, `noc_async_read_barrier`, `cb_push_back(c_0, ntiles_per_block)` |
| 2 | Compute | CB c_0 | CB c_16 | `cb_wait_front(c_0, ntiles_per_block)`, `tilize_block/fast_tilize_block`, `cb_pop_front(c_0)`, `cb_push_back(c_16)` |
| 3 | Writer | CB c_16 | DRAM (interleaved tiles) | `cb_wait_front(c_16)`, `noc_async_write`, `cb_pop_front(c_16)` |

### Reader Kernel Pattern (Primary Focus)

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

The reader kernel processes sticks in groups of 32 (one tile height). For each group of 32 sticks:

1. **Address Resolution**: Resolves NoC addresses for all 32 sticks upfront using `get_noc_addr(stick_id, s)` via TensorAccessor. This translates each logical stick ID to a physical DRAM bank address + offset. The addresses are cached in a local array `base_src_noc_addr[32]`.

2. **Block Read**: Calls `read_tiles()` lambda which:
   - Reserves `ntiles_per_block` pages in CB c_0
   - Iterates over all 32 sticks, issuing `noc_async_read` for each stick's full width (`width_size = block_width_size`)
   - Each read copies one stick row into consecutive L1 positions within the CB
   - After all 32 reads are issued, calls `noc_async_read_barrier()` to wait for completion
   - Pushes `ntiles_per_block` pages to CB c_0

3. **Addressing Advancement**: After reading each stick, the `base_src_noc_addr[k]` pointer advances by `width_size`. This supports the general case where multiple horizontal blocks exist per row, but in this interleaved factory `num_full_blocks_in_row = 1`, so each group of 32 sticks is read in a single read_tiles call covering the full width.

**Key insight for layernorm**: The reader pattern of "resolve 32 stick addresses, then read them all into a CB block" is a reusable pattern for any operation that processes row-major input data in tile-height groups. For layernorm, the same approach would work -- read 32 sticks at a time to fill one tile-row's worth of data.

### Stick-to-Tile Batching

- **32 sticks = 1 tile height**: Every `cb_reserve_back` / `cb_push_back` cycle processes exactly 32 sticks.
- **CB push granularity**: `ntiles_per_block` tiles per push (= number of tiles spanning the width).
- **Physical layout in CB**: The 32 sticks are laid out contiguously in L1, row after row. Each stick occupies `block_width_size` bytes. Total per block = `32 * block_width_size` = `32 * padded_shape[-1] * element_size` bytes = `ntiles_per_block * tile_size` bytes (since tile_size = 32 * 32 * element_size for standard tiles).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input staging (RM sticks) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_output | Output staging (tiled data) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Compute | Writer | Block |

**Input CB (c_0) details**:
- Capacity = `ntiles_per_block * input_single_tile_size` bytes
- This holds exactly one tile-row's worth of row-major data (32 sticks x full width)
- Page size = `input_single_tile_size` (tile-sized pages, even though the data is row-major)
- Single-buffered: capacity equals block size, so reader must wait for compute to consume before writing next block

**Note**: Both CBs are single-buffered (`capacity == block_size`). This means there is no overlap between reader and compute, or between compute and writer, within the pipeline. Each stage must complete before the next can use the CB.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of cores from available grid) |
| **Grid dimensions** | Up to full device `compute_with_storage_grid_size` |
| **Total cores** | `ncores` = ceil(`nblocks` / `nblocks_per_core`) |
| **Work per core** | `nblocks_per_core` blocks (tile-rows), cliff core gets remainder |
| **Load balancing** | Equal distribution with optional cliff core for remainder |

### Work Splitting Algorithm (`split_blocks_for_tilize`)

1. Compute `nblocks_per_core = ceil(nblocks / grid_area)` -- tries to spread blocks evenly.
2. Compute `ncores = ceil(nblocks / nblocks_per_core)`.
3. If `nblocks % nblocks_per_core != 0`, the last core is a "cliff core" processing fewer blocks (`nblocks_per_core_cliff`).
4. Cores are enumerated linearly from the available `CoreRangeSet` using `corerange_to_cores()`.

### Per-Core Work Assignment

For full cores:
- Process `nblocks_per_core` blocks, starting at `row_start_id` (stick index).
- Total sticks = `nblocks_per_core * TILE_HEIGHT` (= `nblocks_per_core * 32`).
- Total tiles = `nblocks_per_core * ntiles_per_block`.

For cliff core (last core, if remainder exists):
- Process `nblocks_per_core_cliff` blocks with proportionally fewer sticks and tiles.

Row start IDs and tile start IDs advance linearly across cores, establishing contiguous ownership of tensor regions.

## Arguments

### Compile-Time Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size (block_size_nbytes) | uint32_t | Size of one stick in bytes: `padded_shape[-1] * element_size` |
| 1+ | TensorAccessorArgs | variable | Accessor metadata for interleaved DRAM buffer (bank mapping, shapes) |

### Runtime Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total sticks for this core: `nblocks_per_core * 32` |
| 2 | block_size_nbytes | uint32_t | Bytes per stick (same as compile-time arg 0) |
| 3 | ntiles_per_block | uint32_t | Tiles per tile-row (width / 32) |
| 4 | block_width_size | uint32_t | Same as block_size_nbytes (full row width in bytes) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for this factory |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial width blocks) |
| 7 | leftover_width_in_row | uint32_t | Always 0 |
| 8 | start_stick_id | uint32_t | First stick index for this core (row_start_id) |

### Compile-Time Arguments (Compute)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt (nblocks_per_core) | uint32_t | Number of blocks (tile-rows) per core |
| 1 | per_core_block_tile_cnt (ntiles_per_block) | uint32_t | Tiles per block (width tiles) |

Note: Cliff cores get a separate compute kernel instance with `nblocks_per_core_cliff` instead.

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (interleaved sticks) | CB c_0 | Read 32 RM sticks per block via NoC |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Uses `TensorAccessor` for address resolution: `get_noc_addr(stick_id, s)` maps logical stick ID to physical DRAM bank + offset.
  - Pre-resolves all 32 addresses in a local array before issuing any reads. This is efficient because it batches address lookups.
  - Issues 32 `noc_async_read` calls per block (one per stick), then one `noc_async_read_barrier`.
  - The CB reservation/push cycle operates at tile granularity (`ntiles_per_block` tiles) even though the data written is raw row-major bytes.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 | CB c_16 | tilize_block / fast_tilize_block |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (dispatches to `compute_kernel_lib::tilize` in `tilize_helpers.hpp/.inl`)
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` to configure unpack/pack registers.
  - Uses symmetric mode (no `total_input_pages` argument) -- both input and output CBs have tile-sized pages.
  - Automatically selects `fast_tilize_block` (hardware-accelerated) when conditions are met: 32x32 tiles, Float32/Float16_b format, half-sync dest mode. Otherwise falls back to `tilize_block`.
  - Processes `per_core_block_cnt` blocks, each producing `per_core_block_tile_cnt` output tiles.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM (interleaved tiles) | Write tiles via NoC |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- De-emphasized per role directive. This is a generic interleaved tile writer with TensorAccessor.

## Index Calculations

The reader maps logical stick IDs to physical DRAM addresses via `TensorAccessor`:

1. **Stick ID**: Linear index into the row-major tensor. Stick 0 is the first row, stick 1 is the second row, etc.
2. **TensorAccessor** resolves `stick_id` to a `(bank_id, offset)` pair using the interleaved distribution: `bank_id = stick_id % num_banks`, `offset = (stick_id / num_banks) * page_size`. The `get_noc_addr()` function then converts this to a physical NoC address.
3. **Per-core offset**: Each core starts at `row_start_id = sum of previous cores' sticks`. Full cores own `nblocks_per_core * 32` contiguous sticks.

## Memory Access Patterns

### Read Pattern
- **Ordering**: Sequential by stick ID within each core's assigned region.
- **Granularity**: One full-width stick per `noc_async_read` call.
- **Batching**: 32 reads are issued asynchronously before a single barrier.
- **Bank distribution**: Sticks are interleaved across DRAM banks in round-robin fashion. Consecutive sticks go to different banks, enabling bank-level parallelism.
- **Stride**: No stride within a stick (contiguous read). Between sticks, the physical addresses may jump between banks.

### Write Pattern
- De-emphasized per role directive. Sequential tile writes with interleaved distribution.

## Implementation Notes

### Relevance to Layernorm

For a layernorm operation normalizing rows of a 2D tensor:

1. **Reader pattern reuse**: The "read 32 sticks, push to CB" pattern directly applies. For layernorm on row-major input, the reader would read 32 rows at a time (one tile-height), giving the compute kernel a full tile-row to compute mean/variance on.

2. **CB sizing consideration**: The input CB holds `ntiles_per_block` tiles = one full tile-row. For layernorm, you may want the same sizing if computing statistics across the full row width, or potentially larger if double-buffering is desired for read-compute overlap.

3. **Single-buffered limitation**: This tilize factory uses single-buffered CBs (capacity = block size). For a compute-heavy operation like layernorm, double-buffering the input CB (capacity = 2 * block_size) would allow the reader to prefetch the next tile-row while compute processes the current one.

4. **Work distribution reuse**: The `split_blocks_for_tilize` utility provides a clean 1D block distribution with cliff handling. This same pattern works for any operation that splits work by tile-rows. The function lives in `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`.

5. **TensorAccessor for interleaved access**: The `TensorAccessorArgs` pattern (compile-time args on host, `TensorAccessor` construction on device) is the standard way to read interleaved tensors. For layernorm, this same pattern would be used for reading the input tensor and potentially gamma/beta parameter tensors.

6. **Block = tile-row**: The fundamental work unit is a group of 32 consecutive sticks covering the full tensor width. This maps naturally to layernorm where each row's statistics are independent.

### Edge Cases

- The `num_full_blocks_in_row` is always 1 and `num_leftover_tiles` is always 0 in this factory, meaning the entire row width is treated as a single block. The reader kernel supports splitting rows into multiple horizontal blocks, but this factory does not use that capability.
- FP32 accumulation is enabled (`fp32_dest_acc_en`) when input dtype is FLOAT32.
- The `sub_core_grids` parameter allows restricting which cores are used, falling back to the full device grid if not specified.

## External Knowledge Sources

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor maps stick IDs to DRAM bank addresses.
   **Key Information**: TensorAccessor handles interleaved bank mapping automatically. Host-side `TensorAccessorArgs` packs buffer metadata into compile-time args. Device-side `TensorAccessor(args, addr, page_size)` reconstructs the accessor. `get_noc_addr(page_id, accessor)` returns the physical NoC address.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major vs tiled layouts and interleaved memory distribution.
   **Key Information**: In row-major layout, each tensor row is one page. In interleaved memory, pages are distributed round-robin across DRAM banks. Tiles are 32x32 with 16x16 faces in row-major face order.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel's tilize implementation.
   **Key Information**: `compute_kernel_lib::tilize` automatically selects fast_tilize when hardware supports it. In symmetric mode (default), both CBs use tile-sized pages. The function handles init/uninit lifecycle, WaitBlock synchronization (per-block cb_wait_front), and supports both fast and standard tilize paths.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the block-to-core distribution algorithm.
   **Key Information**: `split_blocks_for_tilize(grid, nblocks)` computes `nblocks_per_core = ceil(nblocks / grid_area)`, then `ncores = ceil(nblocks / nblocks_per_core)`. The last core may be a cliff core with `nblocks % nblocks_per_core` blocks.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper utility.
   **Key Information**: `create_cb(cb_index, program, cores, page_size, num_pages, data_format)` creates a circular buffer with total size = `num_pages * page_size`. Both input and output CBs in this factory use `ntiles_per_block` pages of tile size.
