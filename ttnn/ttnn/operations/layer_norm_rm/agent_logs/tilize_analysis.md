# Tilize Multi-Core Interleaved Implementation Analysis

## Overview

The tilize operation converts row-major (RM) interleaved data into tiled (32x32) format across multiple cores. This is the interleaved-memory variant of the multi-core tilize, handling tensors stored in DRAM with pages distributed round-robin across banks.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Focus**: This analysis emphasizes the **input_stage** pattern -- how RM sticks are read from DRAM, batched into tile-height groups (32 sticks), and pushed to the compute kernel through circular buffers. This serves as a reference for building a layer_norm_rm operation that needs to read RM interleaved input and tilize it before compute.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (tile-row) |
| **Unit size** | `ntiles_per_block` tiles = one row of tiles across the full tensor width |
| **Total units** | `nblocks = ceil(ntiles / ntiles_per_block)` = total tile-rows in the tensor |
| **Loop structure** | Outer: iterate tile-rows assigned to core; Inner: read 32 sticks, push one block of tiles |

A **block** in this operation is a single tile-row: a horizontal strip of `ntiles_per_block` tiles spanning the full padded width of the tensor. Each block corresponds to 32 consecutive RM sticks (TILE_HEIGHT = 32) across the full width. The total number of blocks equals the number of tile-rows in the tensor: `physical_volume / TILE_HW / ntiles_per_block`, which simplifies to `padded_height / TILE_HEIGHT` (times batch dimensions).

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary N-D, flattened to 2D for memory |
| **Dimension convention** | Inner dim = width (last dim, `-1`) |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | BFLOAT16 or FLOAT32 |
| **Page definition** | One RM stick = one full row of elements = `padded_shape[-1]` elements |
| **Page size (bytes)** | `padded_shape[-1] * element_size` (= `block_size_nbytes`) |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | Same as input dtype |
| **Page definition** | One 32x32 tile |
| **Page size (bytes)** | `tile_size(output_cb_data_format)` |

### Layout Transformation

The tilize operation converts RM sticks into tiles. The reader reads 32 consecutive sticks (each = full tensor width in bytes) into CB c_0. The compute kernel then performs the tilize transform (rearranging the 32 rows into 32x32 tile format with faces) and writes tiles into CB c_16. The writer outputs tiles to DRAM in interleaved tile order.

## Data Flow Pattern

### Step-by-Step Flow (Reader Focus)

1. **Stick address resolution**: For each group of 32 sticks, the reader resolves the NoC address of each stick using `get_noc_addr(stick_id, s)` where `s` is a `TensorAccessor`. This maps a logical stick ID to a physical DRAM bank address + bank-local offset via the interleaved distribution.

2. **32-stick batch read**: The reader pre-computes all 32 NoC addresses (`base_src_noc_addr[32]`), then issues `noc_async_read` for each stick into contiguous L1 memory within the CB. Each read transfers `block_width_size` bytes (= full row width in bytes).

3. **CB push**: After all 32 reads complete (`noc_async_read_barrier`), the reader pushes `ntiles_per_block` tiles worth of data via `cb_push_back(cb_id_in0, num_tiles_per_block)`.

4. **Compute tilize**: The compute kernel waits for the block (`cb_wait_front`), performs tilize_block (rearranging RM data into 32x32 tiled format with faces), pushes to output CB, and pops input CB.

5. **Writer output**: The writer reads tiles one at a time from the output CB and writes them to DRAM using tile-based `noc_async_write_page`.

### Data Flow Table

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (RM sticks) | CB c_0 | `cb_reserve_back(c_0, ntiles_per_block)`, `noc_async_read` x 32 sticks, `noc_async_read_barrier`, `cb_push_back(c_0, ntiles_per_block)` |
| 2 | Compute | CB c_0 | CB c_16 | `cb_wait_front(c_0, ntiles_per_block)`, `tilize_block`, `cb_push_back(c_16, ntiles_per_block)`, `cb_pop_front(c_0, ntiles_per_block)` |
| 3 | Writer | CB c_16 | DRAM (tiles) | `cb_wait_front(c_16, 1)`, `noc_async_write_page`, `cb_pop_front(c_16, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Block Size (pages) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | Input | RM stick staging for tilize | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | Output | Tiled output staging | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Compute | Writer | Block |

**Key sizing detail**: Both CBs are allocated with capacity = `ntiles_per_block` pages, and each block operation uses exactly `ntiles_per_block` pages. This means both are **single-buffered** -- the reader must wait for the compute to consume the previous block before writing the next one.

**CB c_0 page size**: Each page is one tile's worth of data in the **input data format** (RM). The CB holds `ntiles_per_block * input_single_tile_size` bytes total. However, the reader writes into this buffer as 32 contiguous rows, each `block_size_nbytes` wide. The total bytes written per block = `32 * block_size_nbytes` = `32 * padded_width * element_size`, which equals `ntiles_per_block * tile_size` when the tile is 32x32. So the CB capacity perfectly fits one full tile-row of 32 sticks.

**CB c_0 capacity formula for reuse**: `ntiles_per_block * tile_size(input_data_format)` bytes, where `ntiles_per_block = padded_shape[-1] / TILE_WIDTH`.

## Pipeline Pattern Summary

Both CBs use single-buffering (capacity = block size). This means:
- Reader and compute are serialized per block -- no overlap between reading block N+1 and computing block N.
- Compute and writer have partial overlap: the writer processes tiles one at a time from the output CB, but must wait for the full block to be produced.

For a new operation wanting better reader/compute overlap, the input CB capacity should be increased to `2 * ntiles_per_block` tiles (double-buffered).

## Memory Access Patterns

### Read Pattern (Reader Kernel -- Primary Focus)

The reader performs **strided reads** within each tile-row:

1. For each tile-row (32-stick group):
   - Resolve 32 consecutive stick IDs to NoC addresses via TensorAccessor
   - Each `get_noc_addr(stick_id, s)` maps stick_id -> bank_index (round-robin) + bank-local offset
   - Issue 32 `noc_async_read` calls, each reading `block_width_size` bytes (= full row width)
   - L1 destination is contiguous: stick 0 at write_ptr, stick 1 at write_ptr + block_width_size, etc.
   - Barrier after all 32 reads

2. **Access pattern characteristics**:
   - Sticks are sequential in logical index (stick_id increments by 1)
   - Physical DRAM addresses are NOT sequential -- they are distributed round-robin across banks
   - Each noc_async_read is a full-row transfer (potentially large: width * element_size bytes)
   - The 32 reads within a tile-row can be pipelined by the NoC hardware

3. **Stick-to-tile batching**: Exactly 32 sticks are read per CB push. This is fundamental -- 32 rows fill one tile height. After reading 32 sticks of width W, the CB contains data for `W / TILE_WIDTH` tiles (= `ntiles_per_block`), ready for the tilize transform.

### Write Pattern (Brief)

The writer processes tiles one at a time in sequential tile order, writing each to its interleaved DRAM location.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear core assignment from available grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` tile-rows (last core may get `nblocks_per_core_cliff`) |
| **Load balancing** | Near-equal with cliff: `nblocks_per_core = ceil(nblocks / grid_area)`, remainder goes to last core |

### Work Splitting Details

The function `split_blocks_for_tilize(available_grid, nblocks)` performs 1D distribution:

1. Compute `nblocks_per_core = ceil(nblocks / grid_area)` where grid_area = total available cores.
2. Compute `ncores = ceil(nblocks / nblocks_per_core)` -- only as many cores as needed.
3. If `nblocks % nblocks_per_core != 0`, the last core is a "cliff" core with fewer blocks.
4. Full cores get `nblocks_per_core` blocks; cliff core gets `nblocks % nblocks_per_core` blocks.
5. Cores are assigned linearly from the `available_grid` (row-major traversal of the core grid).

### Per-Core Address Tracking

Each core's starting position is tracked by two counters:
- `tile_start_id`: Starting output tile index (for the writer)
- `row_start_id`: Starting input stick index (for the reader)

These advance by `ntiles_per_block * nblocks_per_core` and `TILE_HEIGHT * nblocks_per_core` respectively for each core.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size (block_size_nbytes) | uint32_t | Size of one RM stick in bytes: `padded_shape[-1] * element_size` |
| 1+ | TensorAccessor args | uint32_t[] | Bank mapping metadata for interleaved source buffer (appended by `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)`) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile-rows (blocks) assigned to this core |
| 1 | per_core_block_tile_cnt | uint32_t | Number of tiles per tile-row (`ntiles_per_block = padded_width / TILE_WIDTH`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_16) |
| 1+ | TensorAccessor args | uint32_t[] | Bank mapping metadata for interleaved destination buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total sticks to read: `nblocks_per_core * TILE_HEIGHT` |
| 2 | block_size_nbytes | uint32_t | Bytes per stick (same as CT arg 0 -- redundant for flexibility) |
| 3 | num_tiles_per_block | uint32_t | Tiles per tile-row (`ntiles_per_block`) |
| 4 | block_width_size | uint32_t | Bytes per stick (same as arg 2) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 (full row read as one block) |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial blocks in interleaved case) |
| 7 | leftover_width_in_row | uint32_t | Always 0 |
| 8 | start_stick_id | uint32_t | First stick ID for this core (`row_start_id`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_pages | uint32_t | Total tiles to write: `ntiles_per_block * nblocks_per_core` |
| 2 | start_id | uint32_t | First output tile index for this core (`tile_start_id`) |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per block, push ntiles_per_block tiles |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Creates a `TensorAccessor` from compile-time args (index 1+) with `src_addr` and `stick_size`
  - Outer loop: `num_sticks / tile_height` iterations (= nblocks_per_core)
  - Each iteration: resolves 32 stick NoC addresses into `base_src_noc_addr[32]`
  - Inner loop: `num_full_blocks_in_row` iterations (always 1 for interleaved)
  - `read_tiles` lambda: reserves CB, reads 32 sticks via noc_async_read into contiguous L1, barriers, pushes CB
  - The `base_src_noc_addr[k] += width_size` advancement handles the case where a row spans multiple blocks (not used in interleaved variant where num_full_blocks_in_row = 1)

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_2 (unpack+math+pack) | N/A | CB c_0 | CB c_16 | tilize_block per block |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Uses `compute_kernel_lib::tilize<c_0, c_16, WaitBlock>` helper
  - Processes `per_core_block_cnt` blocks, each with `per_core_block_tile_cnt` tiles
  - WaitBlock mode: waits for each block of input before processing
  - Automatically selects fast_tilize when conditions are met (32x32 tiles, Float32/Float16_b, half-sync dest)
  - Calls `compute_kernel_hw_startup(c_0, c_16)` for hardware initialization

### Writer Kernel (Brief)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- Standard tile-by-tile writer using TensorAccessor for interleaved output. Processes one tile at a time with `cb_wait_front/cb_pop_front` of 1 page.

## Implementation Notes

### Key Patterns for Reuse in layer_norm_rm

1. **RM stick reading pattern**: The reader reads exactly 32 sticks per CB push. For layer_norm_rm, the same pattern applies -- read 32 rows of the input tensor to fill one tile-height of data. The stick_size compile-time arg and TensorAccessor setup are directly reusable.

2. **CB sizing for RM input**: CB c_0 is sized to hold `ntiles_per_block` tiles, which equals 32 sticks * full_width. For layer_norm_rm, the input CB should be sized identically: `ntiles_per_block * tile_size(input_format)` bytes.

3. **Work unit = tile-row**: The natural work unit for any operation on RM interleaved tensors that involves tilize is the tile-row (32 sticks across full width). This is because tilize requires exactly 32 rows to produce one row of tiles.

4. **TensorAccessor for interleaved sticks**: On the host side, `TensorAccessorArgs(*src_buffer).append_to(ct_args)` appends bank mapping as compile-time args. On device, `TensorAccessor(args, addr, page_size)` resolves stick_id -> NoC address. The page_size for RM sticks is `stick_size` (full row width in bytes).

5. **Block count and core assignment**: `nblocks = physical_volume / TILE_HW / ntiles_per_block`. Distribute with `split_blocks_for_tilize` for simple 1D distribution with cliff handling. Each core knows its `start_stick_id` and `num_sticks`.

6. **Compute kernel reuse**: The `compute_kernel_lib::tilize` helper is a clean, configurable template. For layer_norm_rm, use it with `InitOnly`/`Neither`/`UninitOnly` modes if tilize is followed by other compute operations in the same kernel.

### Redundant Runtime Args

The reader runtime args include `block_size_nbytes` (arg 2) and `block_width_size` (arg 4) which are identical values, plus `stick_size` as a compile-time arg. This redundancy exists because the kernel was designed to also handle partial-width blocks (leftover tiles), which is not used in the interleaved variant.

### Single-Buffering Limitation

Both CBs are single-buffered (capacity = 1 block). This serializes the pipeline. For a new operation like layer_norm_rm, consider double-buffering the input CB by setting capacity to `2 * ntiles_per_block` tiles, allowing the reader to fill one buffer while compute processes the other.

### FP32 Accumulation

When input dtype is FLOAT32, the compute kernel enables `fp32_dest_acc_en` for higher precision in the FPU destination register accumulation.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the TensorAccessor and TensorAccessorArgs work for reading interleaved tensors from DRAM in dataflow kernels?"
   **Reason**: Needed to understand the host-side setup and device-side usage of TensorAccessor for resolving stick IDs to NoC addresses.
   **Key Findings**: TensorAccessorArgs on host configures compile-time vs runtime arg passing. On device, `TensorAccessor(args, addr, page_size)` provides `get_noc_addr(page_id)` which maps logical page/stick index to physical NoC address by computing bank_index (round-robin), bank-local offset, and NoC X-Y coordinates.

2. **Query**: "What is the split_blocks_for_tilize function and how does it distribute blocks across cores?"
   **Reason**: Needed to understand the work splitting strategy and what constitutes a "block".
   **Key Findings**: A block = one tile-row (not necessarily 32 sticks -- the size depends on the caller). `split_blocks_for_tilize` does 1D distribution: `nblocks_per_core = ceil(nblocks / grid_area)`, with a cliff core handling remainders. Returns CoreRangeSets for full and cliff cores.

3. **Query**: "What do cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front do in tt-metal?"
   **Reason**: Needed to understand CB synchronization semantics for producer-consumer analysis.
   **Key Findings**: `cb_reserve_back` blocks until N pages are free (producer side). `cb_push_back` signals N pages written. `cb_wait_front` blocks until N pages available (consumer side). `cb_pop_front` frees N pages. Single-buffering occurs when capacity = block size; double-buffering when capacity >= 2x block size.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understand RM page definitions and interleaved memory distribution.
   **Key Information**: RM layout: each row = one page. Interleaved: pages distributed round-robin across banks. Tiled layout: 32x32 tiles with 16x16 faces.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understand host-side and device-side TensorAccessor API for address resolution.
   **Key Information**: Host: `TensorAccessorArgs(buffer)` configures arg passing, `.append_to(ct_args)` adds to compile-time args. Device: `TensorAccessor(args, addr, page_size)` created from args, then `get_noc_addr(page_id)` resolves addresses.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understand the `create_cb` utility function used in the program factory.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a CB with `total_size = num_pages * page_size`, setting page_size per CB index. Returns `(cb_index, cb_handle)`.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understand the compute-side tilize helper library used by the compute kernel.
   **Key Information**: `compute_kernel_lib::tilize<input_cb, output_cb, mode, wait, reconfig>(block_width_tiles, num_blocks)` is a configurable template supporting fast_tilize (for 32x32, Float32/Float16_b), multiple init/uninit modes for back-to-back calls, and WaitBlock/WaitUpfront/NoWait synchronization strategies.
