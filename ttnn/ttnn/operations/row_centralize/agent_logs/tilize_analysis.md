# Tilize Multi-Core Interleaved Implementation Analysis

## Overview

The **tilize** operation converts a tensor from **row-major layout** to **tiled layout** (32x32 tiles). It reads contiguous row-major "sticks" (rows) from DRAM, rearranges them into the tile format expected by Tenstorrent's matrix engine (face-based 32x32 tiles with four 16x16 faces), and writes the resulting tiles back to DRAM in interleaved memory layout.

This is the **multi-core interleaved** variant, meaning the input and output tensors are both stored in interleaved (round-robin across banks) memory layout, and work is distributed across multiple Tensix cores for parallelism.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Role in hybrid pipeline**: This analysis serves as the `input_stage` reference for the `row_centralize` operation, which performs row-wise standardization on row-major interleaved tensors. The tilize stage converts RM input to TILE format for downstream compute.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a block is one tile-row of the tensor) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / TILE_WIDTH`) |
| **Total units** | `nblocks` = `ceil(ntiles / ntiles_per_block)` = number of tile-rows in the tensor |
| **Loop structure** | Outer loop over tile-rows of 32 sticks each; inner loop reads full-width blocks per tile-row |

A single **block** corresponds to one horizontal tile-row of the tensor: all tiles spanning the last dimension at a given set of 32 consecutive rows (TILE_HEIGHT). For example, a tensor with padded shape `[1, 1, 64, 128]` has `ntiles_per_block = 128/32 = 4` tiles per block, and `nblocks = (64*128 / 1024) / 4 = 2` blocks total.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary N-D (treated as 2D: height x width) |
| **Dimension convention** | Last dimension is width (contiguous in memory) |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1, determined by memory config) |
| **Data type** | BFLOAT16, FLOAT32 (any type supported by `datatype_to_dataformat_converter`) |

- **Page size**: One row ("stick") = `padded_shape[-1] * element_size` bytes
- **Page count**: `physical_volume / padded_shape[-1]` = number of sticks (rows)

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input (last dim = width) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1, determined by output_mem_config) |
| **Data type** | Same as input (or output_dtype from TilizeParams) |

- **Page size**: One tile = `tile_size(output_cb_data_format)` bytes (1024 for BF16, 2048 for FP32 with standard 32x32 tiles)
- **Page count**: `ntiles = physical_volume / TILE_HW` = total number of 32x32 tiles

### Layout Transformations

The core transformation is **ROW_MAJOR to TILE_LAYOUT**:
- Input: Data stored as contiguous rows (sticks). Each row is one page.
- Output: Data stored as 32x32 tiles, each internally organized into four 16x16 faces (face0, face1, face2, face3 in row-major order within the tile).
- The `tilize_block` LLK operation on the compute core performs this rearrangement using the hardware unpacker in "tilize mode", which reads row-major data from the input CB and produces tile-formatted data in the output CB.

## Data Flow Pattern

### Step-by-step flow

| Stage | Kernel | Reads From | Writes To | CB Operations | Description |
|-------|--------|------------|-----------|---------------|-------------|
| 1 | Reader | DRAM (interleaved, RM pages) | CB c_0 | `cb_reserve_back`, `cb_push_back` | Reads 32 consecutive sticks (one tile-height) across full tensor width, writes raw row-major data into input CB |
| 2 | Compute | CB c_0 | CB c_16 | `cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back` | Performs tilize: reads row-major block from input CB, rearranges into tile format, writes tiles to output CB |
| 3 | Writer | CB c_16 | DRAM (interleaved, tile pages) | `cb_wait_front`, `cb_pop_front` | Reads one tile at a time from output CB, writes to DRAM using tile-page addressing |

### Detailed Reader Logic

The reader kernel (`reader_unary_stick_layout_split_rows_interleaved.cpp`) works as follows:

1. **Outer loop**: Iterates over groups of `tile_height` (32) sticks. Each group corresponds to one tile-row.
2. **NOC address pre-computation**: For each group of 32 sticks, computes the NOC address of each stick using `get_noc_addr(stick_id, s)` with the TensorAccessor `s`. This maps the logical stick ID to the physical DRAM bank and offset.
3. **Inner loop**: For each full block in the row (`num_full_blocks_in_row`, which is 1 in this variant), calls `read_tiles()`.
4. **read_tiles lambda**: Reserves CB space for `num_tiles_per_block` tiles, then for each of the 32 sticks, issues a `noc_async_read` of `width_size` bytes from the pre-computed NOC address to the L1 write pointer. The write pointer advances by `width_size` per stick, packing 32 sticks worth of data into the CB. After all 32 reads, issues `noc_async_read_barrier()` and pushes the block.

Key insight: The reader packs 32 row-major sticks into the input CB contiguously. The compute kernel's tilize hardware then reinterprets this contiguous row-major block as tile data.

### Detailed Writer Logic

The writer kernel (`writer_unary_interleaved_start_id.cpp`) writes one tile at a time:

1. Loops from `start_id` to `start_id + num_pages`.
2. For each tile: `cb_wait_front(1)` -> read L1 address -> `noc_async_write_page(tile_id, s, l1_addr)` -> `noc_async_writes_flushed()` -> `cb_pop_front(1)`.
3. Final `noc_async_write_barrier()` after loop.

The writer uses `noc_async_writes_flushed()` (not barrier) per tile for pipelining, only issuing a full barrier at the end.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|--------------------|-----------|---------|---------|----|
| c_0 | cb_input | Input staging (row-major sticks) | `ntiles_per_block` | `ntiles_per_block` | Single | Reader | Compute | Block |
| c_16 | cb_output | Output staging (tilized tiles) | `ntiles_per_block` | `ntiles_per_block` | Single | Compute | Writer | Block |

**Capacity calculation**: Both CBs are created with `num_pages = ntiles_per_block`. The page size for c_0 is `input_single_tile_size` and for c_16 is `output_single_tile_size`.

**Important**: Both CBs are **single-buffered** (capacity equals block size). This means:
- The reader must complete writing all `ntiles_per_block` tiles worth of row-major data before compute can start processing.
- The compute must complete writing all `ntiles_per_block` output tiles before the writer can start writing them to DRAM.
- There is no overlap between reader and compute, or between compute and writer, within a single block.

## Pipeline Pattern Summary

Both c_0 and c_16 are single-buffered:
- **c_0**: Capacity = `ntiles_per_block`, Block size = `ntiles_per_block` --> **Single-buffered**
- **c_16**: Capacity = `ntiles_per_block`, Block size = `ntiles_per_block` --> **Single-buffered**

The pipeline operates in a strictly sequential block-by-block fashion: Read block N -> Compute block N -> Write block N -> Read block N+1 -> etc. However, because the writer writes one tile at a time (popping one tile at a time), the compute for block N+1 can potentially begin as soon as the writer has popped enough tiles from c_16 to make room for new output, though in practice the compute waits for the full block from the reader first.

## Index Calculations

### Reader: Stick ID to NOC Address

The reader uses `TensorAccessor` to map logical stick IDs to physical memory locations:

```cpp
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

- `stick_id` is a linear index into the row-major tensor (0, 1, 2, ... up to total number of sticks)
- `get_noc_addr(stick_id, s)` uses the TensorAccessor to compute which DRAM bank holds this stick and at what offset, then returns the full 64-bit NOC address
- The `TensorAccessorArgs` are constructed from `*src0_buffer` on the host and appended to compile-time args, encoding bank coordinates and distribution information
- On device, `TensorAccessorArgs<1>()` reconstructs the accessor from compile-time args starting at index 1 (index 0 is `stick_size`)

### Writer: Tile ID to NOC Address

The writer uses `noc_async_write_page(i, s, l1_read_addr)` where:
- `i` is the linear tile ID (starting from `start_id`)
- `s` is a `TensorAccessor` constructed from `TensorAccessorArgs<1>()` with compile-time args starting at index 1 (index 0 is `cb_id_out`)
- The TensorAccessor maps the tile ID to the correct DRAM bank and offset for interleaved tile pages

### Mapping from core to sticks/tiles

Each core is assigned a contiguous range of blocks:
- `tile_start_id = sum of tiles from cores 0..i-1 = i * ntiles_per_block * nblocks_per_core`
- `row_start_id = sum of rows from cores 0..i-1 = i * TILE_HEIGHT * nblocks_per_core`

## Memory Access Patterns

### Read Pattern

- **Pattern**: Semi-sequential with stride
- The reader reads 32 sticks per block, where each stick is a full row of the tensor. Within a block, it reads stick 0, stick 1, ... stick 31 sequentially by stick ID, but each stick maps to a potentially different DRAM bank (interleaved round-robin).
- Each `noc_async_read` transfers `block_width_size` = `padded_shape[-1] * element_size` bytes (one full row).
- Within a tile-height group, reads are sequential in stick ID but addresses jump between banks.
- Across blocks, stick IDs increase contiguously.

### Write Pattern

- **Pattern**: Sequential, one tile at a time
- The writer writes tiles sequentially starting from `start_id`, incrementing by 1 each iteration.
- Each write is one full tile page (`output_single_tile_size` bytes).
- Due to interleaved layout, consecutive tile IDs map to consecutive banks in round-robin fashion.
- Uses `noc_async_writes_flushed()` per tile (not full barrier) for pipelining, with a single `noc_async_write_barrier()` at the end.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of cores from available grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` (e.g., 8x8 = 64 cores) |
| **Total cores** | `ncores` = `ceil(nblocks / nblocks_per_core)` |
| **Work per core** | `nblocks_per_core` blocks (full cores), `nblocks_per_core_cliff` blocks (cliff core) |
| **Load balancing** | Equal distribution with optional cliff core for remainder |

### Work Split Details

The `split_blocks_for_tilize` function distributes blocks across cores:

1. `nblocks_per_core = ceil(nblocks / grid_area)` -- aims to use all available cores
2. `ncores = ceil(nblocks / nblocks_per_core)` -- actual cores needed
3. `nblocks_per_core_cliff = nblocks % nblocks_per_core` -- remainder for last core (0 means no cliff)
4. All non-cliff cores get `nblocks_per_core` blocks each
5. The cliff core (if any) gets `nblocks_per_core_cliff` blocks

Cores are enumerated linearly from the `available_grid` CoreRangeSet using `corerange_to_cores()`.

## Arguments

### Reader Kernel: Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size (block_size_nbytes) | uint32_t | Size of one row-major stick in bytes: `padded_shape[-1] * element_size` |
| 1+ | TensorAccessorArgs | uint32_t[] | Bank distribution info for source buffer (appended via `TensorAccessorArgs(*src0_buffer).append_to()`) |

### Reader Kernel: Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total number of sticks to read: `nblocks_per_core * TILE_HEIGHT` |
| 2 | block_width_size | uint32_t | Width of one block in bytes (same as stick_size for this variant) |
| 3 | num_tiles_per_block | uint32_t | Number of tiles in one block (width direction) |
| 4 | block_width_size | uint32_t | Width of one block in bytes (repeated, used as `block_width_size` in read_tiles) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for this variant (full row is one block) |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial blocks) |
| 7 | leftover_width_in_row | uint32_t | Always 0 (no partial blocks) |
| 8 | start_stick_id (row_start_id) | uint32_t | First stick ID for this core |

### Writer Kernel: Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_16) |
| 1+ | TensorAccessorArgs | uint32_t[] | Bank distribution info for destination buffer |

### Writer Kernel: Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_pages | uint32_t | Total number of tiles to write: `ntiles_per_block * nblocks_per_core` |
| 2 | start_id (tile_start_id) | uint32_t | First tile ID for this core |

### Compute Kernel: Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt (nblocks_per_core) | uint32_t | Number of tile-row blocks to process on this core |
| 1 | per_core_block_tile_cnt (ntiles_per_block) | uint32_t | Number of tiles per block (width of tensor in tiles) |

Cliff cores receive different compile-time args: `{nblocks_per_core_cliff, ntiles_per_block}`.

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per block via noc_async_read |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Pre-computes 32 NOC addresses per tile-row group using `get_noc_addr(stick_id, s)`
  - Reads 32 sticks (each `block_width_size` bytes) into the input CB contiguously
  - The `read_tiles` lambda issues all 32 reads, then barriers, then pushes
  - After each read, advances `base_src_noc_addr[k] += width_size` to handle row-wide reads in chunks (though in this variant `num_full_blocks_in_row = 1`, so each tile-row is one block)
  - The TensorAccessor handles the interleaved bank mapping automatically

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_2 (MATH/PACK/UNPACK) | N/A | CB c_0 | CB c_16 | tilize_block (LLK hardware tilize) |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` for hardware initialization
  - Delegates to `compute_kernel_lib::tilize<c_0, c_16>(per_core_block_tile_cnt, per_core_block_cnt)` from `tilize_helpers.hpp`
  - Uses default template parameters: `InitAndUninit`, `WaitBlock`, `Standard` speed mode, `NoReconfigure`
  - The helper library handles the init/block-loop/uninit pattern:
    1. `tilize_init(c_0, block_width_tiles, c_16)` -- configures unpacker for tilize mode
    2. For each block: `cb_wait_front(c_0, ntiles_per_block)` -> `cb_reserve_back(c_16, ntiles_per_block)` -> `tilize_block(c_0, ntiles_per_block, c_16)` -> `cb_push_back(c_16, ntiles_per_block)` -> `cb_pop_front(c_0, ntiles_per_block)`
    3. `tilize_uninit(c_0, c_16)` -- restores unpacker state
  - `tilize_block` internally: calls `llk_unpack_tilize_block` to configure unpacker, then for each tile: acquires dest, does `llk_math_eltwise_unary_datacopy` (A2D mode), packs result, releases dest
  - When `a.dtype() == FLOAT32`, `fp32_dest_acc_en` is set to true for FP32 accumulation in dest registers

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_16 | DRAM (tile pages) | Write tiles via noc_async_write_page |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**:
  - Generic writer shared with unary operations (not tilize-specific)
  - Gets page size from CB interface: `get_local_cb_interface(cb_id_out).fifo_page_size`
  - Writes one tile at a time in a simple loop
  - Uses `noc_async_writes_flushed()` per tile (lightweight flush, not full barrier) for pipelined writes
  - Final `noc_async_write_barrier()` ensures all writes complete
  - Supports `OUT_SHARDED` and `BACKWARDS` modes via preprocessor defines (neither used in this variant)

## Implementation Notes

### Program Caching (override_runtime_arguments)

The factory implements `override_runtime_arguments` which only updates buffer addresses (runtime arg index 0 for both reader and writer). This enables efficient program re-use when the same operation runs again with different tensors of the same shape -- only the DRAM addresses change, not the program structure.

### Sub-Core Grid Support

The factory accepts an optional `sub_core_grids` parameter in `TilizeParams`, allowing the caller to restrict which cores are used. This is useful for operations that need to reserve cores for other purposes or for multi-program pipelines.

### FP32 Accumulation

When the input data type is FLOAT32, the compute kernel enables `fp32_dest_acc_en` in the ComputeConfig. This configures the dest registers for 32-bit accumulation, which halves the available dest space (from 8 to 4 tiles) but maintains full precision.

### Reader Kernel Design: Split Rows Pattern

The reader kernel name includes "split_rows" because it is designed to handle cases where a single tile-row of data could be split across multiple blocks (when the tensor is very wide). In this interleaved variant, `num_full_blocks_in_row` is always 1, meaning the entire width is one block. The `num_leftover_tiles` and `leftover_width_in_row` args are 0, indicating no partial blocks. The more general form of this kernel (used by the block variant) can split wide rows into multiple smaller blocks.

### No Sharding Support in This Variant

This factory is specifically for **interleaved** memory layout. Sharded variants exist separately (`tilize_multi_core_block_program_factory`). The reader uses `TensorAccessor` which handles interleaved bank mapping, and the writer similarly uses `TensorAccessor` for interleaved tile-page writes.

### CB Sizing Consideration for Downstream Operations

Both CBs are sized to `ntiles_per_block` (full tensor width in tiles). For a tensor with width 4096 in BF16, that is 128 tiles * 2KB = 256KB per CB, which is a significant portion of L1 (1.5MB total). The single-buffered design means peak L1 usage is `ntiles_per_block * (input_tile_size + output_tile_size)`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in tt-metal? What does tilize_block do at the hardware level?"
   **Reason**: Needed to understand the fundamental tilize hardware mechanism.
   **Key Findings**: DeepWiki was unavailable (500 error). Analysis derived from local source code inspection of `tilize.h`, `tilize_helpers.hpp`, and `tilize_helpers.inl`.

2. **Query**: "What is TensorAccessorArgs and how does append_to work for compile-time arguments?"
   **Reason**: Needed to understand how accessor arguments are serialized for kernel consumption.
   **Key Findings**: DeepWiki was unavailable. Analysis derived from `tensor_accessor_args.hpp` and `tensor_accessor.md`.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major vs tiled layout, page definitions, and face organization within tiles.
   **Key Information**: Row-major pages = one row each. Tiled pages = 32x32 tiles with four 16x16 faces in row-major face order. Interleaved = round-robin page distribution across banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor maps logical page IDs to physical NOC addresses.
   **Key Information**: TensorAccessor provides `get_noc_addr(page_id)` for address calculation. On host, `TensorAccessorArgs(buffer)` creates args that encode bank distribution. On device, `TensorAccessorArgs<N>()` reconstructs from compile-time args at offset N. The accessor handles interleaved bank mapping automatically.

3. **Source**: `tt_metal/include/compute_kernel_api/tilize.h`
   **Reason**: Understanding the hardware-level tilize_block implementation.
   **Key Information**: `tilize_block` calls `llk_unpack_tilize_block` to configure the unpacker for tilize mode, then for each tile in the block: acquires dest, performs `llk_math_eltwise_unary_datacopy` (A2D datacopy in tilize mode), packs to output CB, releases dest. The unpacker handles the row-major to tile rearrangement.

4. **Source**: `tt_metal/include/compute_kernel_api/compute_kernel_hw_startup.h`
   **Reason**: Understanding the hardware initialization requirement before any compute operations.
   **Key Information**: `compute_kernel_hw_startup` must be called exactly once at kernel start. It configures UNPACK, MATH, and PACK hardware units. Performs MMIO writes that require idle state.

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding how blocks are distributed across cores.
   **Key Information**: `split_blocks_for_tilize` computes `nblocks_per_core = ceil(nblocks/grid_area)`, then `ncores = ceil(nblocks/nblocks_per_core)`. Last core may be a "cliff" with fewer blocks. Returns core ranges for full and cliff cores.

6. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used to create circular buffers.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a CB with total size = `num_pages * page_size`, configured with the given data format. Returns `(cb_id, cb_handle)` tuple.
