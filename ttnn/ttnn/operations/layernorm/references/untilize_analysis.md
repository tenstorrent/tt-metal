# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize operation converts tensor data from **TILE layout** (hardware-native 32x32 tiles) to **ROW_MAJOR layout** (contiguous row-major sticks). This is the inverse of the tilize operation. Each tile row of tiles is read, the compute kernel rearranges the tile data into row-major sticks, and the writer kernel writes each resulting stick to the correct output page in DRAM/L1.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Focus of this analysis**: Output stage -- how untilized row-major sticks are written out, output CB sizing, writer kernel pattern, and the `untilize` compute helper signature. This is intended as a reference for building a layernorm operation that produces row-major output.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row of tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (= tensor_width / tile_width for interleaved) |
| **Total units** | `num_tiles_per_col` blocks (= tensor_height / tile_height) |
| **Loop structure** | Outer loop over blocks (tile rows) assigned to this core; inner loop over tile_height stick rows per block |

One "input block" is a single tile-height strip spanning a configurable number of tiles in width. For interleaved inputs this spans the full tensor width; for sharded inputs it spans the shard width.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] (2D+ tensor) | [..., H, W] (same logical shape) |
| **Dimension convention** | Last two dims are height, width | Last two dims are height, width |
| **Tensor layout** | TILE_LAYOUT | ROW_MAJOR |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED, or BLOCK_SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 | Same as input |

### Output Page Structure (Critical for Layernorm Reference)

The output tensor is in ROW_MAJOR layout. The "page" concept depends on the output memory layout:

- **Interleaved or Height-Sharded**: `output_page_width = tensor_width` (one full row = one page)
- **Width-Sharded or Block-Sharded**: `output_page_width = shard_spec.shape[1]` (one shard-width segment = one page)

The output stick size in bytes is: `output_stick_size = output_page_width * element_size`.

Multiple output pages may tile across the width of a single logical row when width/block sharding is used. The variable `output_num_blocks_across_width` tracks how many output pages exist per logical row.

### Layout Transformations

The compute kernel performs the tile-to-row-major conversion. Each 32x32 tile is unpacked and its rows are extracted as contiguous sticks of `tile_width` elements. These sticks from adjacent tiles in the same tile row are concatenated (logically, via CB positioning) to form full-width row-major rows.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) or L1 (sharded) | CB c_0 (input) | reserve_back, push_back |
| 2 | Compute | CB c_0 (input) | CB c_16 (output) | wait_front/pop_front on c_0; reserve_back/push_back on c_16 |
| 3 | Writer | CB c_16 (output) | DRAM/L1 (output buffer) | wait_front, pop_front |

### Detailed Output Stage Flow

1. Compute kernel untilizes one block of `num_tiles_per_input_block` tiles, producing `tile_height` (32) row-major sticks in CB c_16.
2. Writer calls `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` to wait for the full block.
3. Writer reads sticks directly from CB c_16's L1 read pointer (`get_read_ptr`).
4. For each of the `tile_height` rows in the block:
   - Computes the output page_id from the block height index, row offset within the block, and width-wise output block index.
   - Uses `TensorAccessor::get_noc_addr(page_id, byte_offset_within_page)` to get the destination NoC address.
   - Calls `noc_async_write(l1_read_addr, dst_noc_addr, num_bytes_to_write)` to write the stick segment.
   - When input blocks do not align with output pages (width/block sharding), the writer splits a single row's data across multiple output pages via a while loop.
5. After all rows in the block: `noc_async_write_barrier()` followed by `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`.
6. Outer loop repeats for `num_input_blocks_to_process` blocks.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tiles (tiled) | Varies (see below) | num_tiles_per_input_block tiles | Single or Double | Reader | Compute | Block |
| c_16 | cb_output | Output sticks (row-major) | Varies (see below) | num_tiles_per_input_block tiles | Single or Double | Compute | Writer | Block |

### Output CB Sizing (Primary Focus)

The output CB (c_16) capacity is determined by:

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

- **Single-buffered** (`capacity = 1 block`): When core processes only 1 block total -- no overlap needed.
- **Double-buffered** (`capacity = 2 blocks`): When core processes 2+ blocks -- compute can fill the next block while writer drains the current block.

The block size is `num_tiles_per_input_block * output_single_tile_size` bytes. Each tile in the output CB holds `tile_height * tile_width * element_size` bytes of row-major data (even though internally it is still referred to as "tiles" for CB accounting, the data layout within is row-major after untilize).

**Key insight for layernorm**: The output CB should be sized to hold at least one full tile-row width of output data. Double-buffering enables compute/writer overlap when there are multiple blocks.

## Pipeline Pattern Summary

- **Input CB (c_0)**: Single-buffered for 1-block cores or sharded; Double-buffered for multi-block interleaved cores.
- **Output CB (c_16)**: Single-buffered for 1-block cores; Double-buffered for multi-block cores.
- Double-buffering enables compute to process the next block while writer drains the current block.

## Index Calculations

### Output Page ID Calculation (Writer Kernel)

The writer computes the output page_id for each row-major stick using:

```cpp
uint32_t num_rows_already_processed = block_height_index * tile_height + j;  // j = row within tile
uint32_t num_pages_already_processed_in_previous_rows =
    num_rows_already_processed * num_output_blocks_across_width;
uint32_t output_page_id =
    num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;
```

This maps `(block_height_index, row_within_tile, width_block_index)` to a linear page ID. The page ID ordering is: all width blocks for row 0, then all width blocks for row 1, etc.

### TensorAccessor for Address Resolution

The writer creates a `TensorAccessor` from compile-time args starting at index 8:
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

It uses `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` which:
1. Maps page_id to a (bank_id, bank_offset) pair via the distribution spec
2. Adds the byte offset within the page
3. Returns the full 64-bit NoC address for the write

The `output_offset_within_page_in_bytes` is nonzero only for the first output page per row when input blocks don't align with output page boundaries (width/block sharding case).

## Memory Access Patterns

### Write Pattern (Primary Focus)

- **Row-sequential within a block**: For each block of `tile_height` rows, sticks are written in row-major order (row 0 first, then row 1, etc.).
- **Contiguous within a row**: Each row's data is written as one or more contiguous byte segments. For interleaved output, the entire row is a single `noc_async_write`. For width/block sharded output, a row may span multiple output pages with separate writes.
- **Block-sequential across blocks**: Blocks are processed in height order, so writes progress from top to bottom of the tensor.
- **Write barrier per block**: `noc_async_write_barrier()` is called once per block (after all `tile_height` rows), not per row. This batches up to `tile_height` async writes before blocking.

### Read Pattern (De-emphasized)

Reader reads tiles sequentially from DRAM using tile IDs starting from `tile_start_index`. For sharded input, the reader simply signals that local L1 data is ready.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D compute grid) |
| **Grid dimensions** | Up to device compute_with_storage_grid_size |
| **Total cores** | `num_compute_cores` (interleaved: from `split_blocks_for_tilize`; sharded: from shard spec) |
| **Work per core** | `num_input_blocks_per_full_core` blocks (interleaved); `input_shard_height / tile_height` blocks (sharded) |
| **Load balancing** | Near-equal with cliff core for interleaved; shard-defined for sharded |

For **interleaved** input, `split_blocks_for_tilize(grid_size, num_tiles_per_col)` distributes tile-rows across cores. The last core may be a "cliff" core processing fewer blocks.

For **sharded** input, each core processes its local shard. The number of cores equals the shard grid size (or the number of cores with actual data if the grid is over-provisioned).

## Arguments

### Compile-Time Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer index (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output page in bytes (`output_page_width * element_size`) |
| 2 | tile_height | uint32_t | Height of a tile (typically 32) |
| 3 | num_tiles_per_input_block | uint32_t | Number of tiles in one block width-wise |
| 4 | num_output_blocks_across_width | uint32_t | Number of output pages per logical row |
| 5 | output_element_size | uint32_t | Size of one element in bytes |
| 6 | num_cols_per_input_block | uint32_t | `num_tiles_per_input_block * tile_width` |
| 7 | num_cols_per_output_block | uint32_t | `output_page_width` (columns per output page) |
| 8+ | TensorAccessorArgs | ... | Bank mapping and distribution info for output buffer |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | num_input_blocks_to_process | uint32_t | Number of tile-row blocks this core handles |
| 2 | height_wise_input_block_start_index | uint32_t | First block index (row of tiles) for this core |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Actual valid columns (handles uneven sharding) |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page index within each row |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Byte-level offset into first output page |

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | `num_tiles_per_input_block` -- tiles per block width-wise |
| 1 | src_cb_id | uint32_t | Input CB index (c_0) |
| 2 | out_cb_id | uint32_t | Output CB index (c_16) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | `num_input_blocks_to_process` -- blocks for this core |

## Kernel Implementations

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM/L1 output buffer | Write row-major sticks |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - Uses a lambda `write_tiles_in_current_block` that processes one block at a time.
  - Within each block, iterates over `tile_height` rows.
  - For each row, reads contiguous data from the output CB's L1 address.
  - Handles the case where an input block's data spans multiple output pages (width/block sharding) via a while loop that writes segments to successive output pages.
  - Uses `TensorAccessor::get_noc_addr(page_id, offset)` for address resolution -- the `offset` parameter allows writing into the middle of an output page.
  - Calls `noc_async_write_barrier()` once per block (not per row), batching writes.
  - Handles padding: `num_unpadded_cols_per_input_block` may be less than `num_cols_per_input_block` for the last width-wise shard; padded columns are simply not written.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 | CB c_16 | Untilize (tile -> row-major) |

- **File (slow path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **File (fast path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`

#### Untilize Helper Signature (Key for Layernorm Reuse)

```cpp
// From ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp
namespace compute_kernel_lib {

template <
    uint32_t block_width_tiles,    // Tiles per row (compile-time)
    uint32_t input_cb,             // Input CB index (tiled data)
    uint32_t output_cb,            // Output CB index (row-major output)
    untilize_config::InitUninitMode init_uninit_mode = InitAndUninit,
    untilize_config::WaitMode wait_mode = WaitBlock,
    untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
void untilize(uint32_t num_blocks);

}
```

**Usage in untilize compute kernel**:
```cpp
compute_kernel_hw_startup(src_cb_id, out_cb_id);  // Must be called first
compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt);
```

**Template parameters**:
- `block_width_tiles`: Number of tiles per row, must be compile-time. This is the key sizing parameter.
- `input_cb` / `output_cb`: CB indices.
- `init_uninit_mode`: Controls init/uninit lifecycle. Use `InitOnly`/`Neither`/`UninitOnly` for back-to-back calls.
- `wait_mode`: `WaitBlock` (default) waits per block; `WaitUpfront` waits for all tiles; `NoWait` for external sync.
- `reconfig_mode`: For switching data formats mid-kernel.

**Runtime parameter**: `num_blocks` -- number of tile-row blocks to untilize.

**Prerequisite**: `compute_kernel_hw_startup(input_cb, output_cb)` must be called before any untilize call.

**For layernorm usage**: After the normalization compute, call `untilize` with `WaitMode::NoWait` or `WaitMode::WaitUpfront` since the data is already in the CB from the preceding compute stage. Use `InitUninitMode` variants if untilize follows other compute operations in the same kernel.

### Reader Kernel (De-emphasized)

- **Interleaved**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **Sharded**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

## Implementation Notes

### Compute Kernel Selection
Two compute paths are available:
- **Fast path** (`pack_untilize`): Uses hardware-accelerated pack_untilize. Selected when `use_pack_untilize` is true, data type is not UINT16, and for FLOAT32 the block width must be below `MAX_PACK_UNTILIZE_WIDTH`.
- **Slow path** (standard `untilize`): SFPU-based untilize. Used as fallback.

Both use the same `compute_kernel_lib::untilize<>()` helper which internally selects the appropriate implementation based on block width vs DEST capacity.

### FP32 Accumulation
When `fp32_dest_acc_en` is true, unpack-to-dest mode is set to `UnpackToDestFp32` for the input CB. Additionally, INT32/UINT32/FLOAT32 data types trigger `DST_ACCUM_MODE` define which halves the max block count for pack_untilize (4 instead of 8).

### Uneven Sharding Handling
The writer kernel handles two forms of uneven sharding:
1. **Width-wise**: The last shard column may have fewer valid columns. `num_unpadded_cols_per_input_block` tracks the actual number, and only those columns are written (padding is silently dropped).
2. **Height-wise**: The last shard row may have fewer valid tile rows. `num_input_blocks_to_process` is reduced accordingly.

### TensorAccessorArgs Pattern
`TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` appends the bank distribution information to the compile-time args vector. On the device side, `TensorAccessorArgs<8>()` reads these starting at compile-time arg index 8. The resulting `TensorAccessor` object provides `get_noc_addr(page_id, byte_offset)` for address resolution.

### Output Page Width vs Input Block Width
These may differ when input and output have different sharding configurations. The writer handles this mismatch by splitting each row's write across multiple output pages via its inner while loop. The key variables are:
- `num_cols_per_input_block`: Width of data in the input block (from tile grid)
- `num_cols_per_output_block`: Width of one output page (from output shard spec)
- `num_output_blocks_across_width`: Total output pages per logical row

## External Knowledge Sources

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor API used by the writer kernel for address resolution.
   **Key Information**: `TensorAccessor(args, base_addr, page_size)` construction pattern; `get_noc_addr(page_id, offset)` returns 64-bit NoC address with optional byte offset within a page; `TensorAccessorArgs<base_idx>()` on device side reads compile-time args starting at the given index.

2. **Source**: `tt_metal/hw/inc/api/tensor/tensor_accessor.h` (line 100)
   **Reason**: Confirming the `get_noc_addr(page_id, offset)` signature supports an intra-page byte offset.
   **Key Information**: `get_noc_addr(const uint32_t page_id, const uint32_t offset = 0, uint8_t noc = noc_index)` -- offset defaults to 0, allowing writes to arbitrary positions within a page.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
   **Reason**: Understanding the unified untilize compute helper signature and its configuration modes.
   **Key Information**: Template parameters `block_width_tiles`, `input_cb`, `output_cb` plus optional `InitUninitMode`, `WaitMode`, `ReconfigureRegisterDatatypeMode`. Runtime parameter is `num_blocks`. Requires `compute_kernel_hw_startup()` before use.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used to set up circular buffers.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer)` creates a CB with total size = `num_pages * page_size`. When `buffer` is non-null, the CB is backed by the globally-allocated buffer (sharded input pattern).

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding how tile-row blocks are distributed across cores.
   **Key Information**: `split_blocks_for_tilize(grid_size, nblocks)` returns `BlockSplit` with full core range, cliff core range, blocks per full core, and blocks per cliff core. Distributes `nblocks` tile-rows as evenly as possible.
