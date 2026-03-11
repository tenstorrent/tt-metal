# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize operation converts tile-layout tensors to row-major (RM) layout. It reads tiled data from DRAM (or sharded L1), runs it through a compute kernel that rearranges elements from 32x32 tile format into contiguous row-major sticks, then writes those sticks back to DRAM (or sharded L1). This analysis focuses on the **output stage**: how the output CB is sized, how untilized sticks are laid out in L1 after compute, and how the writer kernel extracts and writes those sticks to DRAM as row-major pages.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one "tile-row": a horizontal row of tiles spanning the tensor width) |
| **Unit size** | `num_tiles_per_input_block` tiles (= `tensor_width / tile_width`). For standard 32-wide tiles on a 1024-wide tensor this is 32 tiles. |
| **Total units** | `num_tiles_per_col` = `tensor_height / tile_height` blocks total |
| **Loop structure** | Outer: iterate over blocks assigned to this core. Inner: for each block the compute kernel untilizes `block_width_tiles` tiles producing `tile_height` sticks; the writer extracts those sticks row-by-row. |

One block represents all tiles in a single tile-row of the tensor. After untilize, one block becomes `tile_height` (typically 32) contiguous row-major sticks in L1, each `num_cols_per_input_block * element_size` bytes wide.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] (arbitrary dims, flattened to 2D: height x width) |
| **Dimension convention** | Height = `physical_volume / tensor_width`, Width = `padded_shape[-1]` |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same shape as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED (or sharded for width/block-sharded outputs) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | Same as input |

### Layout Transformations

The sole transformation is tile-to-row-major conversion. No resharding or dtype conversion occurs. The compute kernel (pack_untilize) reads tiles from the input CB and writes row-major sticks to the output CB. The writer kernel then transfers those sticks from L1 to DRAM.

## Data Flow Pattern (Output-Stage Focus)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) or L1 (sharded) | CB_in (c_0) | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB_in (c_0) | CB_out (c_16) | `cb_wait_front(input)`, `cb_reserve_back(output)`, `pack_untilize_block`, `cb_pop_front(input)`, `cb_push_back(output)` |
| 3 | **Writer** | **CB_out (c_16)** | **DRAM (interleaved)** | **`cb_wait_front`, row-by-row `noc_async_write`, `noc_async_write_barrier`, `cb_pop_front`** |

### Detailed Writer Data Flow

1. **Wait for block**: `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- waits until one full block of untilized tiles is available.
2. **Get L1 base address**: `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- points to the start of the untilized data.
3. **Iterate rows**: For each of `tile_height` rows within the block:
   - Compute L1 read address for this row: `base_l1_read_addr + j * num_cols_per_input_block * output_element_size`
   - Compute the output page_id based on the row index and width-wise output block position.
   - Write row data to DRAM using `noc_async_write`, potentially splitting across multiple output pages if the input block spans multiple output pages (width/block sharding case).
4. **Barrier + pop**: `noc_async_write_barrier()` then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging (tiled data) | See note below | `num_tiles_per_input_block` | Double (interleaved, multi-block) or Single (single-block/sharded) | Reader | Compute | Block |
| **c_16** | **cb_output** | **Output staging (RM sticks)** | **See note below** | **`num_tiles_per_input_block`** | **Double (multi-block) or Single (single-block)** | **Compute** | **Writer** | **Block** |

### Output CB Sizing (Key Focus)

The output CB capacity is determined by this logic (program factory lines 149-156):

```
if (num_input_blocks_per_full_core == 1):
    output_cb_num_tiles = num_tiles_per_input_block          # Single-buffered
else:
    output_cb_num_tiles = num_tiles_per_input_block * 2      # Double-buffered
```

**Interpretation**:
- **Block size** = `num_tiles_per_input_block` tiles. This is the number of tiles in one complete tile-row of the tensor.
- **Double-buffered** when a core processes 2+ blocks: capacity = 2 blocks. While the writer drains one block, compute can fill the next.
- **Single-buffered** when a core processes only 1 block: no overlap needed, capacity = 1 block.

**Physical size in bytes**: `output_cb_num_tiles * output_single_tile_size` where `output_single_tile_size = tt::tile_size(output_cb_data_format)` (e.g., 2048 bytes for bfloat16 32x32 tiles).

### What a "tile" means in the output CB

After untilize, a "tile" in the output CB no longer contains tile-format data. The pack_untilize hardware operation writes row-major data but the CB still tracks capacity in tile-sized pages. The output CB page size is `output_single_tile_size` (the byte size of one tile), but the content is now `tile_height` contiguous row-major sticks, each `tile_width * element_size` bytes wide. For `num_tiles_per_input_block` tiles, the entire output block holds `tile_height` sticks each `num_cols_per_input_block * element_size` bytes wide, where `num_cols_per_input_block = num_tiles_per_input_block * tile_width`.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering | Overlap Potential |
|----|----------|------------|-----------|-------------------|
| c_0 (input) | 1x or 2x block (interleaved); full shard (sharded) | `num_tiles_per_input_block` | Single or Double | Reader-Compute overlap when double-buffered |
| **c_16 (output)** | **1x or 2x block** | **`num_tiles_per_input_block`** | **Single or Double** | **Compute-Writer overlap when double-buffered** |

## Index Calculations

### Output Page ID Mapping (Writer Kernel)

The writer must map from its block-local row index to a global DRAM page_id for the output tensor. The output tensor is row-major interleaved, so each page corresponds to one row-major stick (one row of the tensor or one shard width).

For each row `j` within a block at `block_height_index`:

```
num_rows_already_processed = block_height_index * tile_height + j
num_pages_already_processed_in_previous_rows = num_rows_already_processed * num_output_blocks_across_width
output_page_id = num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index
```

For the simple interleaved case (no width/block sharding), `num_output_blocks_across_width = 1` and `width_wise_output_block_start_index = 0`, so `output_page_id = num_rows_already_processed`. This is a 1:1 mapping from row index to page index.

### TensorAccessor Usage

The writer creates a `TensorAccessor` on the device side:
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // compile-time args start at index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

To write a stick, it calls:
```cpp
uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write);
```

The `get_noc_addr(page_id, offset)` method computes the physical bank address for the given page, then adds `offset` bytes. This is used when writing to a position partway through an output page (relevant for width/block sharding where an input block may not align with output page boundaries).

## Memory Access Patterns

### Read Pattern (De-emphasized)

Sequential tile reads from DRAM, one tile at a time via `noc_async_read`. Reader waits for each read to complete before pushing to CB.

### Write Pattern (Key Focus)

**Pattern**: Row-by-row sequential writes to interleaved DRAM.

Within a single block of `tile_height` rows:
1. The writer iterates `j = 0..tile_height-1` (typically 0..31).
2. For each row `j`, it computes the L1 source address as an offset from the CB read pointer: `base + j * num_cols_per_input_block * element_size`. This is a stride of exactly one stick width.
3. It computes the output `page_id` which increments by `num_output_blocks_across_width` for each successive row (equals 1 for simple interleaved, meaning consecutive pages).
4. It issues `noc_async_write(l1_addr, noc_addr, stick_size_bytes)`.
5. After all `tile_height` rows are written, it issues `noc_async_write_barrier()` then `cb_pop_front`.

**Access ordering**: Rows within a block are written in order (row 0 first, row 31 last). Blocks are processed in order of `height_wise_input_block_index` (top to bottom). This produces sequential page writes to the output buffer.

**For the simple interleaved output case**, the write size per row is `output_stick_size = tensor_width * element_size` bytes. This is one complete row of the output tensor. Pages are distributed round-robin across DRAM banks by the TensorAccessor.

### Width/Block Sharding Complication

When the output is width-sharded or block-sharded, `output_page_width < tensor_width`. A single input block row may span multiple output pages. The writer handles this via a `while` loop that:
1. Writes `min(remaining_input_cols, remaining_output_page_cols)` columns.
2. Advances the output page and resets the within-page offset.
3. Repeats until all input columns are written.

For a `layer_norm_rm` operation writing to simple interleaved DRAM, this complication does not apply -- `num_output_blocks_across_width = 1` and each row maps to exactly one page.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` = min(grid_area, num_tiles_per_col) |
| **Work per core** | `num_rows_per_full_core` tile-rows (blocks); cliff core gets `num_rows_per_cliff_core` |
| **Load balancing** | Nearly equal; at most 1 cliff core with fewer blocks |

The function `split_blocks_for_tilize(grid_size, num_tiles_per_col)` divides the total number of tile-rows across available cores. Each core processes a contiguous range of tile-rows. The cliff core (if any) handles the remainder.

## Arguments

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Bytes per output stick/page (`output_page_width * element_size`) |
| 2 | `tile_height` | uint32_t | Tile height (32 for standard tiles) |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per block width (`tensor_width / tile_width`) |
| 4 | `num_output_blocks_across_width` | uint32_t | Output pages per tensor row (1 for interleaved, >1 for width/block sharded) |
| 5 | `output_element_size` | uint32_t | Bytes per element (2 for bf16, 4 for f32) |
| 6 | `num_cols_per_input_block` | uint32_t | Columns per input block (`num_tiles_per_input_block * tile_width`) |
| 7 | `num_cols_per_output_block` | uint32_t | Columns per output page (`output_page_width`) |
| 8+ | TensorAccessor args | (various) | Bank mapping, addresses, etc. appended by `TensorAccessorArgs(*dst_buffer).append_to(...)` |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-rows this core must write |
| 2 | `height_wise_input_block_start_index` | uint32_t | First tile-row index for this core |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Valid (non-padding) columns to write from each block |
| 4 | `width_wise_output_block_start_index` | uint32_t | First output page index within each row for this core |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Byte-offset (in columns) into the first output page |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | Tiles per block (= `num_tiles_per_input_block`) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Compute Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks (tile-rows) to process |

## Kernel Implementations

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB_out (c_16) | DRAM (interleaved) | Extract RM sticks from CB, write to DRAM pages |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - The inner function `write_tiles_in_current_block` processes one block at a time.
  - It calls `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` to wait for a complete block of untilized data.
  - It reads the L1 address with `get_read_ptr(cb_id_out0)` and computes per-row offsets as `j * num_cols_per_input_block * output_element_size`.
  - For each of `tile_height` rows, it computes the output `page_id` and uses `TensorAccessor::get_noc_addr(page_id, offset)` to get the DRAM NOC address.
  - It issues `noc_async_write(l1_addr, noc_addr, bytes)` for each row.
  - After all rows in the block are written, it calls `noc_async_write_barrier()` and `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`.
  - The outer loop iterates over `num_input_blocks_to_process` blocks assigned to this core.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB_in (c_0) | CB_out (c_16) | Untilize (tile-to-RM conversion) |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` (slow path) or `pack_untilize_variable_num_blocks.cpp` (fast path)
- **Key Logic**:
  - Both variants call `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt)`.
  - The unified `untilize` template in `untilize_helpers.inl` handles DEST register limits by splitting wide rows into sub-blocks if `block_width_tiles > DEST_AUTO_LIMIT`.
  - Per block: `cb_wait_front(input, block_width_tiles)`, `cb_reserve_back(output, block_width_tiles)`, `pack_untilize_block(...)`, `cb_pop_front(input)`, `cb_push_back(output, block_width_tiles)`.
  - The `pack_untilize_block` hardware function transforms tile-format data into row-major sticks in the output CB.
  - The fast `pack_untilize` path uses hardware packer acceleration; the slow `untilize` path falls back to SFPU datacopy when pack_untilize is unsupported (UINT16, or FLOAT32 with very wide rows).

### Reader Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | CB_in (c_0) | Read tiles from DRAM |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp` (interleaved) or `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` (sharded)

## Implementation Notes

### Stick Layout in Output CB After Untilize

After `pack_untilize_block` executes, the output CB contains `tile_height` contiguous row-major sticks. Each stick is `num_cols_per_input_block * element_size` bytes wide. The sticks are stored contiguously in L1 starting at `get_read_ptr(cb_id_out0)`. Row `j` starts at offset `j * num_cols_per_input_block * element_size` from the base.

This means the output CB effectively holds a small row-major matrix of shape `[tile_height, num_cols_per_input_block]` after each block is processed. The writer simply iterates through these rows and writes each to its corresponding DRAM page.

### Output CB Page Size vs Stick Size

The output CB is configured with `page_size = output_single_tile_size` (the size of one tile in bytes), and `num_pages = output_cb_num_tiles`. The CB tracks capacity in tile-sized units, but the data written by pack_untilize is row-major. The total byte capacity of the CB is `output_cb_num_tiles * output_single_tile_size`, which equals `num_tiles_per_input_block * tile_height * tile_width * element_size * buffering_factor`. This exactly accommodates `buffering_factor` blocks of `tile_height` sticks.

### Writer Barrier Strategy

The writer issues `noc_async_write_barrier()` after writing all `tile_height` rows of a single block (not after each individual row). This batches the barrier across 32 writes, reducing overhead. The total bytes written before each barrier is `tile_height * output_stick_size` (e.g., 32 * 1024 * 2 = 64KB for a 1024-wide bf16 tensor).

### Unpadded Column Handling

The runtime argument `num_unpadded_cols_per_input_block` allows the writer to skip writing padding columns at the end of a block. This matters for sharded inputs where the last shard in a row may contain padding. The writer writes exactly `num_unpadded_cols_per_input_block` columns per row instead of the full `num_cols_per_input_block`. For simple interleaved input, `num_unpadded_cols_per_input_block == num_cols_per_input_block` (no padding).

### Relevance to layer_norm_rm

For a `layer_norm_rm` operation that:
1. Reads RM sticks from DRAM
2. Tilizes them
3. Performs layer norm compute
4. Untilizes results
5. Writes RM sticks back to DRAM

The output stage (steps 4-5) can follow this untilize pattern directly:
- **Output CB**: Use `c_16` (or any non-conflicting CB index), sized to `num_tiles_per_row` tiles per block, double-buffered if processing multiple blocks.
- **Compute untilize call**: Use `compute_kernel_lib::untilize<block_width_tiles, in_cb, out_cb>` with appropriate `InitUninitMode` if chaining with other compute operations.
- **Writer kernel**: Extract sticks from the output CB row-by-row, compute DRAM page_id per row, and write via `TensorAccessor::get_noc_addr(page_id)` + `noc_async_write`.
- **Page structure**: For interleaved RM output, each page = one tensor row = `tensor_width * element_size` bytes. Page IDs are sequential row indices.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize compute kernel work? What does it do to the data layout - specifically how does it convert tile-layout data to row-major sticks in the output circular buffer?"
   **Reason**: Needed to understand what data format exists in the output CB after compute, so the writer's L1 access pattern could be explained.
   **Key Findings**: The unpack operation extracts elements from tiles, then the pack operation writes them sequentially as row-major data into the output CB. The `pack_untilize` variant uses hardware packer acceleration. After untilize, the output CB contains row-major sticks.

2. **Query**: "What is split_blocks_for_tilize in ttnn work_split_tilize.hpp? How does it split work across cores?"
   **Reason**: Needed to understand core distribution strategy and what a "block" means.
   **Key Findings**: A block is one tile-row (all tiles horizontally at one tile height). `split_blocks_for_tilize` divides total tile-rows across cores. Returns full-core count, cliff-core count, and blocks per each. Cliff core gets the remainder.

3. **Query**: "How does TensorAccessor work with an offset parameter in get_noc_addr?"
   **Reason**: The writer kernel calls `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` and needed to understand what the offset does.
   **Key Findings**: The offset is added to the base address of the page within its bank. This allows writing to an arbitrary byte position within a page, used when an input block only partially fills an output page.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor is constructed on both host and device, and its API for address computation.
   **Key Information**: `TensorAccessorArgs` on host side appends compile-time args. On device, `TensorAccessor(args, base_addr, page_size)` is constructed. `get_noc_addr(page_id)` returns the NOC address for that page, handling bank interleaving automatically.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Confirming page structure for row-major interleaved tensors.
   **Key Information**: For row-major layout, each row of the 2D tensor is one page. Pages are distributed round-robin across DRAM banks in interleaved mode.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the compute kernel's untilize implementation and its CB interaction pattern.
   **Key Information**: The unified `untilize()` template handles DEST limits by splitting wide rows into sub-blocks. Per block: `cb_wait_front(input)` -> `cb_reserve_back(output)` -> `pack_untilize_block` -> `cb_pop_front(input)` -> `cb_push_back(output)`. Supports `WaitMode::WaitBlock` (per-block sync) and `WaitMode::WaitUpfront` (wait for all tiles at start).

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used to configure circular buffers.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer)` creates a CB with total size = `num_pages * page_size`. If `buffer` is non-null, the CB is backed by that globally-allocated buffer (used for sharded inputs).

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` function signature and return values.
   **Key Information**: Returns `BlockSplit{ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff}`. The cliff core handles `nblocks % nblocks_per_core` remaining blocks. The function uses `compute_ncores(grid_area, nblocks)` which computes `ceil(nblocks / grid_area)` blocks per core.
