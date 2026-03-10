# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize operation converts tensor data from **TILE_LAYOUT** (32x32 tiles) to **ROW_MAJOR** layout. This analysis focuses on the multi-core interleaved program factory (`UntilizeMultiCoreProgramFactory`) with emphasis on the **output stage**: how the compute kernel produces row-major sticks in the output CB, how the writer kernel extracts those sticks, and how they are written to DRAM as row-major pages.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

This analysis serves as an **output_stage reference** for a new `layer_norm_rm` operation that follows the pattern: RM input -> tilize -> compute -> untilize -> RM output.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = all tiles across the width dimension at one tile-height) |
| **Unit size** | `num_tiles_per_input_block` tiles (= `tensor_width / tile_width`) |
| **Total units** | `num_tiles_per_col` blocks (= `tensor_height / tile_height`) |
| **Loop structure** | Outer: iterate blocks assigned to this core; Inner (compute): process one tile-row at a time; Inner (writer): iterate 32 stick-rows per block |

A "block" here is one tile-row: a horizontal strip of tiles spanning the full width, with height equal to `tile_height` (32). Each core processes multiple consecutive blocks (tile-rows).

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary, flattened to 2D (height x width) |
| **Dimension convention** | Last dim = width, everything else collapsed to height |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED (or SHARDED -- both supported, but this analysis focuses on interleaved) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED (default) or SHARDED (width/block sharded possible) |
| **Buffer type** | DRAM (interleaved) |
| **Data type** | Same as input |

### Layout Transformations

The compute kernel performs the tile-to-row-major conversion. After `pack_untilize_block`, data in the output CB is arranged with all tile-width elements for row 0 contiguous, then row 1, etc. -- i.e., the output CB contains `tile_height` (32) contiguous sticks, each of width `num_tiles_per_input_block * tile_width` elements.

**Key insight for output stage**: After untilize, the output CB holds one block's worth of row-major data. Each "stick" (row) in the CB has width = `num_cols_per_input_block` = `num_tiles_per_input_block * tile_width` columns. The writer reads these sticks and writes them as row-major pages to DRAM.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved tiles) | CB c_0 (input) | `cb_reserve_back`, `noc_async_read`, `cb_push_back` (1 tile at a time) |
| 2 | Compute | CB c_0 (input, tiled) | CB c_16 (output, row-major) | `cb_wait_front`/`cb_pop_front` on c_0; `cb_reserve_back`/`cb_push_back` on c_16 |
| 3 | Writer | CB c_16 (output, row-major sticks) | DRAM (interleaved RM pages) | `cb_wait_front`, `noc_async_write`, `cb_pop_front` |

### Detailed Output Stage Flow (Writer)

1. **Wait** for one block (`num_tiles_per_input_block` tiles) in output CB c_16.
2. Get the **base L1 read address** from `get_read_ptr(cb_id_out0)`.
3. For each of the `tile_height` (32) rows in the block:
   a. Compute the L1 read address for this row: `base_l1_read_addr + j * num_cols_per_input_block * output_element_size`.
   b. Compute the output page_id based on the row index and width-wise block position.
   c. Iterate through columns of the input block, writing sub-rows to possibly multiple output pages (handles width/block sharding where output pages may be narrower than input blocks).
   d. Use `TensorAccessor::get_noc_addr(page_id, offset_within_page)` to get the DRAM destination address.
   e. Call `noc_async_write(l1_addr, noc_addr, num_bytes)`.
4. **Barrier** with `noc_async_write_barrier()`.
5. **Pop** the block from CB c_16: `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`.
6. Repeat for next block.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_input_block * 2` (interleaved, multi-block) or `num_tiles_per_input_block` (single-block) | `num_tiles_per_input_block` | Double (multi-block) / Single (single-block) | Reader | Compute | Block |
| c_16 | cb_output | **Output row-major staging** | `num_tiles_per_input_block * 2` (multi-block) or `num_tiles_per_input_block` (single-block) | `num_tiles_per_input_block` | Double (multi-block) / Single (single-block) | Compute | Writer | Block |

### Output CB Sizing Details (Focus Area)

The output CB (c_16) capacity is determined by lines 149-163 of the program factory:

```
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

- **Block size** = `num_tiles_per_input_block` tiles = all tiles in one tile-row.
- **Capacity** = 1x or 2x block size depending on whether the core processes multiple blocks.
- **Physical size** = `output_cb_num_tiles * output_single_tile_size` bytes.
- **Page size** per tile = `tt::tile_size(output_cb_data_format)` (e.g., 2048 bytes for BF16 32x32).

**Critical detail**: Even though the CB is allocated in units of "tiles," the data format in the output CB is row-major after untilize. The `pack_untilize_block` function writes row-major data into the CB space. The `output_single_tile_size` is used for allocation, but the actual data layout within that space is 32 contiguous sticks of width `num_tiles_per_input_block * tile_width * element_size` bytes each.

### Stick Layout in the Output CB After Untilize

After `pack_untilize_block` processes one block of `num_tiles_per_input_block` tiles:

```
CB c_16 memory layout (one block):
  offset 0:                  row 0 of all tiles [num_cols_per_input_block elements]
  offset row_stride:         row 1 of all tiles [num_cols_per_input_block elements]
  ...
  offset (31 * row_stride):  row 31 of all tiles [num_cols_per_input_block elements]

where row_stride = num_cols_per_input_block * output_element_size
```

Row 0 contains: T0[0,0..31], T1[0,0..31], T2[0,0..31], ... (all tile columns concatenated).
This is exactly the row-major format needed for the output tensor.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Classification |
|----|----------|------------|-------|----------------|
| c_0 (input) | 2 blocks (multi-block) / 1 block (single-block) | 1 block | 2:1 / 1:1 | Double / Single buffered |
| c_16 (output) | 2 blocks (multi-block) / 1 block (single-block) | 1 block | 2:1 / 1:1 | Double / Single buffered |

When double-buffered, the compute kernel can process block N+1 while the writer writes block N, enabling compute-writer overlap.

## Index Calculations

### Writer Index Mapping (Output Stage Focus)

The writer maps from input block coordinates to output page addresses:

1. **Block height index** (`height_wise_input_block_start_index`): Which tile-row this core starts at. Set as runtime arg.
2. **Row within block** (`j`): 0 to `tile_height - 1` (0..31).
3. **Global row number**: `block_height_index * tile_height + j`.
4. **Output page_id calculation** (per row):
   ```
   num_rows_already_processed = block_height_index * tile_height + j
   num_pages_already_processed = num_rows_already_processed * num_output_blocks_across_width
   output_page_id = num_pages_already_processed + width_wise_output_block_start_index
   ```
5. **L1 read address** for each row:
   ```
   current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size
   ```

For **interleaved output** (the common case), `num_output_blocks_across_width = 1` and `output_page_width = tensor_width`, meaning each output page is one full row of the tensor. The page_id is simply the row index.

### TensorAccessor Usage

The writer uses `TensorAccessor` to convert page_id to a physical DRAM NOC address:

```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
// ...
uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
```

- `TensorAccessorArgs<8>()` starts reading compile-time args from index 8 (after the 8 explicit writer compile-time args).
- `output_stick_size` = `num_cols_per_output_block * output_element_size` = the page size for the output buffer.
- `get_noc_addr(page_id, offset)` maps the logical page_id to a physical DRAM bank address, adding the byte offset within the page.

For interleaved DRAM, the accessor computes:
1. `bank_index` = which DRAM bank holds this page (round-robin across banks).
2. `bank_offset` = byte offset within that bank.
3. Returns `noc_xy | addr` as 64-bit address for NOC transfer.

## Memory Access Patterns

### Read Pattern (De-emphasized, summary only)
- Reader reads tiles sequentially from DRAM, one tile at a time with barrier per tile.
- Tiles are read in row-major tile order (tile (0,0), tile (0,1), ... across width, then next tile-row).

### Write Pattern (Focus Area)

**Pattern**: Row-by-row sequential writes within each block, sequential blocks.

For each block (tile-row):
- 32 sequential writes, one per stick (row).
- Each write covers `output_stick_size` bytes = one full tensor row (for interleaved output).
- Writes go to sequential page_ids (consecutive rows in the output tensor).
- A write barrier (`noc_async_write_barrier()`) is issued after all 32 rows of a block.

**For interleaved output (simple case)**:
- `output_page_width = tensor_width`, `num_output_blocks_across_width = 1`.
- Each stick write is a single `noc_async_write` of the full row width.
- The inner `while` loop executes exactly once per row (no page splitting needed).

**For width-sharded / block-sharded output (complex case)**:
- Output pages may be narrower than input blocks.
- The inner `while` loop may iterate multiple times per row, writing to different output pages.
- The first output page in each row may have a byte offset (`output_offset_within_page_in_bytes`) if a previous core already wrote part of that page.

### Write Granularity

- **Minimum write**: `num_cols_to_write * output_element_size` bytes.
- **Maximum write**: `output_stick_size` bytes (full output page).
- **Barrier frequency**: Once per block (32 rows), not per row.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `num_compute_cores` (from `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` blocks (tile-rows) per full core |
| **Load balancing** | Nearly equal; one cliff core may have fewer blocks |

### Distribution Algorithm

The function `split_blocks_for_tilize(grid_size, num_tiles_per_col)` distributes `num_tiles_per_col` tile-rows across available cores:

1. `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`.
2. `ncores = ceil(num_tiles_per_col / nblocks_per_core)`.
3. `nblocks_per_core_cliff = num_tiles_per_col % nblocks_per_core` (may be 0).
4. If cliff is non-zero, the last core gets fewer blocks.

### Remainder Handling

- **Cliff core**: Gets `num_rows_per_cliff_core` blocks (< `num_rows_per_full_core`).
- Has its own compute kernel handle (`untilize_cliff_kernel_id`) with separate compile-time args.
- Only exists for interleaved input (sharded input has no cliff core).

## Arguments

### Compile-Time Arguments

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16 = 16) |
| 1 | `output_stick_size` | uint32_t | Bytes per output page: `num_cols_per_output_block * element_size` |
| 2 | `tile_height` | uint32_t | Tile height (32). Number of sticks per block. |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per block (= tensor_width / tile_width) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per row (1 for interleaved, >1 for width/block sharded) |
| 5 | `output_element_size` | uint32_t | Bytes per element (2 for BF16, 4 for FP32) |
| 6 | `num_cols_per_input_block` | uint32_t | Columns in input block: `num_tiles_per_input_block * tile_width` |
| 7 | `num_cols_per_output_block` | uint32_t | Columns per output page (= `output_page_width`) |
| 8+ | TensorAccessor args | varies | Auto-appended by `TensorAccessorArgs(*dst_buffer)` |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | = `num_tiles_per_input_block` (tiles per block width) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0 = 0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16 = 16) |

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_in0` | uint32_t | Input CB index (c_0 = 0) |
| 1+ | TensorAccessor args | varies | Auto-appended by `TensorAccessorArgs(*src0_buffer)` |

### Runtime Arguments

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of blocks (tile-rows) this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | First block index for this core |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Valid columns (excludes shard padding) |
| 4 | `width_wise_output_block_start_index` | uint32_t | First output page index in the width dimension |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset into first output page |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks (tile-rows) to process |

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Source buffer base address |
| 1 | `num_tiles` | uint32_t | Total tiles to read |
| 2 | `start_page_id` | uint32_t | First tile page_id for this core |

## Kernel Implementations

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_multi_core | RISCV_1 | NOC1 | CB c_16 (row-major sticks) | DRAM (interleaved RM pages) | `cb_wait_front`, `noc_async_write`, `noc_async_write_barrier`, `cb_pop_front` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - Uses a lambda `write_tiles_in_current_block` that processes one block at a time.
  - Waits for `num_tiles_per_input_block` tiles in the output CB (one full block).
  - Reads the base L1 pointer with `get_read_ptr(cb_id_out0)` -- this is the start of the row-major data.
  - Iterates 32 rows (`tile_height`), computing L1 read address as stride from the base.
  - For each row, computes the output `page_id` and optional byte offset.
  - Handles the case where one input block spans multiple output pages (width/block sharding).
  - Issues `noc_async_write` for each sub-row write.
  - A single `noc_async_write_barrier()` after all 32 rows of the block.
  - `cb_pop_front` to release the block from the output CB.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| pack_untilize_variable_num_blocks (fast) | RISCV_2 (unpack/math/pack) | N/A | CB c_0 (tiled) | CB c_16 (row-major) | `untilize` helper library |
| untilize_variable_num_blocks (slow) | RISCV_2 | N/A | CB c_0 (tiled) | CB c_16 (row-major) | `untilize` helper library |

- **File (fast)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
- **File (slow)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **Key Logic**:
  - Both call `compute_kernel_lib::untilize<block_width, src_cb, out_cb, ...>(num_blocks)`.
  - The unified `untilize` helper (from `untilize_helpers.inl`) automatically selects:
    - **Single-pass pack untilize**: When `block_width_tiles <= DEST_AUTO_LIMIT` (fits in DEST register).
    - **Block-based pack untilize**: When `block_width_tiles > DEST_AUTO_LIMIT`, splits into sub-blocks.
  - The "fast" variant includes `pack_untilize.h` and handles `DST_ACCUM_MODE` (for INT32/UINT32/FP32).
  - The "slow" variant is used when `use_pack_untilize` is false, for UINT16, or for wide FP32.
  - Path selection is in the program factory (lines 232-244).

### Reader Kernel (Summary)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM (interleaved tiles) | CB c_0 | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- Reads tiles one at a time using `TensorAccessor::get_noc_addr(page_id)`.

## Implementation Notes

### Output Stage Patterns Relevant for layer_norm_rm

1. **Output CB sizing**: Use `num_tiles_per_row` (tiles across width) as the block size. Double-buffer when the core processes multiple blocks. The output CB holds row-major data despite being sized in tile units.

2. **TensorAccessor for interleaved RM output**: Construct on host with `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)`. On device, create with `TensorAccessor(args, dst_addr, page_size)` where `page_size = stick_size = row_width * element_size`.

3. **Stick extraction from untilized CB**: After `pack_untilize_block`, data in the output CB is arranged as 32 contiguous sticks. The L1 address for row `j` is `base_l1_read_addr + j * num_cols_per_input_block * element_size`. This formula is the key to extracting RM sticks from the untilized output CB.

4. **Write pattern**: One `noc_async_write` per row (for interleaved output), barrier per block. The page_id is the global row index.

5. **Compute kernel integration**: The `compute_kernel_lib::untilize` helper handles all untilize complexity. It takes `block_width_tiles`, `input_cb`, `output_cb` as template parameters and `num_blocks` as a runtime argument. For a fused operation, use `WaitMode::WaitBlock` (default) or `WaitMode::NoWait` if synchronization is managed externally.

6. **CB c_16 convention**: The untilize operation uses CB index `c_16` (= `tt::CBIndex::c_16`) for the output, keeping it separate from the input CB `c_0`. For a fused kernel, this convention can be followed or adapted.

7. **Writer compile-time arg pattern**: The 8 explicit args (indices 0-7) define the output geometry. TensorAccessor args are appended starting at index 8. This is a clean, reusable pattern.

### Edge Cases

- **Uneven sharding width-wise**: `num_unpadded_cols_per_input_block` may be less than `num_cols_per_input_block` for the last shard in a row. The writer only writes valid columns, skipping garbage padding.
- **Uneven sharding height-wise**: `num_input_blocks_to_process` may be reduced for the last shard in a column.
- **FP32 + wide rows**: Falls back to the "slow" untilize path when `num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH`.
- **DST_ACCUM_MODE**: Enabled for INT32, UINT32, FLOAT32 data types via `compute_kernel_defines["DST_ACCUM_MODE"] = "1"`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize operation work in TTNN? What is the difference between pack_untilize and the standard untilize compute path?"
   **Reason**: Needed to understand the two compute paths and when each is selected.
   **Key Findings**: pack_untilize is the "fast" path using hardware-accelerated packing from DEST to output CB in row-major format. Standard untilize uses llk_unpack_untilize + datacopy + llk_pack. pack_untilize has significantly lower latency per tile.

2. **Query**: "How does split_blocks_for_tilize work in ttnn? How does it distribute tile rows across cores, and what is the cliff core concept?"
   **Reason**: Needed to understand core distribution strategy for the multi-core untilize.
   **Key Findings**: Divides total blocks evenly across cores. Cliff core handles remainder blocks. Returns BlockSplit struct with core ranges and block counts per core type.

3. **Query**: "How does TensorAccessor work on the device side for writing data? How does get_noc_addr(page_id, offset) work for interleaved output tensors?"
   **Reason**: Needed to understand how the writer kernel maps page_ids to DRAM addresses.
   **Key Findings**: For interleaved tensors, get_noc_addr computes bank_index (round-robin), bank_offset, and combines with NOC coordinates. Supports page_id + byte_offset variant for writing to middle of a page.

4. **Query**: "After pack_untilize converts tiles to row-major in the output CB, what is the exact memory layout of the data?"
   **Reason**: Critical for understanding how sticks are arranged in the output CB for the writer to read.
   **Key Findings**: After pack_untilize, rows of tiles across the width are contiguous. Row 0 of T0+T1+T2 is contiguous, then row 1, etc. The pack function increments by `full_ct_dim * page_size` to move between rows.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host-side setup and device-side API.
   **Key Information**: Host: `TensorAccessorArgs(buffer).append_to(compile_args)`. Device: `TensorAccessor(args, addr, page_size)` then `get_noc_addr(page_id, offset)`. All compile-time by default.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize compute helper library.
   **Key Information**: Template params: `block_width_tiles, input_cb, output_cb`. Runtime param: `num_blocks`. Automatically selects between single-pass and block-based pack untilize based on DEST limit. Supports WaitBlock, WaitUpfront, and NoWait synchronization modes.

3. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding how `split_blocks_for_tilize` distributes work across cores.
   **Key Information**: Returns `BlockSplit{ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff}`. Uses `ceil(nblocks/grid_area)` for blocks per core.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper function signature.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, optional_buffer)`. Returns `{cb_index, cb_handle}`. Total CB size = `num_pages * page_size`.
