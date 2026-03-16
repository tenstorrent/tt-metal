# Untilize Multi-Core Implementation Analysis (Output Stage Reference)

## Overview

The untilize operation converts tiled tensor data (TILE_LAYOUT) to row-major format (ROW_MAJOR_LAYOUT). This analysis focuses on the **output stage** patterns: how the writer kernel extracts row-major sticks from the output circular buffer and writes them to interleaved DRAM. This serves as a reference for implementing the output stage of a new `layer_norm_rm` operation that produces row-major interleaved output.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row of tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (= `tensor_width / tile_width` for interleaved) |
| **Total units** | `num_tiles_per_col` blocks (= `tensor_height / tile_height`) |
| **Loop structure** | Outer: blocks (tile-rows), Inner (writer): rows within a tile-height block |

One "input block" is a full row of tiles spanning the tensor width. Each block is `num_tiles_per_input_block` tiles wide and 1 tile tall (32 rows). The compute kernel untilizes one block at a time, producing `tile_height` (32) row-major sticks in the output CB. The writer then reads these sticks and writes them to DRAM.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] (any rank) | [N, C, H, W] (same shape) |
| **Dimension convention** | Last dim = width | Last dim = width |
| **Tensor layout** | TILE_LAYOUT | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED (focus of this analysis) |
| **Buffer type** | DRAM or L1 | DRAM (interleaved) |
| **Data type** | bfloat16 / float32 / int32 / uint32 | Same as input |

### Output Tensor Details (Key for output_stage reference)

For interleaved output:
- **Page definition**: One page = one full-width row of the tensor = `tensor_width * element_size` bytes
- **Page count**: `tensor_height` pages total (one per row)
- **Page distribution**: Round-robin across DRAM banks via TensorAccessor

### Layout Transformations

The compute kernel performs the tile-to-row-major conversion using `pack_untilize`. The input CB holds tiles in face-based tiled format (32x32 tiles split into 16x16 faces). The output CB holds the same data rearranged as contiguous row-major sticks. Each tile-row block of `N` tiles (covering `N*32` columns) is converted into `tile_height` (32) row-major sticks, each `N*32*element_size` bytes long.

## Data Flow Pattern

### Full Pipeline (for context)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved tiles) | CB c_0 (input) | reserve_back, push_back (1 tile at a time) |
| 2 | Compute | CB c_0 (input, tiled) | CB c_16 (output, RM) | wait_front, pop_front (block), reserve_back, push_back (block) |
| 3 | Writer | CB c_16 (output, RM sticks) | DRAM (interleaved RM) | wait_front, pop_front (block) |

### Output Stage Data Flow (Focus)

1. **Compute pushes a block**: `cb_push_back(output_cb, block_width_tiles)` -- announces `num_tiles_per_input_block` tiles worth of RM data in CB c_16
2. **Writer waits**: `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- waits for one block of RM sticks
3. **Writer reads sticks from L1**: Gets the base L1 read address via `get_read_ptr(cb_id_out0)`, then iterates through `tile_height` rows, computing each row's L1 address as `base + j * num_cols_per_input_block * output_element_size`
4. **Writer computes DRAM address**: Uses `TensorAccessor::get_noc_addr(output_page_id, offset_bytes)` to get the target NoC address for each output page (row)
5. **Writer issues NoC write**: `noc_async_write(l1_read_addr, dst_noc_addr, num_bytes_to_write)`
6. **Writer barriers and pops**: `noc_async_write_barrier()` then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tiles (tiled format) | 2 * num_tiles_per_row tiles (if >1 block/core) or 1 * num_tiles_per_row tiles | num_tiles_per_row tiles | Double (if >1 block) / Single (if 1 block) | Reader | Compute | Block |
| **c_16** | **cb_output** | **Output RM sticks** | **2 * num_tiles_per_row tiles (if >1 block/core) or 1 * num_tiles_per_row tiles** | **num_tiles_per_row tiles** | **Double (if >1 block) / Single (if 1 block)** | **Compute** | **Writer** | **Block** |

### Output CB Sizing (Key for output_stage reference)

The output CB (c_16) capacity is determined by the program factory (lines 148-162):

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;          // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;      // Double-buffered
}
```

**Sizing logic**: The CB holds one or two blocks of row-major sticks. Each block is `num_tiles_per_input_block` tiles worth of space. Even though the data in the output CB is row-major sticks, the **capacity is measured in tiles** because the CB page size is set to `output_single_tile_size` (tile_size for the output data format). This is important: the CB infrastructure still uses tile-based page sizes even for RM data in the untilize output CB.

The CB is created with:
```cpp
create_cb(tt::CBIndex::c_16, program, compute_core_range,
          output_single_tile_size, output_cb_num_tiles, output_cb_data_format);
```

Where `output_single_tile_size = tt::tile_size(output_cb_data_format)` is the byte size of one tile in the output data format.

**Key insight for layer_norm_rm**: The output CB uses tile-sized pages even though it holds RM data. The compute kernel's `pack_untilize` writes exactly `block_width_tiles` tile-slots worth of data per push, where the data content is `tile_height` contiguous RM sticks of width `block_width_tiles * tile_width * element_size` bytes.

## Pipeline Pattern Summary

- **Single-buffered** when each core processes exactly 1 block (1 tile-row): No overlap between compute and writer.
- **Double-buffered** when each core processes 2+ blocks: Compute can fill block N+1 while writer drains block N.

## Index Calculations

### Output Page ID Calculation (Writer)

The writer kernel maps each row-major stick to an output page ID in the DRAM-interleaved buffer. The mapping is:

```
output_page_id = num_rows_already_processed * num_output_blocks_across_width
                 + width_wise_output_block_start_index
```

Where:
- `num_rows_already_processed = block_height_index * tile_height + j` (j = row within block, 0..31)
- `num_output_blocks_across_width` = number of output pages per tensor row (1 for interleaved, >1 for width/block-sharded output)
- `width_wise_output_block_start_index` = starting page index for this core's width-wise position

**For simple interleaved output** (the common case for layer_norm_rm):
- `num_output_blocks_across_width = 1`
- `output_page_width = tensor_width` (one page = one full row)
- `output_page_id = block_height_index * tile_height + j` (simply the row index)

### L1 Read Address Calculation (Writer)

After `cb_wait_front`, the writer reads RM sticks from the output CB:

```
base_l1_read_addr = get_read_ptr(cb_id_out0)
row_l1_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size
```

This is a simple stride: each row in the CB is `num_cols_per_input_block * output_element_size` bytes apart.

### TensorAccessor Address Resolution

The writer calls `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` to get the NoC address. For interleaved output:
- The TensorAccessor maps `output_page_id` to the correct DRAM bank and offset via round-robin distribution
- `output_offset_within_page_in_bytes` is 0 for simple cases (writing the full page from the start)

## Memory Access Patterns

### Read Pattern (Reader - brief)

Tiles are read sequentially from DRAM one at a time using TensorAccessor. Each tile is read to the input CB.

### Write Pattern (Writer - KEY for output_stage reference)

The writer writes RM sticks to interleaved DRAM:

1. **Sequential within a block**: For each block, rows 0..31 (tile_height) are written in order
2. **One NoC write per row**: Each row is a single contiguous write of `output_stick_size` bytes
3. **Barrier per block**: `noc_async_write_barrier()` is called after all 32 rows of a block are written, before popping the CB
4. **Ascending page IDs**: Pages are written in order of ascending row index
5. **DRAM round-robin**: Pages land in different DRAM banks per the interleaved layout

**Write size**: `output_stick_size = output_page_width * output_element_size` bytes per NoC write (for interleaved: `tensor_width * element_size`)

**Pattern type**: Sequential row writes with a barrier per tile-height block. This is efficient because each write is a full page and sequential page IDs distribute well across DRAM banks.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` blocks (tile-rows) per full core |
| **Load balancing** | Equal blocks per core, optional cliff core with fewer blocks |

The `split_blocks_for_tilize(grid_size, num_tiles_per_col)` utility divides `num_tiles_per_col` tile-rows across available cores:
- Most cores get `nblocks_per_core` blocks
- One optional "cliff" core gets the remainder (`nblocks % nblocks_per_core`)
- Cores are enumerated in row-major order across the compute grid

## Arguments

### Compile-Time Arguments (Writer)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output CB index (c_16 = 16) |
| 1 | output_stick_size | uint32_t | Bytes per output row/stick (`output_page_width * element_size`) |
| 2 | tile_height | uint32_t | Rows per tile (32) |
| 3 | num_tiles_per_input_block | uint32_t | Tiles per block width (`tensor_width / tile_width`) |
| 4 | num_output_blocks_across_width | uint32_t | Output pages per tensor row (1 for interleaved) |
| 5 | output_element_size | uint32_t | Bytes per element (2 for bfloat16, 4 for float32) |
| 6 | num_cols_per_input_block | uint32_t | Columns per input block (`num_tiles_per_input_block * tile_width`) |
| 7 | num_cols_per_output_block | uint32_t | Columns per output page (`output_page_width`) |
| 8+ | TensorAccessorArgs | uint32_t[] | Auto-appended by `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` |

### Runtime Arguments (Writer)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |
| 1 | num_input_blocks_to_process | uint32_t | Number of tile-row blocks this core processes |
| 2 | height_wise_input_block_start_index | uint32_t | First block index (tile-row) for this core |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Columns to actually write (handles padding) |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page column index (0 for interleaved) |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Column offset within first output page (0 for interleaved) |

### Compile-Time Arguments (Compute)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Tiles per block width (= `num_tiles_per_input_block`) |
| 1 | src_cb_id | uint32_t | Input CB index (c_0 = 0) |
| 2 | out_cb_id | uint32_t | Output CB index (c_16 = 16) |

### Runtime Arguments (Compute)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) to process |

## Kernel Implementations

### Writer Kernel (PRIMARY FOCUS)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 (RM sticks) | DRAM (interleaved) | Write RM sticks row-by-row |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - Uses a lambda `write_tiles_in_current_block` that processes one block at a time
  - For each block: waits for `num_tiles_per_input_block` tiles, then iterates through `tile_height` rows
  - Each row is written via a while loop that handles potential splitting across output pages (for sharded output), but for interleaved output this loop executes once per row
  - Uses `get_read_ptr(cb_id_out0)` to get the L1 address of the RM sticks
  - Row stride in L1: `num_cols_per_input_block * output_element_size`
  - Issues `noc_async_write_barrier()` once per block (not per row) for efficiency
  - Then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)` to free the CB space

### Compute Kernel (Brief)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 (tiled) | CB c_16 (RM) | pack_untilize |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(src_cb_id, out_cb_id)` for hardware init
  - Calls `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt)`
  - The `untilize` template function (in `untilize_helpers.inl`) handles:
    - WaitBlock mode: waits for one block of input tiles before processing
    - `pack_untilize_block` converts tiles to RM sticks in the output CB
    - If `block_width_tiles > DEST_AUTO_LIMIT`, splits into sub-blocks
    - Push/pop semantics: `cb_reserve_back`/`cb_push_back` for output, `cb_wait_front`/`cb_pop_front` for input

### Reader Kernel (Brief)

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- Reads tiles one-by-one from interleaved DRAM to input CB c_0

## Implementation Notes

### Key Patterns for layer_norm_rm Output Stage

1. **Output CB uses tile-sized pages for RM data**: Even though the output contains row-major sticks, the CB is configured with `output_single_tile_size` as the page size and capacity measured in tiles. The `pack_untilize` compute function writes exactly `block_width_tiles` tile-slots per push. A new layer_norm_rm operation that outputs RM data would similarly configure its output CB with tile-based page sizes if using untilize.

2. **Writer reads from CB using `get_read_ptr` + stride**: The writer does not use tile-aware read APIs. It treats the CB as a flat byte buffer, computing row offsets manually: `base + j * row_stride`. This is possible because the data in the output CB is already in RM format after untilize.

3. **TensorAccessor for DRAM writes**: The pattern is:
   - Host: `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` -- appends address generation metadata as compile-time args
   - Device: `constexpr auto dst_args = TensorAccessorArgs<8>()` (8 = starting CT arg index) then `const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size)` -- creates accessor with page_size = stick_size
   - Write: `s.get_noc_addr(page_id, offset_bytes)` then `noc_async_write(...)`

4. **Barrier strategy**: One `noc_async_write_barrier()` per block (not per row). This amortizes barrier cost across `tile_height` (32) writes.

5. **CB pop after barrier**: The sequence is: issue all writes for a block -> barrier -> pop_front. This ensures all writes complete before freeing the CB space.

6. **Double-buffering decision**: Based on whether the core processes more than 1 block. For single-block cores, no overlap benefit exists so single-buffering saves L1 space.

### Relevance to layer_norm_rm

For a `layer_norm_rm` operation that takes RM input and produces RM output:
- The **writer kernel pattern** from untilize is directly applicable: read RM sticks from an output CB and write them to interleaved DRAM
- The **output CB sizing** approach (tile-based pages, double-buffering for multi-block) can be adapted
- The **TensorAccessor setup** for the output buffer follows the same host/device pattern
- Since layer_norm_rm works in RM throughout, there is no untilize compute step; instead the compute kernel would write RM results directly to the output CB
- The writer's row-by-row DRAM write pattern with per-block barriers is an efficient template to follow

### Simplifications for Interleaved-Only Output

When the output is always interleaved (as in layer_norm_rm's initial implementation):
- `num_output_blocks_across_width = 1`
- `output_page_width = tensor_width`
- `width_wise_output_block_start_index = 0`
- `num_cols_already_processed_in_first_output_block = 0`
- The while loop in the writer executes exactly once per row (no page splitting)
- `output_page_id` is simply the absolute row index

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is TensorAccessor and TensorAccessorArgs in tt-metal? How are they used in dataflow kernels to map logical tensor indices to physical NoC addresses?"
   **Reason**: Understanding how the writer kernel resolves page IDs to physical DRAM addresses is essential for the output stage.
   **Key Findings**: TensorAccessorArgs is configured on the host with buffer metadata and appended as compile-time args. On the device, TensorAccessor is constructed from these args plus a base address and page size. `get_noc_addr(page_id, offset)` maps logical page IDs to physical NoC addresses via round-robin bank distribution for interleaved tensors. The `append_to` method on the host appends args_config, rank, num_banks, tensor_shape, shard_shape, and bank_coords as compile-time arguments.

2. **Query**: "How does the untilize compute operation work? What is the data layout in the output CB after untilize?"
   **Reason**: Understanding what the writer kernel receives from compute is critical for the output stage analysis.
   **Key Findings**: The `pack_untilize` operation converts tiled data in the DEST register to row-major format in the output CB. After untilize, the output CB contains contiguous row-major sticks. For a block of N tiles wide, the output is `tile_height` sticks each `N * tile_width * element_size` bytes long, laid out sequentially in the CB.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding page definitions for RM vs tiled tensors, and interleaved memory distribution.
   **Key Information**: For RM layout, each row of the 2D tensor is one page. Interleaved layout distributes pages round-robin across DRAM banks. N-dimensional tensors are flattened to 2D by squeezing outer dimensions.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor device-side API and address calculation.
   **Key Information**: `TensorAccessor(args, base_addr, page_size)` construction; `get_noc_addr(page_id, offset)` for address resolution; host-side `TensorAccessorArgs(buffer).append_to(compile_args)` pattern. For interleaved tensors, the args_config is minimal (just the config flag); for sharded, it includes shape/bank metadata.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl`
   **Reason**: Understanding exact CB push/pop semantics of the compute kernel that feeds the writer.
   **Key Information**: Compute calls `cb_reserve_back(output_cb, block_width_tiles)` before processing and `cb_push_back(output_cb, block_width_tiles)` after. Each push makes `block_width_tiles` tile-slots of RM data available to the writer. WaitBlock mode waits for one input block at a time.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` utility used for output CB creation.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer*)` -- creates a CircularBuffer with total size `num_pages * page_size`, sets per-CB page size. If buffer is non-null, globally allocates the CB at the buffer's address (used for sharded input CB, not output).

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding core distribution and cliff core handling.
   **Key Information**: `split_blocks_for_tilize(grid_size, nblocks)` divides `nblocks` across `grid_size.x * grid_size.y` cores. Returns `nblocks_per_core` for full cores and `nblocks_per_core_cliff` for the cliff core. Cores are laid out row-major in the compute grid.
