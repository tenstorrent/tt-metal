# Untilize Multi-Core Implementation Analysis

## Overview

The **untilize** operation converts tensor data from tile layout (32x32 tiles organized in face format) to row-major layout (contiguous rows/"sticks"). This is the inverse of the tilize operation. The multi-core program factory distributes the untilization work across multiple Tensix cores, each processing a subset of tile rows.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Key architectural insight**: The untilize conversion happens in the **compute kernel** (not the writer). The compute kernel uses the hardware PACK unit to rearrange tile-formatted data into row-major format. The writer kernel then takes this already-untilized (row-major) data from the output CB and writes it to DRAM as sticks.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = all tiles across the width of the tensor for one tile-height worth of rows) |
| **Unit size** | `num_tiles_per_input_block` tiles (= `tensor_width / tile_width`, typically `tensor_width / 32`) |
| **Total units** | `num_tiles_per_col` blocks (= `tensor_height / tile_height`, typically `tensor_height / 32`) |
| **Loop structure** | Outer loop: blocks (tile-rows). Inner loop (in compute): untilize each block of tiles. Writer: iterate rows within each block. |

A single "work unit" is one **input block** -- a horizontal strip of tiles spanning the tensor width and one tile-height tall. For example, a 128x256 tensor in bfloat16 with 32x32 tiles has 4 tile-rows (blocks), each consisting of 8 tiles.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary N-D, flattened to 2D as `[tensor_height, tensor_width]` |
| **Dimension convention** | Last dim = width, all outer dims collapsed into height |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with face sub-structure) |
| **Memory layout** | INTERLEAVED or SHARDED (both supported) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 (all supported) |

When sharded, additional properties apply:
- **Shard Shape**: `[input_shard_height, input_shard_width]` -- height and width of each shard in elements
- **Core Grid**: From `input_shard_spec.grid`
- **Shard Orientation**: ROW_MAJOR or COL_MAJOR (from `shard_spec.orientation`)

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT (sticks/rows) |
| **Memory layout** | INTERLEAVED, WIDTH_SHARDED, or BLOCK_SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or may differ, e.g., fp32_dest_acc_en) |

For sharded output (WIDTH_SHARDED or BLOCK_SHARDED):
- `output_num_blocks_across_width` > 1, meaning each row is split across multiple output shards/pages
- `output_stick_size` = `tensor_width * element_size / output_num_blocks_across_width`

### Layout Transformations

The core transformation is: **TILE_LAYOUT -> ROW_MAJOR_LAYOUT**

- Input tiles are 32x32 with face sub-structure (4 faces of 16x16, stored face0->face1->face2->face3)
- The compute kernel rearranges this internal face ordering into contiguous row-major data
- The output CB contains row-major data where each "tile's worth" of data is `tile_height` sticks of `num_tiles_per_input_block * tile_width` elements each
- The writer extracts individual rows from this untilized data and writes them as pages to DRAM

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) or L1 (sharded) | CB c_0 (input) | `cb_reserve_back(c_0, 1)`, `noc_async_read`, `cb_push_back(c_0, 1)` -- one tile at a time |
| 2 | Compute | CB c_0 (input, tiled) | CB c_16 (output, row-major) | `cb_wait_front(c_0, block_width)`, `cb_reserve_back(c_16, block_width)`, untilize_block/pack_untilize_block, `cb_pop_front(c_0, block_width)`, `cb_push_back(c_16, block_width)` |
| 3 | Writer | CB c_16 (output, row-major) | DRAM | `cb_wait_front(c_16, block_width)`, row-by-row `noc_async_write`, `cb_pop_front(c_16, block_width)` |

### Detailed Data Flow

1. **Reader** reads tiles from DRAM one at a time using `TensorAccessor::get_noc_addr(page_id)`. For interleaved input, tiles are read sequentially starting from `start_page_id`. For sharded input, the reader simply does `cb_push_back` since data is already in L1 (CB is globally allocated to the shard buffer).

2. **Compute** waits for one block's worth of tiles (`num_tiles_per_input_block`), then performs the untilize operation. This converts the tile-ordered data (with face sub-structure) into row-major format. The output CB receives contiguous rows. Three dispatch paths exist:
   - **Pack untilize (fast path)**: Hardware-accelerated, used when `block_width_tiles <= DEST_limit` (typically 8 tiles for half-sync fp16)
   - **Block-based pack untilize**: For integer types with wide rows exceeding DEST, splits into sub-blocks
   - **Standard untilize (slow path)**: Software fallback for wide float rows or UINT16 or FLOAT32 with `num_tiles_per_input_block >= 8`

3. **Writer** waits for one block of untilized data, then iterates row-by-row within that block. For each of the `tile_height` (32) rows, it reads from L1 and writes to the appropriate DRAM page using `TensorAccessor::get_noc_addr(output_page_id, offset)`. The writer handles the case where one input block may span multiple output pages (sharded output) or where multiple input blocks contribute to a single output page.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging (tiled data) | See below | `num_tiles_per_input_block` (one tile-row) | Single or Double (see below) | Reader | Compute | Block |
| c_16 | cb_output | Output staging (row-major data) | See below | `num_tiles_per_input_block` (one tile-row) | Single or Double (see below) | Compute | Writer | Block |

### CB c_0 (Input) Capacity Rules

- **Sharded input**: Capacity = `num_tiles_per_input_block * num_input_blocks_per_full_core` (entire shard). Single-buffered but the entire shard is loaded at once. CB is globally allocated to the shard buffer.
- **Interleaved, 1 block per core**: Capacity = `num_tiles_per_input_block`. Single-buffered (no overlap needed).
- **Interleaved, 2+ blocks per core**: Capacity = `num_tiles_per_input_block * 2`. **Double-buffered** -- reader can fill one block while compute processes the previous one.

### CB c_16 (Output) Capacity Rules

- **1 block per core**: Capacity = `num_tiles_per_input_block`. Single-buffered.
- **2+ blocks per core**: Capacity = `num_tiles_per_input_block * 2`. **Double-buffered** -- compute can fill one block while writer drains the previous one.

### Page Size

- Input CB page size: `tile_size(input_cb_data_format)` -- one tile in the input data format
- Output CB page size: `tile_size(output_cb_data_format)` -- one tile in the output data format (note: even though the data is now row-major, the CB page size is still tile-sized because the compute kernel produces tile-sized chunks)

## Pipeline Pattern Summary

| Condition | CB c_0 | CB c_16 | Pipeline |
|-----------|--------|---------|----------|
| 1 block per core | Single-buffered | Single-buffered | Sequential: Read -> Compute -> Write |
| 2+ blocks per core (interleaved) | Double-buffered | Double-buffered | Pipelined: Read block N+1 overlaps with Compute block N; Compute block N+1 overlaps with Write block N |
| Sharded input | Entire shard (single load) | Single or Double | Compute starts after reader push; writer overlaps if double-buffered |

## Index Calculations

### Reader Index Mapping (Interleaved)

The reader uses `TensorAccessor` initialized with compile-time args at offset 1 (after `cb_id_in0`):
```
constexpr auto src_args = TensorAccessorArgs<1>();
const auto s = TensorAccessor(src_args, src_addr, tile_bytes);
```
Tiles are read sequentially: `page_id` goes from `start_page_id` to `start_page_id + num_tiles`. The `TensorAccessor::get_noc_addr(page_id)` maps the linear tile ID to a physical DRAM bank address using the interleaved round-robin page-to-bank mapping.

### Writer Index Mapping

The writer uses `TensorAccessor` initialized at compile-time arg offset 8:
```
constexpr auto dst_args = TensorAccessorArgs<8>();
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

Output pages are **sticks** (rows), not tiles. The page size is `output_stick_size = tensor_width * element_size / output_num_blocks_across_width`.

The writer computes the output page ID for each row:
```
output_page_id = (block_height_index * tile_height + j) * num_output_blocks_across_width
                 + width_wise_output_block_start_index
```
Where:
- `block_height_index`: Which tile-row this block is (globally)
- `j`: Row index within the current tile-height (0 to `tile_height - 1`)
- `num_output_blocks_across_width`: How many output pages per row (1 for non-sharded output)
- `width_wise_output_block_start_index`: Starting page offset for this core's width position

The writer also handles an offset within a page: `output_offset_within_page_in_bytes = num_cols_already_processed_in_first_output_block * output_element_size`. This allows multiple input blocks (from different cores processing different width shards) to write to different portions of the same output page.

## Memory Access Patterns

### Read Pattern

**Interleaved input**: Sequential tile reads. The reader reads tiles one at a time from `start_page_id` to `start_page_id + num_tiles - 1`. Each tile read uses `noc_async_read` with an immediate barrier (`noc_async_read_barrier` after each tile). Tiles are distributed across DRAM banks in round-robin fashion.

**Sharded input**: No explicit reads. The CB is globally allocated to the shard buffer in L1. The reader kernel simply calls `cb_push_back` to make the data available to compute.

### Write Pattern

**Row-by-row writes**: After the compute kernel produces one untilized block (one tile-row's worth of row-major data), the writer iterates through `tile_height` rows. For each row, it writes the stick data to the output buffer using `noc_async_write`. The write is barrier-synchronized per block (`noc_async_write_barrier` after all rows in a block).

Within each row, the writer may issue multiple writes if the input block spans multiple output pages (sharded output). It tracks `num_input_cols_processed` and writes chunks that fit within output page boundaries.

The L1 read address calculation for each row within the untilized block is:
```
current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size
```
This reflects that after untilization, data in the CB is stored as contiguous rows of `num_cols_per_input_block` elements.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` (from `split_blocks_for_tilize`) |
| **Work per core (full)** | `num_rows_per_full_core` tile-rows (blocks) |
| **Work per core (cliff)** | `num_rows_per_cliff_core` tile-rows (remainder) |
| **Load balancing** | Near-equal distribution with at most 1 cliff core |

### Interleaved Input

Uses `split_blocks_for_tilize(grid_size, num_tiles_per_col)`:
- Computes `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- Full cores: Each processes `num_rows_per_full_core` blocks
- Cliff core (0 or 1): Processes `num_rows_per_cliff_core = num_tiles_per_col % nblocks_per_core` blocks
- Cores are filled row-major across the grid
- `tile_start_index` accumulates across cores: each core starts reading where the previous one left off

### Sharded Input

Uses the shard spec's core grid directly:
- `num_compute_cores = input_shard_spec.grid.num_cores()`
- Each core processes its own shard: `num_input_blocks_per_full_core = input_shard_height / tile_height`
- `num_input_blocks_across_width = ceil(tensor_width / input_shard_width)` -- supports 2D sharding
- No cliff core (but handles uneven shards at edges via `num_input_blocks_to_process` and `num_unpadded_cols_per_input_block` adjustments)
- Core traversal order follows shard orientation (ROW_MAJOR or COL_MAJOR)

## Arguments

### Compile-Time Arguments

#### Reader Kernel (Interleaved: `reader_unary_start_id.cpp`)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |
| 1+ | TensorAccessorArgs | uint32_t[] | Source buffer accessor parameters (bank layout, rank, shapes) |

#### Reader Kernel (Sharded: `reader_unary_sharded.cpp`)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |

#### Writer Kernel (`writer_unary_stick_layout_split_rows_multi_core.cpp`)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer ID (c_16) |
| 1 | output_stick_size | uint32_t | Size in bytes of one output row/page |
| 2 | tile_height | uint32_t | Height of one tile (32 for standard tiles) |
| 3 | num_tiles_per_input_block | uint32_t | Number of tiles across the width in one block |
| 4 | num_output_blocks_across_width | uint32_t | Number of output pages per row (1 for non-sharded output) |
| 5 | output_element_size | uint32_t | Size in bytes of one output element |
| 6 | num_cols_per_input_block | uint32_t | Number of columns (elements) in one input block |
| 7 | num_cols_per_output_block | uint32_t | Number of columns per output page |
| 8+ | TensorAccessorArgs | uint32_t[] | Destination buffer accessor parameters |

#### Compute Kernel (both variants)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Number of tiles per block (= `num_tiles_per_input_block`) |
| 1 | src_cb_id | uint32_t | Input circular buffer ID (c_0) |
| 2 | out_cb_id | uint32_t | Output circular buffer ID (c_16) |

#### Compute Kernel Defines

| Define | Condition | Effect |
|--------|-----------|--------|
| `DST_ACCUM_MODE` | INT32, UINT32, or FLOAT32 data types | Sets DEST accumulator to 32-bit mode (halves capacity) |

### Runtime Arguments

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM base address |
| 1 | num_tiles_to_read | uint32_t | Total tiles for this core (`num_tiles_per_input_block * num_input_blocks_to_process`) |
| 2 | tile_start_index | uint32_t | First tile ID for this core |

#### Reader Kernel (Sharded)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_to_read | uint32_t | Total tiles in the shard for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM base address |
| 1 | num_input_blocks_to_process | uint32_t | Number of tile-row blocks this core processes |
| 2 | height_wise_input_block_start_index | uint32_t | Global tile-row index where this core starts |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Number of valid (non-padding) columns to write per block |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page index for width position |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Column offset within the first output page |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) to process on this core |

## Kernel Implementations

### Reader Kernel (Interleaved)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM (src buffer) | CB c_0 | Read tiles sequentially |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **Key Logic**: Simple sequential tile reader. Uses `TensorAccessor` to resolve tile IDs to NOC addresses. Reads one tile at a time with `noc_async_read` + immediate barrier. Iterates from `start_page_id` to `start_page_id + num_tiles`.

### Reader Kernel (Sharded)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_sharded | RISCV_0 | N/A | L1 (shard) | CB c_0 | Push shard data to CB |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`
- **Key Logic**: Since the CB is globally allocated to the shard buffer, no data movement is needed. The kernel simply calls `cb_push_back(cb_id_in0, num_tiles_per_core)` to make all shard tiles visible to the compute kernel.

### Compute Kernel (Pack Untilize - Fast Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| pack_untilize_variable_num_blocks | Compute (TRISC) | N/A | CB c_0 (tiled) | CB c_16 (row-major) | Pack untilize |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
- **Key Logic**: Calls `compute_kernel_hw_startup(src_cb_id, out_cb_id)` for hardware initialization, then delegates to `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt)`. The unified untilize function auto-selects the optimal dispatch path based on DEST capacity and data format. For narrow widths (typical case), uses `pack_untilize_block` which is hardware-accelerated. Adjusts `max_bct` based on `DST_ACCUM_MODE` (4 for 32-bit, 8 for 16-bit).

### Compute Kernel (Standard Untilize - Slow Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| untilize_variable_num_blocks | Compute (TRISC) | N/A | CB c_0 (tiled) | CB c_16 (row-major) | Standard untilize |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **Key Logic**: Same structure as the fast path. Uses `compute_kernel_lib::untilize<>()` which falls back to `untilize_block()` when the row is too wide for pack_untilize. This path involves all three TRISC threads (UNPACK, MATH, PACK) instead of just PACK.
- **Activation condition**: Used when `use_pack_untilize` is false, or dtype is UINT16, or (FLOAT32 and `num_tiles_per_input_block >= 8`)

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_multi_core | RISCV_1 | NOC1 | CB c_16 (row-major) | DRAM (dst buffer) | Write sticks |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**: This is the most complex kernel. For each block:
  1. Waits for `num_tiles_per_input_block` tiles of untilized data in CB c_16
  2. Computes `base_l1_read_addr` from `get_read_ptr(cb_id_out0)`
  3. For each of `tile_height` rows within the block:
     - Computes the output page ID based on global row index and width-wise position
     - Handles partial writes when input blocks don't align with output pages (sharded output)
     - Uses a while loop to write chunks of the row, advancing across output pages as needed
     - Supports `output_offset_within_page_in_bytes` for mid-page writes
  4. Issues `noc_async_write_barrier()` per block
  5. Pops the block from CB c_16

  The writer is designed to be general: it handles the case where the input block's columns span multiple output pages and where the first output page may already be partially written by a different core (width-sharded input).

## Implementation Notes

### Kernel Selection Logic (Host Side)

The program factory selects between two compute kernels at lines 196-209:
```cpp
if (!use_pack_untilize || a.dtype() == DataType::UINT16 ||
    (a.dtype() == DataType::FLOAT32 && num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH)) {
    // "slow untilize" -- standard path
} else {
    // "fast pack untilize" -- hardware-accelerated path
}
```
`MAX_PACK_UNTILIZE_WIDTH` is 8 (defined in `ttnn/api/ttnn/common/constants.hpp`). Both kernels now use the unified `compute_kernel_lib::untilize<>()` helper which handles dispatch internally, but the host still selects different source files for the two paths.

### FP32 DEST Accumulation

When `fp32_dest_acc_en` is true, the compute kernel unpacks to FP32 DEST registers. This halves DEST capacity (from 8 to 4 tiles in half-sync mode). This is enabled for INT32, UINT32, and FLOAT32 data types, and controlled via the `DST_ACCUM_MODE` define and `UnpackToDestMode::UnpackToDestFp32`.

### Uneven Shard Handling

For sharded input, the last shard in each row/column may be smaller than the full shard size. The program factory computes:
- `num_unpadded_cols_per_input_block`: For the last width-wise shard, this is reduced by the padding amount
- `num_input_blocks_to_process`: For the last height-wise shard, this is reduced to exclude padded tile-rows

These are passed as runtime args so the writer knows how much valid data to write.

### Program Caching and Override

The factory returns a `cached_program_t` with shared variables containing kernel handles and core coordinates. The `override_runtime_arguments` method allows efficient re-invocation: only buffer addresses are updated, not the full program setup. For interleaved input, the reader's `src_addr` is updated; for sharded input, the CB's global address is updated. The writer's `dst_addr` is always updated.

### Writer Stick Extraction Pattern

After untilization, data in CB c_16 is arranged as:
```
Row 0: [col_0, col_1, ..., col_{W-1}]  (W = num_cols_per_input_block)
Row 1: [col_0, col_1, ..., col_{W-1}]
...
Row 31: [col_0, col_1, ..., col_{W-1}]
```

The writer reads each row at offset `j * num_cols_per_input_block * output_element_size` from the CB base. This is the key insight: after untilize, the CB holds `tile_height` contiguous rows, each of `num_cols_per_input_block` elements. The writer then slices these rows and writes them to the appropriate DRAM locations.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize operation work in tt-metal? What is the difference between pack_untilize and standard untilize?"
   **Reason**: Needed to understand the two untilize dispatch paths and which hardware units are involved.
   **Key Findings**: Standard untilize uses all three TRISC threads (UNPACK -> MATH -> PACK), while pack_untilize only uses the PACK thread and is hardware-accelerated. Pack untilize is preferred but has width limitations (DEST register capacity). The choice is made based on data type and tile-row width.

2. **Query**: "How does TensorAccessor and TensorAccessorArgs work in tt-metal kernels?"
   **Reason**: Both reader and writer kernels use TensorAccessor for NOC address computation.
   **Key Findings**: TensorAccessorArgs is configured on the host and appended to compile-time args. On device, TensorAccessor is constructed from these args plus a base address and page size. `get_noc_addr(page_id)` maps logical page IDs to physical DRAM bank addresses, abstracting the interleaved round-robin layout. `get_noc_addr(page_id, offset)` adds a byte offset within a page.

3. **Query**: "What is a 'stick' in tt-metal tensor layouts? When untilizing, how is data organized as sticks versus tiles?"
   **Reason**: Needed to understand the output format and how the writer kernel handles row-major data.
   **Key Findings**: A "stick" is a single row of a tensor in row-major layout. In row-major layout, each row = one page. After untilizing, data is contiguous rows. The writer extracts individual rows from the CB and writes them as pages to DRAM.

4. **Query**: "What does the split_blocks_for_tilize function do in tt-metal?"
   **Reason**: The program factory uses this function for core distribution.
   **Key Findings**: `split_blocks_for_tilize` divides `nblocks` work units across available cores. Returns full core range (equal work) and cliff core range (remainder). Uses `ceil(nblocks / grid_area)` blocks per core. At most one cliff core handles the remainder.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tile vs row-major layout, page definitions, and interleaved memory layout.
   **Key Information**: Row-major: each row = one page. Tiled: 32x32 tiles with 16x16 face sub-structure. Interleaved: pages distributed round-robin across banks.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize compute helper used by both kernel variants.
   **Key Information**: Three dispatch paths: (1) pack_untilize for narrow widths, (2) block-based pack_untilize for wide integer types, (3) standard untilize for wide non-integer types. DEST capacity auto-detected from JIT headers.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity detection.
   **Key Information**: DEST capacity depends on sync mode (Full/Half) and accumulation mode (16-bit/32-bit). Half-sync + fp16 = 8 tiles, Half-sync + fp32 = 4 tiles.

4. **Source**: `tt_metal/include/compute_kernel_api/compute_kernel_hw_startup.h`
   **Reason**: Understanding what hardware initialization the compute kernel performs.
   **Key Information**: Configures UNPACK, MATH, and PACK hardware units. Must be called once at kernel start before any compute operations. Sets up data formats and sync modes.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in the program factory.
   **Key Information**: Convenience wrapper that creates `CircularBufferConfig` with specified page size, num pages, and data format. Optionally sets globally allocated address for sharded buffers.

6. **Source**: `ttnn/api/ttnn/common/constants.hpp`
   **Reason**: Understanding `MAX_PACK_UNTILIZE_WIDTH` constant.
   **Key Information**: `MAX_PACK_UNTILIZE_WIDTH = 8` -- pack untilize does not support > 8 tiles width for FLOAT32 on the host dispatch path.
