# Untilize (Single Core) Implementation Analysis

## Overview

The untilize operation converts tensor data from **TILE_LAYOUT** (32x32 tiles with face-based internal ordering) to **ROW_MAJOR_LAYOUT** (contiguous row sticks). This analysis focuses on the **output stage** -- how the compute kernel produces row-major data into the output CB, and how the writer kernel extracts individual row-major sticks and writes them to DRAM. This serves as a reference for a new rms_norm operation that needs in-kernel untilize as its output stage.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_single_core_program_factory.cpp`

**Focus**: Writer kernel pattern, output CB sizing, stick extraction from tiles.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a horizontal strip of tiles) |
| **Unit size** | `num_tiles_per_block` tiles (a divisor of tiles-per-row that fits in L1) |
| **Total units** | `num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height` |
| **Loop structure** | Outer: tile-rows (height); Middle: column groups (sharding); Inner: blocks per row |

One "block" is a contiguous horizontal strip of `num_tiles_per_block` tiles along one tile-row. The compute kernel untilizes one block at a time, producing `tile_height` row-major sticks of width `num_tiles_per_block * TILE_WIDTH` elements each. The writer then writes those sticks to DRAM.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, ..., H, W] (arbitrary rank) | [N, ..., H, W] (same shape) |
| **Dimension convention** | Last dim is W (width) | Last dim is W (width) |
| **Tensor layout** | TILE_LAYOUT | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED (or sharded) | INTERLEAVED (or WIDTH/BLOCK/ND_SHARDED) |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | bfloat16, float32, int32, uint32 | Same as input |

### Layout Transformations

The core transformation is **tile-to-row-major** (untilize). Inside each tile, data is stored in face order (4 faces of 16x16 per 32x32 tile). The `pack_untilize_block` LLK function extracts the data from the DEST register and packs it into the output CB in row-major order: `tile_height` contiguous rows, each `block_width_tiles * TILE_WIDTH * element_size` bytes wide.

**Key insight for rms_norm**: After compute produces normalized tiled results, the untilize step converts them to row-major sticks. The output CB must hold `num_tiles_per_block` tiles worth of row-major data (same total bytes as tiled, just reordered).

## Data Flow Pattern

This section emphasizes the **output side** of the pipeline (compute -> output CB -> writer -> DRAM).

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM | CB c_0 (input) | reserve_back, push_back |
| 2 | Compute | CB c_0 | CB c_16 (output) | wait_front(input), pop_front(input), reserve_back(output), push_back(output) |
| 3 | Writer | CB c_16 | DRAM | wait_front, pop_front |

### Detailed Output Flow (Compute to DRAM)

1. **Compute** calls `cb_reserve_back(output_cb, block_width_tiles)` to reserve space in CB c_16.
2. **Compute** calls `pack_untilize_block<block_width_tiles>(input_cb, 1, output_cb, 0)` which:
   - Unpacks tiles from input CB into DEST register
   - Rearranges tile face data into row-major order
   - Packs result into output CB
3. **Compute** calls `cb_push_back(output_cb, block_width_tiles)` to make data available to the writer.
4. **Writer** calls `cb_wait_front(output_cb, num_tiles_per_output_block)` to wait for data.
5. **Writer** reads `tile_height` row-major sticks from L1 (sequential, stride = `output_single_block_width_size`).
6. **Writer** issues `noc_async_write` for each stick to its DRAM destination address.
7. **Writer** calls `noc_async_write_barrier()` then `cb_pop_front(output_cb, num_tiles_per_output_block)`.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_output | Output row-major staging | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Compute | Writer | Block |

### Output CB Sizing Details (Critical for rms_norm reuse)

The output CB capacity is calculated as follows in the program factory:

```
max_l1_size = (l1_size_per_core / 2) - allocator_base_addr
max_tiles_per_cb = max_l1_size / (input_single_tile_size + output_single_tile_size)
num_tiles_per_block = largest divisor of num_tiles_per_column_row that is <= max_tiles_per_cb
output_cb_num_tiles = num_tiles_per_block
```

Key observations:
- Input and output CBs are sized **identically** in tile count (`num_tiles_per_block`).
- The capacity split assumes L1 is divided equally between the two CBs.
- Both CBs are **single-buffered** (capacity == block size). This means the pipeline is strictly sequential: compute waits for reader, writer waits for compute. No overlap between stages for the same block.
- The `output_single_tile_size` is `tt::tile_size(output_cb_data_format)`, which for bfloat16 with 32x32 tiles is 2048 bytes (32 * 32 * 2).

**For rms_norm**: The output CB should be sized to hold one block of untilized data. If the rms_norm operation adds intermediate CBs (for squared values, mean, etc.), the L1 budget calculation must account for all CBs, not just input and output.

## Pipeline Pattern Summary

Both CBs use single-buffering (capacity equals block size). This creates a strictly serialized pipeline per block: reader fills input CB, then compute processes it and fills output CB, then writer drains output CB. There is no overlapping of reader/compute/writer for the *same* block, but since the CBs are circular, the next block's reader can begin as soon as compute finishes reading the current block.

## Index Calculations

### Reader Side (brief, de-emphasized)
The reader uses `TensorAccessor` with tile-based page IDs (0 to `num_tiles - 1`). Each page ID maps to one tile. TensorAccessor internally handles the bank interleaving.

### Writer Side (primary focus)

The writer uses `TensorAccessor` with **stick-based page IDs**. Each stick is one row of the output tensor (width = `output_stick_size` bytes). The stick ID calculation in the writer kernel is:

```cpp
// For each tile-row i, each column-group j, each row-within-tile k:
uint32_t num_complete_rows_already_processed = (i * tile_height + k) * num_output_columns_of_blocks;
uint32_t stick_id = num_complete_rows_already_processed + j;
base_dst_noc_addr[k] = s.get_noc_addr(stick_id);
```

This maps `(tile_row_index, sub_row_within_tile, column_group)` to a linear stick ID. The TensorAccessor then converts this stick ID to a physical NoC address, handling bank interleaving automatically.

**For rms_norm**: When writing row-major output, the page size passed to TensorAccessor must be the full stick width (`padded_shape[-1] * element_size`), and page IDs correspond to rows of the output tensor. The stick_id formula handles the decomposition of 2D position into linear row indices.

### Output CB L1 Address Calculation

Within the writer kernel, the L1 read address for each stick is computed as:

```cpp
uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
// For each of tile_height rows:
//   write output_single_block_width_size bytes from l1_read_addr
//   l1_read_addr += output_single_block_width_size
```

This means the output CB data is laid out as `tile_height` contiguous rows, each `output_single_block_width_size` bytes wide (= `num_tiles_per_block * TILE_WIDTH * element_size`). The writer reads them sequentially.

## Memory Access Patterns

### Read Pattern (de-emphasized)
Sequential tile reads from DRAM. One tile per iteration, barrier after each read.

### Write Pattern (primary focus)

For each block of untilized data in the output CB:
1. **Stick-by-stick writes**: The writer iterates over `tile_height` sticks (rows 0..31 within a tile row).
2. **Each write**: `noc_async_write(l1_read_addr, dst_noc_addr, output_single_block_width_size)` -- writes one horizontal strip of the block.
3. **L1 reads are sequential**: stride = `output_single_block_width_size`, no gaps.
4. **DRAM writes may be non-contiguous**: Each stick goes to a different DRAM bank/offset determined by TensorAccessor. With interleaved memory, consecutive sticks round-robin across banks.
5. **Barrier per block**: `noc_async_write_barrier()` is called once per block (after all `tile_height` sticks are issued), not per stick. This batches the barrier cost.

**Key pattern for rms_norm**: Write `tile_height` sticks per block, each `block_width * element_size` bytes. Issue all writes, then barrier once. This amortizes the barrier overhead across multiple NoC transactions.

### Address Progression in Writer

The writer pre-computes `base_dst_noc_addr[tile_height]` for all rows in the current tile-row before entering the inner block loop. As blocks within the same row are processed, `base_dst_noc_addr[k]` is incremented by `output_single_block_width_size` to advance horizontally across the output row.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core (1D) |
| **Grid dimensions** | 1 x 1 |
| **Total cores** | 1 |
| **Work per core** | All tiles |
| **Load balancing** | N/A (single core) |

This is the single-core variant. Multi-core variants exist but are not analyzed here.

## Arguments

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer index (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output row in bytes (`padded_width * element_size`) |
| 2 | tile_height | uint32_t | Height of a tile (32 for standard tiles) |
| 3 | num_blocks_across_height | uint32_t | Number of tile-rows in the tensor |
| 4 | num_output_columns_of_blocks | uint32_t | Number of column groups (1 for interleaved, >1 for width/block sharded) |
| 5 | num_blocks_per_output_column_row | uint32_t | Number of blocks per row within a column group |
| 6 | num_tiles_per_output_block | uint32_t | Tiles per block (determines CB wait/pop granularity) |
| 7 | output_single_block_width_size | uint32_t | Width of one block in bytes (`num_tiles_per_block * TILE_WIDTH * element_size`) |
| 8+ | TensorAccessorArgs | varies | Compile-time args for output TensorAccessor (bank layout, shapes, coordinates) |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output buffer in DRAM/L1 |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_blocks | uint32_t | Total number of blocks to untilize (`cols_of_blocks * blocks_per_col_row * blocks_across_height`) |
| 1 | num_tiles_per_block | uint32_t | Tiles in each block (width of block in tiles) |
| 2 | src_cb_id | uint32_t | Input CB index (c_0) |
| 3 | out_cb_id | uint32_t | Output CB index (c_16) |

### Reader Compile-Time Arguments (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input CB index (c_0) |
| 1+ | TensorAccessorArgs | varies | Compile-time args for input TensorAccessor |

### Reader Runtime Arguments (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_tiles | uint32_t | Total number of tiles to read |
| 2 | start_page_id | uint32_t | Starting tile page ID (always 0 for single core) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (tiled pages) | CB c_0 | Read tiles sequentially |
| compute | RISCV_2 | N/A | CB c_0 | CB c_16 | pack_untilize_block (tile->RM conversion) |
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM (RM sticks) | Write row-major sticks |

### Writer Kernel: `writer_unary_stick_layout_split_rows_single_core.cpp`

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_single_core.cpp`

**Key Logic**:

The writer has a nested loop structure:
```
for i in 0..num_blocks_across_height:       // each tile-row
  for j in 0..num_output_columns_of_blocks: // column groups (sharding)
    precompute base_dst_noc_addr[0..tile_height-1] using TensorAccessor
    for k in 0..num_blocks_per_column_row:  // blocks within row
      write_tiles_in_current_block()
```

The `write_tiles_in_current_block()` lambda:
1. `cb_wait_front(cb_id_out0, num_tiles_per_output_block)` -- wait for compute to produce a block
2. Gets L1 read pointer via `get_read_ptr(cb_id_out0)`
3. For each of `tile_height` rows:
   - `noc_async_write(l1_read_addr, base_dst_noc_addr[l], output_single_block_width_size)`
   - Advances L1 pointer and DRAM pointer by `output_single_block_width_size`
4. `noc_async_write_barrier()` -- wait for all writes to complete
5. `cb_pop_front(cb_id_out0, num_tiles_per_output_block)` -- release CB space

**"Split rows" naming**: The name indicates that each tile's worth of output is split into individual row sticks before writing, rather than writing the entire tile as a contiguous block. This is necessary because the row-major output layout requires each row to map to a separate page (stick) in the output buffer's bank interleaving scheme.

### Compute Kernel: `untilize.cpp`

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp`

**Key Logic**:
- Uses `compute_kernel_lib::untilize<block_width_tiles, src_cb_id, out_cb_id>` from the unified helper library.
- Configuration: `InitAndUninit` mode (single standalone call), `WaitBlock` mode (waits for input per block).
- For blocks wider than DEST register capacity, automatically splits into sub-blocks.
- The helper handles `cb_wait_front` on input, `cb_reserve_back`/`cb_push_back` on output, and `cb_pop_front` on input.

### Untilize Helper Library Signature

```cpp
template <
    uint32_t block_width_tiles,       // tiles per row-block (compile-time)
    uint32_t input_cb,                // input CB index
    uint32_t output_cb,               // output CB index
    InitUninitMode init_uninit_mode,  // init/uninit lifecycle
    WaitMode wait_mode,               // input sync strategy
    ReconfigureRegisterDatatypeMode reconfig_mode  // register reconfig
>
void untilize(uint32_t num_blocks);   // runtime: number of blocks
```

**For rms_norm usage**: The untilize helper can be called as a final step in the compute kernel after normalization is complete. Use `InitOnly`/`Neither`/`UninitOnly` modes if combining with other compute operations (e.g., the rms_norm math) to avoid redundant init/uninit between operations.

## Implementation Notes

### Output CB Data Layout After Untilize

After `pack_untilize_block` writes to the output CB, the data is organized as:
- `tile_height` (32) consecutive rows
- Each row is `num_tiles_per_block * TILE_WIDTH` (e.g., 4 tiles * 32 = 128) elements wide
- Total size = `tile_height * num_tiles_per_block * TILE_WIDTH * element_size`
- This equals `num_tiles_per_block * tile_size` in bytes (since tile_size = tile_height * tile_width * element_size)

The writer reads this data with stride `output_single_block_width_size` per row.

### Block Sizing Strategy

The block size is the **largest divisor** of `num_tiles_per_column_row` (tiles per width) that fits within the L1 budget. This maximizes the amount of work per block while respecting memory constraints. The search is done from `max_tiles_per_cb` downward:

```cpp
for (uint32_t i = max_tiles_per_cb; i > 0; --i) {
    if (num_tiles_per_column_row % i == 0) {
        num_tiles_per_block = i;
        break;
    }
}
```

### TensorAccessor Pattern for Output Writes

The writer creates its TensorAccessor with:
- **Base address**: `dst_addr` (runtime arg -- the output buffer's DRAM address)
- **Page size**: `output_stick_size` (compile-time -- one full row of the output tensor)
- **Args offset**: Starting at compile-time arg index 8

The `get_noc_addr(stick_id)` call handles the bank interleaving: given a linear stick ID (row index), it computes which DRAM bank holds that row and the offset within the bank.

### Sharding Support

The `num_output_columns_of_blocks` parameter handles width/block/ND sharded outputs. When the output is sharded, the full tensor width is divided into column groups, each corresponding to a shard's width. The stick_id calculation incorporates the column group index `j` to address the correct shard. For interleaved outputs, `num_output_columns_of_blocks = 1` and this dimension collapses.

### FP32 Destination Accumulator

When input dtype is INT32, UINT32, or FLOAT32, the compute kernel defines `DST_ACCUM_MODE = 1` and enables `fp32_dest_acc_en`. This halves the DEST register capacity (from 8 to 4 tiles in double-buffer mode), which may cause the untilize helper to use the block-based pack path with sub-blocks.

## Relevance to rms_norm Output Stage

For a new rms_norm operation with in-kernel untilize:

1. **Output CB**: Allocate CB c_16 (or another output index) with capacity = `num_tiles_per_block` tiles using the output data format. Size it using the same L1 budget approach, accounting for all CBs (input, intermediates, output).

2. **Compute kernel**: After computing `x / sqrt(mean(x^2) + eps) * gamma`, call `compute_kernel_lib::untilize<block_width, intermediate_cb, output_cb, InitUninitMode::..., WaitMode::NoWait>(num_blocks)` to convert the result from tiled to row-major. Use `NoWait` if the data is already in the intermediate CB from the previous compute step.

3. **Writer kernel**: Reuse the stick-based writing pattern from this untilize writer. The key elements to replicate:
   - Pre-compute `base_dst_noc_addr[tile_height]` for all rows in the current tile-row
   - Write `tile_height` sticks per block, each `output_single_block_width_size` bytes
   - Single barrier per block
   - Use TensorAccessor with stick-based page size for output addressing

4. **Compile-time args for writer**: Pass `output_stick_size`, `tile_height`, loop bounds, `output_single_block_width_size`, and TensorAccessor args.

5. **Runtime args for writer**: Only the output buffer base address.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor work in tt-metal? What is TensorAccessorArgs and how does get_noc_addr work for mapping stick IDs to physical DRAM addresses?"
   **Reason**: Needed to understand how the writer kernel translates logical stick IDs to physical NoC addresses for DRAM writes.
   **Key Findings**: TensorAccessor abstracts bank-interleaved and sharded memory layouts. On the host side, `TensorAccessorArgs(buffer)` generates compile-time and runtime args. On the device side, `TensorAccessor(args, base_addr, page_size)` provides `get_noc_addr(page_id)` which computes the bank index and offset using round-robin interleaving. The page_size parameter determines the granularity of addressing (tiles for reader, sticks for writer).

2. **Query**: "How does pack_untilize_block work in tt-metal compute kernels?"
   **Reason**: Needed to understand how the compute kernel converts tiled data to row-major format and what the DEST register capacity constraints are.
   **Key Findings**: `pack_untilize_block` unpacks tiles from input CB into DEST, then packs them into output CB in row-major order. DEST capacity is 8 tiles (bfloat16) or 4 tiles (float32) in double-buffer mode. When `block_width_tiles > DEST_LIMIT`, the helper automatically splits into sub-blocks via `compute_num_blocks()`.

3. **Query**: "In tt-metal, what is CBIndex::c_0 vs CBIndex::c_16?"
   **Reason**: Needed to understand the CB index convention used in the factory.
   **Key Findings**: c_0/c_1 are conventionally input CBs, c_16 is conventionally the output CB. These are conventions, not hardware requirements -- any unique index works. c_24-c_31 are often used for intermediates.

4. **Query**: "After pack_untilize_block writes to the output CB, how is the data organized?"
   **Reason**: Needed to understand the exact memory layout in the output CB so the writer can correctly extract sticks.
   **Key Findings**: Data is laid out as `tile_height` contiguous rows, each `block_width_tiles * TILE_WIDTH * element_size` bytes wide. The writer reads sequentially with stride = `output_single_block_width_size`.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host-side setup and device-side usage patterns.
   **Key Information**: Host-side `TensorAccessorArgs(buffer)` generates args; device-side `TensorAccessor(args, addr, page_size)` provides `get_noc_addr(page_id)`. Args can be split between compile-time and common-runtime. The `append_to()` method adds args to a compile-time args vector.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding the difference between tiled and row-major layouts, and how pages map to banks.
   **Key Information**: In row-major layout, each row is one page. In tiled layout, each 32x32 tile is one page. Interleaved memory distributes pages round-robin across banks. Tiles internally use 16x16 face ordering.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize helper API and its configuration options.
   **Key Information**: The helper provides `InitUninitMode` (for chaining multiple calls), `WaitMode` (WaitBlock/WaitUpfront/NoWait for external sync), and `ReconfigureRegisterDatatypeMode`. It automatically handles DEST capacity splitting. For rms_norm, `NoWait` mode is appropriate when data is already in the CB from a previous compute step.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in program factories.
   **Key Information**: `create_cb(cb_index, program, core, page_size, num_pages, data_format)` creates a CB with the given configuration. Returns `(cb_index, cb_handle)`. Supports optional `buffer` parameter for globally-allocated CBs.
