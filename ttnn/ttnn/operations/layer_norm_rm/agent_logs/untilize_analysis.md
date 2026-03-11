# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize multi-core operation converts tensor data from **TILE_LAYOUT** (32x32 tiles stored in face order) back to **ROW_MAJOR_LAYOUT** (contiguous sticks). This analysis focuses on the **output stage** aspects: how the compute kernel produces row-major sticks in an output CB, and how the writer kernel extracts those sticks and writes them to DRAM via TensorAccessor.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Focus**: output_stage -- Output CB sizing, untilize helper signature/usage, writer kernel pattern (how RM sticks are written to DRAM), stick extraction from tiles.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = `num_tiles_per_input_block` tiles wide, `tile_height` rows tall) |
| **Unit size** | `num_tiles_per_input_block` tiles (one full tile-row width of the tensor or shard) |
| **Total units** | `num_tiles_per_col` blocks (= `tensor_height / tile_height` for interleaved) |
| **Loop structure** | Outer: iterate over blocks (tile-rows). Inner: compute untilizes one block at a time, writer extracts `tile_height` RM sticks per block. |

One "input block" is a horizontal strip of tiles spanning the full width of the tensor (interleaved) or the shard width (sharded). It is `num_tiles_per_input_block` tiles wide and `tile_height` (32) rows tall. The compute kernel untilizes this block, converting the tiles into `tile_height` contiguous row-major sticks in the output CB. The writer then extracts these sticks and writes them to DRAM page-by-page.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] | [..., H, W] |
| **Dimension convention** | NHWC (flattened to 2D: `tensor_height x tensor_width`) | NHWC (flattened to 2D) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | ROW_MAJOR_LAYOUT (sticks) |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | bfloat16 / float32 / uint16 / int32 / uint32 | Same as input |

### Layout Transformations

The compute kernel performs the core layout transformation: tiles (32x32, stored as four 16x16 faces) are converted to row-major sticks (contiguous rows of `num_cols_per_input_block` elements). After untilization, the output CB contains `tile_height` consecutive RM sticks, each `num_cols_per_input_block * element_size` bytes wide. The writer then repackages these sticks into output pages whose width may differ from the input block width (e.g., when input is sharded but output is interleaved, or vice versa).

## Data Flow Pattern (Output Stage Focus)

### Step-by-step flow through compute and writer

1. **Compute: Untilize one block of tiles into RM sticks**
   - `cb_wait_front(input_cb, block_width_tiles)` -- wait for reader to fill one tile-row
   - `cb_reserve_back(output_cb, block_width_tiles)` -- reserve output space
   - `pack_untilize_block<block_width_tiles>(input_cb, 1, output_cb, 0)` -- hardware-accelerated tile-to-RM conversion
   - `cb_pop_front(input_cb, block_width_tiles)` -- release input tiles
   - `cb_push_back(output_cb, block_width_tiles)` -- signal output sticks are ready
   - Repeat for `num_blocks` (= `num_input_blocks_to_process`)

2. **Writer: Extract RM sticks from output CB and write to DRAM**
   - `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- wait for compute to produce one untilized block
   - `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- get L1 address of the untilized data
   - For each of `tile_height` rows (j = 0..31):
     - Calculate `current_l1_read_addr = base + j * num_cols_per_input_block * element_size`
     - Determine output `page_id` from `(block_height_index * tile_height + j) * num_output_blocks_across_width + width_offset`
     - Write stick data to DRAM via `noc_async_write(l1_addr, s.get_noc_addr(page_id, byte_offset), num_bytes)`
     - Handle cases where one input block spans multiple output pages (sharding width mismatch)
   - `noc_async_write_barrier()` -- ensure all writes complete
   - `cb_pop_front(cb_id_out0, num_tiles_per_input_block)` -- release output CB space

### Key insight: Output CB data layout after untilize

After the compute kernel untilizes one block, the output CB contains data arranged as:
```
Row 0: [col_0, col_1, ..., col_(W-1)]  -- W = num_cols_per_input_block
Row 1: [col_0, col_1, ..., col_(W-1)]
...
Row 31: [col_0, col_1, ..., col_(W-1)]
```

Each row is `num_cols_per_input_block * element_size` bytes. The writer reads row `j` at offset `j * num_cols_per_input_block * element_size` from the CB base pointer. This is a contiguous RM stick.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 (CB 0) | cb_src0 | Input tile staging | `num_tiles_per_input_block * 2` tiles (interleaved); `num_tiles_per_input_block * num_blocks_per_core` tiles (sharded) | `num_tiles_per_input_block` tiles | Double (interleaved, multi-block); Single (single-block or sharded) | Reader | Compute | Block |
| c_16 (CB 16) | cb_output | **Output RM stick staging** | `num_tiles_per_input_block * 2` tiles (multi-block); `num_tiles_per_input_block` tiles (single-block) | `num_tiles_per_input_block` tiles | **Double** (multi-block); **Single** (single-block) | Compute | Writer | Block |

### Output CB Sizing Details (Primary Focus)

The output CB (c_16) sizing follows this logic from the program factory (lines 149-163):

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

- **Block size** = `num_tiles_per_input_block` tiles = one full tile-row width
- **Capacity** = 1x or 2x the block size
- **Double buffering** is used when a core processes 2+ blocks, allowing the writer to drain one block while compute fills the next
- **Page size** = `output_single_tile_size` (tile-sized pages in the CB, even though data is now RM)
- **Data format** = same as output tensor dtype

**Critical observation for downstream usage**: The output CB page size is `tile_size(output_cb_data_format)` -- a full tile's worth of bytes. Even though the data is row-major after untilization, the CB is configured with tile-sized pages because `pack_untilize_block` writes tile-sized output chunks. The CB capacity in tiles must be >= `block_width_tiles` (the width in tiles of one tile-row).

### Why tile-sized pages in the output CB?

The `pack_untilize_block` hardware function writes output as tile-width chunks. Although the data within the CB is rearranged into RM order, the CB management still operates in units of tiles. The `cb_push_back` / `cb_pop_front` / `cb_wait_front` all operate in tile counts. The writer then reads the raw bytes from L1 using `get_read_ptr()` and manually computes byte offsets for each row.

## Pipeline Pattern Summary

- **Interleaved, multi-block**: Output CB is double-buffered (capacity = 2 * block_size). Compute can fill block N+1 while writer drains block N.
- **Interleaved, single-block**: Output CB is single-buffered (capacity = block_size). No overlap needed since there is only one block.
- **Sharded**: Output CB is single- or double-buffered following the same rule as interleaved (based on `num_input_blocks_per_full_core`).

## Index Calculations

### Writer Kernel Page ID Calculation

The writer uses TensorAccessor for address translation. The key index mapping is:

```
output_page_id = num_rows_already_processed * num_output_blocks_across_width + width_wise_output_block_start_index
```

Where:
- `num_rows_already_processed = block_height_index * tile_height + j` (j = row within tile, 0..31)
- `num_output_blocks_across_width` = 1 for interleaved/height-sharded (each page = full tensor width)
- `width_wise_output_block_start_index` = which output page column this core starts at

For a standard interleaved output with full-width pages:
- `page_id` = row index (since each page = one full row)
- `output_stick_size` = `tensor_width * element_size` bytes

### TensorAccessor with Byte Offset

The writer uses `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` to write to a specific byte offset within a page. This is essential when an input block maps to the middle of an output page (e.g., width-sharded input with interleaved output).

The TensorAccessor signature: `get_noc_addr(page_id, offset = 0, noc = noc_index)` returns the NOC address for the given page with an optional byte offset added.

### Host-side Setup for TensorAccessor

On the host, TensorAccessor args are created and appended to compile-time args:
```cpp
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

On the device kernel, they are reconstructed:
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // starts at compile-time arg index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

The template parameter `8` is the compile-time argument offset where the TensorAccessor args begin (after the 8 explicit writer compile-time args at indices 0-7).

## Memory Access Patterns

### Read Pattern (Writer reads from output CB)

The writer reads from the output CB in L1 using direct pointer arithmetic:
- `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- base of the untilized block in L1
- Row `j` starts at `base + j * num_cols_per_input_block * element_size`
- Access is **sequential within each row**, **strided between rows** (stride = `num_cols_per_input_block * element_size`)
- This is a **row-sequential** pattern: all columns of row 0, then all columns of row 1, etc.

### Write Pattern (Writer writes to DRAM)

- Each RM stick (or portion thereof) is written to DRAM via `noc_async_write`
- Pages are written in **row-major order**: all columns for row 0 across all output pages, then row 1, etc.
- For interleaved output: one `noc_async_write` per row per input block (stick = full tensor width)
- For width/block-sharded output: potentially multiple writes per row if input block spans multiple output pages
- Write barrier (`noc_async_write_barrier()`) is issued once per block (after all `tile_height` rows)

### Padding Handling

The writer supports uneven sharding via `num_unpadded_cols_per_input_block`. When the last shard in a row has padding columns, only the valid columns are written; padding data in the CB is simply ignored by limiting the write width.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear core assignment from device compute grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `num_compute_cores` (derived from `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` blocks (tile-rows) for full cores; `num_rows_per_cliff_core` for cliff core |
| **Load balancing** | Near-equal with optional cliff core for remainder |

### Interleaved Path

`split_blocks_for_tilize(grid_size, num_tiles_per_col)` divides the total number of tile-rows across cores:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- Cliff core gets `num_tiles_per_col % nblocks_per_core` blocks (may be 0)
- Each core processes contiguous tile-rows (blocks)

### Sharded Path

For sharded input, each shard maps to one core. The number of cores equals the number of shards. There is no cliff core. Each core processes `input_shard_height / tile_height` blocks.

## Arguments

### Compile-Time Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output CB index (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output RM page in bytes (`output_page_width * element_size`) |
| 2 | tile_height | uint32_t | Height of tiles (32 for standard tiles) |
| 3 | num_tiles_per_input_block | uint32_t | Number of tiles in one tile-row (block width in tiles) |
| 4 | num_output_blocks_across_width | uint32_t | Number of output pages per tensor row (1 for interleaved, >1 for width/block sharded output) |
| 5 | output_element_size | uint32_t | Size of one element in bytes (2 for bfloat16, 4 for float32) |
| 6 | num_cols_per_input_block | uint32_t | Number of columns per input block (`num_tiles_per_input_block * tile_width`) |
| 7 | num_cols_per_output_block | uint32_t | Number of columns per output page (`output_page_width`) |
| 8+ | TensorAccessor args | uint32_t[] | Auto-appended by `TensorAccessorArgs(*dst_buffer).append_to(...)` |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |
| 1 | num_input_blocks_to_process | uint32_t | Number of tile-row blocks this core processes |
| 2 | height_wise_input_block_start_index | uint32_t | Starting tile-row index for this core |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Valid (non-padding) columns in this core's input block |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page column index |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Byte-offset within the first output page (for mid-page writes) |

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Number of tiles per block (= `num_tiles_per_input_block`) |
| 1 | src_cb_id | uint32_t | Input CB index (c_0) |
| 2 | out_cb_id | uint32_t | Output CB index (c_16) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) this core processes |

## Kernel Implementations

### Compute Kernel: untilize_variable_num_blocks.cpp / pack_untilize_variable_num_blocks.cpp

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (unpack/math/pack) | N/A | CB c_0 (tiled) | CB c_16 (RM sticks) | `compute_kernel_lib::untilize<>()` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` (slow path) or `pack_untilize_variable_num_blocks.cpp` (fast path)
- **Helper used**: `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id, InitAndUninit, WaitBlock, NoReconfigure>(per_core_block_cnt)`
- **Selection logic**: Fast path (`pack_untilize`) is used by default. Falls back to slow path for UINT16, or when FLOAT32 with `num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH`
- **Key behavior**: The helper handles all CB synchronization internally (wait_front on input, reserve_back/push_back on output). The caller only passes `num_blocks`.

#### Untilize Helper Signature and Usage

```cpp
template <
    uint32_t block_width_tiles,     // compile-time: tiles per row
    uint32_t input_cb,              // compile-time: input CB index
    uint32_t output_cb,             // compile-time: output CB index
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
ALWI void untilize(uint32_t num_blocks);   // runtime: number of tile-rows to process
```

**Critical for downstream use**: The helper internally manages all CB operations (`cb_wait_front`, `cb_reserve_back`, `cb_push_back`, `cb_pop_front`). When integrating into a fused kernel, callers must NOT add redundant CB operations around the helper call. The `WaitBlock` mode means the helper waits for `block_width_tiles` input tiles per block iteration.

**WaitMode options relevant for integration**:
- `WaitBlock` (default): Helper calls `cb_wait_front(input_cb, block_width_tiles)` per block. Use when data arrives block-by-block.
- `WaitUpfront`: Helper calls `cb_wait_front(input_cb, total_tiles)` once before processing. Use when all input data is available at start.
- `NoWait`: Caller manages synchronization. Use when untilize is part of a fused pipeline where prior operations already ensured data availability.

**InitUninitMode options for fused kernels**:
- `InitAndUninit` (default): Calls both `pack_untilize_init` and `pack_untilize_uninit`.
- `InitOnly` / `UninitOnly` / `Neither`: For back-to-back untilize calls or when sandwiching untilize between other compute operations.

**Prerequisite**: `compute_kernel_hw_startup(input_cb, output_cb)` must be called before using the untilize helper.

**Width handling**: When `block_width_tiles > DEST_AUTO_LIMIT`, the helper automatically splits the block into sub-blocks that fit in the DEST register, using `compute_num_blocks()` to find the largest divisor of `block_width_tiles` that is <= `DEST_AUTO_LIMIT`.

### Writer Kernel: writer_unary_stick_layout_split_rows_multi_core.cpp

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 (RM sticks in L1) | DRAM (interleaved/sharded output) | Stick extraction, `noc_async_write` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  1. Outer loop: iterates over blocks (`num_input_blocks_to_process`)
  2. Per block: `cb_wait_front` for one untilized block, then inner loop over `tile_height` rows
  3. Per row: reads L1 data at computed offset, writes to DRAM using TensorAccessor
  4. Handles output page boundary crossing when input and output widths differ
  5. Uses `noc_async_write_barrier()` per block (not per row) for efficiency
  6. Handles padding via `num_unpadded_cols_per_input_block`

#### Writer Pattern for DRAM Interleaved Output (Simplest Case)

For interleaved output where output page width = tensor width:
- `num_output_blocks_across_width = 1`
- `output_stick_size = tensor_width * element_size`
- `output_page_id = row_index` (one page per tensor row)
- One `noc_async_write` per row with size = `output_stick_size`
- No page boundary crossing logic needed

This is the most relevant pattern for the layer_norm_rm use case.

### Reader Kernel (De-emphasized)

- **File (interleaved)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **File (sharded)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`
- Reads tiles from DRAM/L1 into input CB c_0. Uses TensorAccessor for address translation.

## Implementation Notes

### Output Stick Extraction from Tiles

After `pack_untilize_block`, the output CB contains `tile_height` RM sticks laid out contiguously. The writer accesses them via direct L1 pointer arithmetic:

```cpp
uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);
// Row j starts at:
uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;
```

This means the untilize hardware has already reordered the face data (4 faces of 16x16) into 32 contiguous RM rows of `block_width_tiles * tile_width` elements each.

### Output CB as Page-Width Staging Area

The output CB holds one block's worth of untilized data: `num_tiles_per_input_block` "tiles" worth of space, but organized as `tile_height` RM sticks. The CB page size remains `tile_size(data_format)` for CB management purposes (push/pop/wait operate in tile counts), but the actual data layout is row-major.

### Write Granularity

The writer performs one `noc_async_write` per row per output page intersection. For the common interleaved case, this means one write per row (32 writes per block). The write barrier is at block granularity, amortizing the barrier cost over 32 writes.

### Relevance for layer_norm_rm Output Stage

For a fused operation that performs computation in tile format and needs to write RM output:

1. **CB Configuration**: Allocate output CB (e.g., c_16) with capacity = `Wt` tiles (single-buffered if only one block per core) or `Wt * 2` (double-buffered). Page size = `tile_size(output_format)`.

2. **Compute Integration**: Use `compute_kernel_lib::untilize<Wt, compute_output_cb, writer_output_cb>(num_blocks)` to convert the final tiled result to RM in the output CB. If untilize follows other compute operations, use `InitOnly`/`UninitOnly` modes and manage init/uninit lifecycle manually.

3. **Writer Pattern**: After `cb_wait_front(output_cb, Wt)`, read `tile_height` sticks from L1 using `get_read_ptr()` + row offset. Write each stick to DRAM via `noc_async_write(l1_addr, tensor_accessor.get_noc_addr(page_id), stick_size)`. Issue `noc_async_write_barrier()` per block. Pop the CB.

4. **TensorAccessor Setup**: Create `TensorAccessorArgs(*dst_buffer)` on host, append to writer compile-time args. On device, construct `TensorAccessor(args, dst_addr, stick_size)` where `stick_size = tensor_width * element_size`.

5. **Page ID Calculation**: For interleaved output, `page_id = absolute_row_index` where `absolute_row_index = core_start_row + block_index * tile_height + row_within_tile`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the untilize operation in tt-metal/TTNN? How does pack_untilize work at the hardware level?"
   **Reason**: Needed to understand how `pack_untilize_block` converts tiled data to RM at the hardware level, specifically what the DEST register limitation means and how sub-blocking works.
   **Key Findings**: `pack_untilize_block` operates in the PACK phase of LLK. It reads tiles from DEST, reorders face data (four 16x16 faces per 32x32 tile) into contiguous rows, and writes to the output CB. The DEST register has a limited capacity (8 tiles in half-sync, 16 in full-sync for 16-bit data), so wide rows must be split into sub-blocks. The `DEST_AUTO_LIMIT` macro handles this automatically.

2. **Query**: "How does TensorAccessor work in tt-metal kernels?"
   **Reason**: The writer kernel uses TensorAccessor for DRAM address translation. Needed to understand the `get_noc_addr(page_id, offset)` API and how compile-time args are set up.
   **Key Findings**: DeepWiki query failed, but documentation from `tech_reports/tensor_accessor/tensor_accessor.md` confirmed: `TensorAccessorArgs(buffer)` on host extracts bank mapping info. `append_to(compile_time_args)` adds them as compile-time args. On device, `TensorAccessorArgs<base_offset>()` reconstructs, and `TensorAccessor(args, base_addr, page_size)` creates the accessor. `get_noc_addr(page_id, offset)` returns the physical NOC address for a given page with optional byte offset.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host setup and device usage patterns for writer kernels.
   **Key Information**: TensorAccessor handles mapping from logical page IDs to physical bank addresses across distributed DRAM banks. Default config (`ArgConfig::None`) passes all args as compile-time. `get_noc_addr(page_id, offset)` is the primary API for address translation.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding the RM page concept (one page = one tensor row for RM interleaved), and how tiles are structured (32x32 with four 16x16 faces).
   **Key Information**: For RM layout, each page is one full row of the flattened 2D tensor. For tiled layout, each page is one tile. Pages are distributed round-robin across banks for interleaved layout.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the compute_kernel_lib::untilize helper API -- its template parameters, WaitMode/InitUninitMode enums, and internal CB management.
   **Key Information**: The helper manages all CB operations internally. `WaitBlock` mode waits per block; `WaitUpfront` waits for all tiles at start; `NoWait` expects caller to manage sync. Wide rows (> DEST limit) are automatically split into sub-blocks. The helper requires `compute_kernel_hw_startup(input_cb, output_cb)` as prerequisite.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding `split_blocks_for_tilize` which determines core count and work distribution.
   **Key Information**: For 1D splitting, `nblocks_per_core = ceil(total_blocks / grid_area)`. One cliff core may handle remainder blocks. Returns `BlockSplit` struct with full and cliff core ranges.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` utility used to configure circular buffers.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer)` creates a CB with `total_size = num_pages * page_size`. If `buffer != nullptr`, the CB is backed by a globally-allocated buffer (used for sharded input CB).
