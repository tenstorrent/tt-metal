# Untilize Multi-Core Implementation Analysis

## Overview

The untilize operation converts tiled (32x32) tensor data into row-major (RM) format. It reads tiles from an input tensor, uses the compute kernel's `pack_untilize` hardware path to extract contiguous row-major sticks from tiles, and writes those sticks to a row-major interleaved (or sharded) output buffer in DRAM/L1. This analysis focuses on the **output stage**: the output circular buffer sizing, how RM sticks are extracted from tiles by compute, and the writer kernel pattern that writes those sticks to DRAM.

**Program Factory**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Role context**: This analysis serves as an `output_stage` reference for a new `layer_norm_rm` operation that will produce row-major interleaved output after performing layer normalization.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile row (one row of tiles across the tensor width) |
| **Unit size** | `num_tiles_per_input_block` tiles (= `tensor_width / tile_width`) |
| **Total units** | `num_tiles_per_col` (= `tensor_height / tile_height`) tile-rows |
| **Loop structure** | Outer: per-block (tile-row); Inner: per-stick within the tile-row (tile_height iterations in writer) |

One "input block" corresponds to one horizontal row of tiles spanning the tensor width. Each block contains `num_tiles_per_input_block` tiles. The compute kernel processes one block at a time, producing `tile_height` (32) contiguous RM sticks in the output CB. The writer then drains these sticks to DRAM one tile-row at a time.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, ..., H, W] (arbitrary rank, flattened to 2D: height x width) |
| **Dimension convention** | Last dim = W, everything else = H |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | bfloat16, float32, uint16, int32, uint32 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED (primary focus), also supports SHARDED |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | Same as input |
| **Page definition** | One page = one RM stick = `output_page_width * element_size` bytes |

For the interleaved case, `output_page_width = tensor_width` (the full row). Each page is one complete row of the tensor. Pages are distributed round-robin across DRAM banks.

### Layout Transformation

The compute kernel transforms tiles into RM sticks:
- Input: `num_tiles_per_input_block` tiles in tile format (face-interleaved 32x32)
- Output: `tile_height` (32) contiguous RM sticks, each `num_tiles_per_input_block * tile_width * element_size` bytes wide
- The `pack_untilize_block` hardware primitive performs this rearrangement in the PACK thread, writing directly to the output CB in RM order

---

## Data Flow Pattern (Output-Stage Focus)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (tiles) | CB c_0 | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB c_0 (tiles) | CB c_16 (RM sticks) | `cb_wait_front` on c_0, `cb_reserve_back` on c_16, `pack_untilize_block`, `cb_pop_front` on c_0, `cb_push_back` on c_16 |
| 3 | **Writer** | **CB c_16 (RM sticks)** | **DRAM (RM pages)** | **`cb_wait_front` on c_16, `noc_async_write`, `cb_pop_front` on c_16** |

### Detailed Output Stage Flow

1. **Compute produces RM sticks into CB c_16**: For each tile-row block, the compute kernel calls `pack_untilize_block` which unpacks tiles from CB c_0 into the DEST register and packs them back out as RM sticks into CB c_16. After processing one block (all tiles in a tile-row), it pushes `num_tiles_per_input_block` tiles worth of RM data to CB c_16.

2. **Writer drains CB c_16 to DRAM**: The writer waits for `num_tiles_per_input_block` tiles in CB c_16, then extracts `tile_height` individual RM sticks from the contiguous CB memory and writes each stick to the appropriate DRAM page using TensorAccessor.

The critical insight is that **after untilize, the data in CB c_16 is laid out as contiguous RM sticks** -- `tile_height` rows, each `num_cols_per_input_block * element_size` bytes wide. The writer reads these sticks sequentially and writes them as separate DRAM pages.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tiles (tiled format) | `num_tiles_per_input_block * 2` tiles (double-buffered) or `num_tiles_per_input_block` (single) | `num_tiles_per_input_block` tiles | Single if 1 block/core, Double if 2+ blocks/core | Reader | Compute | Block |
| **c_16** | **cb_output** | **Output RM sticks** | **`num_tiles_per_input_block * 2` tiles (double-buffered) or `num_tiles_per_input_block` (single)** | **`num_tiles_per_input_block` tiles** | **Single if 1 block/core, Double if 2+ blocks/core** | **Compute** | **Writer** | **Block** |

### Output CB Sizing (c_16) -- Key Detail for layer_norm_rm

The output CB capacity is determined by the program factory at lines 149-163:

```
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // double-buffered
}
```

- **Single-buffered** (`capacity = block_size`): When a core only processes 1 tile-row, there is no overlap benefit.
- **Double-buffered** (`capacity = 2 * block_size`): When a core processes 2+ tile-rows, the output CB is doubled so compute can fill the next block while the writer drains the current one.

The page size for each tile in the output CB is `output_single_tile_size` (from `tt::tile_size(output_cb_data_format)`), even though the data is logically RM. This is because the CB infrastructure measures capacity in tile-sized units; the `pack_untilize` hardware fills the CB with RM data but the capacity accounting still uses tile counts.

**For layer_norm_rm**: Since the output is already RM (no untilize needed), the output CB should be sized in terms of RM stick pages. The CB page size should be `stick_size = tensor_width * element_size` and capacity should be `tile_height` sticks (or `2 * tile_height` for double-buffering), matching the number of rows the compute kernel produces per block.

---

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering | Overlap Potential |
|----|----------|------------|-----------|-------------------|
| c_0 (input) | 1x or 2x block | `num_tiles_per_input_block` tiles | Single or Double | Reader can prefetch next block while compute processes current (when double) |
| c_16 (output) | 1x or 2x block | `num_tiles_per_input_block` tiles | Single or Double | Compute can fill next block while writer drains current (when double) |

---

## Index Calculations

### Output Page ID Calculation (Writer Kernel, lines 50-55)

The writer constructs the output page ID from the block height index and the row within the tile:

```
uint32_t num_rows_already_processed = block_height_index * tile_height + j;
uint32_t num_pages_already_processed_in_previous_rows =
    num_rows_already_processed * num_output_blocks_across_width;
uint32_t output_page_id =
    num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;
```

For the simple interleaved case (`num_output_blocks_across_width = 1`, `width_wise_output_block_start_index = 0`):

```
output_page_id = (block_height_index * tile_height + j) * 1 + 0
               = block_height_index * tile_height + j
```

This means **page_id = row_index**. Each RM stick (row) maps to exactly one page.

### L1 Read Address Calculation (Writer Kernel, line 43)

To read individual sticks from the output CB:

```
uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;
```

Where `base_l1_read_addr = get_read_ptr(cb_id_out0)`. This iterates through `tile_height` rows within the CB, each separated by `num_cols_per_input_block * element_size` bytes. This confirms the CB contains contiguous RM data laid out as sequential rows.

---

## Memory Access Patterns

### Read Pattern (Reader -- brief, not focus)
- Sequential tile reads from DRAM using TensorAccessor
- Each tile read individually with `noc_async_read`, barrier per tile
- Tiles read in order: `start_page_id` through `start_page_id + num_tiles - 1`

### Write Pattern (Writer -- primary focus)

**Pattern**: Row-by-row sequential writes to interleaved DRAM pages.

For each tile-row block:
1. Wait for the entire block of RM data in CB c_16 (`cb_wait_front(cb_id_out0, num_tiles_per_input_block)`)
2. For each of the `tile_height` (32) rows in the block:
   - Compute the L1 source address: `base_l1_read_addr + j * num_cols_per_input_block * element_size`
   - Compute the DRAM target: `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)`
   - Write the stick: `noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write)`
3. Issue write barrier: `noc_async_write_barrier()`
4. Release the CB: `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`

**Key characteristics**:
- Each DRAM write is one complete RM stick (for interleaved output: `tensor_width * element_size` bytes)
- Writes are issued for all 32 rows within a tile-row block before the barrier
- The `noc_async_write_barrier()` is per-block (not per-stick), amortizing barrier overhead
- Page IDs increment sequentially: row 0, row 1, ..., row 31 within each block

### Partial Page Write Support

The writer supports writing to a partial page via `output_offset_within_page_in_bytes`. This is used when the output is sharded with a different shard width than the input. For the interleaved case, this offset is always 0 (line 65-66: `num_cols_already_processed_in_first_output_block` is 0 for interleaved).

The inner while loop (lines 69-97) handles the case where one input block spans multiple output pages (or vice versa). For the simple interleaved case where `num_cols_per_input_block == num_cols_per_output_block`, the loop executes exactly once per row.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` tile-rows (blocks) for full cores |
| **Load balancing** | Equal distribution with optional cliff core |

### Work Splitting via `split_blocks_for_tilize`

The function `split_blocks_for_tilize(grid_size, num_tiles_per_col)` distributes tile-rows across cores:
- `num_tiles_per_col` = total tile-rows = `tensor_height / tile_height`
- `nblocks_per_core` = `ceil(num_tiles_per_col / grid_area)`
- `ncores` = `ceil(num_tiles_per_col / nblocks_per_core)`
- Cliff core (at most 1) gets `num_tiles_per_col % nblocks_per_core` blocks

Returns a `BlockSplit` with:
- `full_compute_core_range`: cores each processing `num_rows_per_full_core` blocks
- `cliff_compute_core_range`: 0 or 1 core processing `num_rows_per_cliff_core` blocks

---

## Arguments

### Compile-Time Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Size of one output RM stick in bytes (`output_page_width * element_size`) |
| 2 | `tile_height` | uint32_t | Height of a tile (32) |
| 3 | `num_tiles_per_input_block` | uint32_t | Number of tiles per tile-row |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages across the width (1 for interleaved) |
| 5 | `output_element_size` | uint32_t | Size of one element in bytes |
| 6 | `num_cols_per_input_block` | uint32_t | Width in elements of one input block (`num_tiles_per_input_block * tile_width`) |
| 7 | `num_cols_per_output_block` | uint32_t | Width in elements of one output page (`output_page_width`) |
| 8+ | TensorAccessor args | various | Appended by `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Destination buffer base address |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-rows this core must write |
| 2 | `height_wise_input_block_start_index` | uint32_t | Starting tile-row index for this core |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Actual data columns (excluding padding) |
| 4 | `width_wise_output_block_start_index` | uint32_t | Starting output page column index (0 for interleaved) |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset within first output page (0 for interleaved) |

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | = `num_tiles_per_input_block` (tiles per row) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | = `num_input_blocks_to_process` (tile-rows to process) |

---

## Kernel Implementations

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 (default for writers) | CB c_16 (RM sticks) | DRAM (interleaved RM pages) | Read RM sticks from CB, write to DRAM pages via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - The TensorAccessor is constructed from compile-time args starting at index 8: `constexpr auto dst_args = TensorAccessorArgs<8>(); const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);`
  - The `output_stick_size` used for TensorAccessor page size is the full output page width in bytes
  - `get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` supports partial-page writes
  - The lambda `write_tiles_in_current_block` processes one tile-row: waits for CB data, iterates `tile_height` rows, writes each stick to DRAM, then barriers and pops
  - The outer loop iterates `num_input_blocks_to_process` times, incrementing `height_wise_input_block_index`

### Compute Kernel (brief)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (TRISC) | N/A | CB c_0 (tiles) | CB c_16 (RM sticks) | `pack_untilize_block` |

- **File** (fast path): `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
- **File** (slow path): `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **Key Logic**: Both use the unified `compute_kernel_lib::untilize<>()` template from `untilize_helpers.hpp`. This template:
  - Determines if `block_width_tiles > DEST_AUTO_LIMIT` and splits into sub-blocks if needed
  - Calls `pack_untilize_block` which uses hardware pack-untilize to rearrange tile data into RM sticks in the output CB
  - Per-block synchronization: `cb_wait_front(input_cb, block_width_tiles)` then `cb_reserve_back(output_cb, block_width_tiles)`, process, pop input, push output

### Reader Kernel (brief)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (tiles) | CB c_0 | Sequential tile reads |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

---

## TensorAccessor Usage Pattern (Critical for layer_norm_rm)

### Host-Side Setup

```cpp
// Writer compile-time args
std::vector<uint32_t> writer_compile_time_args = {
    output_cb_index,
    output_stick_size,      // page size for TensorAccessor
    tile_height,
    num_tiles_per_input_block,
    // ... other args ...
};
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

The `TensorAccessorArgs` constructor takes the output buffer and appends distribution metadata (bank coordinates, tensor shape, shard shape, etc.) as compile-time constants. This means the TensorAccessor knows how to map page IDs to physical DRAM bank addresses at compile time.

### Device-Side Usage

```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // compile-time args start at index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);

// For each stick to write:
uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write);
```

**For layer_norm_rm adaptation**: The same pattern applies. Create a TensorAccessor from the output buffer with `page_size = stick_size`. Each page_id corresponds to one RM row. Use `get_noc_addr(page_id)` (no offset needed if writing full sticks) and `noc_async_write` to write each stick.

---

## Stick Extraction from Tiles -- How It Works

The `pack_untilize_block` hardware primitive rearranges tile-format data into RM sticks. Here is the logical transformation:

### Before (in CB c_0, tile layout)
For a row of N tiles, each 32x32, the data in CB c_0 is stored as N consecutive tiles, each with face-interleaved layout:
```
[Tile0: face0(16x16), face1(16x16), face2(16x16), face3(16x16)]
[Tile1: face0, face1, face2, face3]
...
[TileN-1: face0, face1, face2, face3]
```

### After (in CB c_16, RM layout)
The output CB contains 32 contiguous rows, each N*32 elements wide:
```
Row 0:  [Tile0_row0(32 elems), Tile1_row0(32 elems), ..., TileN-1_row0(32 elems)]
Row 1:  [Tile0_row1(32 elems), Tile1_row1(32 elems), ..., TileN-1_row1(32 elems)]
...
Row 31: [Tile0_row31(32 elems), Tile1_row31(32 elems), ..., TileN-1_row31(32 elems)]
```

Each row is `num_tiles_per_input_block * tile_width * element_size` bytes, which equals `num_cols_per_input_block * element_size`. The writer kernel reads these rows at stride `num_cols_per_input_block * element_size` from `get_read_ptr(cb_id_out0)`.

### Implications for layer_norm_rm

Since layer_norm_rm does NOT need to untilize (input and output are both RM), the compute kernel instead writes RM sticks directly. The output CB for layer_norm_rm should:
- Use a page size equal to `stick_size` (not tile_size)
- Have capacity for at least `tile_height` sticks (to match the number of rows produced per compute block)
- Use the same writer pattern: iterate rows, compute page_id = row_index, write each stick via TensorAccessor

---

## Implementation Notes

### Output CB Page Size Subtlety

In the untilize program factory, the output CB is created with `output_single_tile_size` as the page size (line 157-163):
```cpp
auto [output_cb_index, cb_output] = create_cb(
    tt::CBIndex::c_16, program, compute_core_range,
    output_single_tile_size,    // <-- tile_size, not stick_size!
    output_cb_num_tiles,
    output_cb_data_format);
```

This is because `pack_untilize_block` operates in tile-count units for CB push/pop, and the CB capacity is measured in tiles. The actual RM data fills the same byte region. For a pure RM operation like layer_norm_rm, the CB page size should instead be `stick_size` bytes and capacity measured in stick counts.

### Write Barrier Placement

The write barrier `noc_async_write_barrier()` is placed once per tile-row block (after all 32 sticks are written), not per stick. This batches all 32 async writes before synchronizing, which is more efficient. The same pattern should be used in layer_norm_rm.

### Double-Buffering Decision

The decision to double-buffer the output CB depends on `num_input_blocks_per_full_core`:
- If a core processes only 1 block, there is nothing to overlap, so single-buffering suffices
- If a core processes 2+ blocks, double-buffering allows compute to fill the next block while the writer drains the current one

For layer_norm_rm, the same logic applies: if each core processes multiple row-blocks, double-buffer the output CB.

### Unpadded Column Handling

The writer uses `num_unpadded_cols_per_input_block` (runtime arg index 3) to handle cases where the last shard in a row has padding columns. Only the unpadded columns are written to DRAM. For interleaved tensors, `num_unpadded_cols_per_input_block == num_cols_per_input_block` (no padding).

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize operation work in TTNN? Specifically, how does the pack_untilize compute kernel extract row-major sticks from tiles, and what is the layout of the output circular buffer after untilizing?"
   **Reason**: Needed to confirm that the output CB after untilize contains contiguous RM sticks (not tile-ordered data).
   **Key Findings**: Confirmed that `pack_untilize_block` writes RM sticks directly to the output CB. The `llk_pack_untilize` function calculates `pack_tile_addr` to write row-major data. The output CB is contiguous RM after the untilize compute.

2. **Query**: "How does the TensorAccessor get_noc_addr work with an offset parameter? When writing row-major sticks to interleaved DRAM, what does page_id represent?"
   **Reason**: Needed to understand the page_id semantics for the writer kernel and how partial-page writes work.
   **Key Findings**: `page_id` = stick index (row index) for RM interleaved tensors. `get_noc_addr(page_id, offset)` adds a byte offset within the page. The offset is used for partial writes when sharding causes misalignment between input blocks and output pages.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host-side setup and device-side API for address calculations.
   **Key Information**: `TensorAccessorArgs(buffer)` on host, `TensorAccessorArgs<base_idx>()` + `TensorAccessor(args, addr, page_size)` on device. `get_noc_addr(page_id, offset)` maps logical page to physical NOC address. All compile-time by default.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM page definition and interleaved page distribution.
   **Key Information**: For RM layout, each row = one page. Interleaved distributes pages round-robin across banks. Page size = row_width * element_size.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl`
   **Reason**: Understanding the compute-side untilize implementation and CB synchronization pattern.
   **Key Information**: The unified `untilize<>()` template auto-selects between single-pass and block-based paths depending on whether `block_width_tiles > DEST_AUTO_LIMIT`. CB synchronization: `cb_wait_front` on input, `cb_reserve_back` on output, process, `cb_pop_front` input, `cb_push_back` output.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding `split_blocks_for_tilize` used for core distribution.
   **Key Information**: Distributes `nblocks` across `grid_area` cores. Returns `BlockSplit` with full core range, cliff core range, blocks per core. Cliff core handles remainder.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used to create circular buffers.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer)` -- wraps `CircularBufferConfig` creation. If `buffer != nullptr`, sets globally-allocated address (used for sharded input).
