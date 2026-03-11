# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize operation converts tensor data from **tiled layout** (32x32 tiles) to **row-major layout** (linear sticks). It reads tiles from DRAM (interleaved) or L1 (sharded), uses the compute kernel to reorder tile data into row-major sticks, and writes the resulting RM sticks back to DRAM via TensorAccessor.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Focus**: This analysis emphasizes the **output stage** -- output CB sizing, the writer kernel's RM-stick write pattern, stick extraction from tiles, and the untilize compute helper signature. Reader and input CB details are de-emphasized.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = `num_tiles_per_row` tiles spanning the full tensor width) |
| **Unit size** | `num_tiles_per_input_block` tiles (a row of tiles across one input shard or the full tensor width for interleaved) |
| **Total units** | `num_tiles_per_col` tile-rows (total height in tiles) |
| **Loop structure** | Outer: iterate over tile-rows assigned to this core. Inner (compute): process one block of `num_tiles_per_input_block` tiles. Inner (writer): iterate over `tile_height` (32) stick rows per block, writing each stick to DRAM. |

One "input block" is a horizontal strip of `num_tiles_per_input_block` tiles (one tile-row high). The compute kernel untilizes this block, producing `tile_height` (32) row-major sticks of width `num_cols_per_input_block` elements each. The writer then writes these sticks to the output buffer as RM pages.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary ND (flattened to 2D: height x width) |
| **Dimension convention** | Last dim = width, all others collapsed into height |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT (sticks) |
| **Memory layout** | INTERLEAVED (or SHARDED for width/block sharding) |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | Same as input |

### Layout Transformations

The compute kernel performs the tile-to-row-major conversion. Each 32x32 tile (with face-based internal layout) is converted into 32 contiguous row-major sticks by the hardware's pack_untilize or untilize machinery. The output CB then holds these sticks in a format where each row of `num_cols_per_input_block` elements is contiguous -- ready for the writer to emit as RM pages.

---

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved tiles) or L1 (sharded) | CB c_0 (input) | `cb_reserve_back`, `cb_push_back` |
| 2 | Compute | CB c_0 (input) | CB c_16 (output) | `cb_wait_front(c_0)`, `cb_pop_front(c_0)`, `cb_reserve_back(c_16)`, `cb_push_back(c_16)` |
| 3 | Writer | CB c_16 (output) | DRAM (RM sticks) | `cb_wait_front(c_16)`, `cb_pop_front(c_16)` |

### Detailed Output Stage Flow

1. **Compute produces one block**: The untilize helper converts `num_tiles_per_input_block` tiles into RM data in CB c_16. After conversion, `cb_push_back(c_16, num_tiles_per_input_block)` signals the writer.

2. **Writer waits for block**: `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- the writer waits for one full block-width of untilized data.

3. **Writer extracts sticks**: The writer reads the L1 memory behind CB c_16 directly via `get_read_ptr(cb_id_out0)`. The untilized data is laid out as `tile_height` rows, each `num_cols_per_input_block` elements wide. The writer iterates over each of the `tile_height` (32) rows.

4. **Writer writes sticks to DRAM**: For each row, the writer computes the output page_id (RM page) and byte offset within that page using the TensorAccessor. It then issues `noc_async_write` to push the stick data to DRAM.

5. **Writer releases block**: After writing all rows, `noc_async_write_barrier()` ensures completion, then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)` frees the CB space.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_input_block * 2` (interleaved, multi-block) or `num_tiles_per_input_block` (single block) or entire shard (sharded) | `num_tiles_per_input_block` | Double (interleaved, multi-block) / Single (single block) / Program (sharded) | Reader | Compute | Block |
| c_16 | cb_output | **Output RM staging** | `num_tiles_per_input_block * 2` (multi-block) or `num_tiles_per_input_block` (single block) | `num_tiles_per_input_block` | **Double** (multi-block) / **Single** (single block) | Compute | Writer | Block |

### Output CB Sizing Details (Primary Focus)

The output CB (c_16) sizing logic (lines 149-163 of program factory):

```
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

- **Block size**: `num_tiles_per_input_block` tiles = one full tile-row of the input block width.
- **Capacity**: Either 1x or 2x the block size depending on whether the core processes multiple blocks.
- **Page size**: `output_single_tile_size` = `tile_size(output_cb_data_format)` bytes per tile.
- **Total CB bytes**: `output_cb_num_tiles * output_single_tile_size`.

**Key insight for layer_norm_rm**: When the compute kernel writes untilized data into the output CB, the data is organized as `tile_height` contiguous rows of `num_cols_per_input_block * element_size` bytes. The "tile" page size is still used for CB accounting (the CB tracks tiles), but the actual memory layout within the CB is now row-major after untilize.

---

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Pattern |
|----|----------|------------|-------|---------|
| c_0 (input) | 2 * block (typical) | block | 2x | Double-buffered (reader/compute overlap) |
| c_16 (output) | 2 * block (typical) | block | 2x | Double-buffered (compute/writer overlap) |

When only 1 block is processed per core, both CBs are single-buffered (no overlap possible since there is only one iteration).

---

## Index Calculations

### Output Page ID Mapping (Writer Kernel)

The writer uses a mapping from the tile-block's row index to an output RM page ID. The key calculations:

1. **`block_height_index`**: The global tile-row index being processed. For the i-th block on this core: `height_wise_input_block_start_index + i`.

2. **Row within block**: `j` iterates from 0 to `tile_height - 1` (0 to 31).

3. **`num_rows_already_processed`**: `block_height_index * tile_height + j` -- the global stick-row index.

4. **Output page ID**: `num_rows_already_processed * num_output_blocks_across_width + width_wise_output_block_start_index`. Each stick-row has `num_output_blocks_across_width` pages (one per output shard or 1 for interleaved). The `width_wise_output_block_start_index` offsets to the correct output shard column.

5. **Byte offset within page**: `num_cols_already_processed_in_first_output_block * output_element_size` for the first output block written in each row. This handles the case where an input shard straddles two output pages.

### L1 Read Address for Stick Data

```c
base_l1_read_addr = get_read_ptr(cb_id_out0);
current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;
```

Row `j` of the untilized block starts at a stride of `num_cols_per_input_block * output_element_size` bytes from the CB base.

---

## Memory Access Patterns

### Read Pattern (De-emphasized)

The reader reads tiles sequentially from DRAM using TensorAccessor, one tile at a time (interleaved) or an entire shard at once (sharded).

### Write Pattern (Primary Focus)

The writer's DRAM write pattern is **row-sequential with potential page-splitting**:

1. For each tile-row block, the writer processes all `tile_height` (32) rows sequentially.
2. For each row, it writes a contiguous chunk of `num_bytes_to_write` bytes to a single output page via `noc_async_write`.
3. When input and output sharding differ, a single input row may span multiple output pages. The inner `while` loop handles this by:
   - Writing as many columns as fit in the current output page.
   - Advancing to the next output page_id.
   - Continuing until all unpadded columns are written.
4. For the simple interleaved case (`num_output_blocks_across_width == 1`), each row maps to exactly one page, and the while loop executes exactly once per row.

**Write granularity**: One RM stick per noc_async_write call (or a partial stick if splitting across output pages).

**Barrier**: `noc_async_write_barrier()` is called once per tile-row block (after all 32 rows), not per individual write. This amortizes barrier cost.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D device grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` (e.g., 8x8 = 64 cores) |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` tile-rows (blocks) for full cores; `num_rows_per_cliff_core` for the cliff core |
| **Load balancing** | Near-equal distribution with at most 1 cliff core handling remainder |

For **interleaved input**: `split_blocks_for_tilize(grid_size, num_tiles_per_col)` distributes tile-rows across cores. Each full core gets `ceil(num_tiles_per_col / grid_area)` rows. The last core (cliff) handles `num_tiles_per_col % nblocks_per_core` rows if non-zero.

For **sharded input**: Each shard is processed by its owning core. `num_compute_cores = shard_grid.num_cores()`. No cliff core exists (sharding determines the distribution).

---

## Arguments

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output circular buffer index (c_16) |
| 1 | `output_stick_size` | uint32_t | Bytes per output RM page (= `output_page_width * element_size`) |
| 2 | `tile_height` | uint32_t | Tile height (32) -- number of stick rows per untilized block |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per input block width (used for `cb_wait_front`/`cb_pop_front` granularity) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per stick-row (1 for interleaved/height-sharded) |
| 5 | `output_element_size` | uint32_t | Bytes per element (2 for bfloat16, 4 for float32) |
| 6 | `num_cols_per_input_block` | uint32_t | Elements per input block width (= `num_tiles_per_input_block * tile_width`) |
| 7 | `num_cols_per_output_block` | uint32_t | Elements per output page width |
| 8+ | TensorAccessorArgs | (variable) | Bank mapping, shapes, coords for output buffer address generation |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-row blocks this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Global tile-row index where this core starts |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Actual (non-padding) columns to write per block (handles uneven width sharding) |
| 4 | `width_wise_output_block_start_index` | uint32_t | Starting output page column index for this core |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset within the first output page (for partial page writes) |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | Tiles per block width (`num_tiles_per_input_block`) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Compute Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks (tile-rows) to process on this core |

---

## Kernel Implementations

### Compute Kernel: Untilize Helper

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (compute threads) | N/A | CB c_0 | CB c_16 | `compute_kernel_lib::untilize` |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` (slow path) or `pack_untilize_variable_num_blocks.cpp` (fast path).

**Untilize Helper Signature** (from `untilize_helpers.hpp`):

```cpp
template <
    uint32_t block_width_tiles,   // tiles per row (compile-time)
    uint32_t input_cb,            // input CB index
    uint32_t output_cb,           // output CB index
    InitUninitMode,               // lifecycle: InitAndUninit (default)
    WaitMode,                     // sync: WaitBlock (default)
    ReconfigureRegisterDatatypeMode  // reconfig: NoReconfigure (default)
>
void untilize(uint32_t num_blocks);  // runtime: number of tile-rows
```

**Key behavior**: The `untilize` helper automatically selects between single-pass and block-based pack paths based on whether `block_width_tiles` exceeds `DEST_AUTO_LIMIT`. For each of `num_blocks` iterations:
- `cb_wait_front(input_cb, block_width_tiles)` -- waits for a full tile-row in the input CB.
- `cb_reserve_back(output_cb, block_width_tiles)` -- reserves output space.
- `pack_untilize_block` -- hardware-accelerated conversion of tiles to RM format.
- `cb_pop_front(input_cb, block_width_tiles)` / `cb_push_back(output_cb, block_width_tiles)`.

**PREREQUISITE**: `compute_kernel_hw_startup(input_cb, output_cb)` must be called before `untilize()`.

**Path selection** (in program factory, lines 232-244):
- **Fast path** (`pack_untilize`): Default. Hardware-accelerated untilize using the packer.
- **Slow path** (`untilize`): Used when `use_pack_untilize` is false, dtype is UINT16, or FLOAT32 with width >= `MAX_PACK_UNTILIZE_WIDTH`.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM (RM pages) | `noc_async_write` via TensorAccessor |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Key Logic**:

1. **TensorAccessor setup**: `TensorAccessor(dst_args, dst_addr, output_stick_size)` -- constructed with the output buffer address and RM page size (stick size). The `dst_args = TensorAccessorArgs<8>()` reads compile-time args starting at index 8.

2. **Per-block processing** (`write_tiles_in_current_block` lambda):
   - Waits for `num_tiles_per_input_block` tiles in output CB.
   - Gets `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- pointer to untilized data in L1.
   - Loops over `tile_height` (32) rows.
   - For each row, computes the L1 source address as `base_l1_read_addr + j * num_cols_per_input_block * output_element_size`.
   - Computes the output page_id and byte offset for DRAM destination.
   - Writes via `noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write)`.
   - After all rows: `noc_async_write_barrier()` then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`.

3. **Page-splitting logic**: When input blocks and output pages have different widths (e.g., different sharding), the inner `while` loop splits a single input row across multiple output pages:
   - `num_cols_to_write = min(remaining_input_cols, remaining_output_cols)`.
   - After each partial write, advance both the L1 read pointer and the output page_id.

### Reader Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (tiles) | CB c_0 | `noc_async_read` via TensorAccessor |

**File** (interleaved): `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
**File** (sharded): `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

---

## Implementation Notes

### Output CB Data Layout After Untilize

After the compute kernel's untilize operation, the data in CB c_16 is organized as **`tile_height` contiguous rows**, each `num_cols_per_input_block` elements wide. The CB still tracks pages in tile-sized units for push/pop accounting, but the actual byte layout is row-major. The writer leverages this by computing row addresses as `base + j * row_stride`.

### TensorAccessor for RM Output Writes

The TensorAccessor is constructed with `output_stick_size` as its page size. For interleaved RM output:
- Each output "page" is one full-width RM stick (one row of the tensor).
- `get_noc_addr(page_id, offset)` maps a page_id to a DRAM bank address, plus an optional byte offset within that page.
- The writer uses the offset parameter to handle partial-page writes when input blocks don't align with output pages.

### Uneven Sharding Support

The writer handles uneven input sharding via `num_unpadded_cols_per_input_block`. For the last shard in a row, the actual data width may be less than the shard width. Padding columns are skipped (not written).

### Double-Buffering Decision

Both input and output CBs use double-buffering when the core processes more than one block. This allows:
- Reader to fill block N+1 while compute processes block N (input CB double-buffering).
- Compute to fill block N+1 while writer drains block N (output CB double-buffering).

### FP32 and Integer Type Handling

For FLOAT32, INT32, UINT32 data types, the compute kernel define `DST_ACCUM_MODE` is set. This halves the DEST register capacity (from 8 to 4 tiles max for block-based path), affecting the sub-block splitting in the untilize helper.

---

## Relevance to layer_norm_rm Output Stage

For a `layer_norm_rm` operation that reads RM sticks, tilizes in-kernel, performs normalization, then untilizes back to RM output, the key patterns from this analysis are:

1. **Output CB sizing**: Size the output CB to `Wt * 2` tiles (double-buffered) where `Wt` is the number of tiles across the width. Use `output_single_tile_size` as page size.

2. **Untilize compute helper usage**: Call `compute_kernel_lib::untilize<Wt, input_cb, output_cb>(num_blocks)` after the normalization math. Prerequisite: `compute_kernel_hw_startup` at kernel start.

3. **Writer pattern for RM sticks**: After untilize fills the output CB with RM data, the writer:
   - `cb_wait_front(output_cb, Wt)` for one tile-row of untilized data.
   - `get_read_ptr(output_cb)` to get the L1 base address.
   - Loop over 32 rows, computing `base + j * W * element_size` for each row's L1 address.
   - Use `TensorAccessor::get_noc_addr(page_id)` to map each output stick to its DRAM location.
   - `noc_async_write` per stick, then `noc_async_write_barrier()` + `cb_pop_front`.

4. **TensorAccessor setup**: Host-side: `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)`. Kernel-side: `TensorAccessorArgs<base_idx>()` + `TensorAccessor(args, addr, stick_size)`.

5. **Page ID calculation**: For interleaved RM output with full-width sticks, `page_id = global_row_index` (one page per tensor row). The `height_wise_input_block_start_index` runtime arg tells each core where its first row is.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize operation work in tt-metal? Specifically, how does the compute kernel convert tiles (32x32) to row-major sticks? What is the difference between untilize and pack_untilize?"
   **Reason**: Needed to understand the compute kernel's tile-to-RM conversion mechanism.
   **Key Findings**: The untilize operation uses LLK APIs (`ckernel::untilize_block`, `ckernel::pack_untilize_block`) to convert tiled data to row-major. `pack_untilize` operates on data already in the DEST register (fast path), while `untilize` reads from input CB first. The `compute_kernel_lib::untilize` helper wraps these, automatically selecting the optimal path based on block width vs DEST capacity.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Needed to understand how TensorAccessor maps page IDs to NOC addresses for both reads and writes.
   **Key Information**: `TensorAccessor(args, base_addr, page_size)` is constructed from compile-time args. `get_noc_addr(page_id, offset)` maps a logical page to a physical DRAM bank address with optional byte offset. Host-side setup via `TensorAccessorArgs(*buffer).append_to(compile_time_args)`.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Needed to understand RM vs tiled page structure and interleaved bank distribution.
   **Key Information**: In RM layout, each page is one full row of the 2D tensor. Pages are distributed round-robin across DRAM banks in interleaved mode. Tiles are 32x32 with face-based internal layout (4 faces of 16x16).

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Needed the exact untilize helper signature and its internal behavior for use in a new operation.
   **Key Information**: Template params: `<block_width_tiles, input_cb, output_cb, InitUninitMode, WaitMode, ReconfigMode>`. Runtime param: `num_blocks`. Prerequisite: `compute_kernel_hw_startup(input_cb, output_cb)`. Auto-splits wide rows into DEST-sized sub-blocks. CB push/pop is handled internally by the helper.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Needed to understand the `create_cb` utility signature.
   **Key Information**: `create_cb(cb_id, program, core_spec, page_size, num_pages, data_format, buffer)` creates a circular buffer. Total size = `num_pages * page_size`. Returns `(cb_index, CBHandle)`. Optional `buffer` parameter for sharded CBs that share globally allocated memory.

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Needed to understand how `split_blocks_for_tilize` distributes work.
   **Key Information**: `split_blocks_for_tilize(grid_size, nblocks)` returns `{ncores, all_cores, core_range, cliff_core_range, nblocks_per_core, nblocks_per_core_cliff}`. Distributes `nblocks` tile-rows across `grid_area` cores with `ceil(nblocks/grid_area)` per full core and a single cliff core for remainder.
