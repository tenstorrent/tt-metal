# Untilize Multi-Core Implementation Analysis (Output Stage Reference)

## Overview

The untilize operation converts tensor data from **tiled layout** (32x32 tiles with face-based internal ordering) to **row-major layout** (contiguous rows/sticks). This analysis focuses on the **output stage patterns** -- the output CB sizing, the compute-to-output-CB untilize helper, and the writer kernel that extracts row-major sticks from the output CB and writes them to DRAM. These patterns serve as a reference for the `layer_norm_rm` operation's output stage (compute -> untilize -> RM output to DRAM).

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = `num_tiles_per_row` tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (= tiles spanning the width for this core's shard/column) |
| **Total units** | `num_tiles_per_col` blocks (= total tile rows across the tensor height) |
| **Loop structure** | Outer: blocks (tile-rows assigned to this core). Inner: tile_height rows within each block, then columns within each row. |

A single "input block" is one tile-row: a horizontal strip of `num_tiles_per_input_block` tiles, each `tile_height` rows tall. The compute kernel untilizes one block at a time, producing `tile_height` contiguous row-major sticks in the output CB. The writer then drains these sticks to DRAM.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] (flattened to 2D: height x width) | [N, C, H, W] (same logical shape) |
| **Dimension convention** | Last dim = W (width in elements) | Last dim = W (width in elements) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with face ordering) | ROW_MAJOR_LAYOUT (contiguous row sticks) |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED (primary path), SHARDED possible |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 | Same as input |

### Output Tensor Details (Focus)

The output tensor uses **row-major layout with interleaved memory**. Each "page" in the output buffer corresponds to one row-major stick:
- **Page width** = `output_page_width` elements (= full tensor width for interleaved; = shard width for width/block sharded output)
- **Page size (stick size)** = `output_page_width * output_element_size` bytes
- Pages are distributed round-robin across DRAM banks via the interleaved memory layout

### Layout Transformation

The core transformation is: **Tiled -> Row-major**. The compute kernel's `pack_untilize_block` (or `untilize_block`) reads tiles from the input CB and writes row-major data to the output CB. After one block of `num_tiles_per_input_block` tiles is untilized, the output CB contains `tile_height` contiguous sticks, each `num_tiles_per_input_block * tile_width * element_size` bytes wide.

## Data Flow Pattern

### Full Pipeline (Input -> Output)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved tiles) | CB c_0 (input) | reserve_back, push_back (1 tile at a time) |
| 2 | Compute | CB c_0 (input tiles) | CB c_16 (output RM sticks) | wait_front, pop_front (block), reserve_back, push_back (block) |
| 3 | Writer | CB c_16 (output RM sticks) | DRAM (interleaved RM pages) | wait_front, pop_front (block) |

### Output Stage Detail (Focus for layer_norm_rm)

The output stage consists of two cooperating components:

1. **Compute kernel (untilize helper)**: Calls `compute_kernel_lib::untilize<>()` which internally:
   - Waits for `block_width_tiles` tiles in input CB (`cb_wait_front`)
   - Reserves `block_width_tiles` worth of space in output CB (`cb_reserve_back`)
   - Calls `pack_untilize_block` to convert tile data to row-major sticks in output CB
   - Pops consumed tiles from input CB (`cb_pop_front`)
   - Pushes produced sticks to output CB (`cb_push_back`)
   - Repeats for `num_blocks` (= `num_input_blocks_to_process` per core)

2. **Writer kernel**: For each block:
   - Waits for `num_tiles_per_input_block` tiles worth of RM data in output CB (`cb_wait_front`)
   - Gets `base_l1_read_addr` via `get_read_ptr(cb_id_out0)` -- this is the L1 address of the untilized RM data
   - Iterates over `tile_height` rows, each `num_cols_per_input_block * element_size` bytes apart
   - For each row, computes the output page_id and writes the stick (or sub-stick) to DRAM via `noc_async_write`
   - Calls `noc_async_write_barrier()` then `cb_pop_front` to release the CB space

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_input_block * 2` tiles (double-buffered) or `num_tiles_per_input_block` (single block case) | `num_tiles_per_input_block` tiles | Double (if >=2 blocks/core) or Single (if 1 block/core) | Reader | Compute | Block |
| c_16 | cb_output | Output RM stick staging | `num_tiles_per_input_block * 2` tiles (double-buffered) or `num_tiles_per_input_block` (single block case) | `num_tiles_per_input_block` tiles | Double (if >=2 blocks/core) or Single (if 1 block/core) | Compute | Writer | Block |

### Output CB Sizing Logic (Key for layer_norm_rm)

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

**Key insight**: The output CB capacity is measured in **tile-sized units** even though it holds row-major data. After `pack_untilize_block`, each "tile's worth" of output CB space holds `tile_height` row-major sticks of `tile_width * element_size` bytes each. The total output CB space per block is `num_tiles_per_input_block * tile_size` bytes, which holds `tile_height` full-width sticks.

The `create_cb` helper creates the CB:
```cpp
auto [output_cb_index, cb_output] = create_cb(
    tt::CBIndex::c_16, program, compute_core_range,
    output_single_tile_size, output_cb_num_tiles, output_cb_data_format);
```

Parameters: `page_size = output_single_tile_size`, `num_pages = output_cb_num_tiles`. The total CB size = `output_single_tile_size * output_cb_num_tiles`.

## Pipeline Pattern Summary

- **Input CB (c_0)**: Double-buffered when core processes 2+ blocks (reader can fill next block while compute processes current). Single-buffered when only 1 block.
- **Output CB (c_16)**: Double-buffered when core processes 2+ blocks (compute can fill next block while writer drains current). Single-buffered when only 1 block.
- Double-buffering enables overlap between compute and writer stages.

## Untilize Helper Signature and Usage (Key for layer_norm_rm)

The compute kernel uses a unified helper from `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`:

```cpp
// Template signature
template <
    uint32_t block_width_tiles,     // tiles per row (compile-time)
    uint32_t input_cb,              // input CB index (tiled data)
    uint32_t output_cb,             // output CB index (RM data, must differ from input_cb)
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure
>
ALWI void untilize(uint32_t num_blocks);  // num_blocks is runtime
```

**Usage in untilize compute kernel** (`untilize_variable_num_blocks.cpp`):
```cpp
compute_kernel_hw_startup(src_cb_id, out_cb_id);  // MUST call first
compute_kernel_lib::untilize<
    per_core_block_tile_cnt,  // compile-time: tiles per row
    src_cb_id,                // compile-time: input CB (c_0)
    out_cb_id,                // compile-time: output CB (c_16)
    InitUninitMode::InitAndUninit,
    WaitMode::WaitBlock,
    ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);  // runtime: blocks to process
```

**For layer_norm_rm usage**:
- The untilize helper can be called at the end of a compute kernel after the normalization math is complete
- Use `InitUninitMode::InitOnly` / `Neither` / `UninitOnly` if calling untilize in a loop with other compute between calls
- `WaitMode::WaitBlock` is standard (compute waits for its own data per block)
- `WaitMode::NoWait` can be used if the data is already in the CB (e.g., compute just produced it)
- **The input CB for untilize must contain tiled data; the output CB will contain RM data**

### Internal Dispatch Logic

The helper automatically selects between two paths based on `block_width_tiles` vs `DEST_AUTO_LIMIT`:

1. **Single-pass pack untilize** (when `block_width_tiles <= DEST_AUTO_LIMIT`): Processes the full tile-row in one shot. This is the optimal/fast path.
2. **Block-based pack untilize** (when `block_width_tiles > DEST_AUTO_LIMIT`): Splits the tile-row into sub-blocks that fit in the DEST register. Processes sub-blocks sequentially within each row.

The `DEST_AUTO_LIMIT` is derived from `DST_ACCUM_MODE` and hardware generation. For BFLOAT16 without accumulation mode, it is typically 8 tiles; for FLOAT32 with accumulation mode, typically 4 tiles.

## Index Calculations

### Writer Kernel Page ID Calculation (Key for layer_norm_rm)

The writer kernel computes the DRAM page_id for each row-major stick using a straightforward formula:

```cpp
// For each row j within a tile-height block at block_height_index:
uint32_t num_rows_already_processed = block_height_index * tile_height + j;
uint32_t num_pages_already_processed_in_previous_rows =
    num_rows_already_processed * num_output_blocks_across_width;
uint32_t output_page_id =
    num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;
```

For the simple interleaved case (`num_output_blocks_across_width = 1`, `width_wise_output_block_start_index = 0`):
- `output_page_id = (block_height_index * tile_height + j)` -- simply the global row index.
- Each page is one full-width row-major stick.

### TensorAccessor for Output

```cpp
// Host-side: append TensorAccessorArgs to compile-time args
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

// Device-side (writer kernel):
constexpr auto dst_args = TensorAccessorArgs<8>();  // starts at compile-time arg index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);

// Get NoC address for a specific page with byte offset:
uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
```

The TensorAccessor maps logical `page_id` to the correct DRAM bank and offset within that bank, handling the round-robin interleaved distribution. The optional `byte_offset` parameter allows writing to a specific position within a page (used when output sharding splits a row across multiple output blocks).

## Memory Access Patterns

### Read Pattern (De-emphasized)

The reader reads tiles sequentially from DRAM, one tile at a time, using a linear page_id range.

### Write Pattern (Focus for layer_norm_rm)

The writer uses a **row-sequential, block-at-a-time** pattern:

1. Wait for one full block of untilized data in the output CB (`tile_height` sticks, each `num_cols_per_input_block * element_size` bytes wide)
2. For each of the `tile_height` rows within the block:
   - Compute the L1 read address: `base_l1_read_addr + j * num_cols_per_input_block * element_size`
   - Compute the target page_id (global row index)
   - Issue `noc_async_write(l1_addr, noc_addr, num_bytes_to_write)` for the stick
3. Barrier after all `tile_height` writes: `noc_async_write_barrier()`
4. Pop the block from CB: `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`

**For the simple interleaved case** (no width/block sharding), the inner while loop executes exactly once per row (the entire stick fits in one output page), making the pattern:

```
For each block:
    wait for block in output CB
    For j in 0..tile_height-1:
        l1_addr = base + j * stick_bytes
        page_id = global_row_index
        noc_write(l1_addr, get_noc_addr(page_id), stick_bytes)
    barrier
    pop block from CB
```

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of cores) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` (from `split_blocks_for_tilize`) |
| **Work per core (full)** | `num_rows_per_full_core` tile-rows |
| **Work per core (cliff)** | `num_rows_per_cliff_core` tile-rows (0 or 1 cliff core) |
| **Load balancing** | Approximately equal; ceil-division with at most 1 cliff core |

The `split_blocks_for_tilize(grid_size, num_tiles_per_col)` function distributes `num_tiles_per_col` tile-rows across available cores:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- `ncores = ceil(num_tiles_per_col / nblocks_per_core)`
- Last core may be a "cliff" core processing `num_tiles_per_col % nblocks_per_core` blocks

Each core is assigned a contiguous range of tile-rows. The `tile_start_index` variable tracks which tile each core begins reading from.

## Arguments

### Compile-Time Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output CB index (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output page/stick in bytes (`output_page_width * element_size`) |
| 2 | tile_height | uint32_t | Height of a tile (32 for standard tiles) |
| 3 | num_tiles_per_input_block | uint32_t | Number of tiles across the width of one input block |
| 4 | num_output_blocks_across_width | uint32_t | Number of output pages per row (1 for interleaved) |
| 5 | output_element_size | uint32_t | Size of one element in bytes (2 for BF16, 4 for FP32) |
| 6 | num_cols_per_input_block | uint32_t | Total columns in one input block (`num_tiles_per_input_block * tile_width`) |
| 7 | num_cols_per_output_block | uint32_t | Columns per output page (`output_page_width`) |
| 8+ | TensorAccessorArgs | (variable) | Appended by `TensorAccessorArgs(*dst_buffer).append_to(...)` |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |
| 1 | num_input_blocks_to_process | uint32_t | Number of tile-rows this core processes |
| 2 | height_wise_input_block_start_index | uint32_t | Starting tile-row index for this core |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Actual data columns (handles uneven sharding) |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page column index (0 for interleaved) |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Column offset within first output page (0 for interleaved) |

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Tiles per block (= `num_tiles_per_input_block`) |
| 1 | src_cb_id | uint32_t | Input CB index (c_0) |
| 2 | out_cb_id | uint32_t | Output CB index (c_16) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) to process on this core |

## Kernel Implementations

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 (RM sticks) | DRAM (interleaved RM pages) | Extract sticks from CB, write to DRAM per-page |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - Uses a lambda `write_tiles_in_current_block` that processes one block at a time
  - `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- waits for compute to produce one block of RM data
  - `get_read_ptr(cb_id_out0)` -- gets the L1 base address of the untilized data
  - Row addressing: `base_l1_read_addr + j * num_cols_per_input_block * output_element_size`
  - Uses `TensorAccessor::get_noc_addr(page_id, byte_offset)` for address resolution
  - Inner while-loop handles the case where input block width spans multiple output pages (for width/block sharded output). For simple interleaved output, this loop runs exactly once per row.
  - `noc_async_write_barrier()` after all rows in a block, then `cb_pop_front`

### Compute Kernel (Untilize)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 (tiles) | CB c_16 (RM sticks) | pack_untilize_block (tile -> RM) |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` (slow path) or `pack_untilize_variable_num_blocks.cpp` (fast path)
- **Key Logic**: Both variants call `compute_kernel_lib::untilize<>()` with identical template parameters. The "fast" vs "slow" distinction is now internal to the unified helper (both use pack_untilize internally, the "slow" kernel name is legacy).
- **Prerequisite**: `compute_kernel_hw_startup(src_cb_id, out_cb_id)` must be called before any untilize calls.

### Reader Kernel (De-emphasized)

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- Reads tiles one-at-a-time from DRAM using TensorAccessor, pushing each tile into CB c_0.

## Implementation Notes

### Output Stage Patterns for layer_norm_rm Reuse

1. **Output CB sizing**: Size the output CB to hold `num_tiles_per_row` tiles (one tile-row) worth of RM data. Double-buffer if the core processes 2+ blocks. Use `output_single_tile_size` as the page size and `num_tiles * (1 or 2)` as num_pages.

2. **Compute-side untilize**: At the end of the compute kernel (after layer norm math), call `compute_kernel_lib::untilize<>()` to convert the tile result to RM in the output CB. The helper manages all CB synchronization internally (`cb_wait_front` on input, `cb_reserve_back`/`cb_push_back` on output).

3. **Writer-side RM stick extraction**: The writer reads from the output CB using `get_read_ptr()` to get the L1 base address. Row j within the block is at offset `j * num_cols * element_size` from that base. Each row is written to DRAM as one page using `noc_async_write` with TensorAccessor-resolved addresses.

4. **TensorAccessor setup**: On the host, call `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)`. On the device, instantiate `TensorAccessorArgs<N>()` at the correct compile-time arg offset, then `TensorAccessor(args, addr, page_size)`.

5. **Page ID for interleaved RM output**: For a simple case (no output sharding), `page_id = global_row_index`. The TensorAccessor handles bank mapping.

6. **Barrier placement**: Call `noc_async_write_barrier()` after writing all `tile_height` sticks in a block, before `cb_pop_front`. This ensures all writes are complete before releasing CB space for the next block.

### Compute Kernel Defines

- `DST_ACCUM_MODE=1` is set for INT32, UINT32, and FLOAT32 data types
- `fp32_dest_acc_en` enables FP32 accumulation in DEST register
- The `use_pack_untilize` flag determines which compute kernel file is selected, but both now invoke the same unified helper

### Fast vs Slow Untilize Selection

The program factory selects the "slow" kernel when:
- `use_pack_untilize` is false, OR
- Data type is UINT16, OR
- Data type is FLOAT32 and `num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH`

Otherwise the "fast" kernel is used. For layer_norm_rm with BFLOAT16/FLOAT32 output, the fast path will typically be selected.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the TensorAccessor work for writing data to DRAM in writer kernels?"
   **Reason**: Needed to understand how the writer kernel maps page_ids to physical DRAM addresses.
   **Key Findings**: TensorAccessor is constructed with (TensorAccessorArgs, base_addr, page_size). `get_noc_addr(page_id, byte_offset)` returns the full NoC address including bank mapping. `byte_offset` allows writing to a sub-page position.

2. **Query**: "How does the untilize operation work in tt-metal? What is pack_untilize vs standard untilize?"
   **Reason**: Needed to understand what the compute kernel does to convert tile data to row-major.
   **Key Findings**: Both paths ultimately use `pack_untilize_block` which unpacks tiles from input CB, moves data through DEST register, and packs in row-major format to output CB. The distinction between "fast" and "slow" is now largely internal to the unified helper.

3. **Query**: "After pack_untilize_block writes to the output CB, what is the data layout?"
   **Reason**: Critical for understanding what the writer kernel reads from the output CB.
   **Key Findings**: The output CB contains `tile_height` contiguous row-major sticks, each `num_tiles * tile_width * element_size` bytes wide. Sticks are laid out sequentially in the CB.

4. **Query**: "What is the split_blocks_for_tilize function and how does it work?"
   **Reason**: Needed to understand core distribution strategy.
   **Key Findings**: Distributes tile-rows across cores using ceil-division. Returns full core range and optional cliff core range. The cliff core processes the remainder blocks.

5. **Query**: "What do get_read_ptr and get_write_ptr return for a circular buffer?"
   **Reason**: Needed to understand how the writer kernel accesses data in the output CB.
   **Key Findings**: `get_read_ptr(cb)` returns the L1 byte address of the current read position in the CB. Used after `cb_wait_front` to access data pushed by the producer. `cb_pop_front` advances the read pointer.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding page structure for row-major tensors and interleaved memory distribution.
   **Key Information**: For row-major tensors, each page is one row. Pages are distributed round-robin across DRAM banks in interleaved layout. Tiled tensors use 32x32 tiles with 16x16 face ordering.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize compute helper API and its dispatch logic.
   **Key Information**: The helper selects between single-pass and block-based paths based on `block_width_tiles` vs `DEST_AUTO_LIMIT`. Supports `WaitMode::NoWait` for cases where data is already in the CB. Supports `InitUninitMode` for lifecycle control in multi-call scenarios.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper function signature.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer)` creates a circular buffer. Total CB size = `page_size * num_pages`. If `buffer` is non-null, the CB is mapped to that buffer (used for sharded input).

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` 1D work distribution function.
   **Key Information**: Returns `BlockSplit{ncores, all_cores, core_range, cliff_core_range, nblocks_per_core, nblocks_per_core_cliff}`. Uses ceil-division for even distribution with at most one cliff core.
