# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize multi-core operation converts tensor data from **tiled layout** (32x32 tiles with face sub-structure) to **row-major layout** (linear sticks). It reads tiles from DRAM (interleaved) or L1 (sharded), runs the pack_untilize compute kernel to reorder tile data into row-major sticks in the output CB, then the writer kernel writes those sticks out as row-major pages to DRAM.

**Focus of this analysis**: Output stage -- how row-major sticks are extracted from tiles and written to DRAM interleaved memory. This serves as a reference for implementing the output stage of a new `layer_norm_rm` operation.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row of tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (= `tensor_width / tile_width`) |
| **Total units** | `num_tiles_per_col` blocks (= `tensor_height / tile_height`) |
| **Loop structure** | Each core processes `num_input_blocks_per_full_core` consecutive tile-rows. A cliff core may process fewer. |

One "input block" is a single tile-row spanning the full width of the tensor (for interleaved input) or the shard width (for sharded input). The compute kernel untilizes one block at a time, producing `tile_height` (typically 32) row-major sticks per block.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D (flattened to 2D: height x width) |
| **Dimension convention** | Last dim = width, all others collapsed into height |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with face sub-structure) |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, UINT16, INT32, UINT32 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Last dim = width, all others collapsed into height |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED (may also be width/block sharded) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded output) |
| **Data type** | Same as input |

### Layout Transformations

The compute kernel converts tiled data to row-major. Specifically, `pack_untilize_block` reorders tile face data (4 faces of 16x16 within each 32x32 tile) into contiguous row-major sticks. After compute, the output CB contains `tile_height` (32) contiguous sticks, each of width `num_tiles_per_input_block * tile_width` elements.

**Key insight for output stage reuse**: The output CB data after pack_untilize is already in row-major stick format. The writer kernel's job is to take these sticks from L1 and write them as pages to DRAM.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) or L1 (sharded) | CB_in (c_0) | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB_in (c_0) | CB_out (c_16) | `cb_wait_front(input)`, `cb_reserve_back(output)`, `pack_untilize_block`, `cb_pop_front(input)`, `cb_push_back(output)` |
| 3 | Writer | CB_out (c_16) | DRAM (interleaved) | `cb_wait_front`, `get_read_ptr`, `noc_async_write`, `noc_async_write_barrier`, `cb_pop_front` |

### Detailed Writer Data Flow (Output Stage Focus)

For each input block (tile-row) processed by the compute kernel:

1. **Wait**: `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- wait for compute to produce one full tile-row of untilized data.
2. **Get base L1 address**: `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- pointer to start of row-major stick data in L1.
3. **Iterate over `tile_height` rows** (j = 0..31): Each row is one stick of width `num_cols_per_input_block * output_element_size` bytes in L1.
   - Compute L1 read address: `base_l1_read_addr + j * num_cols_per_input_block * output_element_size`
   - Compute output page_id: `(block_height_index * tile_height + j) * num_output_blocks_across_width + width_wise_output_block_start_index`
   - Write partial or full sticks to one or more output pages via `noc_async_write`.
4. **Barrier**: `noc_async_write_barrier()` -- ensures all writes for this block have departed.
5. **Release**: `cb_pop_front(cb_id_out0, num_tiles_per_input_block)` -- free CB space.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_input_block * 2` tiles (interleaved), or entire shard (sharded) | `num_tiles_per_input_block` tiles | Double (interleaved, multi-block), Single (interleaved, single-block or sharded) | Reader | Compute | Block |
| c_16 | cb_output | Output RM stick staging | `num_tiles_per_input_block * 2` tiles (multi-block), or `num_tiles_per_input_block` tiles (single-block) | `num_tiles_per_input_block` tiles | Double (multi-block), Single (single-block) | Compute | Writer | Block |

### Output CB Sizing Details (Key for layer_norm_rm)

The output CB capacity is determined by the number of blocks per core:

```
if (num_input_blocks_per_full_core == 1):
    output_cb_num_tiles = num_tiles_per_input_block          // Single-buffered
else:
    output_cb_num_tiles = num_tiles_per_input_block * 2      // Double-buffered
```

**Rationale**: When a core processes only 1 block, there is no overlap opportunity, so single-buffering suffices. When processing 2+ blocks, double-buffering allows the compute kernel to produce block N+1 while the writer writes block N.

**Physical size of each output tile slot**: `output_single_tile_size = tt::tile_size(output_cb_data_format)`. Note that even though the output is logically row-major, the CB is still sized in "tile" units because the pack_untilize writes `tile_height * tile_width * element_size` bytes per tile position.

**After pack_untilize, the output CB data layout is**:
- For a block of `num_tiles_per_input_block` tiles across one tile-row:
  - 32 contiguous sticks (rows), each of width `num_tiles_per_input_block * tile_width` elements
  - Stick j starts at byte offset `j * num_cols_per_input_block * output_element_size` from the CB read pointer

## Pipeline Pattern Summary

- **Input CB (c_0)**: Double-buffered for interleaved multi-block case. Enables reader to prefetch the next tile-row while compute processes the current one.
- **Output CB (c_16)**: Double-buffered for multi-block case. Enables compute to produce the next block while writer writes the current one.
- **Single-block case**: Both CBs are single-buffered since there is only one block to process.

## Index Calculations

### Output Page ID Calculation (Writer Kernel)

The writer maps each stick (row within a tile-row block) to an output page_id for DRAM interleaved writes:

```
// For each stick j within a block at height_wise index block_height_index:
num_rows_already_processed = block_height_index * tile_height + j
num_pages_already_processed_in_previous_rows = num_rows_already_processed * num_output_blocks_across_width
output_page_id = num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index
```

**For the simple interleaved-input, interleaved-output case** (most relevant for layer_norm_rm):
- `num_output_blocks_across_width = 1` (one output page per row)
- `width_wise_output_block_start_index = 0`
- So: `output_page_id = block_height_index * tile_height + j`
- This means page 0 = row 0, page 1 = row 1, etc. -- one page per row-major stick.

### TensorAccessor Address Resolution

The writer uses `TensorAccessor` to convert `output_page_id` into a physical NOC address:

```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // compile-time args start at index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
```

- `output_stick_size` = `num_cols_per_output_block * output_element_size` is the page size for address generation.
- For interleaved output: pages are distributed round-robin across DRAM banks. `get_noc_addr(page_id, offset)` computes the bank index and bank-local offset, then translates to a NOC address with an additional byte offset within the page.

## Memory Access Patterns

### Read Pattern (Reader -- de-emphasized)

- **Interleaved input**: Sequential tile reads via `noc_async_read`, one tile at a time with `cb_reserve_back(1)` and `cb_push_back(1)`. Tiles are read from contiguous page_ids starting at `tile_start_index`.
- **Sharded input**: No NoC reads needed; the input CB is backed by the shard buffer directly.

### Write Pattern (Writer -- primary focus)

**Access ordering**: The writer processes blocks sequentially. Within each block, it iterates over `tile_height` sticks (rows 0..31).

**For simple interleaved output** (1 output block across width):
- Each stick is written as a complete row-major page to DRAM.
- Write size = `output_stick_size` = `tensor_width * output_element_size` bytes.
- Pages are written in ascending page_id order (row 0, row 1, ..., row 31 for each block).
- Across DRAM banks: pages are interleaved round-robin.

**For sharded or multi-block output** (multiple output blocks across width):
- A single input block stick may span multiple output pages (or a fraction of one).
- The writer uses a `while` loop to split the stick across output page boundaries.
- `num_cols_already_processed_in_first_output_block` handles the case where the current core's data starts mid-page.

**Write granularity**: `noc_async_write(l1_addr, noc_addr, num_bytes)` with num_bytes = `num_cols_to_write * output_element_size`. For the common case, this is the full stick.

**Barrier strategy**: `noc_async_write_barrier()` is called once per block (after all `tile_height` stick writes). This ensures all writes for the block have departed before the CB space is freed.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` tile-rows (blocks) for full cores; `num_rows_per_cliff_core` for the cliff core |
| **Load balancing** | Near-equal distribution with at most 1 cliff core handling remainder |

### Work Splitting (Interleaved Input)

`split_blocks_for_tilize(grid_size, num_tiles_per_col)` distributes `num_tiles_per_col` tile-rows across available cores:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- `ncores = ceil(num_tiles_per_col / nblocks_per_core)`
- If `num_tiles_per_col % nblocks_per_core != 0`, the last core is a "cliff core" with fewer blocks.

### Work Splitting (Sharded Input)

Each shard maps to one core. The number of cores equals the number of shards. There is no cliff core. Edge shards may have fewer valid tile-rows (handled by `num_input_blocks_to_process` runtime arg) or fewer valid columns (handled by `num_unpadded_cols_per_input_block`).

## Arguments

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Bytes per output page: `num_cols_per_output_block * output_element_size` |
| 2 | `tile_height` | uint32_t | Height of tile (typically 32) |
| 3 | `num_tiles_per_input_block` | uint32_t | Number of tiles in one tile-row (determines CB wait size) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per row (1 for simple interleaved) |
| 5 | `output_element_size` | uint32_t | Bytes per element (2 for BF16, 4 for FP32) |
| 6 | `num_cols_per_input_block` | uint32_t | Width of input block in elements: `num_tiles_per_input_block * tile_width` |
| 7 | `num_cols_per_output_block` | uint32_t | Width of output page in elements (= tensor_width for interleaved) |
| 8+ | TensorAccessorArgs | various | Bank mapping, addresses, shapes for output buffer |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Base address of output buffer in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-rows this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Starting tile-row index for this core |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Valid columns to write (handles padding in last width-wise shard) |
| 4 | `width_wise_output_block_start_index` | uint32_t | Starting output page column index |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset within first output page (for sharded output, 0 for simple interleaved) |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | Number of tiles per block (width in tiles) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Compute Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks (tile-rows) this core processes |

### Reader Compile-Time Arguments (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_in0` | uint32_t | Input CB index (c_0) |
| 1+ | TensorAccessorArgs | various | Bank mapping for input buffer |

### Reader Runtime Arguments (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Base address of input buffer |
| 1 | `num_tiles` | uint32_t | Total tiles to read: `num_tiles_per_input_block * num_input_blocks_to_process` |
| 2 | `start_page_id` | uint32_t | First tile page_id for this core |

## Kernel Implementations

### Writer Kernel (Primary Focus)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB_out (c_16) | DRAM (interleaved) | Write RM sticks to DRAM pages |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - Uses `TensorAccessor` with `TensorAccessorArgs<8>()` for address resolution. The `<8>` indicates compile-time args for the accessor start at index 8 in the compile-time arg list.
  - Inner lambda `write_tiles_in_current_block` processes one tile-row at a time.
  - For each of `tile_height` sticks, computes the output page_id and writes the stick (or portions of it) via `noc_async_write`.
  - The `while` loop inside the stick processing handles the case where one input stick spans multiple output pages (relevant for width/block sharded output).
  - Barrier is per-block, not per-stick: all sticks in one tile-row are issued before `noc_async_write_barrier()`.
  - **For simple interleaved output**: The inner while loop executes exactly once per stick (since `num_cols_per_input_block == num_cols_per_output_block` and `num_cols_already_processed_in_first_output_block == 0`).

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB_in (c_0) | CB_out (c_16) | pack_untilize (tile to RM conversion) |

- **File (fast path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
- **File (slow path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **Key Logic**:
  - Both files call `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt)`.
  - The unified `untilize` helper automatically selects between pack_untilize (hardware-accelerated) and standard untilize based on whether the block width exceeds DEST register capacity.
  - The fast path is selected unless: `!use_pack_untilize`, dtype is UINT16, or (dtype is FLOAT32 and width too large for pack_untilize).
  - `DST_ACCUM_MODE` define is set for INT32/UINT32/FLOAT32 dtypes, reducing max block column tiles from 8 to 4.

### Reader Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (interleaved) | CB_in (c_0) | Read tiles from DRAM |

- **File (interleaved)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **File (sharded)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

## Implementation Notes

### Key Design Patterns for Output Stage Reuse

1. **Output CB as row-major staging area**: The output CB (c_16) serves as an intermediate buffer holding row-major sticks. The compute kernel converts tiles to sticks and writes them here. The writer reads sticks from here. This is the fundamental pattern for any operation that produces row-major output from tiled compute.

2. **TensorAccessor for address generation**: The writer uses `TensorAccessor` initialized from compile-time args (starting at index 8) with the output buffer's base address and page size (stick size). `get_noc_addr(page_id, byte_offset)` resolves interleaved bank mapping automatically.

3. **Page = one row-major stick**: For interleaved row-major output, each page is one complete row of the tensor. Page_id corresponds directly to row index. `output_stick_size = tensor_width * element_size`.

4. **Per-block barrier**: Writes for all `tile_height` sticks within a block are issued before a single `noc_async_write_barrier()`. This batches the NoC operations efficiently.

5. **Block-level CB synchronization**: The writer does `cb_wait_front` for a full block of tiles (`num_tiles_per_input_block`), not individual tiles. After processing all sticks in the block, it does `cb_pop_front` for the same count. This matches the compute kernel's `cb_push_back` granularity.

6. **Double-buffering condition**: Output CB is double-buffered only when `num_input_blocks_per_full_core > 1`. The cost is 2x CB memory but enables compute/writer overlap.

### Simplifications for layer_norm_rm

For a `layer_norm_rm` operation producing interleaved row-major output:
- `num_output_blocks_across_width = 1` (no width sharding on output)
- `width_wise_output_block_start_index = 0` (all cores start at column 0)
- `num_cols_already_processed_in_first_output_block = 0` (no partial page offsets)
- `num_cols_per_input_block = num_cols_per_output_block = tensor_width`
- The writer's inner `while` loop executes exactly once per stick
- Page_id = absolute row index

### untilize Helper Signature and Usage

The compute-side `untilize` helper is defined in `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`:

```cpp
template <
    uint32_t block_width_tiles,   // tiles per row (compile-time)
    uint32_t input_cb,            // input CB index
    uint32_t output_cb,           // output CB index
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
ALWI void untilize(uint32_t num_blocks);  // num_blocks = runtime
```

Key usage patterns:
- `WaitBlock` (default): The helper calls `cb_wait_front(input_cb, block_width_tiles)` per block.
- `WaitUpfront`: Waits for all tiles before processing (used by GroupNorm pattern).
- `NoWait`: Caller manages synchronization externally (useful when data is already in CB from a prior compute step).
- `InitOnly` / `UninitOnly` / `Neither`: For chaining multiple untilize calls without redundant init/uninit.

For `layer_norm_rm`, the `WaitUpfront` or `NoWait` modes may be relevant if the data is produced by a preceding compute step in the same kernel, rather than read from a separate CB.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor and TensorAccessorArgs work in tt-metal kernels?"
   **Reason**: Needed to understand how the writer kernel maps page_ids to physical DRAM addresses.
   **Key Findings**: TensorAccessor abstracts interleaved bank mapping. `get_noc_addr(page_id)` computes bank index via round-robin, then translates to NOC x,y coordinates + bank-local offset. TensorAccessorArgs encodes bank layout as compile-time args starting at a configurable index.

2. **Query**: "How does the untilize compute operation work in tt-metal?"
   **Reason**: Needed to understand what the compute kernel produces in the output CB.
   **Key Findings**: `pack_untilize` is the fast hardware-accelerated path. It reads tiled data from the input CB, converts it to row-major format in the DEST register, and packs it into the output CB. The result is contiguous row-major sticks. Standard `untilize` is a slower fallback.

3. **Query**: "How does split_blocks_for_tilize work and what is a cliff core?"
   **Reason**: Needed to understand core distribution and work splitting.
   **Key Findings**: `split_blocks_for_tilize` distributes tile-rows (blocks) across cores. `nblocks_per_core = ceil(total / grid_area)`. A "cliff core" handles the remainder when work doesn't divide evenly. Returns separate CoreRangeSet for full cores and cliff core.

4. **Query**: "After pack_untilize, how is data laid out in the output CB?"
   **Reason**: Critical for understanding the L1 memory layout the writer reads from.
   **Key Findings**: Data is in row-major format. For a block of N tiles across one tile-row, the output CB contains `tile_height` contiguous sticks, each of width `N * tile_width` elements. Sticks are laid out sequentially in memory.

5. **Query**: "How does noc_async_write work in writer kernels?"
   **Reason**: Needed to understand the write mechanism and barrier semantics.
   **Key Findings**: `noc_async_write(l1_addr, noc_addr, size)` initiates an async transfer from L1 to the NOC destination. `noc_async_write_barrier()` blocks until all writes have departed (not completed). The barrier does not guarantee delivery, only departure from the core.

6. **Query**: "What does TensorAccessor::get_noc_addr(page_id, offset) do with two arguments?"
   **Reason**: The writer kernel calls `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` with an offset parameter.
   **Key Findings**: The offset parameter adds a byte offset within the page to the resulting NOC address. This enables writing to a specific position within a page, not just the start. For simple interleaved output the offset is always 0.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding host-side setup and device-side usage of TensorAccessor.
   **Key Information**: `TensorAccessorArgs(buffer).append_to(compile_args)` adds bank layout as compile-time args. Device side: `TensorAccessorArgs<start_idx>()` recovers them. `get_noc_addr(page_id)` returns full NOC address. Zero-cost construction when rank is static.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major page layout and interleaved bank distribution.
   **Key Information**: Row-major layout: each row = one page. Interleaved: pages distributed round-robin across DRAM banks. Tiled layout: each page = one 32x32 tile (internally 4 faces of 16x16).

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize compute helper API.
   **Key Information**: Template-based compile-time dispatch. Automatically splits wide rows into sub-blocks if they exceed DEST capacity. Supports WaitBlock, WaitUpfront, and NoWait synchronization modes. InitUninitMode controls lifecycle for chained calls.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` work distribution function.
   **Key Information**: Returns `BlockSplit` with full_core_range, cliff_core_range, nblocks_per_core, and nblocks_per_core_cliff. Cliff core is the last core and handles the remainder blocks.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper function.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer)` creates a circular buffer. If `buffer` is non-null, the CB is backed by that buffer (used for sharded input where CB maps directly to L1 shard). Returns `(cb_index, cb_handle)`.
