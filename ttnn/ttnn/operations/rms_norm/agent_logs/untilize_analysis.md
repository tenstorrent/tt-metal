# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize operation converts a tensor from TILE_LAYOUT (32x32 tiles with face-interleaved internal storage) back to ROW_MAJOR_LAYOUT (contiguous row-major sticks). This analysis focuses on the **multi-core program factory** variant (`UntilizeMultiCoreProgramFactory`), with emphasis on the **output stage**: how the compute kernel produces row-major sticks in the output CB, how the writer kernel extracts and writes those sticks to DRAM, and the output CB sizing strategy.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Focus areas** (per output_stage role):
- Untilize helper signature and usage
- Writer kernel pattern (how RM sticks are written to DRAM)
- Output CB sizing
- Stick extraction from tiles

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row of tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (one full tile-row width) |
| **Total units** | `num_tiles_per_col` (total tile-rows across the tensor height) |
| **Loop structure** | Outer loop: `num_input_blocks_to_process` (tile-rows per core); inner: compute untilizes one tile-row, writer extracts `tile_height` (32) RM sticks from it |

A "block" is one horizontal row of tiles spanning the input width (or shard width for sharded inputs). The compute kernel processes one block at a time, converting all tiles in that row from tile format to row-major format in the output CB. The writer then extracts `tile_height` individual row-major sticks from that block and writes them to DRAM.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, ...dims..., H, W] (arbitrary rank, flattened to 2D) |
| **Dimension convention** | Last dim = W, second-to-last = H, others collapsed |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with 16x16 face sub-structure) |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT (one page = one row-major stick) |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input |

### Layout Transformations

The core transformation is **tile-to-row-major conversion**:
- **Input**: Data stored in tile format (32x32 tiles, internally organized as 4 faces of 16x16, face-major order within each tile).
- **Output**: Data stored as contiguous row-major sticks. Each page in the output is one full row of elements (for interleaved/height-sharded) or a shard-width portion of a row (for width/block-sharded).
- The compute kernel's `pack_untilize_block` hardware instruction rearranges tile data into row-major order within the output CB. After compute finishes one block, the output CB contains `tile_height` contiguous sticks, each of width `num_tiles_per_input_block * tile_width` elements.

## Data Flow Pattern

### Stage 1: Reader Kernel (de-emphasized)
- **Interleaved**: Reads tiles sequentially from DRAM using `TensorAccessor.get_noc_addr(page_id)`, one tile at a time, into CB_src0 (c_0).
- **Sharded**: CB_src0 is backed by the shard buffer directly; reader just signals tile availability.

### Stage 2: Compute Kernel (Untilize)
- Waits for `num_tiles_per_input_block` tiles in CB_src0 (one full tile-row).
- Calls `pack_untilize_block` (or standard untilize) which:
  1. Unpacks tiles from input CB into DEST registers.
  2. Rearranges tile face data into row-major order.
  3. Packs the row-major result into output CB_out (c_16).
- After processing one block: output CB contains `num_tiles_per_input_block` tiles' worth of data, but now laid out as `tile_height` (32) contiguous row-major sticks, each `num_tiles_per_input_block * tile_width` elements wide.
- Pops input tiles, pushes output tiles.
- Repeats for `num_input_blocks_to_process` blocks (runtime arg).

### Stage 3: Writer Kernel (OUTPUT STAGE - PRIMARY FOCUS)
- Waits for `num_tiles_per_input_block` tiles in output CB (one block's worth of untilized data).
- Iterates through `tile_height` (32) rows within the block.
- For each row, calculates the output page ID and writes the stick to DRAM using `TensorAccessor`.
- Handles width-sharded/block-sharded output by splitting sticks across multiple output pages when input block width differs from output page width.
- Issues `noc_async_write_barrier()` after each block.
- Pops output CB tiles.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | Varies (see below) | `num_tiles_per_input_block` | Single or Double | Reader | Compute | Block |
| c_16 | cb_output | **Output RM stick staging** | Varies (see below) | `num_tiles_per_input_block` | **Single or Double** | Compute | Writer | Block |

### Output CB Sizing Logic (Primary Focus)

The output CB (c_16) sizing is determined by the number of tile-rows (blocks) the core processes:

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    // Single block: no need for double buffering
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    // Multiple blocks: double buffer for compute/writer overlap
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

**Key insight for rms_norm**: The output CB capacity is `num_tiles_per_input_block` (single-buffered) or `2 * num_tiles_per_input_block` (double-buffered). This means the output CB holds either 1 or 2 complete tile-rows of untilized (row-major) data. Each "tile" in the output CB is `tile_size(output_data_format)` bytes, but the data within is now row-major rather than tile-ordered.

**Important**: The output CB page size is still `output_single_tile_size` (the size of one tile in the output data format), even though the data is row-major. This is because the compute kernel's `pack_untilize_block` writes data in units of tiles (the output CB tracks capacity in tile-sized pages), but the actual data layout within those pages is row-major sticks concatenated together. The writer kernel reads the CB using raw L1 addresses (`get_read_ptr`) rather than tile-based addressing.

### Input CB Sizing (De-emphasized)
- **Sharded**: Entire shard at once (`num_tiles_per_input_block * num_input_blocks_per_full_core`).
- **Interleaved, single block**: `num_tiles_per_input_block` (single-buffered).
- **Interleaved, multiple blocks**: `num_tiles_per_input_block * 2` (double-buffered).

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Pattern |
|----|----------|------------|-------|---------|
| c_0 (input, interleaved multi-block) | 2x block | 1x block | 2 | Double-buffered |
| c_0 (input, sharded) | full shard | full shard | 1 | Single-buffered (no reader overlap needed) |
| c_16 (output, multi-block) | 2x block | 1x block | 2 | Double-buffered |
| c_16 (output, single-block) | 1x block | 1x block | 1 | Single-buffered |

When double-buffered, compute can begin untilizing the next block while the writer is still writing sticks from the previous block.

## Index Calculations

### Writer Kernel Index Mapping (Primary Focus)

The writer kernel performs a critical mapping from the untilized CB data to output DRAM pages. The key variables:

1. **`height_wise_input_block_start_index`**: Which tile-row this core starts processing at (globally). Computed on host as `(core_index / num_input_blocks_across_width) * num_input_blocks_per_full_core`.

2. **Output page ID calculation** (in writer kernel, per stick):
   ```cpp
   uint32_t num_rows_already_processed = block_height_index * tile_height + j;
   uint32_t num_pages_already_processed_in_previous_rows =
       num_rows_already_processed * num_output_blocks_across_width;
   uint32_t output_page_id =
       num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;
   ```
   This maps a (block_index, row_within_block) pair to a global output page ID. Each row-major stick occupies one output page (for interleaved/height-sharded) or is split across multiple output pages (for width/block-sharded).

3. **Width splitting across output pages**: When input block width differs from output page width (width/block sharding), a single stick from the CB may span multiple output pages. The writer uses a `while` loop:
   ```cpp
   while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
       num_cols_to_write = min(remaining_input_cols, remaining_output_block_cols);
       // write partial stick to current output page
       // advance to next output page if needed
   }
   ```

4. **L1 read address within CB**: The writer reads raw L1 memory from the output CB:
   ```cpp
   uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);
   uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;
   ```
   This assumes row-major layout: row `j` starts at offset `j * row_width_bytes` from the CB read pointer. This is the layout that `pack_untilize_block` produces.

### TensorAccessor Usage

Both reader and writer use `TensorAccessor` for DRAM address resolution:
- **Host side**: `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` serializes bank mapping information into compile-time args.
- **Device side**: `TensorAccessor(dst_args, dst_addr, output_stick_size)` reconstructs the accessor. The `output_stick_size` parameter is the page size for address calculation.
- **Address resolution**: `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` returns the NoC address for a specific page, optionally with a byte offset within the page (used for width-sharded partial writes).

## Memory Access Patterns

### Read Pattern (De-emphasized)
- **Interleaved**: Sequential tile reads from DRAM, one tile at a time, page IDs incrementing.
- **Sharded**: Direct L1 access, no NoC reads needed.

### Write Pattern (Primary Focus)

The writer kernel writes **row-major sticks** to DRAM/L1:

1. **Per-block**: Writer waits for one complete block of untilized data in the output CB.
2. **Per-row within block**: Iterates through `tile_height` (32) rows sequentially.
3. **Per-row write**: For each row, writes the stick to the output buffer.
   - **Simple case** (interleaved or height-sharded, output page width == input block width): One `noc_async_write` per row, writing `output_stick_size` bytes.
   - **Complex case** (width/block-sharded, output page width != input block width): Multiple `noc_async_write` calls per row, each writing a portion of the stick to a different output page.
4. **Barrier**: `noc_async_write_barrier()` after all rows in a block are written.
5. **Pattern**: Effectively sequential within a block (rows 0 through 31), with block-level pipelining via double buffering.

**Write size per call**: `num_cols_to_write * output_element_size` bytes. In the common case (no width sharding), this equals `output_stick_size` = `tensor_width * element_size` bytes per write.

**Padding handling**: The `num_unpadded_cols_per_input_block` runtime arg allows the writer to skip padding columns at the end of the last width-wise shard.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear core allocation) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` (interleaved) or shard grid cores (sharded) |
| **Work per core** | `num_input_blocks_per_full_core` tile-rows (interleaved: `ceil(num_tiles_per_col / grid_area)`) |
| **Load balancing** | Nearly equal; last core (cliff) may process fewer rows |
| **Remainder handling** | Single cliff core with `num_input_blocks_per_cliff_core = num_tiles_per_col % nblocks_per_core` |

**Interleaved path**: Uses `split_blocks_for_tilize(grid_size, num_tiles_per_col)` which divides tile-rows across cores with at most one cliff core.

**Sharded path**: Each core processes its own shard. `num_input_blocks_across_width` tracks width-wise sharding. Core iteration uses `i / num_input_blocks_across_width` for height index and `i % num_input_blocks_across_width` for width index.

## Arguments

### Compile-Time Arguments

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Size of one output page in bytes (`output_page_width * element_size`) |
| 2 | `tile_height` | uint32_t | Height of tiles (32) -- number of RM sticks per block |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per input block (for CB wait/pop) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per row (1 for interleaved/height-sharded) |
| 5 | `output_element_size` | uint32_t | Bytes per element (2 for bf16, 4 for fp32) |
| 6 | `num_cols_per_input_block` | uint32_t | Elements per input block row (`num_tiles_per_input_block * tile_width`) |
| 7 | `num_cols_per_output_block` | uint32_t | Elements per output page (`output_page_width`) |
| 8+ | TensorAccessor args | uint32_t[] | Bank mapping info for output buffer (serialized by `TensorAccessorArgs`) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | `num_tiles_per_input_block` -- tiles per row (block width) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Runtime Arguments

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-rows this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Global tile-row start index for this core |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Actual data columns (excludes shard padding) |
| 4 | `width_wise_output_block_start_index` | uint32_t | Starting output page index within each row |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset into first output page (for width-sharded partial writes) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | `num_input_blocks_to_process` -- number of blocks (tile-rows) |

## Kernel Implementations

### Compute Kernel: Untilize Helper

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | TRISC (unpack/math/pack) | N/A | CB c_0 (tiles) | CB c_16 (RM sticks) | pack_untilize_block |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` (slow path) or `pack_untilize_variable_num_blocks.cpp` (fast path)

**Both kernels delegate to the unified helper**:
```cpp
compute_kernel_lib::untilize<
    per_core_block_tile_cnt,    // block width in tiles (compile-time)
    src_cb_id,                  // input CB
    out_cb_id,                  // output CB
    InitUninitMode::InitAndUninit,
    WaitMode::WaitBlock,        // wait for input per block
    ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);  // runtime: num blocks
```

**Untilize helper signature** (`untilize_helpers.hpp`):
```cpp
template <
    uint32_t block_width_tiles,     // Tiles per row (FIRST template param, compile-time)
    uint32_t input_cb,              // Input CB index
    uint32_t output_cb,             // Output CB index
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
ALWI void untilize(uint32_t num_blocks);  // Runtime: number of rows/blocks
```

**Key implementation details** (from `untilize_helpers.inl`):
- **DEST limit check**: `constexpr bool use_block_based_pack = (block_width_tiles > dest_limit)`. DEST capacity is 8 tiles (half-sync, fp16) or 4 tiles (half-sync, fp32).
- **Wide row splitting**: If `block_width_tiles > dest_limit`, the row is split into `num_sub_blocks` sub-blocks, each `sub_block_width` tiles wide. Each sub-block is processed separately via `pack_untilize_block<sub_block_width, block_width_tiles>(input_cb, 1, output_cb, b)` where `b` is the sub-block index.
- **Narrow row path**: If `block_width_tiles <= dest_limit`, the entire row fits in DEST. A single `pack_untilize_block<block_width_tiles, block_width_tiles>(input_cb, 1, output_cb, 0)` processes the full row.
- **Per-block CB operations**: For each block: `cb_wait_front(input, block_width)`, `cb_reserve_back(output, block_width)`, process, `cb_pop_front(input, block_width)`, `cb_push_back(output, block_width)`.

**Path selection** (program factory, line 232-244):
- **Slow untilize**: Used when `!use_pack_untilize`, or dtype is UINT16, or (FLOAT32 and width >= 8 tiles). Falls back to standard untilize (still uses `pack_untilize_block` internally via the unified helper, but with reduced DEST capacity for FP32).
- **Fast pack untilize**: Used otherwise. Hardware-accelerated path.
- `MAX_PACK_UNTILIZE_WIDTH = 8` (defined in `ttnn/api/ttnn/common/constants.hpp`).

**What pack_untilize_block produces**: After the compute kernel processes one block, the output CB contains `tile_height` (32) contiguous row-major sticks. Each stick is `num_tiles_per_input_block * tile_width` elements wide. The sticks are laid out sequentially in L1: stick 0 at offset 0, stick 1 at offset `num_cols_per_input_block * element_size`, etc.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 (RM sticks) | DRAM | noc_async_write |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Key Logic**:

1. **TensorAccessor setup**: `TensorAccessor(dst_args, dst_addr, output_stick_size)` -- uses `output_stick_size` as page size for bank address calculation.

2. **Block processing loop**: For each block `i` in `[0, num_input_blocks_to_process)`:
   - Wait for `num_tiles_per_input_block` tiles in output CB.
   - Get base L1 read address: `base_l1_read_addr = get_read_ptr(cb_id_out0)`.
   - For each row `j` in `[0, tile_height)`:
     - Compute L1 source address: `base_l1_read_addr + j * num_cols_per_input_block * output_element_size`.
     - Compute output page ID from global row index and width position.
     - Write stick (or partial sticks for width-sharded output) via `noc_async_write`.
   - Barrier and pop CB.

3. **Width-sharded output handling**: When `num_cols_per_output_block != num_cols_per_input_block`, one untilized row may span multiple output pages. The writer iterates through output pages, writing partial sticks. The `num_cols_already_processed_in_first_output_block` runtime arg handles the case where this core's data starts mid-page.

4. **Padding column skip**: `num_unpadded_cols_per_input_block` limits how many columns are written, skipping padding at the shard boundary.

### Reader Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | CB c_0 | noc_async_read |

**File (interleaved)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
**File (sharded)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

## Implementation Notes

### Output CB Data Layout After Untilize

This is the most critical detail for reuse in rms_norm. After the compute kernel's `pack_untilize_block` processes one block of `N` tiles (one tile-row), the output CB contains the data in row-major order:

```
Offset 0:                      Row 0 of all N tiles concatenated (N * tile_width elements)
Offset row_width_bytes:        Row 1 of all N tiles concatenated
...
Offset (tile_height-1)*rw:     Row 31 of all N tiles concatenated
```

Where `row_width_bytes = num_tiles_per_input_block * tile_width * element_size` = `num_cols_per_input_block * output_element_size`.

The writer reads this using raw pointer arithmetic: `get_read_ptr(cb) + row_index * row_width_bytes`.

### Compute Kernel Selection for rms_norm Context

For a new rms_norm operation that needs in-kernel untilize at the output stage:
- Use `compute_kernel_lib::untilize<block_width_tiles, input_cb, output_cb>` from `untilize_helpers.hpp`.
- `block_width_tiles` must be a compile-time constant (the number of tiles in the width dimension of the output).
- Call `compute_kernel_hw_startup(input_cb, output_cb)` before using the helper.
- The helper handles all DEST splitting, init/uninit, and CB synchronization internally.
- For back-to-back operations (e.g., compute rms_norm then untilize), use `InitUninitMode::InitOnly` / `Neither` / `UninitOnly` to avoid redundant init/uninit between stages.

### DST_ACCUM_MODE and FP32 Handling

When processing INT32, UINT32, or FLOAT32 data:
- `DST_ACCUM_MODE` define is set to "1".
- `fp32_dest_acc_en` is set in `ComputeConfig`.
- This halves DEST capacity (from 8 to 4 tiles in half-sync mode).
- For FLOAT32 with width >= 8 tiles, the fast pack_untilize path is disabled.

### Uneven Shard Handling

The program factory computes `num_unpadded_cols_per_input_block` and `num_input_blocks_to_process` per-core to handle:
- **Width-wise uneven shards**: Last shard in a row may have fewer valid columns.
- **Height-wise uneven shards**: Last shard in a column may have fewer valid tile-rows.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is pack_untilize_block and how does it convert tile format data to row-major sticks in the output circular buffer?"
   **Reason**: Needed to understand the fundamental mechanism by which tiles are converted to row-major sticks.
   **Key Findings**: `pack_untilize_block` is an LLK function that reads from DEST registers and writes row-major data to the output CB. It processes tiles row by row, unpacking from tile format, performing unary data copy through MATH, and then packing in row-major order. The output in the CB is contiguous row-major sticks. Maximum block size is limited by DEST register capacity.

2. **Query**: "How does the writer kernel write row-major sticks to DRAM after untilize? What is the output page structure?"
   **Reason**: Needed to understand the output memory layout and page mapping.
   **Key Findings**: DeepWiki query failed; information was obtained from direct code analysis of the writer kernel and tensor_layouts tech report.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding page structure for row-major vs tiled tensors.
   **Key Information**: Row-major layout: each row = one page. Tiled layout: each 32x32 tile = one page. Interleaved: pages distributed round-robin across banks. This confirms that after untilize, output pages are individual rows (sticks).

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how `TensorAccessor` maps page IDs to NoC addresses.
   **Key Information**: `TensorAccessor` handles bank-to-core mapping for interleaved and sharded tensors. `get_noc_addr(page_id)` returns the full NoC address for a page. Supports optional byte offset within page via second parameter. Host-side setup via `TensorAccessorArgs(*buffer).append_to(compile_time_args)`.

3. **Source**: `ttnn/api/ttnn/common/constants.hpp`
   **Reason**: Finding the value of `MAX_PACK_UNTILIZE_WIDTH`.
   **Key Information**: `MAX_PACK_UNTILIZE_WIDTH = 8` -- pack untilize hardware path does not support block widths > 8 tiles.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity and how it affects untilize block splitting.
   **Key Information**: DEST capacity table: SyncFull+16bit=16, SyncFull+32bit=8, SyncHalf+16bit=8, SyncHalf+32bit=4 tiles. The untilize helper uses `DEST_AUTO_LIMIT` to automatically determine if block splitting is needed.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize helper API and its implementation.
   **Key Information**: The helper provides a single `untilize<>()` template function that handles all variants (narrow/wide, fast/slow, init modes, wait modes). Template parameter `block_width_tiles` must be compile-time. Runtime parameter `num_blocks` controls how many rows to process. Automatically splits wide rows into DEST-sized sub-blocks.

6. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding core distribution strategy for interleaved path.
   **Key Information**: `split_blocks_for_tilize(grid_size, num_tiles_per_col)` distributes tile-rows across cores, with `ceil(nblocks/grid_area)` rows per core and at most one cliff core handling the remainder.
