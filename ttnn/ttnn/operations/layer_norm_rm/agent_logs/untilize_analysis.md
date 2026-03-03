# Untilize Multi-Core Implementation Analysis

## Overview

The untilize operation converts tile-layout tensor data (32x32 tiles with face structure) back into row-major (RM) format, producing contiguous row "sticks" in the output buffer. This analysis covers the `UntilizeMultiCoreProgramFactory`, the general-purpose multi-core variant that handles both interleaved and sharded inputs, writing interleaved RM output to DRAM.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Focus**: Output stage -- how RM sticks are extracted from untilized tiles and written to DRAM, output CB sizing, and the `untilize` helper signature/usage pattern.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = one row of tiles spanning the tensor width) |
| **Unit size** | `num_tiles_per_input_block` tiles (tiles spanning one tile-row of the assigned width) |
| **Total units** | `num_tiles_per_col` tile-rows (for interleaved), distributed across cores |
| **Loop structure** | Outer loop over assigned blocks (tile-rows); inner loop over `tile_height` stick rows within each block |

A single "input block" is one tile-row: `num_tiles_per_input_block` tiles arranged horizontally. Each block contains `tile_height` (32) rows of sticks. The compute kernel untilizes one block at a time, converting the face-structured tile data into `tile_height` contiguous RM rows in the output CB. The writer then extracts those rows and writes them as individual sticks to DRAM.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D (flattened to 2D: height x width) |
| **Dimension convention** | Last dim = width, remaining dims flattened to height |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with 16x16 face sub-structure) |
| **Memory layout** | INTERLEAVED or SHARDED (height, width, or block) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT (contiguous sticks) |
| **Memory layout** | INTERLEAVED (primary path); also WIDTH_SHARDED or BLOCK_SHARDED possible |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | Same as input |

### Layout Transformations

The core transformation is tile-to-row-major conversion:
- **Compute kernel**: Reads tiles from input CB (face-structured 32x32), untilizes them via hardware pack/unpack units, writes contiguous RM rows to output CB. Each block of `num_tiles_per_input_block` tiles produces `tile_height` rows, each `num_tiles_per_input_block * tile_width` elements wide.
- **Writer kernel**: Reads these contiguous RM rows from the output CB and writes them as sticks to DRAM pages.

**Page mapping change**: Input pages are tiles (32x32 elements each). Output pages are sticks (one full tensor row or one shard width, depending on output memory layout). For interleaved output, each page = one full row of the tensor = `tensor_width * element_size` bytes.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) or L1 (sharded) | CB c_0 (input) | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB c_0 (input) | CB c_16 (output) | `cb_wait_front` / `cb_pop_front` on c_0; `cb_reserve_back` / `cb_push_back` on c_16 (managed inside `untilize` helper) |
| 3 | Writer | CB c_16 (output) | DRAM (interleaved) | `cb_wait_front`, `noc_async_write`, `noc_async_write_barrier`, `cb_pop_front` |

**Detailed data flow for the output stage (writer)**:
1. Writer calls `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` to wait for one full block of untilized data.
2. Gets the base L1 read address via `get_read_ptr(cb_id_out0)`.
3. Iterates over `tile_height` rows within the block. For each row `j`:
   - Computes L1 read address: `base + j * num_cols_per_input_block * output_element_size`
   - Computes the output page_id (stick index) accounting for height and width offsets
   - Writes partial or full sticks to DRAM using `noc_async_write` with addresses from `TensorAccessor::get_noc_addr(page_id, offset)`
4. After all rows in the block are written, calls `noc_async_write_barrier()` then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`.
5. Repeats for the next block.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | See formula below | `num_tiles_per_input_block` tiles | Single or Double | Reader | Compute | Block |
| c_16 | cb_output | Output RM stick staging | See formula below | `num_tiles_per_input_block` tiles | Single or Double | Compute | Writer | Block |

### Input CB (c_0) Sizing -- summarized for completeness

- **Sharded**: `num_tiles_per_input_block * num_input_blocks_per_full_core` tiles (entire shard at once; backed by the sharded L1 buffer)
- **Interleaved, 1 block/core**: `num_tiles_per_input_block` tiles (single-buffered)
- **Interleaved, 2+ blocks/core**: `num_tiles_per_input_block * 2` tiles (double-buffered)

### Output CB (c_16) Sizing -- PRIMARY FOCUS

The output CB capacity is determined by how many tile-rows (blocks) the core processes:

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

**Key insight for reuse**: The output CB is sized in units of tiles (using `output_single_tile_size`), even though the data in it is row-major. After the compute kernel untilizes a block of tiles, the output CB contains `tile_height` contiguous RM rows, each `num_tiles_per_input_block * tile_width * element_size` bytes. The CB capacity in bytes = `output_cb_num_tiles * output_single_tile_size`.

**Block size**: One block = `num_tiles_per_input_block` tiles = one tile-row of untilized data.

**Buffering strategy**: When the core processes 2+ blocks, the output CB is double-buffered so the compute kernel can fill the next block while the writer drains the current block. With only 1 block, no overlap is needed.

**Output CB is NOT backed by a globally-allocated buffer** -- it is always a local L1 circular buffer (the `create_cb` call passes `nullptr` for the buffer argument for c_16, unlike the sharded input case for c_0).

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering | Overlap Potential |
|----|----------|------------|-----------|-------------------|
| c_0 (input) | 1x or 2x block (interleaved); full shard (sharded) | `num_tiles_per_input_block` tiles | Single (1 block) or Double (2+ blocks) | Reader/Compute overlap when double-buffered |
| c_16 (output) | 1x or 2x block | `num_tiles_per_input_block` tiles | Single (1 block) or Double (2+ blocks) | Compute/Writer overlap when double-buffered |

## Index Calculations

### Output Page ID Calculation (Writer Kernel)

The writer maps each RM row within a block to an output page_id (stick index in the interleaved output buffer):

```cpp
// For row j within a block at height_wise index block_height_index:
uint32_t num_rows_already_processed = block_height_index * tile_height + j;
uint32_t num_pages_already_processed_in_previous_rows =
    num_rows_already_processed * num_output_blocks_across_width;
uint32_t output_page_id =
    num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;
```

For the common interleaved-output case where `num_output_blocks_across_width = 1`, this simplifies to:
```
output_page_id = (block_height_index * tile_height + j) * 1 + 0
               = block_height_index * tile_height + j
```
This means page_id = absolute row index, which aligns with the RM page convention: one page per tensor row.

### L1 Read Address Calculation (Writer Kernel)

After untilize, the output CB contains `tile_height` contiguous RM rows:

```cpp
uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);
uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;
```

Each row is `num_cols_per_input_block * output_element_size` bytes. The writer reads row `j` at offset `j * row_bytes` from the CB base.

### Width-Splitting Logic (Writer Kernel)

When the output is width-sharded or block-sharded, a single input block's worth of columns may span multiple output pages. The writer handles this with a `while` loop:

```cpp
while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
    uint32_t num_cols_to_write = std::min(
        num_unpadded_cols_per_input_block - num_input_cols_processed,
        num_cols_remaining_in_current_output_block);
    // ... write partial stick, advance to next output page
}
```

For simple interleaved output where `num_cols_per_input_block == num_cols_per_output_block`, the loop body executes exactly once per row (the common case).

### TensorAccessor Usage

The writer creates a `TensorAccessor` from compile-time args appended by `TensorAccessorArgs(*dst_buffer)`:

```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // compile-time args start at index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

The `get_noc_addr` call takes a page_id and optional byte offset within the page:
```cpp
uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
```

This maps the logical stick index to a physical DRAM bank address, handling interleaved round-robin distribution automatically. The offset parameter enables writing to the middle of a page (used when an input block does not align with output page boundaries).

## Memory Access Patterns

### Read Pattern (Reader -- summarized)

- **Interleaved**: Sequential tile reads via `noc_async_read`, one tile at a time, starting from `start_page_id`. Each tile read is followed by a barrier before pushing to CB.
- **Sharded**: No reads needed; input CB is backed by the shard buffer directly. Reader just signals `cb_push_back` for the whole shard.

### Write Pattern (Writer -- PRIMARY FOCUS)

- **Granularity**: Individual RM sticks (one per tensor row per block iteration)
- **Ordering**: Row-major within each block. For block `i` at `height_wise_input_block_start_index`, rows are written from row `i*tile_height` through `i*tile_height + tile_height - 1`, in order.
- **Access pattern**: For interleaved output, each stick goes to a different DRAM bank (round-robin), so writes are distributed across banks. Writes are sequential within a block (row 0, row 1, ..., row 31), producing ascending page_ids.
- **Barrier placement**: `noc_async_write_barrier()` is called once per block (after all `tile_height` rows are written), NOT per row. This batches the NoC write barrier for efficiency.
- **Partial writes**: When input block width does not align with output page width (width/block sharding), the writer performs multiple partial writes per row, advancing across output pages.

### Write Pattern Summary for Interleaved Output (Common Case)

Per block: 32 sequential `noc_async_write` calls (one per row), each writing `tensor_width * element_size` bytes, followed by a single `noc_async_write_barrier`. The CB is then popped (`cb_pop_front`).

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D compute grid) |
| **Grid dimensions** | Determined by `split_blocks_for_tilize(grid_size, num_tiles_per_col)` |
| **Total cores** | `num_compute_cores` (varies based on tensor height and grid size) |
| **Work per core** | `num_rows_per_full_core` tile-rows (blocks); cliff core gets `num_rows_per_cliff_core` |
| **Load balancing** | Nearly equal; at most one cliff core handles the remainder |

**Work splitting**: `split_blocks_for_tilize` divides `num_tiles_per_col` (total tile-rows) across the available grid. Each full core gets `nblocks_per_core` blocks. If there is a remainder, a single cliff core handles `nblocks_per_core_cliff` blocks.

**Sharded input**: When input is sharded, the core distribution follows the shard spec's grid. Each core processes its local shard. There is no cliff core for sharded input.

## Arguments

### Compile-Time Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Size of one output page in bytes (`output_page_width * element_size`) |
| 2 | `tile_height` | uint32_t | Number of rows per tile (32 for standard tiles) |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per block (tile-row width) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per tensor row (1 for interleaved, >1 for width/block sharded) |
| 5 | `output_element_size` | uint32_t | Bytes per output element (2 for BF16, 4 for FP32) |
| 6 | `num_cols_per_input_block` | uint32_t | Total columns in one input block (`num_tiles_per_input_block * tile_width`) |
| 7 | `num_cols_per_output_block` | uint32_t | Columns per output page (= `output_page_width`) |
| 8+ | TensorAccessorArgs | varies | Auto-appended by `TensorAccessorArgs(*dst_buffer)` for DRAM addressing |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of blocks (tile-rows) this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Starting block index for this core (global tile-row offset) |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Actual data columns (excludes padding from uneven sharding) |
| 4 | `width_wise_output_block_start_index` | uint32_t | Starting output page offset within each row |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset within the first output page (for mid-page writes) |

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | Tiles per block (`num_tiles_per_input_block`) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks to untilize (`num_input_blocks_to_process`) |

### Compile-Time Arguments (Reader Kernel -- interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_in0` | uint32_t | Input CB index (c_0) |
| 1+ | TensorAccessorArgs | varies | Auto-appended by `TensorAccessorArgs(*src0_buffer)` |

### Runtime Arguments (Reader Kernel -- interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Source buffer base address |
| 1 | `num_tiles` | uint32_t | Total tiles to read for this core |
| 2 | `start_page_id` | uint32_t | First tile index to read |

## Kernel Implementations

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM (interleaved) | Extract RM sticks, write to DRAM via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - The writer defines a lambda `write_tiles_in_current_block` that processes one block at a time.
  - For each of `tile_height` (32) rows within a block, it calculates the L1 source address as an offset from the CB read pointer and the destination DRAM page_id via arithmetic over block indices.
  - The inner `while` loop handles cases where input block columns span multiple output pages (width/block sharding). For simple interleaved output, this loop iterates exactly once.
  - `noc_async_write_barrier()` is called once per block (NOT per row), enabling all 32 writes to be pipelined.
  - `cb_pop_front` releases the block after all rows are written.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 | CB c_16 | Untilize (tile-to-RM conversion) |

- **File (slow path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **File (fast path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
- **Key Logic**:
  - Both variants call `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt)`.
  - The unified `untilize` helper (from `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`) automatically selects between standard untilize and pack_untilize based on block width vs DEST capacity and data format.
  - Standard path: unpack tiles from input CB, datacopy to DEST, pack as RM to output CB.
  - Pack untilize path: hardware-accelerated reordering during pack stage (faster for supported types/widths).
  - The slow path is forced for UINT16 or for FLOAT32 when `num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH`.

### Reader Kernel (summarized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (interleaved) | CB c_0 | Read tiles sequentially |

- **File (interleaved)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **File (sharded)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

## The `untilize` Helper -- Signature and Usage

The `compute_kernel_lib::untilize` helper is the primary compute-side API for tile-to-RM conversion. Its signature:

```cpp
template <
    uint32_t block_width_tiles,   // Tiles per row (compile-time)
    uint32_t input_cb,            // Input CB index
    uint32_t output_cb,           // Output CB index
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
void untilize(uint32_t num_blocks);  // num_blocks = runtime
```

**Template parameters**:
- `block_width_tiles`: Number of tiles per row -- this is the width of one "block" being untilized. Must be a compile-time constant.
- `input_cb` / `output_cb`: CB indices for tiled input and RM output.
- `init_uninit_mode`: Controls whether `untilize_init`/`untilize_uninit` are called. Default `InitAndUninit` is correct for standalone calls. Use `InitOnly`/`Neither`/`UninitOnly` for back-to-back calls to avoid redundant init overhead.
- `wait_mode`: `WaitBlock` (default) waits for input tiles per block. `WaitUpfront` waits for all tiles before processing (useful when data is already in CB). `NoWait` defers synchronization to the caller.
- `reconfig_mode`: For switching data formats mid-kernel (e.g., after a different compute op).

**Usage in this operation**:
```cpp
compute_kernel_hw_startup(src_cb_id, out_cb_id);  // Must come first
compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt);
```

**For a new operation (like layer_norm_rm) reusing this pattern**:
- If your compute kernel produces tiles in a CB and you need RM output, you can call `untilize` at the end.
- Use `WaitMode::WaitUpfront` or `WaitMode::NoWait` if the tiles are already available (e.g., produced by a preceding compute stage in the same kernel).
- Use `InitUninitMode::InitOnly` / `UninitOnly` if untilize is one of multiple operations in the same kernel.

## Implementation Notes

### Output Stick Size Calculation

The output stick size (page size for RM output) depends on the output memory layout:

```cpp
uint32_t output_page_width = tensor_width;  // Default: full row for interleaved
if (output is WIDTH_SHARDED or BLOCK_SHARDED) {
    output_page_width = shard_spec.shape[1];  // Shard width
}
uint32_t output_stick_size = output_page_width * output_element_size;
```

For a new operation writing interleaved RM output: `stick_size = tensor_width * element_size`.

### Padding Handling

When input is unevenly sharded, the last shard width-wise may contain padding columns. The writer uses `num_unpadded_cols_per_input_block` (a runtime arg) to write only valid data, skipping garbage padding. This is computed on the host:

```cpp
if (is_last_input_shard_in_row) {
    num_unpadded_cols_per_input_block =
        num_cols_per_input_block - (round_up(tensor_width, input_shard_width) - tensor_width);
}
```

### Write Barrier Strategy

The write barrier is placed per-block, NOT per-stick. This allows all 32 (tile_height) `noc_async_write` calls within a block to be pipelined by the NoC, with a single barrier at the end. This is more efficient than per-row barriers.

### CB Pop Timing

`cb_pop_front` is called AFTER the write barrier, ensuring the compute kernel does not overwrite the output CB while the writer is still reading from it.

### Compute Kernel Selection

The program factory selects between two compute kernels:
- **Fast path** (`pack_untilize`): Used when `use_pack_untilize` is true AND dtype is not UINT16 AND (dtype is not FLOAT32 OR `num_tiles_per_input_block < MAX_PACK_UNTILIZE_WIDTH`).
- **Slow path** (`untilize`): Fallback for unsupported types/widths. Forces `UnpackToDestMode::Default` for SFPU-based untilize.

Both kernels use the same unified `compute_kernel_lib::untilize` helper which internally routes to the appropriate LLK path.

### Runtime Argument Override

The `override_runtime_arguments` method updates only buffer addresses (src/dst) for program cache reuse. This enables efficient re-execution when only tensor addresses change between calls.

## Relevance to layer_norm_rm

For a new `layer_norm_rm` operation that computes on RM interleaved input and writes RM interleaved output:

1. **Output CB sizing**: Follow the same pattern -- size the output CB as `num_tiles_per_row * output_tile_size` (single-buffered for 1 block, double-buffered for 2+). Even though the data is RM, CB capacity is measured in tile units.

2. **Writer pattern**: The writer kernel here demonstrates the canonical pattern for writing RM sticks to interleaved DRAM:
   - Use `TensorAccessor` with `output_stick_size` = `tensor_width * element_size`
   - Each page_id = row index
   - `noc_async_write` with `get_noc_addr(page_id)` for full-row writes
   - Barrier per block (not per row) for efficiency

3. **Untilize helper usage**: If layer_norm_rm computes in tiles and needs RM output, call `compute_kernel_lib::untilize` with appropriate `WaitMode` and `InitUninitMode` settings depending on whether it follows other compute operations.

4. **If input is already RM**: The untilize compute step is not needed. The reader can read RM sticks directly (page = row), and the writer can write them back after compute. The CB sizing and writer pattern still apply.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the TensorAccessor work in tt-metal kernels? What is TensorAccessorArgs, and how does get_noc_addr with a page_id and optional offset work?"
   **Reason**: The writer kernel uses `TensorAccessor` with a `page_id` and `offset_bytes` parameter, and understanding this mapping is critical for the output stage.
   **Key Findings**: TensorAccessor abstracts logical page indices to physical DRAM bank addresses. `get_noc_addr(page_id, offset_bytes)` computes the NoC address for a given page with an optional byte offset within the page. For RM interleaved tensors, pages correspond to rows, distributed round-robin across DRAM banks.

2. **Query**: "How does the untilize compute operation work in tt-metal? What happens during pack_untilize vs regular untilize?"
   **Reason**: Needed to understand what the compute kernel produces in the output CB and how the data is laid out after untilization.
   **Key Findings**: Untilize converts face-structured tile data to contiguous RM rows. The standard path uses unpack-datacopy-pack; pack_untilize uses hardware-accelerated reordering during the pack phase. Both produce the same output format: `tile_height` contiguous RM rows per block.

3. **Query**: "What does split_blocks_for_tilize do in tt-metal? How does it distribute tile rows across cores?"
   **Reason**: The program factory uses this to determine core distribution and work splitting.
   **Key Findings**: Divides total tile-rows evenly across available cores. Full cores get `nblocks_per_core` blocks each. A single "cliff" core handles the remainder (if any). Returns core ranges for kernel placement.

4. **Query**: "How does noc_async_write work in tt-metal writer kernels? What is the noc_async_write_barrier used for?"
   **Reason**: The writer kernel's write pattern (barrier per block vs per row) is a key design decision.
   **Key Findings**: `noc_async_write` is asynchronous -- it enqueues NoC write requests and returns immediately. `noc_async_write_barrier` blocks until all outstanding writes complete. Batching multiple writes before a single barrier is an established optimization pattern.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding the page/stick model for RM vs tiled tensors.
   **Key Information**: In RM layout, each tensor row = one page. In tiled layout, each 32x32 tile = one page. Interleaved memory distributes pages round-robin across DRAM banks. Tiles contain 4 faces (16x16) stored contiguously.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how `TensorAccessorArgs` is set up on host and used on device.
   **Key Information**: `TensorAccessorArgs(buffer)` extracts addressing metadata from the buffer. On device, `TensorAccessorArgs<offset>()` reconstructs from compile-time args. `get_noc_addr(page_id)` handles bank mapping; `get_noc_addr(page_id, offset)` adds a byte offset within the page.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
   **Reason**: Understanding the unified untilize helper's template interface and configuration options.
   **Key Information**: Template parameters control block width, CB indices, init/uninit lifecycle, wait strategy, and register reconfiguration. The helper auto-selects between standard untilize and pack_untilize based on DEST capacity and data format. `WaitMode::WaitUpfront` and `InitUninitMode` variants enable integration into multi-stage compute kernels.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used to create circular buffers.
   **Key Information**: `create_cb` creates a CircularBufferConfig with `num_pages * page_size` total capacity. When a `Buffer*` is passed (non-null), it calls `set_globally_allocated_address` to back the CB with an existing buffer (used for sharded inputs). When null, the CB is allocated fresh in L1.
